//! Prepass utilities for MIR lowering: ensures expressions have values and resolves consts.

use super::*;
use hir::analysis::name_resolution::{PathRes, resolve_path};
use hir::analysis::ty::const_eval::{
    ConstValue, eval_const_expr, evaluated_const_to_value, try_eval_const_body, try_eval_const_ref,
};
use hir::analysis::ty::const_ty::{ConstTyData, EvaluatedConstTy};
use hir::analysis::ty::fold::TyFoldable;
use hir::analysis::ty::normalize::normalize_ty;
use hir::analysis::ty::trait_resolution::PredicateListId;

use crate::{
    ConstData, pack_inline_string_word, serialize_const_data_to_bytes,
    serialize_const_u8_array_bytes,
};

impl<'db, 'a> MirBuilder<'db, 'a> {
    fn resolve_const_ref_under_generics(
        &self,
        cref: hir::analysis::ty::ty_check::ConstRef<'db>,
    ) -> hir::analysis::ty::ty_check::ConstRef<'db> {
        struct GenericSubst<'a, 'db> {
            generic_args: &'a [TyId<'db>],
        }

        impl<'db> hir::analysis::ty::fold::TyFolder<'db> for GenericSubst<'_, 'db> {
            fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
                match ty.data(db) {
                    hir::analysis::ty::ty_def::TyData::TyParam(param) => {
                        self.generic_args.get(param.idx).copied().unwrap_or(ty)
                    }
                    hir::analysis::ty::ty_def::TyData::ConstTy(const_ty) => {
                        if let ConstTyData::TyParam(param, _) = const_ty.data(db)
                            && let Some(rep) = self.generic_args.get(param.idx).copied()
                        {
                            return rep;
                        }
                        ty.super_fold_with(db, self)
                    }
                    _ => ty.super_fold_with(db, self),
                }
            }
        }

        let mut subst = GenericSubst {
            generic_args: self.generic_args,
        };
        let cref = cref.fold_with(self.db, &mut subst);
        if let hir::analysis::ty::ty_check::ConstRef::TraitConst(assoc) = cref
            && matches!(
                assoc.inst().self_ty(self.db).data(self.db),
                hir::analysis::ty::ty_def::TyData::TyParam(_)
                    | hir::analysis::ty::ty_def::TyData::TyVar(_)
            )
            && let Some(&self_arg) = self.generic_args.first()
        {
            let mut args = assoc.inst().args(self.db).to_vec();
            if let Some(arg) = args.first_mut() {
                *arg = self_arg;
            }
            return hir::analysis::ty::ty_check::ConstRef::TraitConst(
                hir::analysis::ty::assoc_const::AssocConstUse::new(
                    assoc.origin_scope(),
                    assoc.assumptions(),
                    hir::analysis::ty::trait_def::TraitInstId::new(
                        self.db,
                        assoc.inst().def(self.db),
                        args,
                        assoc.inst().assoc_type_bindings(self.db).clone(),
                    ),
                    assoc.name(),
                ),
            );
        }

        cref
    }

    /// Helper to iterate expressions and conditionally force value lowering.
    ///
    /// # Parameters
    /// - `predicate`: Predicate that selects which expressions to visit.
    /// - `ensure`: Callback invoked for each matching expression to perform lowering.
    ///
    /// # Returns
    /// Nothing; mutates internal caches for selected expressions.
    pub(super) fn ensure_expr_values<P, F>(&mut self, predicate: P, mut ensure: F)
    where
        P: Fn(&Expr<'db>) -> bool,
        F: FnMut(&mut Self, ExprId),
    {
        let exprs = self.body.exprs(self.db);
        for expr_id in exprs.keys() {
            let Partial::Present(expr) = &exprs[expr_id] else {
                continue;
            };
            if predicate(expr) {
                ensure(self, expr_id);
            }
        }
    }

    /// Forces all const path expressions to lower into synthetic literals.
    ///
    /// # Returns
    /// Nothing; caches literal `ValueId`s for const paths.
    pub(super) fn ensure_const_expr_values(&mut self) {
        self.ensure_expr_values(
            |expr| matches!(expr, Expr::Path(..)),
            |this, expr_id| {
                let expr_ty = this.typed_body.expr_ty(this.db, expr_id);
                let expr_ty = expr_ty
                    .as_capability(this.db)
                    .map(|(_, inner)| inner)
                    .unwrap_or(expr_ty);
                if expr_ty.is_array(this.db) {
                    return;
                }
                if let Some(value_id) = this.try_const_expr(expr_id) {
                    if let Some(&existing) = this.builder.body.expr_values.get(&expr_id) {
                        if matches!(
                            this.builder.body.values[existing.index()].origin,
                            ValueOrigin::Expr(_)
                        ) {
                            this.builder.body.expr_values.insert(expr_id, value_id);
                        }
                    } else {
                        this.builder.body.expr_values.insert(expr_id, value_id);
                    }
                }
            },
        );
    }

    /// Ensure that the given expression has a corresponding MIR value.
    ///
    /// # Parameters
    /// - `expr`: Expression to materialize into a value.
    ///
    /// # Returns
    /// The `ValueId` bound to the expression.
    pub(super) fn ensure_value(&mut self, expr: ExprId) -> ValueId {
        if let Some(&val) = self.builder.body.expr_values.get(&expr) {
            return val;
        }

        let value = match expr.data(self.db, self.body) {
            Partial::Present(Expr::Block(stmts)) => {
                let last_expr = stmts.iter().rev().find_map(|stmt_id| {
                    let Partial::Present(stmt) = stmt_id.data(self.db, self.body) else {
                        return None;
                    };
                    if let Stmt::Expr(expr_id) = stmt {
                        Some(*expr_id)
                    } else {
                        None
                    }
                });
                if let Some(inner) = last_expr {
                    let val = self.ensure_value(inner);
                    self.builder.body.expr_values.insert(expr, val);
                    return val;
                }
                self.alloc_expr_value(expr)
            }
            _ => self.alloc_expr_value(expr),
        };

        self.builder.body.expr_values.insert(expr, value);
        value
    }

    /// Allocate the MIR value slot for an expression, handling special cases.
    ///
    /// # Parameters
    /// - `expr`: Expression to allocate a value for.
    ///
    /// # Returns
    /// The allocated `ValueId` (lowered call/field/const where applicable).
    pub(super) fn alloc_expr_value(&mut self, expr: ExprId) -> ValueId {
        if let Some(value) = self.try_const_expr(expr) {
            self.builder.body.values[value.index()].source = self.source_for_expr(expr);
            return value;
        }

        let ty = self.typed_body.expr_ty(self.db, expr);
        if let Partial::Present(Expr::Lit(LitKind::String(str_id))) = expr.data(self.db, self.body)
            && !self.is_core_dyn_string_ty(ty)
        {
            match self.alloc_bytes_value(ty, str_id.data(self.db).as_bytes().to_vec()) {
                Ok(value_id) => {
                    self.builder.body.values[value_id.index()].source = self.source_for_expr(expr);
                    return value_id;
                }
                Err(message) => self.defer_materialization_error(expr, ty, &message),
            }
        }

        let mut repr = self.value_repr_for_expr(expr, ty);
        let origin = match expr.data(self.db, self.body) {
            Partial::Present(Expr::Lit(LitKind::Int(int_id))) => {
                ValueOrigin::Synthetic(SyntheticValue::Int(int_id.data(self.db).clone()))
            }
            Partial::Present(Expr::Lit(LitKind::Bool(flag))) => {
                ValueOrigin::Synthetic(SyntheticValue::Bool(*flag))
            }
            Partial::Present(Expr::Path(_)) => {
                let expr_prop = self.typed_body.expr_prop(self.db, expr);
                if let Some(binding) = expr_prop.binding {
                    if self
                        .hir_func
                        .is_some_and(|func| extract_contract_function(self.db, func).is_some())
                        && matches!(
                            binding,
                            LocalBinding::EffectParam { .. }
                                | LocalBinding::Param {
                                    site: ParamSite::EffectField(_),
                                    ..
                                }
                        )
                    {
                        // TODO: document/enforce this rule:
                        //   effect params on contract_init/contract_runtime must be zero-sized concrete types
                        debug_assert!(
                            crate::layout::ty_size_bytes(self.db, ty) == Some(0),
                            "contract entrypoint effect params must be concrete zero-sized providers; got `{}`",
                            ty.pretty_print(self.db)
                        );
                        ValueOrigin::Unit
                    } else if let Some(target) = self.code_region_target_from_ty(ty) {
                        ValueOrigin::CodeRegionRef(target)
                    } else if let Some(local) = self.local_for_binding(binding) {
                        if self.effect_param_key_is_type(binding)
                            && matches!(repr, ValueRepr::Word)
                            && !crate::layout::is_zero_sized_ty(self.db, ty)
                        {
                            // Type-effect params are addressable providers; even when their runtime
                            // representation is a single word, treat them as `Ref` roots so lowering
                            // can emit `Place`-based loads/stores (including transparent newtype
                            // peeling over field-0 projections).
                            repr = ValueRepr::Ref(self.address_space_for_binding(&binding));
                        }
                        ValueOrigin::Local(local)
                    } else if let Some(target) = self.code_region_target(expr) {
                        ValueOrigin::CodeRegionRef(target)
                    } else {
                        ValueOrigin::Expr(expr)
                    }
                } else if let Some(target) = self.code_region_target(expr) {
                    ValueOrigin::CodeRegionRef(target)
                } else {
                    ValueOrigin::Expr(expr)
                }
            }
            Partial::Present(Expr::Un(inner, op)) => ValueOrigin::Unary {
                op: *op,
                inner: self.ensure_value(*inner),
            },
            Partial::Present(Expr::Cast(inner, _)) => ValueOrigin::TransparentCast {
                value: self.ensure_value(*inner),
            },
            Partial::Present(Expr::Bin(lhs, rhs, op)) => {
                if matches!(op, hir::hir_def::expr::BinOp::Index) {
                    ValueOrigin::Expr(expr)
                } else {
                    ValueOrigin::Binary {
                        op: *op,
                        lhs: self.ensure_value(*lhs),
                        rhs: self.ensure_value(*rhs),
                    }
                }
            }
            Partial::Present(Expr::If(..) | Expr::Match(..)) => {
                ValueOrigin::ControlFlowResult { expr }
            }
            Partial::Present(Expr::Block(..)) => ValueOrigin::Unit,
            _ if ty.is_tuple(self.db) && ty.field_count(self.db) == 0 => ValueOrigin::Unit,
            _ => ValueOrigin::Expr(expr),
        };

        let value_id = self.alloc_value(ty, origin, repr);
        self.builder.body.values[value_id.index()].source = self.source_for_expr(expr);
        value_id
    }

    /// Collect all argument expressions and their lowered values for a call or method call.
    ///
    /// # Parameters
    /// - `expr`: Expression id representing the call or method call.
    ///
    /// # Returns
    /// A tuple of lowered argument `ValueId`s and their corresponding `ExprId`s.
    pub(super) fn collect_call_args(
        &mut self,
        expr: ExprId,
    ) -> Option<(Vec<ValueId>, Vec<ExprId>)> {
        let exprs = self.body.exprs(self.db);
        let Partial::Present(expr_data) = &exprs[expr] else {
            return None;
        };
        match expr_data {
            Expr::Call(_, call_args) => {
                let mut args = Vec::with_capacity(call_args.len());
                let mut arg_exprs = Vec::with_capacity(call_args.len());
                for arg in call_args.iter() {
                    arg_exprs.push(arg.expr);
                    args.push(self.lower_expr(arg.expr));
                }
                Some((args, arg_exprs))
            }
            Expr::MethodCall(receiver, _, _, call_args) => {
                let mut args = Vec::with_capacity(call_args.len() + 1);
                let mut arg_exprs = Vec::with_capacity(call_args.len() + 1);
                arg_exprs.push(*receiver);
                args.push(self.lower_expr(*receiver));
                for arg in call_args.iter() {
                    arg_exprs.push(arg.expr);
                    args.push(self.lower_expr(arg.expr));
                }
                Some((args, arg_exprs))
            }
            // Operator expressions desugared to trait method calls:
            // binary `a + b` → Add::add(a, b), unary `-a` → Neg::neg(a), etc.
            Expr::Bin(lhs, rhs, _) => {
                let mut args = Vec::with_capacity(2);
                let mut arg_exprs = Vec::with_capacity(2);
                arg_exprs.push(*lhs);
                args.push(self.lower_expr(*lhs));
                arg_exprs.push(*rhs);
                args.push(self.lower_expr(*rhs));
                Some((args, arg_exprs))
            }
            Expr::Un(inner, _) => {
                let arg_exprs = vec![*inner];
                let args = vec![self.lower_expr(*inner)];
                Some((args, arg_exprs))
            }
            _ => None,
        }
    }

    /// Attempts to resolve a path expression to a literal `const` value.
    ///
    /// # Parameters
    /// - `expr`: Path expression to resolve.
    ///
    /// # Returns
    /// A MIR `ValueId` referencing a synthetic literal when successful.
    pub(super) fn try_const_expr(&mut self, expr: ExprId) -> Option<ValueId> {
        let Partial::Present(Expr::Path(path)) = expr.data(self.db, self.body) else {
            return None;
        };
        let path = path.to_opt()?;

        if let Some(cref) = self.typed_body.expr_const_ref(expr) {
            if let hir::analysis::ty::ty_check::ConstRef::Const(const_def) = cref
                && let Some(&cached) = self.const_cache.get(&const_def)
                && self.is_const_cache_value_reusable(cached)
            {
                return Some(cached);
            }
            let cref = self.resolve_const_ref_under_generics(cref);

            let ty = self.typed_body.expr_ty(self.db, expr);
            let assumptions = self.typed_body.assumptions();
            let expected_ty = normalize_ty(self.db, ty, self.body.scope(), assumptions);
            let capability_expected_ty = expected_ty.as_capability(self.db).map(|(_, ty)| ty);
            let base_expected_ty = expected_ty.base_ty(self.db);
            if let Some(value) = try_eval_const_ref(self.db, cref, expected_ty)
                .or_else(|| {
                    capability_expected_ty.and_then(|inner_expected_ty| {
                        try_eval_const_ref(self.db, cref, inner_expected_ty)
                    })
                })
                .or_else(|| {
                    (base_expected_ty != expected_ty)
                        .then(|| try_eval_const_ref(self.db, cref, base_expected_ty))
                        .flatten()
                })
                .or_else(|| {
                    eval_const_expr(self.db, self.body, self.typed_body, self.generic_args, expr)
                        .ok()
                        .flatten()
                })
            {
                // Const arrays lower to an allocation + store.
                // Reusing that ValueId across control-flow paths is unsound, so we
                // materialize arrays at each use site and do not cache their ValueId.
                if expected_ty.is_array(self.db)
                    && matches!(value, ConstValue::ConstArray(_) | ConstValue::Bytes(_))
                    && let Some(value_id) =
                        self.try_emit_const_array(expected_ty, value.clone().into())
                {
                    return Some(value_id);
                }
                let value_id = match self.alloc_const_scalar_value(expected_ty, value) {
                    Ok(Some(value_id)) => value_id,
                    Ok(None) => return None,
                    Err(message) => {
                        self.defer_materialization_error(expr, expected_ty, &message);
                        return None;
                    }
                };
                if let hir::analysis::ty::ty_check::ConstRef::Const(const_def) = cref {
                    self.const_cache.insert(const_def, value_id);
                }
                return Some(value_id);
            }
        }

        // Const generic parameter (e.g. `const SALT: u256`).
        if self.generic_args.is_empty() {
            return None;
        }
        if self.typed_body.expr_prop(self.db, expr).binding.is_some() {
            return None;
        }

        let assumptions = self.typed_body.assumptions();
        let resolved = resolve_path(self.db, path, self.body.scope(), assumptions, true).ok()?;
        let ty = match resolved {
            PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => ty,
            _ => return None,
        };

        let TyData::ConstTy(const_ty) = ty.data(self.db) else {
            return None;
        };
        let ConstTyData::TyParam(param, _) = const_ty.data(self.db) else {
            return None;
        };
        let arg = *self.generic_args.get(param.idx)?;
        let TyData::ConstTy(const_arg) = arg.data(self.db) else {
            return None;
        };

        let expected_ty = self.typed_body.expr_ty(self.db, expr);
        let expected_ty = normalize_ty(self.db, expected_ty, self.body.scope(), assumptions);
        let capability_expected_ty = expected_ty.as_capability(self.db).map(|(_, ty)| ty);
        let base_expected_ty = expected_ty.base_ty(self.db);
        match const_arg.data(self.db) {
            ConstTyData::Evaluated(EvaluatedConstTy::LitInt(value), _) => {
                Some(self.alloc_synthetic_value(
                    expected_ty,
                    SyntheticValue::Int(value.data(self.db).clone()),
                ))
            }
            ConstTyData::Evaluated(EvaluatedConstTy::LitBool(flag), _) => {
                Some(self.alloc_synthetic_value(expected_ty, SyntheticValue::Bool(*flag)))
            }
            ConstTyData::Evaluated(EvaluatedConstTy::EnumVariant(variant), _) => {
                Some(self.alloc_synthetic_value(
                    expected_ty,
                    SyntheticValue::Int(BigUint::from(variant.idx as u64)),
                ))
            }
            ConstTyData::Evaluated(EvaluatedConstTy::Bytes(bytes), _) => {
                if expected_ty.is_array(self.db) {
                    self.try_emit_const_array(expected_ty, ConstData::Bytes(bytes.clone()))
                } else {
                    match self.alloc_bytes_value(expected_ty, bytes.clone()) {
                        Ok(value_id) => Some(value_id),
                        Err(message) => {
                            self.defer_materialization_error(expr, expected_ty, &message);
                            None
                        }
                    }
                }
            }
            ConstTyData::Evaluated(EvaluatedConstTy::Array(elems), _) => {
                let values: Option<Vec<_>> = elems
                    .iter()
                    .map(|elem_ty| {
                        let TyData::ConstTy(elem_const) = elem_ty.data(self.db) else {
                            return None;
                        };
                        evaluated_const_to_value(self.db, *elem_const)
                    })
                    .collect();
                if let Some(elems) = values {
                    self.try_emit_const_array(
                        expected_ty,
                        ConstData::Array(elems.into_iter().map(ConstData::from).collect()),
                    )
                } else {
                    None
                }
            }
            ConstTyData::UnEvaluated { body, .. } => {
                match try_eval_const_body(self.db, *body, expected_ty)
                    .or_else(|| {
                        capability_expected_ty.and_then(|inner_expected_ty| {
                            try_eval_const_body(self.db, *body, inner_expected_ty)
                        })
                    })
                    .or_else(|| {
                        (base_expected_ty != expected_ty)
                            .then(|| try_eval_const_body(self.db, *body, base_expected_ty))
                            .flatten()
                    }) {
                    Some(ConstValue::ConstArray(ref elems)) => self.try_emit_const_array(
                        expected_ty,
                        ConstData::Array(elems.iter().cloned().map(ConstData::from).collect()),
                    ),
                    Some(ConstValue::Bytes(bytes)) => {
                        if expected_ty.is_array(self.db) {
                            self.try_emit_const_array(expected_ty, ConstData::Bytes(bytes))
                        } else {
                            match self.alloc_bytes_value(expected_ty, bytes) {
                                Ok(value_id) => Some(value_id),
                                Err(message) => {
                                    self.defer_materialization_error(expr, expected_ty, &message);
                                    None
                                }
                            }
                        }
                    }
                    Some(value) => match self.alloc_const_scalar_value(expected_ty, value) {
                        Ok(value) => value,
                        Err(message) => {
                            self.defer_materialization_error(expr, expected_ty, &message);
                            None
                        }
                    },
                    None => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn const_array_region_for_expr(
        &mut self,
        expr: ExprId,
        array_ty: TyId<'db>,
    ) -> Option<ConstRegionId> {
        let Partial::Present(Expr::Path(path)) = expr.data(self.db, self.body) else {
            return None;
        };
        let path = path.to_opt()?;
        let assumptions = PredicateListId::empty_list(self.db);
        let expected_ty = normalize_ty(self.db, array_ty, self.body.scope(), assumptions);

        if let Some(cref) = self.typed_body.expr_const_ref(expr) {
            let cref = self.resolve_const_ref_under_generics(cref);

            let capability_expected_ty = expected_ty.as_capability(self.db).map(|(_, ty)| ty);
            let base_expected_ty = expected_ty.base_ty(self.db);
            if let Some(value) = try_eval_const_ref(self.db, cref, expected_ty)
                .or_else(|| {
                    capability_expected_ty.and_then(|inner_expected_ty| {
                        try_eval_const_ref(self.db, cref, inner_expected_ty)
                    })
                })
                .or_else(|| {
                    (base_expected_ty != expected_ty)
                        .then(|| try_eval_const_ref(self.db, cref, base_expected_ty))
                        .flatten()
                })
                .or_else(|| {
                    eval_const_expr(self.db, self.body, self.typed_body, self.generic_args, expr)
                        .ok()
                        .flatten()
                })
            {
                match self.const_array_region_for_value(&cref, expected_ty, &value) {
                    Ok(Some(region)) => return Some(region),
                    Ok(None) => {}
                    Err(message) => {
                        self.defer_const_materialization_error(expr, expected_ty, &message);
                        return None;
                    }
                }
            }
        }

        if self.generic_args.is_empty()
            || self.typed_body.expr_prop(self.db, expr).binding.is_some()
        {
            return None;
        }

        let resolved = resolve_path(self.db, path, self.body.scope(), assumptions, true).ok()?;
        let ty = match resolved {
            PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => ty,
            _ => return None,
        };
        let TyData::ConstTy(const_ty) = ty.data(self.db) else {
            return None;
        };
        let ConstTyData::TyParam(param, _) = const_ty.data(self.db) else {
            return None;
        };
        let arg = *self.generic_args.get(param.idx)?;
        let TyData::ConstTy(const_arg) = arg.data(self.db) else {
            return None;
        };
        let capability_expected_ty = expected_ty.as_capability(self.db).map(|(_, ty)| ty);
        let base_expected_ty = expected_ty.base_ty(self.db);
        match const_arg.data(self.db) {
            ConstTyData::Evaluated(EvaluatedConstTy::Bytes(bytes), _) => {
                match self.intern_const_u8_array_region(expected_ty, bytes) {
                    Ok(region) => Some(region),
                    Err(message) => {
                        self.defer_const_materialization_error(expr, expected_ty, &message);
                        None
                    }
                }
            }
            ConstTyData::Evaluated(EvaluatedConstTy::Array(elems), _) => {
                let values: Option<Vec<_>> = elems
                    .iter()
                    .map(|elem_ty| {
                        let TyData::ConstTy(elem_const) = elem_ty.data(self.db) else {
                            return None;
                        };
                        evaluated_const_to_value(self.db, *elem_const)
                    })
                    .collect();
                match values {
                    Some(elems) => match self.intern_const_array_region(expected_ty, &elems) {
                        Ok(region) => Some(region),
                        Err(message) => {
                            self.defer_const_materialization_error(expr, expected_ty, &message);
                            None
                        }
                    },
                    None => None,
                }
            }
            ConstTyData::UnEvaluated { body, .. } => {
                match try_eval_const_body(self.db, *body, expected_ty)
                    .or_else(|| {
                        capability_expected_ty.and_then(|inner_expected_ty| {
                            try_eval_const_body(self.db, *body, inner_expected_ty)
                        })
                    })
                    .or_else(|| {
                        (base_expected_ty != expected_ty)
                            .then(|| try_eval_const_body(self.db, *body, base_expected_ty))
                            .flatten()
                    }) {
                    Some(ConstValue::ConstArray(elems)) => {
                        match self.intern_const_array_region(expected_ty, &elems) {
                            Ok(region) => Some(region),
                            Err(message) => {
                                self.defer_const_materialization_error(expr, expected_ty, &message);
                                None
                            }
                        }
                    }
                    Some(ConstValue::Bytes(bytes)) => {
                        match self.intern_const_u8_array_region(expected_ty, &bytes) {
                            Ok(region) => Some(region),
                            Err(message) => {
                                self.defer_const_materialization_error(expr, expected_ty, &message);
                                None
                            }
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Allocates a synthetic literal value with the provided type.
    ///
    /// # Parameters
    /// - `ty`: Type of the synthetic literal.
    /// - `value`: Literal content to store.
    ///
    /// # Returns
    /// The new `ValueId` stored in the MIR body.
    pub(super) fn alloc_synthetic_value(
        &mut self,
        ty: TyId<'db>,
        value: SyntheticValue,
    ) -> ValueId {
        self.alloc_value(ty, ValueOrigin::Synthetic(value), ValueRepr::Word)
    }

    fn alloc_const_region_value(
        &mut self,
        ty: TyId<'db>,
        data: ConstData,
    ) -> Result<ValueId, String> {
        if serialize_const_data_to_bytes(self.db, &crate::layout::EVM_LAYOUT, ty, &data).is_none() {
            return Err(format!(
                "unsupported EVM const-data layout or serialization for `{}`",
                ty.pretty_print(self.db)
            ));
        }
        let region = self.builder.body.intern_const_region(ty, data);
        Ok(self.alloc_value(
            ty,
            ValueOrigin::ConstRegion(region),
            self.value_repr_for_ty(ty, AddressSpaceKind::Code),
        ))
    }

    fn alloc_bytes_value(&mut self, ty: TyId<'db>, bytes: Vec<u8>) -> Result<ValueId, String> {
        if ty.is_string(self.db) {
            let packed = pack_inline_string_word(&bytes).ok_or_else(|| {
                format!(
                    "string values are currently limited to {} bytes",
                    hir::analysis::ty::ty_def::MAX_INLINE_STRING_BYTES
                )
            })?;
            return Ok(self.alloc_synthetic_value(ty, SyntheticValue::Bytes(packed)));
        }
        if bytes.len() <= 32 {
            return Ok(self.alloc_synthetic_value(ty, SyntheticValue::Bytes(bytes)));
        }
        self.alloc_const_region_value(ty, ConstData::Bytes(bytes))
    }

    fn alloc_const_scalar_value(
        &mut self,
        ty: TyId<'db>,
        value: ConstValue,
    ) -> Result<Option<ValueId>, String> {
        Ok(Some(match value {
            ConstValue::Int(value) => self.alloc_synthetic_value(ty, SyntheticValue::Int(value)),
            ConstValue::Bool(flag) => self.alloc_synthetic_value(ty, SyntheticValue::Bool(flag)),
            ConstValue::Bytes(bytes) => self.alloc_bytes_value(ty, bytes)?,
            ConstValue::EnumVariant(idx) => {
                self.alloc_synthetic_value(ty, SyntheticValue::Int(BigUint::from(idx as u64)))
            }
            ConstValue::ConstArray(_) => return Ok(None),
        }))
    }

    fn defer_materialization_error(&mut self, expr: ExprId, ty: TyId<'db>, detail: &str) {
        if self.deferred_error.is_some() {
            return;
        }
        let func_name = self
            .hir_func
            .map(|func| func.pretty_print_signature(self.db))
            .unwrap_or_else(|| "<body owner>".to_owned());
        let expr_context = super::format_hir_expr_context(self.db, self.body, expr);
        self.deferred_error = Some(MirLowerError::Unsupported {
            func_name,
            message: format!(
                "failed to materialize `{expr_context}` as `{}`: {detail}",
                ty.pretty_print(self.db)
            ),
        });
    }

    fn defer_const_materialization_error(&mut self, expr: ExprId, ty: TyId<'db>, detail: &str) {
        if self.deferred_error.is_some() {
            return;
        }
        let func_name = self
            .hir_func
            .map(|func| func.pretty_print_signature(self.db))
            .unwrap_or_else(|| "<body owner>".to_owned());
        let expr_context = super::format_hir_expr_context(self.db, self.body, expr);
        self.deferred_error = Some(MirLowerError::Unsupported {
            func_name,
            message: format!(
                "failed to materialize const `{expr_context}` as `{}`: {detail}",
                ty.pretty_print(self.db)
            ),
        });
    }

    fn is_const_cache_value_reusable(&self, value_id: ValueId) -> bool {
        matches!(
            self.builder.body.value(value_id).origin,
            ValueOrigin::Synthetic(_) | ValueOrigin::ConstRegion(_)
        )
    }

    fn const_array_region_for_value(
        &mut self,
        cref: &hir::analysis::ty::ty_check::ConstRef<'db>,
        array_ty: TyId<'db>,
        value: &ConstValue,
    ) -> Result<Option<ConstRegionId>, String> {
        if !array_ty.is_array(self.db) {
            return Ok(None);
        }
        match value {
            ConstValue::ConstArray(elems) => self
                .const_array_region_for_ref(cref, array_ty, elems)
                .map(Some),
            ConstValue::Bytes(raw) => self
                .const_u8_array_region_for_ref(cref, array_ty, raw)
                .map(Some),
            _ => Ok(None),
        }
    }

    fn const_array_region_for_ref(
        &mut self,
        cref: &hir::analysis::ty::ty_check::ConstRef<'db>,
        array_ty: TyId<'db>,
        elems: &[ConstValue],
    ) -> Result<ConstRegionId, String> {
        let hir::analysis::ty::ty_check::ConstRef::Const(const_def) = cref else {
            return self.intern_const_array_region(array_ty, elems);
        };

        if let Some((cached_ty, region)) = self.const_array_region_cache.get(const_def)
            && *cached_ty == array_ty
        {
            return Ok(*region);
        }

        let region = self.intern_const_array_region(array_ty, elems)?;
        self.const_array_region_cache
            .insert(*const_def, (array_ty, region));
        Ok(region)
    }

    fn const_u8_array_region_for_ref(
        &mut self,
        cref: &hir::analysis::ty::ty_check::ConstRef<'db>,
        array_ty: TyId<'db>,
        raw: &[u8],
    ) -> Result<ConstRegionId, String> {
        let hir::analysis::ty::ty_check::ConstRef::Const(const_def) = cref else {
            return self.intern_const_u8_array_region(array_ty, raw);
        };

        if let Some((cached_ty, region)) = self.const_array_region_cache.get(const_def)
            && *cached_ty == array_ty
        {
            return Ok(*region);
        }

        let region = self.intern_const_u8_array_region(array_ty, raw)?;
        self.const_array_region_cache
            .insert(*const_def, (array_ty, region));
        Ok(region)
    }

    fn intern_const_array_region(
        &mut self,
        array_ty: TyId<'db>,
        elems: &[ConstValue],
    ) -> Result<ConstRegionId, String> {
        let data = ConstData::Array(elems.iter().cloned().map(ConstData::from).collect());
        let Some(_) =
            serialize_const_data_to_bytes(self.db, &crate::layout::EVM_LAYOUT, array_ty, &data)
        else {
            return Err(format!(
                "unsupported EVM const-array layout or element serialization for `{}`",
                array_ty.pretty_print(self.db)
            ));
        };
        Ok(self.builder.body.intern_const_region(array_ty, data))
    }

    fn intern_const_u8_array_region(
        &mut self,
        array_ty: TyId<'db>,
        raw: &[u8],
    ) -> Result<ConstRegionId, String> {
        let Some(_) =
            serialize_const_u8_array_bytes(self.db, &crate::layout::EVM_LAYOUT, array_ty, raw)
        else {
            return Err(format!(
                "unsupported EVM `[u8; N]` const-array layout or byte serialization for `{}`",
                array_ty.pretty_print(self.db)
            ));
        };
        Ok(self
            .builder
            .body
            .intern_const_region(array_ty, ConstData::Bytes(raw.to_vec())))
    }

    /// Emits a typed constant aggregate materialization for the array value.
    fn try_emit_const_array(&mut self, array_ty: TyId<'db>, data: ConstData) -> Option<ValueId> {
        self.current_block()?;

        let dest = self.alloc_temp_local(array_ty, false, "const_array");
        self.set_local_address_space(dest, AddressSpaceKind::Memory);

        self.assign(
            None,
            Some(dest),
            Rvalue::ConstAggregate { data, ty: array_ty },
        );

        Some(self.alloc_value(
            array_ty,
            ValueOrigin::Local(dest),
            ValueRepr::Ref(self.builder.body.local(dest).address_space),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::ConstValue;
    use hir::{
        analysis::{
            HirAnalysisDb,
            ty::{
                const_ty::{ConstTyData, ConstTyId, EvaluatedConstTy},
                ty_def::{PrimTy, TyBase, TyData, TyId},
            },
        },
        hir_def::IntegerId,
        test_db::HirAnalysisTestDb,
    };
    use num_bigint::BigUint;

    use crate::{ConstData, serialize_const_data_to_bytes};

    fn array_with_len<'db>(db: &'db dyn HirAnalysisDb, elem: TyId<'db>, len: usize) -> TyId<'db> {
        let array_ctor = TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::Array)));
        let array = TyId::app(db, array_ctor, elem);
        let len_kind = array
            .applicable_ty(db)
            .and_then(|prop| prop.const_ty)
            .expect("array type should have a length const parameter");
        let len_const = ConstTyId::new(
            db,
            ConstTyData::Evaluated(
                EvaluatedConstTy::LitInt(IntegerId::new(db, BigUint::from(len))),
                len_kind,
            ),
        );
        TyId::app(db, array, TyId::new(db, TyData::ConstTy(len_const)))
    }

    #[test]
    fn serialize_nested_bool_const_array_with_padding() {
        let db = HirAnalysisTestDb::default();
        let bool_ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::Bool)));
        let inner = array_with_len(&db, bool_ty, 3);
        let outer = array_with_len(&db, inner, 2);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            outer,
            &ConstData::Array(vec![
                ConstData::Array(vec![
                    ConstData::Bool(true),
                    ConstData::Bool(false),
                    ConstData::Bool(true),
                ]),
                ConstData::Array(vec![
                    ConstData::Bool(false),
                    ConstData::Bool(true),
                    ConstData::Bool(false),
                ]),
            ]),
        )
        .expect("nested bool array should serialize");

        assert_eq!(data.len(), 64);
        assert_eq!(&data[0..3], &[1, 0, 1]);
        assert!(data[3..32].iter().all(|b| *b == 0));
        assert_eq!(&data[32..35], &[0, 1, 0]);
        assert!(data[35..64].iter().all(|b| *b == 0));
    }

    #[test]
    fn serialize_nested_u8_bytes_array_with_word_padding() {
        let db = HirAnalysisTestDb::default();
        let u8_ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::U8)));
        let inner = array_with_len(&db, u8_ty, 2);
        let outer = array_with_len(&db, inner, 2);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            outer,
            &ConstData::Array(vec![
                ConstData::Bytes(vec![1, 2]),
                ConstData::Bytes(vec![3, 4]),
            ]),
        )
        .expect("nested u8 arrays should serialize");

        assert_eq!(data.len(), 64);
        assert_eq!(data[0], 0x01);
        assert_eq!(data[1], 0x02);
        assert!(data[2..32].iter().all(|b| *b == 0));
        assert_eq!(data[32], 0x03);
        assert_eq!(data[33], 0x04);
        assert!(data[34..64].iter().all(|b| *b == 0));
    }

    #[test]
    fn serialize_mixed_scalar_const_array_words() {
        let db = HirAnalysisTestDb::default();
        let u256_ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::U256)));
        let arr = array_with_len(&db, u256_ty, 3);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            arr,
            &ConstData::Array(vec![
                ConstData::from(ConstValue::Int(1u8.into())),
                ConstData::EnumVariant(2),
                ConstData::Bool(true),
            ]),
        )
        .expect("scalars should serialize");
        assert_eq!(data.len(), 96);
        assert_eq!(data[31], 0x01);
        assert_eq!(data[63], 0x02);
        assert_eq!(data[95], 0x01);
    }
}

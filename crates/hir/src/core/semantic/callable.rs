//! Source-level callable semantics shared by editor features.
//!
//! A direct call through `Fn`/`FnOnce` is physically a trait-method call with
//! a receiver and tuple argument.  Those implementation details are useful to
//! lowering, but are the wrong interface for hover, completion, signature
//! help, and navigation.  This module projects a type-checked call back to the
//! logical signature written by the user.

use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{
            adt_def::AdtRef,
            closure::ClosureCallTrait,
            fold::TyFoldable,
            trait_def::TraitInstId,
            trait_lower::lower_impl_trait,
            trait_resolution::{GoalSatisfiability, TraitSolveCx, is_goal_satisfiable},
            ty_check::{Callable, TypedBody},
            ty_def::{ClosureCallMode, ClosureParamMode, ClosureTy, TyId},
            unify::UnificationTable,
        },
    },
    hir_def::{
        Body, CallableDef, Expr, FuncParamName, IdentId, ImplTrait, Partial, params::FuncParamMode,
    },
};

use super::reference::{enclosing_assumptions, typed_body_for_body};
use super::{CallSiteKind, CallSiteView};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalCallableCapability {
    Reusable,
    Consuming,
}

impl LogicalCallableCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reusable => "reusable",
            Self::Consuming => "consuming",
        }
    }
}

impl From<ClosureCallMode> for LogicalCallableCapability {
    fn from(mode: ClosureCallMode) -> Self {
        match mode {
            ClosureCallMode::Reusable => Self::Reusable,
            ClosureCallMode::Consuming => Self::Consuming,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalCallableParamMode {
    Own,
    View,
    Ref,
    Mut,
}

impl LogicalCallableParamMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::View => "view",
            Self::Ref => "ref",
            Self::Mut => "mut",
        }
    }
}

impl From<ClosureParamMode> for LogicalCallableParamMode {
    fn from(mode: ClosureParamMode) -> Self {
        match mode {
            ClosureParamMode::Own => Self::Own,
            ClosureParamMode::View => Self::View,
            ClosureParamMode::Ref => Self::Ref,
            ClosureParamMode::Mut => Self::Mut,
        }
    }
}

impl From<FuncParamMode> for LogicalCallableParamMode {
    fn from(mode: FuncParamMode) -> Self {
        match mode {
            FuncParamMode::Own => Self::Own,
            FuncParamMode::View => Self::View,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalCallableTarget<'db> {
    Definition(CallableDef<'db>),
    Closure(ClosureTy<'db>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalCallableParam<'db> {
    pub name: Option<IdentId<'db>>,
    pub mode: LogicalCallableParamMode,
    /// The parameter payload, with its ownership carrier represented by
    /// `mode` rather than duplicated in the type.
    pub ty: TyId<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalCallableSignature<'db> {
    pub target: LogicalCallableTarget<'db>,
    pub params: Vec<LogicalCallableParam<'db>>,
    pub ret_ty: TyId<'db>,
    pub capability: Option<LogicalCallableCapability>,
}

impl<'db> LogicalCallableSignature<'db> {
    /// Build the logical signature selected by type checking for `site`.
    pub fn for_call_site(db: &'db dyn HirAnalysisDb, site: &CallSiteView<'db>) -> Option<Self> {
        let typed_body = typed_body_for_body(db, site.body)?;
        let callable = typed_body.callable_expr(site.expr_id)?;
        Some(Self::from_selected_callable(
            db,
            site.body,
            typed_body,
            callable,
            matches!(site.kind, CallSiteKind::MethodCall { .. }),
        ))
    }

    /// Build a signature for a concrete closure value even when it is not
    /// currently being called (for hover and completion).
    pub fn for_closure(db: &'db dyn HirAnalysisDb, closure: ClosureTy<'db>) -> Self {
        let names = closure_param_names(db, closure);
        let params = closure
            .params(db)
            .iter()
            .copied()
            .zip(closure.param_modes(db).iter().copied())
            .enumerate()
            .map(|(idx, (carrier, mode))| LogicalCallableParam {
                name: names.get(idx).copied().flatten(),
                mode: mode.into(),
                ty: mode.payload(db, carrier),
            })
            .collect();
        Self {
            target: LogicalCallableTarget::Closure(closure),
            params,
            ret_ty: closure.ret_ty(db),
            capability: Some(closure.call_mode(db).into()),
        }
    }

    pub fn for_ty(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Option<Self> {
        let payload = ty.as_capability(db).map_or(ty, |(_, payload)| payload);
        payload
            .as_closure(db)
            .map(|closure| Self::for_closure(db, closure))
    }

    /// Enumerate the logical call signatures supported by a value in this
    /// scope. This covers concrete closures, generic `Fn`/`FnOnce` bounds,
    /// and user-defined callable implementations. Equivalent `Fn` and
    /// `FnOnce` signatures are de-duplicated in favor of reusable `Fn`.
    pub fn for_ty_in_scope(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        scope: crate::hir_def::scope_graph::ScopeId<'db>,
    ) -> Vec<Self> {
        if let Some(signature) = Self::for_ty(db, ty) {
            return vec![signature];
        }
        let payload = ty.as_capability(db).map_or(ty, |(_, payload)| payload);
        let mut candidates = Vec::new();

        for &inst in enclosing_assumptions(db, scope).list(db) {
            if inst.self_ty(db) != payload {
                continue;
            }
            let Some(call_trait) = ClosureCallTrait::for_trait(db, scope, inst.def(db)) else {
                continue;
            };
            let target = inst
                .def(db)
                .methods(db)
                .find(|method| {
                    method
                        .name(db)
                        .to_opt()
                        .is_some_and(|name| name.data(db) == call_trait.method_name())
                })
                .map(CallableDef::Func);
            if let Some(signature) = signature_from_trait_inst(db, inst, call_trait, target) {
                candidates.push(signature);
            }
        }

        for (impl_trait, inst, call_trait) in
            applicable_callable_impl_candidates(db, payload, scope)
        {
            let target = impl_trait
                .scope()
                .child_items(db)
                .filter_map(|item| match item {
                    crate::hir_def::ItemKind::Func(method) => Some(method),
                    _ => None,
                })
                .find(|method| {
                    method
                        .name(db)
                        .to_opt()
                        .is_some_and(|name| name.data(db) == call_trait.method_name())
                })
                .map(CallableDef::Func);
            if let Some(signature) = signature_from_trait_inst(db, inst, call_trait, target) {
                candidates.push(signature);
            }
        }

        candidates.sort_by_key(|signature| {
            !matches!(
                signature.capability,
                Some(LogicalCallableCapability::Reusable)
            )
        });
        let mut deduped: Vec<Self> = Vec::new();
        for candidate in candidates {
            if deduped.iter().any(|existing| {
                existing.params == candidate.params && existing.ret_ty == candidate.ret_ty
            }) {
                continue;
            }
            deduped.push(candidate);
        }
        deduped
    }

    fn from_selected_callable(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        _typed_body: &TypedBody<'db>,
        callable: &Callable<'db>,
        is_method_call: bool,
    ) -> Self {
        let closure = concrete_closure_for_callable(db, callable);
        if let Some(pack) = callable.call_trait_args_pack_ty(db, body.scope()) {
            let closure_modes = closure.map(|closure| closure.param_modes(db));
            let closure_names = closure
                .map(|closure| closure_param_names(db, closure))
                .unwrap_or_default();
            let params = pack
                .field_types(db)
                .into_iter()
                .enumerate()
                .map(|(idx, carrier)| {
                    let mode = closure_modes
                        .and_then(|modes| modes.get(idx).copied())
                        .or_else(|| ClosureParamMode::try_from_carrier(db, carrier))
                        .unwrap_or(ClosureParamMode::View);
                    LogicalCallableParam {
                        name: closure_names.get(idx).copied().flatten(),
                        mode: mode.into(),
                        ty: mode.payload(db, carrier),
                    }
                })
                .collect();
            let call_trait = callable
                .trait_inst()
                .and_then(|inst| ClosureCallTrait::for_trait(db, body.scope(), inst.def(db)));
            let capability =
                closure
                    .map(|closure| closure.call_mode(db).into())
                    .or(match call_trait {
                        Some(ClosureCallTrait::Fn) => Some(LogicalCallableCapability::Reusable),
                        Some(ClosureCallTrait::FnOnce) => {
                            Some(LogicalCallableCapability::Consuming)
                        }
                        None => None,
                    });
            return Self {
                target: closure.map_or(
                    LogicalCallableTarget::Definition(callable.callable_def()),
                    LogicalCallableTarget::Closure,
                ),
                params,
                ret_ty: callable.ret_ty(db),
                capability,
            };
        }

        let callable_def = callable.callable_def();
        let skip = usize::from(is_method_call);
        let params = (skip..callable_def.arg_tys(db).len())
            .filter_map(|idx| {
                let carrier = callable.arg_ty(db, idx)?;
                let mode = match callable_def {
                    CallableDef::Func(func) => func
                        .params(db)
                        .nth(idx)
                        .map(|param| param.mode(db).into())
                        .unwrap_or(LogicalCallableParamMode::View),
                    CallableDef::VariantCtor(_) => ClosureParamMode::try_from_carrier(db, carrier)
                        .unwrap_or(ClosureParamMode::View)
                        .into(),
                };
                let name = callable_def
                    .param_label_or_name(db, idx)
                    .and_then(|name| match name {
                        FuncParamName::Ident(name) => Some(name),
                        FuncParamName::Underscore => None,
                    });
                let ty = match mode {
                    LogicalCallableParamMode::Own => carrier,
                    LogicalCallableParamMode::View
                    | LogicalCallableParamMode::Ref
                    | LogicalCallableParamMode::Mut => carrier
                        .as_capability(db)
                        .map_or(carrier, |(_, payload)| payload),
                };
                Some(LogicalCallableParam { name, mode, ty })
            })
            .collect();

        Self {
            target: LogicalCallableTarget::Definition(callable_def),
            params,
            ret_ty: callable.ret_ty(db),
            capability: None,
        }
    }
}

/// Return the source impl blocks that can make `ty` callable in `scope`.
///
/// Generic self types are specialized against `ty`, and impls with
/// unsatisfied bounds are excluded.
pub fn applicable_callable_impls<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: crate::hir_def::scope_graph::ScopeId<'db>,
) -> Vec<ImplTrait<'db>> {
    let payload = ty.as_capability(db).map_or(ty, |(_, payload)| payload);
    applicable_callable_impl_candidates(db, payload, scope)
        .into_iter()
        .map(|(impl_trait, _, _)| impl_trait)
        .collect()
}

fn applicable_callable_impl_candidates<'db>(
    db: &'db dyn HirAnalysisDb,
    payload: TyId<'db>,
    scope: crate::hir_def::scope_graph::ScopeId<'db>,
) -> Vec<(ImplTrait<'db>, TraitInstId<'db>, ClosureCallTrait)> {
    let impl_traits = match payload.adt_ref(db) {
        Some(AdtRef::Struct(struct_)) => struct_.all_impl_traits(db),
        Some(AdtRef::Enum(enum_)) => enum_.all_impl_traits(db),
        None => Vec::new(),
    };
    impl_traits
        .into_iter()
        .filter_map(|impl_trait| {
            let inst = instantiate_impl_trait_for_ty(db, impl_trait, payload, scope)?;
            let call_trait = ClosureCallTrait::for_trait(db, scope, inst.def(db))?;
            Some((impl_trait, inst, call_trait))
        })
        .collect()
}

fn instantiate_impl_trait_for_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_trait: ImplTrait<'db>,
    ty: TyId<'db>,
    scope: crate::hir_def::scope_graph::ScopeId<'db>,
) -> Option<TraitInstId<'db>> {
    let implementor = lower_impl_trait(db, impl_trait)?;
    let mut table = UnificationTable::new(db);
    let implementor = table.instantiate_with_fresh_vars(implementor);
    table.unify(implementor.self_ty(db), ty).ok()?;
    let solve_cx = TraitSolveCx::new(db, scope).with_assumptions(enclosing_assumptions(db, scope));
    for &constraint in implementor.constraints(db).list(db) {
        let constraint = constraint.fold_with(db, &mut table);
        if matches!(
            is_goal_satisfiable(db, solve_cx, constraint),
            GoalSatisfiability::UnSat(_)
        ) {
            return None;
        }
    }
    Some(implementor.trait_inst(db).fold_with(db, &mut table))
}

fn signature_from_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    inst: TraitInstId<'db>,
    call_trait: ClosureCallTrait,
    target: Option<CallableDef<'db>>,
) -> Option<LogicalCallableSignature<'db>> {
    let args = inst.args(db);
    let pack = *args.get(1)?;
    if !pack.is_tuple(db) {
        return None;
    }
    let ret_ty = *args.get(2)?;
    let params = pack
        .field_types(db)
        .into_iter()
        .map(|carrier| {
            let mode =
                ClosureParamMode::try_from_carrier(db, carrier).unwrap_or(ClosureParamMode::View);
            LogicalCallableParam {
                name: None,
                mode: mode.into(),
                ty: mode.payload(db, carrier),
            }
        })
        .collect();
    let capability = match call_trait {
        ClosureCallTrait::Fn => LogicalCallableCapability::Reusable,
        ClosureCallTrait::FnOnce => LogicalCallableCapability::Consuming,
    };
    Some(LogicalCallableSignature {
        target: LogicalCallableTarget::Definition(target?),
        params,
        ret_ty,
        capability: Some(capability),
    })
}

impl<'db> Body<'db> {
    /// Type-check this body regardless of whether it belongs to a function,
    /// contract initializer, or contract receive arm.
    pub fn typed_body(self, db: &'db dyn HirAnalysisDb) -> Option<&'db TypedBody<'db>> {
        typed_body_for_body(db, self)
    }
}

fn concrete_closure_for_callable<'db>(
    db: &'db dyn HirAnalysisDb,
    callable: &Callable<'db>,
) -> Option<ClosureTy<'db>> {
    let self_ty = callable.trait_inst()?.self_ty(db);
    let payload = self_ty
        .as_capability(db)
        .map_or(self_ty, |(_, payload)| payload);
    payload.as_closure(db)
}

fn closure_param_names<'db>(
    db: &'db dyn HirAnalysisDb,
    closure: ClosureTy<'db>,
) -> Vec<Option<IdentId<'db>>> {
    let def = closure.def(db);
    let Partial::Present(Expr::Closure { params, .. }) = def.expr.data(db, def.body) else {
        return Vec::new();
    };
    params.data(db).iter().map(|param| param.name()).collect()
}

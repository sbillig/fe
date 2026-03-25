use crate::core::hir_def::{
    ConstGenericArgValue, GenericArg, GenericArgListId, GenericParam, GenericParamOwner,
    GenericParamView, IdentId, KindBound as HirKindBound, Partial, PathId, PathKind,
    TypeAlias as HirTypeAlias, TypeBound, TypeId as HirTyId, TypeKind as HirTyKind, TypeMode,
    scope_graph::ScopeId,
};
use salsa::Update;
use smallvec::smallvec;

use super::{
    assoc_items::{TraitConstUseResolution, resolve_trait_const_use},
    collect_layout_hole_tys_in_order,
    const_expr::{ConstExpr, ConstExprId},
    const_ty::{ConstTyData, ConstTyId, EvaluatedConstTy, const_ty_from_selected_assoc_const},
    context::{AnalysisCx, ImplOverlay, LoweringMode, ProofCx},
    effects::{EffectKeyKind, effect_key_kind, resolve_normalized_type_effect_key},
    fold::{TyFoldable, TyFolder},
    trait_lower::lower_trait_ref_in_cx,
    trait_resolution::{PredicateListId, constraint::collect_constraints},
    ty_def::{InvalidCause, Kind, TyData, TyId, TyParam},
};
use crate::analysis::name_resolution::{
    PathRes, PathResErrorKind, ReceiverPathResolutionCx, resolve_path,
    resolve_path_from_receiver_ty,
};
use crate::analysis::{HirAnalysisDb, ty::binder::Binder};

/// Lowers the given HirTy to `TyId`.
#[salsa::tracked(cycle_fn=lower_hir_ty_cycle_recover, cycle_initial=lower_hir_ty_cycle_initial)]
pub fn lower_hir_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: HirTyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    lower_hir_ty_in_mode(db, ty, scope, assumptions, LoweringMode::Normal)
}

fn analysis_cx_for_mode<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    mode: LoweringMode<'db>,
) -> AnalysisCx<'db> {
    AnalysisCx::new(ProofCx::new(db, scope).with_assumptions(assumptions))
        .with_overlay(
            mode.current_impl()
                .map(ImplOverlay::with_current_impl)
                .unwrap_or_default(),
        )
        .with_mode(mode)
}

pub fn lower_hir_ty_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: HirTyId<'db>,
    scope: ScopeId<'db>,
    cx: &AnalysisCx<'db>,
) -> TyId<'db> {
    match ty.data(db) {
        HirTyKind::Ptr(pointee) => {
            let pointee = lower_opt_hir_ty_in_cx(db, *pointee, scope, cx);
            let ptr = TyId::ptr(db);
            TyId::app(db, ptr, pointee)
        }

        HirTyKind::Mode(type_mode, inner) => {
            let inner = lower_opt_hir_ty_in_cx(db, *inner, scope, cx);
            match type_mode {
                TypeMode::Mut => TyId::borrow_mut_of(db, inner),
                TypeMode::Ref => TyId::borrow_ref_of(db, inner),
                TypeMode::Own => inner,
            }
        }

        HirTyKind::Path(path) => lower_path_in_cx(db, scope, *path, cx),

        HirTyKind::Tuple(tuple_id) => {
            let elems = tuple_id.data(db);
            let len = elems.len();
            let tuple = TyId::tuple(db, len);
            elems.iter().fold(tuple, |acc, &elem| {
                let elem_ty = lower_opt_hir_ty_in_cx(db, elem, scope, cx);
                if !elem_ty.has_star_kind(db) {
                    return TyId::invalid(db, InvalidCause::NotFullyApplied);
                }

                TyId::app(db, acc, elem_ty)
            })
        }

        HirTyKind::Array(hir_elem_ty, len) => {
            let elem_ty = lower_opt_hir_ty_in_cx(db, *hir_elem_ty, scope, cx);
            let len_ty = ConstTyId::from_opt_body_in_mode(db, *len, cx.mode);
            let len_ty = TyId::const_ty(db, len_ty);
            let array = TyId::array(db, elem_ty);
            TyId::app(db, array, len_ty)
        }

        HirTyKind::Never => TyId::never(db),
    }
}

/// Compatibility wrapper for callers that only have `(scope, assumptions,
/// mode)`. Contextual callers should prefer `lower_hir_ty_in_cx(...)` so they
/// preserve the live proof frontier.
pub fn lower_hir_ty_in_mode<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: HirTyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    mode: LoweringMode<'db>,
) -> TyId<'db> {
    let cx = analysis_cx_for_mode(db, scope, assumptions, mode);
    lower_hir_ty_in_cx(db, ty, scope, &cx)
}

pub fn lower_opt_hir_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: Partial<HirTyId<'db>>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    lower_opt_hir_ty_in_mode(db, ty, scope, assumptions, LoweringMode::Normal)
}

pub fn lower_opt_hir_ty_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: Partial<HirTyId<'db>>,
    scope: ScopeId<'db>,
    cx: &AnalysisCx<'db>,
) -> TyId<'db> {
    ty.to_opt()
        .map(|hir_ty| lower_hir_ty_in_cx(db, hir_ty, scope, cx))
        .unwrap_or_else(|| TyId::invalid(db, InvalidCause::ParseError))
}

pub fn lower_opt_hir_ty_in_mode<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: Partial<HirTyId<'db>>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    mode: LoweringMode<'db>,
) -> TyId<'db> {
    let cx = analysis_cx_for_mode(db, scope, assumptions, mode);
    lower_opt_hir_ty_in_cx(db, ty, scope, &cx)
}

pub(crate) fn contextual_path_resolution_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: PathId<'db>,
    resolve_tail_as_value: bool,
    cx: &AnalysisCx<'db>,
) -> Option<PathRes<'db>> {
    let mode_trait_inst = cx.mode.trait_inst()?;
    let current_self_ty = cx
        .mode
        .self_ty()
        .unwrap_or_else(|| mode_trait_inst.self_ty(db));
    if path.is_self_ty(db) && path.generic_args(db).is_empty(db) {
        return Some(PathRes::Ty(current_self_ty));
    }
    if let PathKind::QualifiedType { type_, trait_ } = path.kind(db) {
        let receiver_ty = if type_.is_self_ty(db) {
            current_self_ty
        } else {
            lower_hir_ty_in_cx(db, type_, scope, cx)
        };
        let trait_inst = lower_trait_ref_in_cx(db, receiver_ty, trait_, scope, *cx, None).ok()?;
        return Some(PathRes::Ty(TyId::qualified_ty(db, trait_inst)));
    }

    let receiver = path.parent(db)?;
    let name = path.ident(db).to_opt()?;
    let receiver_res = contextual_path_resolution_in_cx(db, scope, receiver, false, cx)
        .or_else(|| resolve_path(db, receiver, scope, cx.proof.assumptions(), false).ok())?;
    let receiver_ty = match receiver_res {
        PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => ty,
        _ => return None,
    };

    if receiver_ty == current_self_ty {
        let trait_inst = super::trait_def::specialize_trait_const_inst_to_receiver(
            db,
            receiver_ty,
            mode_trait_inst,
        );
        if resolve_tail_as_value && trait_inst.def(db).const_(db, name).is_some() {
            return Some(PathRes::TraitConst(receiver_ty, trait_inst, name));
        }
        let assoc_ty = trait_inst.assoc_ty(db, name)?;
        let seg_args = lower_generic_arg_list_in_cx(db, path.generic_args(db), scope, cx);
        let assoc_ty = if seg_args.is_empty() {
            assoc_ty
        } else {
            TyId::foldl(db, assoc_ty, &seg_args)
        };
        return Some(PathRes::Ty(assoc_ty));
    }

    resolve_path_from_receiver_ty(
        db,
        receiver_ty,
        ReceiverPathResolutionCx {
            parent_res: Some(receiver_res),
            path,
            scope,
            assumptions: cx.proof.assumptions(),
            resolve_tail_as_value,
            is_tail: true,
        },
    )
    .ok()
}

fn lower_trait_const_path_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    recv_ty: TyId<'db>,
    inst: crate::analysis::ty::trait_def::TraitInstId<'db>,
    name: IdentId<'db>,
) -> Option<TyId<'db>> {
    let inst = super::trait_def::specialize_trait_const_inst_to_receiver(db, recv_ty, inst);
    let resolution = resolve_trait_const_use(db, cx, inst, name)?;
    let ty = match resolution {
        TraitConstUseResolution::Concrete(selection) => {
            TyId::const_ty(db, const_ty_from_selected_assoc_const(db, &selection)?)
        }
        TraitConstUseResolution::Abstract {
            trait_inst,
            name,
            declared_ty,
        } => {
            let expr = ConstExprId::new(
                db,
                ConstExpr::TraitConst {
                    inst: trait_inst,
                    name,
                },
            );
            TyId::const_ty(
                db,
                ConstTyId::new(db, ConstTyData::Abstract(expr, declared_ty)),
            )
        }
        TraitConstUseResolution::MissingConcreteImpl {
            trait_inst, name, ..
        } => TyId::const_ty(
            db,
            ConstTyId::invalid(
                db,
                InvalidCause::TraitConstNotImplemented {
                    inst: trait_inst,
                    name,
                },
            ),
        ),
    };
    Some(ty)
}

fn lower_path_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: Partial<PathId<'db>>,
    cx: &AnalysisCx<'db>,
) -> TyId<'db> {
    let Some(path) = path.to_opt() else {
        return TyId::invalid(db, InvalidCause::ParseError);
    };
    let use_mode_for_generic_args =
        !matches!(cx.mode, LoweringMode::Normal) && !path.generic_args(db).is_empty(db);
    let resolve_path_id = if use_mode_for_generic_args {
        path.strip_generic_args(db)
    } else {
        path
    };

    let contextual = contextual_path_resolution_in_cx(db, scope, path, false, cx);
    let used_contextual = contextual.is_some();
    match contextual
        .map(Ok)
        .unwrap_or_else(|| resolve_path(db, resolve_path_id, scope, cx.proof.assumptions(), false))
    {
        Ok(PathRes::Ty(ty) | PathRes::TyAlias(_, ty) | PathRes::Func(ty)) => {
            if use_mode_for_generic_args && !used_contextual {
                let seg_args = lower_generic_arg_list_in_cx(db, path.generic_args(db), scope, cx);
                TyId::foldl(db, ty, &seg_args)
            } else {
                ty
            }
        }
        Ok(res) => TyId::invalid(db, InvalidCause::NotAType(res)),
        Err(err) => {
            // Try to resolve as a value, to find a matching `const` definition
            if matches!(err.kind, PathResErrorKind::NotFound { .. })
                && let Ok(resolved) = contextual_path_resolution_in_cx(db, scope, path, true, cx)
                    .map(Ok)
                    .unwrap_or_else(|| {
                        resolve_path(db, resolve_path_id, scope, cx.proof.assumptions(), true)
                    })
            {
                return match resolved {
                    PathRes::Const(const_def, ty) => {
                        if let Some(body) = const_def.body(db).to_opt() {
                            let const_ty =
                                ConstTyId::from_body(db, body, Some(ty), Some(const_def));
                            TyId::const_ty(db, const_ty)
                        } else {
                            TyId::invalid(db, InvalidCause::ParseError)
                        }
                    }
                    PathRes::TraitConst(recv_ty, inst, name) => {
                        lower_trait_const_path_in_cx(db, cx, recv_ty, inst, name)
                            .unwrap_or_else(|| TyId::invalid(db, InvalidCause::Other))
                    }
                    other => TyId::invalid(db, InvalidCause::NotAType(other)),
                };
            }

            TyId::invalid(db, InvalidCause::PathResolutionFailed { path })
        }
    }
}

fn lower_path_in_mode<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: Partial<PathId<'db>>,
    assumptions: PredicateListId<'db>,
    mode: LoweringMode<'db>,
) -> TyId<'db> {
    let cx = analysis_cx_for_mode(db, scope, assumptions, mode);
    lower_path_in_cx(db, scope, path, &cx)
}

fn lower_hir_ty_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    _ty: HirTyId<'db>,
    _scope: ScopeId<'db>,
    _assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    // On cycles during type lowering, treat the type as invalid so that
    // callers can emit a suitable diagnostic without recursing further.
    TyId::invalid(db, InvalidCause::Other)
}

fn lower_hir_ty_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &TyId<'db>,
    _count: u32,
    _ty: HirTyId<'db>,
    _scope: ScopeId<'db>,
    _assumptions: PredicateListId<'db>,
) -> salsa::CycleRecoveryAction<TyId<'db>> {
    // Keep iterating until we reach a fixpoint; the initial value is
    // already marked invalid, so subsequent iterations will converge
    // quickly without panicking.
    salsa::CycleRecoveryAction::Iterate
}

fn lower_const_ty_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    ty: HirTyId<'db>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    let HirTyKind::Path(path) = ty.data(db) else {
        return TyId::invalid(db, InvalidCause::InvalidConstParamTy);
    };

    if !path
        .to_opt()
        .map(|p| p.generic_args(db).is_empty(db))
        .unwrap_or(true)
    {
        return TyId::invalid(db, InvalidCause::InvalidConstParamTy);
    }
    let ty = lower_path_in_mode(db, scope, *path, assumptions, LoweringMode::Normal);

    if ty.has_invalid(db)
        || ty.is_integral(db)
        || ty.is_bool(db)
        || ty.is_unit_variant_only_enum(db)
    {
        ty
    } else {
        TyId::invalid(db, InvalidCause::InvalidConstParamTy)
    }
}

/// Collects the generic parameters of the given generic parameter owner.
#[salsa::tracked(
    cycle_initial=collect_generic_params_cycle_initial,
    cycle_fn=collect_generic_params_cycle_recover
)]
pub(crate) fn collect_generic_params<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: GenericParamOwner<'db>,
) -> GenericParamTypeSet<'db> {
    GenericParamCollector::new(db, owner).finalize()
}

fn collect_generic_params_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: GenericParamOwner<'db>,
) -> GenericParamTypeSet<'db> {
    GenericParamTypeSet::empty(db, owner.scope())
}

fn collect_generic_params_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &GenericParamTypeSet<'db>,
    _count: u32,
    _owner: GenericParamOwner<'db>,
) -> salsa::CycleRecoveryAction<GenericParamTypeSet<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

pub(crate) fn method_receiver_layout_hole_tys<'db>(
    db: &'db dyn HirAnalysisDb,
    func: crate::hir_def::Func<'db>,
) -> Vec<TyId<'db>> {
    if !func.is_method(db) {
        return Vec::new();
    }
    let Some(expected_self_ty) = func.expected_self_ty(db) else {
        return Vec::new();
    };
    collect_layout_hole_tys_in_order(db, expected_self_ty)
}

/// Lowers the given type alias to [`TyAlias`].
#[salsa::tracked(return_ref, cycle_fn=lower_type_alias_cycle_recover, cycle_initial=lower_type_alias_cycle_initial)]
pub(crate) fn lower_type_alias<'db>(
    db: &'db dyn HirAnalysisDb,
    alias: HirTypeAlias<'db>,
) -> TyAlias<'db> {
    crate::core::semantic::lower_type_alias_body(db, alias)
}

pub(crate) fn lower_type_alias_from_hir<'db>(
    db: &'db dyn HirAnalysisDb,
    alias: HirTypeAlias<'db>,
    alias_type_ref: Option<HirTyId<'db>>,
) -> TyAlias<'db> {
    let param_set = collect_generic_params(db, alias.into());

    let Some(hir_ty) = alias_type_ref else {
        return TyAlias {
            alias,
            alias_to: Binder::bind(TyId::invalid(db, InvalidCause::ParseError)),
            param_set,
        };
    };

    let assumptions = collect_constraints(db, alias.into()).instantiate_identity();
    let alias_to = lower_hir_ty(db, hir_ty, alias.scope(), assumptions);
    let alias_to = if let TyData::Invalid(InvalidCause::AliasCycle(cycle)) = alias_to.data(db) {
        if cycle.contains(&alias) {
            alias_to
        } else {
            let mut cycle = cycle.clone();
            cycle.push(alias);
            TyId::invalid(db, InvalidCause::AliasCycle(cycle))
        }
    } else if alias_to.has_invalid(db) {
        // Should be reported by TypeAliasAnalysisPass
        TyId::invalid(db, InvalidCause::Other)
    } else {
        alias_to
    };
    TyAlias {
        alias,
        alias_to: Binder::bind(alias_to),
        param_set,
    }
}

fn lower_type_alias_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    alias: HirTypeAlias<'db>,
) -> TyAlias<'db> {
    TyAlias {
        alias,
        alias_to: Binder::bind(TyId::invalid(
            db,
            InvalidCause::AliasCycle(smallvec![alias]),
        )),
        param_set: GenericParamTypeSet::empty(db, alias.scope()),
    }
}

fn lower_type_alias_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &TyAlias<'db>,
    _count: u32,
    _alias: HirTypeAlias<'db>,
) -> salsa::CycleRecoveryAction<TyAlias<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

#[doc(hidden)]
#[salsa::tracked(return_ref, cycle_initial=evaluate_params_precursor_cycle_initial, cycle_fn=evaluate_params_precursor_cycle_recover)]
pub(crate) fn evaluate_params_precursor<'db>(
    db: &'db dyn HirAnalysisDb,
    set: GenericParamTypeSet<'db>,
) -> Vec<TyId<'db>> {
    set.params_precursor(db)
        .iter()
        .enumerate()
        .map(|(i, p)| p.evaluate(db, set.scope(db), i))
        .collect()
}

fn evaluate_params_precursor_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    set: GenericParamTypeSet<'db>,
) -> Vec<TyId<'db>> {
    set.params_precursor(db)
        .iter()
        .map(|_| TyId::invalid(db, InvalidCause::Other))
        .collect()
}

fn evaluate_params_precursor_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &Vec<TyId<'db>>,
    _count: u32,
    _set: GenericParamTypeSet<'db>,
) -> salsa::CycleRecoveryAction<Vec<TyId<'db>>> {
    salsa::CycleRecoveryAction::Iterate
}

/// Represents a lowered type alias. `TyAlias` itself isn't a type, but
/// can be instantiated to a `TyId` by substituting its type
/// parameters with actual types.
///
/// NOTE: `TyAlias` can't become an alias to partial applied types, i.e., the
/// right hand side of the alias declaration must be a fully applied type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Update)]
pub struct TyAlias<'db> {
    pub alias: HirTypeAlias<'db>,
    pub alias_to: Binder<TyId<'db>>,
    pub param_set: GenericParamTypeSet<'db>,
}

impl<'db> TyAlias<'db> {
    pub fn params(&self, db: &'db dyn HirAnalysisDb) -> &'db [TyId<'db>] {
        self.param_set.params(db)
    }
}

pub(crate) fn lower_generic_arg_list<'db>(
    db: &'db dyn HirAnalysisDb,
    args: GenericArgListId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> Vec<TyId<'db>> {
    lower_generic_arg_list_in_mode(db, args, scope, assumptions, LoweringMode::Normal)
}

pub(crate) fn lower_generic_arg_list_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    args: GenericArgListId<'db>,
    scope: ScopeId<'db>,
    cx: &AnalysisCx<'db>,
) -> Vec<TyId<'db>> {
    let assumptions = cx.proof.assumptions();
    args.data(db)
        .iter()
        .map(|arg| match arg {
            GenericArg::Type(ty_arg) => {
                // Generic args are syntactically ambiguous: `String<N>` may parse `N` as a type
                // even when `String` expects a const generic arg. When a type-arg is a path that
                // resolves as a value const/trait-const, lower it as a const-ty argument so
                // downstream `TyId::app` sees a const generic.
                if let Some(hir_ty) = ty_arg.ty.to_opt()
                    && let HirTyKind::Path(path) = hir_ty.data(db)
                    && let Some(path) = path.to_opt()
                    && let Ok(resolved) =
                        contextual_path_resolution_in_cx(db, scope, path, true, cx)
                            .map(Ok)
                            .unwrap_or_else(|| resolve_path(db, path, scope, assumptions, true))
                {
                    match resolved {
                        PathRes::Const(const_def, ty) => {
                            if let Some(body) = const_def.body(db).to_opt() {
                                let const_ty =
                                    ConstTyId::from_body(db, body, Some(ty), Some(const_def));
                                return TyId::const_ty(db, const_ty);
                            }
                            return TyId::invalid(db, InvalidCause::ParseError);
                        }
                        PathRes::TraitConst(recv_ty, inst, name) => {
                            if let Some(ty) =
                                lower_trait_const_path_in_cx(db, cx, recv_ty, inst, name)
                            {
                                return ty;
                            }
                        }
                        PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => {
                            if let TyData::ConstTy(const_ty) = ty.data(db) {
                                return TyId::const_ty(db, *const_ty);
                            }
                        }
                        PathRes::EnumVariant(variant)
                            if variant.ty.is_unit_variant_only_enum(db) =>
                        {
                            let evaluated = EvaluatedConstTy::EnumVariant(variant.variant);
                            let const_ty =
                                ConstTyId::new(db, ConstTyData::Evaluated(evaluated, variant.ty));
                            return TyId::const_ty(db, const_ty);
                        }
                        _ => {}
                    }
                }
                lower_opt_hir_ty_in_cx(db, ty_arg.ty, scope, cx)
            }
            GenericArg::Const(const_arg) => match const_arg.value {
                ConstGenericArgValue::Expr(body) => {
                    let const_ty = ConstTyId::from_opt_body_in_mode(db, body, cx.mode);
                    TyId::const_ty(db, const_ty)
                }
                ConstGenericArgValue::Hole => TyId::const_ty(db, ConstTyId::hole(db)),
            },

            GenericArg::AssocType(_assoc_type_arg) => {
                // TODO: ?
                TyId::invalid(db, InvalidCause::Other)
            }
        })
        .collect()
}

pub(crate) fn lower_generic_arg_list_in_mode<'db>(
    db: &'db dyn HirAnalysisDb,
    args: GenericArgListId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    mode: LoweringMode<'db>,
) -> Vec<TyId<'db>> {
    let cx = analysis_cx_for_mode(db, scope, assumptions, mode);
    lower_generic_arg_list_in_cx(db, args, scope, &cx)
}

#[salsa::interned]
#[derive(Debug)]
pub struct GenericParamTypeSet<'db> {
    #[return_ref]
    pub(crate) params_precursor: Vec<TyParamPrecursor<'db>>,
    pub(crate) scope: ScopeId<'db>,
    offset_to_explicit: usize,
}

impl<'db> GenericParamTypeSet<'db> {
    pub(crate) fn params(self, db: &'db dyn HirAnalysisDb) -> &'db [TyId<'db>] {
        evaluate_params_precursor(db, self)
    }

    pub(crate) fn explicit_params(self, db: &'db dyn HirAnalysisDb) -> &'db [TyId<'db>] {
        let offset = self.offset_to_explicit(db);
        &self.params(db)[offset..]
    }

    pub(crate) fn explicit_const_param_default_is_hole(
        self,
        db: &'db dyn HirAnalysisDb,
        explicit_idx: usize,
    ) -> bool {
        let idx = self.offset_to_explicit(db) + explicit_idx;
        let Some(param) = self.params_precursor(db).get(idx) else {
            return false;
        };
        matches!(param.variant, Variant::Const(_))
            && matches!(param.default_hir_const, Some(ConstGenericArgValue::Hole))
    }

    pub(crate) fn empty(db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> Self {
        Self::new(db, Vec::new(), scope, 0)
    }

    pub(crate) fn trait_self(&self, db: &'db dyn HirAnalysisDb) -> Option<TyId<'db>> {
        let params = self.params_precursor(db);
        let cand = params.first()?;

        if cand.is_trait_self() {
            Some(cand.evaluate(db, self.scope(db), 0))
        } else {
            None
        }
    }

    pub(crate) fn offset_to_explicit_params_position(&self, db: &dyn HirAnalysisDb) -> usize {
        self.offset_to_explicit(db)
    }

    pub(crate) fn param_by_original_idx(
        &self,
        db: &'db dyn HirAnalysisDb,
        original_idx: usize,
    ) -> Option<TyId<'db>> {
        let idx = self.offset_to_explicit(db) + original_idx;
        self.params_precursor(db)
            .get(idx)
            .map(|p| p.evaluate(db, self.scope(db), idx))
    }

    /// Given explicit generic args provided at the use site, append any trailing
    /// defaults from this param set and return the completed explicit arg list.
    ///
    /// - `provided_explicit`: args corresponding to the explicit params (i.e.,
    ///   skipping implicit ones like trait `Self`).
    /// - `implicit_bindings`: mapping of (lowered_idx -> TyId) for implicit
    ///   parameters that should be available when evaluating defaults (e.g.,
    ///   trait `Self` at index 0).
    pub(crate) fn complete_explicit_args_with_defaults(
        self,
        db: &'db dyn HirAnalysisDb,
        trait_self: Option<TyId<'db>>,
        provided_explicit: &[TyId<'db>],
        assumptions: PredicateListId<'db>,
    ) -> Vec<TyId<'db>> {
        let total = self.params_precursor(db).len();
        let offset = self.offset_to_explicit(db);

        // mapping from lowered param idx -> bound arg, used to substitute in defaults
        let mut mapping = vec![];
        if let Some(self_ty) = trait_self {
            mapping.push(Some(self_ty));
        }
        mapping.extend(provided_explicit.iter().map(|ty| Some(*ty)));
        mapping.resize(total, None);

        // Helper folder to substitute known params when lowering defaults
        struct ParamSubst<'a, 'db> {
            db: &'db dyn HirAnalysisDb,
            mapping: &'a [Option<TyId<'db>>],
        }
        impl<'a, 'db> TyFolder<'db> for ParamSubst<'a, 'db> {
            fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
                match ty.data(self.db) {
                    TyData::TyParam(param) => {
                        if let Some(Some(rep)) = self.mapping.get(param.idx) {
                            return *rep;
                        }
                        ty.super_fold_with(db, self)
                    }
                    TyData::ConstTy(const_ty) => {
                        if let super::const_ty::ConstTyData::TyParam(param, _) =
                            const_ty.data(self.db)
                            && let Some(Some(rep)) = self.mapping.get(param.idx)
                        {
                            return *rep;
                        }
                        ty.super_fold_with(db, self)
                    }
                    _ => ty.super_fold_with(db, self),
                }
            }
        }

        // Build the returned explicit arg list, appending defaults where available.
        let mut result: Vec<TyId<'db>> = provided_explicit.to_vec();
        let scope = self.scope(db);
        for i in (offset + provided_explicit.len())..total {
            let prec = &self.params_precursor(db)[i];

            if let Some(hir_ty) = prec.default_hir_ty {
                let lowered = lower_hir_ty(db, hir_ty, scope, assumptions);
                let lowered = {
                    let mut subst = ParamSubst {
                        db,
                        mapping: &mapping,
                    };
                    lowered.fold_with(db, &mut subst)
                };
                mapping[i] = Some(lowered);
                result.push(lowered);
                continue;
            }

            if let Some(default) = prec.default_hir_const {
                let expected = prec.evaluate(db, scope, i).const_ty_ty(db);
                let lowered = match default {
                    ConstGenericArgValue::Expr(default) => {
                        let const_ty = ConstTyId::from_opt_body(db, default);
                        let lowered = TyId::const_ty(db, const_ty);
                        lowered
                            .evaluate_const_ty(db, expected)
                            .unwrap_or_else(|cause| TyId::invalid(db, cause))
                    }
                    ConstGenericArgValue::Hole => TyId::const_ty(
                        db,
                        ConstTyId::hole_with_ty(
                            db,
                            expected.unwrap_or_else(|| TyId::invalid(db, InvalidCause::Other)),
                        ),
                    ),
                };

                let lowered = {
                    let mut subst = ParamSubst {
                        db,
                        mapping: &mapping,
                    };
                    lowered.fold_with(db, &mut subst)
                };

                mapping[i] = Some(lowered);
                result.push(lowered);
                continue;
            }

            break; // Missing non-default; stop filling further params
        }

        result
    }
}

struct GenericParamCollector<'db> {
    db: &'db dyn HirAnalysisDb,
    owner: GenericParamOwner<'db>,
    params: Vec<TyParamPrecursor<'db>>,
    offset_to_original: usize,
}

impl<'db> GenericParamCollector<'db> {
    fn new(db: &'db dyn HirAnalysisDb, owner: GenericParamOwner<'db>) -> Self {
        let mut params = match owner {
            GenericParamOwner::Trait(_) => {
                vec![TyParamPrecursor::trait_self(db, None)]
            }

            GenericParamOwner::Func(func) if func.is_associated_func(db) => {
                let parent = owner.parent(db).unwrap();
                collect_generic_params(db, parent)
                    .params_precursor(db)
                    .to_vec()
            }

            _ => vec![],
        };

        if let GenericParamOwner::Func(func) = owner {
            for (layout_idx, hole_ty) in method_receiver_layout_hole_tys(db, func)
                .into_iter()
                .enumerate()
            {
                let name = IdentId::new(db, format!("__self_layout{layout_idx}"));
                params.push(TyParamPrecursor::implicit_const_param(
                    db,
                    Partial::Present(name),
                    hole_ty,
                ));
            }

            let assumptions = PredicateListId::empty_list(db);
            for effect in func.effect_params(db) {
                let Some(key_path) = effect.key_path(db) else {
                    continue;
                };
                if !matches!(
                    effect_key_kind(db, key_path, func.scope()),
                    EffectKeyKind::Type
                ) {
                    continue;
                }

                let Some(key_ty) =
                    resolve_normalized_type_effect_key(db, key_path, func.scope(), assumptions)
                else {
                    continue;
                };
                for (layout_idx, hole_ty) in collect_layout_hole_tys_in_order(db, key_ty)
                    .into_iter()
                    .enumerate()
                {
                    let name =
                        IdentId::new(db, format!("__efflayout{}_{}", effect.index(), layout_idx));
                    params.push(TyParamPrecursor::implicit_const_param(
                        db,
                        Partial::Present(name),
                        hole_ty,
                    ));
                }
            }
        }

        // For each effect parameter, insert an implicit generic parameter that carries the
        // concrete "provider type". This allows monomorphization to treat effects as ordinary
        // implicit generics and substitute a concrete provider at call sites.
        if let GenericParamOwner::Func(func) = owner {
            let mut provider_idx = 0usize;
            for effect in func.effect_params(db) {
                let Some(key_path) = effect.key_path(db) else {
                    continue;
                };
                if !matches!(
                    effect_key_kind(db, key_path, func.scope()),
                    EffectKeyKind::Type | EffectKeyKind::Trait
                ) {
                    continue;
                }

                let name = IdentId::new(db, format!("__effprov{provider_idx}"));
                provider_idx += 1;
                let prec_idx = params.len();
                params.push(TyParamPrecursor::effect_provider_param(
                    Partial::Present(name),
                    prec_idx,
                ));
            }
        }

        let offset_to_original = params.len();
        Self {
            db,
            owner,
            params,
            offset_to_original,
        }
    }

    fn collect_generic_params(&mut self) {
        let hir_db = self.db;
        let params = self.owner.params(hir_db);
        for (idx, param) in params
            .map(|GenericParamView { param, .. }| param)
            .enumerate()
        {
            let idx = idx + self.offset_to_original;

            match param {
                GenericParam::Type(param) => {
                    let name = param.name;

                    let kind = lower_kind_in_bounds(param.bounds.as_slice());
                    let default_hir_ty = param.default_ty;
                    self.params
                        .push(TyParamPrecursor::ty_param(name, idx, kind, default_hir_ty));
                }

                GenericParam::Const(param) => {
                    let name = param.name;
                    let hir_ty = param.ty.to_opt();
                    let default = param.default;

                    self.params
                        .push(TyParamPrecursor::const_ty_param(name, idx, hir_ty, default))
                }
            }
        }
    }

    fn collect_kind_in_where_clause(&mut self) {
        let Some(where_clause_owner) = self.owner.where_clause_owner() else {
            return;
        };

        let hir_db = self.db;
        let where_clause = where_clause_owner.clause(hir_db);
        for pred in where_clause.predicates(hir_db) {
            let Some(kind) = pred.kind(self.db) else {
                continue;
            };

            // Kind bound on a concrete type parameter in this owner.
            if let Some(orig_idx) = pred.param_original_index(hir_db) {
                let idx = orig_idx + self.offset_to_original;
                if let Some(param) = self.params.get_mut(idx)
                    && param.kind.is_none()
                    && !param.is_const_ty()
                {
                    param.kind = Some(kind.clone());
                }
                continue;
            }

            // Kind bound on `Self` in a trait owner.
            if pred.is_self_subject(hir_db)
                && matches!(self.owner, GenericParamOwner::Trait(_))
                && let Some(trait_self) = self.trait_self_ty_mut()
                && trait_self.kind.is_none()
            {
                trait_self.kind = Some(kind);
            }
        }
    }

    fn finalize(mut self) -> GenericParamTypeSet<'db> {
        self.collect_generic_params();
        self.collect_kind_in_where_clause();

        GenericParamTypeSet::new(
            self.db,
            self.params,
            self.owner.scope(),
            self.offset_to_original,
        )
    }

    fn trait_self_ty_mut(&mut self) -> Option<&mut TyParamPrecursor<'db>> {
        let cand = self.params.get_mut(0)?;
        cand.is_trait_self().then_some(cand)
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TyParamPrecursor<'db> {
    name: Partial<IdentId<'db>>,
    original_idx: Option<usize>,
    kind: Option<Kind>,
    variant: Variant<'db>,
    default_hir_ty: Option<HirTyId<'db>>, // Only used for type params
    default_hir_const: Option<ConstGenericArgValue<'db>>, // Only used for const params
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Variant<'db> {
    TraitSelf,
    Normal,
    Const(Option<HirTyId<'db>>),
    EffectProvider,
    ImplicitConst(TyId<'db>),
}

impl<'db> TyParamPrecursor<'db> {
    fn evaluate(
        &self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        lowered_idx: usize,
    ) -> TyId<'db> {
        let Partial::Present(name) = self.name else {
            return TyId::invalid(db, InvalidCause::Other);
        };

        let kind = self.kind.clone().unwrap_or(Kind::Star);

        match self.variant {
            Variant::TraitSelf => {
                let param = TyParam::trait_self(db, kind, scope);
                TyId::new(db, TyData::TyParam(param))
            }
            Variant::Normal => {
                let param = TyParam::normal_param(name, lowered_idx, kind, scope);
                TyId::new(db, TyData::TyParam(param))
            }
            Variant::EffectProvider => {
                let param = TyParam::effect_provider_param(name, lowered_idx, scope);
                TyId::new(db, TyData::TyParam(param))
            }
            Variant::Const(Some(ty)) => {
                let param = TyParam::normal_param(name, lowered_idx, kind, scope);
                let ty = lower_const_ty_ty(db, scope, ty, PredicateListId::empty_list(db)); // xxx fixme
                let const_ty = ConstTyId::new(db, ConstTyData::TyParam(param, ty));
                TyId::new(db, TyData::ConstTy(const_ty))
            }
            Variant::Const(None) => TyId::invalid(db, InvalidCause::Other),
            Variant::ImplicitConst(const_ty_ty) => {
                let param = TyParam::implicit_param(name, lowered_idx, kind, scope);
                let const_ty = ConstTyId::new(db, ConstTyData::TyParam(param, const_ty_ty));
                TyId::new(db, TyData::ConstTy(const_ty))
            }
        }
    }

    fn ty_param(
        name: Partial<IdentId<'db>>,
        idx: usize,
        kind: Option<Kind>,
        default_hir_ty: Option<HirTyId<'db>>,
    ) -> Self {
        Self {
            name,
            original_idx: idx.into(),
            kind,
            variant: Variant::Normal,
            default_hir_ty,
            default_hir_const: None,
        }
    }

    fn const_ty_param(
        name: Partial<IdentId<'db>>,
        idx: usize,
        ty: Option<HirTyId<'db>>,
        default: Option<ConstGenericArgValue<'db>>,
    ) -> Self {
        Self {
            name,
            original_idx: idx.into(),
            kind: None,
            variant: Variant::Const(ty),
            default_hir_ty: None,
            default_hir_const: default,
        }
    }

    fn effect_provider_param(name: Partial<IdentId<'db>>, idx: usize) -> Self {
        Self {
            name,
            original_idx: idx.into(),
            kind: Some(Kind::Star),
            variant: Variant::EffectProvider,
            default_hir_ty: None,
            default_hir_const: None,
        }
    }

    fn implicit_const_param(
        db: &'db dyn HirAnalysisDb,
        name: Partial<IdentId<'db>>,
        ty: TyId<'db>,
    ) -> Self {
        Self {
            name,
            original_idx: None,
            kind: Some(ty.kind(db).clone()),
            variant: Variant::ImplicitConst(ty),
            default_hir_ty: None,
            default_hir_const: None,
        }
    }

    fn trait_self(db: &'db dyn HirAnalysisDb, kind: Option<Kind>) -> Self {
        let name = Partial::Present(IdentId::make_self_ty(db));
        Self {
            name,
            original_idx: None,
            kind,
            variant: Variant::TraitSelf,
            default_hir_ty: None,
            default_hir_const: None,
        }
    }

    fn is_trait_self(&self) -> bool {
        matches!(self.variant, Variant::TraitSelf)
    }

    fn is_const_ty(&self) -> bool {
        matches!(self.variant, Variant::Const(_) | Variant::ImplicitConst(_))
    }
}

pub(super) fn lower_kind(kind: &HirKindBound) -> Kind {
    match kind {
        HirKindBound::Mono => Kind::Star,
        HirKindBound::Abs(lhs, rhs) => match (lhs, rhs) {
            (Partial::Present(lhs), Partial::Present(rhs)) => {
                Kind::Abs(Box::new((lower_kind(lhs), lower_kind(rhs))))
            }
            (Partial::Present(lhs), Partial::Absent) => {
                Kind::Abs(Box::new((lower_kind(lhs), Kind::Any)))
            }
            (Partial::Absent, Partial::Present(rhs)) => {
                Kind::Abs(Box::new((Kind::Any, lower_kind(rhs))))
            }
            (Partial::Absent, Partial::Absent) => Kind::Abs(Box::new((Kind::Any, Kind::Any))),
        },
    }
}

/// Helper for extracting a lowered kind from a slice of HIR `TypeBound`s.
/// Returns the first kind bound if present.
pub(super) fn lower_kind_in_bounds<'db>(bounds: &[TypeBound<'db>]) -> Option<Kind> {
    for bound in bounds {
        if let TypeBound::Kind(Partial::Present(k)) = bound {
            return Some(lower_kind(k));
        }
    }
    None
}

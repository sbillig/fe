//! This module implements the trait and impl trait lowering process.

use crate::{
    core::hir_def::{
        AssocTypeGenericArg, ConstGenericArgValue, HirIngot, IdentId, ItemKind, Partial, PathId,
        Trait, TraitRefId, params::GenericArg, scope_graph::ScopeId,
    },
    hir_def::Func,
};
use common::{indexmap::IndexMap, ingot::Ingot};
use rustc_hash::FxHashMap;
use salsa::Update;

use super::{
    admission::{AdmissionEngine, AdmissionSummary, TraitImplTable},
    const_ty::ConstTyId,
    context::{AnalysisCx, ProofCx},
    fold::{TyFoldable, TyFolder},
    trait_def::{ImplementorId, TraitInstId},
    trait_resolution::{LocalImplementorSet, PredicateListId},
    ty_def::{InvalidCause, TyId},
    ty_lower::lower_hir_ty,
};
use crate::analysis::{
    HirAnalysisDb,
    name_resolution::{PathRes, PathResError, resolve_path},
    ty::{
        ty_def::{Kind, TyData},
        ty_lower::lower_opt_hir_ty,
    },
};

fn collect_trait_impls_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _ingot: Ingot<'db>,
) -> TraitImplTable<'db> {
    TraitImplTable::default()
}

fn collect_trait_impls_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &TraitImplTable<'db>,
    _count: u32,
    _ingot: Ingot<'db>,
) -> salsa::CycleRecoveryAction<TraitImplTable<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

/// Internal fixed-point frontier used during recursive trait-env admission.
///
/// Callers that need the stable final admitted impl set should use
/// [`collect_trait_impls`] instead.
#[salsa::tracked(
    return_ref,
    cycle_fn=collect_trait_impls_cycle_recover,
    cycle_initial=collect_trait_impls_cycle_initial
)]
pub(crate) fn collect_trait_impls_frontier<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> TraitImplTable<'db> {
    let const_impls = ingot
        .resolved_external_ingots(db)
        .iter()
        .map(|(_, external)| collect_trait_impls(db, *external))
        .collect();

    let impl_traits = ingot.all_impl_traits(db);
    AdmissionEngine::new(db, const_impls).collect(impl_traits)
}

fn admission_summary_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _ingot: Ingot<'db>,
) -> AdmissionSummary<'db> {
    AdmissionSummary::default()
}

fn admission_summary_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &AdmissionSummary<'db>,
    _count: u32,
    _ingot: Ingot<'db>,
) -> salsa::CycleRecoveryAction<AdmissionSummary<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

/// Collect all trait implementors in the ingot.
/// The returned table doesn't contain the const(external) ingot
/// implementors. If you need to obtain the environment that contains all
/// available implementors in the ingot, please use
/// [`TraitEnv`](super::trait_def::TraitEnv).
#[salsa::tracked(
    return_ref,
    cycle_fn=collect_trait_impls_cycle_recover,
    cycle_initial=collect_trait_impls_cycle_initial
)]
pub(crate) fn collect_trait_impls<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> TraitImplTable<'db> {
    admission_summary(db, ingot).admitted.clone()
}

pub(crate) fn final_local_implementors<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> LocalImplementorSet<'db> {
    let implementors_by_impl: FxHashMap<_, _> = admission_summary(db, ingot)
        .admitted
        .values()
        .flat_map(|implementors| {
            implementors
                .iter()
                .map(|implementor| (implementor.skip_binder().hir_impl_trait(db), *implementor))
        })
        .collect();
    let implementors: Vec<_> = ingot
        .all_impl_traits(db)
        .iter()
        .filter_map(|impl_trait| implementors_by_impl.get(impl_trait).copied())
        .collect();
    LocalImplementorSet::new(db, implementors)
}

#[salsa::tracked(
    return_ref,
    cycle_fn=admission_summary_cycle_recover,
    cycle_initial=admission_summary_cycle_initial
)]
pub(crate) fn admission_summary<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> AdmissionSummary<'db> {
    let const_impls = ingot
        .resolved_external_ingots(db)
        .iter()
        .map(|(_, external)| &admission_summary(db, *external).admitted)
        .collect();

    let impl_traits = ingot.all_impl_traits(db);
    AdmissionEngine::new(db, const_impls).summarize(impl_traits)
}

/// Lower a trait reference to a trait instance.
///
/// When `owner_self` is provided, it is used for substituting `Self` references in generic
/// arguments and associated type bindings, while `self_ty` is used as the implementor (args[0]).
/// This is needed for associated type bounds like `type Assoc: Encode<Self>` where `Self`
/// refers to the owner trait's Self, not the associated type.
#[salsa::tracked]
pub(crate) fn lower_trait_ref_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    self_ty: TyId<'db>,
    trait_ref: TraitRefId<'db>,
    scope: ScopeId<'db>,
    cx: AnalysisCx<'db>,
    owner_self: Option<TyId<'db>>,
) -> Result<TraitInstId<'db>, TraitRefLowerError<'db>> {
    let Partial::Present(path) = trait_ref.path(db) else {
        return Err(TraitRefLowerError::Ignored);
    };

    let self_subst = owner_self.unwrap_or(self_ty);

    match resolve_path(db, path, scope, cx.proof.assumptions(), false) {
        Ok(PathRes::Trait(t)) => {
            let mut args = t.args(db).clone();

            struct SelfSubst<'db> {
                db: &'db dyn HirAnalysisDb,
                self_subst: TyId<'db>,
            }

            impl<'db> TyFolder<'db> for SelfSubst<'db> {
                fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
                    match ty.data(self.db) {
                        TyData::TyParam(p) if p.is_trait_self() => self.self_subst,
                        _ => ty.super_fold_with(db, self),
                    }
                }
            }

            let mut folder = SelfSubst { db, self_subst };
            args[0] = self_ty;
            args.iter_mut()
                .skip(1)
                .for_each(|arg| *arg = arg.fold_with(db, &mut folder));

            let mut assoc_bindings = t.assoc_type_bindings(db).clone();
            assoc_bindings
                .iter_mut()
                .for_each(|(_, ty)| *ty = (*ty).fold_with(db, &mut folder));

            Ok(TraitInstId::new(db, t.key(db), args, assoc_bindings))
        }
        Ok(res) => Err(TraitRefLowerError::InvalidDomain(res)),
        Err(err) => Err(TraitRefLowerError::PathResError(err)),
    }
}

#[salsa::tracked]
pub(crate) fn lower_trait_ref<'db>(
    db: &'db dyn HirAnalysisDb,
    self_ty: TyId<'db>,
    trait_ref: TraitRefId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    owner_self: Option<TyId<'db>>,
) -> Result<TraitInstId<'db>, TraitRefLowerError<'db>> {
    let cx = AnalysisCx::new(ProofCx::new(db, scope).with_assumptions(assumptions));
    lower_trait_ref_in_cx(db, self_ty, trait_ref, scope, cx, owner_self)
}

pub(crate) enum TraitArgError<'db> {
    ArgNumMismatch {
        expected: usize,
        given: usize,
    },
    ArgKindMisMatch {
        // TODO: add index, improve diag display
        expected: Kind,
        given: TyId<'db>,
    },
    ArgTypeMismatch {
        expected: Option<TyId<'db>>,
        given: Option<TyId<'db>>,
    },
    ConstHoleNotAllowed {
        arg_idx: usize,
    },
    Ignored,
}

pub(crate) fn lower_trait_ref_impl<'db>(
    db: &'db (dyn HirAnalysisDb + 'static),
    path: PathId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    t: Trait<'db>,
) -> Result<TraitInstId<'db>, TraitArgError<'db>> {
    let trait_params: &[TyId<'db>] = t.params(db);
    let args = path.generic_args(db).data(db);

    // Lower provided explicit args (excluding Self)
    let mut provided_explicit = Vec::new();
    for (arg_idx, arg) in args.iter().enumerate() {
        match arg {
            GenericArg::Type(ty_arg) => {
                provided_explicit.push(lower_opt_hir_ty(db, ty_arg.ty, scope, assumptions));
            }
            GenericArg::Const(const_arg) => match const_arg.value {
                ConstGenericArgValue::Expr(body) => {
                    provided_explicit.push(TyId::const_ty(db, ConstTyId::from_opt_body(db, body)));
                }
                ConstGenericArgValue::Hole => {
                    return Err(TraitArgError::ConstHoleNotAllowed { arg_idx });
                }
            },
            GenericArg::AssocType(..) => {}
        }
    }

    // Fill trailing defaults using the trait's param set. Bind Self (idx 0).
    let non_self_completed = t.param_set(db).complete_explicit_args_with_defaults(
        db,
        Some(t.self_param(db)),
        &provided_explicit,
        assumptions,
    );

    if non_self_completed.len() != trait_params.len() - 1 {
        return Err(TraitArgError::ArgNumMismatch {
            expected: trait_params.len() - 1,
            given: non_self_completed.len(),
        });
    }

    let mut final_args: Vec<TyId<'db>> = Vec::with_capacity(trait_params.len());
    final_args.push(t.self_param(db));
    final_args.extend(non_self_completed);

    for (expected_ty, actual_ty) in trait_params.iter().zip(final_args.iter_mut()).skip(1) {
        if !expected_ty.kind(db).does_match(actual_ty.kind(db)) {
            return Err(TraitArgError::ArgKindMisMatch {
                expected: expected_ty.kind(db).clone(),
                given: *actual_ty,
            });
        }

        let expected_const_ty = match expected_ty.data(db) {
            TyData::ConstTy(expected_ty) => expected_ty.ty(db).into(),
            _ => None,
        };

        match actual_ty.evaluate_const_ty(db, expected_const_ty) {
            Ok(evaluated_ty) => *actual_ty = evaluated_ty,
            Err(InvalidCause::ConstTyMismatch { expected, given }) => {
                return Err(TraitArgError::ArgTypeMismatch {
                    expected: Some(expected),
                    given: Some(given),
                });
            }
            Err(InvalidCause::ConstTyExpected { expected }) => {
                return Err(TraitArgError::ArgTypeMismatch {
                    expected: Some(expected),
                    given: None,
                });
            }
            Err(InvalidCause::NormalTypeExpected { given }) => {
                return Err(TraitArgError::ArgTypeMismatch {
                    expected: None,
                    given: Some(given),
                });
            }
            _ => return Err(TraitArgError::Ignored),
        }
    }

    let assoc_bindings: IndexMap<IdentId<'db>, TyId<'db>> = args
        .iter()
        .filter_map(|arg| match arg {
            GenericArg::AssocType(AssocTypeGenericArg { name, ty }) => {
                let (Some(name), Some(ty)) = (name.to_opt(), ty.to_opt()) else {
                    return None;
                };
                Some((name, lower_hir_ty(db, ty, scope, assumptions)))
            }
            _ => None,
        })
        .collect();

    Ok(TraitInstId::new(db, t, final_args, assoc_bindings))
}

#[salsa::tracked(return_ref)]
pub(crate) fn collect_implementor_methods<'db>(
    db: &'db dyn HirAnalysisDb,
    implementor: ImplementorId<'db>,
) -> IndexMap<IdentId<'db>, Func<'db>> {
    let mut methods = IndexMap::default();
    let impl_trait = match implementor.origin(db) {
        super::trait_def::ImplementorOrigin::Hir(impl_trait) => impl_trait,
        super::trait_def::ImplementorOrigin::VirtualContract(_)
        | super::trait_def::ImplementorOrigin::Assumption => return methods,
    };
    let scope = impl_trait.scope();
    let graph = scope.scope_graph(db);
    for method in graph.child_items(scope).filter_map(|item| match item {
        ItemKind::Func(func) => Some(func),
        _ => None,
    }) {
        let name = method.name(db).to_opt().expect("impl methods have names");
        methods.insert(name, method);
    }

    methods
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Update)]
pub(crate) enum TraitRefLowerError<'db> {
    PathResError(PathResError<'db>),
    InvalidDomain(PathRes<'db>),
    /// Error is expected to be reported elsewhere.
    Ignored,
}

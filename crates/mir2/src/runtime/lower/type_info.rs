use hir::analysis::ty::{
    normalize::normalize_ty,
    trait_resolution::PredicateListId,
    ty_def::{TyData, TyId},
};

use crate::db::MirDb;

#[derive(Clone, Copy)]
pub(crate) struct RuntimeTypeEnv<'db> {
    pub(crate) scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    pub(crate) assumptions: PredicateListId<'db>,
}

impl<'db> RuntimeTypeEnv<'db> {
    pub(crate) fn new(
        scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        Self { scope, assumptions }
    }
}

pub(crate) fn runtime_repr_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    let mut ty = scope.map_or(ty, |scope| normalize_ty(db, ty, scope, assumptions));
    while let Some(inner) = ty.as_view(db) {
        ty = scope.map_or(inner, |scope| normalize_ty(db, inner, scope, assumptions));
    }
    ty
}

pub(crate) fn is_zero_sized_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    runtime_zero_sized_ty(db, ty, scope, assumptions)
}

#[salsa::tracked(
    cycle_fn=runtime_zero_sized_ty_cycle_recover,
    cycle_initial=runtime_zero_sized_ty_cycle_initial
)]
fn runtime_zero_sized_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if repr_ty != ty {
        return runtime_zero_sized_ty(db, repr_ty, scope, assumptions);
    }
    if repr_ty.is_never(db)
        || matches!(
            repr_ty.base_ty(db).data(db),
            TyData::TyBase(hir::analysis::ty::ty_def::TyBase::Func(_))
        )
    {
        return true;
    }
    if repr_ty.is_array(db) {
        let (_, args) = repr_ty.decompose_ty_app(db);
        return repr_ty.array_len(db).is_some_and(|len| {
            len == 0
                || args
                    .first()
                    .copied()
                    .is_some_and(|elem| runtime_zero_sized_ty(db, elem, scope, assumptions))
        });
    }
    if repr_ty.is_tuple(db) || repr_ty.is_struct(db) {
        return repr_ty
            .field_types(db)
            .into_iter()
            .all(|field| runtime_zero_sized_ty(db, field, scope, assumptions));
    }
    false
}

fn runtime_zero_sized_ty_cycle_initial<'db>(
    _db: &'db dyn MirDb,
    _ty: TyId<'db>,
    _scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    _assumptions: PredicateListId<'db>,
) -> bool {
    false
}

fn runtime_zero_sized_ty_cycle_recover<'db>(
    _db: &'db dyn MirDb,
    _value: &bool,
    _count: u32,
    _ty: TyId<'db>,
    _scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    _assumptions: PredicateListId<'db>,
) -> salsa::CycleRecoveryAction<bool> {
    salsa::CycleRecoveryAction::Iterate
}

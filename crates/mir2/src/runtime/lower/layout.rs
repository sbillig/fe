use hir::analysis::ty::{
    const_ty::{ConstTyData, EvaluatedConstTy},
    normalize::normalize_ty,
    trait_resolution::PredicateListId,
    ty_def::{TyData, TyId},
};
use num_traits::ToPrimitive;

use crate::{
    db::MirDb,
    runtime::{ArrayLayout, EnumLayoutKey, EnumVariantLayout, LayoutId, LayoutKey, StructLayout},
};

use super::class::stored_class_for_ty_in_context;

pub(crate) fn runtime_repr_ty<'db>(db: &'db dyn MirDb, mut ty: TyId<'db>) -> TyId<'db> {
    while let Some(inner) = ty.as_view(db) {
        ty = inner;
    }
    ty
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
    fn inner<'db>(
        db: &'db dyn MirDb,
        ty: TyId<'db>,
        scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
        assumptions: PredicateListId<'db>,
        visiting: &mut rustc_hash::FxHashSet<TyId<'db>>,
    ) -> bool {
        let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
        if !visiting.insert(ty) {
            return false;
        }
        let result = if ty.is_never(db)
            || matches!(
                ty.base_ty(db).data(db),
                TyData::TyBase(hir::analysis::ty::ty_def::TyBase::Func(_))
            ) {
            true
        } else if ty.is_array(db) {
            let (_, args) = ty.decompose_ty_app(db);
            let Some(elem) = args.first().copied() else {
                return false;
            };
            array_len(db, ty)
                .is_some_and(|len| len == 0 || inner(db, elem, scope, assumptions, visiting))
        } else if ty.is_tuple(db) || ty.is_struct(db) {
            ty.field_types(db)
                .into_iter()
                .all(|field| inner(db, field, scope, assumptions, visiting))
        } else {
            false
        };
        visiting.remove(&ty);
        result
    }

    inner(
        db,
        ty,
        scope,
        assumptions,
        &mut rustc_hash::FxHashSet::default(),
    )
}

pub(crate) fn layout_for_ty<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> LayoutId<'db> {
    layout_for_ty_in_context(db, ty, ty.as_scope(db), PredicateListId::empty_list(db))
}

pub(crate) fn layout_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> LayoutId<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if ty.as_enum(db).is_some() {
        return LayoutId::new(
            db,
            LayoutKey::Enum(enum_layout_key(db, ty, scope, assumptions)),
        );
    }
    if ty.is_array(db) {
        let (_, args) = ty.decompose_ty_app(db);
        let elem = args.first().copied().expect("array element type");
        return LayoutId::new(
            db,
            LayoutKey::Array(ArrayLayout {
                source_ty: ty,
                elem: stored_class_for_ty_in_context(db, elem, scope, assumptions),
                len: array_len(db, ty).expect("array length"),
            }),
        );
    }
    LayoutId::new(
        db,
        LayoutKey::Struct(StructLayout {
            source_ty: ty,
            fields: ty
                .field_types(db)
                .into_iter()
                .map(|field| stored_class_for_ty_in_context(db, field, scope, assumptions))
                .collect(),
        }),
    )
}

fn enum_layout_key<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> EnumLayoutKey<'db> {
    let enum_ = ty.as_enum(db).expect("enum layout requested for non-enum");
    let args = ty.generic_args(db);
    EnumLayoutKey {
        source_ty: ty,
        variants: enum_
            .variants(db)
            .map(|variant| EnumVariantLayout {
                name: variant
                    .name(db)
                    .map(|name| name.data(db).to_string())
                    .unwrap_or_else(|| format!("variant_{}", variant.idx)),
                fields: variant
                    .field_tys(db)
                    .into_iter()
                    .map(|field| {
                        stored_class_for_ty_in_context(
                            db,
                            field.instantiate(db, args),
                            scope,
                            assumptions,
                        )
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn array_len<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> Option<u64> {
    let (_, args) = ty.decompose_ty_app(db);
    let len_ty = *args.get(1)?;
    let TyData::ConstTy(const_ty) = len_ty.data(db) else {
        return None;
    };
    match const_ty.data(db) {
        ConstTyData::Evaluated(EvaluatedConstTy::LitInt(int_id), _) => int_id.data(db).to_u64(),
        _ => None,
    }
}

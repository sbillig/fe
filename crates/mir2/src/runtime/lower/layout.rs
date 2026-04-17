use hir::{
    analysis::{
        semantic::FieldIndex,
        ty::{
            normalize::normalize_ty,
            trait_resolution::PredicateListId,
            ty_def::{TyData, TyId},
        },
    },
    projection::IndexSource,
};

use crate::{
    db::MirDb,
    runtime::{
        ArrayLayout, EnumLayoutKey, EnumVariantLayout, Layout, LayoutId, LayoutKey, PlaceElem,
        RuntimeClass, StructLayout,
    },
};

use super::class::stored_class_for_ty_in_context;

#[derive(Clone, Copy)]
pub(crate) struct RuntimeTypeEnv<'db> {
    pub(crate) scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    pub(crate) assumptions: PredicateListId<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AggregateCtorElem<'db> {
    pub(crate) elem: PlaceElem<'db>,
    pub(crate) class: RuntimeClass<'db>,
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

pub(crate) fn layout_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
) -> LayoutId<'db> {
    layout_for_ty_in_context(db, ty, env.scope, env.assumptions)
}

pub(crate) fn layout_for_aggregate_instance_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
    field_classes: &[RuntimeClass<'db>],
) -> LayoutId<'db> {
    layout_for_aggregate_instance_in_context(db, ty, field_classes, env.scope, env.assumptions)
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
                len: ty.array_len(db).expect("array length") as u64,
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

pub(crate) fn layout_for_aggregate_instance_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    field_classes: &[RuntimeClass<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> LayoutId<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if ty.as_enum(db).is_some() {
        panic!("aggregate instance layout requested for enum type");
    }
    if ty.is_array(db) {
        let (_, args) = ty.decompose_ty_app(db);
        let elem_ty = args.first().copied().expect("array element type");
        let len = ty.array_len(db).expect("array length");
        assert_eq!(
            len,
            field_classes.len(),
            "aggregate instance arity mismatch for array type {}",
            ty.pretty_print(db),
        );
        let elem = field_classes
            .first()
            .cloned()
            .unwrap_or_else(|| stored_class_for_ty_in_context(db, elem_ty, scope, assumptions));
        assert!(
            field_classes.iter().all(|class| class == &elem),
            "array aggregate instance requires homogeneous runtime element classes: ty={} field_classes={field_classes:?}",
            ty.pretty_print(db),
        );
        return LayoutId::new(
            db,
            LayoutKey::Array(ArrayLayout {
                source_ty: ty,
                elem,
                len: len as u64,
            }),
        );
    }
    assert_eq!(
        ty.field_types(db).len(),
        field_classes.len(),
        "aggregate instance arity mismatch for struct/tuple type {}",
        ty.pretty_print(db),
    );
    LayoutId::new(
        db,
        LayoutKey::Struct(StructLayout {
            source_ty: ty,
            fields: field_classes.to_vec().into_boxed_slice(),
        }),
    )
}

pub(crate) fn aggregate_ctor_elems_for_layout<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    arity: usize,
) -> Box<[AggregateCtorElem<'db>]> {
    match layout.data(db) {
        Layout::Struct(layout) => {
            assert_eq!(
                layout.fields.len(),
                arity,
                "aggregate constructor arity mismatch for struct layout {layout:?}"
            );
            layout
                .fields
                .iter()
                .enumerate()
                .map(|(idx, class)| AggregateCtorElem {
                    elem: PlaceElem::Field(FieldIndex(idx as u16)),
                    class: class.clone(),
                })
                .collect()
        }
        Layout::Array(layout) => {
            assert_eq!(
                layout.len as usize, arity,
                "aggregate constructor arity mismatch for array layout {layout:?}"
            );
            (0..arity)
                .map(|idx| AggregateCtorElem {
                    elem: PlaceElem::Index(IndexSource::Constant(idx)),
                    class: layout.elem.clone(),
                })
                .collect()
        }
        Layout::Enum(_) => panic!("AggregateMake must not lower enum layouts"),
    }
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

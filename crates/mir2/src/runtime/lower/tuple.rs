use hir::analysis::ty::ty_def::TyId;

use crate::{
    db::MirDb,
    runtime::{
        AddressSpaceKind, PlaceElem, PlaceRoot, RBlockId, RExpr, RLocalId, RStmt, RuntimeCarrier,
        RuntimeClass, RuntimeLocalRoot, RuntimePlace,
    },
};

pub(crate) trait RuntimeTupleFieldEmitter<'db> {
    fn db(&self) -> &'db dyn MirDb;

    fn tuple_value_class(&self, tuple: RLocalId) -> Option<RuntimeClass<'db>>;

    fn tuple_local_root(&self, tuple: RLocalId) -> RuntimeLocalRoot<'db>;

    fn alloc_tuple_temp(
        &mut self,
        semantic_ty: TyId<'db>,
        carrier: RuntimeCarrier<'db>,
    ) -> RLocalId;

    fn push_tuple_stmt(&mut self, bb: RBlockId, stmt: RStmt<'db>);
}

pub(crate) fn extract_runtime_tuple_fields<'db, E, I, F>(
    emitter: &mut E,
    bb: RBlockId,
    tuple: RLocalId,
    tuple_ty: TyId<'db>,
    field_indices: I,
    mut field_class: F,
) -> Vec<RLocalId>
where
    E: RuntimeTupleFieldEmitter<'db>,
    I: IntoIterator<Item = usize>,
    F: FnMut(&mut E, TyId<'db>) -> RuntimeClass<'db>,
{
    let Some(tuple_class) = emitter.tuple_value_class(tuple) else {
        return Vec::new();
    };
    let tuple_root = tuple_place_root(emitter, bb, tuple, tuple_ty, tuple_class);
    let field_tys = tuple_ty.field_types(emitter.db());
    field_indices
        .into_iter()
        .map(|idx| {
            let field_ty = field_tys
                .get(idx)
                .copied()
                .unwrap_or_else(|| panic!("tuple field index {idx} out of bounds"));
            let class = field_class(emitter, field_ty);
            let dst = emitter.alloc_tuple_temp(field_ty, RuntimeCarrier::Value(class));
            emitter.push_tuple_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Load {
                        place: RuntimePlace {
                            root: tuple_root.clone(),
                            path: vec![PlaceElem::Field(hir::analysis::semantic::FieldIndex(
                                idx as u16,
                            ))]
                            .into_boxed_slice(),
                        },
                    },
                },
            );
            dst
        })
        .collect()
}

fn tuple_place_root<'db>(
    emitter: &mut impl RuntimeTupleFieldEmitter<'db>,
    bb: RBlockId,
    tuple: RLocalId,
    tuple_ty: TyId<'db>,
    tuple_class: RuntimeClass<'db>,
) -> PlaceRoot<'db> {
    match tuple_class {
        RuntimeClass::Ref { .. } => PlaceRoot::Ref(tuple),
        RuntimeClass::AggregateValue { layout } => match emitter.tuple_local_root(tuple) {
            RuntimeLocalRoot::Slot(_) => PlaceRoot::Slot(tuple),
            RuntimeLocalRoot::Ref(_) => PlaceRoot::Ref(tuple),
            RuntimeLocalRoot::Ptr { .. } | RuntimeLocalRoot::None => {
                let handle = emitter.alloc_tuple_temp(
                    tuple_ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
                );
                emitter.push_tuple_stmt(
                    bb,
                    RStmt::Assign {
                        dst: handle,
                        expr: RExpr::MaterializeToObject { src: tuple },
                    },
                );
                PlaceRoot::Ref(handle)
            }
        },
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {
            panic!("tuple extraction requires aggregate carrier")
        }
    }
}

pub(crate) fn memory_fallback_class<'db>() -> RuntimeClass<'db> {
    RuntimeClass::RawAddr {
        space: AddressSpaceKind::Memory,
        target: None,
    }
}

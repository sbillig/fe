use hir::analysis::semantic::{FieldIndex, SEffectArg};
use hir::analysis::ty::ty_check::ConcreteBorrowProvider;

use crate::{
    db::MirDb,
    runtime::{AddressSpaceKind, HandleView, RuntimeClass},
};

pub(super) fn project_field_class<'db>(
    db: &'db dyn MirDb,
    class: RuntimeClass<'db>,
    field: FieldIndex,
) -> RuntimeClass<'db> {
    let layout = match class {
        RuntimeClass::AggregateValue { layout } | RuntimeClass::Handle { layout, .. } => layout,
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => layout,
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => {
            panic!("invalid field projection class")
        }
    };
    match layout.data(db) {
        crate::runtime::Layout::Struct(layout) => layout.fields[field.0 as usize].clone(),
        _ => panic!("invalid field projection layout"),
    }
}

pub(super) fn project_index_class<'db>(
    db: &'db dyn MirDb,
    class: RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    let layout = match class {
        RuntimeClass::AggregateValue { layout } | RuntimeClass::Handle { layout, .. } => layout,
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => layout,
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => {
            panic!("invalid index projection class")
        }
    };
    match layout.data(db) {
        crate::runtime::Layout::Array(layout) => layout.elem,
        _ => panic!("invalid index projection layout"),
    }
}

pub(super) fn project_variant_field_class<'db>(
    db: &'db dyn MirDb,
    class: RuntimeClass<'db>,
    field: FieldIndex,
) -> RuntimeClass<'db> {
    let RuntimeClass::Handle {
        view: HandleView::EnumVariant(variant),
        ..
    } = class
    else {
        panic!("invalid variant-field projection class");
    };
    variant.layout(db).expect("variant layout").variants[variant.index as usize].fields
        [field.0 as usize]
        .clone()
}

pub(super) fn address_space_from_provider(provider: ConcreteBorrowProvider) -> AddressSpaceKind {
    match provider {
        ConcreteBorrowProvider::Memory => AddressSpaceKind::Memory,
        ConcreteBorrowProvider::Storage => AddressSpaceKind::Storage,
        ConcreteBorrowProvider::TransientStorage => AddressSpaceKind::Transient,
        ConcreteBorrowProvider::Calldata => AddressSpaceKind::Calldata,
    }
}

pub(super) fn effect_arg_address_space(arg: &SEffectArg<'_>) -> AddressSpaceKind {
    address_space_from_provider(arg.provider.expect(
        "effect/provider args must carry an explicit resolved address space before rMIR lowering",
    ))
}

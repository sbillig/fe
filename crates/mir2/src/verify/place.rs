use hir::analysis::semantic::FieldIndex;

use crate::{
    db::MirDb,
    runtime::{
        ConstScalar, HandleView, Layout, LayoutId, LocalSlotKind, PlaceElem, PlaceRoot,
        RuntimeBody, RuntimeClass, RuntimeProgramView, ScalarClass, ScalarRepr, ScalarRole,
        VariantId,
    },
    verify::VerifyError,
};

pub(super) fn project_place<'db>(
    db: &'db dyn MirDb,
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    place: &crate::runtime::RuntimePlace,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let mut current = match place.root {
        PlaceRoot::Slot(local) => match &body
            .local(local)
            .ok_or(VerifyError::MissingRuntimeLocal(local))?
            .slot
        {
            LocalSlotKind::None => {
                return Err(VerifyError::InvalidPlace(RuntimeClass::RawAddr {
                    space: crate::runtime::AddressSpaceKind::Memory,
                    target: None,
                }));
            }
            LocalSlotKind::Slot(class) => class.clone(),
        },
        PlaceRoot::Handle(value) => body
            .value_class(value)
            .cloned()
            .ok_or(VerifyError::ErasedRuntimeValue(value))?,
    };

    for elem in place.path.iter() {
        current = match elem {
            PlaceElem::Field(field) => project_field(program, current, *field)?,
            PlaceElem::Index(index) => {
                let _ = body
                    .value_class(*index)
                    .ok_or(VerifyError::ErasedRuntimeValue(*index))?;
                project_index(program, current)?
            }
            PlaceElem::VariantField(field) => project_variant_field(db, current, *field)?,
        };
    }

    Ok(current)
}

pub(super) fn runtime_value_class<'a, 'db>(
    body: &'a RuntimeBody<'db>,
    value: crate::runtime::RValueId,
) -> Result<&'a RuntimeClass<'db>, VerifyError<'db>> {
    body.value_class(value)
        .ok_or(VerifyError::ErasedRuntimeValue(value))
}

pub(super) fn scalar_class_from_const(value: &ConstScalar) -> ScalarClass<'_> {
    match value {
        ConstScalar::Bool(_) => ScalarClass {
            repr: ScalarRepr::Bool,
            role: ScalarRole::Plain,
        },
        ConstScalar::Int { bits, signed, .. } => ScalarClass {
            repr: ScalarRepr::Int {
                bits: *bits,
                signed: *signed,
            },
            role: ScalarRole::Plain,
        },
        ConstScalar::FixedBytes(bytes) => ScalarClass {
            repr: ScalarRepr::FixedBytes {
                len: bytes.len() as u16,
            },
            role: ScalarRole::Plain,
        },
        ConstScalar::Address { bits, .. } => ScalarClass {
            repr: ScalarRepr::Address { bits: *bits },
            role: ScalarRole::Plain,
        },
    }
}

pub(super) fn enum_tag_class<'db>(
    enum_layout: LayoutId<'db>,
    program: &impl RuntimeProgramView<'db>,
) -> ScalarClass<'db> {
    let Layout::Enum(layout) = program.layout(enum_layout) else {
        unreachable!();
    };
    layout.tag
}

pub(super) fn enum_tag_class_from_value<'db>(
    db: &'db dyn MirDb,
    body: &RuntimeBody<'db>,
    value: crate::runtime::RValueId,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let enum_layout = match runtime_value_class(body, value)? {
        RuntimeClass::AggregateValue { layout } | RuntimeClass::Handle { layout, .. } => layout,
        class => return Err(VerifyError::InvalidPlace(class.clone())),
    };
    Ok(RuntimeClass::Scalar(ScalarClass {
        repr: match enum_layout.data(db) {
            Layout::Enum(layout) => layout.tag.repr,
            Layout::Struct(_) | Layout::Array(_) => {
                return Err(VerifyError::InvalidEnumTag(*enum_layout));
            }
        },
        role: ScalarRole::EnumTag {
            enum_layout: *enum_layout,
        },
    }))
}

pub(super) fn verify_enum_handle<'db>(
    body: &RuntimeBody<'db>,
    root: crate::runtime::RValueId,
    variant: VariantId<'db>,
    program: &impl RuntimeProgramView<'db>,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let class = runtime_value_class(body, root)?.clone();
    let RuntimeClass::Handle {
        layout,
        kind,
        view: HandleView::Whole,
    } = class
    else {
        return Err(VerifyError::InvalidVariantPlace(class));
    };
    if layout != variant.enum_layout || !matches!(program.layout(layout), Layout::Enum(_)) {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    }
    Ok(RuntimeClass::Handle {
        layout,
        kind,
        view: HandleView::Whole,
    })
}

pub(super) fn verify_enum_write_variant<'db>(
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    root: crate::runtime::RValueId,
    variant: VariantId<'db>,
    fields: &[crate::runtime::RValueId],
) -> Result<(), VerifyError<'db>> {
    let RuntimeClass::Handle { layout, .. } = verify_enum_handle(body, root, variant, program)?
    else {
        unreachable!();
    };
    let Layout::Enum(enum_layout) = program.layout(layout) else {
        return Err(VerifyError::InvalidEnumTag(layout));
    };
    let Some(variant_layout) = enum_layout.variants.get(variant.index as usize) else {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    };
    if variant_layout.fields.len() != fields.len() {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    }
    for (field, expected) in fields.iter().zip(variant_layout.fields.iter()) {
        if runtime_value_class(body, *field)? != expected {
            return Err(VerifyError::InvalidVariant(layout, variant.index));
        }
    }
    Ok(())
}

pub(super) fn verify_value_enum_variant<'db>(
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    value_class: RuntimeClass<'db>,
    variant: VariantId<'db>,
    fields: &[crate::runtime::RValueId],
) -> Result<(), VerifyError<'db>> {
    let RuntimeClass::AggregateValue { layout } = value_class else {
        return Err(VerifyError::InvalidVariantPlace(value_class));
    };
    if layout != variant.enum_layout {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    }
    let Layout::Enum(enum_layout) = program.layout(layout) else {
        return Err(VerifyError::InvalidEnumTag(layout));
    };
    let Some(variant_layout) = enum_layout.variants.get(variant.index as usize) else {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    };
    if variant_layout.fields.len() != fields.len() {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    }
    for (field, expected) in fields.iter().zip(variant_layout.fields.iter()) {
        if runtime_value_class(body, *field)? != expected {
            return Err(VerifyError::InvalidVariant(layout, variant.index));
        }
    }
    Ok(())
}

pub(super) fn enum_extract_class<'db>(
    db: &'db dyn MirDb,
    body: &RuntimeBody<'db>,
    value: crate::runtime::RValueId,
    variant: VariantId<'db>,
    field: FieldIndex,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let RuntimeClass::AggregateValue { layout } = runtime_value_class(body, value)?.clone() else {
        return Err(VerifyError::InvalidVariantPlace(
            runtime_value_class(body, value)?.clone(),
        ));
    };
    if layout != variant.enum_layout {
        return Err(VerifyError::InvalidVariant(layout, variant.index));
    }
    let enum_layout = variant
        .layout(db)
        .ok_or(VerifyError::InvalidVariant(layout, variant.index))?;
    enum_layout
        .variants
        .get(variant.index as usize)
        .and_then(|variant| variant.fields.get(field.0 as usize))
        .cloned()
        .ok_or(VerifyError::InvalidVariant(layout, variant.index))
}

fn project_field<'db>(
    program: &impl RuntimeProgramView<'db>,
    current: RuntimeClass<'db>,
    field: FieldIndex,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let layout_id =
        layout_for_projection(current.clone()).ok_or(VerifyError::InvalidPlace(current))?;
    match program.layout(layout_id) {
        Layout::Struct(layout) => {
            layout
                .fields
                .get(field.0 as usize)
                .cloned()
                .ok_or(VerifyError::InvalidPlace(RuntimeClass::AggregateValue {
                    layout: layout_id,
                }))
        }
        Layout::Array(_) | Layout::Enum(_) => {
            Err(VerifyError::InvalidPlace(RuntimeClass::AggregateValue {
                layout: layout_id,
            }))
        }
    }
}

fn project_index<'db>(
    program: &impl RuntimeProgramView<'db>,
    current: RuntimeClass<'db>,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let layout =
        layout_for_projection(current.clone()).ok_or(VerifyError::InvalidPlace(current))?;
    match program.layout(layout) {
        Layout::Array(layout) => Ok(layout.elem),
        Layout::Struct(_) | Layout::Enum(_) => {
            Err(VerifyError::InvalidPlace(RuntimeClass::AggregateValue {
                layout,
            }))
        }
    }
}

fn project_variant_field<'db>(
    db: &'db dyn MirDb,
    current: RuntimeClass<'db>,
    field: FieldIndex,
) -> Result<RuntimeClass<'db>, VerifyError<'db>> {
    let RuntimeClass::Handle {
        layout,
        view: HandleView::EnumVariant(variant),
        ..
    } = current
    else {
        return Err(VerifyError::InvalidVariantPlace(current));
    };
    let enum_layout = variant
        .layout(db)
        .ok_or(VerifyError::InvalidVariant(layout, variant.index))?;
    let variant_layout = enum_layout
        .variants
        .get(variant.index as usize)
        .ok_or(VerifyError::InvalidVariant(layout, variant.index))?;
    variant_layout
        .fields
        .get(field.0 as usize)
        .cloned()
        .ok_or(VerifyError::InvalidVariant(layout, variant.index))
}

fn layout_for_projection<'db>(class: RuntimeClass<'db>) -> Option<LayoutId<'db>> {
    match class {
        RuntimeClass::AggregateValue { layout }
        | RuntimeClass::Handle { layout, .. }
        | RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => Some(layout),
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => None,
    }
}

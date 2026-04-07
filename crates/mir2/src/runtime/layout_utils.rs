use common::layout::TargetDataLayout;
use hir::analysis::semantic::FieldIndex;

use crate::{
    db::MirDb,
    runtime::{
        ConstNode, ConstRegionId, ConstScalar, Layout, LayoutId, LowerError, RuntimeClass,
        ScalarClass, ScalarRepr, VariantId,
    },
};

pub fn layout_size_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    target: TargetDataLayout,
) -> usize {
    match layout.data(db) {
        Layout::Struct(data) => data
            .fields
            .iter()
            .map(|field| memory_size_bytes_for_class(db, field, target))
            .sum(),
        Layout::Array(data) => round_up_to_word(
            data.len as usize * array_elem_size_bytes(db, layout, target),
            target,
        ),
        Layout::Enum(data) => {
            let payload = data
                .variants
                .iter()
                .map(|variant| {
                    variant
                        .fields
                        .iter()
                        .map(|field| memory_size_bytes_for_class(db, field, target))
                        .sum()
                })
                .max()
                .unwrap_or(0);
            target.discriminant_size_bytes + payload
        }
    }
}

pub fn struct_field_offset_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    field: FieldIndex,
    target: TargetDataLayout,
) -> usize {
    let Layout::Struct(data) = layout.data(db) else {
        panic!("struct_field_offset_bytes called for non-struct layout {layout:?}");
    };
    data.fields
        .iter()
        .take(field.0 as usize)
        .map(|field| memory_size_bytes_for_class(db, field, target))
        .sum()
}

pub fn array_elem_size_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    target: TargetDataLayout,
) -> usize {
    let Layout::Array(data) = layout.data(db) else {
        panic!("array_elem_size_bytes called for non-array layout {layout:?}");
    };
    if packed_array_scalar_stride(&data.elem).is_some() {
        1
    } else {
        memory_size_bytes_for_class(db, &data.elem, target)
    }
}

pub fn enum_tag_size_bytes<'db>(
    _db: &'db dyn MirDb,
    _layout: LayoutId<'db>,
    target: TargetDataLayout,
) -> usize {
    target.discriminant_size_bytes
}

pub fn enum_variant_field_offset_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    variant: VariantId<'db>,
    field: FieldIndex,
    target: TargetDataLayout,
) -> usize {
    let Layout::Enum(data) = layout.data(db) else {
        panic!("enum_variant_field_offset_bytes called for non-enum layout {layout:?}");
    };
    let variant = data
        .variants
        .get(variant.index as usize)
        .unwrap_or_else(|| panic!("invalid enum variant {} for {layout:?}", variant.index));
    variant
        .fields
        .iter()
        .take(field.0 as usize)
        .map(|field| memory_size_bytes_for_class(db, field, target))
        .sum()
}

pub fn serialize_const_region_bytes<'db>(
    db: &'db dyn MirDb,
    region: ConstRegionId<'db>,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    let region = region.data(db);
    serialize_const_node_to_layout_bytes(db, region.layout, &region.value, target)
}

fn serialize_const_node_to_layout_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    node: &ConstNode<'db>,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    match layout.data(db) {
        Layout::Struct(data) => serialize_struct_bytes(db, layout, &data.fields, node, target),
        Layout::Array(data) => serialize_array_bytes(db, layout, &data, node, target),
        Layout::Enum(data) => serialize_enum_bytes(db, layout, &data, node, target),
    }
}

fn serialize_struct_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    fields: &[RuntimeClass<'db>],
    node: &ConstNode<'db>,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    let ConstNode::Aggregate {
        layout: node_layout,
        fields: nodes,
    } = node
    else {
        return Err(LowerError::Unsupported(format!(
            "struct const region `{layout:?}` is not an aggregate node"
        )));
    };
    if *node_layout != layout || nodes.len() != fields.len() {
        return Err(LowerError::Unsupported(format!(
            "struct const region `{layout:?}` does not match expected field shape"
        )));
    }
    let mut out = Vec::with_capacity(layout_size_bytes(db, layout, target));
    for (field_class, field_node) in fields.iter().zip(nodes.iter()) {
        out.extend(serialize_const_node_for_class(
            db,
            field_class,
            field_node,
            target,
        )?);
    }
    Ok(out)
}

fn serialize_array_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    data: &crate::runtime::ArrayLayout<'db>,
    node: &ConstNode<'db>,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    let ConstNode::Aggregate {
        layout: node_layout,
        fields,
    } = node
    else {
        return Err(LowerError::Unsupported(format!(
            "array const region `{layout:?}` is not an aggregate node"
        )));
    };
    if *node_layout != layout || fields.len() != data.len as usize {
        return Err(LowerError::Unsupported(format!(
            "array const region `{layout:?}` does not match expected element shape"
        )));
    }
    let stride = array_elem_size_bytes(db, layout, target);
    let total = layout_size_bytes(db, layout, target);
    let mut out = Vec::with_capacity(total);
    for elem in fields {
        out.extend(serialize_const_node_with_size(
            db, &data.elem, elem, stride, target,
        )?);
    }
    if out.len() > total {
        return Err(LowerError::Unsupported(format!(
            "array const region `{layout:?}` overflows serialized size"
        )));
    }
    out.resize(total, 0);
    Ok(out)
}

fn serialize_enum_bytes<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    data: &crate::runtime::EnumLayout<'db>,
    node: &ConstNode<'db>,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    let ConstNode::Aggregate {
        layout: node_layout,
        fields,
    } = node
    else {
        return Err(LowerError::Unsupported(format!(
            "enum const region `{layout:?}` is not an aggregate node"
        )));
    };
    if *node_layout != layout || fields.is_empty() {
        return Err(LowerError::Unsupported(format!(
            "enum const region `{layout:?}` does not contain a tag node"
        )));
    }
    let tag_size = enum_tag_size_bytes(db, layout, target);
    let tag = serialize_const_scalar_with_size(enum_tag_scalar(fields[0].clone())?, tag_size)?;
    let variant_index = enum_variant_index(&fields[0])?;
    let Some(variant) = data.variants.get(variant_index) else {
        return Err(LowerError::Unsupported(format!(
            "enum const region `{layout:?}` has invalid variant index {variant_index}"
        )));
    };
    if fields.len() != 1 + variant.fields.len() {
        return Err(LowerError::Unsupported(format!(
            "enum const region `{layout:?}` payload shape does not match variant {variant_index}"
        )));
    }
    let payload_capacity = layout_size_bytes(db, layout, target) - tag_size;
    let mut payload = Vec::with_capacity(payload_capacity);
    for (field_class, field_node) in variant.fields.iter().zip(fields.iter().skip(1)) {
        payload.extend(serialize_const_node_for_class(
            db,
            field_class,
            field_node,
            target,
        )?);
    }
    if payload.len() > payload_capacity {
        return Err(LowerError::Unsupported(format!(
            "enum const region `{layout:?}` overflows serialized payload size"
        )));
    }
    payload.resize(payload_capacity, 0);
    let mut out = tag;
    out.extend(payload);
    Ok(out)
}

fn serialize_const_node_for_class<'db>(
    db: &'db dyn MirDb,
    class: &RuntimeClass<'db>,
    node: &ConstNode<'db>,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    let size = memory_size_bytes_for_class(db, class, target);
    serialize_const_node_with_size(db, class, node, size, target)
}

fn serialize_const_node_with_size<'db>(
    db: &'db dyn MirDb,
    class: &RuntimeClass<'db>,
    node: &ConstNode<'db>,
    size: usize,
    target: TargetDataLayout,
) -> Result<Vec<u8>, LowerError> {
    match class {
        RuntimeClass::Scalar(class) => {
            let ConstNode::Scalar(scalar) = node else {
                return Err(LowerError::Unsupported(
                    "scalar const node expected scalar payload".to_string(),
                ));
            };
            serialize_const_scalar_with_size(repr_coerced_scalar(class.repr, scalar.clone())?, size)
        }
        RuntimeClass::Ref { .. } => Err(LowerError::Unsupported(
            "reference const nodes are not serializable runtime data".to_string(),
        )),
        RuntimeClass::AggregateValue { layout }
        | RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => serialize_const_node_to_layout_bytes(db, *layout, node, target),
        RuntimeClass::RawAddr { target: None, .. } => {
            let ConstNode::Scalar(scalar) = node else {
                return Err(LowerError::Unsupported(
                    "raw address const node expected scalar payload".to_string(),
                ));
            };
            serialize_const_scalar_with_size(scalar.clone(), size)
        }
    }
}

fn memory_size_bytes_for_class<'db>(
    db: &'db dyn MirDb,
    class: &RuntimeClass<'db>,
    target: TargetDataLayout,
) -> usize {
    match class {
        RuntimeClass::Scalar(class) => {
            round_up_to_word(scalar_storage_size_bytes(class.repr), target)
        }
        RuntimeClass::AggregateValue { layout } => layout_size_bytes(db, *layout, target),
        RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. } => target.word_size_bytes,
    }
}

fn scalar_storage_size_bytes(repr: ScalarRepr) -> usize {
    match repr {
        ScalarRepr::Bool => 1,
        ScalarRepr::Int { bits, .. } | ScalarRepr::Address { bits } => bits.div_ceil(8) as usize,
        ScalarRepr::FixedBytes { len } => len as usize,
    }
}

fn packed_array_scalar_stride(class: &RuntimeClass<'_>) -> Option<usize> {
    match class {
        RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Bool,
            ..
        })
        | RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int { bits: 8, .. },
            ..
        }) => Some(1),
        RuntimeClass::Scalar(_)
        | RuntimeClass::AggregateValue { .. }
        | RuntimeClass::Ref { .. }
        | RuntimeClass::RawAddr { .. } => None,
    }
}

fn round_up_to_word(size: usize, target: TargetDataLayout) -> usize {
    if size == 0 {
        0
    } else {
        size.div_ceil(target.word_size_bytes) * target.word_size_bytes
    }
}

fn serialize_const_scalar_with_size(
    scalar: ConstScalar,
    size: usize,
) -> Result<Vec<u8>, LowerError> {
    let raw = match scalar {
        ConstScalar::Bool(flag) => vec![u8::from(flag)],
        ConstScalar::Int { words, .. } => words,
        ConstScalar::FixedBytes(bytes) => bytes,
        ConstScalar::Address { bytes, .. } => bytes,
    };
    if raw.len() > size {
        return Err(LowerError::Unsupported(format!(
            "const scalar of {} bytes does not fit into {size} bytes",
            raw.len()
        )));
    }
    let mut out = vec![0; size];
    let start = size - raw.len();
    out[start..].copy_from_slice(&raw);
    Ok(out)
}

fn repr_coerced_scalar(repr: ScalarRepr, scalar: ConstScalar) -> Result<ConstScalar, LowerError> {
    match (repr, scalar) {
        (ScalarRepr::Bool, ConstScalar::Bool(flag)) => Ok(ConstScalar::Bool(flag)),
        (
            ScalarRepr::Int { bits, signed },
            ConstScalar::Int {
                bits: scalar_bits,
                signed: scalar_signed,
                words,
            },
        ) if bits == scalar_bits && signed == scalar_signed => Ok(ConstScalar::Int {
            bits,
            signed,
            words,
        }),
        (ScalarRepr::FixedBytes { len }, ConstScalar::FixedBytes(bytes))
            if len as usize == bytes.len() =>
        {
            Ok(ConstScalar::FixedBytes(bytes))
        }
        (
            ScalarRepr::Address { bits },
            ConstScalar::Address {
                bits: scalar_bits,
                bytes,
            },
        ) if bits == scalar_bits => Ok(ConstScalar::Address { bits, bytes }),
        (expected, actual) => Err(LowerError::Unsupported(format!(
            "const scalar `{actual:?}` does not match expected repr `{expected:?}`"
        ))),
    }
}

fn enum_variant_index(node: &ConstNode<'_>) -> Result<usize, LowerError> {
    let ConstNode::Scalar(ConstScalar::Int { words, .. }) = node else {
        return Err(LowerError::Unsupported(
            "enum tag const node is not an integer scalar".to_string(),
        ));
    };
    Ok(words
        .iter()
        .fold(0usize, |acc, byte| (acc << 8) | (*byte as usize)))
}

fn enum_tag_scalar(node: ConstNode<'_>) -> Result<ConstScalar, LowerError> {
    let ConstNode::Scalar(scalar) = node else {
        return Err(LowerError::Unsupported(
            "enum tag const node is not a scalar".to_string(),
        ));
    };
    Ok(scalar)
}

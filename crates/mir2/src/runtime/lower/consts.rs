use hir::analysis::semantic::{
    SemConstId, SemConstScalar, SemConstValue, VariantIndex, normalize_int_to_shape, sem_const_ty,
};

use crate::{
    db::MirDb,
    runtime::{ConstNode, ConstRegionId, ConstScalar, LayoutId, ScalarRepr},
};

use super::{class::scalar_class_for_ty, layout::layout_for_ty};

pub(super) fn const_scalar_from_value<'db>(
    db: &'db dyn MirDb,
    value: SemConstId<'db>,
) -> Option<ConstScalar> {
    let ty = sem_const_ty(db, value);
    match value.value(db) {
        SemConstValue::Unit
        | SemConstValue::Tuple { .. }
        | SemConstValue::Struct { .. }
        | SemConstValue::Array { .. }
        | SemConstValue::Enum { .. } => None,
        SemConstValue::Scalar { value, .. } => match value {
            SemConstScalar::Bool(value) => Some(ConstScalar::Bool(value)),
            SemConstScalar::Int { value } => {
                let scalar = scalar_class_for_ty(db, ty)?;
                match scalar.repr {
                    ScalarRepr::Bool => None,
                    ScalarRepr::Int { bits, signed } => Some(ConstScalar::Int {
                        bits,
                        signed,
                        words: encode_int_words(&value, bits, signed),
                    }),
                    ScalarRepr::FixedBytes { .. } => None,
                    ScalarRepr::Address { bits } => Some(ConstScalar::Address {
                        bits,
                        bytes: encode_int_words(&value, bits, false),
                    }),
                }
            }
            SemConstScalar::Bytes(bytes) => Some(ConstScalar::FixedBytes(bytes.clone())),
        },
    }
}

fn encode_int_words(value: &num_bigint::BigInt, bits: u16, signed: bool) -> Vec<u8> {
    let normalized = normalize_int_to_shape(value.clone(), bits, signed);
    let unsigned = normalize_int_to_shape(normalized, bits, false);
    let (_, bytes) = unsigned.to_bytes_be();
    bytes
}

pub(super) fn lower_const_region<'db>(
    db: &'db dyn MirDb,
    value: SemConstId<'db>,
) -> Option<ConstRegionId<'db>> {
    let layout = layout_for_ty(db, sem_const_ty(db, value));
    let value = lower_const_node(db, value)?;
    Some(ConstRegionId::new(db, layout, value))
}

fn lower_const_node<'db>(db: &'db dyn MirDb, value: SemConstId<'db>) -> Option<ConstNode<'db>> {
    if let Some(scalar) = const_scalar_from_value(db, value) {
        return Some(ConstNode::Scalar(scalar));
    }
    let ty = sem_const_ty(db, value);
    match value.value(db) {
        SemConstValue::Unit | SemConstValue::Scalar { .. } => None,
        SemConstValue::Tuple { elems, .. }
        | SemConstValue::Struct { fields: elems, .. }
        | SemConstValue::Array { elems, .. } => Some(ConstNode::Aggregate {
            layout: layout_for_ty(db, ty),
            fields: elems
                .iter()
                .copied()
                .map(|value| lower_const_node(db, value))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        }),
        SemConstValue::Enum {
            variant, fields, ..
        } => {
            let layout = layout_for_ty(db, ty);
            let mut nodes = Vec::with_capacity(fields.len() + 1);
            nodes.push(ConstNode::Scalar(enum_tag_scalar(db, layout, variant)?));
            nodes.extend(
                fields
                    .iter()
                    .copied()
                    .map(|field| lower_const_node(db, field))
                    .collect::<Option<Vec<_>>>()?,
            );
            Some(ConstNode::Aggregate {
                layout,
                fields: nodes.into_boxed_slice(),
            })
        }
    }
}

pub(super) fn enum_tag_scalar<'db>(
    db: &'db dyn MirDb,
    layout: LayoutId<'db>,
    variant: VariantIndex,
) -> Option<ConstScalar> {
    let crate::runtime::Layout::Enum(layout_data) = layout.data(db) else {
        return None;
    };
    let ScalarRepr::Int { bits, signed } = layout_data.tag.repr else {
        return None;
    };
    Some(ConstScalar::Int {
        bits,
        signed,
        words: if variant.0 == 0 {
            Vec::new()
        } else {
            vec![variant.0 as u8]
        },
    })
}

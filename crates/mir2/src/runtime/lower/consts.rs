use hir::analysis::{
    semantic::{
        SemConstId, SemConstScalar, SemConstValue, VariantIndex, normalize_int_to_shape,
        sem_const_ty,
    },
    ty::const_ty::{ConstTyData, EvaluatedConstTy, evaluate_type_level_int_const_expr},
    ty::ty_def::TyData,
};

use crate::{
    db::MirDb,
    runtime::{ConstNode, ConstRegionId, ConstScalar, LayoutId, ScalarClass, ScalarRepr},
};

use super::{
    layout::layout_for_ty_in_env,
    type_info::{RuntimeTypeEnv, scalar_class_for_ty_in_env},
};

pub(super) fn const_scalar_from_value<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
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
                let scalar = scalar_class_for_ty_in_env(db, env, ty)?;
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
            SemConstScalar::Bytes(bytes) => {
                scalar_class_for_ty_in_env(db, env, ty).and_then(|scalar| {
                    matches!(scalar.repr, ScalarRepr::FixedBytes { .. })
                        .then(|| ConstScalar::FixedBytes(bytes.clone()))
                })
            }
        },
        SemConstValue::TypeLevel { ty, const_ty } => {
            let TyData::ConstTy(const_ty) = const_ty.data(db) else {
                return None;
            };
            let evaluated = const_ty.evaluate(db, Some(ty));
            let evaluated = if let ConstTyData::Abstract(expr, expected_ty) = evaluated.data(db) {
                evaluate_type_level_int_const_expr(db, *expr, *expected_ty).unwrap_or(evaluated)
            } else {
                evaluated
            };
            match evaluated.data(db) {
                ConstTyData::Evaluated(EvaluatedConstTy::LitBool(value), _) => {
                    Some(ConstScalar::Bool(*value))
                }
                ConstTyData::Evaluated(EvaluatedConstTy::LitInt(int_id), _) => {
                    let value = num_bigint::BigInt::from(int_id.data(db).clone());
                    let scalar = scalar_class_for_ty_in_env(db, env, ty)?;
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
                ConstTyData::Evaluated(EvaluatedConstTy::Bytes(bytes), _) => {
                    scalar_class_for_ty_in_env(db, env, ty).and_then(|scalar| {
                        matches!(scalar.repr, ScalarRepr::FixedBytes { .. })
                            .then(|| ConstScalar::FixedBytes(bytes.clone()))
                    })
                }
                ConstTyData::Evaluated(
                    EvaluatedConstTy::Unit
                    | EvaluatedConstTy::Tuple(_)
                    | EvaluatedConstTy::Array(_)
                    | EvaluatedConstTy::Record(_)
                    | EvaluatedConstTy::EnumVariant(_)
                    | EvaluatedConstTy::Invalid,
                    _,
                )
                | ConstTyData::TyVar(_, _)
                | ConstTyData::TyParam(_, _)
                | ConstTyData::Hole(_, _)
                | ConstTyData::Abstract(_, _)
                | ConstTyData::UnEvaluated { .. } => None,
            }
        }
    }
}

pub(super) fn const_scalar_for_class(
    value: &SemConstScalar,
    class: &ScalarClass<'_>,
) -> Option<ConstScalar> {
    match value {
        SemConstScalar::Bool(value) => {
            matches!(class.repr, ScalarRepr::Bool).then_some(ConstScalar::Bool(*value))
        }
        SemConstScalar::Int { value } => match class.repr {
            ScalarRepr::Bool => None,
            ScalarRepr::Int { bits, signed } => Some(ConstScalar::Int {
                bits,
                signed,
                words: encode_int_words(value, bits, signed),
            }),
            ScalarRepr::FixedBytes { .. } => None,
            ScalarRepr::Address { bits } => Some(ConstScalar::Address {
                bits,
                bytes: encode_int_words(value, bits, false),
            }),
        },
        SemConstScalar::Bytes(bytes) => matches!(class.repr, ScalarRepr::FixedBytes { .. })
            .then(|| ConstScalar::FixedBytes(bytes.clone())),
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
    env: RuntimeTypeEnv<'db>,
    value: SemConstId<'db>,
) -> Option<ConstRegionId<'db>> {
    let layout = layout_for_ty_in_env(db, env, sem_const_ty(db, value));
    let value = lower_const_node(db, env, value)?;
    Some(ConstRegionId::new(db, layout, value))
}

fn lower_const_node<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    value: SemConstId<'db>,
) -> Option<ConstNode<'db>> {
    if let Some(scalar) = const_scalar_from_value(db, env, value) {
        return Some(ConstNode::Scalar(scalar));
    }
    let ty = sem_const_ty(db, value);
    if let SemConstValue::Scalar {
        value: SemConstScalar::Bytes(bytes),
        ..
    } = value.value(db)
        && ty.is_array(db)
    {
        return Some(ConstNode::Aggregate {
            layout: layout_for_ty_in_env(db, env, ty),
            fields: bytes
                .iter()
                .map(|byte| {
                    ConstNode::Scalar(ConstScalar::Int {
                        bits: 8,
                        signed: false,
                        words: if *byte == 0 { Vec::new() } else { vec![*byte] },
                    })
                })
                .collect(),
        });
    }
    match value.value(db) {
        SemConstValue::Unit | SemConstValue::Scalar { .. } | SemConstValue::TypeLevel { .. } => {
            None
        }
        SemConstValue::Tuple { elems, .. }
        | SemConstValue::Struct { fields: elems, .. }
        | SemConstValue::Array { elems, .. } => Some(ConstNode::Aggregate {
            layout: layout_for_ty_in_env(db, env, ty),
            fields: elems
                .iter()
                .copied()
                .map(|value| lower_const_node(db, env, value))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        }),
        SemConstValue::Enum {
            variant, fields, ..
        } => {
            let layout = layout_for_ty_in_env(db, env, ty);
            let mut nodes = Vec::with_capacity(fields.len() + 1);
            nodes.push(ConstNode::Scalar(enum_tag_scalar(db, layout, variant)?));
            nodes.extend(
                fields
                    .iter()
                    .copied()
                    .map(|field| lower_const_node(db, env, field))
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

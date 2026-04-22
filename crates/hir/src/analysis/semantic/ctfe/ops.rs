use crate::analysis::{
    HirAnalysisDb,
    semantic::{
        CtfeError, SemConstId, SemConstScalar, SemConstValue, SemOrigin, array_const, enum_const,
        int_const, struct_const, tuple_const,
    },
    ty::ty_def::{PrimTy, TyBase, TyData, TyId},
};

use super::machine::CtfePathElem;

pub(super) fn project_const<'db>(
    db: &'db dyn HirAnalysisDb,
    mut value: SemConstId<'db>,
    path: &[CtfePathElem],
    origin: SemOrigin<'db>,
) -> Result<SemConstId<'db>, CtfeError<'db>> {
    for elem in path {
        value = match (value.value(db), elem) {
            (SemConstValue::Tuple { elems, .. }, CtfePathElem::Field(field))
            | (SemConstValue::Struct { fields: elems, .. }, CtfePathElem::Field(field)) => elems
                .get(field.0 as usize)
                .copied()
                .ok_or(CtfeError::OutOfBounds { origin })?,
            (
                SemConstValue::Enum {
                    variant: actual,
                    fields,
                    ..
                },
                CtfePathElem::VariantField { variant, field },
            ) if actual == *variant => fields
                .get(field.0 as usize)
                .copied()
                .ok_or(CtfeError::OutOfBounds { origin })?,
            (
                SemConstValue::Scalar {
                    value: SemConstScalar::Bytes(bytes),
                    ..
                },
                CtfePathElem::Index(index),
            ) => {
                let byte = *bytes.get(*index).ok_or(CtfeError::OutOfBounds { origin })?;
                int_const(
                    db,
                    TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U8))),
                    byte.into(),
                )
            }
            (SemConstValue::Array { elems, .. }, CtfePathElem::Index(index)) => elems
                .get(*index)
                .copied()
                .ok_or(CtfeError::OutOfBounds { origin })?,
            _ => {
                return Err(CtfeError::InvalidOperation {
                    origin,
                    message: "invalid const projection".into(),
                });
            }
        };
    }
    Ok(value)
}

pub(super) fn store_const<'db>(
    db: &'db dyn HirAnalysisDb,
    root: SemConstId<'db>,
    path: &[CtfePathElem],
    new_value: SemConstId<'db>,
    origin: SemOrigin<'db>,
) -> Result<SemConstId<'db>, CtfeError<'db>> {
    let Some((head, tail)) = path.split_first() else {
        return Ok(new_value);
    };
    match root.value(db) {
        SemConstValue::Tuple { ty, elems } => {
            let mut elems = elems.to_vec();
            let CtfePathElem::Field(field) = head else {
                return Err(CtfeError::InvalidOperation {
                    origin,
                    message: "tuple store requires field projection".into(),
                });
            };
            let Some(slot) = elems.get_mut(field.0 as usize) else {
                return Err(CtfeError::OutOfBounds { origin });
            };
            *slot = store_const(db, *slot, tail, new_value, origin)?;
            Ok(tuple_const(db, ty, elems.into_boxed_slice()))
        }
        SemConstValue::Struct { ty, fields } => {
            let mut fields = fields.to_vec();
            let CtfePathElem::Field(field) = head else {
                return Err(CtfeError::InvalidOperation {
                    origin,
                    message: "struct store requires field projection".into(),
                });
            };
            let Some(slot) = fields.get_mut(field.0 as usize) else {
                return Err(CtfeError::OutOfBounds { origin });
            };
            *slot = store_const(db, *slot, tail, new_value, origin)?;
            Ok(struct_const(db, ty, fields.into_boxed_slice()))
        }
        SemConstValue::Array { ty, elems } => {
            let mut elems = elems.to_vec();
            let CtfePathElem::Index(index) = head else {
                return Err(CtfeError::InvalidOperation {
                    origin,
                    message: "array store requires index projection".into(),
                });
            };
            let Some(slot) = elems.get_mut(*index) else {
                return Err(CtfeError::OutOfBounds { origin });
            };
            *slot = store_const(db, *slot, tail, new_value, origin)?;
            Ok(array_const(db, ty, elems.into_boxed_slice()))
        }
        SemConstValue::Enum {
            ty,
            variant,
            fields,
        } => {
            let mut fields = fields.to_vec();
            let CtfePathElem::VariantField {
                variant: expected,
                field,
            } = head
            else {
                return Err(CtfeError::InvalidOperation {
                    origin,
                    message: "enum store requires variant field projection".into(),
                });
            };
            if variant != *expected {
                return Err(CtfeError::VariantMismatch { origin });
            }
            let Some(slot) = fields.get_mut(field.0 as usize) else {
                return Err(CtfeError::OutOfBounds { origin });
            };
            *slot = store_const(db, *slot, tail, new_value, origin)?;
            Ok(enum_const(db, ty, variant, fields.into_boxed_slice()))
        }
        _ => Err(CtfeError::InvalidOperation {
            origin,
            message: "invalid CTFE store target".into(),
        }),
    }
}

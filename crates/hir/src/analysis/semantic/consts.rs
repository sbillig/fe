use num_bigint::{BigInt, Sign};
use salsa::Update;

use crate::analysis::{
    HirAnalysisDb,
    ty::{
        const_ty::{ConstTyData, EvaluatedConstTy},
        ty_def::{TyData, TyId},
    },
};

use super::ir::VariantIndex;

#[salsa::interned]
#[derive(Debug)]
pub struct SemConstId<'db> {
    pub value: SemConstValue<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemConstValue<'db> {
    Unit,
    Scalar {
        ty: TyId<'db>,
        value: SemConstScalar,
    },
    TypeLevel {
        ty: TyId<'db>,
        const_ty: TyId<'db>,
    },
    Tuple {
        ty: TyId<'db>,
        elems: Box<[SemConstId<'db>]>,
    },
    Struct {
        ty: TyId<'db>,
        fields: Box<[SemConstId<'db>]>,
    },
    Array {
        ty: TyId<'db>,
        elems: Box<[SemConstId<'db>]>,
    },
    Enum {
        ty: TyId<'db>,
        variant: VariantIndex,
        fields: Box<[SemConstId<'db>]>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemConstScalar {
    Bool(bool),
    Int { value: BigInt },
    Bytes(Vec<u8>),
}

pub fn sem_const_ty<'db>(db: &'db dyn HirAnalysisDb, value: SemConstId<'db>) -> TyId<'db> {
    match value.value(db) {
        SemConstValue::Unit => TyId::unit(db),
        SemConstValue::Scalar { ty, .. }
        | SemConstValue::TypeLevel { ty, .. }
        | SemConstValue::Tuple { ty, .. }
        | SemConstValue::Struct { ty, .. }
        | SemConstValue::Array { ty, .. }
        | SemConstValue::Enum { ty, .. } => ty,
    }
}

pub fn sem_const_from_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
) -> Option<SemConstId<'db>> {
    let TyData::ConstTy(const_ty) = ty.data(db) else {
        return None;
    };

    match const_ty.data(db) {
        ConstTyData::Evaluated(value, expected_ty) => match value {
            EvaluatedConstTy::LitInt(int) => Some(int_const(
                db,
                *expected_ty,
                BigInt::from(int.data(db).clone()),
            )),
            EvaluatedConstTy::LitBool(value) => Some(bool_const(db, *value)),
            EvaluatedConstTy::Unit => Some(unit_const(db)),
            EvaluatedConstTy::Tuple(elems) => Some(tuple_const(
                db,
                *expected_ty,
                elems
                    .iter()
                    .map(|elem| sem_const_from_ty(db, *elem))
                    .collect::<Option<Vec<_>>>()?
                    .into_boxed_slice(),
            )),
            EvaluatedConstTy::Array(elems) => Some(array_const(
                db,
                *expected_ty,
                elems
                    .iter()
                    .map(|elem| sem_const_from_ty(db, *elem))
                    .collect::<Option<Vec<_>>>()?
                    .into_boxed_slice(),
            )),
            EvaluatedConstTy::Bytes(bytes) => Some(bytes_const(db, *expected_ty, bytes.clone())),
            EvaluatedConstTy::Record(fields) => Some(struct_const(
                db,
                *expected_ty,
                fields
                    .iter()
                    .map(|field| sem_const_from_ty(db, *field))
                    .collect::<Option<Vec<_>>>()?
                    .into_boxed_slice(),
            )),
            EvaluatedConstTy::EnumVariant(variant) => Some(enum_const(
                db,
                *expected_ty,
                VariantIndex(variant.idx),
                Box::new([]),
            )),
            EvaluatedConstTy::Invalid => None,
        },
        ConstTyData::TyVar(_, value_ty)
        | ConstTyData::TyParam(_, value_ty)
        | ConstTyData::Hole(value_ty, _)
        | ConstTyData::Abstract(_, value_ty)
        | ConstTyData::UnEvaluated {
            ty: Some(value_ty), ..
        } => Some(SemConstId::new(
            db,
            SemConstValue::TypeLevel {
                ty: *value_ty,
                const_ty: ty,
            },
        )),
        ConstTyData::UnEvaluated { ty: None, .. } => None,
    }
}

pub fn unit_const<'db>(db: &'db dyn HirAnalysisDb) -> SemConstId<'db> {
    SemConstId::new(db, SemConstValue::Unit)
}

pub fn bool_const<'db>(db: &'db dyn HirAnalysisDb, value: bool) -> SemConstId<'db> {
    SemConstId::new(
        db,
        SemConstValue::Scalar {
            ty: TyId::bool(db),
            value: SemConstScalar::Bool(value),
        },
    )
}

pub fn int_const<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>, value: BigInt) -> SemConstId<'db> {
    SemConstId::new(
        db,
        SemConstValue::Scalar {
            ty,
            value: SemConstScalar::Int {
                value: normalize_int(db, ty, value),
            },
        },
    )
}

pub fn bytes_const<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    bytes: Vec<u8>,
) -> SemConstId<'db> {
    SemConstId::new(
        db,
        SemConstValue::Scalar {
            ty,
            value: SemConstScalar::Bytes(bytes),
        },
    )
}

pub fn tuple_const<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    elems: Box<[SemConstId<'db>]>,
) -> SemConstId<'db> {
    SemConstId::new(db, SemConstValue::Tuple { ty, elems })
}

pub fn struct_const<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    fields: Box<[SemConstId<'db>]>,
) -> SemConstId<'db> {
    SemConstId::new(db, SemConstValue::Struct { ty, fields })
}

pub fn array_const<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    elems: Box<[SemConstId<'db>]>,
) -> SemConstId<'db> {
    SemConstId::new(db, SemConstValue::Array { ty, elems })
}

pub fn enum_const<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    variant: VariantIndex,
    fields: Box<[SemConstId<'db>]>,
) -> SemConstId<'db> {
    SemConstId::new(
        db,
        SemConstValue::Enum {
            ty,
            variant,
            fields,
        },
    )
}

pub fn scalar_bool(value: SemConstScalar) -> Option<bool> {
    let SemConstScalar::Bool(value) = value else {
        return None;
    };
    Some(value)
}

pub fn scalar_int(value: &SemConstScalar) -> Option<&BigInt> {
    let SemConstScalar::Int { value } = value else {
        return None;
    };
    Some(value)
}

pub fn normalize_int<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>, value: BigInt) -> BigInt {
    let Some((bits, signed)) = int_ty_shape(db, ty) else {
        return value;
    };
    normalize_int_to_shape(value, bits, signed)
}

pub fn normalize_int_to_shape(value: BigInt, bits: u16, signed: bool) -> BigInt {
    if bits == 0 {
        return BigInt::default();
    }

    let modulus = BigInt::from(1u8) << usize::from(bits);
    let mut normalized = value % &modulus;
    if normalized.sign() == Sign::Minus {
        normalized += &modulus;
    }
    if signed {
        let sign_bit = BigInt::from(1u8) << usize::from(bits - 1);
        if normalized >= sign_bit {
            normalized -= modulus;
        }
    }
    normalized
}

pub fn int_ty_shape<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Option<(u16, bool)> {
    use crate::analysis::ty::ty_def::{PrimTy, TyBase, TyData};

    let TyData::TyBase(TyBase::Prim(prim)) = ty.base_ty(db).data(db) else {
        return None;
    };
    Some(match prim {
        PrimTy::U8 => (8, false),
        PrimTy::U16 => (16, false),
        PrimTy::U32 => (32, false),
        PrimTy::U64 => (64, false),
        PrimTy::U128 => (128, false),
        PrimTy::U256 | PrimTy::Usize => (256, false),
        PrimTy::I8 => (8, true),
        PrimTy::I16 => (16, true),
        PrimTy::I32 => (32, true),
        PrimTy::I64 => (64, true),
        PrimTy::I128 => (128, true),
        PrimTy::I256 | PrimTy::Isize => (256, true),
        PrimTy::Bool
        | PrimTy::String
        | PrimTy::Array
        | PrimTy::Tuple(_)
        | PrimTy::Ptr
        | PrimTy::View
        | PrimTy::BorrowMut
        | PrimTy::BorrowRef => return None,
    })
}

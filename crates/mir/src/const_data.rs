use hir::analysis::{
    HirAnalysisDb,
    ty::ty_def::{MAX_INLINE_STRING_BYTES, TyId},
};
use num_bigint::BigUint;

use crate::layout::{self, TargetDataLayout};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstData {
    Int(BigUint),
    Bool(bool),
    Bytes(Vec<u8>),
    EnumVariant(u16),
    Array(Vec<ConstData>),
}

impl From<hir::analysis::ty::const_eval::ConstValue> for ConstData {
    fn from(value: hir::analysis::ty::const_eval::ConstValue) -> Self {
        match value {
            hir::analysis::ty::const_eval::ConstValue::Int(value) => Self::Int(value),
            hir::analysis::ty::const_eval::ConstValue::Bool(value) => Self::Bool(value),
            hir::analysis::ty::const_eval::ConstValue::Bytes(value) => Self::Bytes(value),
            hir::analysis::ty::const_eval::ConstValue::EnumVariant(value) => {
                Self::EnumVariant(value)
            }
            hir::analysis::ty::const_eval::ConstValue::ConstArray(items) => {
                Self::Array(items.into_iter().map(ConstData::from).collect())
            }
        }
    }
}

pub fn serialize_const_data_to_bytes<'db>(
    db: &'db dyn HirAnalysisDb,
    layout: &TargetDataLayout,
    ty: TyId<'db>,
    data: &ConstData,
) -> Option<Vec<u8>> {
    if let ConstData::Bytes(raw) = data
        && matches!(
            ty.base_ty(db).data(db),
            hir::analysis::ty::ty_def::TyData::TyBase(hir::analysis::ty::ty_def::TyBase::Prim(
                hir::analysis::ty::ty_def::PrimTy::String
            ))
        )
    {
        return pack_inline_string_word(raw);
    }
    let size = layout::ty_memory_size_in(db, layout, ty)?;
    if size == 0 {
        return Some(Vec::new());
    }
    if let ConstData::Bytes(raw) = data
        && ty.is_array(db)
    {
        return serialize_const_u8_array_bytes(db, layout, ty, raw);
    }
    serialize_const_leaf_to_bytes(db, layout, ty, data)
}

pub fn pack_inline_string_word(raw: &[u8]) -> Option<Vec<u8>> {
    if raw.len() > MAX_INLINE_STRING_BYTES {
        return None;
    }

    let mut bytes = vec![0u8; 32];
    bytes[0] = raw.len() as u8;
    bytes[1..1 + raw.len()].copy_from_slice(raw);
    Some(bytes)
}

pub fn serialize_const_u8_array_bytes<'db>(
    db: &'db dyn HirAnalysisDb,
    layout: &TargetDataLayout,
    array_ty: TyId<'db>,
    raw: &[u8],
) -> Option<Vec<u8>> {
    let len = layout::array_len(db, array_ty)?;
    let elem_ty = layout::array_elem_ty(db, array_ty)?;
    if len != raw.len() {
        return None;
    }
    if !matches!(
        elem_ty.base_ty(db).data(db),
        hir::analysis::ty::ty_def::TyData::TyBase(hir::analysis::ty::ty_def::TyBase::Prim(
            hir::analysis::ty::ty_def::PrimTy::U8
        ))
    ) {
        return None;
    }

    let elem_size = layout::array_elem_stride_memory_in(db, layout, array_ty)?;
    let array_size = layout::ty_memory_size_in(db, layout, array_ty)?;
    let mut bytes = Vec::with_capacity(array_size);
    for &byte in raw {
        bytes.extend(pad_be_bytes(&[byte], elem_size)?);
    }
    if bytes.len() > array_size {
        return None;
    }
    bytes.extend(std::iter::repeat_n(0u8, array_size - bytes.len()));
    Some(bytes)
}

fn serialize_const_leaf_to_bytes<'db>(
    db: &'db dyn HirAnalysisDb,
    layout: &TargetDataLayout,
    ty: TyId<'db>,
    data: &ConstData,
) -> Option<Vec<u8>> {
    if ty.is_array(db) {
        let ConstData::Array(items) = data else {
            return None;
        };
        let len = layout::array_len(db, ty)?;
        if items.len() != len {
            return None;
        }
        let elem_ty = layout::array_elem_ty(db, ty)?;
        let elem_size = layout::array_elem_stride_memory_in(db, layout, ty)?;
        let array_size = layout::ty_memory_size_in(db, layout, ty)?;
        let mut bytes = Vec::new();
        for item in items {
            bytes.extend(if elem_ty.is_array(db) {
                serialize_const_data_to_bytes(db, layout, elem_ty, item)?
            } else {
                serialize_const_value_to_sized_bytes(item, elem_size)?
            });
        }
        if bytes.len() > array_size {
            return None;
        }
        bytes.extend(std::iter::repeat_n(0u8, array_size - bytes.len()));
        return Some(bytes);
    }

    let size = layout::ty_memory_size_in(db, layout, ty)?;
    if size == 0 {
        return Some(Vec::new());
    }
    serialize_const_value_to_sized_bytes(data, size)
}

fn serialize_const_value_to_sized_bytes(data: &ConstData, size: usize) -> Option<Vec<u8>> {
    let bytes = match data {
        ConstData::Int(int) => pad_be_bytes(&int.to_bytes_be(), size)?,
        ConstData::Bool(flag) => {
            let raw = if *flag { [1u8] } else { [0u8] };
            pad_be_bytes(&raw, size)?
        }
        ConstData::Bytes(raw) => pad_be_bytes(raw, size)?,
        ConstData::EnumVariant(idx) => pad_be_bytes(&idx.to_be_bytes(), size)?,
        ConstData::Array(_) => return None,
    };
    Some(bytes)
}

fn pad_be_bytes(raw: &[u8], size: usize) -> Option<Vec<u8>> {
    if raw.len() > size {
        return None;
    }
    let mut bytes = vec![0u8; size];
    let start = size.checked_sub(raw.len())?;
    bytes[start..].copy_from_slice(raw);
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use hir::analysis::ty::ty_def::{PrimTy, TyBase, TyData, TyId};
    use num_bigint::BigUint;

    use super::{ConstData, serialize_const_data_to_bytes};

    fn array_with_len<'db>(
        db: &'db dyn hir::analysis::HirAnalysisDb,
        elem: TyId<'db>,
        len: usize,
    ) -> TyId<'db> {
        use hir::analysis::ty::const_ty::{ConstTyData, ConstTyId, EvaluatedConstTy};
        use hir::hir_def::IntegerId;

        let array_ctor = TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::Array)));
        let array = TyId::app(db, array_ctor, elem);
        let len_kind = array
            .applicable_ty(db)
            .and_then(|prop| prop.const_ty)
            .expect("array type should have a length const parameter");
        let len_const = ConstTyId::new(
            db,
            ConstTyData::Evaluated(
                EvaluatedConstTy::LitInt(IntegerId::new(db, BigUint::from(len))),
                len_kind,
            ),
        );
        TyId::app(db, array, TyId::new(db, TyData::ConstTy(len_const)))
    }

    #[test]
    fn serialize_nested_bool_const_array_with_padding() {
        let db = driver::DriverDataBase::default();
        let bool_ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::Bool)));
        let inner = array_with_len(&db, bool_ty, 3);
        let outer = array_with_len(&db, inner, 2);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            outer,
            &ConstData::Array(vec![
                ConstData::Array(vec![
                    ConstData::Bool(true),
                    ConstData::Bool(false),
                    ConstData::Bool(true),
                ]),
                ConstData::Array(vec![
                    ConstData::Bool(false),
                    ConstData::Bool(true),
                    ConstData::Bool(false),
                ]),
            ]),
        )
        .expect("bool arrays should serialize");

        assert_eq!(data.len(), 64);
        assert_eq!(&data[0..3], &[1, 0, 1]);
        assert!(data[3..32].iter().all(|b| *b == 0));
        assert_eq!(&data[32..35], &[0, 1, 0]);
        assert!(data[35..64].iter().all(|b| *b == 0));
    }

    #[test]
    fn serialize_u8_const_array_words() {
        let db = driver::DriverDataBase::default();
        let elem = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::U8)));
        let array_ty = array_with_len(&db, elem, 2);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            array_ty,
            &ConstData::Bytes(vec![1, 2]),
        )
        .expect("u8 array should serialize");

        assert_eq!(data.len(), 32);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
    }

    #[test]
    fn serialize_mixed_scalar_const_array_words() {
        let db = driver::DriverDataBase::default();
        let elem_ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::U256)));
        let array_ty = array_with_len(&db, elem_ty, 3);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            array_ty,
            &ConstData::Array(vec![
                ConstData::Int(BigUint::from(0x11u64)),
                ConstData::EnumVariant(2),
                ConstData::Bool(true),
            ]),
        )
        .expect("scalar array should serialize");

        assert_eq!(data.len(), 96);
        assert_eq!(data[31], 0x11);
        assert_eq!(data[63], 2);
        assert_eq!(data[95], 1);
    }

    #[test]
    fn serialize_nested_u8_const_arrays_keep_inner_word_padding() {
        let db = driver::DriverDataBase::default();
        let u8_ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(PrimTy::U8)));
        let inner = array_with_len(&db, u8_ty, 2);
        let outer = array_with_len(&db, inner, 2);

        let data = serialize_const_data_to_bytes(
            &db,
            &crate::layout::EVM_LAYOUT,
            outer,
            &ConstData::Array(vec![
                ConstData::Bytes(vec![1, 2]),
                ConstData::Bytes(vec![3, 4]),
            ]),
        )
        .expect("nested u8 arrays should serialize with inner padding");

        assert_eq!(data.len(), 64);
        assert_eq!(&data[0..2], &[1, 2]);
        assert!(data[2..32].iter().all(|byte| *byte == 0));
        assert_eq!(&data[32..34], &[3, 4]);
        assert!(data[34..64].iter().all(|byte| *byte == 0));
    }
}

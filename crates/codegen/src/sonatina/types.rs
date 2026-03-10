//! Type mapping from Fe MIR to Sonatina IR types.

use driver::DriverDataBase;
use hir::analysis::ty::ty_def::{CapabilityKind, PrimTy, TyBase, TyData, TyId};
use mir::{
    LocalId, MirBody, MirInst, Rvalue, ValueData, ValueOrigin, layout::TargetDataLayout, repr,
};
use sonatina_ir::{Immediate, Type, ValueId, builder::FunctionBuilder, func_cursor::FuncCursor};

use super::is_erased_runtime_ty;

/// Returns the Sonatina scalar type for a Fe primitive, when one exists.
pub fn prim_scalar_type(prim: PrimTy) -> Option<Type> {
    match prim {
        PrimTy::Bool => Some(Type::I1),
        PrimTy::U8 | PrimTy::I8 => Some(Type::I8),
        PrimTy::U16 | PrimTy::I16 => Some(Type::I16),
        PrimTy::U32 | PrimTy::I32 => Some(Type::I32),
        PrimTy::U64 | PrimTy::I64 => Some(Type::I64),
        PrimTy::U128 | PrimTy::I128 => Some(Type::I128),
        PrimTy::U256 | PrimTy::I256 | PrimTy::Usize | PrimTy::Isize => Some(Type::I256),
        PrimTy::String
        | PrimTy::Array
        | PrimTy::Tuple(_)
        | PrimTy::Ptr
        | PrimTy::View
        | PrimTy::BorrowMut
        | PrimTy::BorrowRef => None,
    }
}

/// Returns the Sonatina runtime value type for a Fe scalar-or-pointer runtime value.
pub fn value_type(db: &DriverDataBase, ty: TyId<'_>) -> Type {
    let ty = match ty.as_capability(db) {
        Some((CapabilityKind::View, inner)) => inner,
        _ => ty,
    };
    let leaf_ty = repr::word_conversion_leaf_ty(db, ty);
    if let TyData::TyBase(TyBase::Prim(prim)) = leaf_ty.base_ty(db).data(db)
        && let Some(ty) = prim_scalar_type(*prim)
    {
        return ty;
    }

    Type::I256
}

/// Returns the Sonatina runtime type for a Fe value, including erased types.
pub fn runtime_type(db: &DriverDataBase, target_layout: &TargetDataLayout, ty: TyId<'_>) -> Type {
    if is_erased_runtime_ty(db, target_layout, ty) {
        Type::Unit
    } else {
        value_type(db, ty)
    }
}

fn repr_runtime_type(
    db: &DriverDataBase,
    target_layout: &TargetDataLayout,
    ty: TyId<'_>,
    repr_has_address_space: bool,
) -> Type {
    if repr_has_address_space {
        Type::I256
    } else {
        runtime_type(db, target_layout, ty)
    }
}

fn local_has_pointer_repr(body: &MirBody<'_>, local_id: LocalId) -> bool {
    body.values.iter().any(|value| {
        matches!(value.origin, ValueOrigin::Local(local) if local == local_id)
            && value.repr.address_space().is_some()
    }) || body
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .any(|inst| {
            matches!(
                inst,
                MirInst::Assign {
                    dest: Some(dest_local),
                    rvalue: Rvalue::Alloc { .. },
                    ..
                } if *dest_local == local_id
            )
        })
}

/// Returns the Sonatina runtime type for a MIR local after repr lowering.
pub fn local_runtime_type(
    db: &DriverDataBase,
    target_layout: &TargetDataLayout,
    body: &MirBody<'_>,
    local_id: LocalId,
) -> Type {
    let local = &body.locals[local_id.index()];
    repr_runtime_type(
        db,
        target_layout,
        local.ty,
        local_has_pointer_repr(body, local_id),
    )
}

/// Returns the Sonatina runtime type for a MIR value after repr lowering.
pub fn value_runtime_type(
    db: &DriverDataBase,
    target_layout: &TargetDataLayout,
    value: &ValueData<'_>,
) -> Type {
    repr_runtime_type(
        db,
        target_layout,
        value.ty,
        value.repr.address_space().is_some(),
    )
}

/// Creates a zero/undef value of the given Sonatina type.
pub fn zero_value<C: FuncCursor>(fb: &mut FunctionBuilder<C>, ty: Type) -> ValueId {
    if ty == Type::Unit {
        fb.make_undef_value(Type::Unit)
    } else {
        fb.make_imm_value(Immediate::zero(ty))
    }
}

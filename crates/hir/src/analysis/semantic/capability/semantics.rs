use super::array::ArrayLength;
use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{
            adt_def::{AdtRef, instantiate_adt_field_shape},
            normalize::normalize_ty,
            provider::ProviderAddressSpace,
            trait_resolution::PredicateListId,
            ty_def::{BorrowKind, CapabilityKind, TyId},
        },
    },
    hir_def::scope_graph::ScopeId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CapabilityClass {
    Borrow(BorrowKind),
    View,
    /// A copyable raw memory address. Passing it does not reserve its referent.
    Pointer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilitySemantics<'db> {
    pub class: CapabilityClass,
    pub target_ty: TyId<'db>,
    pub representation_ty: TyId<'db>,
}

pub fn capability_semantics<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
) -> Option<CapabilitySemantics<'db>> {
    if let Some((kind, target_ty)) = ty.as_capability(db) {
        let class = match kind {
            CapabilityKind::Mut => CapabilityClass::Borrow(BorrowKind::Mut),
            CapabilityKind::Ref => CapabilityClass::Borrow(BorrowKind::Ref),
            CapabilityKind::View => CapabilityClass::View,
        };
        return Some(CapabilitySemantics {
            class,
            target_ty,
            representation_ty: ty,
        });
    }
    let (target_ty, ProviderAddressSpace::Memory) = ty.as_ptr(db)? else {
        return None;
    };
    Some(CapabilitySemantics {
        class: CapabilityClass::Pointer,
        target_ty,
        representation_ty: ty,
    })
}

/// Whether a value of `ty` holds a capability carrier or raw pointer, without
/// following one to its target.
#[salsa::tracked(cycle_fn=contains_capability_cycle_recover, cycle_initial=contains_capability_cycle_initial)]
pub fn contains_capability<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    ty: TyId<'db>,
) -> bool {
    let ty = normalize_ty(db, ty, scope, assumptions);
    if capability_semantics(db, ty).is_some() {
        return true;
    }
    if ty.is_array(db) {
        ArrayLength::from_ty(db, ty.generic_args(db)[1]) != Some(ArrayLength::Known(0))
            && contains_capability(db, scope, assumptions, ty.generic_args(db)[0])
    } else if ty.is_product(db) {
        ty.field_types(db)
            .into_iter()
            .any(|field| contains_capability(db, scope, assumptions, field))
    } else if let Some(adt) = ty.adt_def(db)
        && matches!(adt.adt_ref(db), AdtRef::Enum(_))
    {
        adt.fields(db).iter().enumerate().any(|(variant, fields)| {
            (0..fields.num_types()).any(|field| {
                let field_ty =
                    instantiate_adt_field_shape(db, adt, variant, field, ty.generic_args(db));
                contains_capability(db, scope, assumptions, field_ty)
            })
        })
    } else {
        false
    }
}

fn contains_capability_cycle_initial<'db>(
    _: &'db dyn HirAnalysisDb,
    _: ScopeId<'db>,
    _: PredicateListId<'db>,
    _: TyId<'db>,
) -> bool {
    false
}

fn contains_capability_cycle_recover<'db>(
    _: &'db dyn HirAnalysisDb,
    _: &bool,
    _: u32,
    _: ScopeId<'db>,
    _: PredicateListId<'db>,
    _: TyId<'db>,
) -> salsa::CycleRecoveryAction<bool> {
    salsa::CycleRecoveryAction::Iterate
}

/// Whether a semantic-IR value of `ty` carries accesses: a carrier, or a
/// projection's tuple or sum shape holding one.
pub fn holds_carrier<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> bool {
    ty.as_capability(db).is_some()
        || ty
            .decompose_ty_app(db)
            .1
            .iter()
            .any(|arg| holds_carrier(db, *arg))
}

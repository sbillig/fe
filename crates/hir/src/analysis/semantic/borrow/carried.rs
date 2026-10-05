//! Which capabilities a value of a type can hold.
use salsa::Update;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::capability::{
            array::ArrayLength,
            semantics::{CapabilityClass, capability_semantics},
        },
        ty::{
            adt_def::{AdtRef, instantiate_adt_field_shape},
            corelib::resolve_core_trait,
            normalize::normalize_ty,
            trait_def::TraitInstId,
            trait_resolution::{
                GoalSatisfiability, PredicateListId, TraitSolveCx, is_goal_satisfiable,
            },
            ty_def::{BorrowKind, TyId},
        },
    },
    hir_def::scope_graph::ScopeId,
};

/// Capabilities reachable inside a value without following a borrow, view,
/// handle or raw pointer to its target. A value carries one implicit lifetime:
/// every borrow it holds is tracked as a unit.
///
/// A type parameter carries nothing: as in Rust, a value of a generic type
/// never borrows through a signature's elided inputs. A specialization that
/// fills it with a borrowing type is checked as its own concrete instance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Update)]
pub struct Carried {
    /// `ref`, `mut`, or view capabilities.
    pub borrows: bool,
    /// `mut` capabilities.
    pub mut_borrows: bool,
    /// Provider handles.
    pub handles: bool,
    /// Raw pointers.
    pub pointers: bool,
    /// Raw pointers outside any `Borrows` view: memory the value may own.
    pub owns: bool,
    /// Some raw pointer in the value points to a type that holds borrows.
    pub pointer_to_borrow: bool,
    /// Values of a `core::marker::Borrows` type: raw-pointer views that hold
    /// a shared borrow of the values they were derived from.
    pub views: bool,
    /// Some `mut` capability's referent can itself hold capabilities, so a
    /// callee given this value may store borrows through it.
    pub mut_referents_hold: bool,
}

impl Carried {
    pub fn any(self) -> bool {
        self.borrows || self.handles || self.views
    }

    /// Whether the value holds a borrow, including a view's.
    pub fn borrowing(self) -> bool {
        self.borrows || self.views
    }

    /// Whether the value holds any capability, raw pointers included.
    pub fn contains_capability(self) -> bool {
        self.borrows || self.handles || self.pointers
    }

    fn join(&mut self, other: Self) {
        self.borrows |= other.borrows;
        self.mut_borrows |= other.mut_borrows;
        self.handles |= other.handles;
        self.pointers |= other.pointers;
        self.owns |= other.owns;
        self.pointer_to_borrow |= other.pointer_to_borrow;
        self.views |= other.views;
        self.mut_referents_hold |= other.mut_referents_hold;
    }
}

#[salsa::tracked(cycle_fn=carried_cycle_recover, cycle_initial=carried_cycle_initial)]
pub fn carried_capabilities<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    ty: TyId<'db>,
) -> Carried {
    let ty = normalize_ty(db, ty, scope, assumptions);
    if let Ok(Some(semantics)) = capability_semantics(db, scope, assumptions, ty) {
        return match semantics.class {
            CapabilityClass::Borrow(kind) => Carried {
                borrows: true,
                mut_borrows: kind == BorrowKind::Mut,
                mut_referents_hold: kind == BorrowKind::Mut
                    && carried_capabilities(db, scope, assumptions, semantics.target_ty).any(),
                ..Carried::default()
            },
            CapabilityClass::View => Carried {
                borrows: true,
                ..Carried::default()
            },
            CapabilityClass::Handle => Carried {
                handles: true,
                ..Carried::default()
            },
            CapabilityClass::Pointer => {
                let pointee = carried_capabilities(db, scope, assumptions, semantics.target_ty);
                Carried {
                    pointers: true,
                    owns: true,
                    pointer_to_borrow: pointee.borrows || pointee.pointer_to_borrow,
                    ..Carried::default()
                }
            }
        };
    }
    let view = implements_borrows(db, scope, assumptions, ty);
    let mut carried = Carried {
        views: view,
        ..Carried::default()
    };
    if ty.is_array(db) {
        if ArrayLength::from_ty(db, ty.generic_args(db)[1]) != Some(ArrayLength::Known(0)) {
            carried.join(carried_capabilities(
                db,
                scope,
                assumptions,
                ty.generic_args(db)[0],
            ));
        }
    } else if ty.is_tuple(db) || ty.is_struct(db) {
        for field in ty.field_types(db) {
            carried.join(carried_capabilities(db, scope, assumptions, field));
        }
    } else if let Some(adt) = ty.adt_def(db)
        && matches!(adt.adt_ref(db), AdtRef::Enum(_))
    {
        for (variant, fields) in adt.fields(db).iter().enumerate() {
            for field in 0..fields.num_types() {
                let field_ty =
                    instantiate_adt_field_shape(db, adt, variant, field, ty.generic_args(db));
                carried.join(carried_capabilities(db, scope, assumptions, field_ty));
            }
        }
    }
    carried.owns &= !view;
    carried
}

/// Whether `ty` implements the `core::marker::Borrows` marker.
fn implements_borrows<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    ty: TyId<'db>,
) -> bool {
    resolve_core_trait(db, scope, &["marker", "Borrows"]).is_some_and(|borrows| {
        matches!(
            is_goal_satisfiable(
                db,
                TraitSolveCx::new(db, scope).with_assumptions(assumptions),
                TraitInstId::new_simple(db, borrows, vec![ty]),
            ),
            GoalSatisfiability::Satisfied(_)
        )
    })
}

fn carried_cycle_initial<'db>(
    _: &'db dyn HirAnalysisDb,
    _: ScopeId<'db>,
    _: PredicateListId<'db>,
    _: TyId<'db>,
) -> Carried {
    Carried::default()
}

fn carried_cycle_recover<'db>(
    _: &'db dyn HirAnalysisDb,
    _: &Carried,
    _: u32,
    _: ScopeId<'db>,
    _: PredicateListId<'db>,
    _: TyId<'db>,
) -> salsa::CycleRecoveryAction<Carried> {
    salsa::CycleRecoveryAction::Iterate
}

//! Type normalization module
//!
//! This module provides functionality to normalize types by resolving associated types
//! to concrete types when possible. This happens before type unification to ensure
//! that types are in their most resolved form.

use crate::core::hir_def::{ImplTrait, scope_graph::ScopeId};
use common::indexmap::{IndexMap, IndexSet};
use rustc_hash::FxHashMap;

use super::{
    canonical::Canonical,
    canonical::Canonicalized,
    closure::callable_out_ty,
    fold::{TyFoldable, TyFolder},
    trait_def::{
        ImplementorId, TraitInstId, TraitRefId, impl_where_clause_refuted, impls_for_trait_inst,
    },
    trait_lower::complete_impl_assoc_ty,
    trait_resolution::{PredicateListId, TraitSolveCx},
    ty_def::{AssocTy, InvalidCause, TyData, TyId, TyParam, collect_variables},
    unify::UnificationTable,
    visitor::{TyVisitor, walk_ty},
};
use crate::analysis::HirAnalysisDb;

/// Normalizes a type by resolving all associated types to concrete types when possible.
///
/// This function takes a type and attempts to resolve any associated types within it
/// using the provided assumptions and scope context. It handles:
/// - Simple associated types (e.g., `T::Output`)
/// - Nested associated types (e.g., `T::Encoder::Output`)
/// - Associated types with generic parameters
pub fn normalize_ty<'db, T>(
    db: &'db dyn HirAnalysisDb,
    ty: T,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> T
where
    T: TyFoldable<'db>,
{
    ty.fold_with(db, &mut TypeNormalizer::new(db, scope, assumptions, true))
}

/// Apply declared associated equalities without implementation selection.
/// Structural slot planning uses declaration coordinates before the slots it
/// is discovering exist, so implementation lookup would create a query cycle.
pub fn normalize_from_assumptions<'db, T>(
    db: &'db dyn HirAnalysisDb,
    value: T,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> T
where
    T: TyFoldable<'db>,
{
    value.fold_with(db, &mut TypeNormalizer::new(db, scope, assumptions, false))
}

/// Apply the associated equalities carried by a single trait predicate, as
/// [`normalize_from_assumptions`] does.
pub fn normalize_with_trait_evidence<'db, T>(
    db: &'db dyn HirAnalysisDb,
    value: T,
    scope: ScopeId<'db>,
    evidence: TraitInstId<'db>,
) -> T
where
    T: TyFoldable<'db>,
{
    normalize_from_assumptions(db, value, scope, PredicateListId::new(db, vec![evidence]))
}

/// How many projections may be resolved inside each other, and how deeply the
/// generic arguments of one instance may nest.
pub(crate) const PROJECTION_DEPTH_LIMIT: usize = 64;

/// How many type nodes, counted as a tree, a projection being resolved may
/// have, as may the generic arguments of one instance.
pub(crate) const PROJECTION_WORK_LIMIT: usize = 65536;

/// Folds a value, normalizing each projection in it with
/// [`normalize_projection`].
struct TypeNormalizer<'db> {
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    /// Whether impls, and not only the assumptions, resolve projections.
    resolve_impls: bool,
    /// How many projections may still be resolved inside each other.
    budget: usize,
}

impl<'db> TypeNormalizer<'db> {
    fn new(
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        assumptions: PredicateListId<'db>,
        resolve_impls: bool,
    ) -> Self {
        Self {
            db,
            scope,
            assumptions,
            resolve_impls,
            budget: PROJECTION_DEPTH_LIMIT,
        }
    }
}

/// A projection to normalize, with what normalizing it depends on.
#[salsa::interned]
#[derive(Debug)]
struct ProjectionQuery<'db> {
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    resolve_impls: bool,
    /// How many projections may be resolved inside each other, this one
    /// included.
    budget: usize,
}

/// The normalized form of a projection.
///
/// A cycle through an impl that defines a projection as itself is left
/// unresolved; other repetitions reach the budget. A projection whose impl
/// defines it through a larger one, such as `type Out = <W<(T, T)> as
/// Tr>::Out`, never repeats, and its types can double at each step, staying
/// small as interned values while growing exponentially as trees, which is how
/// resolution walks them. So both the nesting and the size of each projection
/// are limited. The limits depend only on the query, so a projection
/// normalizes the same way wherever and in whichever order it is met.
#[salsa::tracked(
    cycle_fn = normalize_projection_cycle_recover,
    cycle_initial = normalize_projection_cycle_initial
)]
fn normalize_projection<'db>(db: &'db dyn HirAnalysisDb, query: ProjectionQuery<'db>) -> TyId<'db> {
    let (ty, budget) = (query.ty(db), query.budget(db));
    let TyData::AssocTy(assoc) = ty.data(db) else {
        unreachable!("only projections are normalized one at a time")
    };
    if budget == 0 || tree_size_within(db, ty, PROJECTION_WORK_LIMIT).is_none() {
        return TyId::invalid(db, InvalidCause::TypeNormalizationLimit);
    }
    let mut normalizer = TypeNormalizer {
        db,
        scope: query.scope(db),
        assumptions: query.assumptions(db),
        resolve_impls: query.resolve_impls(db),
        budget: budget - 1,
    };
    match normalizer.try_resolve_assoc_ty(ty, assoc) {
        Some(replacement) if replacement != ty => normalizer.fold_ty(db, replacement),
        // Not resolved; still fold internals (e.g., normalize self type)
        _ => ty.super_fold_with(db, &mut normalizer),
    }
}

/// Resolving a projection can lower an impl's associated type, which can ask
/// for the projection again through trait solving; within that cycle it is
/// unresolved.
fn normalize_projection_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    query: ProjectionQuery<'db>,
) -> TyId<'db> {
    query.ty(db)
}

fn normalize_projection_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &TyId<'db>,
    _count: u32,
    _query: ProjectionQuery<'db>,
) -> salsa::CycleRecoveryAction<TyId<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

/// The types `ty` is made of, one level down.
fn children<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Vec<TyId<'db>> {
    struct Children<'db> {
        db: &'db dyn HirAnalysisDb,
        children: Vec<TyId<'db>>,
    }
    impl<'db> TyVisitor<'db> for Children<'db> {
        fn db(&self) -> &'db dyn HirAnalysisDb {
            self.db
        }
        fn visit_ty(&mut self, ty: TyId<'db>) {
            self.children.push(ty);
        }
        // An application is one type with its constructor: `(A, B)` is made
        // of `A` and `B`, not of `(A,)` applied to `B`.
        fn visit_app(&mut self, abs: TyId<'db>, arg: TyId<'db>) {
            walk_ty(self, abs);
            self.children.push(arg);
        }
    }
    let mut children = Children {
        db,
        children: Vec::new(),
    };
    walk_ty(&mut children, ty);
    children.children
}

/// The number of nodes in `ty` counted as a tree, if it is at most `limit`.
pub(crate) fn tree_size_within<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    limit: usize,
) -> Option<usize> {
    fn size<'db>(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        limit: usize,
        memo: &mut FxHashMap<TyId<'db>, usize>,
    ) -> Option<usize> {
        if let Some(&known) = memo.get(&ty) {
            return Some(known);
        }
        let mut total = 1usize;
        for child in children(db, ty) {
            total = total.checked_add(size(db, child, limit, memo)?)?;
            if total > limit {
                return None;
            }
        }
        memo.insert(ty, total);
        Some(total)
    }
    if limit == 0 {
        return None;
    }
    size(db, ty, limit, &mut FxHashMap::default())
}

/// Whether `ty`, as a tree, nests at most `limit` levels of types.
pub(crate) fn tree_depth_within<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    limit: usize,
) -> bool {
    fn depth<'db>(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        limit: usize,
        memo: &mut FxHashMap<TyId<'db>, usize>,
    ) -> Option<usize> {
        if let Some(&known) = memo.get(&ty) {
            return (known <= limit).then_some(known);
        }
        let below = limit.checked_sub(1)?;
        let mut deepest = 0;
        for child in children(db, ty) {
            deepest = deepest.max(depth(db, child, below, memo)?);
        }
        memo.insert(ty, deepest + 1);
        Some(deepest + 1)
    }
    depth(db, ty, limit, &mut FxHashMap::default()).is_some()
}

impl<'db> TyFolder<'db> for TypeNormalizer<'db> {
    fn fold_ty_app(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        abs: TyId<'db>,
        arg: TyId<'db>,
    ) -> TyId<'db> {
        if self.resolve_impls {
            TyId::app(db, abs, arg)
        } else {
            TyId::app_structural(db, abs, arg)
        }
    }

    fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
        match ty.data(self.db) {
            TyData::TyParam(p @ TyParam { owner, .. }) if p.is_trait_self() => {
                if let Some(impl_) = owner.resolve_to::<ImplTrait>(self.db) {
                    // Use the item method to obtain the implementor's self type.
                    let lowered = impl_.ty(self.db);
                    return self.fold_ty(db, lowered);
                }
                ty
            }
            TyData::AssocTy(_) => normalize_projection(
                db,
                ProjectionQuery::new(
                    db,
                    ty,
                    self.scope,
                    self.assumptions,
                    self.resolve_impls,
                    self.budget,
                ),
            ),
            _ => ty.super_fold_with(db, self),
        }
    }
}

impl<'db> TypeNormalizer<'db> {
    fn try_resolve_assoc_ty(&mut self, ty: TyId<'db>, assoc: &AssocTy<'db>) -> Option<TyId<'db>> {
        // Equality evidence is separate from the projection's identity. Match
        // its entire trait reference before using an assumption's binding.
        let target = assoc.trait_.fold_with(self.db, self);
        let mut matching_bounds: IndexMap<TyId<'db>, ()> = IndexMap::new();
        for &pred in self.assumptions.list(self.db) {
            let Some(bound) = pred.bound_assoc_ty(self.db, assoc.name) else {
                continue;
            };
            if self.trait_refs_match(target, pred.trait_ref(self.db)) {
                matching_bounds.insert(self.fold_ty(self.db, bound), ());
            }
        }
        if matching_bounds.len() > 1 {
            return None;
        }
        if let Some((&bound, _)) = matching_bounds.first() {
            return (bound != ty).then_some(bound);
        }

        if !self.resolve_impls {
            return None;
        }

        if let Some(out) = callable_out_ty(
            self.db,
            self.fold_ty(self.db, assoc.trait_.self_ty(self.db)),
            assoc.trait_.def(self.db),
            assoc.name,
        ) {
            return Some(out);
        }

        self.try_resolve_assoc_ty_from_impls(assoc)
    }

    /// A pure normalization query may observe established equality, but may
    /// not choose a binding by assigning an unresolved caller inference var.
    fn trait_refs_match(&mut self, target: TraitRefId<'db>, candidate: TraitRefId<'db>) -> bool {
        if target.def(self.db) != candidate.def(self.db) {
            return false;
        }
        let candidate = candidate.fold_with(self.db, self);
        if target == candidate {
            return true;
        }
        if !collect_variables(self.db, &target).is_empty()
            || !collect_variables(self.db, &candidate).is_empty()
        {
            return false;
        }
        UnificationTable::new(self.db)
            .unify::<TraitRefId<'db>>(target, candidate)
            .is_ok()
    }

    /// Resolves `assoc` through the impls whose header, every trait argument
    /// included, matches its trait reference. Nothing is proved: where clauses
    /// only choose between impls whose headers both match, and the trait
    /// reference itself is checked where the projection is written. So an
    /// impl whose where clause names the projection it defines, as
    /// `impl Tr for S where <S as Tr>::Out: Show`, needs nothing special.
    fn try_resolve_assoc_ty_from_impls(&mut self, assoc: &AssocTy<'db>) -> Option<TyId<'db>> {
        let trait_inst = assoc.trait_.fold_with(self.db, self).as_predicate(self.db);
        let canonical_inst = Canonical::new(self.db, trait_inst);

        let solve_cx = TraitSolveCx::new(self.db, self.scope).with_assumptions(self.assumptions);
        let (primary, secondary) = solve_cx.search_ingots_for_trait_inst(self.db, trait_inst);
        let mut candidates: IndexSet<ImplementorId<'db>> = [Some(primary), secondary]
            .into_iter()
            .flatten()
            .flat_map(|ingot| impls_for_trait_inst(self.db, ingot, canonical_inst))
            .copied()
            .collect();
        if candidates.len() > 1 {
            candidates.retain(|&implementor| {
                !impl_where_clause_refuted(self.db, solve_cx, implementor, canonical_inst)
            });
        }

        let mut dedup: IndexSet<TyId<'db>> = IndexSet::default();
        // Canonicalize the target trait instance so we can unify against it in a
        // fresh table without mixing inference keys from other tables.
        let canonical_target = Canonicalized::new(self.db, trait_inst);
        canonical_target.with_materialized(self.db, |cx| {
            let target_inst = cx.query();
            let original_target = cx.try_extract::<TraitInstId<'db>>(target_inst);
            for implementor in candidates {
                let Some(implementor) = complete_impl_assoc_ty(self.db, implementor, assoc.name)
                else {
                    continue;
                };
                let candidate = cx.with_impl_assoc_ty(
                    implementor,
                    target_inst.self_ty(self.db),
                    assoc.name,
                    |cx, inst, assoc_ty| {
                        cx.unify::<TraitInstId<'db>>(inst, target_inst).ok()?;
                        if cx.try_extract::<TraitInstId<'db>>(target_inst) != original_target {
                            return None;
                        }
                        let assoc_ty = cx.resolve::<TyId<'db>>(assoc_ty);
                        cx.try_extract::<TyId<'db>>(assoc_ty)
                    },
                );

                // Extract into the caller's inference environment before
                // continuing normalization, so scratch-local vars never
                // leak into the cache.
                if let Some(Some(folded)) = candidate {
                    dedup.insert(self.fold_ty(self.db, folded));
                }
            }
        });

        match dedup.len() {
            1 => dedup.first().copied(),
            _ => None,
        }
    }
}

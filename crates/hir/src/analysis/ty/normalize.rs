//! Type normalization module
//!
//! This module provides functionality to normalize types by resolving associated types
//! to concrete types when possible. This happens before type unification to ensure
//! that types are in their most resolved form.

use std::collections::hash_map::Entry;

use crate::core::hir_def::{ImplTrait, scope_graph::ScopeId};
use common::indexmap::IndexMap;
use rustc_hash::FxHashMap;

use super::{
    canonical::Canonical,
    canonical::Canonicalized,
    fold::{TyFoldable, TyFolder},
    trait_def::{impls_for_ty_with_constraints, impls_for_ty_with_constraints_in_cx},
    trait_resolution::{PredicateListId, TraitSolveCx},
    ty_def::{AssocTy, InvalidCause, TyData, TyId, TyParam},
    unify::UnificationTable,
    visitor::{TyVisitable, TyVisitor},
};
use crate::analysis::{HirAnalysisDb, name_resolution::find_associated_type};

/// Normalizes a type by resolving all associated types to concrete types when possible.
///
/// This function takes a type and attempts to resolve any associated types within it
/// using the provided assumptions and scope context. It handles:
/// - Simple associated types (e.g., `T::Output`)
/// - Nested associated types (e.g., `T::Encoder::Output`)
/// - Associated types with generic parameters
pub fn normalize_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    normalize_ty_with_solve_cx(db, ty, scope, assumptions, None)
}

pub fn normalize_ty_with_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    solve_cx: Option<TraitSolveCx<'db>>,
) -> TyId<'db> {
    normalize_ty_impl(db, ty, scope, assumptions, solve_cx, true)
}

pub(crate) fn normalize_ty_without_consts_with_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    solve_cx: Option<TraitSolveCx<'db>>,
) -> TyId<'db> {
    normalize_ty_impl(db, ty, scope, assumptions, solve_cx, false)
}

fn normalize_ty_impl<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    solve_cx: Option<TraitSolveCx<'db>>,
    normalize_const_tys: bool,
) -> TyId<'db> {
    let mut normalizer = TypeNormalizer::new(db, scope, assumptions, solve_cx, normalize_const_tys);
    ty.fold_with(db, &mut normalizer)
}

pub struct TypeNormalizer<'db> {
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    solve_cx: Option<TraitSolveCx<'db>>,
    normalize_const_tys: bool,
    // Projection cache: None = in progress (cycle guard), Some(ty) = normalized result
    cache: FxHashMap<AssocTy<'db>, Option<TyId<'db>>>,
}

#[derive(Clone, Copy)]
struct AssumptionAssocBindingMatch<'db> {
    lhs_self: TyId<'db>,
    rhs_self: TyId<'db>,
    bound: TyId<'db>,
}

impl<'db> TyVisitable<'db> for AssumptionAssocBindingMatch<'db> {
    fn visit_with<V>(&self, visitor: &mut V)
    where
        V: TyVisitor<'db> + ?Sized,
    {
        self.lhs_self.visit_with(visitor);
        self.rhs_self.visit_with(visitor);
        self.bound.visit_with(visitor);
    }
}

impl<'db> TyFoldable<'db> for AssumptionAssocBindingMatch<'db> {
    fn super_fold_with<F>(self, db: &'db dyn HirAnalysisDb, folder: &mut F) -> Self
    where
        F: TyFolder<'db>,
    {
        Self {
            lhs_self: self.lhs_self.fold_with(db, folder),
            rhs_self: self.rhs_self.fold_with(db, folder),
            bound: self.bound.fold_with(db, folder),
        }
    }
}

impl<'db> TypeNormalizer<'db> {
    pub fn new(
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        assumptions: PredicateListId<'db>,
        solve_cx: Option<TraitSolveCx<'db>>,
        normalize_const_tys: bool,
    ) -> Self {
        Self {
            db,
            scope,
            assumptions,
            solve_cx,
            normalize_const_tys,
            cache: FxHashMap::default(),
        }
    }

    fn solve_cx(&self) -> TraitSolveCx<'db> {
        self.solve_cx.unwrap_or_else(|| {
            TraitSolveCx::new(self.db, self.scope).with_assumptions(self.assumptions)
        })
    }

    fn normalize_const_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
        let folded = ty.super_fold_with(db, self);
        let TyData::ConstTy(const_ty) = folded.data(db) else {
            return folded;
        };
        let evaluated = const_ty.evaluate_with_solve_cx(db, Some(const_ty.ty(db)), self.solve_cx());
        if matches!(
            evaluated.ty(db).invalid_cause(db),
            Some(InvalidCause::ConstEvalUnsupported { .. })
        ) {
            return folded;
        }
        TyId::const_ty(db, evaluated)
    }
}

impl<'db> TyFolder<'db> for TypeNormalizer<'db> {
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
            TyData::AssocTy(assoc_ty) => {
                match self.cache.entry(*assoc_ty) {
                    Entry::Occupied(entry) => match entry.get() {
                        Some(cached) => return *cached,
                        None => return ty, // cycle: leave unresolved
                    },
                    Entry::Vacant(entry) => {
                        entry.insert(None);
                    }
                }

                if let Some(replacement) = self.try_resolve_assoc_ty(ty, assoc_ty) {
                    let normalized = self.fold_ty(db, replacement);
                    self.cache.insert(*assoc_ty, Some(normalized));
                    return normalized;
                }

                // Not resolved; still fold internals (e.g., normalize self type)
                let folded = ty.super_fold_with(db, self);
                self.cache.insert(*assoc_ty, Some(folded));
                folded
            }
            TyData::ConstTy(_) if self.normalize_const_tys => self.normalize_const_ty(db, ty),
            _ => ty.super_fold_with(db, self),
        }
    }
}

impl<'db> TypeNormalizer<'db> {
    fn try_resolve_assoc_ty(&mut self, ty: TyId<'db>, assoc: &AssocTy<'db>) -> Option<TyId<'db>> {
        // 1) Check if the trait instance itself carries an explicit binding
        if let Some(&bound_ty) = assoc.trait_.assoc_type_bindings(self.db).get(&assoc.name) {
            return Some(bound_ty);
        }

        // 2) Check assumptions for an equivalent trait instance that carries
        //    an explicit associated type binding (e.g. from where-clauses).
        //    Canonicalize the whole match candidate first so we can safely
        //    unify in a fresh table without mixing outer inference vars.
        let lhs_self = self.fold_ty(self.db, assoc.trait_.self_ty(self.db));
        for &pred in self.assumptions.list(self.db) {
            if pred.def(self.db) != assoc.trait_.def(self.db) {
                continue;
            }

            let Some(&bound) = pred.assoc_type_bindings(self.db).get(&assoc.name) else {
                continue;
            };

            let candidate = AssumptionAssocBindingMatch {
                lhs_self,
                rhs_self: self.fold_ty(self.db, pred.self_ty(self.db)),
                bound,
            };
            let canonical_candidate = Canonicalized::new(self.db, candidate);
            let mut table = UnificationTable::new(self.db);
            let candidate = canonical_candidate.value.extract_identity(&mut table);

            if table.unify(candidate.lhs_self, candidate.rhs_self).is_ok() {
                let bound = candidate.bound.fold_with(self.db, &mut table);
                return Some(canonical_candidate.decanonicalize(self.db, bound));
            }
        }

        let self_ty = self.fold_ty(self.db, assoc.trait_.self_ty(self.db));
        if matches!(self_ty.data(self.db), TyData::TyParam(_) | TyData::TyVar(_)) {
            return None;
        }

        // 3) Fall back to the general associated type search used by path resolution,
        //    but restrict results to the same trait as `assoc` and deduplicate by
        //    the resulting type. If all viable candidates agree on a single type,
        //    normalize to that type.
        //
        // First attempt an impl-based lookup across relevant ingots (Self's + trait's),
        // mirroring trait-method resolution. This allows normalization to succeed even
        // when the calling scope is in a different ingot (e.g., core code instantiated
        // with std types).
        if let Some(resolved) = self.try_resolve_assoc_ty_from_impls(assoc) {
            return Some(resolved);
        }

        //    Search by the trait's self type: `SelfTy::assoc.name`.
        let mut raw_cands = find_associated_type(
            self.db,
            self.scope,
            Canonical::new(self.db, self_ty),
            assoc.name,
            self.assumptions,
        );

        // Keep only candidates from the same trait as `assoc`.
        raw_cands.retain(|(inst, _)| inst.def(self.db) == assoc.trait_.def(self.db));

        // Deduplicate by normalized result type (to handle cases where multiple
        // impls yield the same associated type, e.g., Output = Self for all impls).
        let mut dedup: IndexMap<TyId<'db>, ()> = IndexMap::new();
        for (_, t) in raw_cands.into_iter() {
            // Continue folding so nested associated types are also normalized
            let norm_t = self.fold_ty(self.db, t);
            dedup.entry(norm_t).or_insert(());
        }

        match dedup.len() {
            0 => None,
            1 => {
                let (unique, _) = dedup.first().unwrap();
                // Only replace if we're actually making progress
                if *unique != ty { Some(*unique) } else { None }
            }
            _ => None,
        }
    }

    fn try_resolve_assoc_ty_from_impls(&mut self, assoc: &AssocTy<'db>) -> Option<TyId<'db>> {
        let trait_inst = assoc.trait_.fold_with(self.db, self);
        let trait_def = trait_inst.def(self.db);
        let canonical_self_ty = Canonical::new(self.db, trait_inst.self_ty(self.db));

        let mut dedup: IndexMap<TyId<'db>, ()> = IndexMap::new();

        let solve_cx = self.solve_cx();
        let (primary, secondary) = solve_cx.search_ingots_for_trait_inst(self.db, trait_inst);
        let search_ingots = if primary.is_none() && secondary.is_none() {
            vec![None]
        } else {
            vec![primary, secondary]
        };

        // Canonicalize the target trait instance so we can unify against it in a
        // fresh table without mixing inference keys from other tables.
        let canonical_target = Canonicalized::new(self.db, trait_inst);
        let canonical_inst = canonical_target.value;

        let mut table = UnificationTable::new(self.db);
        let target_inst = canonical_inst.extract_identity(&mut table);

        for ingot in search_ingots {
            let implementors = if let Some(solve_cx) = self.solve_cx {
                impls_for_ty_with_constraints_in_cx(self.db, ingot, canonical_self_ty, solve_cx)
            } else if let Some(ingot) = ingot {
                impls_for_ty_with_constraints(self.db, ingot, canonical_self_ty, self.assumptions)
            } else {
                continue;
            };

            for implementor in implementors {
                let snapshot = table.snapshot();
                let implementor = table.instantiate_with_fresh_vars(implementor);

                // Filter by trait before unifying (cheap early out).
                if implementor.trait_def(self.db) != trait_def {
                    table.rollback_to(snapshot);
                    continue;
                }

                if table
                    .unify(implementor.trait_(self.db), target_inst)
                    .is_err()
                {
                    table.rollback_to(snapshot);
                    continue;
                }

                let Some(assoc_ty) = implementor
                    .types(self.db)
                    .get(&assoc.name)
                    .copied()
                    .or_else(|| {
                        implementor
                            .trait_(self.db)
                            .assoc_type_bindings(self.db)
                            .get(&assoc.name)
                            .copied()
                    })
                else {
                    table.rollback_to(snapshot);
                    continue;
                };

                // Apply substitutions, then decanonicalize back to the original
                // inference vars before further normalization.
                let folded = assoc_ty.fold_with(self.db, &mut table);
                let folded = canonical_target.decanonicalize(self.db, folded);
                let norm = self.fold_ty(self.db, folded);
                dedup.entry(norm).or_insert(());

                table.rollback_to(snapshot);
            }
        }

        match dedup.len() {
            0 => None,
            1 => Some(*dedup.first().unwrap().0),
            _ => None,
        }
    }
}

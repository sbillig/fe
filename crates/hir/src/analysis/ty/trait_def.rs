//! This module contains all trait related types definitions.

use crate::{
    analysis::ty::{
        method_cmp::compare_impl_method,
        trait_lower::{collect_trait_impls, collect_trait_impls_frontier},
        trait_resolution::{GoalSatisfiability, PredicateListId, Selection},
    },
    hir_def::{Contract, Func, HirIngot, IdentId, ImplTrait, Trait},
};
use common::{
    indexmap::{IndexMap, IndexSet},
    ingot::{Ingot, IngotKind},
};
use rustc_hash::FxHashMap;
use salsa::Update;

pub use super::assoc_items::AssocConstBodyOrigin;
use super::{
    assoc_items::normalize_ty_for_trait_inst as normalize_assoc_ty_for_trait_inst,
    binder::Binder,
    canonical::{Canonical, Canonicalized},
    context::{AnalysisCx, ImplOverlay, ProofCx},
    diagnostics::{ImplDiag, TyDiagCollection},
    fold::TyFoldable as _,
    trait_lower::collect_implementor_methods,
    trait_resolution::{TraitSolveCx, constraint::collect_constraints, is_goal_satisfiable},
    ty_def::TyId,
    unify::UnificationTable,
    visitor::{TyVisitable, TyVisitor},
};
use crate::analysis::HirAnalysisDb;

/// Returns [`TraitEnv`] for the given ingot.
#[salsa::tracked(return_ref, cycle_fn=ingot_trait_env_cycle_recover, cycle_initial=ingot_trait_env_cycle_initial)]
pub(crate) fn ingot_trait_env<'db>(db: &'db dyn HirAnalysisDb, ingot: Ingot<'db>) -> TraitEnv<'db> {
    TraitEnv::collect(db, ingot)
}

/// Returns all implementors for the given trait definition.
///
/// Note: this intentionally does **not** pre-filter implementors by unifying with a
/// specific goal instance. Projection-heavy goals (e.g. involving associated type
/// projections) often only become unifiable after normalization, and unification
/// rejects unresolved associated types. The solver normalizes before unifying
/// candidates, so any filtering here must be an over-approximation.
pub(crate) fn impls_for_trait_def<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
    trait_def: Trait<'db>,
) -> Vec<Binder<ImplementorId<'db>>> {
    let env = ingot_trait_env(db, ingot);
    let mut out = env.impls.get(&trait_def).cloned().unwrap_or_default();

    if is_std_evm_contract_trait_def(db, trait_def) {
        out.extend(contract_virtual_impls(db, ingot).iter().copied());
    }

    out
}

/// Returns all implementors for the given trait inst, searching across a
/// deterministic set of ingots.
///
/// This is used to avoid "pick an ingot" footguns where impl lookup depends on
/// the caller's current module ingot and can miss impls that live in either:
/// - the trait's ingot, or
/// - the implementor type's ingot.
pub(crate) fn impls_for_trait_in_ingots<'db>(
    db: &'db dyn HirAnalysisDb,
    primary: Option<Ingot<'db>>,
    secondary: Option<Ingot<'db>>,
    trait_def: Trait<'db>,
) -> Vec<Binder<ImplementorId<'db>>> {
    let mut dedup: IndexSet<Binder<ImplementorId<'db>>> = IndexSet::default();
    for ingot in [primary, secondary].into_iter().flatten() {
        dedup.extend(impls_for_trait_def(db, ingot, trait_def).iter().copied());
    }
    dedup.into_iter().collect()
}

fn is_std_evm_contract_trait_def<'db>(db: &'db dyn HirAnalysisDb, trait_def: Trait<'db>) -> bool {
    let Some(name) = trait_def.name(db).to_opt() else {
        return false;
    };
    if name.data(db) != "Contract" {
        return false;
    }
    if trait_def.top_mod(db).ingot(db).kind(db) != IngotKind::Std {
        return false;
    }
    true
}

#[salsa::tracked(return_ref)]
fn contract_virtual_impls<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> Vec<Binder<ImplementorId<'db>>> {
    let Some(contract_trait) = std_evm_contract_trait_def(db, ingot) else {
        return Vec::new();
    };

    let init_args_ident = IdentId::new(db, "InitArgs".to_string());

    let mut out = Vec::new();
    for top_mod in ingot.all_modules(db) {
        for &contract in top_mod.all_contracts(db).iter() {
            let self_ty = TyId::contract(db, contract);
            let trait_inst = TraitInstId::new(db, contract_trait, vec![self_ty], IndexMap::new());

            let init_args_ty = contract.init_args_ty(db);
            let mut types = IndexMap::new();
            types.insert(init_args_ident, init_args_ty);

            let implementor = ImplementorId::new(
                db,
                trait_inst,
                Vec::new(),
                types,
                ImplementorOrigin::VirtualContract(contract),
            );
            out.push(Binder::bind(implementor));
        }
    }

    out
}

#[salsa::tracked]
fn std_evm_contract_trait_def<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> Option<Trait<'db>> {
    use crate::analysis::name_resolution::resolve_path;
    use common::ingot::IngotKind;

    let scope = ingot.root_mod(db).scope();
    let assumptions = PredicateListId::empty_list(db);

    let std_root = if ingot.kind(db) == IngotKind::Std {
        IdentId::make_ingot(db)
    } else {
        IdentId::new(db, "std".to_string())
    };

    let path = crate::hir_def::PathId::from_ident(db, std_root)
        .push_ident(db, IdentId::new(db, "evm".to_string()))
        .push_ident(db, IdentId::new(db, "Contract".to_string()));

    match resolve_path(db, path, scope, assumptions, false).ok()? {
        crate::analysis::name_resolution::PathRes::Trait(inst) => Some(inst.def(db)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum ImplementorOrigin<'db> {
    Hir(ImplTrait<'db>),
    VirtualContract(Contract<'db>),
    Assumption,
}

#[derive(Debug, Clone, Copy)]
struct ImplementorTyCheckHeader<'db> {
    self_ty: TyId<'db>,
    constraints: PredicateListId<'db>,
}

impl<'db> TyVisitable<'db> for ImplementorTyCheckHeader<'db> {
    fn visit_with<V>(&self, visitor: &mut V)
    where
        V: TyVisitor<'db> + ?Sized,
    {
        self.self_ty.visit_with(visitor);
        self.constraints.visit_with(visitor);
    }
}

impl<'db> super::fold::TyFoldable<'db> for ImplementorTyCheckHeader<'db> {
    fn super_fold_with<F>(self, db: &'db dyn HirAnalysisDb, folder: &mut F) -> Self
    where
        F: super::fold::TyFolder<'db>,
    {
        Self {
            self_ty: self.self_ty.fold_with(db, folder),
            constraints: self.constraints.fold_with(db, folder),
        }
    }
}

fn ingot_trait_env_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
) -> TraitEnv<'db> {
    // When local impl admission recursively re-enters trait-env lookup, use the
    // current fixed-point frontier from `collect_trait_impls_frontier` alongside the
    // stable external env so projection/method/const checks can see already
    // admitted local helpers during the next iteration.
    TraitEnv::from_impl_maps(
        db,
        ingot,
        ingot
            .resolved_external_ingots(db)
            .iter()
            .map(|(_, external)| collect_trait_impls(db, *external))
            .chain(std::iter::once(collect_trait_impls_frontier(db, ingot))),
    )
}

fn ingot_trait_env_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &TraitEnv<'db>,
    _count: u32,
    _ingot: Ingot<'db>,
) -> salsa::CycleRecoveryAction<TraitEnv<'db>> {
    // Continue iterating to try to resolve the cycle
    salsa::CycleRecoveryAction::Iterate
}

/// Resolves the concrete HIR function that implements `method` for the given
/// trait instance, returning both the function and the impl's instantiated
/// generic arguments.
pub fn resolve_trait_method_instance<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    inst: TraitInstId<'db>,
    method: IdentId<'db>,
) -> Option<(Func<'db>, Vec<TyId<'db>>)> {
    // Normalize the trait instance arguments before searching for implementors.
    //
    // This is important for cases where the `Self` type is an associated type
    // projection (e.g. `<Sol as Abi>::Decoder<I>`). Without normalization, the
    // projection has no declaring ingot, which prevents us from searching the
    // ingot that contains the actual implementor type (e.g. `SolDecoder<I>`).
    //
    // Monomorphization happens with concrete substitutions, so we can safely
    // normalize using a scope derived from the instantiated arguments.
    let assumptions = solve_cx.assumptions();
    let norm_scope = solve_cx.normalization_scope_for_trait_inst(db, inst);
    let inst = inst.normalize_with_solve_cx(db, solve_cx, norm_scope, assumptions);

    let implementor = match solve_cx.select_impl(db, inst) {
        Selection::Unique(implementor) => implementor,
        Selection::Ambiguous(_ambiguous) => return None,
        Selection::NotFound => return None,
    };
    let &func = implementor.methods(db).get(&method)?;

    let mut table = UnificationTable::new(db);
    let implementor = table.instantiate_with_fresh_vars(Binder::bind(implementor));
    table.unify(implementor.trait_inst(db), inst).ok()?;
    let impl_args = implementor
        .params(db)
        .iter()
        .map(|&ty| ty.fold_with(db, &mut table))
        .collect();
    Some((func, impl_args))
}

/// Returns all implementors for the given `ty` that satisfy the given assumptions.
pub(crate) fn impls_for_ty_with_constraints<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
    ty: Canonical<TyId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Vec<Binder<ImplementorId<'db>>> {
    let mut table = UnificationTable::new(db);
    let ty = ty.extract_identity(&mut table);

    let env = ingot_trait_env(db, ingot);
    let solve_cx = TraitSolveCx::new(db, ingot.root_mod(db).scope()).with_assumptions(assumptions);
    if ty.has_invalid(db) {
        return vec![];
    }

    let mut cands = vec![];
    for (key, insts) in env.ty_to_implementors.iter() {
        let snapshot = table.snapshot();
        let key = table.instantiate_with_fresh_vars(*key);
        if table.unify(key, ty.base_ty(db)).is_ok() {
            cands.push(insts);
        }

        table.rollback_to(snapshot);
    }

    let mut raw_impls: Vec<Binder<ImplementorId<'db>>> =
        cands.into_iter().flatten().copied().collect();

    if ty.as_contract(db).is_some() {
        raw_impls.extend(contract_virtual_impls(db, ingot).iter().copied());
    }

    raw_impls
        .into_iter()
        .filter(|impl_| {
            let snapshot = table.snapshot();
            let inst = table.instantiate_with_fresh_vars(Binder::bind(ImplementorTyCheckHeader {
                self_ty: impl_.instantiate_identity().self_ty(db),
                constraints: impl_.instantiate_identity().constraints(db),
            }));
            let impl_ty = table.instantiate_to_term(inst.self_ty);
            let ty_term = table.instantiate_to_term(ty);
            let unifies = table.unify(impl_ty, ty_term).is_ok();

            if unifies {
                // Filter out impls that don't satisfy assumptions
                if inst.constraints.is_empty(db) {
                    table.rollback_to(snapshot);
                    return true;
                }

                for &constraint in inst.constraints.list(db) {
                    let constraint = Canonicalized::new(db, constraint);
                    match is_goal_satisfiable(db, solve_cx, constraint.value) {
                        GoalSatisfiability::UnSat(_) => {
                            table.rollback_to(snapshot);
                            return false;
                        }
                        _ => {
                            // Ignoring the NeedsConfirmation case for now
                        }
                    }
                }
            }

            table.rollback_to(snapshot);
            unifies
        })
        .collect()
}

/// Returns all implementors for the given `ty` that satisfy the given solve context,
/// including admission-local overlay implementors.
pub(crate) fn impls_for_ty_with_constraints_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Option<Ingot<'db>>,
    ty: Canonical<TyId<'db>>,
    solve_cx: TraitSolveCx<'db>,
) -> Vec<Binder<ImplementorId<'db>>> {
    let mut dedup: IndexSet<_> = ingot
        .map(|ingot| impls_for_ty_with_constraints(db, ingot, ty, solve_cx.assumptions()))
        .unwrap_or_default()
        .into_iter()
        .collect();

    let mut table = UnificationTable::new(db);
    let ty = ty.extract_identity(&mut table);
    if ty.has_invalid(db) {
        return dedup.into_iter().collect();
    }

    for &implementor in solve_cx.local_implementors(db) {
        let snapshot = table.snapshot();
        let inst = table.instantiate_with_fresh_vars(Binder::bind(ImplementorTyCheckHeader {
            self_ty: implementor.instantiate_identity().self_ty(db),
            constraints: implementor.instantiate_identity().constraints(db),
        }));
        let impl_ty = table.instantiate_to_term(inst.self_ty);
        let ty_term = table.instantiate_to_term(ty);
        let mut is_ok = table.unify(impl_ty, ty_term).is_ok();

        if is_ok {
            for &constraint in inst.constraints.list(db) {
                let constraint = Canonicalized::new(db, constraint);
                if matches!(
                    is_goal_satisfiable(db, solve_cx, constraint.value),
                    GoalSatisfiability::UnSat(_)
                ) {
                    is_ok = false;
                    break;
                }
            }
        }

        table.rollback_to(snapshot);
        if is_ok {
            dedup.insert(implementor);
        }
    }

    dedup.into_iter().collect()
}

/// Returns all implementors for the given `ty`.
#[salsa::tracked(return_ref)]
pub(crate) fn impls_for_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
    ty: Canonical<TyId<'db>>,
) -> Vec<Binder<ImplementorId<'db>>> {
    let mut table = UnificationTable::new(db);
    let ty = ty.extract_identity(&mut table);

    let env = ingot_trait_env(db, ingot);
    if ty.has_invalid(db) {
        return vec![];
    }

    let mut cands = vec![];
    for (key, insts) in env.ty_to_implementors.iter() {
        let snapshot = table.snapshot();
        let key = table.instantiate_with_fresh_vars(*key);
        if table.unify(key, ty.base_ty(db)).is_ok() {
            cands.push(insts);
        }
        table.rollback_to(snapshot);
    }

    let mut raw_impls: Vec<Binder<ImplementorId<'db>>> =
        cands.into_iter().flatten().copied().collect();

    if ty.as_contract(db).is_some() {
        raw_impls.extend(contract_virtual_impls(db, ingot).iter().copied());
    }

    raw_impls
        .into_iter()
        .filter(|impl_| {
            let snapshot = table.snapshot();

            let self_ty = table.instantiate_with_fresh_vars(Binder::bind(
                impl_.instantiate_identity().self_ty(db),
            ));
            let impl_ty = table.instantiate_to_term(self_ty);
            let ty_term = table.instantiate_to_term(ty);
            let is_ok = table.unify(impl_ty, ty_term).is_ok();

            table.rollback_to(snapshot);

            is_ok
        })
        .collect()
}

pub(crate) fn normalize_ty_for_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    ty: TyId<'db>,
    trait_inst: TraitInstId<'db>,
) -> TyId<'db> {
    normalize_assoc_ty_for_trait_inst(db, &AnalysisCx::from_solve_cx(solve_cx), ty, trait_inst)
}

pub(crate) fn specialize_trait_const_inst_to_receiver<'db>(
    db: &'db dyn HirAnalysisDb,
    recv_ty: TyId<'db>,
    inst: TraitInstId<'db>,
) -> TraitInstId<'db> {
    if inst.self_ty(db) == recv_ty {
        return inst;
    }

    let mut table = UnificationTable::new(db);
    let specialized = table.instantiate_with_fresh_vars(Binder::bind(inst));
    if table.unify(specialized.self_ty(db), recv_ty).is_err() {
        return inst;
    }

    specialized.fold_with(db, &mut table)
}

/// Represents the trait environment of an ingot, which maintain all trait
/// implementors which can be used in the ingot.
#[derive(Debug, PartialEq, Eq, Clone, Update)]
pub(crate) struct TraitEnv<'db> {
    /// Implementors grouped by trait definition.
    pub(crate) impls: FxHashMap<Trait<'db>, Vec<Binder<ImplementorId<'db>>>>,

    /// This maintains a mapping from the base type to the implementors.
    ty_to_implementors: FxHashMap<Binder<TyId<'db>>, Vec<Binder<ImplementorId<'db>>>>,

    ingot: Ingot<'db>,
}

impl<'db> TraitEnv<'db> {
    fn from_impl_maps<'a>(
        db: &'db dyn HirAnalysisDb,
        ingot: Ingot<'db>,
        impl_maps: impl IntoIterator<Item = &'a FxHashMap<Trait<'db>, Vec<Binder<ImplementorId<'db>>>>>,
    ) -> Self
    where
        'db: 'a,
    {
        let mut impls: FxHashMap<Trait<'db>, Vec<Binder<ImplementorId<'db>>>> =
            FxHashMap::default();
        let mut ty_to_implementors: FxHashMap<Binder<TyId>, Vec<Binder<ImplementorId<'db>>>> =
            FxHashMap::default();

        for impl_map in impl_maps {
            for (trait_def, implementors) in impl_map.iter() {
                impls
                    .entry(*trait_def)
                    .or_default()
                    .extend(implementors.iter().copied());

                for implementor in implementors {
                    let self_ty = implementor.instantiate_identity().self_ty(db);
                    ty_to_implementors
                        .entry(Binder::bind(self_ty.base_ty(db)))
                        .or_default()
                        .push(*implementor);
                }
            }
        }

        Self {
            impls,
            ty_to_implementors,
            ingot,
        }
    }

    fn collect(db: &'db dyn HirAnalysisDb, ingot: Ingot<'db>) -> Self {
        Self::from_impl_maps(
            db,
            ingot,
            ingot
                .resolved_external_ingots(db)
                .iter()
                .map(|(_, external)| collect_trait_impls(db, *external))
                .chain(std::iter::once(collect_trait_impls(db, ingot))),
        )
    }
}

/// Represents a slim, internal view of a trait impl, derived from an
/// `ImplTrait` item and its lowered trait instance.
#[salsa::interned]
#[derive(Debug)]
pub struct ImplementorId<'db> {
    /// The trait instance that this impl realizes.
    pub(crate) trait_: TraitInstId<'db>,

    /// The type parameters of this implementor.
    #[return_ref]
    pub(crate) params: Vec<TyId<'db>>,

    #[return_ref]
    pub(crate) types: IndexMap<IdentId<'db>, TyId<'db>>,

    pub(crate) origin: ImplementorOrigin<'db>,
}

impl<'db> ImplementorId<'db> {
    pub(crate) fn assumption(db: &'db dyn HirAnalysisDb, inst: TraitInstId<'db>) -> Self {
        ImplementorId::new(
            db,
            inst,
            Vec::new(),
            IndexMap::new(),
            ImplementorOrigin::Assumption,
        )
    }

    pub(crate) fn hir_impl_trait(self, db: &'db dyn HirAnalysisDb) -> ImplTrait<'db> {
        match self.origin(db) {
            ImplementorOrigin::Hir(impl_trait) => impl_trait,
            ImplementorOrigin::VirtualContract(contract) => panic!(
                "requested HIR impl-trait for virtual implementor (contract={})",
                contract
                    .name(db)
                    .to_opt()
                    .map(|n| n.data(db).to_string())
                    .unwrap_or_else(|| "<unknown>".to_string())
            ),
            ImplementorOrigin::Assumption => {
                panic!("requested HIR impl-trait for assumption-based implementor")
            }
        }
    }

    /// Associated type defined in this impl, if any.
    pub(crate) fn assoc_ty(
        self,
        db: &'db dyn HirAnalysisDb,
        name: IdentId<'db>,
    ) -> Option<TyId<'db>> {
        self.types(db)
            .get(&name)
            .copied()
            .or_else(|| self.trait_(db).assoc_type_bindings(db).get(&name).copied())
    }

    /// Trait definition implemented by this impl.
    pub(crate) fn trait_def(self, db: &'db dyn HirAnalysisDb) -> Trait<'db> {
        self.trait_(db).def(db)
    }

    /// Semantic self type of this impl.
    pub(crate) fn self_ty(self, db: &'db dyn HirAnalysisDb) -> TyId<'db> {
        self.trait_(db).self_ty(db)
    }

    /// Trait instance realized by this impl, including its associated type definitions.
    pub(crate) fn trait_inst(self, db: &'db dyn HirAnalysisDb) -> TraitInstId<'db> {
        let trait_inst = self.trait_(db);
        let mut assoc_type_bindings = trait_inst.assoc_type_bindings(db).clone();
        match self.origin(db) {
            ImplementorOrigin::Hir(_)
            | ImplementorOrigin::VirtualContract(_)
            | ImplementorOrigin::Assumption => {
                if self.types(db).is_empty() {
                    return trait_inst;
                }
                for (name, ty) in self.types(db) {
                    assoc_type_bindings.insert(*name, *ty);
                }
            }
        }

        TraitInstId::new(
            db,
            trait_inst.def(db),
            trait_inst.args(db).to_vec(),
            assoc_type_bindings,
        )
    }

    /// Returns the constraints that the implementor requires when the
    /// implementation is selected.
    pub(crate) fn constraints(self, db: &'db dyn HirAnalysisDb) -> PredicateListId<'db> {
        match self.origin(db) {
            ImplementorOrigin::Hir(impl_trait) => {
                collect_constraints(db, impl_trait.into()).instantiate(db, self.params(db))
            }
            ImplementorOrigin::VirtualContract(_) | ImplementorOrigin::Assumption => {
                PredicateListId::empty_list(db)
            }
        }
    }

    /// Method map for this impl, keyed by name.
    pub(crate) fn methods(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> &'db IndexMap<IdentId<'db>, Func<'db>> {
        collect_implementor_methods(db, self)
    }

    /// Compare impl methods vs. trait methods and report missing/mismatched ones.
    pub(crate) fn diags_method_conformance(
        self,
        db: &'db dyn HirAnalysisDb,
        solve_cx: TraitSolveCx<'db>,
    ) -> Vec<TyDiagCollection<'db>> {
        if !matches!(self.origin(db), ImplementorOrigin::Hir(_)) {
            return Vec::new();
        }
        let mut diags = vec![];
        let impl_methods = self.methods(db);
        let hir_trait = self.trait_def(db);
        let trait_methods = self.trait_def(db).method_defs(db);
        let base_trait_inst = self.trait_(db);
        let mut method_cmp_assoc_type_bindings = base_trait_inst.assoc_type_bindings(db).clone();
        method_cmp_assoc_type_bindings.extend(self.types(db).iter().map(|(&name, &ty)| (name, ty)));
        let method_cmp_trait_inst = TraitInstId::new(
            db,
            base_trait_inst.def(db),
            base_trait_inst.args(db).to_vec(),
            method_cmp_assoc_type_bindings,
        );
        let mut required_methods: IndexSet<_> = trait_methods
            .iter()
            .filter_map(|(name, &trait_method)| trait_method.body(db).is_none().then_some(*name))
            .collect();

        for (name, impl_m) in impl_methods {
            let Some(trait_m) = trait_methods.get(name) else {
                diags.push(
                    ImplDiag::MethodNotDefinedInTrait {
                        primary: self.hir_impl_trait(db).span().trait_ref().into(),
                        method_name: *name,
                        trait_: hir_trait,
                    }
                    .into(),
                );
                continue;
            };
            compare_impl_method(
                db,
                impl_m.as_callable(db).unwrap(),
                trait_m.as_callable(db).unwrap(),
                method_cmp_trait_inst,
                &AnalysisCx::new(ProofCx::from_solve_cx(solve_cx))
                    .with_overlay(ImplOverlay::with_current_impl(self)),
                &mut diags,
            );
            required_methods.remove(name);
        }

        if !required_methods.is_empty() {
            diags.push(
                ImplDiag::NotAllTraitItemsImplemented {
                    primary: self.hir_impl_trait(db).span().ty().into(),
                    not_implemented: required_methods.into_iter().collect(),
                }
                .into(),
            );
        }

        diags
    }
}

/// Returns `true` if the given two implementors conflict.
///
/// This mirrors the legacy `Implementor`-based semantics:
/// - instantiate both implementors with fresh vars and unify them;
/// - then check that the merged constraints are satisfiable.
pub(crate) fn does_impl_trait_conflict<'db>(
    db: &'db dyn HirAnalysisDb,
    a: Binder<ImplementorId<'db>>,
    b: Binder<ImplementorId<'db>>,
) -> bool {
    let mut table = UnificationTable::new(db);
    let a = table.instantiate_with_fresh_vars(a);
    let b = table.instantiate_with_fresh_vars(b);

    if table.unify(a, b).is_err() {
        return false;
    }

    let a_constraints = a.constraints(db);
    let b_constraints = b.constraints(db);

    if a_constraints.is_empty(db) && b_constraints.is_empty(db) {
        return true;
    }

    // Check if all constraints from both implementations would be satisfiable
    // when the types are unified.
    let merged_constraints = a_constraints.merge(db, b_constraints);
    let solve_cx = TraitSolveCx::new(db, a.trait_def(db).scope())
        .with_assumptions(PredicateListId::empty_list(db));

    for &constraint in merged_constraints.list(db) {
        let constraint = Canonicalized::new(db, constraint.fold_with(db, &mut table));

        match is_goal_satisfiable(db, solve_cx, constraint.value) {
            GoalSatisfiability::UnSat(_) | GoalSatisfiability::ContainsInvalid => {
                return false;
            }
            _ => {
                // Constraint is satisfiable or needs more information, continue checking.
            }
        }
    }

    true
}

/// Represents an instantiated trait, which can be thought of as a trait
/// reference from a HIR perspective.
#[salsa::interned]
#[derive(Debug)]
pub struct TraitInstId<'db> {
    pub key: Trait<'db>,
    /// Regular type and const parameters: [Self, ExplicitTypeParam1, ..., ExplicitConstParamN]
    #[return_ref]
    pub args: Vec<TyId<'db>>,

    /// Associated type bounds specified by user, eg `Iterator<Item=i32>`
    #[return_ref]
    pub assoc_type_bindings: IndexMap<IdentId<'db>, TyId<'db>>,
}

impl<'db> TraitInstId<'db> {
    pub fn def(self, db: &'db dyn HirAnalysisDb) -> Trait<'db> {
        self.key(db)
    }

    pub fn new_simple(db: &'db dyn HirAnalysisDb, def: Trait<'db>, args: Vec<TyId<'db>>) -> Self {
        Self::new(db, def, args, IndexMap::new())
    }

    pub fn with_fresh_vars(
        db: &'db dyn HirAnalysisDb,
        def: Trait<'db>,
        table: &mut UnificationTable<'db>,
    ) -> Self {
        let args = def
            .params(db)
            .iter()
            .map(|ty| table.new_var_from_param(*ty))
            .collect::<Vec<_>>();
        Self::new(db, def, args, IndexMap::new())
    }

    pub fn assoc_ty_bindings(self, db: &'db dyn HirAnalysisDb) -> Vec<(IdentId<'db>, TyId<'db>)> {
        self.assoc_type_bindings(db)
            .iter()
            .map(|(&name, &ty)| (name, ty))
            .collect()
    }

    pub fn assoc_ty(self, db: &'db dyn HirAnalysisDb, name: IdentId<'db>) -> Option<TyId<'db>> {
        if let Some(ty) = self.assoc_type_bindings(db).get(&name) {
            return Some(*ty);
        }
        if self.def(db).assoc_ty(db, name).is_some() {
            return Some(TyId::assoc_ty(db, self, name));
        }
        None
    }

    /// Normalize arguments of this trait instance.
    pub(crate) fn normalize(
        self,
        db: &'db dyn HirAnalysisDb,
        scope: crate::core::hir_def::scope_graph::ScopeId<'db>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        self.normalize_with_solve_cx(
            db,
            TraitSolveCx::new(db, scope).with_assumptions(assumptions),
            scope,
            assumptions,
        )
    }

    pub(crate) fn normalize_with_solve_cx(
        self,
        db: &'db dyn HirAnalysisDb,
        solve_cx: TraitSolveCx<'db>,
        scope: crate::core::hir_def::scope_graph::ScopeId<'db>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        let normalized_args: Vec<_> = self
            .args(db)
            .iter()
            .map(|&arg| {
                crate::analysis::ty::normalize::normalize_ty_with_solve_cx(
                    db,
                    arg,
                    scope,
                    assumptions,
                    Some(solve_cx),
                )
            })
            .collect();
        Self::new(
            db,
            self.def(db),
            normalized_args,
            self.assoc_type_bindings(db).clone(),
        )
    }

    pub fn pretty_print(self, db: &dyn HirAnalysisDb, as_pred: bool) -> String {
        if as_pred {
            let inst = self.pretty_print(db, false);
            let self_ty = self.self_ty(db);
            format! {"{}: {}", self_ty.pretty_print(db), inst}
        } else {
            let mut s = self
                .def(db)
                .name(db)
                .to_opt()
                .map(|n| n.data(db).as_str())
                .unwrap_or("<unknown>")
                .to_string();

            let mut args = self.args(db).iter().map(|ty| ty.pretty_print(db));
            // Skip the first type parameter since it's the implementor type.
            args.next();

            let mut has_generics = false;
            if let Some(first) = args.next() {
                s.push('<');
                s.push_str(first);
                for arg in args {
                    s.push_str(", ");
                    s.push_str(arg);
                }
                has_generics = true;
            }

            // Add associated type bindings
            if !self.assoc_type_bindings(db).is_empty() {
                if !has_generics {
                    s.push('<');
                } else {
                    s.push_str(", ");
                }

                let mut first_assoc = true;
                for (name, ty) in self.assoc_type_bindings(db) {
                    if !first_assoc {
                        s.push_str(", ");
                    }
                    first_assoc = false;
                    s.push_str(name.data(db));
                    s.push_str(" = ");
                    s.push_str(ty.pretty_print(db));
                }
                has_generics = true;
            }

            if has_generics {
                s.push('>');
            }

            s
        }
    }

    pub fn self_ty(self, db: &'db dyn HirAnalysisDb) -> TyId<'db> {
        self.args(db)[0]
    }

    pub(crate) fn ingot(self, db: &'db dyn HirAnalysisDb) -> Ingot<'db> {
        self.def(db).ingot(db)
    }
}

// Represents a trait definition.
// (TraitDef struct and impl removed)

// (TraitMethod struct and impl removed)

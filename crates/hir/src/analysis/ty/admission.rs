use rustc_hash::{FxHashMap, FxHashSet};
use thin_vec::ThinVec;

use crate::analysis::{
    HirAnalysisDb,
    name_resolution::{ExpectedPathKind, diagnostics::PathResDiag},
    ty::{
        self,
        assoc_items::{analysis_cx_for_selected_assoc_const_body, resolve_assoc_const_selection},
        binder::Binder,
        canonical::Canonicalized,
        context::{AnalysisCx, ImplOverlay, LoweringMode, ProofCx},
        diagnostics::{
            BodyDiag, FuncBodyDiag, ImplDiag, TraitConstraintDiag, TraitLowerDiag,
            TyDiagCollection, TyLowerDiag,
        },
        fold::TyFoldable as _,
        trait_def::{ImplementorId, TraitInstId},
        trait_lower::TraitRefLowerError,
        trait_resolution::{
            GoalSatisfiability, LocalImplementorSet, WellFormedness, check_trait_inst_wf,
            constraint::ty_constraints, is_goal_satisfiable,
        },
        ty_def::{InvalidCause, TyFlags, TyId},
        unify::UnificationTable,
        visitor::TyVisitable,
    },
};
use crate::hir_def::{IdentId, ImplTrait};
use crate::span::DynLazySpan;
use common::ingot::IngotKind;

pub(crate) type TraitImplTable<'db> =
    FxHashMap<crate::hir_def::Trait<'db>, Vec<Binder<ImplementorId<'db>>>>;

#[derive(Debug, Clone, PartialEq, Eq, Default, salsa::Update)]
pub(crate) struct AdmissionSummary<'db> {
    pub admitted: TraitImplTable<'db>,
    pub header_issues: FxHashMap<ImplTrait<'db>, Vec<ImplHeaderIssue<'db>>>,
    pub interface_issues: FxHashMap<ImplTrait<'db>, Vec<ImplInterfaceIssue<'db>>>,
}

#[derive(Default)]
struct AdmissionCaches<'db> {
    header_issues: FxHashMap<ImplTrait<'db>, Vec<ImplHeaderIssue<'db>>>,
    interface_issues: FxHashMap<ImplTrait<'db>, Vec<ImplInterfaceIssue<'db>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Candidate<'db> {
    impl_trait: ImplTrait<'db>,
    implementor: Binder<ImplementorId<'db>>,
}

#[derive(Debug, Clone, Copy)]
struct ConflictCheckHeader<'db> {
    trait_inst: TraitInstId<'db>,
    constraints: ty::trait_resolution::PredicateListId<'db>,
}

impl<'db> TyVisitable<'db> for ConflictCheckHeader<'db> {
    fn visit_with<V>(&self, visitor: &mut V)
    where
        V: ty::visitor::TyVisitor<'db> + ?Sized,
    {
        self.trait_inst.visit_with(visitor);
        self.constraints.visit_with(visitor);
    }
}

impl<'db> ty::fold::TyFoldable<'db> for ConflictCheckHeader<'db> {
    fn super_fold_with<F>(self, db: &'db dyn HirAnalysisDb, folder: &mut F) -> Self
    where
        F: ty::fold::TyFolder<'db>,
    {
        Self {
            trait_inst: self.trait_inst.fold_with(db, folder),
            constraints: self.constraints.fold_with(db, folder),
        }
    }
}

pub(crate) struct AdmissionEngine<'db> {
    db: &'db dyn HirAnalysisDb,
    admitted: TraitImplTable<'db>,
    const_impl_maps: Vec<&'db TraitImplTable<'db>>,
    caches: AdmissionCaches<'db>,
}

impl<'db> AdmissionEngine<'db> {
    pub(crate) fn new(
        db: &'db dyn HirAnalysisDb,
        const_impl_maps: Vec<&'db TraitImplTable<'db>>,
    ) -> Self {
        Self {
            db,
            admitted: TraitImplTable::default(),
            const_impl_maps,
            caches: AdmissionCaches::default(),
        }
    }

    pub(crate) fn collect(mut self, impl_traits: &[ImplTrait<'db>]) -> TraitImplTable<'db> {
        let mut remaining = impl_traits
            .iter()
            .filter_map(|&impl_trait| {
                Some(Candidate {
                    impl_trait,
                    implementor: self.lowered_implementor(impl_trait)?,
                })
            })
            .collect::<Vec<_>>();
        self.admitted = TraitImplTable::default();

        loop {
            self.clear_round_caches();
            let mut round_admissible = Vec::new();
            for candidate in remaining.iter().copied() {
                if self.is_round_admissible(candidate) {
                    round_admissible.push(candidate);
                }
            }
            let round_survivors = self.filter_round_conflicts(&round_admissible);

            if round_survivors.is_empty() {
                break;
            }

            self.add_to_admitted(&round_survivors);
            self.remove_from_remaining(&mut remaining, &round_survivors);
        }

        self.admitted
    }

    pub(crate) fn summarize(mut self, impl_traits: &[ImplTrait<'db>]) -> AdmissionSummary<'db> {
        let candidates = impl_traits
            .iter()
            .filter_map(|&impl_trait| {
                Some(Candidate {
                    impl_trait,
                    implementor: self.lowered_implementor(impl_trait)?,
                })
            })
            .collect::<Vec<_>>();
        let mut remaining = candidates.clone();
        self.admitted = TraitImplTable::default();

        loop {
            self.clear_round_caches();
            let mut round_admissible = Vec::new();
            for candidate in remaining.iter().copied() {
                if self.is_round_admissible(candidate) {
                    round_admissible.push(candidate);
                }
            }
            let round_survivors = self.filter_round_conflicts(&round_admissible);

            if round_survivors.is_empty() {
                break;
            }

            self.add_to_admitted(&round_survivors);
            self.remove_from_remaining(&mut remaining, &round_survivors);
        }

        // Recompute every candidate's issues once the helper frontier is final so diagnostics
        // render the same semantic truth the admitted trait env uses.
        self.clear_round_caches();
        let header_issues = candidates
            .iter()
            .copied()
            .map(|candidate| {
                (
                    candidate.impl_trait,
                    self.header_issues_for_candidate(candidate).clone(),
                )
            })
            .collect();
        let interface_issues = candidates
            .iter()
            .copied()
            .map(|candidate| {
                (
                    candidate.impl_trait,
                    self.interface_issues_for_candidate(candidate).clone(),
                )
            })
            .collect();

        AdmissionSummary {
            admitted: self.admitted,
            header_issues,
            interface_issues,
        }
    }

    fn is_round_admissible(&mut self, candidate: Candidate<'db>) -> bool {
        if matches!(
            candidate
                .impl_trait
                .top_mod(self.db)
                .ingot(self.db)
                .kind(self.db),
            IngotKind::Core | IngotKind::Std
        ) {
            // `core`/`std` are trusted bootstrap ingots. Keeping them on the
            // permissive path avoids dropping foundational impls whose
            // signatures are only validated once the surrounding trait env is
            // available.
            return true;
        }

        self.header_issues_for_candidate(candidate).is_empty()
            && self.interface_issues_for_candidate(candidate).is_empty()
    }

    fn clear_round_caches(&mut self) {
        self.caches.header_issues.clear();
        self.caches.interface_issues.clear();
    }

    fn conflict_local_implementors(&self) -> LocalImplementorSet<'db> {
        LocalImplementorSet::new(
            self.db,
            self.const_impl_maps
                .iter()
                .chain(std::iter::once(&&self.admitted))
                .flat_map(|impl_map| impl_map.values())
                .flat_map(|implementors| implementors.iter().copied())
                .collect::<common::indexmap::IndexSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
        )
    }

    fn solve_cx(&self, impl_trait: ImplTrait<'db>) -> ty::trait_resolution::TraitSolveCx<'db> {
        ty::trait_resolution::TraitSolveCx::new(self.db, impl_trait.scope())
            .with_assumptions(impl_trait.elaborated_assumptions(self.db))
            .with_local_implementors(self.conflict_local_implementors())
            .without_ingot_search()
    }

    fn analysis_cx(&self, impl_trait: ImplTrait<'db>) -> AnalysisCx<'db> {
        AnalysisCx::new(ProofCx::from_solve_cx(self.solve_cx(impl_trait)))
    }

    fn interface_cx(
        &self,
        impl_trait: ImplTrait<'db>,
        current_impl: ImplementorId<'db>,
    ) -> AnalysisCx<'db> {
        AnalysisCx::new(ProofCx::from_solve_cx(self.solve_cx(impl_trait)))
            .with_overlay(ImplOverlay::with_current_impl(current_impl))
            .with_mode(LoweringMode::ImplTraitSignature {
                trait_inst: current_impl.trait_inst(self.db),
                self_ty: current_impl.self_ty(self.db),
                current_impl: Some(current_impl),
            })
    }

    fn header_issues_for_candidate(
        &mut self,
        candidate: Candidate<'db>,
    ) -> &Vec<ImplHeaderIssue<'db>> {
        let db = self.db;
        let cx = self.analysis_cx(candidate.impl_trait);
        self.caches
            .header_issues
            .entry(candidate.impl_trait)
            .or_insert_with(|| impl_header_issues(db, &cx, candidate.impl_trait))
    }

    fn interface_issues_for_candidate(
        &mut self,
        candidate: Candidate<'db>,
    ) -> &Vec<ImplInterfaceIssue<'db>> {
        let db = self.db;
        let current_impl = candidate.implementor.instantiate_identity();
        let cx = self.interface_cx(candidate.impl_trait, current_impl);
        self.caches
            .interface_issues
            .entry(candidate.impl_trait)
            .or_insert_with(|| {
                impl_interface_issues_with_assoc_type_bound_solve_cx(
                    db,
                    &cx,
                    candidate.impl_trait,
                    current_impl,
                    None,
                )
            })
    }

    fn lowered_implementor(
        &self,
        impl_trait: ImplTrait<'db>,
    ) -> Option<Binder<ImplementorId<'db>>> {
        let cx = self.analysis_cx(impl_trait);
        impl_trait
            .lowered_implementor_preconditions_in_cx(self.db, &cx)
            .ok()
    }

    fn add_to_admitted(&mut self, candidates: &[Candidate<'db>]) {
        for candidate in candidates {
            let current_impl = candidate.implementor.instantiate_identity();
            self.admitted
                .entry(current_impl.trait_def(self.db))
                .or_default()
                .push(candidate.implementor);
        }
    }

    fn remove_from_remaining(
        &self,
        remaining: &mut Vec<Candidate<'db>>,
        admitted: &[Candidate<'db>],
    ) {
        let admitted_impls = admitted
            .iter()
            .map(|candidate| candidate.impl_trait)
            .collect::<FxHashSet<_>>();
        remaining.retain(|candidate| !admitted_impls.contains(&candidate.impl_trait));
    }

    fn filter_round_conflicts(&self, round_admissible: &[Candidate<'db>]) -> Vec<Candidate<'db>> {
        round_admissible
            .iter()
            .copied()
            .filter(|candidate| {
                !self.does_conflict_with_admitted_or_external(candidate.implementor)
                    && !self.does_conflict_with_round(*candidate, round_admissible)
            })
            .collect()
    }

    fn does_conflict_with_admitted_or_external(
        &self,
        implementor: Binder<ImplementorId<'db>>,
    ) -> bool {
        self.const_impl_maps
            .iter()
            .chain(std::iter::once(&&self.admitted))
            .any(|impl_map| self.does_conflict_with_impl_map(implementor, impl_map))
    }

    fn does_conflict_with_round(
        &self,
        candidate: Candidate<'db>,
        round_admissible: &[Candidate<'db>],
    ) -> bool {
        let current_impl = candidate.implementor.instantiate_identity();
        round_admissible.iter().any(|other| {
            other.impl_trait != candidate.impl_trait
                && other.implementor.instantiate_identity().trait_def(self.db)
                    == current_impl.trait_def(self.db)
                && self.do_implementors_conflict(other.implementor, candidate.implementor)
        })
    }

    fn does_conflict_with_impl_map(
        &self,
        implementor: Binder<ImplementorId<'db>>,
        impl_map: &TraitImplTable<'db>,
    ) -> bool {
        let current_impl = implementor.instantiate_identity();
        let Some(implementors) = impl_map.get(&current_impl.trait_def(self.db)) else {
            return false;
        };

        implementors.iter().copied().any(|admitted| {
            admitted.instantiate_identity().origin(self.db) != current_impl.origin(self.db)
                && self.do_implementors_conflict(admitted, implementor)
        })
    }

    fn do_implementors_conflict(
        &self,
        a: Binder<ImplementorId<'db>>,
        b: Binder<ImplementorId<'db>>,
    ) -> bool {
        let mut table = UnificationTable::new(self.db);
        let a = table.instantiate_with_fresh_vars(Binder::bind(ConflictCheckHeader {
            trait_inst: a.instantiate_identity().trait_(self.db),
            constraints: a.instantiate_identity().constraints(self.db),
        }));
        let b = table.instantiate_with_fresh_vars(Binder::bind(ConflictCheckHeader {
            trait_inst: b.instantiate_identity().trait_(self.db),
            constraints: b.instantiate_identity().constraints(self.db),
        }));

        if table.unify(a.trait_inst, b.trait_inst).is_err() {
            return false;
        }

        if a.constraints.is_empty(self.db) && b.constraints.is_empty(self.db) {
            return true;
        }

        let merged_constraints = a.constraints.merge(self.db, b.constraints);
        let solve_cx =
            ty::trait_resolution::TraitSolveCx::new(self.db, a.trait_inst.def(self.db).scope())
                .with_assumptions(ty::trait_resolution::PredicateListId::empty_list(self.db))
                .with_local_implementors(self.conflict_local_implementors())
                .without_ingot_search();

        for &constraint in merged_constraints.list(self.db) {
            let constraint = Canonicalized::new(self.db, constraint.fold_with(self.db, &mut table));
            match is_goal_satisfiable(self.db, solve_cx, constraint.value) {
                GoalSatisfiability::UnSat(_) | GoalSatisfiability::ContainsInvalid => {
                    return false;
                }
                GoalSatisfiability::Satisfied(_) | GoalSatisfiability::NeedsConfirmation(_) => {}
            }
        }

        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, salsa::Update)]
pub(crate) enum ImplHeaderIssue<'db> {
    InvalidTraitRef(TraitRefLowerError<'db>),
    ImplementorIllFormed {
        goal: TraitInstId<'db>,
        subgoal: Option<TraitInstId<'db>>,
    },
    TraitInstIllFormed {
        goal: TraitInstId<'db>,
        subgoal: Option<TraitInstId<'db>>,
    },
    SelfKindMismatch {
        expected: ty::ty_def::Kind,
        actual: TyId<'db>,
    },
    SupertraitUnmet {
        goal: TraitInstId<'db>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, salsa::Update)]
pub enum ImplInterfaceIssue<'db> {
    ExtraMethod {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        method_name: IdentId<'db>,
    },
    MissingMethod {
        primary: DynLazySpan<'db>,
        not_implemented: ThinVec<IdentId<'db>>,
    },
    MethodSignatureMismatch(ImplDiag<'db>),
    ExtraAssocType {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        type_name: IdentId<'db>,
    },
    MissingAssocType {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        type_name: IdentId<'db>,
    },
    AssocTypeInvalid(TyDiagCollection<'db>),
    AssocTypeBoundViolation {
        span: DynLazySpan<'db>,
        primary_goal: TraitInstId<'db>,
    },
    ExtraAssocConst {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        const_name: IdentId<'db>,
    },
    MissingAssocConst {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        const_name: IdentId<'db>,
    },
    MissingAssocConstValue {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        const_name: IdentId<'db>,
    },
    AssocConstInvalidDiag(TyDiagCollection<'db>),
    AssocConstInvalid {
        primary: DynLazySpan<'db>,
        trait_: crate::hir_def::Trait<'db>,
        const_name: IdentId<'db>,
        body_diags: Vec<FuncBodyDiag<'db>>,
    },
}

fn const_ty_mismatch_diag<'db>(
    span: DynLazySpan<'db>,
    expected: TyId<'db>,
    given: TyId<'db>,
) -> TyDiagCollection<'db> {
    TyLowerDiag::ConstTyMismatch {
        span,
        expected,
        given,
    }
    .into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssocConstDependencyState {
    Valid,
    Abstract,
    Invalid,
    Cyclic,
}

type AssocConstDependencyNode<'db> = (TraitInstId<'db>, IdentId<'db>);
type AssocConstDeps<'db> = Vec<(TraitInstId<'db>, IdentId<'db>)>;
type AssocConstBodyAnalysis<'db> = (Vec<FuncBodyDiag<'db>>, AssocConstDeps<'db>);

struct AssocConstDependencyCx<'db> {
    cx: AnalysisCx<'db>,
    visiting: FxHashSet<AssocConstDependencyNode<'db>>,
    memo: FxHashMap<AssocConstDependencyNode<'db>, AssocConstDependencyState>,
}

impl<'db> AssocConstDependencyCx<'db> {
    fn new(cx: AnalysisCx<'db>) -> Self {
        Self {
            cx,
            visiting: FxHashSet::default(),
            memo: FxHashMap::default(),
        }
    }

    fn state(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        node: AssocConstDependencyNode<'db>,
    ) -> AssocConstDependencyState {
        if let Some(&state) = self.memo.get(&node) {
            return state;
        }
        if !self.visiting.insert(node) {
            return AssocConstDependencyState::Cyclic;
        }

        let state =
            if let Some(selection) = resolve_assoc_const_selection(db, &self.cx, node.0, node.1) {
                if let Some((body_diags, deps)) =
                    assoc_const_body_diags_and_deps(db, &selection, self.cx)
                {
                    if !body_diags.is_empty() {
                        AssocConstDependencyState::Invalid
                    } else {
                        let mut state = AssocConstDependencyState::Valid;
                        for dep in deps {
                            match self.state(db, dep) {
                                AssocConstDependencyState::Cyclic => {
                                    state = AssocConstDependencyState::Cyclic;
                                    break;
                                }
                                AssocConstDependencyState::Invalid => {
                                    state = AssocConstDependencyState::Invalid
                                }
                                AssocConstDependencyState::Abstract
                                | AssocConstDependencyState::Valid => {}
                            }
                        }
                        state
                    }
                } else {
                    abstract_assoc_const_dependency_state(db, selection.trait_inst)
                }
            } else {
                abstract_assoc_const_dependency_state(db, node.0)
            };

        self.visiting.remove(&node);
        self.memo.insert(node, state);
        state
    }
}

fn abstract_assoc_const_dependency_state<'db>(
    db: &'db dyn HirAnalysisDb,
    trait_inst: TraitInstId<'db>,
) -> AssocConstDependencyState {
    let flags = ty::visitor::collect_flags(db, trait_inst);
    if flags.intersects(TyFlags::HAS_PARAM | TyFlags::HAS_VAR)
        && !flags.contains(TyFlags::HAS_INVALID)
    {
        AssocConstDependencyState::Abstract
    } else {
        AssocConstDependencyState::Invalid
    }
}

fn check_selected_assoc_const_body<'db>(
    db: &'db dyn HirAnalysisDb,
    proof: ProofCx<'db>,
    selection: &ty::assoc_items::AssocConstSelection<'db>,
) -> (
    Vec<FuncBodyDiag<'db>>,
    crate::analysis::ty::ty_check::TypedBody<'db>,
) {
    let analysis_cx = analysis_cx_for_selected_assoc_const_body(db, proof, selection)
        .expect("selected assoc const body analysis requires a selected body");
    let body = selection
        .body
        .as_ref()
        .expect("selected assoc const body analysis requires a selected body")
        .body;
    let result = ty::ty_check::check_anon_const_body_in_mode(
        db,
        body,
        selection.declared_ty,
        analysis_cx.proof.solve_cx(),
        analysis_cx.mode,
    );
    (result.0.clone(), result.1.clone())
}

fn assoc_const_body_diags_and_deps<'db>(
    db: &'db dyn HirAnalysisDb,
    selection: &ty::assoc_items::AssocConstSelection<'db>,
    cx: AnalysisCx<'db>,
) -> Option<AssocConstBodyAnalysis<'db>> {
    let source = selection.body.as_ref()?;
    let (body_diags, typed_body) = check_selected_assoc_const_body(db, cx.proof, selection);
    let typed_body = ty::ctfe::instantiate_typed_body_for_trait_inst(
        db,
        typed_body,
        selection.trait_inst,
        &source.impl_args,
    );

    let mut deps = Vec::new();
    for (expr, _) in source.body.exprs(db).iter() {
        if let Some(ty::ty_check::ConstRef::TraitConst { inst, name }) =
            typed_body.expr_const_ref(expr)
        {
            deps.push((inst, name));
        }
    }
    Some((body_diags, deps))
}

pub(crate) fn check_impl_ty_wf_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: ty::trait_resolution::TraitSolveCx<'db>,
    ty: TyId<'db>,
) -> WellFormedness<'db> {
    let (_, args) = ty.decompose_ty_app(db);
    for &arg in args {
        let wf = check_impl_ty_wf_in_cx(db, solve_cx, arg);
        if !matches!(wf, WellFormedness::WellFormed) {
            return wf;
        }
    }

    let assumptions = solve_cx.assumptions();
    let normalized_constraints = {
        let scope = solve_cx.origin_scope(db);
        let normalized: Vec<_> = ty_constraints(db, ty)
            .list(db)
            .iter()
            .map(|&goal| goal.normalize(db, scope, assumptions))
            .collect();
        ty::trait_resolution::PredicateListId::new(db, normalized)
    };

    for &goal in normalized_constraints.list(db) {
        let mut table = UnificationTable::new(db);
        let canonical_goal = Canonicalized::new(db, goal);
        match is_goal_satisfiable(db, solve_cx, canonical_goal.value) {
            GoalSatisfiability::Satisfied(_)
            | GoalSatisfiability::NeedsConfirmation(_)
            | GoalSatisfiability::ContainsInvalid => {}
            GoalSatisfiability::UnSat(subgoal) => {
                let subgoal =
                    subgoal.map(|subgoal| canonical_goal.extract_solution(&mut table, subgoal));
                return WellFormedness::IllFormed { goal, subgoal };
            }
        }
    }

    WellFormedness::WellFormed
}

pub(crate) fn check_impl_trait_implementor_wf_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_trait: ImplTrait<'db>,
    cx: &AnalysisCx<'db>,
) -> WellFormedness<'db> {
    check_impl_ty_wf_in_cx(db, cx.proof.solve_cx(), impl_trait.ty_in_cx(db, cx))
}

pub(crate) fn impl_header_issues<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    impl_trait: ImplTrait<'db>,
) -> Vec<ImplHeaderIssue<'db>> {
    let mut issues = Vec::new();
    let self_ty = impl_trait.ty_in_cx(db, cx);
    match check_impl_trait_implementor_wf_in_cx(db, impl_trait, cx) {
        WellFormedness::WellFormed => {}
        WellFormedness::IllFormed { goal, subgoal } => {
            issues.push(ImplHeaderIssue::ImplementorIllFormed { goal, subgoal });
        }
    }

    let trait_inst = match impl_trait.trait_inst_result_in_cx(db, cx) {
        Ok(trait_inst) => trait_inst,
        Err(err) => {
            issues.push(ImplHeaderIssue::InvalidTraitRef(err));
            return issues;
        }
    };

    let expected_kind = trait_inst.def(db).self_param(db).kind(db);
    if self_ty.kind(db) != expected_kind {
        issues.push(ImplHeaderIssue::SelfKindMismatch {
            expected: expected_kind.clone(),
            actual: self_ty,
        });
    }

    if let WellFormedness::IllFormed { goal, subgoal } =
        check_trait_inst_wf(db, cx.proof.solve_cx(), trait_inst)
    {
        issues.push(ImplHeaderIssue::TraitInstIllFormed { goal, subgoal });
    }

    for super_trait in trait_inst.def(db).super_traits(db) {
        let goal = super_trait.instantiate(db, trait_inst.args(db));
        if matches!(
            is_goal_satisfiable(
                db,
                cx.proof.solve_cx(),
                ty::canonical::Canonical::new(db, goal)
            ),
            GoalSatisfiability::UnSat(_)
        ) {
            issues.push(ImplHeaderIssue::SupertraitUnmet { goal });
        }
    }

    issues
}

fn method_conformance_issues<'db>(
    db: &'db dyn HirAnalysisDb,
    implementor: ImplementorId<'db>,
    solve_cx: ty::trait_resolution::TraitSolveCx<'db>,
) -> Vec<ImplInterfaceIssue<'db>> {
    let mut issues = Vec::new();
    for diag in implementor.diags_method_conformance(db, solve_cx) {
        let TyDiagCollection::Impl(diag) = diag else {
            continue;
        };
        match diag {
            ImplDiag::MethodNotDefinedInTrait {
                primary,
                trait_,
                method_name,
            } => issues.push(ImplInterfaceIssue::ExtraMethod {
                primary,
                trait_,
                method_name,
            }),
            ImplDiag::NotAllTraitItemsImplemented {
                primary,
                not_implemented,
            } => issues.push(ImplInterfaceIssue::MissingMethod {
                primary,
                not_implemented,
            }),
            diag => issues.push(ImplInterfaceIssue::MethodSignatureMismatch(diag)),
        }
    }
    issues
}

pub fn impl_interface_issues<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    impl_trait: ImplTrait<'db>,
    current_impl: ImplementorId<'db>,
) -> Vec<ImplInterfaceIssue<'db>> {
    impl_interface_issues_with_assoc_type_bound_solve_cx(db, cx, impl_trait, current_impl, None)
}

fn impl_interface_issues_with_assoc_type_bound_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    impl_trait: ImplTrait<'db>,
    current_impl: ImplementorId<'db>,
    assoc_type_bound_solve_cx: Option<ty::trait_resolution::TraitSolveCx<'db>>,
) -> Vec<ImplInterfaceIssue<'db>> {
    let cx = cx
        .with_overlay(ty::context::ImplOverlay::with_current_impl(current_impl))
        .with_mode(ty::context::LoweringMode::ImplTraitSignature {
            trait_inst: current_impl.trait_inst(db),
            self_ty: current_impl.self_ty(db),
            current_impl: Some(current_impl),
        });
    let mut issues = method_conformance_issues(db, current_impl, cx.proof.solve_cx());
    issues.extend(
        impl_trait
            .assoc_types(db)
            .flat_map(|assoc| assoc.ty_diags_in_cx(db, &cx))
            .map(ImplInterfaceIssue::AssocTypeInvalid),
    );

    let trait_hir = current_impl.trait_def(db);
    let impl_types = current_impl.types(db);
    for assoc in trait_hir.assoc_types(db) {
        let Some(name) = assoc.name(db) else { continue };
        let has_impl = impl_types.get(&name).is_some();
        let has_default = assoc.default_ty(db).is_some();
        if !has_impl && !has_default {
            issues.push(ImplInterfaceIssue::MissingAssocType {
                primary: impl_trait.span().ty().into(),
                trait_: trait_hir,
                type_name: name,
            });
        }
    }

    for assoc in impl_trait.assoc_types(db) {
        let Some(name) = assoc.name(db) else { continue };
        if trait_hir.assoc_ty(db, name).is_none() {
            issues.push(ImplInterfaceIssue::ExtraAssocType {
                primary: assoc.span().name().into(),
                trait_: trait_hir,
                type_name: name,
            });
        }
    }

    {
        use ty::fold::TyFoldable as _;

        struct TraitScopeSubstFolder<'db, 'a> {
            trait_scope: crate::hir_def::scope_graph::ScopeId<'db>,
            trait_args: &'a [TyId<'db>],
        }

        impl<'db> ty::fold::TyFolder<'db> for TraitScopeSubstFolder<'db, '_> {
            fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
                match ty.data(db) {
                    ty::ty_def::TyData::TyParam(param)
                        if !param.is_effect() && param.owner == self.trait_scope =>
                    {
                        self.trait_args.get(param.idx).copied().unwrap_or(ty)
                    }
                    ty::ty_def::TyData::ConstTy(const_ty) => match const_ty.data(db) {
                        ty::const_ty::ConstTyData::TyParam(param, _)
                            if !param.is_effect() && param.owner == self.trait_scope =>
                        {
                            self.trait_args.get(param.idx).copied().unwrap_or(ty)
                        }
                        _ => ty.super_fold_with(db, self),
                    },
                    _ => ty.super_fold_with(db, self),
                }
            }
        }

        let trait_args = current_impl.trait_(db).args(db);
        let trait_scope = current_impl.trait_def(db).scope();
        let assoc_type_bound_solve_cx = assoc_type_bound_solve_cx.unwrap_or(cx.proof.solve_cx());
        for assoc in current_impl.assoc_type_views(db) {
            let Some(name) = assoc.name(db) else { continue };
            for bound_inst in assoc.bounds(db) {
                let mut folder = TraitScopeSubstFolder {
                    trait_scope,
                    trait_args,
                };
                let bound_inst = bound_inst.fold_with(db, &mut folder);
                let canonical_bound = ty::canonical::Canonical::new(db, bound_inst);
                if matches!(
                    is_goal_satisfiable(db, assoc_type_bound_solve_cx, canonical_bound),
                    GoalSatisfiability::UnSat(_)
                ) {
                    let assoc_ty_span = impl_trait
                        .associated_type_span(db, name)
                        .map(|s| s.ty().into())
                        .unwrap_or_else(|| impl_trait.span().ty().into());
                    issues.push(ImplInterfaceIssue::AssocTypeBoundViolation {
                        span: assoc_ty_span,
                        primary_goal: bound_inst,
                    });
                }
            }
        }
    }

    for trait_const in trait_hir.assoc_consts(db) {
        let Some(name) = trait_const.name(db) else {
            continue;
        };
        let has_impl = impl_trait.const_(db, name).is_some();
        let has_default = trait_const.has_default(db);
        if !has_impl && !has_default {
            issues.push(ImplInterfaceIssue::MissingAssocConst {
                primary: impl_trait.span().ty().into(),
                trait_: trait_hir,
                const_name: name,
            });
        }
    }

    let mut dep_cx = AssocConstDependencyCx::new(cx);
    for impl_const in impl_trait.assoc_consts(db) {
        let Some(name) = impl_const.name(db) else {
            continue;
        };
        if trait_hir.const_(db, name).is_none() {
            issues.push(ImplInterfaceIssue::ExtraAssocConst {
                primary: impl_const.span().name().into(),
                trait_: trait_hir,
                const_name: name,
            });
        }
    }

    for trait_const in trait_hir.assoc_consts(db) {
        let Some(name) = trait_const.name(db) else {
            continue;
        };
        let impl_const = impl_trait.const_(db, name);

        if let Some(impl_const) = impl_const
            && !impl_const.has_value(db)
        {
            issues.push(ImplInterfaceIssue::MissingAssocConstValue {
                primary: impl_const.span().ty().into(),
                trait_: trait_hir,
                const_name: name,
            });
            continue;
        }

        let Some(expected_ty) =
            ty::assoc_items::assoc_const_declared_ty(db, &cx, current_impl.trait_inst(db), name)
        else {
            continue;
        };

        let mut header_valid = true;
        if let Some(impl_const) = impl_const {
            for diag in impl_const.ty_diags_in_cx(db, &cx) {
                issues.push(ImplInterfaceIssue::AssocConstInvalidDiag(diag));
                header_valid = false;
            }

            if header_valid {
                let Some(actual_ty) = impl_const.ty_in_cx(db, &cx) else {
                    continue;
                };
                let actual_ty = ty::trait_def::normalize_ty_for_trait_inst(
                    db,
                    cx.proof.solve_cx(),
                    actual_ty,
                    current_impl.trait_inst(db),
                );
                if actual_ty != expected_ty {
                    issues.push(ImplInterfaceIssue::AssocConstInvalidDiag(
                        const_ty_mismatch_diag(
                            impl_const.span().ty().into(),
                            expected_ty,
                            actual_ty,
                        ),
                    ));
                    header_valid = false;
                }
            }
        }

        if !header_valid {
            continue;
        }

        let Some(selection) =
            resolve_assoc_const_selection(db, &cx, current_impl.trait_inst(db), name)
        else {
            continue;
        };
        let Some(source) = selection.body.as_ref() else {
            continue;
        };
        let primary: DynLazySpan<'db> = impl_const
            .map(|const_| const_.span().ty().into())
            .unwrap_or_else(|| trait_const.span().ty().into());
        let Some((body_diags, deps)) = assoc_const_body_diags_and_deps(db, &selection, cx) else {
            continue;
        };
        if !body_diags.is_empty() {
            issues.push(ImplInterfaceIssue::AssocConstInvalid {
                primary: primary.clone(),
                trait_: trait_hir,
                const_name: name,
                body_diags,
            });
            continue;
        }

        let root = (selection.trait_inst, name);
        dep_cx.visiting.insert(root);
        let mut dep_state = AssocConstDependencyState::Valid;
        for dep in deps {
            match dep_cx.state(db, dep) {
                AssocConstDependencyState::Cyclic => {
                    dep_state = AssocConstDependencyState::Cyclic;
                    break;
                }
                AssocConstDependencyState::Invalid => {
                    dep_state = AssocConstDependencyState::Invalid
                }
                AssocConstDependencyState::Abstract | AssocConstDependencyState::Valid => {}
            }
        }
        dep_cx.visiting.remove(&root);

        match dep_state {
            AssocConstDependencyState::Abstract | AssocConstDependencyState::Valid => {}
            AssocConstDependencyState::Invalid => {
                issues.push(ImplInterfaceIssue::AssocConstInvalid {
                    primary,
                    trait_: trait_hir,
                    const_name: name,
                    body_diags: Vec::new(),
                })
            }
            AssocConstDependencyState::Cyclic => {
                issues.push(ImplInterfaceIssue::AssocConstInvalidDiag(
                    TyId::invalid(
                        db,
                        InvalidCause::ConstEvalRecursionLimitExceeded {
                            body: source.body,
                            expr: source.body.expr(db),
                        },
                    )
                    .emit_diag(db, primary)
                    .expect("const recursion should emit a diagnostic"),
                ))
            }
        }
    }

    issues
}

impl<'db> ImplInterfaceIssue<'db> {
    pub fn to_diags(&self) -> Vec<TyDiagCollection<'db>> {
        match self {
            Self::ExtraMethod {
                primary,
                trait_,
                method_name,
            } => vec![
                ImplDiag::MethodNotDefinedInTrait {
                    primary: primary.clone(),
                    trait_: *trait_,
                    method_name: *method_name,
                }
                .into(),
            ],
            Self::MissingMethod {
                primary,
                not_implemented,
            } => vec![
                ImplDiag::NotAllTraitItemsImplemented {
                    primary: primary.clone(),
                    not_implemented: not_implemented.clone(),
                }
                .into(),
            ],
            Self::MethodSignatureMismatch(diag) => vec![diag.clone().into()],
            Self::ExtraAssocType {
                primary,
                trait_,
                type_name,
            } => vec![
                ImplDiag::AssocTypeNotDefinedInTrait {
                    primary: primary.clone(),
                    trait_: *trait_,
                    type_name: *type_name,
                }
                .into(),
            ],
            Self::MissingAssocType {
                primary,
                trait_,
                type_name,
            } => vec![
                ImplDiag::MissingAssociatedType {
                    primary: primary.clone(),
                    trait_: *trait_,
                    type_name: *type_name,
                }
                .into(),
            ],
            Self::AssocTypeInvalid(diag) => vec![diag.clone()],
            Self::AssocTypeBoundViolation { span, primary_goal } => vec![
                TraitConstraintDiag::TraitBoundNotSat {
                    span: span.clone(),
                    primary_goal: *primary_goal,
                    unsat_subgoal: None,
                }
                .into(),
            ],
            Self::ExtraAssocConst {
                primary,
                trait_,
                const_name,
            } => vec![
                ImplDiag::ConstNotDefinedInTrait {
                    primary: primary.clone(),
                    trait_: *trait_,
                    const_name: *const_name,
                }
                .into(),
            ],
            Self::MissingAssocConst {
                primary,
                trait_,
                const_name,
            } => vec![
                ImplDiag::MissingAssociatedConst {
                    primary: primary.clone(),
                    trait_: *trait_,
                    const_name: *const_name,
                }
                .into(),
            ],
            Self::MissingAssocConstValue {
                primary,
                trait_,
                const_name,
            } => vec![
                ImplDiag::MissingAssociatedConstValue {
                    primary: primary.clone(),
                    trait_: *trait_,
                    const_name: *const_name,
                }
                .into(),
            ],
            Self::AssocConstInvalidDiag(diag) => vec![diag.clone()],
            Self::AssocConstInvalid {
                primary,
                trait_,
                const_name,
                body_diags,
            } => {
                let mut diags = Vec::new();
                let mut needs_fallback = false;
                for diag in body_diags {
                    match diag {
                        FuncBodyDiag::Ty(diag) => diags.push(diag.clone()),
                        FuncBodyDiag::NameRes(diag) => diags.push(diag.clone().into()),
                        FuncBodyDiag::Body(BodyDiag::TypeMismatch {
                            span,
                            expected,
                            given,
                        }) => diags.push(const_ty_mismatch_diag(span.clone(), *expected, *given)),
                        FuncBodyDiag::Body(_) => needs_fallback = true,
                    }
                }
                if diags.is_empty() || needs_fallback {
                    diags.push(
                        ImplDiag::InvalidAssociatedConst {
                            primary: primary.clone(),
                            trait_: *trait_,
                            const_name: *const_name,
                        }
                        .into(),
                    );
                }
                diags
            }
        }
    }
}

impl<'db> ImplHeaderIssue<'db> {
    pub fn to_diags(
        &self,
        db: &'db dyn HirAnalysisDb,
        impl_trait: ImplTrait<'db>,
    ) -> Vec<TyDiagCollection<'db>> {
        match self {
            Self::InvalidTraitRef(err) => match err {
                TraitRefLowerError::PathResError(err) => impl_trait
                    .hir_trait_ref(db)
                    .to_opt()
                    .and_then(|trait_ref| {
                        err.clone().into_diag(
                            db,
                            trait_ref.path(db).unwrap(),
                            impl_trait.span().trait_ref().path(),
                            ExpectedPathKind::Trait,
                        )
                    })
                    .map(Into::into)
                    .into_iter()
                    .collect(),
                TraitRefLowerError::InvalidDomain(res) => impl_trait
                    .hir_trait_ref(db)
                    .to_opt()
                    .map(|trait_ref| {
                        PathResDiag::ExpectedTrait(
                            impl_trait.span().trait_ref().path().into(),
                            trait_ref.path(db).unwrap().ident(db).unwrap(),
                            res.kind_name(),
                        )
                        .into()
                    })
                    .into_iter()
                    .collect(),
                TraitRefLowerError::Ignored => {
                    vec![TraitLowerDiag::ExternalTraitForExternalType(impl_trait).into()]
                }
            },
            Self::ImplementorIllFormed { goal, subgoal } => vec![
                TraitConstraintDiag::TraitBoundNotSat {
                    span: impl_trait.span().ty().into(),
                    primary_goal: *goal,
                    unsat_subgoal: *subgoal,
                }
                .into(),
            ],
            Self::TraitInstIllFormed { goal, subgoal } => vec![
                TraitConstraintDiag::TraitBoundNotSat {
                    span: impl_trait.span().trait_ref().into(),
                    primary_goal: *goal,
                    unsat_subgoal: *subgoal,
                }
                .into(),
            ],
            Self::SelfKindMismatch { expected, actual } => vec![
                TraitConstraintDiag::TraitArgKindMismatch {
                    span: impl_trait.span().trait_ref(),
                    expected: expected.clone(),
                    actual: *actual,
                }
                .into(),
            ],
            Self::SupertraitUnmet { goal } => vec![
                TraitConstraintDiag::TraitBoundNotSat {
                    span: impl_trait.span().ty().into(),
                    primary_goal: *goal,
                    unsat_subgoal: None,
                }
                .into(),
            ],
        }
    }
}

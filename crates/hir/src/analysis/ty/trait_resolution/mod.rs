use super::{
    binder::Binder,
    canonical::{Canonical, Canonicalized, Solution},
    const_expr::ConstExpr,
    const_ty::{ConstTyData, StoredAnalysisCx, const_body_simple_path},
    context::AnalysisCx,
    fold::{AssocTySubst, TyFoldable},
    normalize::normalize_ty_without_consts_with_solve_cx,
    trait_def::{ImplementorId, TraitInstId},
    ty_def::{InvalidCause, TyData, TyFlags, TyId, TyParam},
    ty_lower::contextual_path_resolution_in_cx,
};
use crate::analysis::{
    HirAnalysisDb,
    name_resolution::{PathRes, resolve_path, resolve_path_in_cx},
    ty::{
        trait_resolution::{constraint::ty_constraints, proof_forest::ProofForest},
        unify::UnificationTable,
        visitor::{TyVisitable, TyVisitor, collect_flags, walk_const_ty, walk_ty},
    },
};
use crate::{
    Ingot,
    hir_def::{HirIngot, scope_graph::ScopeId},
};
use common::indexmap::{IndexMap, IndexSet};
use constraint::collect_constraints;
use rustc_hash::FxHashSet;
use salsa::Update;

pub(crate) mod constraint;
mod proof_forest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct TraitSolverQuery<'db> {
    pub goal: TraitInstId<'db>,
    pub assumptions: PredicateListId<'db>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalGoalQuery<'db> {
    raw: TraitSolverQuery<'db>,
    canonical: Canonical<TraitSolverQuery<'db>>,
    original: Canonicalized<'db, TraitSolverQuery<'db>>,
}

impl<'db> CanonicalGoalQuery<'db> {
    pub fn new(
        db: &'db dyn HirAnalysisDb,
        goal: TraitInstId<'db>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        Self::from_query(
            db,
            TraitSolverQuery {
                goal,
                assumptions: assumptions.extend_all_bounds(db),
            },
        )
    }

    pub fn from_query(db: &'db dyn HirAnalysisDb, raw: TraitSolverQuery<'db>) -> Self {
        let original = Canonicalized::new(db, raw);
        Self {
            raw,
            canonical: original.value,
            original,
        }
    }

    pub fn goal(&self) -> TraitInstId<'db> {
        self.raw.goal
    }

    pub fn assumptions(&self) -> PredicateListId<'db> {
        self.raw.assumptions
    }

    pub fn canonical(&self) -> Canonical<TraitSolverQuery<'db>> {
        self.canonical
    }

    pub fn extract_solution<S, U>(
        &self,
        table: &mut crate::analysis::ty::unify::UnificationTableBase<'db, S>,
        solution: Solution<U>,
    ) -> U
    where
        S: crate::analysis::ty::unify::UnificationStore<'db>,
        U: TyFoldable<'db> + Update,
    {
        self.original.extract_solution(table, solution)
    }

    pub fn extract_subgoal<S>(
        &self,
        table: &mut crate::analysis::ty::unify::UnificationTableBase<'db, S>,
        solution: Solution<TraitInstId<'db>>,
    ) -> TraitInstId<'db>
    where
        S: crate::analysis::ty::unify::UnificationStore<'db>,
    {
        self.extract_solution(table, solution)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Selection<T> {
    Unique(T),
    Ambiguous(IndexSet<T>),
    NotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct TraitSolveCx<'db> {
    origin_ingot: Ingot<'db>,
    assumptions: PredicateListId<'db>,
    local_implementors: Option<LocalImplementorSet<'db>>,
}

#[salsa::interned]
#[derive(Debug)]
pub(crate) struct LocalImplementorSet<'db> {
    #[return_ref]
    pub(crate) implementors: Vec<Binder<ImplementorId<'db>>>,
}

impl<'db> TraitSolveCx<'db> {
    pub fn new(db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> Self {
        Self {
            origin_ingot: scope.ingot(db),
            assumptions: PredicateListId::empty_list(db),
            local_implementors: None,
        }
    }

    pub fn with_assumptions(self, assumptions: PredicateListId<'db>) -> Self {
        Self {
            assumptions,
            ..self
        }
    }

    pub(crate) fn with_local_implementors(
        self,
        local_implementors: LocalImplementorSet<'db>,
    ) -> Self {
        Self {
            local_implementors: Some(local_implementors),
            ..self
        }
    }

    pub fn assumptions(self) -> PredicateListId<'db> {
        self.assumptions
    }

    pub(crate) fn local_implementors(self) -> Option<LocalImplementorSet<'db>> {
        self.local_implementors
    }

    pub(crate) fn origin_ingot(self) -> Ingot<'db> {
        self.origin_ingot
    }

    pub(crate) fn select_impl(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> Selection<ImplementorId<'db>> {
        let inst = normalize_trait_inst_preserving_validity_with_solve_cx(db, inst, self);
        let query = CanonicalGoalQuery::new(db, inst, self.assumptions());
        match is_goal_query_satisfiable(db, self, &query) {
            GoalSatisfiability::Satisfied(solution) => {
                let mut table = UnificationTable::new(db);
                let solution = query.extract_solution(&mut table, solution);
                Selection::Unique(solution.implementor)
            }
            GoalSatisfiability::NeedsConfirmation(ambiguous) => Selection::Ambiguous(
                ambiguous
                    .iter()
                    .map(|&solution| {
                        let mut table = UnificationTable::new(db);
                        query.extract_solution(&mut table, solution).implementor
                    })
                    .collect(),
            ),
            GoalSatisfiability::ContainsInvalid | GoalSatisfiability::UnSat(_) => {
                Selection::NotFound
            }
        }
    }

    pub(crate) fn search_ingots_for_trait_inst(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> (Ingot<'db>, Option<Ingot<'db>>) {
        Self::search_ingots_for_trait_inst_with_origin(db, self.origin_ingot, inst)
    }

    pub(crate) fn search_ingots_for_trait_inst_with_origin(
        db: &'db dyn HirAnalysisDb,
        origin_ingot: Ingot<'db>,
        inst: TraitInstId<'db>,
    ) -> (Ingot<'db>, Option<Ingot<'db>>) {
        let trait_ingot = inst.def(db).ingot(db);
        let self_ty = inst.self_ty(db);
        let self_ingot = self_ty.ingot(db).or_else(|| {
            // For projection `Self` types that still don't yield an ingot (e.g. all-trait-param
            // args), fall back to other trait arguments as a best-effort proxy.
            match self_ty.data(db) {
                TyData::AssocTy(_) | TyData::QualifiedTy(_) => {
                    inst.args(db).iter().skip(1).find_map(|ty| ty.ingot(db))
                }
                _ => None,
            }
        });

        let primary = self_ingot.unwrap_or(origin_ingot);
        if primary == trait_ingot {
            (primary, None)
        } else {
            (primary, Some(trait_ingot))
        }
    }

    pub(crate) fn normalization_scope_for_trait_inst(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> ScopeId<'db> {
        Self::normalization_scope_for_trait_inst_with_origin(db, self.origin_ingot, inst)
    }

    pub(crate) fn normalization_scope_for_trait_inst_with_origin(
        db: &'db dyn HirAnalysisDb,
        origin_ingot: Ingot<'db>,
        inst: TraitInstId<'db>,
    ) -> ScopeId<'db> {
        let norm_ingot = inst
            .self_ty(db)
            .ingot(db)
            .or_else(|| inst.args(db).iter().find_map(|ty| ty.ingot(db)))
            .unwrap_or(origin_ingot);
        norm_ingot.root_mod(db).scope()
    }
}

pub(crate) fn normalize_trait_inst_preserving_validity<'db>(
    db: &'db dyn HirAnalysisDb,
    inst: TraitInstId<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> TraitInstId<'db> {
    let normalized_args: Vec<_> = inst
        .args(db)
        .iter()
        .map(|&arg| normalize_ty_without_consts_with_solve_cx(db, arg, scope, assumptions, None))
        .collect();
    let normalized_assoc_type_bindings = inst
        .assoc_type_bindings(db)
        .iter()
        .map(|(&name, &ty)| {
            (
                name,
                normalize_ty_without_consts_with_solve_cx(db, ty, scope, assumptions, None),
            )
        })
        .collect::<IndexMap<_, _>>();
    let normalized = TraitInstId::new(
        db,
        inst.def(db),
        normalized_args,
        normalized_assoc_type_bindings,
    );
    let original_has_invalid = inst.args(db).iter().copied().any(|ty| ty.has_invalid(db))
        || inst
            .assoc_type_bindings(db)
            .values()
            .copied()
            .any(|ty| ty.has_invalid(db));
    let normalized_has_invalid = normalized
        .args(db)
        .iter()
        .copied()
        .any(|ty| ty.has_invalid(db))
        || normalized
            .assoc_type_bindings(db)
            .values()
            .copied()
            .any(|ty| ty.has_invalid(db));
    if !original_has_invalid && normalized_has_invalid {
        inst
    } else {
        normalized
    }
}

pub(crate) fn normalize_trait_inst_preserving_validity_with_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    inst: TraitInstId<'db>,
    solve_cx: TraitSolveCx<'db>,
) -> TraitInstId<'db> {
    let scope = solve_cx.normalization_scope_for_trait_inst(db, inst);
    let assumptions = solve_cx.assumptions();
    if solve_cx.local_implementors().is_none() {
        return normalize_trait_inst_preserving_validity(db, inst, scope, assumptions);
    }

    let normalized_args: Vec<_> = inst
        .args(db)
        .iter()
        .map(|&arg| {
            normalize_ty_without_consts_with_solve_cx(db, arg, scope, assumptions, Some(solve_cx))
        })
        .collect();
    let normalized_assoc_type_bindings = inst
        .assoc_type_bindings(db)
        .iter()
        .map(|(&name, &ty)| {
            (
                name,
                normalize_ty_without_consts_with_solve_cx(
                    db,
                    ty,
                    scope,
                    assumptions,
                    Some(solve_cx),
                ),
            )
        })
        .collect::<IndexMap<_, _>>();
    let normalized = TraitInstId::new(
        db,
        inst.def(db),
        normalized_args,
        normalized_assoc_type_bindings,
    );
    let original_has_invalid = inst.args(db).iter().copied().any(|ty| ty.has_invalid(db))
        || inst
            .assoc_type_bindings(db)
            .values()
            .copied()
            .any(|ty| ty.has_invalid(db));
    let normalized_has_invalid = normalized
        .args(db)
        .iter()
        .copied()
        .any(|ty| ty.has_invalid(db))
        || normalized
            .assoc_type_bindings(db)
            .values()
            .copied()
            .any(|ty| ty.has_invalid(db));
    if !original_has_invalid && normalized_has_invalid {
        inst
    } else {
        normalized
    }
}

fn is_query_satisfiable_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _origin_ingot: Ingot<'db>,
    _local_implementors: Option<LocalImplementorSet<'db>>,
    _query: Canonical<TraitSolverQuery<'db>>,
) -> GoalSatisfiability<'db> {
    GoalSatisfiability::NeedsConfirmation(IndexSet::default())
}

fn is_query_satisfiable_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &GoalSatisfiability<'db>,
    _count: u32,
    _origin_ingot: Ingot<'db>,
    _local_implementors: Option<LocalImplementorSet<'db>>,
    _query: Canonical<TraitSolverQuery<'db>>,
) -> salsa::CycleRecoveryAction<GoalSatisfiability<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

#[salsa::tracked(
    return_ref,
    cycle_fn=is_query_satisfiable_cycle_recover,
    cycle_initial=is_query_satisfiable_cycle_initial
)]
fn is_query_satisfiable<'db>(
    db: &'db dyn HirAnalysisDb,
    origin_ingot: Ingot<'db>,
    local_implementors: Option<LocalImplementorSet<'db>>,
    query: Canonical<TraitSolverQuery<'db>>,
) -> GoalSatisfiability<'db> {
    let flags = collect_flags(db, query.value);
    if flags.contains(TyFlags::HAS_INVALID) {
        return GoalSatisfiability::ContainsInvalid;
    };

    ProofForest::new(db, origin_ingot, local_implementors, query).solve()
}

pub fn is_goal_query_satisfiable<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    query: &CanonicalGoalQuery<'db>,
) -> GoalSatisfiability<'db> {
    is_query_satisfiable(
        db,
        solve_cx.origin_ingot(),
        solve_cx.local_implementors(),
        query.canonical(),
    )
    .clone()
}

pub fn is_goal_satisfiable<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    goal: TraitInstId<'db>,
) -> GoalSatisfiability<'db> {
    let query = CanonicalGoalQuery::new(db, goal, solve_cx.assumptions());
    is_goal_query_satisfiable(db, solve_cx, &query)
}

fn check_ty_wf_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _solve_cx: TraitSolveCx<'db>,
    _ty: TyId<'db>,
) -> WellFormedness<'db> {
    WellFormedness::WellFormed
}

fn check_ty_wf_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &WellFormedness<'db>,
    _count: u32,
    _solve_cx: TraitSolveCx<'db>,
    _ty: TyId<'db>,
) -> salsa::CycleRecoveryAction<WellFormedness<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

fn check_trait_inst_wf_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _solve_cx: TraitSolveCx<'db>,
    _trait_inst: TraitInstId<'db>,
) -> WellFormedness<'db> {
    WellFormedness::WellFormed
}

fn check_trait_inst_wf_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &WellFormedness<'db>,
    _count: u32,
    _solve_cx: TraitSolveCx<'db>,
    _trait_inst: TraitInstId<'db>,
) -> salsa::CycleRecoveryAction<WellFormedness<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

pub(crate) fn first_invalid_trait_const_goal_with_solve_cx<'db, T>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    value: &T,
) -> Option<TraitInstId<'db>>
where
    T: TyVisitable<'db>,
{
    struct InvalidTraitConstGoalFinder<'db> {
        db: &'db dyn HirAnalysisDb,
        solve_cx: TraitSolveCx<'db>,
        goal: Option<TraitInstId<'db>>,
    }

    impl<'db> TyVisitor<'db> for InvalidTraitConstGoalFinder<'db> {
        fn db(&self) -> &'db dyn HirAnalysisDb {
            self.db
        }

        fn visit_invalid(&mut self, cause: &InvalidCause<'db>) {
            if self.goal.is_some() {
                return;
            }

            match cause {
                InvalidCause::TraitConstNotImplemented { inst, .. } => self.goal = Some(*inst),
                InvalidCause::KindMismatch { given, .. }
                | InvalidCause::NormalTypeExpected { given } => self.visit_ty(*given),
                InvalidCause::ConstTyMismatch { expected, given } => {
                    self.visit_ty(*expected);
                    if self.goal.is_none() {
                        self.visit_ty(*given);
                    }
                }
                InvalidCause::ConstTyExpected { expected } => self.visit_ty(*expected),
                InvalidCause::NotFullyApplied
                | InvalidCause::TooManyGenericArgs { .. }
                | InvalidCause::InvalidConstParamTy
                | InvalidCause::RecursiveConstParamTy
                | InvalidCause::UnboundTypeAliasParam { .. }
                | InvalidCause::AliasCycle(..)
                | InvalidCause::InvalidConstTyExpr { .. }
                | InvalidCause::ConstEvalUnsupported { .. }
                | InvalidCause::ConstEvalNonConstCall { .. }
                | InvalidCause::ConstEvalDivisionByZero { .. }
                | InvalidCause::ConstEvalStepLimitExceeded { .. }
                | InvalidCause::ConstEvalRecursionLimitExceeded { .. }
                | InvalidCause::ParseError
                | InvalidCause::PathResolutionFailed { .. }
                | InvalidCause::NotAType(..)
                | InvalidCause::Other => {}
            }
        }

        fn visit_ty(&mut self, ty: TyId<'db>) {
            if self.goal.is_some() {
                return;
            }
            let Some(InvalidCause::TraitConstNotImplemented { inst, .. }) =
                ty.invalid_cause(self.db)
            else {
                walk_ty(self, ty);
                return;
            };
            self.goal = Some(inst);
        }

        fn visit_const_ty(&mut self, const_ty: &super::const_ty::ConstTyId<'db>) {
            if self.goal.is_some() {
                return;
            }

            match const_ty.data(self.db) {
                ConstTyData::Abstract(expr, _) => {
                    let ConstExpr::TraitConst(assoc) = expr.data(self.db) else {
                        walk_const_ty(self, const_ty);
                        return;
                    };
                    let solve_cx = self.solve_cx.with_assumptions(assoc.assumptions());
                    if is_current_impl_trait_const_projection(
                        self.db,
                        solve_cx,
                        assoc.analysis_cx(self.db, Some(solve_cx)),
                        assoc.inst(),
                    ) {
                        walk_const_ty(self, const_ty);
                        return;
                    }
                    self.goal = concretized_missing_trait_const_goal(
                        self.db,
                        solve_cx,
                        assoc.inst(),
                        assoc.name(),
                    );
                }
                ConstTyData::UnEvaluated {
                    body,
                    unevaluated_cx,
                    ..
                } => {
                    let body_cx = unevaluated_cx.map(StoredAnalysisCx::get);
                    let body_solve_cx = body_cx
                        .map(|cx| cx.proof.solve_cx())
                        .unwrap_or(self.solve_cx.with_assumptions(self.solve_cx.assumptions()));
                    if let Some(path) = const_body_simple_path(self.db, *body)
                        && let Some(PathRes::TraitConst(recv_ty, inst, name)) = body_cx
                            .and_then(|cx| {
                                contextual_path_resolution_in_cx(
                                    self.db,
                                    body.scope(),
                                    path,
                                    true,
                                    &cx,
                                )
                                .or_else(|| {
                                    resolve_path_in_cx(self.db, path, body.scope(), true, &cx).ok()
                                })
                            })
                            .or_else(|| {
                                resolve_path(
                                    self.db,
                                    path,
                                    body.scope(),
                                    body_solve_cx.assumptions(),
                                    true,
                                )
                                .ok()
                            })
                    {
                        let mut args = inst.args(self.db).clone();
                        if let Some(self_arg) = args.first_mut() {
                            *self_arg = recv_ty;
                        }
                        let inst = TraitInstId::new(
                            self.db,
                            inst.def(self.db),
                            args,
                            inst.assoc_type_bindings(self.db).clone(),
                        );
                        if is_current_impl_trait_const_projection(
                            self.db,
                            body_solve_cx,
                            body_cx,
                            inst,
                        ) {
                            walk_const_ty(self, const_ty);
                            return;
                        }
                        self.goal = concretized_missing_trait_const_goal(
                            self.db,
                            body_solve_cx,
                            inst,
                            name,
                        );
                    }
                }
                _ => {}
            }

            if self.goal.is_none() {
                walk_const_ty(self, const_ty);
            }
        }
    }

    let mut finder = InvalidTraitConstGoalFinder {
        db,
        solve_cx,
        goal: None,
    };
    value.visit_with(&mut finder);
    finder.goal
}

pub(crate) fn concretized_missing_trait_const_goal<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    inst: TraitInstId<'db>,
    name: crate::hir_def::IdentId<'db>,
) -> Option<TraitInstId<'db>> {
    let inst = normalize_trait_inst_preserving_validity_with_solve_cx(db, inst, solve_cx);
    let flags = collect_flags(db, inst);
    if flags.intersects(TyFlags::HAS_PARAM | TyFlags::HAS_VAR | TyFlags::HAS_INVALID) {
        return None;
    }

    match solve_cx.select_impl(db, inst) {
        Selection::NotFound => Some(inst),
        Selection::Unique(_) | Selection::Ambiguous(_) => {
            let _ = name;
            None
        }
    }
}

fn first_ill_formed_nested_ty<'db, T>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    value: &T,
    skip_root: Option<TyId<'db>>,
) -> Option<WellFormedness<'db>>
where
    T: TyVisitable<'db>,
{
    struct NestedTyCollector<'db> {
        db: &'db dyn HirAnalysisDb,
        skip_root: Option<TyId<'db>>,
        tys: IndexSet<TyId<'db>>,
    }

    impl<'db> TyVisitor<'db> for NestedTyCollector<'db> {
        fn db(&self) -> &'db dyn HirAnalysisDb {
            self.db
        }

        fn visit_const_param(&mut self, _param: &TyParam<'db>, const_ty_ty: TyId<'db>) {
            self.visit_ty(const_ty_ty);
        }

        fn visit_ty(&mut self, ty: TyId<'db>) {
            let should_walk = if Some(ty) == self.skip_root {
                true
            } else {
                self.tys.insert(ty)
            };
            if should_walk {
                walk_ty(self, ty);
            }
        }
    }

    let mut collector = NestedTyCollector {
        db,
        skip_root,
        tys: IndexSet::new(),
    };
    value.visit_with(&mut collector);
    collector.tys.into_iter().find_map(|ty| {
        if let Some(inst) = first_invalid_trait_const_goal_with_solve_cx(db, solve_cx, &ty) {
            let goal = normalize_trait_inst_preserving_validity_with_solve_cx(db, inst, solve_cx);
            return Some(WellFormedness::IllFormed {
                goal,
                subgoal: None,
            });
        }

        if collect_flags(db, ty).contains(TyFlags::HAS_INVALID) {
            return None;
        }

        let wf = check_ty_wf(db, solve_cx, ty);
        (!wf.is_wf()).then_some(wf)
    })
}

pub(crate) fn check_ty_wf_nested<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    ty: TyId<'db>,
) -> WellFormedness<'db> {
    if let Some(wf) = first_ill_formed_nested_ty(db, solve_cx, &ty, Some(ty))
        && !wf.is_wf()
    {
        return wf;
    }

    check_ty_wf(db, solve_cx, ty)
}

/// Checks if the given type is well-formed, i.e., the arguments of the given
/// type applications satisfies the constraints under the given assumptions.
#[salsa::tracked(
    cycle_fn=check_ty_wf_cycle_recover,
    cycle_initial=check_ty_wf_cycle_initial
)]
pub(crate) fn check_ty_wf<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    ty: TyId<'db>,
) -> WellFormedness<'db> {
    if ty_has_recursive_assoc_projection(db, ty) {
        return WellFormedness::WellFormed;
    }

    if let Some(goal) = current_projection_goal(db, solve_cx, ty)
        && let Some(wf) = check_projection_goal_wf(db, solve_cx, goal)
    {
        return wf;
    }

    let (_, args) = ty.decompose_ty_app(db);

    for &arg in args {
        let wf = check_ty_wf(db, solve_cx, arg);
        if !wf.is_wf() {
            return wf;
        }
    }

    let constraints = ty_constraints(db, ty);
    let assumptions = solve_cx.assumptions();

    // Normalize constraints to resolve associated types
    let normalized_constraints = {
        let normalized_list: Vec<_> = constraints
            .list(db)
            .iter()
            .map(|&goal| normalize_trait_inst_preserving_validity_with_solve_cx(db, goal, solve_cx))
            .collect();
        PredicateListId::new(db, normalized_list)
    };

    for &goal in normalized_constraints.list(db) {
        let mut table = UnificationTable::new(db);
        let query = CanonicalGoalQuery::new(db, goal, assumptions);

        if let GoalSatisfiability::UnSat(subgoal) = is_goal_query_satisfiable(db, solve_cx, &query)
        {
            let subgoal = subgoal.map(|subgoal| query.extract_subgoal(&mut table, subgoal));
            return WellFormedness::IllFormed { goal, subgoal };
        }
    }

    WellFormedness::WellFormed
}

fn current_projection_goal<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    ty: TyId<'db>,
) -> Option<TraitInstId<'db>> {
    match ty.data(db) {
        TyData::AssocTy(assoc_ty) => Some(assoc_ty.trait_),
        TyData::QualifiedTy(trait_inst) => Some(*trait_inst),
        TyData::ConstTy(const_ty) => {
            let ConstTyData::Abstract(expr, _) = const_ty.data(db) else {
                return None;
            };
            let ConstExpr::TraitConst(assoc) = expr.data(db) else {
                return None;
            };
            let solve_cx = solve_cx.with_assumptions(assoc.assumptions());
            if is_current_impl_trait_const_projection(
                db,
                solve_cx,
                assoc.analysis_cx(db, Some(solve_cx)),
                assoc.inst(),
            ) {
                return None;
            }
            Some(assoc.inst())
        }
        _ => None,
    }
}

fn check_projection_goal_wf<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    goal: TraitInstId<'db>,
) -> Option<WellFormedness<'db>> {
    let wf = check_trait_inst_wf(db, solve_cx, goal);
    if !wf.is_wf() {
        return Some(wf);
    }

    let mut table = UnificationTable::new(db);
    let query = CanonicalGoalQuery::new(db, goal, solve_cx.assumptions());
    if let GoalSatisfiability::UnSat(subgoal) = is_goal_query_satisfiable(db, solve_cx, &query) {
        let subgoal = subgoal.map(|subgoal| query.extract_subgoal(&mut table, subgoal));
        return Some(WellFormedness::IllFormed { goal, subgoal });
    }

    None
}

fn is_current_impl_trait_const_projection<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    cx: Option<AnalysisCx<'db>>,
    inst: TraitInstId<'db>,
) -> bool {
    let Some(cx) = cx else {
        return false;
    };
    let Some(current_impl) = cx.mode.current_impl().or_else(|| cx.overlay.current_impl()) else {
        return false;
    };

    normalize_trait_inst_preserving_validity_with_solve_cx(db, inst, solve_cx)
        == normalize_trait_inst_preserving_validity_with_solve_cx(
            db,
            current_impl.trait_inst(db),
            solve_cx,
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Update)]
pub(crate) enum WellFormedness<'db> {
    WellFormed,
    IllFormed {
        goal: TraitInstId<'db>,
        subgoal: Option<TraitInstId<'db>>,
    },
}

impl WellFormedness<'_> {
    fn is_wf(self) -> bool {
        matches!(self, WellFormedness::WellFormed)
    }
}

/// Checks if the given trait instance are well-formed, i.e., the arguments of
/// the trait satisfies all constraints under the given assumptions.
#[salsa::tracked(
    cycle_fn=check_trait_inst_wf_cycle_recover,
    cycle_initial=check_trait_inst_wf_cycle_initial
)]
pub(crate) fn check_trait_inst_wf<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    trait_inst: TraitInstId<'db>,
) -> WellFormedness<'db> {
    if predicate_has_recursive_assoc_projection(db, trait_inst) {
        return WellFormedness::WellFormed;
    }

    let is_trait_header_root = matches!(
        trait_inst.self_ty(db).data(db),
        TyData::TyParam(param)
            if !param.is_effect() && param.owner == trait_inst.def(db).scope() && param.idx == 0
    );
    if !is_trait_header_root
        && let Some(wf) = first_ill_formed_nested_ty(db, solve_cx, &trait_inst, None)
        && !wf.is_wf()
    {
        return wf;
    }

    let constraints =
        collect_constraints(db, trait_inst.def(db).into()).instantiate(db, trait_inst.args(db));
    let assumptions = solve_cx.assumptions();

    // Normalize constraints after instantiation to resolve associated types
    let normalized_constraints = {
        let normalized_list: Vec<_> = constraints
            .list(db)
            .iter()
            .map(|&goal| normalize_trait_inst_preserving_validity_with_solve_cx(db, goal, solve_cx))
            .collect();
        PredicateListId::new(db, normalized_list)
    };

    for &goal in normalized_constraints.list(db) {
        let mut table = UnificationTable::new(db);
        let query = CanonicalGoalQuery::new(db, goal, assumptions);
        if let GoalSatisfiability::UnSat(subgoal) = is_goal_query_satisfiable(db, solve_cx, &query)
        {
            let subgoal = subgoal.map(|subgoal| query.extract_subgoal(&mut table, subgoal));
            return WellFormedness::IllFormed { goal, subgoal };
        }
    }

    WellFormedness::WellFormed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct TraitGoalSolution<'db> {
    pub(crate) inst: TraitInstId<'db>,
    pub(crate) implementor: ImplementorId<'db>,
}

#[derive(Debug, Clone, PartialEq, Eq, Update)]
pub enum GoalSatisfiability<'db> {
    /// Goal is satisfied with the unique solution.
    Satisfied(Solution<TraitGoalSolution<'db>>),
    /// Goal might be satisfied, but needs more type information to determine
    /// satisfiability and uniqueness.
    NeedsConfirmation(IndexSet<Solution<TraitGoalSolution<'db>>>),

    /// Goal contains invalid.
    ContainsInvalid,
    /// The gaol is not satisfied.
    /// It contains an unsatisfied subgoal if we can know the exact subgoal
    /// that makes the proof step stuck.
    UnSat(Option<Solution<TraitInstId<'db>>>),
}

impl GoalSatisfiability<'_> {
    pub fn is_satisfied(&self) -> bool {
        matches!(
            self,
            Self::Satisfied(_) | Self::NeedsConfirmation(_) | Self::ContainsInvalid
        )
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct PredicateListId<'db> {
    #[return_ref]
    pub list: Vec<TraitInstId<'db>>,
}

impl<'db> PredicateListId<'db> {
    pub fn pretty_print(&self, db: &'db dyn HirAnalysisDb) -> String {
        format!(
            "{{{}}}",
            self.list(db)
                .iter()
                .map(|pred| pred.pretty_print(db, true))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    pub(super) fn merge(self, db: &'db dyn HirAnalysisDb, other: Self) -> Self {
        let mut predicates = self.list(db).clone();
        predicates.extend(other.list(db));
        PredicateListId::new(db, predicates)
    }

    pub fn empty_list(db: &'db dyn HirAnalysisDb) -> Self {
        Self::new(db, Vec::new())
    }

    pub fn is_empty(self, db: &'db dyn HirAnalysisDb) -> bool {
        self.list(db).is_empty()
    }

    /// Transitively extends the predicate list with all implied bounds:
    /// - Super trait bounds
    /// - Associated type bounds from trait definitions
    pub fn extend_all_bounds(self, db: &'db dyn HirAnalysisDb) -> Self {
        let mut all_predicates: IndexSet<TraitInstId<'db>> =
            self.list(db).iter().copied().collect();

        let mut worklist: Vec<TraitInstId<'db>> = self.list(db).to_vec();

        while let Some(pred) = worklist.pop() {
            // 1. Collect super traits
            for super_trait in pred.def(db).super_traits(db) {
                // Instantiate with current predicate's args
                let inst = super_trait.instantiate(db, pred.args(db));

                // Also substitute `Self` and associated types using current predicate's
                // assoc-type bindings so derived bounds are as concrete as possible.
                let mut subst = AssocTySubst::new(pred);
                let inst = inst.fold_with(db, &mut subst);
                if predicate_has_recursive_assoc_projection(db, inst) {
                    continue;
                }

                if all_predicates.insert(inst) {
                    // New predicate added, add to worklist for further processing
                    worklist.push(inst);
                }
            }

            // 2. Collect associated type bounds
            let hir_trait = pred.def(db);
            for trait_type in hir_trait.assoc_types(db) {
                // Get the associated type name
                let Some(assoc_ty_name) = trait_type.name(db) else {
                    continue;
                };

                // Create the associated type: Self::AssocType
                let assoc_ty = TyId::assoc_ty(db, pred, assoc_ty_name);

                let assoc_bounds = assoc_ty
                    .assoc_type_bounds(db, trait_type)
                    .collect::<Vec<_>>();

                for mut trait_inst in assoc_bounds {
                    // Substitute `Self` and associated types using the original predicate instance
                    let mut subst = AssocTySubst::new(pred);
                    trait_inst = trait_inst.fold_with(db, &mut subst);
                    if predicate_has_recursive_assoc_projection(db, trait_inst) {
                        continue;
                    }
                    if all_predicates.insert(trait_inst) {
                        worklist.push(trait_inst);
                    }
                }
            }
        }

        Self::new(db, all_predicates.into_iter().collect::<Vec<_>>())
    }
}

pub(crate) fn predicate_has_recursive_assoc_projection<'db>(
    db: &'db dyn HirAnalysisDb,
    pred: TraitInstId<'db>,
) -> bool {
    pred.args(db)
        .iter()
        .any(|&arg| ty_has_recursive_assoc_projection(db, arg))
}

pub(crate) fn ty_has_recursive_assoc_projection<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
) -> bool {
    fn impl_<'db>(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        visited_tys: &mut FxHashSet<TyId<'db>>,
        seen_assoc_keys: &mut FxHashSet<(crate::hir_def::Trait<'db>, crate::hir_def::IdentId<'db>)>,
    ) -> bool {
        if !visited_tys.insert(ty) {
            return false;
        }

        let has_cycle = match ty.data(db) {
            TyData::ConstTy(const_ty) => impl_(db, const_ty.ty(db), visited_tys, seen_assoc_keys),
            TyData::AssocTy(assoc_ty) => {
                let key = (assoc_ty.trait_.def(db), assoc_ty.name);
                if !seen_assoc_keys.insert(key) {
                    true
                } else {
                    let has_cycle = assoc_ty
                        .trait_
                        .args(db)
                        .iter()
                        .copied()
                        .any(|arg| impl_(db, arg, visited_tys, seen_assoc_keys));
                    seen_assoc_keys.remove(&key);
                    has_cycle
                }
            }
            TyData::QualifiedTy(trait_inst) => {
                let args_have_cycle = trait_inst
                    .args(db)
                    .iter()
                    .copied()
                    .any(|arg| impl_(db, arg, visited_tys, seen_assoc_keys));
                let assoc_bindings_have_cycle = trait_inst
                    .assoc_type_bindings(db)
                    .values()
                    .copied()
                    .any(|ty| impl_(db, ty, visited_tys, seen_assoc_keys));
                args_have_cycle || assoc_bindings_have_cycle
            }
            TyData::TyApp(lhs, rhs) => {
                impl_(db, *lhs, visited_tys, seen_assoc_keys)
                    || impl_(db, *rhs, visited_tys, seen_assoc_keys)
            }
            _ => false,
        };

        visited_tys.remove(&ty);
        has_cycle
    }

    impl_(db, ty, &mut FxHashSet::default(), &mut FxHashSet::default())
}

#[cfg(test)]
mod tests {
    use common::indexmap::IndexMap;

    use super::{
        CanonicalGoalQuery, GoalSatisfiability, TraitInstId, TraitSolveCx, WellFormedness,
        check_trait_inst_wf, is_goal_query_satisfiable,
    };
    use crate::{
        analysis::ty::{
            trait_resolution::constraint::collect_func_def_constraints, ty_def::TyId,
            ty_lower::collect_generic_params,
        },
        hir_def::{Func, Trait},
        test_db::HirAnalysisTestDb,
    };

    #[test]
    fn solver_query_includes_assumptions() {
        fn query_for<'db>(
            db: &'db HirAnalysisTestDb,
            func: Func<'db>,
            needs_a: Trait<'db>,
        ) -> (CanonicalGoalQuery<'db>, TraitSolveCx<'db>) {
            let ty_param = collect_generic_params(db, func.into()).explicit_params(db)[0];
            let assumptions =
                collect_func_def_constraints(db, func.into(), true).instantiate_identity();
            let goal =
                TraitInstId::new(db, needs_a, vec![TyId::unit(db), ty_param], IndexMap::new());
            let query = CanonicalGoalQuery::new(db, goal, assumptions);
            let solve_cx = TraitSolveCx::new(db, func.scope()).with_assumptions(assumptions);
            (query, solve_cx)
        }

        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(
            "trait_solver_query_includes_assumptions.fe".into(),
            r#"
trait A {}
trait NeedsA<T> {}

impl<T: A> NeedsA<T> for () {}

fn with_a<T: A>() -> bool {
    true
}

fn without_a<T>() -> bool {
    true
}
"#,
        );
        let (top_mod, _) = db.top_mod(file);
        db.assert_no_diags(top_mod);

        let needs_a = top_mod
            .all_traits(&db)
            .iter()
            .copied()
            .find(|trait_| {
                trait_
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "NeedsA")
            })
            .unwrap();
        let with_a = top_mod
            .all_funcs(&db)
            .iter()
            .copied()
            .find(|func| {
                func.name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "with_a")
            })
            .unwrap();
        let without_a = top_mod
            .all_funcs(&db)
            .iter()
            .copied()
            .find(|func| {
                func.name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "without_a")
            })
            .unwrap();

        let (with_query, with_cx) = query_for(&db, with_a, needs_a);
        let (without_query, without_cx) = query_for(&db, without_a, needs_a);

        assert_eq!(
            with_query.goal().pretty_print(&db, true),
            without_query.goal().pretty_print(&db, true)
        );
        assert_ne!(with_query.canonical(), without_query.canonical());
        assert!(matches!(
            is_goal_query_satisfiable(&db, with_cx, &with_query),
            GoalSatisfiability::Satisfied(_)
        ));
        assert!(matches!(
            is_goal_query_satisfiable(&db, without_cx, &without_query),
            GoalSatisfiability::UnSat(_)
        ));
    }

    #[test]
    fn concrete_nested_trait_const_arg_is_ill_formed() {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(
            "concrete_nested_trait_const_arg_is_ill_formed.fe".into(),
            r#"
trait HasN {
    const N: u32
}

struct Slot<const N: u32> {}

trait Foo<T> {}

struct Missing {}
struct S {}

impl Foo<Slot<{ <Missing as HasN>::N }>> for S {}
"#,
        );
        let (top_mod, _) = db.top_mod(file);
        let impl_trait = top_mod.all_impl_traits(&db)[0];
        let trait_inst = impl_trait
            .trait_inst_result(&db)
            .expect("trait ref should lower before WF checking");
        let solve_cx = TraitSolveCx::new(&db, impl_trait.scope())
            .with_assumptions(impl_trait.assumptions(&db));

        assert!(matches!(
            check_trait_inst_wf(&db, solve_cx, trait_inst),
            WellFormedness::IllFormed { .. }
        ));
    }
}

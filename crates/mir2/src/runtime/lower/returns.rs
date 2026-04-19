use std::collections::VecDeque;
use std::sync::Arc;

use cranelift_entity::EntityRef;
use hir::analysis::semantic::{
    SLocalId, SemanticInstance,
    borrowck::{NSTerminatorKind, NormalizedSemanticBody, normalize_semantic_body},
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db::MirDb,
    instance::RuntimeInstanceKey,
    runtime::{RuntimeCarrier, RuntimeClass},
};

use super::{
    classify::{
        BodyEnv, BodyStaticFacts, InferClassCache, RuntimeVisibleReturnPlan, default_return_class,
        desired_runtime_return_plan, visible_return_class_for_local,
    },
    infer::merge_runtime_carrier,
    interface::runtime_visible_binding_plans,
};

#[derive(Clone)]
pub(crate) struct RuntimeReturnSummary<'db> {
    pub(crate) typed_body: &'db hir::analysis::ty::ty_check::TypedBody<'db>,
    pub(crate) semantic_body: NormalizedSemanticBody<'db>,
    pub(crate) facts: BodyStaticFacts<'db>,
    pub(crate) return_plan: RuntimeVisibleReturnPlan<'db>,
    pub(crate) default_return_class: Option<RuntimeClass<'db>>,
    pub(crate) param_locals: Box<[SLocalId]>,
    pub(crate) return_locals: Box<[SLocalId]>,
    pub(crate) slice_assignment_ids: Box<[usize]>,
    pub(crate) slice_assignment_positions: Box<[Option<usize>]>,
    pub(crate) slice_assignments_by_local: Vec<Vec<usize>>,
    pub(crate) slice_dynamic_dependents_by_local: Vec<Vec<SLocalId>>,
}

impl<'db> RuntimeReturnSummary<'db> {
    fn build(db: &'db dyn MirDb, semantic: SemanticInstance<'db>) -> Self {
        let typed_body = semantic.key(db).typed_body(db);
        let semantic_body = normalize_semantic_body(db, semantic).unwrap_or_else(|err| {
            panic!(
                "semantic normalization failed for {:?}: {err:?}",
                semantic.key(db)
            )
        });
        let facts = BodyStaticFacts::new(db, &semantic_body);
        let return_plan = desired_runtime_return_plan(db, typed_body);
        let default_return_class = default_return_class(db, typed_body);
        let param_locals = runtime_visible_binding_plans(db, semantic)
            .iter()
            .map(|entry| entry.local)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let return_locals = semantic_body
            .blocks
            .iter()
            .filter_map(|block| match &block.terminator.kind {
                NSTerminatorKind::Return(Some(value)) => Some(value.local),
                NSTerminatorKind::Goto(_)
                | NSTerminatorKind::Branch { .. }
                | NSTerminatorKind::MatchEnum { .. }
                | NSTerminatorKind::Return(None) => None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut def_assignment_by_local = vec![None; semantic_body.locals.len()];
        for (assign_id, assignment) in facts.assignments().iter().enumerate() {
            def_assignment_by_local[assignment.dst.index()] = Some(assign_id);
        }

        let mut needed_assignments = FxHashSet::default();
        let mut needed_locals = FxHashSet::default();
        let mut pending = return_locals.iter().copied().collect::<VecDeque<_>>();
        while let Some(local) = pending.pop_front() {
            if !needed_locals.insert(local) {
                continue;
            }
            for dependency in facts.source_locals(local).iter().copied() {
                pending.push_back(dependency);
            }
            if let Some(assign_id) = def_assignment_by_local[local.index()]
                && needed_assignments.insert(assign_id)
                && let Some(assignment) = facts.assignment(assign_id)
            {
                for used in assignment.uses().iter().copied() {
                    pending.push_back(used);
                }
            }
        }

        let mut slice_assignment_ids = needed_assignments.into_iter().collect::<Vec<_>>();
        slice_assignment_ids.sort_unstable();
        let mut slice_assignment_positions = vec![None; facts.assignments().len()];
        for (slice_idx, &assign_id) in slice_assignment_ids.iter().enumerate() {
            slice_assignment_positions[assign_id] = Some(slice_idx);
        }

        let mut slice_assignments_by_local = vec![Vec::new(); semantic_body.locals.len()];
        for &assign_id in &slice_assignment_ids {
            let assignment = facts
                .assignment(assign_id)
                .unwrap_or_else(|| panic!("missing sliced assignment {assign_id}"));
            for used in assignment.uses().iter().copied() {
                slice_assignments_by_local[used.index()].push(assign_id);
            }
        }

        let mut slice_dynamic_dependents_by_local = vec![Vec::new(); semantic_body.locals.len()];
        for local in needed_locals.iter().copied() {
            for dependency in facts.source_locals(local).iter().copied() {
                slice_dynamic_dependents_by_local[dependency.index()].push(local);
            }
        }

        Self {
            typed_body,
            semantic_body,
            facts,
            return_plan,
            default_return_class,
            param_locals,
            return_locals,
            slice_assignment_ids: slice_assignment_ids.into_boxed_slice(),
            slice_assignment_positions: slice_assignment_positions.into_boxed_slice(),
            slice_assignments_by_local,
            slice_dynamic_dependents_by_local,
        }
    }

    fn env(&self, db: &'db dyn MirDb) -> BodyEnv<'_, 'db> {
        BodyEnv::new(db, &self.semantic_body, self.typed_body, &self.facts)
    }
}

#[derive(Clone)]
enum ReturnNodeState<'db> {
    Evaluating { current: Option<RuntimeClass<'db>> },
    Ready(Option<RuntimeClass<'db>>),
}

impl<'db> ReturnNodeState<'db> {
    fn current(&self) -> Option<RuntimeClass<'db>> {
        match self {
            Self::Evaluating { current } | Self::Ready(current) => current.clone(),
        }
    }
}

pub(crate) struct RuntimeReturnAnalysisCx<'db> {
    db: &'db dyn MirDb,
    summary_cache: FxHashMap<SemanticInstance<'db>, Arc<RuntimeReturnSummary<'db>>>,
    value_cache: FxHashMap<RuntimeInstanceKey<'db>, ReturnNodeState<'db>>,
    dependents: FxHashMap<RuntimeInstanceKey<'db>, FxHashSet<RuntimeInstanceKey<'db>>>,
    solve_stack: Vec<RuntimeInstanceKey<'db>>,
    pending: VecDeque<RuntimeInstanceKey<'db>>,
    pending_set: FxHashSet<RuntimeInstanceKey<'db>>,
    active_nodes: FxHashSet<RuntimeInstanceKey<'db>>,
}

impl<'db> RuntimeReturnAnalysisCx<'db> {
    pub(crate) fn new(db: &'db dyn MirDb) -> Self {
        Self {
            db,
            summary_cache: FxHashMap::default(),
            value_cache: FxHashMap::default(),
            dependents: FxHashMap::default(),
            solve_stack: Vec::new(),
            pending: VecDeque::new(),
            pending_set: FxHashSet::default(),
            active_nodes: FxHashSet::default(),
        }
    }

    pub(crate) fn return_class_for_key(
        &mut self,
        key: RuntimeInstanceKey<'db>,
    ) -> Option<RuntimeClass<'db>> {
        if self.solve_stack.is_empty() {
            self.solve_from_root(key)
        } else {
            self.lookup_during_solve(key)
        }
    }

    fn summary(&mut self, semantic: SemanticInstance<'db>) -> Arc<RuntimeReturnSummary<'db>> {
        self.summary_cache
            .entry(semantic)
            .or_insert_with(|| Arc::new(RuntimeReturnSummary::build(self.db, semantic)))
            .clone()
    }

    fn current_value(&self, key: RuntimeInstanceKey<'db>) -> Option<RuntimeClass<'db>> {
        self.value_cache
            .get(&key)
            .and_then(ReturnNodeState::current)
    }

    fn ensure_node(&mut self, key: RuntimeInstanceKey<'db>) {
        if self.value_cache.contains_key(&key) {
            return;
        }
        let initial = key
            .semantic(self.db)
            .map(|semantic| self.summary(semantic).default_return_class.clone())
            .unwrap_or(None);
        self.value_cache
            .insert(key, ReturnNodeState::Evaluating { current: initial });
    }

    fn ensure_enqueued(&mut self, key: RuntimeInstanceKey<'db>) {
        self.ensure_node(key);
        if let Some(ReturnNodeState::Ready(current)) = self.value_cache.get(&key).cloned() {
            self.value_cache
                .insert(key, ReturnNodeState::Evaluating { current });
        }
        self.active_nodes.insert(key);
        if self.pending_set.insert(key) {
            self.pending.push_back(key);
        }
    }

    fn lookup_during_solve(&mut self, key: RuntimeInstanceKey<'db>) -> Option<RuntimeClass<'db>> {
        if let Some(&caller) = self.solve_stack.last() {
            self.dependents.entry(key).or_default().insert(caller);
        }
        self.ensure_enqueued(key);
        self.current_value(key)
    }

    fn solve_from_root(&mut self, root: RuntimeInstanceKey<'db>) -> Option<RuntimeClass<'db>> {
        self.ensure_enqueued(root);
        while let Some(key) = self.pending.pop_front() {
            self.pending_set.remove(&key);
            let old = self.current_value(key);
            let new = self.evaluate_key_once(key);
            if new != old {
                if let Some(state) = self.value_cache.get_mut(&key) {
                    *state = ReturnNodeState::Evaluating {
                        current: new.clone(),
                    };
                }
                let dependents = self
                    .dependents
                    .get(&key)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<Vec<_>>();
                for dependent in dependents {
                    self.ensure_enqueued(dependent);
                }
            }
        }
        let active = self.active_nodes.drain().collect::<Vec<_>>();
        for key in active {
            let current = self.current_value(key);
            self.value_cache
                .insert(key, ReturnNodeState::Ready(current));
        }
        self.current_value(root)
    }

    fn evaluate_key_once(&mut self, key: RuntimeInstanceKey<'db>) -> Option<RuntimeClass<'db>> {
        let semantic = key.semantic(self.db)?;
        let summary = self.summary(semantic);
        self.solve_stack.push(key);
        let carriers = ReturnSliceInferer::new(self.db, &summary, key.params(self.db), self).run();
        let env = summary.env(self.db);
        let mut returned = Vec::new();
        for local in summary.return_locals.iter().copied() {
            let Some(class) =
                visible_return_class_for_local(env, local, &summary.return_plan, &carriers)
            else {
                self.solve_stack.pop();
                return summary.default_return_class.clone();
            };
            returned.push(class);
        }
        self.solve_stack.pop();
        let Some(first) = returned.pop() else {
            return summary.default_return_class.clone();
        };
        if returned.iter().all(|class| class == &first) {
            Some(first)
        } else {
            summary.default_return_class.clone()
        }
    }
}

struct ReturnSliceInferer<'summary, 'cx, 'db> {
    db: &'db dyn MirDb,
    summary: &'summary RuntimeReturnSummary<'db>,
    carriers: Vec<RuntimeCarrier<'db>>,
    class_cache: InferClassCache<'db>,
    returns: &'cx mut RuntimeReturnAnalysisCx<'db>,
}

impl<'summary, 'cx, 'db> ReturnSliceInferer<'summary, 'cx, 'db> {
    fn new(
        db: &'db dyn MirDb,
        summary: &'summary RuntimeReturnSummary<'db>,
        params: &[RuntimeClass<'db>],
        returns: &'cx mut RuntimeReturnAnalysisCx<'db>,
    ) -> Self {
        let mut carriers = vec![RuntimeCarrier::Erased; summary.semantic_body.locals.len()];
        for (class, local) in params.iter().zip(summary.param_locals.iter().copied()) {
            carriers[local.index()] = RuntimeCarrier::Value(class.clone());
        }
        Self {
            db,
            summary,
            carriers,
            class_cache: InferClassCache::new(summary.semantic_body.locals.len()),
            returns,
        }
    }

    fn run(mut self) -> Vec<RuntimeCarrier<'db>> {
        self.infer_carriers();
        self.carriers
    }

    fn infer_carriers(&mut self) {
        let mut queued = vec![true; self.summary.slice_assignment_ids.len()];
        let mut worklist = VecDeque::from_iter(0..self.summary.slice_assignment_ids.len());
        while let Some(slice_idx) = worklist.pop_front() {
            queued[slice_idx] = false;
            let assign_id = self.summary.slice_assignment_ids[slice_idx];
            let assign = self.summary.facts.assignment(assign_id).unwrap_or_else(|| {
                panic!("missing sliced assignment facts for statement {assign_id}")
            });
            let stmt = &self.summary.semantic_body.blocks[assign.block_idx].stmts[assign.stmt_idx];
            let expr = match &stmt.kind {
                hir::analysis::semantic::NSStmtKind::Assign { expr, .. } => expr,
                hir::analysis::semantic::NSStmtKind::Store { .. } => {
                    panic!(
                        "sliced assignment facts point to non-assignment statement: block={} stmt={}",
                        assign.block_idx, assign.stmt_idx
                    )
                }
            };
            let desired = match self.summary.env(self.db).expr_direct_class(
                &self.carriers,
                assign.block_idx,
                assign.stmt_idx,
                expr,
                Some(&mut self.class_cache),
                self.returns,
            ) {
                Some(class) => RuntimeCarrier::Value(class),
                None => continue,
            };
            if self.set_carrier(assign.dst, desired) {
                self.propagate_local_change(assign.dst, &mut worklist, &mut queued);
            }
        }
    }

    fn set_carrier(&mut self, local: SLocalId, desired: RuntimeCarrier<'db>) -> bool {
        let current = self
            .carriers
            .get(local.index())
            .cloned()
            .unwrap_or(RuntimeCarrier::Erased);
        let desired = merge_runtime_carrier(self.db, current, desired);
        if self.carriers[local.index()] == desired {
            return false;
        }
        self.carriers[local.index()] = desired;
        self.class_cache.note_carrier_changed(local);
        true
    }

    fn propagate_local_change(
        &mut self,
        changed_local: SLocalId,
        worklist: &mut VecDeque<usize>,
        queued: &mut [bool],
    ) {
        let mut pending = VecDeque::from([changed_local]);
        let mut seen = vec![false; self.summary.semantic_body.locals.len()];
        while let Some(local) = pending.pop_front() {
            if std::mem::replace(&mut seen[local.index()], true) {
                continue;
            }
            self.class_cache.invalidate_local_dynamic_facts(local);
            for &assign_id in &self.summary.slice_assignments_by_local[local.index()] {
                let Some(slice_idx) = self.summary.slice_assignment_positions[assign_id] else {
                    continue;
                };
                if !queued[slice_idx] {
                    queued[slice_idx] = true;
                    worklist.push_back(slice_idx);
                }
            }
            for dependent in self.summary.slice_dynamic_dependents_by_local[local.index()]
                .iter()
                .copied()
            {
                pending.push_back(dependent);
            }
        }
    }
}

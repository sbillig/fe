use std::sync::Arc;
use std::{collections::VecDeque, convert::Infallible};

use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap, entity_impl};
use dataflow::{SparseAnalysis, solve_sparse};
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
        AssignmentId, BodyEnv, BodyStaticFacts, InferClassCache, RuntimeVisibleReturnPlan,
        default_return_class, desired_runtime_return_plan, selected_visible_return_for_local,
    },
    infer::{desired_runtime_value_carrier, merge_runtime_carrier, seed_root_provider_carriers},
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
    pub(crate) slice_assignment_ids: PrimaryMap<SliceAssignmentId, AssignmentId>,
    pub(crate) slice_assignment_positions: SecondaryMap<AssignmentId, Option<SliceAssignmentId>>,
    pub(crate) slice_assignments_by_local: Vec<Vec<AssignmentId>>,
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
        let env = BodyEnv::new(db, &semantic_body, typed_body, &facts);
        let mut return_plan = desired_runtime_return_plan(db, typed_body);
        let mut default_return_class = default_return_class(db, typed_body);
        if matches!(return_plan, RuntimeVisibleReturnPlan::Erased) {
            let mut fallback = None;
            let mut all_fallbacks_match = true;
            for local in return_locals.iter().copied() {
                let Some(class) = env.root_transport_fallback_class(local) else {
                    continue;
                };
                match &fallback {
                    Some(fallback) if fallback != &class => all_fallbacks_match = false,
                    None => fallback = Some(class),
                    Some(_) => {}
                }
            }
            if fallback.is_some() {
                return_plan = RuntimeVisibleReturnPlan::PassActual;
                if default_return_class.is_none() && all_fallbacks_match {
                    default_return_class = fallback;
                }
            }
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
            for assign_id in facts.assignments_defining_local(local).iter().copied() {
                if needed_assignments.insert(assign_id) && facts.assignment(assign_id).is_some() {
                    for used in facts.assignment_uses(assign_id).iter().copied() {
                        pending.push_back(used);
                    }
                }
            }
        }

        let mut slice_assignment_ids = needed_assignments.into_iter().collect::<Vec<_>>();
        slice_assignment_ids.sort_unstable();
        let slice_assignment_ids: PrimaryMap<SliceAssignmentId, AssignmentId> =
            slice_assignment_ids.into_iter().collect();
        let mut slice_assignment_positions = SecondaryMap::new();
        slice_assignment_positions.resize(facts.assignments().len());
        for (slice_idx, &assign_id) in slice_assignment_ids.iter() {
            slice_assignment_positions[assign_id] = Some(slice_idx);
        }

        let mut slice_assignments_by_local = vec![Vec::new(); semantic_body.locals.len()];
        for (_, &assign_id) in slice_assignment_ids.iter() {
            facts
                .assignment(assign_id)
                .unwrap_or_else(|| panic!("missing sliced assignment {assign_id:?}"));
            for used in facts.assignment_uses(assign_id).iter().copied() {
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
            slice_assignment_ids,
            slice_assignment_positions,
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
            let Some(selected) =
                selected_visible_return_for_local(env, local, &summary.return_plan, &carriers)
            else {
                self.solve_stack.pop();
                return summary.default_return_class.clone();
            };
            returned.push(selected.class);
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
    pending_dependents: Vec<SliceAssignmentId>,
    returns: &'cx mut RuntimeReturnAnalysisCx<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SliceAssignmentId(u32);
entity_impl!(SliceAssignmentId);

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
            pending_dependents: Vec::new(),
            returns,
        }
    }

    fn run(mut self) -> Vec<RuntimeCarrier<'db>> {
        seed_root_provider_carriers(self.summary.env(self.db), &mut self.carriers);
        solve_sparse(&mut self, &mut ());
        self.carriers
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

    fn collect_local_change_dependents(&mut self, changed_local: SLocalId) {
        let mut pending = vec![changed_local];
        let mut seen = vec![false; self.summary.semantic_body.locals.len()];
        let mut queued = SecondaryMap::with_default(false);
        queued.resize(self.summary.slice_assignment_ids.len());
        self.pending_dependents.clear();
        while let Some(local) = pending.pop() {
            if std::mem::replace(&mut seen[local.index()], true) {
                continue;
            }
            self.class_cache.invalidate_local_dynamic_facts(local);
            for &assign_id in &self.summary.slice_assignments_by_local[local.index()] {
                let Some(slice_id) = self.summary.slice_assignment_positions[assign_id] else {
                    continue;
                };
                if !queued[slice_id] {
                    queued[slice_id] = true;
                    self.pending_dependents.push(slice_id);
                }
            }
            for dependent in self.summary.slice_dynamic_dependents_by_local[local.index()]
                .iter()
                .copied()
            {
                pending.push(dependent);
            }
        }
    }
}

impl<'summary, 'cx, 'db> SparseAnalysis for ReturnSliceInferer<'summary, 'cx, 'db> {
    type Node = SliceAssignmentId;
    type State = ();
    type Error = Infallible;

    fn node_count(&self) -> usize {
        self.summary.slice_assignment_ids.len()
    }

    fn seed_nodes(&self) -> Vec<Self::Node> {
        self.summary.slice_assignment_ids.keys().collect()
    }

    fn step(&mut self, node: Self::Node, _: &mut Self::State) -> Result<bool, Self::Error> {
        self.pending_dependents.clear();
        let assign_id = self.summary.slice_assignment_ids[node];
        let assign = self.summary.facts.assignment(assign_id).unwrap_or_else(|| {
            panic!("missing sliced assignment facts for statement {assign_id:?}")
        });
        let local = &self.summary.semantic_body.locals[assign.dst.index()];
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
        let class = self.summary.env(self.db).expr_direct_class(
            &self.carriers,
            assign.block_idx,
            assign.stmt_idx,
            expr,
            Some(&mut self.class_cache),
            self.returns,
        );
        let Some(class) = class else {
            return Ok(false);
        };
        let desired = desired_runtime_value_carrier(local, class);
        if !self.set_carrier(assign.dst, desired) {
            return Ok(false);
        }
        self.collect_local_change_dependents(assign.dst);
        Ok(true)
    }

    fn dependents(&self, _node: Self::Node, out: &mut Vec<Self::Node>) {
        out.extend(self.pending_dependents.iter().copied());
    }
}

#[cfg(test)]
mod tests {
    use common::InputDb;
    use driver::DriverDataBase;
    use hir::{
        analysis::{
            semantic::{
                SemanticLocalKind, get_or_build_semantic_instance, root_semantic_instance_key,
            },
            ty::ty_check::BodyOwner,
        },
        hir_def::TopLevelMod,
    };
    use url::Url;

    use crate::{
        build_runtime_package,
        runtime::{RuntimeCarrier, RuntimeClass},
    };

    use super::*;
    use crate::runtime::{
        lower::{infer::LocalStateInferer, interface::runtime_param_locals},
        package::runtime_instance_for_semantic,
    };

    fn semantic_instance_for_named_func<'db>(
        db: &'db DriverDataBase,
        top_mod: TopLevelMod<'db>,
        name: &str,
    ) -> SemanticInstance<'db> {
        let func = top_mod
            .all_funcs(db)
            .iter()
            .copied()
            .find(|func| {
                func.name(db)
                    .to_opt()
                    .is_some_and(|func_name| func_name.data(db) == name)
            })
            .unwrap_or_else(|| panic!("missing function `{name}`"));
        let key = root_semantic_instance_key(db, BodyOwner::Func(func))
            .unwrap_or_else(|err| panic!("failed to root semantic function instance: {err:?}"));
        get_or_build_semantic_instance(db, key)
    }

    fn legacy_return_class_for_key<'db>(
        db: &'db DriverDataBase,
        key: RuntimeInstanceKey<'db>,
    ) -> Option<RuntimeClass<'db>> {
        let semantic = key
            .semantic(db)
            .expect("legacy return-class inference only applies to semantic runtime instances");
        let summary = RuntimeReturnSummary::build(db, semantic);
        let env = summary.env(db);
        let mut returns = RuntimeReturnAnalysisCx::new(db);
        let inferred = LocalStateInferer::new(
            env,
            key.params(db),
            &runtime_param_locals(db, semantic, key.params(db)),
            &mut returns,
        )
        .run();
        let mut returned = Vec::new();
        for local in summary.return_locals.iter().copied() {
            let Some(selected) = selected_visible_return_for_local(
                env,
                local,
                &summary.return_plan,
                &inferred.carriers,
            ) else {
                return summary.default_return_class.clone();
            };
            returned.push(selected.class);
        }
        let Some(first) = returned.pop() else {
            return summary.default_return_class.clone();
        };
        if returned.iter().all(|class| class == &first) {
            Some(first)
        } else {
            summary.default_return_class.clone()
        }
    }

    #[test]
    fn return_slice_includes_all_definitions_of_returned_local() {
        let mut db = DriverDataBase::default();
        let file_url =
            Url::parse("file:///return_slice_includes_all_definitions_of_returned_local.fe")
                .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                r#"
fn choose(_ flag: bool) -> u256 {
    let mut x = 1
    if flag {
        x = 2
    }
    x
}
"#
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let semantic = semantic_instance_for_named_func(&db, top_mod, "choose");
        let summary = RuntimeReturnSummary::build(&db, semantic);
        let return_local = *summary
            .return_locals
            .first()
            .expect("choose should return one local");
        let return_defs = summary
            .facts
            .assignments()
            .iter()
            .filter(|(_, assignment)| assignment.dst == return_local)
            .count();
        let sliced_return_defs = summary
            .slice_assignment_ids
            .iter()
            .filter(|&(_, &assign_id)| {
                summary
                    .facts
                    .assignment(assign_id)
                    .is_some_and(|assignment| assignment.dst == return_local)
            })
            .count();

        assert_eq!(
            return_defs, 2,
            "expected `choose` to define the returned local twice"
        );
        assert_eq!(
            sliced_return_defs, return_defs,
            "return slice should keep every definition of the returned local"
        );
    }

    #[test]
    fn provider_root_return_slice_matches_full_inference() {
        let mut db = DriverDataBase::default();
        let file_url =
            Url::parse("file:///provider_root_return_slice_matches_full_inference.fe").unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                r#"
struct Pair {
    a: u256,
    b: u256,
}

fn id_ctx() -> Pair uses (ctx: Pair) {
    ctx
}

msg Msg {
    #[selector = 1]
    Go -> u256
}

pub contract C {
    ctx: Pair

    init() uses (mut ctx) {
        ctx = Pair { a: 1, b: 2 }
    }

    recv Msg {
        Go -> u256 uses (ctx) {
            let pair = with (ctx) { id_ctx() }
            pair.a + pair.b
        }
    }
}
"#
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let package = build_runtime_package(&db, top_mod).expect("runtime package");
        let function = package
            .functions(&db)
            .iter()
            .copied()
            .find(|function| function.symbol(&db).contains("id_ctx"))
            .expect("missing specialized id_ctx runtime function");
        let key = function.instance(&db).key(&db);

        assert_eq!(
            RuntimeReturnAnalysisCx::new(&db).return_class_for_key(key),
            legacy_return_class_for_key(&db, key),
            "provider-root return slice should match full-body carrier inference:\ninstance={key:#?}"
        );
    }

    #[test]
    fn owned_aggregate_temporaries_match_full_inference_in_return_slices() {
        let mut db = DriverDataBase::default();
        let file_url = Url::parse(
            "file:///owned_aggregate_temporaries_match_full_inference_in_return_slices.fe",
        )
        .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                r#"
fn first(_ arr: [u8; 4]) -> u8 {
    let local = arr
    local[0]
}
"#
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let semantic = semantic_instance_for_named_func(&db, top_mod, "first");
        let instance = runtime_instance_for_semantic(&db, semantic);
        let key = instance.key(&db);
        let summary = RuntimeReturnSummary::build(&db, semantic);
        let env = summary.env(&db);

        let mut legacy_returns = RuntimeReturnAnalysisCx::new(&db);
        let legacy = LocalStateInferer::new(
            env,
            key.params(&db),
            &runtime_param_locals(&db, semantic, key.params(&db)),
            &mut legacy_returns,
        )
        .run();
        let mut slice_returns = RuntimeReturnAnalysisCx::new(&db);
        let sliced =
            ReturnSliceInferer::new(&db, &summary, key.params(&db), &mut slice_returns).run();
        let locals = summary
            .semantic_body
            .locals
            .iter()
            .enumerate()
            .filter_map(|(idx, local)| {
                (idx >= summary.param_locals.len()
                    && matches!(local.facts.interface, SemanticLocalKind::DirectValue)
                    && local.facts.root_demand.needs_projectable_owned_storage())
                .then_some(SLocalId::from_u32(idx as u32))
            })
            .collect::<Vec<_>>();
        assert_eq!(locals.len(), 1, "expected one owned aggregate temporary");
        let local = locals[0];

        assert_eq!(
            sliced[local.index()],
            legacy.carriers[local.index()],
            "return slice should infer the same owned aggregate temporary carrier as the full solver"
        );
        assert!(matches!(
            sliced[local.index()],
            RuntimeCarrier::Value(RuntimeClass::Ref { .. })
        ));
    }
}

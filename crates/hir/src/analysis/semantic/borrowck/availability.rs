//! Ownership dataflow shared by diagnostics and summary construction.
use super::validity::NativeValidity;
use crate::analysis::semantic::diagnostics::{
    SemanticDiagnostic, SemanticDiagnosticKind, SemanticDiagnosticSpan,
};
use std::collections::BTreeMap;

use cranelift_entity::EntityRef;

use super::{
    access::ResolvedAccess,
    ir::{AvailabilityRequirement, AvailabilitySummary},
    solver::Borrowck,
    summary::CallInputs,
};
use crate::analysis::{
    semantic::{
        SemOrigin,
        capability::{
            external::ExternalOrigin,
            guard::{Guard, ValueOccurrence},
            index::{BinderScope, IndexExpr},
            path::RegionPath,
            region::{OverlapResult, RegionRoot, RegionSet},
            source::{InputOrigin, SourceExpr},
            state::BorrowState,
        },
        normalized::{NBlockId, NStatementKind, NTerminatorKind, NValueId},
    },
    ty::{corelib::MemoryAccessKind, ty_is_copy},
};

impl<'db> AvailabilitySummary<'db> {
    pub fn empty() -> Self {
        Self {
            incoming: Vec::new(),
            reinitialized: RegionSet::empty(&BinderScope::default()),
            unavailable: RegionSet::empty(&BinderScope::default()),
        }
    }
}

#[derive(Clone)]
pub(super) struct ResolvedAvailability<'db> {
    pub incoming: Vec<(AvailabilityRequirement<'db>, NativeValidity<'db>)>,
    pub reinitialized: RegionSet<'db>,
    pub unavailable: RegionSet<'db>,
}

pub(super) struct AvailabilityAnalysis<'db> {
    pub summary: AvailabilitySummary<'db>,
    pub native_validity: NativeValidity<'db>,
    pub diagnostic: Option<SemanticDiagnostic<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MoveFact<'db> {
    origin: SemOrigin<'db>,
    region: RegionSet<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AvailabilityState<'db> {
    moved: BTreeMap<(usize, usize, usize), MoveFact<'db>>,
    initialized: RegionSet<'db>,
    guard: Guard<'db>,
}

impl<'db> AvailabilityState<'db> {
    fn new() -> Self {
        let scope = BinderScope::default();
        Self {
            moved: BTreeMap::new(),
            initialized: RegionSet::empty(&scope),
            guard: Guard::always(&scope),
        }
    }

    fn restrict(&mut self, guard: &Guard<'db>) -> Option<()> {
        self.guard = self.guard.and(guard)?;
        self.initialized = self.initialized.with_guard(&self.guard);
        for fact in self.moved.values_mut() {
            fact.region = fact.region.with_guard(&self.guard);
        }
        Some(())
    }

    fn join(&mut self, other: Self) {
        // Initialization is a must fact. Preserve alternatives only where their
        // path guards are disjoint; overlapping paths require both guarantees.
        let mut initialized = self.initialized.proven_intersection(&other.initialized);
        if let Some(guard) = self.guard.difference(&other.guard) {
            initialized = initialized.union(&self.initialized.with_guard(&guard));
        }
        if let Some(guard) = other.guard.difference(&self.guard) {
            initialized = initialized.union(&other.initialized.with_guard(&guard));
        }
        self.initialized = initialized;
        self.guard = self.guard.or(&other.guard);
        for (site, fact) in other.moved {
            self.moved
                .entry(site)
                .and_modify(|old| old.region = old.region.union(&fact.region))
                .or_insert(fact);
        }
    }

    fn initialize(&mut self, written: &RegionSet<'db>, exported: &RegionSet<'db>) {
        for fact in self.moved.values_mut() {
            fact.region = fact.region.remove_covered(written);
        }
        self.moved.retain(|_, fact| !fact.region.is_empty());
        self.initialized = self.initialized.union(exported);
    }

    fn consume(
        &mut self,
        site: (usize, usize, usize),
        region: RegionSet<'db>,
        origin: SemOrigin<'db>,
    ) {
        let region = region.with_guard(&self.guard);
        if !region.is_empty() {
            self.moved
                .entry(site)
                .and_modify(|fact| fact.region = fact.region.union(&region))
                .or_insert(MoveFact { origin, region });
        }
    }
}

impl<'db> Borrowck<'db> {
    pub fn call_availability(
        &mut self,
        state: &BorrowState<'db>,
        result: NValueId,
        inputs: CallInputs<'_, 'db>,
    ) -> Result<Option<ResolvedAvailability<'db>>, SemanticDiagnostic<'db>> {
        let Some(call) = self.calls.get(&result).cloned() else {
            return Ok(None);
        };
        let scope = BinderScope::default();
        let mut resolved = ResolvedAvailability {
            incoming: Vec::new(),
            reinitialized: RegionSet::empty(&scope),
            unavailable: RegionSet::empty(&scope),
        };
        // Every source is substituted against the same pre-call snapshot. A
        // guaranteed callee destination can still be ambiguous in its caller.
        for (region, kind, definite) in call
            .summary
            .availability
            .incoming
            .iter()
            .map(|requirement| (&requirement.region, Some(requirement.kind), false))
            .chain([
                (&call.summary.availability.reinitialized, None, true),
                (&call.summary.availability.unavailable, None, false),
            ])
        {
            let mut target_region = RegionSet::empty(&scope);
            let mut invalidated = NativeValidity::default();
            for clause in region.clauses() {
                let Some(guard) = self.instantiate_guard(&clause.guard, result, inputs)? else {
                    continue;
                };
                let source =
                    SourceExpr::from_place(&clause.payload).expect("verified availability source");
                if let ExternalOrigin::Input(input) = &source.source.origin
                    && matches!(input.origin(), InputOrigin::Place(_))
                    && input.dereferences().is_empty()
                    && source.source.dereferences().is_empty()
                    && let Some(arg) = inputs.args.get(input.param() as usize)
                    && state.value(arg.value).shape().direct(self.db).is_none()
                {
                    continue;
                }
                let target =
                    self.instantiate_source(state, &source, result, guard.scope(), inputs)?;
                invalidated |= target.invalidated;
                if kind.is_some_and(|kind| kind != MemoryAccessKind::Write) {
                    let ty = source.referent_ty(self.db, call.instance).ok_or_else(|| {
                        self.internal_diag(
                            inputs.origin,
                            "availability has no typed referent".into(),
                        )
                    })?;
                    let shape = self.shape(ty)?;
                    if shape.contains_capability(self.db) {
                        let contents = self.read_region(
                            state,
                            &target.region,
                            shape,
                            ValueOccurrence::Value(result),
                            inputs.origin,
                        )?;
                        invalidated |=
                            self.value_validity(&contents, ValueOccurrence::Value(result));
                    }
                }
                let target = target.region.with_guard(&guard).close_existentials(&scope);
                if !definite || target.definite_write().is_some() {
                    target_region = target_region.union(&target);
                }
            }
            if let Some(kind) = kind {
                resolved.incoming.push((
                    AvailabilityRequirement {
                        kind,
                        region: target_region,
                    },
                    invalidated,
                ));
            } else if definite {
                resolved.reinitialized = target_region;
            } else {
                resolved.unavailable = target_region;
            }
        }
        Ok(Some(resolved))
    }

    pub fn analyze_availability(&self) -> AvailabilityAnalysis<'db> {
        let mut entries = vec![None; self.body.blocks.len()];
        entries[self.body.entry.index()] = Some(AvailabilityState::new());
        loop {
            let mut changed = false;
            for (block_index, block) in self.body.blocks.iter().enumerate() {
                let Some(mut state) = entries[block_index].clone() else {
                    continue;
                };
                for (index, _) in self.before[block_index].iter().enumerate() {
                    self.transfer_availability(&mut state, block_index, index);
                }
                let Some(terminal) = &self.terminal[block_index] else {
                    continue;
                };
                for successor in block.terminator.kind.successors() {
                    let Some(guard) = self.edge_guard(block, successor) else {
                        continue;
                    };
                    let mut edge = state.clone();
                    if edge.restrict(&guard).is_none() {
                        continue;
                    }
                    for (index, access) in block
                        .terminator
                        .kind
                        .access(self.db, &self.body)
                        .into_iter()
                        .chain(successor.accesses(self.db, &self.body))
                        .enumerate()
                    {
                        let access = self.resolve_access(terminal, access, block.terminator.origin);
                        if access.kind == MemoryAccessKind::Move {
                            edge.consume(
                                (block_index, block.statements.len(), index),
                                access.region,
                                access.origin,
                            );
                        }
                    }
                    for (parameter, argument) in self.body.blocks[successor.block.index()]
                        .params
                        .iter()
                        .zip(&successor.args)
                    {
                        self.initialize_availability(
                            &mut edge,
                            &RegionSet::singleton(
                                &BinderScope::default(),
                                RegionRoot::Value(*parameter),
                                RegionPath::default(),
                            ),
                        );
                        for (site, fact) in &state.moved {
                            let region = RegionSet::new(
                                fact.region.scope(),
                                fact.region
                                    .clauses()
                                    .iter()
                                    .filter(|clause| {
                                        clause.payload.root == RegionRoot::Value(argument.value)
                                    })
                                    .cloned()
                                    .map(|mut clause| {
                                        clause.payload.root = RegionRoot::Value(*parameter);
                                        clause
                                    }),
                            );
                            edge.consume(*site, region, fact.origin);
                        }
                    }
                    if let Some(iteration) = self
                        .inventory
                        .loops
                        .feedback(NBlockId::new(block_index), successor.block)
                    {
                        let repeated = self.inventory.loops.repeated(iteration);
                        let repeats_index = |index| {
                            matches!(index, IndexExpr::Iteration(region) if region == iteration)
                                || matches!(index, IndexExpr::Runtime(value) if repeated.contains(&value))
                        };
                        let repeats_occurrence = |occurrence| {
                            self.inventory
                                .loops
                                .repeats_occurrence(iteration, occurrence)
                        };
                        for fact in edge.moved.values_mut() {
                            fact.region = fact.region.forget_iteration(
                                self.db,
                                repeats_index,
                                repeats_occurrence,
                            );
                        }
                        // Forgetting a witness is safe for may facts, but cannot
                        // turn a previous iteration's write into a must fact.
                        edge.initialized = RegionSet::new(
                            edge.initialized.scope(),
                            edge.initialized
                                .clauses()
                                .iter()
                                .filter(|clause| {
                                    !clause
                                        .guard
                                        .indices()
                                        .into_iter()
                                        .chain(clause.payload.root.indices())
                                        .chain(clause.payload.path.indices())
                                        .any(repeats_index)
                                        && !clause
                                            .guard
                                            .occurrences()
                                            .into_iter()
                                            .any(repeats_occurrence)
                                })
                                .cloned(),
                        );
                        edge.guard = edge
                            .guard
                            .forget_indices(repeats_index)
                            .forget_occurrences(repeats_occurrence);
                    }
                    edge.moved.retain(|_, fact| !fact.region.is_empty());
                    let entry = &mut entries[successor.block.index()];
                    if let Some(previous) = entry {
                        let old = previous.clone();
                        previous.join(edge);
                        changed |= *previous != old;
                    } else {
                        *entry = Some(edge);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut analysis = AvailabilityAnalysis {
            summary: AvailabilitySummary::empty(),
            native_validity: NativeValidity::default(),
            diagnostic: None,
        };
        let mut returned: Option<AvailabilityState<'db>> = None;
        for (block_index, block) in self.body.blocks.iter().enumerate() {
            let Some(mut state) = entries[block_index].clone() else {
                continue;
            };
            for (index, statement) in block
                .statements
                .iter()
                .take(self.before[block_index].len())
                .enumerate()
            {
                let operation = &self.operations[block_index][index];
                analysis.native_validity |= operation.native_validity.clone();
                for call in &operation.calls {
                    analysis.native_validity |= call.invalidated.clone();
                }
                if analysis.native_validity.invalid && analysis.diagnostic.is_none() {
                    analysis.diagnostic = Some(self.invalidated_diag(statement.origin));
                }
                for access in &operation.accesses {
                    self.require_access(&mut analysis, &state, access);
                }
                if let Some(call) = &operation.availability {
                    for (requirement, invalidated) in &call.incoming {
                        self.require_available(
                            &mut analysis,
                            &state,
                            requirement.kind,
                            &requirement.region,
                            statement.origin,
                        );
                        analysis.native_validity |= invalidated.clone();
                        if invalidated.invalid && analysis.diagnostic.is_none() {
                            analysis.diagnostic = Some(self.invalidated_diag(statement.origin));
                        }
                    }
                }
                self.transfer_availability(&mut state, block_index, index);
            }
            let Some(terminal) = &self.terminal[block_index] else {
                continue;
            };
            if let Some(access) = block.terminator.kind.access(self.db, &self.body) {
                let access = self.resolve_access(terminal, access, block.terminator.origin);
                self.require_access(&mut analysis, &state, &access);
                if access.kind == MemoryAccessKind::Move {
                    state.consume(
                        (block_index, block.statements.len(), 0),
                        access.region,
                        access.origin,
                    );
                }
            }
            for successor in block.terminator.kind.successors() {
                let Some(guard) = self.edge_guard(block, successor) else {
                    continue;
                };
                let mut edge = state.clone();
                if edge.restrict(&guard).is_none() {
                    continue;
                }
                for (index, access) in successor.accesses(self.db, &self.body).enumerate() {
                    let access = self.resolve_access(terminal, access, block.terminator.origin);
                    self.require_access(&mut analysis, &edge, &access);
                    if access.kind == MemoryAccessKind::Move {
                        edge.consume(
                            (block_index, block.statements.len(), index + 1),
                            access.region,
                            access.origin,
                        );
                    }
                }
            }
            if matches!(block.terminator.kind, NTerminatorKind::Return(_)) {
                if let Some(previous) = &mut returned {
                    previous.join(state);
                } else {
                    returned = Some(state);
                }
            }
        }
        if let Some(returned) = returned {
            analysis.summary.reinitialized = returned.initialized;
            for fact in returned.moved.values() {
                analysis.summary.unavailable = analysis.summary.unavailable.union(&fact.region);
            }
        }
        analysis.summary.incoming.sort();
        analysis.summary.incoming.dedup();
        analysis
    }

    fn initialize_availability(
        &self,
        state: &mut AvailabilityState<'db>,
        written: &RegionSet<'db>,
    ) {
        // Capability-free Copy storage cannot acquire its own moved fact. Byte
        // writes cannot reinitialize differently typed moved aggregates either.
        // Native slots still need restoration facts even when their handles are
        // Copy, so calls do not require validity before a guaranteed typed store.
        let exported = RegionSet::new(
            written.scope(),
            written
                .clauses()
                .iter()
                .filter(|clause| {
                    let RegionRoot::External(source) = &clause.payload.root else {
                        return false;
                    };
                    !matches!(source.origin, ExternalOrigin::Local(_))
                        && !source.is_fresh_allocation()
                        && (!ty_is_copy(
                            self.db,
                            self.instance
                                .key(self.db)
                                .impl_env(self.db)
                                .normalization_scope(self.db),
                            source.contract.ty,
                            self.instance.assumptions(self.db),
                        ) || self
                            .shape(source.contract.ty)
                            .expect("resolved storage has an admitted capability shape")
                            .contains_capability(self.db))
                })
                .cloned(),
        );
        state.initialize(written, &exported);
    }

    fn transfer_availability(
        &self,
        state: &mut AvailabilityState<'db>,
        block: usize,
        index: usize,
    ) {
        let statement = &self.body.blocks[block].statements[index];
        let operation = &self.operations[block][index];
        for (access_index, access) in operation.accesses.iter().enumerate() {
            if access.kind == MemoryAccessKind::Move {
                state.consume(
                    (block, index, access_index),
                    access.region.clone(),
                    access.origin,
                );
            }
        }
        for access in &operation.accesses {
            if access.kind == MemoryAccessKind::Write
                && let Some(write) = access.region.definite_write()
            {
                self.initialize_availability(state, write.region());
            }
        }
        if let Some(call) = &operation.availability {
            self.initialize_availability(state, &call.reinitialized);
            state.consume(
                (block, index, operation.accesses.len()),
                call.unavailable.clone(),
                statement.origin,
            );
        }
        if let NStatementKind::Define { result, .. } = statement.kind {
            self.initialize_availability(
                state,
                &RegionSet::singleton(
                    &BinderScope::default(),
                    RegionRoot::Value(result),
                    RegionPath::default(),
                ),
            );
        }
    }

    fn require_access(
        &self,
        analysis: &mut AvailabilityAnalysis<'db>,
        state: &AvailabilityState<'db>,
        access: &ResolvedAccess<'db>,
    ) {
        analysis.native_validity |= access.invalidated.clone();
        self.require_available(analysis, state, access.kind, &access.region, access.origin);
        if analysis.diagnostic.is_none() {
            if access.forbidden_move {
                analysis.diagnostic = Some(self.diag(
                    SemanticDiagnosticKind::MoveConflict,
                    access.origin,
                    "cannot move out of a view parameter or through a borrow handle".into(),
                ));
            } else if access.invalidated.invalid {
                analysis.diagnostic = Some(self.invalidated_diag(access.origin));
            }
        }
    }

    fn require_available(
        &self,
        analysis: &mut AvailabilityAnalysis<'db>,
        state: &AvailabilityState<'db>,
        kind: MemoryAccessKind,
        region: &RegionSet<'db>,
        origin: SemOrigin<'db>,
    ) {
        let region = region.with_guard(&state.guard);
        if analysis.diagnostic.is_none()
            && let Some(fact) = state.moved.values().find(|fact| {
                if kind != MemoryAccessKind::Write {
                    return !matches!(fact.region.overlap(&region), OverlapResult::Disjoint);
                }
                // Requirements are conjunctive. A whole-value write elsewhere
                // cannot authorize an earlier field write through a moved owner.
                region.clauses().iter().any(|write| {
                    let written = RegionSet::new(region.scope(), [write.clone()]);
                    fact.region.clauses().iter().any(|moved| {
                        let mut unavailable = RegionSet::new(fact.region.scope(), [moved.clone()]);
                        if write.guard.scope() == region.scope() {
                            unavailable = unavailable.with_guard(&write.guard);
                        }
                        !matches!(written.overlap(&unavailable), OverlapResult::Disjoint)
                            && !written.provably_covers(&unavailable)
                    })
                })
            })
        {
            let message = if kind == MemoryAccessKind::Write {
                "cannot write through a moved value"
            } else {
                "cannot use a value after it was moved"
            };
            let mut diagnostic =
                self.diag(SemanticDiagnosticKind::MoveConflict, origin, message.into());
            diagnostic.push_secondary(
                "value is moved here".into(),
                SemanticDiagnosticSpan::OriginWithTemplateFallback {
                    owner: self.instance.key(self.db).owner(self.db),
                    template_owner: self.body.template_owner,
                    origin: fact.origin,
                },
            );
            analysis.diagnostic = Some(diagnostic);
        }
        // Local reads still need diagnostics, but cannot be caller preconditions.
        // Fresh allocations cannot impose incoming obligations either. Filtering
        // them here also avoids subtracting unrelated byte-offset writes.
        let region = RegionSet::new(region.scope(), region.clauses().iter().filter(|clause| {
            matches!(&clause.payload.root, RegionRoot::External(source) if !matches!(source.origin, ExternalOrigin::Local(_)) && !source.is_fresh_allocation())
        }).cloned()).remove_covered(&state.initialized);
        if !region.is_empty() {
            analysis.summary.incoming.push(AvailabilityRequirement {
                kind: if kind == MemoryAccessKind::Write {
                    kind
                } else {
                    MemoryAccessKind::Read
                },
                region,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{
            semantic::capability::{source::InputSource, test_roots},
            ty::ty_check::BodyOwner,
        },
        test_db::{HirAnalysisTestDb, find_func},
    };

    #[test]
    fn ownership_join_and_composition_cover_two_cell_executions() {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone("availability_model.fe".into(), "fn inspect() {}");
        let (module, _) = db.top_mod(file);
        let origin = SemOrigin::Body(BodyOwner::Func(find_func(&db, module, "inspect")));
        let scope = BinderScope::default();
        let cells = [0, 1].map(|id| {
            RegionSet::singleton(
                &scope,
                RegionRoot::External(test_roots::input(&db, InputSource::place(id))),
                RegionPath::default(),
            )
        });
        let subset = |mask: u32| {
            cells
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .fold(RegionSet::empty(&scope), |set, (_, cell)| set.union(cell))
        };
        let programs: Vec<_> = [0_u32, 1, 2, 3]
            .into_iter()
            .flat_map(|written| {
                [0_u32, 1, 2, 3].map(|moved| {
                    let mut state = AvailabilityState::new();
                    for (index, cell) in cells.iter().enumerate() {
                        if written & (1 << index) != 0 {
                            state.initialize(cell, cell);
                        }
                        if moved & (1 << index) != 0 {
                            state.consume((0, 0, index), cell.clone(), origin);
                        }
                    }
                    (written, moved, state)
                })
            })
            .collect();
        for (left_written, left_moved, left) in &programs {
            for (right_written, right_moved, right) in &programs {
                let mut joined = left.clone();
                joined.join(right.clone());
                for before in [0_u32, 1, 2, 3] {
                    let mut abstract_join = subset(before).remove_covered(&joined.initialized);
                    for fact in joined.moved.values() {
                        abstract_join = abstract_join.union(&fact.region);
                    }
                    let concrete_left = (before & !left_written) | left_moved;
                    let concrete_right = (before & !right_written) | right_moved;
                    assert!(abstract_join.provably_covers(&subset(concrete_left | concrete_right)));

                    let mut composed = subset(before).remove_covered(&left.initialized);
                    for fact in left.moved.values() {
                        composed = composed.union(&fact.region);
                    }
                    composed = composed.remove_covered(&right.initialized);
                    for fact in right.moved.values() {
                        composed = composed.union(&fact.region);
                    }
                    let concrete_composed = (concrete_left & !right_written) | right_moved;
                    assert_eq!(composed, subset(concrete_composed));
                }
            }
        }
    }
}

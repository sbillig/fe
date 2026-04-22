use std::convert::Infallible;

use common::diagnostics::CompleteDiagnostic;
use cranelift_entity::{EntityRef, SecondaryMap};
use dataflow::{BackwardCfgAnalysis, ForwardCfgAnalysis, JoinSemiLattice, SparseAnalysis};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::{
    diagnostics::SpannedHirAnalysisDb,
    semantic::{
        SBlockId, SLocalId, SemanticInstance,
        borrowck::ir::{NEffectArgValue, NExpr, NSStmtKind},
        get_or_build_semantic_instance,
    },
    ty::ty_check::EffectPassMode,
};

use super::{
    canon::{BorrowCanonCx, CanonPlace, CfgAdjacency, Loan, LoanId, MovedPlaces, State},
    check::{Borrowck, semantic_borrow_summary},
    ir::{BorrowInputRef, NormalizedSemanticBody},
};

pub(super) struct BorrowLoanTargetState<'a, 'db> {
    pub(super) loans: &'a mut [Loan<'db>],
}

pub(super) struct BorrowLoanTargetAnalysis<'a, 'db> {
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &'a NormalizedSemanticBody<'db>,
    entry_state: &'a SecondaryMap<SBlockId, State>,
    loan_for_local: &'a FxHashMap<SLocalId, LoanId>,
}

impl<'a, 'db> BorrowLoanTargetAnalysis<'a, 'db> {
    pub(super) fn new(
        db: &'db dyn SpannedHirAnalysisDb,
        instance: SemanticInstance<'db>,
        body: &'a NormalizedSemanticBody<'db>,
        entry_state: &'a SecondaryMap<SBlockId, State>,
        loan_for_local: &'a FxHashMap<SLocalId, LoanId>,
    ) -> Self {
        Self {
            db,
            instance,
            body,
            entry_state,
            loan_for_local,
        }
    }

    fn canon<'b>(&'b self, loans: &'b [Loan<'db>]) -> BorrowCanonCx<'b, 'db> {
        BorrowCanonCx::new(
            self.db,
            self.instance,
            self.body,
            loans,
            self.loan_for_local,
        )
    }

    fn extend_loan(
        &self,
        loans: &mut [Loan<'db>],
        loan_id: LoanId,
        targets: FxHashSet<CanonPlace<'db>>,
        parents: FxHashSet<LoanId>,
    ) -> bool {
        let loan = &mut loans[loan_id.0 as usize];
        let before_targets = loan.targets.len();
        let before_parents = loan.parents.len();
        loan.targets.extend(targets);
        loan.parents.extend(parents);
        before_targets != loan.targets.len() || before_parents != loan.parents.len()
    }

    fn update_loan_from_stmt(
        &self,
        loans: &mut [Loan<'db>],
        state: &State,
        stmt: &super::ir::NSStmt<'db>,
    ) -> Result<bool, CompleteDiagnostic> {
        let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
            return Ok(false);
        };
        let Some(&loan_id) = self.loan_for_local.get(dst) else {
            return Ok(false);
        };
        match expr {
            NExpr::Borrow { place, .. } => {
                let (targets, parents) = {
                    let canon = self.canon(loans);
                    (
                        canon.canonicalize_place(state, place, stmt.origin)?,
                        canon.mut_loans_for_place(state, place),
                    )
                };
                Ok(self.extend_loan(loans, loan_id, targets, parents))
            }
            NExpr::Call {
                callee,
                args,
                effect_args,
            } => {
                let summary = semantic_borrow_summary(
                    self.db,
                    get_or_build_semantic_instance(self.db, callee.key),
                )?;
                let Some(summary) = summary else {
                    return Ok(false);
                };
                let (targets, parents) = {
                    let canon = self.canon(loans);
                    let mut targets = FxHashSet::default();
                    let mut parents = FxHashSet::default();
                    for transform in &summary {
                        match transform.input {
                            BorrowInputRef::Param(idx) => {
                                if let Some(arg) = args.get(idx as usize) {
                                    for base in canon.canonicalize_value_base(state, arg.local) {
                                        targets.insert(CanonPlace {
                                            root: base.root,
                                            proj: base.proj.concat(&transform.proj),
                                        });
                                    }
                                    parents.extend(canon.mut_loans_for_value(state, arg.local));
                                }
                            }
                            BorrowInputRef::EffectArg(idx) => {
                                let Some(effect_arg) = effect_args.get(idx as usize) else {
                                    continue;
                                };
                                if matches!(effect_arg.pass_mode, EffectPassMode::ByPlace)
                                    && let NEffectArgValue::Place(place) = &effect_arg.arg
                                {
                                    for base in
                                        canon.canonicalize_place(state, place, stmt.origin)?
                                    {
                                        targets.insert(CanonPlace {
                                            root: base.root,
                                            proj: base.proj.concat(&transform.proj),
                                        });
                                    }
                                    parents.extend(canon.mut_loans_for_place(state, place));
                                }
                            }
                        }
                    }
                    (targets, parents)
                };
                Ok(self.extend_loan(loans, loan_id, targets, parents))
            }
            _ => Ok(false),
        }
    }
}

impl<'a, 'db> SparseAnalysis for BorrowLoanTargetAnalysis<'a, 'db> {
    type Node = SBlockId;
    type State = BorrowLoanTargetState<'a, 'db>;
    type Error = CompleteDiagnostic;

    fn node_count(&self) -> usize {
        self.body.blocks.len()
    }

    fn seed_nodes(&self) -> Vec<Self::Node> {
        (0..self.body.blocks.len()).map(SBlockId::new).collect()
    }

    fn step(&mut self, node: Self::Node, state: &mut Self::State) -> Result<bool, Self::Error> {
        let mut local_state = self.entry_state[node].clone();
        let mut changed = false;
        for stmt in &self.body.blocks[node.index()].stmts {
            changed |= self.update_loan_from_stmt(&mut *state.loans, &local_state, stmt)?;
            self.canon(state.loans)
                .apply_stmt_state(&mut local_state, stmt);
        }
        Ok(changed)
    }

    fn dependents(&self, _node: Self::Node, out: &mut Vec<Self::Node>) {
        out.extend((0..self.body.blocks.len()).map(SBlockId::new));
    }
}

pub(super) struct BorrowEntryStateAnalysis<'a, 'db> {
    borrowck: &'a Borrowck<'db>,
    successors: CfgAdjacency,
}

impl<'a, 'db> BorrowEntryStateAnalysis<'a, 'db> {
    pub(super) fn new(borrowck: &'a Borrowck<'db>) -> Self {
        Self {
            borrowck,
            successors: borrowck.cfg_successor_indices(),
        }
    }
}

impl ForwardCfgAnalysis for BorrowEntryStateAnalysis<'_, '_> {
    type Block = SBlockId;
    type State = State;
    type Error = Infallible;

    fn block_count(&self) -> usize {
        self.borrowck.body.blocks.len()
    }

    fn seed_blocks(&self) -> Vec<Self::Block> {
        (!self.borrowck.body.blocks.is_empty())
            .then_some(SBlockId::new(0))
            .into_iter()
            .collect()
    }

    fn bottom(&self) -> Self::State {
        State::default()
    }

    fn initialize(
        &mut self,
        entry_states: &mut SecondaryMap<Self::Block, Self::State>,
    ) -> Result<(), Self::Error> {
        if !self.borrowck.body.blocks.is_empty() {
            let entry = &mut entry_states[SBlockId::new(0)];
            for (&local, &loan) in &self.borrowck.param_loan_for_local {
                entry.assign_loans(local, FxHashSet::from_iter([loan]));
            }
        }
        Ok(())
    }

    fn transfer(
        &mut self,
        block: Self::Block,
        in_state: &Self::State,
    ) -> Result<Self::State, Self::Error> {
        let mut state = in_state.clone();
        for stmt in &self.borrowck.body.blocks[block.index()].stmts {
            self.borrowck.canon().apply_stmt_state(&mut state, stmt);
        }
        Ok(state)
    }

    fn successors(&self, block: Self::Block) -> &[Self::Block] {
        &self.successors[block]
    }
}

#[derive(Clone, Default)]
pub(super) struct MovedState<'db>(pub(super) MovedPlaces<'db>);

impl JoinSemiLattice for MovedState<'_> {
    fn join_into(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (place, site) in &other.0 {
            changed |= self.0.insert(place.clone(), site.clone()).is_none();
        }
        changed
    }
}

pub(super) struct BorrowMovedStateAnalysis<'a, 'db> {
    borrowck: &'a Borrowck<'db>,
    successors: CfgAdjacency,
}

impl<'a, 'db> BorrowMovedStateAnalysis<'a, 'db> {
    pub(super) fn new(borrowck: &'a Borrowck<'db>) -> Self {
        Self {
            borrowck,
            successors: borrowck.cfg_successor_indices(),
        }
    }
}

impl<'db> ForwardCfgAnalysis for BorrowMovedStateAnalysis<'_, 'db> {
    type Block = SBlockId;
    type State = MovedState<'db>;
    type Error = CompleteDiagnostic;

    fn block_count(&self) -> usize {
        self.borrowck.body.blocks.len()
    }

    fn seed_blocks(&self) -> Vec<Self::Block> {
        (!self.borrowck.body.blocks.is_empty())
            .then_some(SBlockId::new(0))
            .into_iter()
            .collect()
    }

    fn bottom(&self) -> Self::State {
        MovedState::default()
    }

    fn transfer(
        &mut self,
        block: Self::Block,
        in_state: &Self::State,
    ) -> Result<Self::State, Self::Error> {
        let mut state = self.borrowck.entry_state[block].clone();
        let mut moved = in_state.0.clone();
        for stmt in &self.borrowck.body.blocks[block.index()].stmts {
            self.borrowck
                .update_moved_for_stmt(&state, &mut moved, stmt)?;
            self.borrowck.canon().apply_stmt_state(&mut state, stmt);
        }
        Ok(MovedState(moved))
    }

    fn successors(&self, block: Self::Block) -> &[Self::Block] {
        &self.successors[block]
    }
}

#[derive(Clone, Default)]
pub(super) struct LiveSet(pub(super) FxHashSet<SLocalId>);

impl JoinSemiLattice for LiveSet {
    fn join_into(&mut self, other: &Self) -> bool {
        let before = self.0.len();
        self.0.extend(other.0.iter().copied());
        before != self.0.len()
    }
}

pub(super) struct BorrowLivenessAnalysis<'a, 'db> {
    borrowck: &'a Borrowck<'db>,
    predecessors: CfgAdjacency,
}

impl<'a, 'db> BorrowLivenessAnalysis<'a, 'db> {
    pub(super) fn new(borrowck: &'a Borrowck<'db>) -> Self {
        Self {
            borrowck,
            predecessors: borrowck.cfg_predecessor_indices(),
        }
    }
}

impl<'db> BackwardCfgAnalysis for BorrowLivenessAnalysis<'_, 'db> {
    type Block = SBlockId;
    type State = LiveSet;

    fn block_count(&self) -> usize {
        self.borrowck.body.blocks.len()
    }

    fn seed_blocks(&self) -> Vec<Self::Block> {
        (0..self.borrowck.body.blocks.len())
            .map(SBlockId::new)
            .collect()
    }

    fn bottom(&self) -> Self::State {
        LiveSet::default()
    }

    fn initialize(&mut self, _exit_states: &mut SecondaryMap<Self::Block, Self::State>) {}

    fn transfer(&mut self, block: Self::Block, out_state: &Self::State) -> Self::State {
        let block_data = &self.borrowck.body.blocks[block.index()];
        let mut live = out_state.0.clone();
        live.extend(self.borrowck.facts.terminator_uses(block));
        for (stmt_idx, _) in block_data.stmts.iter().enumerate().rev() {
            live = self.borrowck.live_before_stmt(block, stmt_idx, &live);
        }
        LiveSet(live)
    }

    fn predecessors(&self, block: Self::Block) -> &[Self::Block] {
        &self.predecessors[block]
    }
}

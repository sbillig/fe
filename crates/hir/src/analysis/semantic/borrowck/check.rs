use std::collections::VecDeque;

use common::diagnostics::CompleteDiagnostic;
use cranelift_entity::{EntityRef, SecondaryMap};
use dataflow::{solve_backward_cfg, solve_forward_cfg, try_solve_forward_cfg, try_solve_sparse};
use num_traits::ToPrimitive;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::{
        HirAnalysisDb,
        analysis_pass::ModuleAnalysisPass,
        diagnostics::{DiagnosticVoucher, SpannedHirAnalysisDb},
        semantic::{
            BorrowActivation, LayoutBackingProjection, SBlockId, SConst, SStmtId, SemConstScalar,
            SemConstValue, SemOrigin, SemanticInstance, SemanticInstanceKey,
            get_or_build_semantic_instance, identity_semantic_instance_key,
        },
        ty::{
            ty_check::{BodyOwner, EffectParamSite, LocalBinding},
            ty_contains_borrow,
            ty_def::{BorrowKind, TyId},
        },
    },
    core::semantic::EffectEnvView,
    hir_def::{Body, Expr, FuncParamMode, ItemKind, Partial, TopLevelMod},
    projection::Projection,
};

use super::{
    analyses::{
        BorrowEntryStateAnalysis, BorrowLivenessAnalysis, BorrowLoanTargetAnalysis,
        BorrowLoanTargetState, BorrowMovedStateAnalysis, BorrowSummaryMode,
    },
    canon::{
        BlockAdjacency, BorrowCanonCx, BorrowRoot, CanonPlace, CfgAdjacency, Loan, LoanId,
        MoveSite, MovedPlaces, State, place_set_overlaps, places_overlap,
    },
    diagnostics::operand_origin,
    facts::NormalizedBodyFacts,
    ir::{
        BorrowDiagnosticId, BorrowInput, BorrowResult, BorrowSummary, BorrowSummaryId,
        BorrowTransform, NBorrowRoot, NBorrowRootId, NEffectArgValue, NExpr, NOperand, NSPlace,
        NSPlaceRoot, NSProjectionPath, NSStmtKind, NSTerminatorKind, NormalizedBindingLowering,
        NormalizedSemanticBody, ReadMode, SemanticBorrowCheckResult, SemanticBorrowDiagKind,
        SemanticBorrowDiagnostic, SemanticBorrowDiagnosticSpan, SemanticBorrowSummaryResult,
        borrow_results_in_ty, layout_path_for_semantic_projection,
        local_has_runtime_move_semantics,
    },
    normalize::{normalize_provisional_semantic_body, normalize_semantic_body_for_analysis},
    verify::verify_normalized_semantic_body,
};

#[salsa::tracked(
    cycle_fn=semantic_borrow_summary_cycle_recover,
    cycle_initial=semantic_borrow_summary_cycle_initial
)]
fn semantic_borrow_summary_query<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowSummaryResult<'db> {
    if !instance_returns_borrowing_value(db, instance) {
        return SemanticBorrowSummaryResult::Ok(None);
    }
    if instance.key(db).owner(db).body(db).is_none() {
        return SemanticBorrowSummaryResult::Ok(Some(BorrowSummaryId::new(
            db,
            conservative_signature_borrow_summary(db, instance),
        )));
    }
    match Borrowck::new_for_summary(db, instance).and_then(Borrowck::borrow_summary) {
        Ok(summary) => SemanticBorrowSummaryResult::Ok(
            summary.map(|summary| BorrowSummaryId::new(db, summary)),
        ),
        Err(diag) => SemanticBorrowSummaryResult::Err(BorrowDiagnosticId::new(db, diag)),
    }
}

#[salsa::tracked(
    cycle_fn=semantic_borrow_summary_cycle_recover,
    cycle_initial=semantic_borrow_summary_cycle_initial
)]
fn provisional_borrow_summary_query<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowSummaryResult<'db> {
    if !instance_returns_borrowing_value(db, instance) {
        return SemanticBorrowSummaryResult::Ok(None);
    }
    if instance.key(db).owner(db).body(db).is_none() {
        return SemanticBorrowSummaryResult::Ok(Some(BorrowSummaryId::new(
            db,
            conservative_signature_borrow_summary(db, instance),
        )));
    }
    let body = match normalize_provisional_semantic_body(db, instance) {
        Ok(body) => body,
        Err(diag) => return SemanticBorrowSummaryResult::Err(BorrowDiagnosticId::new(db, diag)),
    };
    match Borrowck::new_with_body(db, instance, body, BorrowSummaryMode::Provisional)
        .and_then(Borrowck::borrow_summary)
    {
        Ok(summary) => SemanticBorrowSummaryResult::Ok(
            summary.map(|summary| BorrowSummaryId::new(db, summary)),
        ),
        Err(diag) => SemanticBorrowSummaryResult::Err(BorrowDiagnosticId::new(db, diag)),
    }
}

pub fn semantic_borrow_summary<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<Option<BorrowSummary>, CompleteDiagnostic> {
    semantic_borrow_summary_voucher(db, instance).map_err(|diag| diag.to_complete(db))
}

pub(super) fn semantic_borrow_summary_voucher<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<Option<BorrowSummary>, SemanticBorrowDiagnostic<'db>> {
    match semantic_borrow_summary_query(db, instance) {
        SemanticBorrowSummaryResult::Ok(summary) => {
            Ok(summary.map(|summary| summary.items(db).clone()))
        }
        SemanticBorrowSummaryResult::Err(diag) => Err(diag.diag(db).clone()),
    }
}

pub(super) fn provisional_borrow_summary_voucher<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<Option<BorrowSummary>, SemanticBorrowDiagnostic<'db>> {
    match provisional_borrow_summary_query(db, instance) {
        SemanticBorrowSummaryResult::Ok(summary) => {
            Ok(summary.map(|summary| summary.items(db).clone()))
        }
        SemanticBorrowSummaryResult::Err(diag) => Err(diag.diag(db).clone()),
    }
}

pub fn check_semantic_borrows<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<(), CompleteDiagnostic> {
    match semantic_borrow_check_query(db, instance) {
        SemanticBorrowCheckResult::Ok => Ok(()),
        SemanticBorrowCheckResult::Err(diag) => Err(diag.to_complete(db)),
    }
}

#[salsa::tracked]
fn semantic_borrow_check_query<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowCheckResult<'db> {
    match Borrowck::new(db, instance).and_then(Borrowck::check) {
        Ok(()) => SemanticBorrowCheckResult::Ok,
        Err(diag) => SemanticBorrowCheckResult::Err(BorrowDiagnosticId::new(db, diag)),
    }
}

pub struct SemanticBorrowAnalysisPass;

impl ModuleAnalysisPass for SemanticBorrowAnalysisPass {
    fn run_on_module<'db>(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        top_mod: TopLevelMod<'db>,
    ) -> Vec<Box<dyn DiagnosticVoucher + 'db>> {
        collect_semantic_borrow_diagnostic_vouchers(db, top_mod)
    }
}

pub fn collect_semantic_borrow_diagnostic_vouchers<'db>(
    db: &'db dyn HirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Vec<Box<dyn DiagnosticVoucher + 'db>> {
    let mut diags = Vec::new();
    let mut pending = VecDeque::new();
    let mut seen_diags = FxHashSet::default();
    collect_top_mod_semantic_borrow_diagnostic_vouchers(db, top_mod, &mut pending);
    let mut seen_instances = FxHashSet::default();
    while let Some(instance) = pending.pop_front() {
        if !seen_instances.insert(instance.key(db)) {
            continue;
        }
        collect_instance(db, instance, &mut pending, &mut seen_diags, &mut diags);
    }
    diags
}

fn collect_top_mod_semantic_borrow_diagnostic_vouchers<'db>(
    db: &'db dyn HirAnalysisDb,
    top_mod: TopLevelMod<'db>,
    pending: &mut VecDeque<SemanticInstance<'db>>,
) {
    for item in top_mod
        .all_items(db)
        .iter()
        .filter(|item| item.top_mod(db) == top_mod)
    {
        match item {
            ItemKind::Func(func) => pending.push_back(get_or_build_semantic_instance(
                db,
                identity_semantic_instance_key(db, BodyOwner::Func(*func)),
            )),
            ItemKind::Const(const_) => pending.push_back(get_or_build_semantic_instance(
                db,
                identity_semantic_instance_key(db, BodyOwner::Const(*const_)),
            )),
            ItemKind::Contract(contract) => {
                pending.push_back(get_or_build_semantic_instance(
                    db,
                    identity_semantic_instance_key(
                        db,
                        BodyOwner::ContractInit {
                            contract: *contract,
                        },
                    ),
                ));
                for (recv_idx, recv) in contract.recvs(db).data(db).iter().enumerate() {
                    for arm_idx in 0..recv.arms.data(db).len() {
                        pending.push_back(get_or_build_semantic_instance(
                            db,
                            identity_semantic_instance_key(
                                db,
                                BodyOwner::ContractRecvArm {
                                    contract: *contract,
                                    recv_idx: recv_idx as u32,
                                    arm_idx: arm_idx as u32,
                                },
                            ),
                        ));
                    }
                }
            }
            ItemKind::Mod(_)
            | ItemKind::Struct(_)
            | ItemKind::Enum(_)
            | ItemKind::Trait(_)
            | ItemKind::Impl(_)
            | ItemKind::ImplTrait(_)
            | ItemKind::TypeAlias(_)
            | ItemKind::StaticAssert(_)
            | ItemKind::Use(_)
            | ItemKind::TopMod(_)
            | ItemKind::Body(_) => {}
        }
    }
}

fn collect_instance<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    pending: &mut VecDeque<SemanticInstance<'db>>,
    seen_diags: &mut FxHashSet<BorrowDiagnosticId<'db>>,
    diags: &mut Vec<Box<dyn DiagnosticVoucher + 'db>>,
) {
    let key = instance.key(db);
    let owner = key.owner(db);
    let typed_body = key.typed_body(db);
    let has_closures = typed_body.closure_infos().next().is_some();
    let is_identity = key == identity_semantic_instance_key(db, owner);
    if is_identity || has_closures || matches!(owner, BodyOwner::Closure { .. }) {
        if let SemanticBorrowCheckResult::Err(diag) = semantic_borrow_check_query(db, instance)
            && seen_diags.insert(diag)
        {
            diags.push(Box::new(diag));
        }
        if let super::ir::SemanticBorrowCheckResult::Err(diag) =
            super::noesc::semantic_noesc_check_query(db, instance)
            && seen_diags.insert(diag)
        {
            diags.push(Box::new(diag));
        }
    }
    if has_closures && !matches!(owner, BodyOwner::Closure { .. }) && owner.body(db).is_some() {
        for (expr, _) in typed_body.closure_infos() {
            if let Some(closure_ty) = typed_body.expr_ty(db, expr).as_closure(db) {
                pending.push_back(get_or_build_semantic_instance(
                    db,
                    identity_semantic_instance_key(db, BodyOwner::closure(db, closure_ty)),
                ));
            }
        }
    }
    // Address spaces supplied by effect handles exist only on finalized callee
    // instances. Walk through every reachable specialization because an ordinary
    // generic wrapper may sit between the root and a closure-bearing effect call.
    // Unrelated monomorphizations are left to their parametric identity-owner
    // check above.
    pending.extend(
        instance
            .callees(db)
            .iter()
            .filter(|callee| is_fully_instantiated_key(db, callee.key))
            .map(|callee| get_or_build_semantic_instance(db, callee.key)),
    );
}

fn is_fully_instantiated_key<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
) -> bool {
    let args_are_concrete = key
        .subst(db)
        .generic_args(db)
        .iter()
        .all(|arg| !arg.has_param(db) && !arg.has_var(db));
    let providers = key.effect_providers(db).providers(db);
    let providers_are_concrete = providers.iter().all(|specialization| {
        let provider = &specialization.provider;
        [
            provider.provider_ty,
            provider.semantics.provider_ty,
            provider.effective_target_ty(),
        ]
        .into_iter()
        .all(|ty| !ty.has_param(db) && !ty.has_var(db))
    });
    let has_all_effect_providers = match key.owner(db) {
        BodyOwner::Func(func) => {
            EffectEnvView::new(EffectParamSite::Func(func))
                .requirements(db)
                .is_empty()
                || !providers.is_empty()
        }
        _ => true,
    };
    args_are_concrete && providers_are_concrete && has_all_effect_providers
}

pub(super) struct Borrowck<'db> {
    pub(super) db: &'db dyn HirAnalysisDb,
    pub(super) instance: SemanticInstance<'db>,
    pub(super) body: NormalizedSemanticBody<'db>,
    pub(super) facts: NormalizedBodyFacts,
    pub(super) summary_mode: BorrowSummaryMode,
    hir_body: Option<Body<'db>>,
    param_modes: Vec<FuncParamMode>,
    param_index_of_local: FxHashMap<crate::analysis::semantic::SLocalId, u32>,
    pub(super) loan_for_local: FxHashMap<crate::analysis::semantic::SLocalId, LoanId>,
    pub(super) param_loan_for_local: FxHashMap<crate::analysis::semantic::SLocalId, LoanId>,
    loans: Vec<Loan<'db>>,
    pub(super) entry_state: SecondaryMap<SBlockId, State>,
    call_result_loans: FxHashMap<SStmtId, Vec<(BorrowResult, LoanId)>>,
    call_loan_transforms: FxHashMap<LoanId, Vec<BorrowTransform>>,
    constant_indices: SecondaryMap<crate::analysis::semantic::SLocalId, Option<usize>>,
    moved_entry: SecondaryMap<SBlockId, MovedPlaces<'db>>,
    live_before: Vec<Vec<FxHashSet<crate::analysis::semantic::SLocalId>>>,
    live_before_term: SecondaryMap<SBlockId, FxHashSet<crate::analysis::semantic::SLocalId>>,
}

struct CallAccess<'db> {
    group: usize,
    role: CallAccessRole,
    kind: BorrowKind,
    targets: FxHashSet<CanonPlace<'db>>,
    origin: SemOrigin<'db>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallAccessRole {
    Container,
    BorrowSlot,
}

impl<'db> Borrowck<'db> {
    pub(super) fn new(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
    ) -> Result<Self, SemanticBorrowDiagnostic<'db>> {
        let body = normalize_semantic_body_for_analysis(db, instance)?;
        Self::new_with_body(db, instance, body, BorrowSummaryMode::FinalCheck)
    }

    fn new_for_summary(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
    ) -> Result<Self, SemanticBorrowDiagnostic<'db>> {
        let body = normalize_semantic_body_for_analysis(db, instance)?;
        Self::new_with_body(db, instance, body, BorrowSummaryMode::FinalSummary)
    }

    pub(super) fn new_with_body(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
        body: NormalizedSemanticBody<'db>,
        summary_mode: BorrowSummaryMode,
    ) -> Result<Self, SemanticBorrowDiagnostic<'db>> {
        verify_normalized_semantic_body(db, instance, &body)?;
        let owner = instance.key(db).owner(db);
        let param_modes = instance
            .key(db)
            .callable_body(db)
            .param_bindings(db)
            .into_iter()
            .map(|binding| match binding {
                crate::analysis::ty::ty_check::LocalBinding::Param { mode, .. } => mode,
                crate::analysis::ty::ty_check::LocalBinding::Local { .. }
                | crate::analysis::ty::ty_check::LocalBinding::EffectParam { .. } => {
                    FuncParamMode::Own
                }
            })
            .collect();
        let mut param_index_of_local = FxHashMap::default();
        for root_id in 0..body.borrow_roots.len() {
            let root_id = NBorrowRootId::from_u32(root_id as u32);
            match body.root(root_id).expect("borrow root") {
                NBorrowRoot::Param { local, param_idx } => {
                    param_index_of_local.insert(*local, *param_idx);
                }
                NBorrowRoot::Provider { .. } | NBorrowRoot::LocalSlot { .. } => {}
            }
        }
        let facts = NormalizedBodyFacts::new(&body);
        let mut constant_candidates = FxHashMap::default();
        let mut stored_locals = FxHashSet::default();
        for stmt in body.blocks.iter().flat_map(|block| &block.stmts) {
            match &stmt.kind {
                NSStmtKind::Assign { dst, expr } => {
                    let value = match expr {
                        NExpr::Const(SConst::Value(value)) => match value.value(db) {
                            SemConstValue::Scalar {
                                value: SemConstScalar::Int { value },
                                ..
                            } => value.to_usize(),
                            _ => None,
                        },
                        _ => None,
                    };
                    constant_candidates
                        .entry(*dst)
                        .and_modify(|candidate| {
                            if *candidate != value {
                                *candidate = None;
                            }
                        })
                        .or_insert(value);
                }
                NSStmtKind::Store {
                    dst:
                        NSPlace {
                            root: NSPlaceRoot::Root(root),
                            ..
                        },
                    ..
                } => match body.root(*root) {
                    Some(NBorrowRoot::Param { local, .. })
                    | Some(NBorrowRoot::LocalSlot { local }) => {
                        stored_locals.insert(*local);
                    }
                    Some(NBorrowRoot::Provider { .. }) | None => {}
                },
                NSStmtKind::Store { .. } => {}
            }
        }
        let mut constant_indices = SecondaryMap::new();
        constant_indices.resize(body.locals.len());
        for (local, value) in constant_candidates {
            if !stored_locals.contains(&local) {
                constant_indices[local] = value;
            }
        }
        let mut checker = Self {
            db,
            instance,
            hir_body: owner.body(db),
            body,
            facts,
            summary_mode,
            param_modes,
            param_index_of_local,
            loan_for_local: FxHashMap::default(),
            param_loan_for_local: FxHashMap::default(),
            loans: Vec::new(),
            entry_state: SecondaryMap::new(),
            call_result_loans: FxHashMap::default(),
            call_loan_transforms: FxHashMap::default(),
            constant_indices,
            moved_entry: SecondaryMap::new(),
            live_before: Vec::new(),
            live_before_term: SecondaryMap::new(),
        };
        checker.init_loans()?;
        Ok(checker)
    }

    pub(super) fn canon(&self) -> BorrowCanonCx<'_, 'db> {
        BorrowCanonCx::new(
            self.db,
            self.instance,
            &self.body,
            &self.loans,
            &self.loan_for_local,
            &self.constant_indices,
        )
    }

    fn borrow_summary(mut self) -> Result<Option<BorrowSummary>, SemanticBorrowDiagnostic<'db>> {
        let key = self.instance.key(self.db);
        if !ty_contains_borrow(self.db, key.callable_body(self.db).result_ty(self.db))
            || key.owner(self.db).body(self.db).is_none()
        {
            return Ok(None);
        }
        self.compute_entry_states();
        self.compute_loan_targets()?;
        self.compute_return_summary().map(Some)
    }

    fn check(mut self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        self.compute_entry_states();
        self.compute_loan_targets()?;
        self.compute_moved_states()?;
        self.compute_liveness();
        self.check_conflicts()?;
        let key = self.instance.key(self.db);
        if ty_contains_borrow(self.db, key.callable_body(self.db).result_ty(self.db)) {
            let _ = self.compute_return_summary()?;
        }
        Ok(())
    }

    fn compute_liveness(&mut self) {
        let live_out = solve_backward_cfg(&mut BorrowLivenessAnalysis::new(self));
        self.live_before = self
            .body
            .blocks
            .iter()
            .map(|block| vec![FxHashSet::default(); block.stmts.len()])
            .collect();
        self.live_before_term = SecondaryMap::new();
        self.live_before_term.resize(self.body.blocks.len());

        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            let bb = SBlockId::new(bb_idx);
            let mut live = live_out[bb].0.clone();
            live.extend(self.facts.terminator_uses(bb));
            self.live_before_term[bb] = live.clone();
            for (stmt_idx, _) in block.stmts.iter().enumerate().rev() {
                live = self.live_before_stmt(bb, stmt_idx, &live);
                self.live_before[bb_idx][stmt_idx] = live.clone();
            }
        }
    }

    pub(super) fn live_before_stmt(
        &self,
        block: SBlockId,
        stmt_idx: usize,
        live_after: &FxHashSet<crate::analysis::semantic::SLocalId>,
    ) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        let mut live = live_after.clone();
        let stmt = &self.body.blocks[block.index()].stmts[stmt_idx];
        match &stmt.kind {
            NSStmtKind::Assign { dst, .. } => {
                live.remove(dst);
                live.extend(self.facts.stmt_uses(block, stmt_idx));
            }
            NSStmtKind::Store { .. } => live.extend(self.facts.stmt_uses(block, stmt_idx)),
        }
        live
    }

    fn init_loans(&mut self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        for local_id in 0..self.body.locals.len() {
            let local_id = crate::analysis::semantic::SLocalId::from_u32(local_id as u32);
            let Some(local) = self.body.local(local_id) else {
                continue;
            };
            if let Some((kind, _)) = local.ty.as_borrow(self.db)
                && let Some(&param_idx) = self.param_index_of_local.get(&local_id)
                && !matches!(
                    local.lowering,
                    NormalizedBindingLowering::CarrierLocal { .. }
                )
            {
                let mut targets = FxHashSet::default();
                targets.insert(CanonPlace {
                    root: BorrowRoot::Param(param_idx),
                    proj: NSProjectionPath::default(),
                });
                let loan = self.allocate_loan(Loan {
                    kind,
                    activation: BorrowActivation::Immediate,
                    targets,
                    parents: FxHashSet::default(),
                    origin: crate::analysis::semantic::SemOrigin::Body(self.body.template_owner),
                });
                self.param_loan_for_local.insert(local_id, loan);
            }
        }

        let stmts = self
            .body
            .blocks
            .iter()
            .flat_map(|block| block.stmts.iter().cloned())
            .collect::<Vec<_>>();
        for stmt in stmts {
            let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
                continue;
            };
            let Some(result_ty) = self.body.local(*dst).map(|local| local.ty) else {
                continue;
            };
            if let Some((kind, _)) = result_ty.as_borrow(self.db)
                && matches!(
                    expr,
                    NExpr::Borrow { .. } | NExpr::Call { .. } | NExpr::Use(_)
                )
            {
                let loan = self.allocate_loan(Loan {
                    kind,
                    activation: match expr {
                        NExpr::Borrow { activation, .. } => *activation,
                        NExpr::Call { .. } | NExpr::Use(_) => BorrowActivation::Immediate,
                        _ => unreachable!(),
                    },
                    targets: FxHashSet::default(),
                    parents: FxHashSet::default(),
                    origin: stmt.origin,
                });
                self.loan_for_local.insert(*dst, loan);
                if let NExpr::Call { callee, args, .. } = expr
                    && let Some(summary) = self.call_borrow_summary(callee.key)?
                {
                    self.validate_call_borrow_summary(result_ty, args, &summary, stmt.origin)?;
                    self.call_loan_transforms.insert(loan, summary);
                }
                continue;
            }

            let NExpr::Call { callee, args, .. } = expr else {
                continue;
            };
            if !ty_contains_borrow(self.db, result_ty) {
                continue;
            }
            let Some(summary) = self.call_borrow_summary(callee.key)? else {
                continue;
            };
            self.validate_call_borrow_summary(result_ty, args, &summary, stmt.origin)?;
            self.call_result_loans.entry(stmt.id).or_default();
            let mut groups = Vec::<(BorrowResult, Vec<BorrowTransform>)>::new();
            for transform in summary {
                if let Some((_, transforms)) = groups
                    .iter_mut()
                    .find(|(result, _)| *result == transform.result)
                {
                    transforms.push(transform);
                } else {
                    groups.push((transform.result.clone(), vec![transform]));
                }
            }
            for (result, transforms) in groups {
                let loan = self.allocate_loan(Loan {
                    kind: result.kind,
                    activation: BorrowActivation::Immediate,
                    targets: FxHashSet::default(),
                    parents: FxHashSet::default(),
                    origin: stmt.origin,
                });
                self.call_result_loans
                    .get_mut(&stmt.id)
                    .expect("call loan entry")
                    .push((result, loan));
                self.call_loan_transforms.insert(loan, transforms);
            }
        }
        Ok(())
    }

    fn allocate_loan(&mut self, loan: Loan<'db>) -> LoanId {
        let id = LoanId(self.loans.len() as u32);
        self.loans.push(loan);
        id
    }

    fn call_borrow_summary(
        &self,
        key: SemanticInstanceKey<'db>,
    ) -> Result<Option<BorrowSummary>, SemanticBorrowDiagnostic<'db>> {
        let instance = get_or_build_semantic_instance(self.db, key);
        match self.summary_mode {
            BorrowSummaryMode::FinalCheck | BorrowSummaryMode::FinalSummary => {
                semantic_borrow_summary_voucher(self.db, instance)
            }
            BorrowSummaryMode::Provisional => provisional_borrow_summary_voucher(self.db, instance),
        }
    }

    fn validate_call_borrow_summary(
        &self,
        result_ty: crate::analysis::ty::ty_def::TyId<'db>,
        args: &[NOperand],
        summary: &[BorrowTransform],
        origin: SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let results = borrow_results_in_ty(self.db, result_ty);
        if let Some(transform) = summary
            .iter()
            .find(|transform| !results.contains(&transform.result))
        {
            return Err(self.internal_diag(
                origin,
                format!(
                    "callee borrow summary contains invalid result slot {:?}",
                    transform.result
                ),
            ));
        }
        if let Some(transform) = summary
            .iter()
            .find(|transform| transform.input.param() as usize >= args.len())
        {
            return Err(self.internal_diag(
                origin,
                format!(
                    "callee borrow summary references missing input {}",
                    transform.input.param()
                ),
            ));
        }
        if self.summary_mode == BorrowSummaryMode::FinalCheck
            && let Some(result) = results.iter().find(|result| {
                !result.projection.iter().any(|projection| {
                    matches!(projection, LayoutBackingProjection::VariantField { .. })
                }) && !summary.iter().any(|transform| transform.result == **result)
            })
        {
            return Err(self.internal_diag(
                origin,
                format!(
                    "callee borrow summary is missing result slot {:?}",
                    result.projection
                ),
            ));
        }
        Ok(())
    }

    pub(super) fn compute_entry_states(&mut self) {
        self.entry_state = solve_forward_cfg(&mut BorrowEntryStateAnalysis::new(self));
    }

    pub(super) fn compute_loan_targets(&mut self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let mut analysis = BorrowLoanTargetAnalysis::new(
            self.db,
            &self.body,
            &self.entry_state,
            &self.loan_for_local,
            &self.constant_indices,
            &self.call_result_loans,
            &self.call_loan_transforms,
        );
        let mut state = BorrowLoanTargetState {
            loans: &mut self.loans,
        };
        try_solve_sparse(&mut analysis, &mut state)
    }

    pub(super) fn apply_stmt_state(&self, state: &mut State, stmt: &super::ir::NSStmt<'db>) {
        self.canon().apply_stmt_state_with_call_loans(
            state,
            stmt,
            self.call_result_loans.get(&stmt.id).map(Vec::as_slice),
        );
    }

    fn compute_moved_states(&mut self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        self.moved_entry = try_solve_forward_cfg(&mut BorrowMovedStateAnalysis::new(self))?
            .iter()
            .map(|(bb, state)| (bb, state.0.clone()))
            .collect();
        Ok(())
    }

    fn check_conflicts(&self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            let bb = SBlockId::new(bb_idx);
            let mut state = self.entry_state[bb].clone();
            let mut moved = self.moved_entry[bb].clone();
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                self.check_stmt(&state, &moved, &self.live_before[bb_idx][stmt_idx], stmt)?;
                self.update_moved_for_stmt(&state, &mut moved, stmt)?;
                self.apply_stmt_state(&mut state, stmt);
            }
            self.check_terminator(
                &state,
                &moved,
                &self.live_before_term[bb],
                &block.terminator,
            )?;
        }
        Ok(())
    }

    fn check_stmt(
        &self,
        state: &State,
        moved: &MovedPlaces<'db>,
        live: &FxHashSet<crate::analysis::semantic::SLocalId>,
        stmt: &super::ir::NSStmt<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let active = self.effective_loans(state, live);
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => {
                match expr {
                    NExpr::ReadPlace { place, mode } => {
                        self.check_place_read(state, moved, &active, place, *mode, stmt.origin)?;
                    }
                    NExpr::Borrow { place, kind, .. } => {
                        let targets = self.canon().canonicalize_place(state, place, stmt.origin)?;
                        self.check_moved_overlap(
                            moved,
                            &targets,
                            stmt.origin,
                            "cannot borrow a moved value",
                        )?;
                        let loan = self.loan_for_local.get(dst).copied();
                        if let Some(kind) =
                            loan.map_or(Some(*kind), |loan| self.loan_conflict_kind(loan))
                        {
                            self.check_loan_conflict(&active, loan, kind, &targets, stmt.origin)?;
                        }
                    }
                    NExpr::ExtractEnumField {
                        value,
                        variant,
                        field,
                    } => {
                        let targets =
                            self.extract_enum_field_move_targets(state, *value, *variant, *field);
                        self.check_moved_overlap(
                            moved,
                            &targets,
                            stmt.origin,
                            "cannot use a value after it was moved",
                        )?;
                        if value.mode == ReadMode::Move {
                            self.check_move_targets_out(&active, &targets, stmt.origin)?;
                        } else {
                            let authorized =
                                self.canon()
                                    .loans_for_value_targets(state, value.local, &targets);
                            self.check_read_targets(&active, &authorized, &targets, stmt.origin)?;
                        }
                    }
                    _ => {
                        let expression_moved =
                            self.check_expr_operands(state, moved, &active, stmt.origin, expr)?;
                        let mut call_accesses =
                            self.check_call_argument_accesses(state, &active, stmt.origin, expr)?;
                        self.check_effect_place_accesses(
                            state,
                            &expression_moved,
                            &active,
                            stmt.origin,
                            expr,
                            &mut call_accesses,
                        )?;
                    }
                }
                if matches!(expr, NExpr::Call { .. } | NExpr::Use(_)) {
                    self.check_assigned_loan_conflicts(&active, stmt.id, *dst, stmt.origin)?;
                }
                self.check_assignment_write(state, &active, *dst, stmt.origin)?;
            }
            NSStmtKind::Store { dst, src } => {
                self.check_operand(
                    state,
                    moved,
                    &active,
                    *src,
                    stmt.origin,
                    "cannot use a value after it was moved",
                )?;
                let targets = self.canon().canonicalize_place(state, dst, stmt.origin)?;
                self.check_moved_parent(moved, &targets, stmt.origin)?;
                let authorized = self.canon().mut_loans_for_place(state, dst);
                self.check_write_targets(&active, &authorized, &targets, stmt.origin)?;
            }
        }
        Ok(())
    }

    fn check_assignment_write(
        &self,
        state: &State,
        active: &[LoanId],
        dst: crate::analysis::semantic::SLocalId,
        origin: SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let Some(place) = self
            .body
            .local(dst)
            .filter(|local| local.source.is_some_and(|binding| binding.is_mut()))
            .and_then(|local| local.lowering.place())
        else {
            return Ok(());
        };
        let targets = self.canon().canonicalize_place(state, place, origin)?;
        self.check_write_targets(active, &FxHashSet::default(), &targets, origin)
    }

    fn check_place_read(
        &self,
        state: &State,
        moved: &MovedPlaces<'db>,
        active: &[LoanId],
        place: &NSPlace<'db>,
        mode: ReadMode,
        origin: SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let targets = self.canon().canonicalize_place(state, place, origin)?;
        self.check_moved_overlap(
            moved,
            &targets,
            origin,
            "cannot use a value after it was moved",
        )?;
        if mode == ReadMode::Move {
            self.check_move_out(active, place, &targets, origin)
        } else {
            let authorized = self.canon().loans_for_place(state, place);
            self.check_read_targets(active, &authorized, &targets, origin)
        }
    }

    fn check_effect_place_accesses(
        &self,
        state: &State,
        moved: &MovedPlaces<'db>,
        active: &[LoanId],
        origin: SemOrigin<'db>,
        expr: &NExpr<'db>,
        accesses: &mut Vec<CallAccess<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let NExpr::Call {
            args, effect_args, ..
        } = expr
        else {
            return Ok(());
        };
        for (idx, effect_arg) in effect_args.iter().enumerate() {
            let group = args.len() + idx;
            let (targets, authorized, arg_origin) = match &effect_arg.arg {
                NEffectArgValue::Place(place) => {
                    let targets = self.canon().canonicalize_place(state, place, origin)?;
                    self.check_moved_overlap(
                        moved,
                        &targets,
                        origin,
                        "cannot use a value after it was moved",
                    )?;
                    let authorized = if effect_arg.required_mut {
                        self.canon().mut_loans_for_place(state, place)
                    } else {
                        self.canon().loans_for_place(state, place)
                    };
                    (targets, authorized, origin)
                }
                NEffectArgValue::Value(value) => {
                    let targets = self.canon().canonicalize_value_base(state, value.local);
                    let authorized = if effect_arg.required_mut {
                        self.canon()
                            .mut_loans_for_value_targets(state, value.local, &targets)
                    } else {
                        self.canon()
                            .loans_for_value_targets(state, value.local, &targets)
                    };
                    (targets, authorized, operand_origin(*value, origin))
                }
            };
            if effect_arg.required_mut {
                self.check_write_targets(active, &authorized, &targets, arg_origin)?;
                self.record_call_access(
                    accesses,
                    group,
                    CallAccessRole::Container,
                    BorrowKind::Mut,
                    targets,
                    arg_origin,
                )?;
            } else {
                self.check_read_targets(active, &authorized, &targets, arg_origin)?;
                self.record_call_access(
                    accesses,
                    group,
                    CallAccessRole::Container,
                    BorrowKind::Ref,
                    targets,
                    arg_origin,
                )?;
            }

            let target_ty = effect_arg.target_ty.or_else(|| match &effect_arg.arg {
                NEffectArgValue::Value(value) => self.body.local(value.local).map(|local| local.ty),
                NEffectArgValue::Place(_) => None,
            });
            let Some(target_ty) = target_ty else {
                continue;
            };
            for result in borrow_results_in_ty(self.db, target_ty) {
                let targets = match &effect_arg.arg {
                    NEffectArgValue::Place(place) => {
                        self.canon().canonicalize_place_layout_projection(
                            state,
                            place,
                            target_ty,
                            &result.projection,
                        )
                    }
                    NEffectArgValue::Value(value) => {
                        self.canon().canonicalize_value_layout_projection(
                            state,
                            value.local,
                            &result.projection,
                        )
                    }
                };
                let authorized = match &effect_arg.arg {
                    NEffectArgValue::Place(place) if result.kind == BorrowKind::Mut => self
                        .canon()
                        .mut_loans_for_place_targets(state, place, &targets),
                    NEffectArgValue::Place(place) => {
                        self.canon().loans_for_place_targets(state, place, &targets)
                    }
                    NEffectArgValue::Value(value) if result.kind == BorrowKind::Mut => self
                        .canon()
                        .mut_loans_for_value_targets(state, value.local, &targets),
                    NEffectArgValue::Value(value) => {
                        self.canon()
                            .loans_for_value_targets(state, value.local, &targets)
                    }
                };
                if result.kind == BorrowKind::Mut {
                    self.check_write_targets(active, &authorized, &targets, arg_origin)?;
                } else {
                    self.check_read_targets(active, &authorized, &targets, arg_origin)?;
                }
                self.record_call_access(
                    accesses,
                    group,
                    CallAccessRole::BorrowSlot,
                    result.kind,
                    targets,
                    arg_origin,
                )?;
            }
        }
        Ok(())
    }

    fn check_call_argument_accesses(
        &self,
        state: &State,
        active: &[LoanId],
        origin: SemOrigin<'db>,
        expr: &NExpr<'db>,
    ) -> Result<Vec<CallAccess<'db>>, SemanticBorrowDiagnostic<'db>> {
        let NExpr::Call { callee, args, .. } = expr else {
            return Ok(Vec::new());
        };
        let instance = get_or_build_semantic_instance(self.db, callee.key);
        let callable = callee.key.callable_body(self.db);
        let mut accesses = Vec::with_capacity(args.len());
        for (idx, arg) in args.iter().copied().enumerate() {
            let Some(binding @ LocalBinding::Param { mode, .. }) =
                callable.param_binding(self.db, idx)
            else {
                return Err(
                    self.internal_diag(origin, format!("callee is missing value parameter {idx}"))
                );
            };
            let ty = instance.normalized_binding_ty(self.db, binding);
            let moves_value =
                arg.mode == ReadMode::Move && self.local_has_runtime_move_semantics(arg.local);
            let arg_origin = operand_origin(arg, origin);
            let mutably_passed_by_place = mode != FuncParamMode::Own && binding.is_mut();
            if ty.as_borrow(self.db).is_none()
                && (arg.mode != ReadMode::Copy || mutably_passed_by_place)
            {
                let kind = if mutably_passed_by_place || moves_value {
                    BorrowKind::Mut
                } else {
                    BorrowKind::Ref
                };
                let targets = self.canon().canonicalize_value_base(state, arg.local);
                if kind == BorrowKind::Mut && !moves_value {
                    let authorized = self
                        .canon()
                        .mut_loans_for_value_targets(state, arg.local, &targets);
                    self.check_write_targets(active, &authorized, &targets, arg_origin)?;
                }
                self.record_call_access(
                    &mut accesses,
                    idx,
                    CallAccessRole::Container,
                    kind,
                    targets,
                    arg_origin,
                )?;
            }
            for result in borrow_results_in_ty(self.db, ty) {
                let targets = self.canon().canonicalize_value_layout_projection(
                    state,
                    arg.local,
                    &result.projection,
                );
                let authorized = if result.kind == BorrowKind::Mut {
                    self.canon()
                        .mut_loans_for_value_targets(state, arg.local, &targets)
                } else {
                    self.canon()
                        .loans_for_value_targets(state, arg.local, &targets)
                };
                if result.kind == BorrowKind::Mut {
                    self.check_write_targets(active, &authorized, &targets, arg_origin)?;
                } else {
                    self.check_read_targets(active, &authorized, &targets, arg_origin)?;
                }
                self.record_call_access(
                    &mut accesses,
                    idx,
                    CallAccessRole::BorrowSlot,
                    result.kind,
                    targets,
                    arg_origin,
                )?;
            }
        }
        Ok(accesses)
    }

    fn record_call_access(
        &self,
        accesses: &mut Vec<CallAccess<'db>>,
        group: usize,
        role: CallAccessRole,
        kind: BorrowKind,
        targets: FxHashSet<CanonPlace<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        // An argument owns its container and the capability slots stored inside
        // it, but distinct capability slots must still obey aliasing rules.
        let conflict = accesses.iter().find(|access| {
            (access.group != group
                || access.role == CallAccessRole::BorrowSlot && role == CallAccessRole::BorrowSlot)
                && !matches!((access.kind, kind), (BorrowKind::Ref, BorrowKind::Ref))
                && place_set_overlaps(&access.targets, &targets)
        });
        if let Some(conflict) = conflict {
            let mut diag = SemanticBorrowDiagnostic::new(
                self.instance,
                SemanticBorrowDiagKind::BorrowConflict,
                "call arguments require conflicting access to the same place".to_string(),
                SemanticBorrowDiagnosticSpan::Origin {
                    owner: self.instance.key(self.db).owner(self.db),
                    origin,
                },
            );
            self.push_secondary_origin(
                &mut diag,
                conflict.origin,
                "overlapping argument access occurs here".to_string(),
            );
            return Err(diag);
        }
        if !targets.is_empty() {
            accesses.push(CallAccess {
                group,
                role,
                kind,
                targets,
                origin,
            });
        }
        Ok(())
    }

    fn check_assigned_loan_conflicts(
        &self,
        active: &[LoanId],
        stmt: SStmtId,
        local: crate::analysis::semantic::SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let direct = self.loan_for_local.get(&local).copied().into_iter();
        let aggregate = self
            .call_result_loans
            .get(&stmt)
            .into_iter()
            .flatten()
            .map(|(_, loan)| *loan);
        for loan_id in direct.chain(aggregate) {
            let loan = &self.loans[loan_id.0 as usize];
            self.check_loan_conflict(active, Some(loan_id), loan.kind, &loan.targets, origin)?;
        }
        Ok(())
    }

    fn check_loan_conflict(
        &self,
        active: &[LoanId],
        new_loan: Option<LoanId>,
        kind: BorrowKind,
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let Some(conflict) = self.first_loan_conflict(active, new_loan, kind, targets) {
            return Err(self.borrow_conflict_diag(
                origin,
                self.overlapping_loans_msg(conflict, kind),
                conflict,
            ));
        }
        Ok(())
    }

    fn check_read_targets(
        &self,
        active: &[LoanId],
        authorized: &FxHashSet<LoanId>,
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let conflict = active.iter().copied().find(|loan| {
            !authorized.contains(loan)
                && self.loan_conflict_kind(*loan) == Some(BorrowKind::Mut)
                && place_set_overlaps(&self.loans[loan.0 as usize].targets, targets)
        });
        if let Some(conflict) = conflict {
            return Err(self.borrow_conflict_diag(
                origin,
                "cannot read this place while a mutable borrow is active".to_string(),
                conflict,
            ));
        }
        Ok(())
    }

    fn check_write_targets(
        &self,
        active: &[LoanId],
        authorized: &FxHashSet<LoanId>,
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let conflict = active.iter().copied().find(|loan| {
            !authorized.contains(loan)
                && self.loan_conflict_kind(*loan).is_some()
                && place_set_overlaps(&self.loans[loan.0 as usize].targets, targets)
        });
        if let Some(conflict) = conflict {
            return Err(self.borrow_conflict_diag(
                origin,
                "cannot write to this place while it is borrowed".to_string(),
                conflict,
            ));
        }
        Ok(())
    }

    fn check_terminator(
        &self,
        state: &State,
        moved: &MovedPlaces<'db>,
        live: &FxHashSet<crate::analysis::semantic::SLocalId>,
        term: &super::ir::NSTerminator<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        match &term.kind {
            NSTerminatorKind::Goto(_) | NSTerminatorKind::Assert { .. } => {}
            NSTerminatorKind::Branch { cond, .. } | NSTerminatorKind::Return(Some(cond)) => {
                let active = self.effective_loans(state, live);
                self.check_operand(
                    state,
                    moved,
                    &active,
                    *cond,
                    term.origin,
                    "cannot use a value after it was moved",
                )?;
            }
            NSTerminatorKind::MatchEnum { value, .. } => {
                let active = self.effective_loans(state, live);
                self.check_operand(
                    state,
                    moved,
                    &active,
                    NOperand {
                        mode: ReadMode::Read,
                        ..*value
                    },
                    term.origin,
                    "cannot use a value after it was moved",
                )?;
            }
            NSTerminatorKind::Return(None) => {}
        }
        if let NSTerminatorKind::Return(Some(value)) = term.kind
            && self
                .body
                .local(value.local)
                .is_some_and(|local| local.ty.as_borrow(self.db).is_some())
            && self
                .canon()
                .borrow_local_targets(state, value.local)
                .is_empty()
        {
            return Err(self.internal_diag(
                term.origin,
                "borrow return local has no tracked loan targets".to_string(),
            ));
        }
        if let NSTerminatorKind::Return(Some(value)) = term.kind
            && self
                .body
                .local(value.local)
                .is_some_and(|local| local.ty.as_borrow(self.db).is_none())
        {
            for loan_id in state.loans_in(value.local) {
                let loan = &self.loans[loan_id.0 as usize];
                if let Some(local) = loan.targets.iter().find_map(|target| match target.root {
                    BorrowRoot::Local(local) => Some(local),
                    BorrowRoot::Param(_) | BorrowRoot::Provider(_) => None,
                }) {
                    let name = self.pretty_local_name(local);
                    let mut diag = self.invalid_return_diag(
                        term.origin,
                        format!("cannot return a value that holds a borrow of local `{name}`"),
                    );
                    self.push_secondary_origin(
                        &mut diag,
                        self.loan_origin(loan_id),
                        "borrow created here".to_string(),
                    );
                    return Err(diag);
                }
            }
        }
        Ok(())
    }

    fn compute_return_summary(&self) -> Result<BorrowSummary, SemanticBorrowDiagnostic<'db>> {
        let mut out = Vec::new();
        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            let NSTerminatorKind::Return(Some(value)) = block.terminator.kind else {
                continue;
            };
            let mut state = self.entry_state[SBlockId::new(bb_idx)].clone();
            for stmt in &block.stmts {
                self.apply_stmt_state(&mut state, stmt);
            }
            let Some(return_local) = self.body.local(value.local) else {
                continue;
            };
            for result in borrow_results_in_ty(self.db, return_local.ty) {
                let targets = self.canon().canonicalize_value_layout_projection(
                    &state,
                    value.local,
                    &result.projection,
                );
                if targets.is_empty() {
                    if self.summary_mode != BorrowSummaryMode::FinalCheck
                        || result.projection.iter().any(|projection| {
                            matches!(projection, LayoutBackingProjection::VariantField { .. })
                        })
                    {
                        continue;
                    }
                    return Err(self.internal_diag(
                        block.terminator.origin,
                        format!(
                            "borrow result slot {:?} has no tracked source",
                            result.projection
                        ),
                    ));
                }
                for target in targets {
                    match &target.root {
                        BorrowRoot::Param(idx) => {
                            let Some(input_projection) =
                                layout_path_for_semantic_projection(&target.proj)
                            else {
                                return Err(self.internal_diag(
                                    block.terminator.origin,
                                    "cannot summarize borrow source projection".to_string(),
                                ));
                            };
                            out.push(BorrowTransform {
                                result: result.clone(),
                                input: BorrowInput::Place {
                                    param: *idx,
                                    projection: input_projection,
                                },
                            });
                        }
                        BorrowRoot::Provider(_) => {
                            return Err(self.invalid_return_diag(
                                block.terminator.origin,
                                "cannot return a borrow derived from an effect parameter"
                                    .to_string(),
                            ));
                        }
                        BorrowRoot::Local(local) => {
                            let name = self.pretty_local_name(*local);
                            return Err(self.invalid_return_diag(
                                block.terminator.origin,
                                format!("cannot return a borrow to local `{name}`"),
                            ));
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    fn effective_loans(
        &self,
        state: &State,
        live: &FxHashSet<crate::analysis::semantic::SLocalId>,
    ) -> Vec<LoanId> {
        let active = state
            .local_loans
            .iter()
            .filter(|(local, _)| live.contains(local))
            .flat_map(|(_, held)| held.values())
            .flatten()
            .copied()
            .collect::<FxHashSet<_>>();
        let mut suspended = FxHashSet::default();
        let mut worklist: Vec<_> = active.iter().copied().collect();
        while let Some(loan) = worklist.pop() {
            for parent in &self.loans[loan.0 as usize].parents {
                if suspended.insert(*parent) {
                    worklist.push(*parent);
                }
            }
        }
        let mut active: Vec<_> = active
            .into_iter()
            .filter(|loan| !suspended.contains(loan))
            .collect();
        active.sort_by_key(|loan| loan.0);
        active
    }

    fn first_loan_conflict(
        &self,
        active: &[LoanId],
        new_loan: Option<LoanId>,
        new_kind: BorrowKind,
        targets: &FxHashSet<CanonPlace<'db>>,
    ) -> Option<LoanId> {
        let reborrow_parents = new_loan.map(|loan| &self.loans[loan.0 as usize].parents);
        active
            .iter()
            .copied()
            .filter(|loan| reborrow_parents.is_none_or(|parents| !parents.contains(loan)))
            .find(|loan_id| {
                let loan = &self.loans[loan_id.0 as usize];
                self.loan_conflict_kind(*loan_id).is_some_and(|kind| {
                    !matches!((kind, new_kind), (BorrowKind::Ref, BorrowKind::Ref))
                }) && place_set_overlaps(&loan.targets, targets)
            })
    }

    fn loan_conflict_kind(&self, loan: LoanId) -> Option<BorrowKind> {
        let loan = &self.loans[loan.0 as usize];
        // Receiver reservations remain dormant while later arguments are
        // evaluated. The call-access checks perform their activation.
        if loan.activation == BorrowActivation::AtCall {
            None
        } else {
            Some(loan.kind)
        }
    }

    fn check_move_out(
        &self,
        active: &[LoanId],
        place: &NSPlace<'db>,
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let NSPlaceRoot::CarrierDerefLocal(local) = place.root {
            if self.body.locals[local.index()]
                .source
                .is_some_and(|binding| {
                    matches!(
                        binding,
                        crate::analysis::ty::ty_check::LocalBinding::Param {
                            mode: FuncParamMode::View,
                            ..
                        }
                    )
                })
            {
                return Err(self.move_conflict_diag(
                    origin,
                    "cannot move out of a view parameter".to_string(),
                ));
            }
            return Err(self.move_conflict_diag(
                origin,
                "cannot move out through a borrow handle".to_string(),
            ));
        }
        self.check_move_targets_out(active, targets, origin)?;
        Ok(())
    }

    fn check_move_targets_out(
        &self,
        active: &[LoanId],
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        for target in targets {
            if let BorrowRoot::Param(idx) = target.root
                && self
                    .param_modes
                    .get(idx as usize)
                    .copied()
                    .is_some_and(|mode| mode == FuncParamMode::View)
            {
                return Err(self.move_conflict_diag(
                    origin,
                    "cannot move out of a view parameter".to_string(),
                ));
            }
        }
        if let Some(loan) = active
            .iter()
            .copied()
            .find(|loan| place_set_overlaps(&self.loans[loan.0 as usize].targets, targets))
        {
            return Err(self.borrow_conflict_diag(
                origin,
                "cannot move out of a value while it is borrowed".to_string(),
                loan,
            ));
        }
        Ok(())
    }

    pub(super) fn update_moved_for_stmt(
        &self,
        state: &State,
        moved: &mut MovedPlaces<'db>,
        stmt: &super::ir::NSStmt<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => {
                if let Some(root) = self
                    .local_root(*dst)
                    .and_then(|root| self.canon().root_to_borrow_root(root))
                {
                    moved.retain(|place, _| place.root != root);
                }
                if let NExpr::ReadPlace {
                    place,
                    mode: ReadMode::Move,
                } = expr
                {
                    let site = MoveSite {
                        origin: stmt.origin,
                        note: "value is moved here".to_string(),
                    };
                    for place in self.canon().canonicalize_place(state, place, stmt.origin)? {
                        moved.insert(place, site.clone());
                    }
                }
                if let NExpr::ExtractEnumField {
                    value,
                    variant,
                    field,
                } = expr
                {
                    if value.mode == ReadMode::Move {
                        let site = self.move_site(*value, operand_origin(*value, stmt.origin));
                        for place in
                            self.extract_enum_field_move_targets(state, *value, *variant, *field)
                        {
                            moved.insert(place, site.clone());
                        }
                    }
                } else {
                    self.record_expr_moves(state, moved, stmt.origin, expr)?;
                }
            }
            NSStmtKind::Store { dst, src } => {
                self.record_operand_move(state, moved, *src, stmt.origin)?;
                let written = self.canon().canonicalize_place(state, dst, stmt.origin)?;
                moved.retain(|place, _| {
                    !written.iter().any(|written| {
                        written.root == place.root && written.proj.is_prefix_of(&place.proj)
                    })
                });
            }
        }
        Ok(())
    }

    fn extract_enum_field_move_targets(
        &self,
        state: &State,
        source: NOperand,
        variant: crate::analysis::semantic::VariantIndex,
        field: crate::analysis::semantic::FieldIndex,
    ) -> FxHashSet<CanonPlace<'db>> {
        let Some(source_local) = self.body.local(source.local) else {
            return FxHashSet::default();
        };
        let projection = Projection::VariantField {
            variant,
            enum_ty: source_local.ty,
            field_idx: field.0 as usize,
        };
        self.canon()
            .canonicalize_value_base(state, source.local)
            .into_iter()
            .map(|mut target| {
                target.proj.push(projection.clone());
                target
            })
            .collect()
    }

    fn check_expr_operands(
        &self,
        state: &State,
        moved: &MovedPlaces<'db>,
        active: &[LoanId],
        origin: crate::analysis::semantic::SemOrigin<'db>,
        expr: &NExpr<'db>,
    ) -> Result<MovedPlaces<'db>, SemanticBorrowDiagnostic<'db>> {
        let mut moved = moved.clone();
        expr.try_for_each_value_operand(|value| {
            self.check_operand(
                state,
                &moved,
                active,
                value,
                origin,
                "cannot use a value after it was moved",
            )?;
            self.record_operand_move(state, &mut moved, value, origin)
        })?;
        Ok(moved)
    }

    fn check_operand(
        &self,
        state: &State,
        moved: &MovedPlaces<'db>,
        active: &[LoanId],
        operand: NOperand,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: &str,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let origin = operand_origin(operand, origin);
        let targets = self.canon().canonicalize_value_base(state, operand.local);
        if targets.is_empty() {
            return Ok(());
        }
        self.check_moved_overlap(moved, &targets, origin, message)?;
        if operand.mode == ReadMode::Move && self.local_has_runtime_move_semantics(operand.local) {
            self.check_move_targets_out(active, &targets, origin)
        } else {
            let authorized = self
                .canon()
                .loans_for_value_targets(state, operand.local, &targets);
            self.check_read_targets(active, &authorized, &targets, origin)
        }
    }

    fn record_expr_moves(
        &self,
        state: &State,
        moved: &mut MovedPlaces<'db>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        expr: &NExpr<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        expr.try_for_each_value_operand(|value| {
            self.record_operand_move(state, moved, value, origin)
        })
    }

    fn record_operand_move(
        &self,
        state: &State,
        moved: &mut MovedPlaces<'db>,
        operand: NOperand,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let origin = operand_origin(operand, origin);
        if operand.mode == ReadMode::Move && self.local_has_runtime_move_semantics(operand.local) {
            let site = self.move_site(operand, origin);
            for place in self.canon().canonicalize_value_base(state, operand.local) {
                moved.insert(place, site.clone());
            }
        }
        Ok(())
    }

    fn move_site(&self, operand: NOperand, origin: SemOrigin<'db>) -> MoveSite<'db> {
        MoveSite {
            origin,
            note: self.moved_operand_name(operand).map_or_else(
                || "value is moved here".to_string(),
                |name| format!("`{name}` is moved here"),
            ),
        }
    }

    fn moved_operand_name(&self, operand: NOperand) -> Option<String> {
        let expr = operand.origin?;
        let body = self.hir_body?;
        let Partial::Present(Expr::Path(Partial::Present(path))) = expr.data(self.db, body) else {
            return None;
        };
        path.as_ident(self.db)
            .map(|ident| ident.data(self.db).to_string())
    }

    fn local_has_runtime_move_semantics(&self, local: crate::analysis::semantic::SLocalId) -> bool {
        self.body.local(local).is_some_and(|local| {
            local_has_runtime_move_semantics(self.db, local, &self.body.borrow_roots)
        })
    }

    fn check_moved_overlap(
        &self,
        moved: &MovedPlaces<'db>,
        accessed: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: &str,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let Some((_, site)) = moved.iter().find(|(moved, _)| {
            accessed
                .iter()
                .any(|accessed| places_overlap(moved, accessed))
        }) {
            let mut diag = self.move_conflict_diag(origin, message.to_string());
            self.push_secondary_origin(&mut diag, site.origin, site.note.clone());
            return Err(diag);
        }
        Ok(())
    }

    fn check_moved_parent(
        &self,
        moved: &MovedPlaces<'db>,
        written: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let Some((_, site)) = moved.iter().find(|(moved, _)| {
            written.iter().any(|written| {
                written.root == moved.root
                    && moved.proj.is_prefix_of(&written.proj)
                    && moved.proj != written.proj
            })
        }) {
            let mut diag =
                self.move_conflict_diag(origin, "cannot write through a moved value".to_string());
            self.push_secondary_origin(&mut diag, site.origin, site.note.clone());
            return Err(diag);
        }
        Ok(())
    }

    fn local_root(&self, local: crate::analysis::semantic::SLocalId) -> Option<NBorrowRootId> {
        self.body.local(local)?.lowering.root()
    }

    fn successors(&self, term: &NSTerminatorKind<'db>) -> BlockAdjacency {
        let mut out = BlockAdjacency::new();
        match term {
            NSTerminatorKind::Goto(bb) => out.push(*bb),
            NSTerminatorKind::Branch {
                then_bb, else_bb, ..
            } => {
                out.push(*then_bb);
                out.push(*else_bb);
            }
            NSTerminatorKind::MatchEnum { cases, default, .. } => {
                out.extend(cases.iter().map(|(_, bb)| *bb));
                if let Some(default) = default {
                    out.push(*default);
                }
            }
            NSTerminatorKind::Assert { .. } | NSTerminatorKind::Return(_) => {}
        }
        out
    }

    pub(super) fn cfg_successor_indices(&self) -> CfgAdjacency {
        let mut successors = CfgAdjacency::new();
        successors.resize(self.body.blocks.len());
        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            successors[SBlockId::new(bb_idx)] = self.successors(&block.terminator.kind);
        }
        successors
    }

    pub(super) fn cfg_predecessor_indices(&self) -> CfgAdjacency {
        let mut predecessors = CfgAdjacency::new();
        predecessors.resize(self.body.blocks.len());
        for (bb, successors) in self.cfg_successor_indices().iter() {
            for succ in successors.iter().copied() {
                predecessors[succ].push(bb);
            }
        }
        predecessors
    }

    fn borrow_conflict_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
        loan: LoanId,
    ) -> SemanticBorrowDiagnostic<'db> {
        let mut diag = self.diag(SemanticBorrowDiagKind::BorrowConflict, origin, message);
        self.push_secondary_origin(
            &mut diag,
            self.loan_origin(loan),
            "borrow created here".to_string(),
        );
        diag
    }

    fn loan_origin(&self, mut loan: LoanId) -> SemOrigin<'db> {
        let mut seen = FxHashSet::default();
        while seen.insert(loan) {
            let data = &self.loans[loan.0 as usize];
            let Some(parent) = data.parents.iter().copied().next() else {
                return data.origin;
            };
            if data.parents.len() != 1 {
                return data.origin;
            }
            loan = parent;
        }
        self.loans[loan.0 as usize].origin
    }

    fn move_conflict_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> SemanticBorrowDiagnostic<'db> {
        self.diag(SemanticBorrowDiagKind::MoveConflict, origin, message)
    }

    fn invalid_return_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> SemanticBorrowDiagnostic<'db> {
        self.diag(SemanticBorrowDiagKind::InvalidReturnBorrow, origin, message)
    }

    fn internal_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> SemanticBorrowDiagnostic<'db> {
        self.diag(SemanticBorrowDiagKind::Internal, origin, message)
    }

    pub(super) fn diag(
        &self,
        kind: SemanticBorrowDiagKind,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> SemanticBorrowDiagnostic<'db> {
        SemanticBorrowDiagnostic::new(
            self.instance,
            kind,
            message,
            SemanticBorrowDiagnosticSpan::Origin {
                owner: self.instance.key(self.db).owner(self.db),
                origin,
            },
        )
    }

    fn push_secondary_origin(
        &self,
        diag: &mut SemanticBorrowDiagnostic<'db>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) {
        diag.push_secondary(
            message,
            SemanticBorrowDiagnosticSpan::Origin {
                owner: self.instance.key(self.db).owner(self.db),
                origin,
            },
        );
    }

    fn overlapping_loans_msg(&self, loan: LoanId, new_kind: BorrowKind) -> String {
        match (
            new_kind,
            self.loan_conflict_kind(loan)
                .expect("dormant loans do not produce conflicts"),
        ) {
            (BorrowKind::Mut, BorrowKind::Mut) => {
                "cannot mutably borrow this place while a mut borrow is active".to_string()
            }
            (BorrowKind::Mut, BorrowKind::Ref) => {
                "cannot mutably borrow this place while an immutable borrow is active".to_string()
            }
            (BorrowKind::Ref, BorrowKind::Mut) => {
                "cannot immutably borrow this place while a mutable borrow is active".to_string()
            }
            (BorrowKind::Ref, BorrowKind::Ref) => unreachable!(),
        }
    }

    fn pretty_local_name(&self, local: crate::analysis::semantic::SLocalId) -> String {
        self.hir_body
            .zip(self.body.local(local).and_then(|local| local.source))
            .map(|(body, source)| source.pretty_name_in_body(self.db, body))
            .unwrap_or_else(|| format!("%{}", local.index()))
    }
}

fn semantic_borrow_summary_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowSummaryResult<'db> {
    SemanticBorrowSummaryResult::Ok(
        instance_returns_borrowing_value(db, instance)
            .then(|| BorrowSummaryId::new(db, Vec::new())),
    )
}

fn instance_returns_borrowing_value<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> bool {
    ty_contains_borrow(db, instance.normalized_result_ty(db))
}

fn conservative_signature_borrow_summary<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> BorrowSummary {
    let inputs = instance
        .key(db)
        .callable_body(db)
        .param_bindings(db)
        .into_iter()
        .filter_map(|binding| match binding {
            LocalBinding::Param {
                idx, ty, is_mut, ..
            } => u32::try_from(idx)
                .ok()
                .map(|idx| (idx, instance.normalized_ty(db, ty), is_mut)),
            LocalBinding::Local { .. } | LocalBinding::EffectParam { .. } => None,
        })
        .collect::<Vec<_>>();
    let mut summary = Vec::new();
    for result in borrow_results_in_ty(db, instance.normalized_result_ty(db)) {
        for (idx, ty, is_mut) in inputs.iter().copied() {
            if signature_input_is_unresolved(db, ty) {
                summary.push(BorrowTransform {
                    result: result.clone(),
                    input: BorrowInput::AnyInParam(idx),
                });
                continue;
            }
            if result.kind == BorrowKind::Ref || is_mut {
                summary.push(BorrowTransform {
                    result: result.clone(),
                    input: BorrowInput::Place {
                        param: idx,
                        projection: Vec::new(),
                    },
                });
            }
            for input in borrow_results_in_ty(db, ty) {
                if result.kind == BorrowKind::Ref || input.kind == BorrowKind::Mut {
                    summary.push(BorrowTransform {
                        result: result.clone(),
                        input: BorrowInput::Place {
                            param: idx,
                            projection: input.projection,
                        },
                    });
                }
            }
        }
    }
    summary.sort_unstable();
    summary.dedup();
    summary
}

fn signature_input_is_unresolved(db: &dyn HirAnalysisDb, input_ty: TyId<'_>) -> bool {
    input_ty.has_param(db)
        || input_ty.has_var(db)
        || input_ty.has_projection(db)
        || input_ty.has_invalid(db)
}

fn semantic_borrow_summary_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &SemanticBorrowSummaryResult<'db>,
    _count: u32,
    _instance: SemanticInstance<'db>,
) -> salsa::CycleRecoveryAction<SemanticBorrowSummaryResult<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

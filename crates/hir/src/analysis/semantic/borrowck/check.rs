use std::{collections::VecDeque, mem};

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
            BorrowActivation, FieldIndex, LayoutBackingProjection, SBlockId, SConst, SLocalId,
            SStmtId, SemConstScalar, SemConstValue, SemOrigin, SemanticInstance,
            SemanticInstanceKey, get_or_build_semantic_instance, identity_semantic_instance_key,
        },
        ty::{
            ProviderAddressSpace,
            ty_check::{
                BodyOwner, CLOSURE_ARGS_PARAM_IDX, EffectParamSite, LocalBinding, ParamSite,
            },
            ty_def::{BorrowKind, ClosureParamMode, TyId},
            ty_is_borrow,
        },
    },
    core::semantic::EffectEnvView,
    hir_def::{Body, Expr, FuncParamMode, ItemKind, Partial, TopLevelMod},
    projection::Projection,
};

use super::{
    analyses::{
        BorrowEntryStateAnalysis, BorrowLivenessAnalysis, BorrowLoanTargetAnalysis,
        BorrowLoanTargetInputs, BorrowLoanTargetState, BorrowMovedStateAnalysis, BorrowSummaryMode,
    },
    canon::{
        BlockAdjacency, BorrowCanonCx, BorrowRoot, CanonPlace, CfgAdjacency, Loan, LoanId,
        MoveSite, MovedPlaces, State, address_space_rank, known_address_space_for_borrow_root,
        place_set_overlaps, places_overlap,
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
        local_has_runtime_move_semantics, return_borrow_results_in_ty, semantic_projection_ty,
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
    if typed_body.has_smir_lowering_blocker(db) {
        return;
    }
    let has_closures = typed_body.closure_infos().next().is_some();
    let is_identity = key == identity_semantic_instance_key(db, owner);
    let has_concrete_provider_specialization =
        !key.effect_providers(db).providers(db).is_empty() && is_fully_instantiated_key(db, key);
    if is_identity
        || has_closures
        || matches!(owner, BodyOwner::Closure { .. })
        || has_concrete_provider_specialization
    {
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
    fresh_call_args: FxHashSet<(SStmtId, SLocalId)>,
    fresh_local_allocations: FxHashSet<(SStmtId, SLocalId)>,
    fresh_local_sites: FxHashMap<SLocalId, Vec<SStmtId>>,
    fresh_status_at_stmt: FreshStatusAtStmt<'db>,
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

#[derive(Clone, Copy)]
struct ConflictCheckCx<'a, 'db> {
    state: &'a State,
    moved: &'a MovedPlaces<'db>,
    active: &'a [LoanId],
}

type FreshRootStates<'db> = FxHashMap<FreshStorageRoot, Vec<FreshRootStatus<'db>>>;
type FreshLocalStatus<'db> = FxHashMap<SLocalId, FreshValueStatus<'db>>;
type FreshStatusAtStmt<'db> = FxHashMap<(SStmtId, SLocalId), FreshValueStatus<'db>>;
type FreshCarrierPath = Vec<LayoutBackingProjection>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FreshValueStatus<'db> {
    roots: FreshRootStates<'db>,
    /// Whether `roots` accounts for every dynamically allocated root which
    /// this value can carry. Missing roots are impossible only when this is
    /// true.
    complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FreshRootStatus<'db> {
    generation: FreshGeneration,
    /// The latest static retention sites through which this dynamic instance
    /// reached storage that can outlive the allocation's current loop
    /// iteration.
    ///
    /// Intersecting sets conservatively denote possibly identical instances.
    /// Disjoint sets let the checker retain separation between instances from
    /// one allocation site that were retained by mutually exclusive
    /// statements. A later retention renames every current alias together, so
    /// copying one instance through multiple statements cannot make it appear
    /// disjoint from itself.
    retention_sites: FxHashSet<SStmtId>,
    /// Locations within the containing value which carry this instance.
    carriers: FxHashSet<FreshCarrierPath>,
    claimed: FxHashSet<NSProjectionPath<'db>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FreshStorageRoot {
    Call {
        stmt: SStmtId,
        source: SLocalId,
    },
    Local {
        definition: SStmtId,
        local: SLocalId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreshGeneration {
    /// Produced by the current dynamic execution of its allocation site.
    Current,
    /// May have crossed a backedge since its allocation site executed.
    Stale,
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
        let mut local_use_sites = vec![Vec::<Option<SStmtId>>::new(); body.locals.len()];
        for (block_idx, block) in body.blocks.iter().enumerate() {
            let block_id = SBlockId::new(block_idx);
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                for &local in facts.stmt_uses(block_id, stmt_idx) {
                    local_use_sites[local.index()].push(Some(stmt.id));
                }
            }
            for &local in facts.terminator_uses(block_id) {
                local_use_sites[local.index()].push(None);
            }
        }
        let mut fresh_call_args = FxHashSet::default();
        for (block_idx, block) in body.blocks.iter().enumerate() {
            let block_id = SBlockId::new(block_idx);
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                let NSStmtKind::Assign {
                    dst: call_result,
                    expr: NExpr::Call { callee, args, .. },
                } = &stmt.kind
                else {
                    continue;
                };
                let Some(assignment) = facts.stmt_assignment(block_id, stmt_idx) else {
                    continue;
                };
                let params = callee.key.callable_body(db).param_bindings(db);
                for (arg_idx, arg) in args.iter().enumerate() {
                    let local = arg.local;
                    let is_owned_param = matches!(
                        params.get(arg_idx),
                        Some(LocalBinding::Param {
                            mode: FuncParamMode::Own,
                            ..
                        })
                    );
                    let is_single_use_temporary = is_owned_param
                        && body.local(local).is_some_and(|local_data| {
                            local_data.source.is_none()
                                && local_data.lowering.root().is_some_and(|root| {
                                    matches!(
                                        body.root(root),
                                        Some(NBorrowRoot::LocalSlot { local: root_local })
                                            if *root_local == local
                                    )
                                })
                        })
                        && facts.defs_by_local(local).len() == 1
                        && facts.assignments_using_local(local) == [assignment]
                        && facts
                            .dynamic_dependents(local)
                            .iter()
                            .all(|dependent| *dependent == local || dependent == call_result)
                        && local_use_sites[local.index()] == [Some(stmt.id)]
                        && args
                            .iter()
                            .filter(|candidate| candidate.local == local)
                            .count()
                            == 1;
                    if is_single_use_temporary {
                        fresh_call_args.insert((stmt.id, local));
                    }
                }
            }
        }
        let mut fresh_local_allocations = FxHashSet::default();
        let mut fresh_local_sites = FxHashMap::<SLocalId, Vec<SStmtId>>::default();
        for (idx, local_data) in body.locals.iter().enumerate() {
            let local = SLocalId::new(idx);
            let demand = local_data.facts.root_demand;
            let storage_can_escape = demand.borrowed_or_addr_taken
                || demand.mut_borrowed_or_addr_taken
                || demand.passed_by_place;
            let is_self_rooted = local_data.lowering.root().is_some_and(|root| {
                matches!(
                    body.root(root),
                    Some(NBorrowRoot::LocalSlot { local: root_local }) if *root_local == local
                )
            });
            if !storage_can_escape || !is_self_rooted {
                continue;
            }
            let definitions = facts.defs_by_local(local);
            let allocation_definitions = if local_data.source.is_some() {
                definitions.get(..1).unwrap_or_default()
            } else {
                definitions
            };
            for &definition in allocation_definitions {
                let Some(assignment) = facts.assignment(definition) else {
                    continue;
                };
                let stmt = body.blocks[assignment.block.index()].stmts[assignment.stmt_idx].id;
                fresh_local_allocations.insert((stmt, local));
                fresh_local_sites.entry(local).or_default().push(stmt);
            }
        }
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
            fresh_call_args,
            fresh_local_allocations,
            fresh_local_sites,
            fresh_status_at_stmt: FxHashMap::default(),
            constant_indices,
            moved_entry: SecondaryMap::new(),
            live_before: Vec::new(),
            live_before_term: SecondaryMap::new(),
        };
        let repeating_blocks = cfg_cycle_blocks(&checker.cfg_successor_indices());
        let repeating_stmts = checker
            .body
            .blocks
            .iter()
            .enumerate()
            .filter(|(idx, _)| repeating_blocks.contains(&SBlockId::new(*idx)))
            .flat_map(|(_, block)| block.stmts.iter().map(|stmt| stmt.id))
            .collect::<FxHashSet<_>>();
        checker
            .fresh_local_allocations
            .retain(|(stmt, _)| repeating_stmts.contains(stmt));
        for sites in checker.fresh_local_sites.values_mut() {
            sites.retain(|stmt| repeating_stmts.contains(stmt));
        }
        checker
            .fresh_local_sites
            .retain(|_, sites| !sites.is_empty());
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
        if return_borrow_results_in_ty(self.db, key.callable_body(self.db).result_ty(self.db))
            .is_empty()
            || key.owner(self.db).body(self.db).is_none()
        {
            return Ok(None);
        }
        self.compute_entry_states();
        self.compute_loan_targets()?;
        self.check_provider_provenance()?;
        self.compute_return_summary().map(Some)
    }

    fn check(mut self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        self.compute_entry_states();
        self.compute_loan_targets()?;
        self.compute_fresh_value_roots()?;
        self.check_provider_provenance()?;
        self.compute_moved_states()?;
        self.compute_liveness();
        self.check_conflicts()?;
        let key = self.instance.key(self.db);
        if !return_borrow_results_in_ty(self.db, key.callable_body(self.db).result_ty(self.db))
            .is_empty()
        {
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
            if let Some((kind, _)) = ty_is_borrow(self.db, local.ty)
                && let Some(&param_idx) = self.param_index_of_local.get(&local_id)
                && (!matches!(
                    local.lowering,
                    NormalizedBindingLowering::CarrierLocal { .. }
                ) || self
                    .param_modes
                    .get(param_idx as usize)
                    .is_some_and(|mode| *mode != FuncParamMode::Own)
                    && !matches!(
                        local.source,
                        Some(LocalBinding::Param {
                            site: ParamSite::ClosureEnv(_),
                            ..
                        })
                    ))
            {
                let kind = if matches!(local.source, Some(LocalBinding::Param { is_mut: true, .. }))
                {
                    BorrowKind::Mut
                } else {
                    kind
                };
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
            if let Some((kind, _)) = ty_is_borrow(self.db, result_ty)
                && matches!(
                    expr,
                    NExpr::Borrow { .. }
                        | NExpr::Call { .. }
                        | NExpr::ReadPlace { .. }
                        | NExpr::Use(_)
                )
                && !matches!(
                    expr,
                    NExpr::ReadPlace { place, .. }
                        if self.read_place_copies_capability(place)
                )
            {
                let loan = self.allocate_loan(Loan {
                    kind,
                    activation: match expr {
                        NExpr::Borrow { activation, .. } => *activation,
                        NExpr::Call { .. } | NExpr::ReadPlace { .. } | NExpr::Use(_) => {
                            BorrowActivation::Immediate
                        }
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
            if return_borrow_results_in_ty(self.db, result_ty).is_empty() {
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

    fn read_place_copies_capability(&self, place: &NSPlace<'db>) -> bool {
        self.body
            .place_root_ty(&place.root)
            .and_then(|ty| semantic_projection_ty(self.db, ty, &place.path))
            .is_some_and(|(ty, _)| ty.as_capability(self.db).is_some())
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
        let results = return_borrow_results_in_ty(self.db, result_ty);
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
        let mut analysis = BorrowLoanTargetAnalysis::new(BorrowLoanTargetInputs {
            db: self.db,
            body: &self.body,
            entry_state: &self.entry_state,
            loan_for_local: &self.loan_for_local,
            constant_indices: &self.constant_indices,
            call_result_loans: &self.call_result_loans,
            call_loan_transforms: &self.call_loan_transforms,
            fresh_call_args: &self.fresh_call_args,
        });
        let mut state = BorrowLoanTargetState {
            loans: &mut self.loans,
        };
        try_solve_sparse(&mut analysis, &mut state)
    }

    fn compute_fresh_value_roots(&mut self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let successors = self.cfg_successor_indices();
        let predecessors = self.cfg_predecessor_indices();
        let backedges = cfg_backedges(&successors);

        let mut entry = vec![FreshLocalStatus::default(); self.body.blocks.len()];
        let mut exit = entry.clone();
        loop {
            let mut changed = false;
            for block_idx in 0..self.body.blocks.len() {
                let block = SBlockId::new(block_idx);
                let mut ignored = FreshStatusAtStmt::default();
                let new_exit =
                    self.transfer_fresh_value_roots(block, &entry[block_idx], &mut ignored)?;
                if new_exit != exit[block_idx] {
                    exit[block_idx] = new_exit;
                    changed = true;
                }
            }
            for (block_idx, block_entry) in entry.iter_mut().enumerate().skip(1) {
                let block = SBlockId::new(block_idx);
                let preds = &predecessors[block];
                let mut candidates = FreshLocalStatus::default();
                let mut predecessor_counts = FxHashMap::<SLocalId, usize>::default();
                for pred in preds {
                    let crosses_backedge = backedges.contains(&(*pred, block));
                    for (&local, status) in &exit[pred.index()] {
                        *predecessor_counts.entry(local).or_default() += 1;
                        let candidate =
                            candidates.entry(local).or_insert_with(|| FreshValueStatus {
                                roots: FreshRootStates::default(),
                                complete: true,
                            });
                        candidate.complete &= status.complete;
                        for (&root, root_statuses) in &status.roots {
                            for root_status in root_statuses {
                                let mut root_status = root_status.clone();
                                if crosses_backedge {
                                    root_status.generation = FreshGeneration::Stale;
                                }
                                merge_fresh_root_status(&mut candidate.roots, root, root_status);
                            }
                        }
                    }
                }
                for (&local, status) in &mut candidates {
                    status.complete &= predecessor_counts.get(&local) == Some(&preds.len());
                }
                if candidates != *block_entry {
                    *block_entry = candidates;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut status_at_stmt = FreshStatusAtStmt::default();
        for (block_idx, block_entry) in entry.iter().enumerate() {
            self.transfer_fresh_value_roots(
                SBlockId::new(block_idx),
                block_entry,
                &mut status_at_stmt,
            )?;
        }
        self.fresh_status_at_stmt = status_at_stmt;
        Ok(())
    }

    fn transfer_fresh_value_roots(
        &self,
        block: SBlockId,
        entry: &FreshLocalStatus<'db>,
        status_at_stmt: &mut FreshStatusAtStmt<'db>,
    ) -> Result<FreshLocalStatus<'db>, SemanticBorrowDiagnostic<'db>> {
        let mut current = entry.clone();
        let mut state = self.entry_state[block].clone();
        for stmt in &self.body.blocks[block.index()].stmts {
            let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
                let NSStmtKind::Store { dst, src } = &stmt.kind else {
                    unreachable!()
                };
                if let Some(status) = current.get(&src.local) {
                    status_at_stmt.insert((stmt.id, src.local), status.clone());
                }
                if let NSPlaceRoot::CarrierDerefLocal(local) = dst.root
                    && let Some(status) = current.get(&local)
                {
                    status_at_stmt.insert((stmt.id, local), status.clone());
                }
                let Some(base) = dst
                    .root
                    .borrow_root()
                    .and_then(|root| self.canon().root_base_local(root))
                else {
                    self.apply_stmt_state(&mut state, stmt);
                    continue;
                };
                let Some(projection) = self.canon().layout_path(&dst.path) else {
                    self.apply_stmt_state(&mut state, stmt);
                    current.remove(&base);
                    continue;
                };
                if self
                    .body
                    .place_root_ty(&dst.root)
                    .and_then(|ty| semantic_projection_ty(self.db, ty, &dst.path))
                    .is_some_and(|(_, traverses_capability)| traverses_capability)
                {
                    self.apply_stmt_state(&mut state, stmt);
                    continue;
                }

                let source = current
                    .get(&src.local)
                    .cloned()
                    .unwrap_or_else(|| self.unknown_fresh_status(&state, src.local));
                self.mark_fresh_retention_site(&mut current, &source, stmt.id);
                let replacement = current
                    .get(&src.local)
                    .cloned()
                    .unwrap_or_else(|| self.unknown_fresh_status(&state, src.local));
                let mut status = current
                    .remove(&base)
                    .unwrap_or_else(|| self.unknown_fresh_status(&state, base));
                replace_fresh_value_projection(&mut status, &projection, replacement);
                self.apply_stmt_state(&mut state, stmt);
                self.filter_fresh_status(&state, base, &mut status);
                if !status.roots.is_empty() {
                    current.insert(base, status);
                }
                continue;
            };
            expr.for_each_value_operand(|value| {
                if let Some(status) = current.get(&value.local) {
                    status_at_stmt.insert((stmt.id, value.local), status.clone());
                }
            });
            let retains_operands = matches!(
                expr,
                NExpr::AggregateMake { .. } | NExpr::EnumMake { .. } | NExpr::ArrayRepeat { .. }
            ) || self
                .body
                .local(*dst)
                .is_some_and(|local| local.source.is_some());
            if retains_operands {
                let mut operands = Vec::new();
                expr.for_each_value_operand(|value| operands.push(value.local));
                for local in operands {
                    let source = current
                        .get(&local)
                        .cloned()
                        .unwrap_or_else(|| self.unknown_fresh_status(&state, local));
                    self.mark_fresh_retention_site(&mut current, &source, stmt.id);
                }
            }
            let previous_dst_status = current.get(dst).cloned();

            let mut status = match expr {
                NExpr::Use(value) | NExpr::Cast { value, .. } => {
                    current.get(&value.local).cloned().unwrap_or_default()
                }
                NExpr::ArrayRepeat { value, .. } => prefix_fresh_value_status(
                    current.get(&value.local).cloned().unwrap_or_default(),
                    &[LayoutBackingProjection::Index(None)],
                ),
                NExpr::ExtractEnumField {
                    value,
                    variant,
                    field,
                } => current
                    .get(&value.local)
                    .map(|status| {
                        project_fresh_value_status(
                            status,
                            &[LayoutBackingProjection::VariantField {
                                variant: *variant,
                                field: *field,
                            }],
                        )
                    })
                    .unwrap_or_default(),
                NExpr::ReadPlace { place, .. } | NExpr::Borrow { place, .. } => {
                    let base = self.canon().place_base_local(place);
                    let projection = self.canon().layout_path(&place.path).unwrap_or_default();
                    let mut status = base
                        .and_then(|local| current.get(&local))
                        .map(|status| project_fresh_value_status(status, &projection))
                        .unwrap_or_default();
                    if status.roots.is_empty()
                        && let Some(local) = base.and_then(|local| self.body.local(local))
                    {
                        for source in local.layout_backing_sources() {
                            let Some(source_status) = self
                                .canon()
                                .place_base_local(&source.source)
                                .and_then(|local| current.get(&local))
                            else {
                                continue;
                            };
                            merge_fresh_value_status(&mut status, source_status.clone());
                        }
                    }
                    status
                }
                NExpr::AggregateMake { ty, fields } => {
                    let mut status = FreshValueStatus {
                        roots: FreshRootStates::default(),
                        complete: true,
                    };
                    for (idx, field) in fields.iter().enumerate() {
                        if let Some(field_status) = current.get(&field.local) {
                            let projection = if ty.is_array(self.db) {
                                LayoutBackingProjection::Index(Some(idx))
                            } else {
                                let Ok(field) = u16::try_from(idx) else {
                                    status.complete = false;
                                    continue;
                                };
                                LayoutBackingProjection::Field(FieldIndex(field))
                            };
                            merge_fresh_value_status(
                                &mut status,
                                prefix_fresh_value_status(field_status.clone(), &[projection]),
                            );
                        } else if !self
                            .fresh_storage_roots_for_value(&state, field.local)
                            .is_empty()
                        {
                            status.complete = false;
                        }
                    }
                    status
                }
                NExpr::EnumMake {
                    variant, fields, ..
                } => {
                    let mut status = FreshValueStatus {
                        roots: FreshRootStates::default(),
                        complete: true,
                    };
                    for (idx, field) in fields.iter().enumerate() {
                        if let Some(field_status) = current.get(&field.local) {
                            let Ok(field) = u16::try_from(idx) else {
                                status.complete = false;
                                continue;
                            };
                            merge_fresh_value_status(
                                &mut status,
                                prefix_fresh_value_status(
                                    field_status.clone(),
                                    &[LayoutBackingProjection::VariantField {
                                        variant: *variant,
                                        field: FieldIndex(field),
                                    }],
                                ),
                            );
                        } else if !self
                            .fresh_storage_roots_for_value(&state, field.local)
                            .is_empty()
                        {
                            status.complete = false;
                        }
                    }
                    status
                }
                NExpr::Call { args, .. } => {
                    let direct = self
                        .loan_for_local
                        .get(dst)
                        .copied()
                        .map(|loan| (Vec::new(), loan))
                        .into_iter();
                    let aggregate = self
                        .call_result_loans
                        .get(&stmt.id)
                        .into_iter()
                        .flatten()
                        .map(|(result, loan)| (result.projection.clone(), *loan));
                    let mut status = FreshValueStatus {
                        roots: FreshRootStates::default(),
                        complete: true,
                    };
                    for (result_projection, loan_id) in direct.chain(aggregate) {
                        let Some(transforms) = self.call_loan_transforms.get(&loan_id) else {
                            status.complete = false;
                            continue;
                        };
                        for transform in transforms {
                            let Some(arg) = args.get(transform.input.param() as usize) else {
                                status.complete = false;
                                continue;
                            };
                            let targets = self.canon().canonicalize_call_input(
                                &state,
                                stmt.id,
                                arg.local,
                                &transform.input,
                                self.fresh_call_args.contains(&(stmt.id, arg.local)),
                            );
                            for target in targets {
                                let arg_status = match &transform.input {
                                    BorrowInput::Place { projection, .. } => current
                                        .get(&arg.local)
                                        .map(|status| {
                                            project_fresh_value_status(status, projection)
                                        })
                                        .unwrap_or_else(|| {
                                            self.unknown_fresh_status(&state, arg.local)
                                        }),
                                    BorrowInput::AnyInParam(_) => {
                                        current.get(&arg.local).cloned().unwrap_or_else(|| {
                                            self.unknown_fresh_status(&state, arg.local)
                                        })
                                    }
                                };
                                let fresh_call_root = FreshStorageRoot::Call {
                                    stmt: stmt.id,
                                    source: arg.local,
                                };
                                if matches!(
                                    target.root,
                                    BorrowRoot::FreshCall {
                                        stmt: root_stmt,
                                        source,
                                    } if root_stmt == stmt.id && source == arg.local
                                ) {
                                    merge_fresh_root_status(
                                        &mut status.roots,
                                        fresh_call_root,
                                        FreshRootStatus {
                                            generation: FreshGeneration::Current,
                                            retention_sites: FxHashSet::default(),
                                            carriers: FxHashSet::from_iter([
                                                result_projection.clone()
                                            ]),
                                            claimed: FxHashSet::default(),
                                        },
                                    );
                                    continue;
                                }

                                let mut found = false;
                                for (&root, root_statuses) in &arg_status.roots {
                                    if !fresh_storage_root_matches(&root, &target.root) {
                                        continue;
                                    }
                                    found = true;
                                    for root_status in root_statuses {
                                        merge_fresh_root_status(
                                            &mut status.roots,
                                            root,
                                            prefix_fresh_root_status(
                                                root_status.clone(),
                                                &result_projection,
                                            ),
                                        );
                                    }
                                }
                                if !found && !arg_status.complete {
                                    status.complete = false;
                                }
                            }
                        }
                    }
                    status
                }
                NExpr::CodeRegionRef { .. }
                | NExpr::Const(_)
                | NExpr::Unary { .. }
                | NExpr::Binary { .. }
                | NExpr::GetEnumTag { .. }
                | NExpr::IsEnumVariant { .. }
                | NExpr::CodeRegionOffset { .. }
                | NExpr::CodeRegionLen { .. } => FreshValueStatus {
                    roots: FreshRootStates::default(),
                    complete: true,
                },
            };
            if self.fresh_local_allocations.contains(&(stmt.id, *dst)) {
                merge_fresh_root_status(
                    &mut status.roots,
                    FreshStorageRoot::Local {
                        definition: stmt.id,
                        local: *dst,
                    },
                    FreshRootStatus {
                        generation: FreshGeneration::Current,
                        retention_sites: FxHashSet::default(),
                        carriers: FxHashSet::from_iter([FreshCarrierPath::new()]),
                        claimed: FxHashSet::default(),
                    },
                );
            } else if self.fresh_local_sites.contains_key(dst)
                && let Some(previous_dst_status) = previous_dst_status
            {
                for (root, root_statuses) in previous_dst_status.roots {
                    if matches!(
                        root,
                        FreshStorageRoot::Local {
                            local: root_local,
                            ..
                        } if root_local == *dst
                    ) {
                        for root_status in root_statuses {
                            merge_fresh_root_status(&mut status.roots, root, root_status);
                        }
                    }
                }
            }
            current.remove(dst);
            self.apply_stmt_state(&mut state, stmt);
            self.filter_fresh_status(&state, *dst, &mut status);
            status_at_stmt.insert((stmt.id, *dst), status.clone());
            let direct = self.loan_for_local.get(dst).copied().into_iter();
            let aggregate = self
                .call_result_loans
                .get(&stmt.id)
                .into_iter()
                .flatten()
                .map(|(_, loan)| *loan);
            let claimed_targets = direct
                .chain(aggregate)
                .flat_map(|loan| self.loans[loan.0 as usize].targets.iter())
                .collect::<Vec<_>>();
            for value_status in current.values_mut() {
                claim_fresh_value_status(value_status, &status, &claimed_targets);
            }
            if !status.roots.is_empty() {
                current.insert(*dst, status);
            }
        }
        Ok(current)
    }

    fn unknown_fresh_status(&self, state: &State, local: SLocalId) -> FreshValueStatus<'db> {
        FreshValueStatus {
            roots: FreshRootStates::default(),
            complete: self.fresh_storage_roots_for_value(state, local).is_empty(),
        }
    }

    fn filter_fresh_status(
        &self,
        state: &State,
        local: SLocalId,
        status: &mut FreshValueStatus<'db>,
    ) {
        let possible_roots = self.fresh_storage_roots_for_value(state, local);
        status
            .roots
            .retain(|root, statuses| possible_roots.contains(root) && !statuses.is_empty());
        if possible_roots
            .iter()
            .all(|root| status.roots.contains_key(root))
        {
            status.complete = true;
        }
    }

    fn mark_fresh_retention_site(
        &self,
        current: &mut FreshLocalStatus<'db>,
        source: &FreshValueStatus<'db>,
        stmt: SStmtId,
    ) {
        for value_status in current.values_mut() {
            for (&root, source_statuses) in &source.roots {
                let Some(statuses) = value_status.roots.get_mut(&root) else {
                    continue;
                };
                for status in &mut *statuses {
                    if status.generation == FreshGeneration::Current
                        && source_statuses.iter().any(|source_status| {
                            fresh_root_statuses_may_alias(status, source_status)
                        })
                    {
                        // Rename every alias of the current allocation
                        // together. Sequential copies therefore keep one
                        // identity, while mutually exclusive retention sites
                        // can distinguish values retained by later loop
                        // iterations.
                        status.retention_sites.clear();
                        status.retention_sites.insert(stmt);
                    }
                }
                normalize_fresh_root_statuses(statuses);
            }
        }
    }

    fn fresh_storage_roots_for_value(
        &self,
        state: &State,
        local: SLocalId,
    ) -> FxHashSet<FreshStorageRoot> {
        let mut roots = FxHashSet::default();
        for target in self.canon().all_value_targets(state, local) {
            self.extend_fresh_storage_roots(&mut roots, &target.root);
        }
        roots
    }

    fn extend_fresh_storage_roots(
        &self,
        roots: &mut FxHashSet<FreshStorageRoot>,
        root: &BorrowRoot<'db>,
    ) {
        match root {
            BorrowRoot::FreshCall { stmt, source } => {
                roots.insert(FreshStorageRoot::Call {
                    stmt: *stmt,
                    source: *source,
                });
            }
            BorrowRoot::Local(local) => {
                roots.extend(self.fresh_local_sites.get(local).into_iter().flatten().map(
                    |definition| FreshStorageRoot::Local {
                        definition: *definition,
                        local: *local,
                    },
                ));
            }
            BorrowRoot::Param(_) | BorrowRoot::Provider(_) => {}
        }
    }

    pub(super) fn apply_stmt_state(&self, state: &mut State, stmt: &super::ir::NSStmt<'db>) {
        self.canon().apply_stmt_state_with_call_loans(
            state,
            stmt,
            self.call_result_loans.get(&stmt.id).map(Vec::as_slice),
        );
    }

    fn check_provider_provenance(&self) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let mut observed = FxHashMap::<
            (
                crate::analysis::semantic::SLocalId,
                Vec<LayoutBackingProjection>,
            ),
            (ProviderAddressSpace, SemOrigin<'db>),
        >::default();
        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            let mut state = self.entry_state[SBlockId::new(bb_idx)].clone();
            self.observe_provider_slots(&state, &mut observed)?;
            for stmt in &block.stmts {
                self.apply_stmt_state(&mut state, stmt);
                self.observe_provider_slots(&state, &mut observed)?;
            }
        }
        Ok(())
    }

    fn observe_provider_slots(
        &self,
        state: &State,
        observed: &mut FxHashMap<
            (
                crate::analysis::semantic::SLocalId,
                Vec<LayoutBackingProjection>,
            ),
            (ProviderAddressSpace, SemOrigin<'db>),
        >,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        for (&local, held) in &state.local_loans {
            for (path, loans) in held {
                let slot = path
                    .iter()
                    .copied()
                    .map(|projection| match projection {
                        LayoutBackingProjection::Index(_) => LayoutBackingProjection::Index(None),
                        projection => projection,
                    })
                    .collect::<Vec<_>>();
                let key = (local, slot);
                for &loan in loans {
                    let origin = self.loan_origin(loan);
                    for target in &self.loans[loan.0 as usize].targets {
                        let Some(space) = known_address_space_for_borrow_root(&target.root) else {
                            continue;
                        };
                        let Some((previous_space, previous_origin)) = observed.get(&key) else {
                            observed.insert(key.clone(), (space, origin));
                            continue;
                        };
                        if *previous_space == space {
                            continue;
                        }
                        let mut spaces = [*previous_space, space];
                        spaces.sort_by_key(|space| address_space_rank(*space));
                        let mut diag = self.diag(
                            SemanticBorrowDiagKind::ProviderProvenanceConflict,
                            origin,
                            format!(
                                "borrow slot may come from multiple address spaces: {}, {}",
                                spaces[0].pretty(),
                                spaces[1].pretty(),
                            ),
                        );
                        self.push_secondary_origin(
                            &mut diag,
                            *previous_origin,
                            format!(
                                "the same borrow slot is {}-backed here",
                                previous_space.pretty()
                            ),
                        );
                        return Err(diag);
                    }
                }
            }
        }
        Ok(())
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
                let live_after = self.live_before[bb_idx]
                    .get(stmt_idx + 1)
                    .unwrap_or(&self.live_before_term[bb]);
                self.check_stmt(
                    &state,
                    &moved,
                    &self.live_before[bb_idx][stmt_idx],
                    live_after,
                    stmt,
                )?;
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
        live_before: &FxHashSet<crate::analysis::semantic::SLocalId>,
        live_after: &FxHashSet<crate::analysis::semantic::SLocalId>,
        stmt: &super::ir::NSStmt<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let active = self.effective_loans(state, live_before);
        let mut state_after = state.clone();
        self.apply_stmt_state(&mut state_after, stmt);
        let mut live_at_write = live_after.clone();
        match &stmt.kind {
            NSStmtKind::Assign { dst, .. } => {
                live_at_write.insert(*dst);
            }
            NSStmtKind::Store { dst, .. } => {
                if let Some(base) = self.canon().place_base_local(dst) {
                    live_at_write.insert(base);
                }
            }
        }
        let active_at_write = self.effective_loans(&state_after, &live_at_write);
        let check = ConflictCheckCx {
            state,
            moved,
            active: &active,
        };
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => {
                match expr {
                    NExpr::ReadPlace { place, mode } => {
                        self.check_place_read(
                            check,
                            place,
                            *mode,
                            stmt.origin,
                            self.fresh_status_at_stmt.get(&(stmt.id, *dst)),
                        )?;
                    }
                    NExpr::Borrow { place, kind, .. } => {
                        let targets = self.canon().canonicalize_place(state, place, stmt.origin)?;
                        let authorized = self.canon().loans_for_place(state, place);
                        self.check_moved_overlap(
                            moved,
                            &targets,
                            &authorized,
                            stmt.origin,
                            "cannot borrow a moved value",
                        )?;
                        let loan = self.loan_for_local.get(dst).copied();
                        if let Some(kind) =
                            loan.map_or(Some(*kind), |loan| self.loan_conflict_kind(loan))
                        {
                            self.check_loan_conflict(
                                &active,
                                loan,
                                kind,
                                &targets,
                                stmt.origin,
                                self.fresh_status_at_stmt.get(&(stmt.id, *dst)),
                            )?;
                        }
                    }
                    NExpr::ExtractEnumField {
                        value,
                        variant,
                        field,
                    } => {
                        let targets =
                            self.extract_enum_field_move_targets(state, *value, *variant, *field);
                        let authorized =
                            self.canon()
                                .loans_for_value_targets(state, value.local, &targets);
                        self.check_moved_overlap(
                            moved,
                            &targets,
                            &authorized,
                            stmt.origin,
                            "cannot use a value after it was moved",
                        )?;
                        if value.mode == ReadMode::Move {
                            self.check_move_targets_out(
                                &active,
                                &targets,
                                stmt.origin,
                                self.fresh_status_at_stmt.get(&(stmt.id, *dst)),
                            )?;
                        } else {
                            self.check_read_targets(
                                &active,
                                &authorized,
                                &targets,
                                stmt.origin,
                                self.fresh_status_at_stmt.get(&(stmt.id, *dst)),
                            )?;
                        }
                    }
                    _ => {
                        let expression_moved = self.check_expr_operands(
                            state,
                            moved,
                            &active,
                            stmt.id,
                            stmt.origin,
                            expr,
                        )?;
                        let mut call_accesses = self.check_call_argument_accesses(
                            state,
                            &active,
                            stmt.id,
                            stmt.origin,
                            expr,
                        )?;
                        self.check_effect_place_accesses(
                            ConflictCheckCx {
                                moved: &expression_moved,
                                ..check
                            },
                            stmt.id,
                            stmt.origin,
                            expr,
                            &mut call_accesses,
                        )?;
                    }
                }
                if matches!(expr, NExpr::Call { .. } | NExpr::Use(_)) {
                    self.check_assigned_loan_conflicts(
                        &active,
                        stmt.id,
                        *dst,
                        stmt.origin,
                        self.fresh_status_at_stmt.get(&(stmt.id, *dst)),
                    )?;
                }
                self.check_assignment_write(
                    state,
                    &active_at_write,
                    *dst,
                    stmt.origin,
                    self.fresh_status_at_stmt.get(&(stmt.id, *dst)),
                )?;
            }
            NSStmtKind::Store { dst, src } => {
                self.check_operand(
                    check,
                    *src,
                    Some(stmt.id),
                    stmt.origin,
                    "cannot use a value after it was moved",
                )?;
                let targets = self.canon().canonicalize_place(state, dst, stmt.origin)?;
                let authorized = self.canon().mut_loans_for_place(state, dst);
                self.check_moved_parent(moved, &targets, &authorized, stmt.origin)?;
                let fresh_status = match dst.root {
                    NSPlaceRoot::CarrierDerefLocal(local) => {
                        self.fresh_status_at_stmt.get(&(stmt.id, local))
                    }
                    NSPlaceRoot::Root(_) => None,
                };
                self.check_write_targets(
                    &active_at_write,
                    &authorized,
                    &targets,
                    stmt.origin,
                    fresh_status,
                )?;
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
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let Some(local) = self
            .body
            .local(dst)
            .filter(|local| local.source.is_some_and(|binding| binding.is_mut()))
        else {
            return Ok(());
        };
        if local.is_derived_place_bound_alias() {
            return Ok(());
        }
        let Some(place) = local.lowering.place() else {
            return Ok(());
        };
        let targets = self.canon().canonicalize_place(state, place, origin)?;
        self.check_write_targets(
            active,
            &FxHashSet::default(),
            &targets,
            origin,
            fresh_status,
        )
    }

    fn check_place_read(
        &self,
        check: ConflictCheckCx<'_, 'db>,
        place: &NSPlace<'db>,
        mode: ReadMode,
        origin: SemOrigin<'db>,
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let targets = self
            .canon()
            .canonicalize_place(check.state, place, origin)?;
        let authorized = self.canon().loans_for_place(check.state, place);
        self.check_moved_overlap(
            check.moved,
            &targets,
            &authorized,
            origin,
            "cannot use a value after it was moved",
        )?;
        if mode == ReadMode::Move {
            self.check_move_out(check.active, place, &targets, origin, fresh_status)
        } else {
            self.check_read_targets(check.active, &authorized, &targets, origin, fresh_status)
        }
    }

    fn check_effect_place_accesses(
        &self,
        check: ConflictCheckCx<'_, 'db>,
        stmt: SStmtId,
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
                    let targets = self
                        .canon()
                        .canonicalize_place(check.state, place, origin)?;
                    let authorized = if effect_arg.required_mut {
                        self.canon().mut_loans_for_place(check.state, place)
                    } else {
                        self.canon().loans_for_place(check.state, place)
                    };
                    self.check_moved_overlap(
                        check.moved,
                        &targets,
                        &authorized,
                        origin,
                        "cannot use a value after it was moved",
                    )?;
                    (targets, authorized, origin)
                }
                NEffectArgValue::Value(value) => {
                    let targets = self
                        .canon()
                        .canonicalize_value_base(check.state, value.local);
                    let authorized = if effect_arg.required_mut {
                        self.canon()
                            .mut_loans_for_value_targets(check.state, value.local, &targets)
                    } else {
                        self.canon()
                            .loans_for_value_targets(check.state, value.local, &targets)
                    };
                    (targets, authorized, operand_origin(*value, origin))
                }
            };
            let fresh_status = match &effect_arg.arg {
                NEffectArgValue::Value(value) => {
                    self.fresh_status_at_stmt.get(&(stmt, value.local))
                }
                NEffectArgValue::Place(_) => None,
            };
            if effect_arg.required_mut {
                self.check_write_targets(
                    check.active,
                    &authorized,
                    &targets,
                    arg_origin,
                    fresh_status,
                )?;
                self.record_call_access(
                    accesses,
                    group,
                    CallAccessRole::Container,
                    BorrowKind::Mut,
                    targets,
                    arg_origin,
                )?;
            } else {
                self.check_read_targets(
                    check.active,
                    &authorized,
                    &targets,
                    arg_origin,
                    fresh_status,
                )?;
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
                            check.state,
                            place,
                            target_ty,
                            &result.projection,
                        )
                    }
                    NEffectArgValue::Value(value) => {
                        self.canon().canonicalize_value_layout_projection(
                            check.state,
                            value.local,
                            &result.projection,
                        )
                    }
                };
                let authorized = match &effect_arg.arg {
                    NEffectArgValue::Place(place) if result.kind == BorrowKind::Mut => self
                        .canon()
                        .mut_loans_for_place_targets(check.state, place, &targets),
                    NEffectArgValue::Place(place) => {
                        self.canon()
                            .loans_for_place_targets(check.state, place, &targets)
                    }
                    NEffectArgValue::Value(value) if result.kind == BorrowKind::Mut => self
                        .canon()
                        .mut_loans_for_value_targets(check.state, value.local, &targets),
                    NEffectArgValue::Value(value) => {
                        self.canon()
                            .loans_for_value_targets(check.state, value.local, &targets)
                    }
                };
                if result.kind == BorrowKind::Mut {
                    self.check_write_targets(
                        check.active,
                        &authorized,
                        &targets,
                        arg_origin,
                        fresh_status,
                    )?;
                } else {
                    self.check_read_targets(
                        check.active,
                        &authorized,
                        &targets,
                        arg_origin,
                        fresh_status,
                    )?;
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
        stmt: SStmtId,
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
            let fresh_status = self.fresh_status_at_stmt.get(&(stmt, arg.local));
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
                    self.check_write_targets(
                        active,
                        &authorized,
                        &targets,
                        arg_origin,
                        fresh_status,
                    )?;
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
                    self.check_write_targets(
                        active,
                        &authorized,
                        &targets,
                        arg_origin,
                        fresh_status,
                    )?;
                } else {
                    self.check_read_targets(
                        active,
                        &authorized,
                        &targets,
                        arg_origin,
                        fresh_status,
                    )?;
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
        fresh_status: Option<&FreshValueStatus<'db>>,
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
            self.check_loan_conflict(
                active,
                Some(loan_id),
                loan.kind,
                &loan.targets,
                origin,
                fresh_status,
            )?;
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
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let Some(conflict) =
            self.first_loan_conflict(active, new_loan, kind, targets, fresh_status)
        {
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
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let conflict = active.iter().copied().find(|loan| {
            !authorized.contains(loan)
                && self.loan_conflict_kind(*loan) == Some(BorrowKind::Mut)
                && place_sets_conflict(
                    &self.loans[loan.0 as usize].targets,
                    targets,
                    fresh_status,
                    &self.fresh_local_sites,
                )
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
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let conflict = active.iter().copied().find(|loan| {
            !authorized.contains(loan)
                && self.loan_conflict_kind(*loan).is_some()
                && place_sets_conflict(
                    &self.loans[loan.0 as usize].targets,
                    targets,
                    fresh_status,
                    &self.fresh_local_sites,
                )
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
                    ConflictCheckCx {
                        state,
                        moved,
                        active: &active,
                    },
                    *cond,
                    None,
                    term.origin,
                    "cannot use a value after it was moved",
                )?;
            }
            NSTerminatorKind::MatchEnum { value, .. } => {
                let active = self.effective_loans(state, live);
                self.check_operand(
                    ConflictCheckCx {
                        state,
                        moved,
                        active: &active,
                    },
                    NOperand {
                        mode: ReadMode::Read,
                        ..*value
                    },
                    None,
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
                .is_some_and(|local| ty_is_borrow(self.db, local.ty).is_some())
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
                .is_some_and(|local| ty_is_borrow(self.db, local.ty).is_none())
        {
            for loan_id in state.loans_in(value.local) {
                let loan = &self.loans[loan_id.0 as usize];
                let targets = self.resolve_return_targets(state, &loan.targets, term.origin)?;
                if let Some(target) = targets.iter().find(|target| {
                    matches!(
                        target.root,
                        BorrowRoot::Local(_) | BorrowRoot::FreshCall { .. }
                    )
                }) {
                    let message = match target.root {
                        BorrowRoot::Local(local) => {
                            let name = self.pretty_local_name(local);
                            format!("cannot return a value that holds a borrow of local `{name}`")
                        }
                        BorrowRoot::FreshCall { .. } => {
                            "cannot return a value that holds a borrow of a temporary created in this function"
                                .to_string()
                        }
                        BorrowRoot::Param(_) | BorrowRoot::Provider(_) => unreachable!(),
                    };
                    let mut diag = self.invalid_return_diag(term.origin, message);
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

    fn resolve_return_targets(
        &self,
        state: &State,
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<FxHashSet<CanonPlace<'db>>, SemanticBorrowDiagnostic<'db>> {
        let mut pending = targets.iter().cloned().collect::<VecDeque<_>>();
        let mut seen = FxHashSet::default();
        let mut resolved = FxHashSet::default();
        while let Some(target) = pending.pop_front() {
            if !seen.insert(target.clone()) {
                continue;
            }
            let BorrowRoot::Local(local) = &target.root else {
                resolved.insert(target);
                continue;
            };
            let Some(local) = self.body.local(*local) else {
                resolved.insert(target);
                continue;
            };
            // Snapshot provenance describes a value's physical source. Layout
            // backing is deliberately not used here: it can point at the
            // argument that supplied a fresh aggregate field's layout without
            // making that freshly allocated field an alias of the argument.
            let Some(mut source) = local.snapshot_source_place().cloned() else {
                resolved.insert(target);
                continue;
            };
            source.path = source.path.concat(&target.proj);
            let mut advanced = false;
            for source in self.canon().canonicalize_place(state, &source, origin)? {
                if source == target {
                    continue;
                }
                advanced = true;
                pending.push_back(source);
            }
            if !advanced {
                resolved.insert(target);
            }
        }
        Ok(resolved)
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
            for result in return_borrow_results_in_ty(self.db, return_local.ty) {
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
                let targets =
                    self.resolve_return_targets(&state, &targets, block.terminator.origin)?;
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
                        BorrowRoot::FreshCall { .. } => {
                            return Err(self.invalid_return_diag(
                                block.terminator.origin,
                                "cannot return a borrow to a temporary created in this function"
                                    .to_string(),
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
        fresh_status: Option<&FreshValueStatus<'db>>,
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
                }) && place_sets_conflict(
                    &loan.targets,
                    targets,
                    fresh_status,
                    &self.fresh_local_sites,
                )
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
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let NSPlaceRoot::CarrierDerefLocal(local) = place.root {
            if let Some(message) = self
                .logical_closure_param_mode_for_local(local)
                .and_then(Self::logical_param_move_message)
            {
                return Err(self.move_conflict_diag(origin, message.to_string()));
            }
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
        self.check_move_targets_out(active, targets, origin, fresh_status)?;
        Ok(())
    }

    fn check_move_targets_out(
        &self,
        active: &[LoanId],
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        fresh_status: Option<&FreshValueStatus<'db>>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        for target in targets {
            if let Some(message) = self
                .logical_closure_param_mode_for_target(target)
                .and_then(Self::logical_param_move_message)
            {
                return Err(self.move_conflict_diag(origin, message.to_string()));
            }
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
        if let Some(loan) = active.iter().copied().find(|loan| {
            place_sets_conflict(
                &self.loans[loan.0 as usize].targets,
                targets,
                fresh_status,
                &self.fresh_local_sites,
            )
        }) {
            return Err(self.borrow_conflict_diag(
                origin,
                "cannot move out of a value while it is borrowed".to_string(),
                loan,
            ));
        }
        Ok(())
    }

    fn logical_closure_param_mode_for_local(
        &self,
        local: crate::analysis::semantic::SLocalId,
    ) -> Option<ClosureParamMode> {
        let source = self
            .body
            .local(local)?
            .facts
            .snapshot_source_place
            .as_ref()?;
        let NSPlaceRoot::Root(root) = source.root else {
            return None;
        };
        let NBorrowRoot::Param { param_idx, .. } = self.body.root(root)? else {
            return None;
        };
        self.logical_closure_param_mode(*param_idx, &source.path)
    }

    fn logical_closure_param_mode_for_target(
        &self,
        target: &CanonPlace<'db>,
    ) -> Option<ClosureParamMode> {
        let BorrowRoot::Param(param_idx) = target.root else {
            return None;
        };
        self.logical_closure_param_mode(param_idx, &target.proj)
    }

    fn logical_closure_param_mode(
        &self,
        param_idx: u32,
        projection: &NSProjectionPath<'db>,
    ) -> Option<ClosureParamMode> {
        let BodyOwner::Closure { ty, .. } = self.instance.key(self.db).owner(self.db) else {
            return None;
        };
        if usize::try_from(param_idx).ok()? != CLOSURE_ARGS_PARAM_IDX {
            return None;
        }
        let Projection::Field(field_idx) = projection.iter().next()? else {
            return None;
        };
        ty.param_modes(self.db).get(*field_idx).copied()
    }

    fn logical_param_move_message(mode: ClosureParamMode) -> Option<&'static str> {
        match mode {
            ClosureParamMode::Own => None,
            ClosureParamMode::View => Some("cannot move out of a view parameter"),
            ClosureParamMode::Ref | ClosureParamMode::Mut => {
                Some("cannot move out through a borrow handle")
            }
        }
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
        stmt: SStmtId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        expr: &NExpr<'db>,
    ) -> Result<MovedPlaces<'db>, SemanticBorrowDiagnostic<'db>> {
        let mut moved = moved.clone();
        expr.try_for_each_value_operand(|value| {
            self.check_operand(
                ConflictCheckCx {
                    state,
                    moved: &moved,
                    active,
                },
                value,
                Some(stmt),
                origin,
                "cannot use a value after it was moved",
            )?;
            self.record_operand_move(state, &mut moved, value, origin)
        })?;
        Ok(moved)
    }

    fn check_operand(
        &self,
        check: ConflictCheckCx<'_, 'db>,
        operand: NOperand,
        stmt: Option<SStmtId>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: &str,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        let origin = operand_origin(operand, origin);
        let targets = self
            .canon()
            .canonicalize_value_base(check.state, operand.local);
        if targets.is_empty() {
            return Ok(());
        }
        let authorized = self
            .canon()
            .loans_for_value_targets(check.state, operand.local, &targets);
        self.check_moved_overlap(check.moved, &targets, &authorized, origin, message)?;
        if operand.mode == ReadMode::Move && self.local_has_runtime_move_semantics(operand.local) {
            self.check_move_targets_out(
                check.active,
                &targets,
                origin,
                stmt.and_then(|stmt| self.fresh_status_at_stmt.get(&(stmt, operand.local))),
            )
        } else {
            self.check_read_targets(
                check.active,
                &authorized,
                &targets,
                origin,
                stmt.and_then(|stmt| self.fresh_status_at_stmt.get(&(stmt, operand.local))),
            )
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
        authorized: &FxHashSet<LoanId>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: &str,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let Some((_, site)) = moved.iter().find(|(moved, _)| {
            accessed.iter().any(|accessed| {
                places_overlap(moved, accessed)
                    && !self.loan_authorizes_access(authorized, accessed)
            })
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
        authorized: &FxHashSet<LoanId>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), SemanticBorrowDiagnostic<'db>> {
        if let Some((_, site)) = moved.iter().find(|(moved, _)| {
            written.iter().any(|written| {
                written.root == moved.root
                    && moved.proj.is_prefix_of(&written.proj)
                    && moved.proj != written.proj
                    && !self.loan_authorizes_access(authorized, written)
            })
        }) {
            let mut diag =
                self.move_conflict_diag(origin, "cannot write through a moved value".to_string());
            self.push_secondary_origin(&mut diag, site.origin, site.note.clone());
            return Err(diag);
        }
        Ok(())
    }

    fn loan_authorizes_access(
        &self,
        authorized: &FxHashSet<LoanId>,
        accessed: &CanonPlace<'db>,
    ) -> bool {
        authorized.iter().any(|loan| {
            self.loans[loan.0 as usize].targets.iter().any(|target| {
                target.root == accessed.root && target.proj.is_prefix_of(&accessed.proj)
            })
        })
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

fn place_sets_conflict<'db>(
    active: &FxHashSet<CanonPlace<'db>>,
    new: &FxHashSet<CanonPlace<'db>>,
    fresh_status: Option<&FreshValueStatus<'db>>,
    fresh_local_sites: &FxHashMap<SLocalId, Vec<SStmtId>>,
) -> bool {
    active.iter().any(|active_target| {
        new.iter().any(|new_target| {
            let current_root_conflicts = |status: &FreshRootStatus<'db>| {
                (status.generation == FreshGeneration::Stale && status.retention_sites.is_empty())
                    || status.claimed.iter().any(|claimed| {
                        places_overlap(
                            &CanonPlace {
                                root: new_target.root.clone(),
                                proj: claimed.clone(),
                            },
                            new_target,
                        )
                    })
            };
            let needs_conflict_check = match (&new_target.root, fresh_status) {
                (
                    BorrowRoot::FreshCall { stmt, source },
                    Some(FreshValueStatus { roots, complete }),
                ) => match roots.get(&FreshStorageRoot::Call {
                    stmt: *stmt,
                    source: *source,
                }) {
                    // The allocation site just executed, so active loans with
                    // the same static root refer to earlier dynamic instances
                    // unless this path has already loaned the same projection.
                    Some(statuses) => statuses.iter().any(current_root_conflicts),
                    // Canonical loan targets are path-unioned. A root omitted
                    // by a complete value state is unreachable on this path.
                    None => !complete,
                },
                (BorrowRoot::Local(local), Some(FreshValueStatus { roots, complete }))
                    if fresh_local_sites.contains_key(local) =>
                {
                    let mut statuses = roots.iter().filter_map(|(root, status)| {
                        matches!(
                            root,
                            FreshStorageRoot::Local {
                                local: root_local,
                                ..
                            } if root_local == local
                        )
                        .then_some(status.iter())
                    });
                    let first = statuses.next();
                    first.is_none() && !complete
                        || first
                            .into_iter()
                            .chain(statuses)
                            .flatten()
                            .any(current_root_conflicts)
                }
                _ => true,
            };
            needs_conflict_check && places_overlap(active_target, new_target)
        })
    })
}

fn fresh_storage_root_matches<'db>(root: &FreshStorageRoot, target: &BorrowRoot<'db>) -> bool {
    matches!(
        (root, target),
        (
            FreshStorageRoot::Call { stmt, source },
            BorrowRoot::FreshCall {
                stmt: target_stmt,
                source: target_source,
            },
        ) if stmt == target_stmt && source == target_source
    ) || matches!(
        (root, target),
        (
            FreshStorageRoot::Local { local, .. },
            BorrowRoot::Local(target_local),
        ) if local == target_local
    )
}

fn prefix_fresh_root_status<'db>(
    mut status: FreshRootStatus<'db>,
    prefix: &[LayoutBackingProjection],
) -> FreshRootStatus<'db> {
    status.carriers = status
        .carriers
        .iter()
        .map(|carrier| {
            let mut prefixed = prefix.to_vec();
            prefixed.extend_from_slice(carrier);
            prefixed
        })
        .collect();
    status
}

fn claim_fresh_value_status<'db>(
    candidate: &mut FreshValueStatus<'db>,
    source: &FreshValueStatus<'db>,
    targets: &[&CanonPlace<'db>],
) {
    for target in targets {
        let matching_source = source
            .roots
            .iter()
            .filter(|(root, _)| fresh_storage_root_matches(root, &target.root))
            .flat_map(|(_, statuses)| statuses)
            .collect::<Vec<_>>();
        for (root, statuses) in &mut candidate.roots {
            if !fresh_storage_root_matches(root, &target.root) {
                continue;
            }
            for status in &mut *statuses {
                if matching_source.is_empty() && !source.complete
                    || matching_source
                        .iter()
                        .any(|source| fresh_root_statuses_may_alias(status, source))
                {
                    status.claimed.insert(target.proj.clone());
                }
            }
            normalize_fresh_root_statuses(statuses);
        }
    }
}

fn normalize_fresh_root_statuses<'db>(statuses: &mut Vec<FreshRootStatus<'db>>) {
    let previous = mem::take(statuses);
    for status in previous {
        merge_fresh_root_statuses(statuses, status);
    }
}

fn merge_fresh_root_status<'db>(
    roots: &mut FreshRootStates<'db>,
    root: FreshStorageRoot,
    status: FreshRootStatus<'db>,
) {
    let statuses = roots.entry(root).or_default();
    merge_fresh_root_statuses(statuses, status);
}

fn merge_fresh_root_statuses<'db>(
    statuses: &mut Vec<FreshRootStatus<'db>>,
    status: FreshRootStatus<'db>,
) {
    let mut merged = status;
    let mut idx = 0;
    while idx < statuses.len() {
        if fresh_root_statuses_may_alias(&statuses[idx], &merged) {
            let candidate = statuses.swap_remove(idx);
            merged.retention_sites.extend(candidate.retention_sites);
            merged.carriers.extend(candidate.carriers);
            merged.claimed.extend(candidate.claimed);
        } else {
            idx += 1;
        }
    }
    statuses.push(merged);
}

fn fresh_root_statuses_may_alias(lhs: &FreshRootStatus<'_>, rhs: &FreshRootStatus<'_>) -> bool {
    lhs.generation == rhs.generation
        && (lhs.retention_sites.is_empty() && rhs.retention_sites.is_empty()
            || !lhs.retention_sites.is_disjoint(&rhs.retention_sites))
}

fn merge_fresh_value_status<'db>(into: &mut FreshValueStatus<'db>, from: FreshValueStatus<'db>) {
    into.complete &= from.complete;
    for (root, statuses) in from.roots {
        for status in statuses {
            merge_fresh_root_status(&mut into.roots, root, status);
        }
    }
}

fn fresh_layout_projection_matches(
    lhs: LayoutBackingProjection,
    rhs: LayoutBackingProjection,
) -> bool {
    lhs == rhs
        || matches!(
            (lhs, rhs),
            (
                LayoutBackingProjection::Index(None),
                LayoutBackingProjection::Index(_)
            ) | (
                LayoutBackingProjection::Index(_),
                LayoutBackingProjection::Index(None)
            )
        )
}

fn fresh_layout_path_is_prefix(
    prefix: &[LayoutBackingProjection],
    path: &[LayoutBackingProjection],
) -> bool {
    prefix.len() <= path.len()
        && prefix
            .iter()
            .copied()
            .zip(path.iter().copied())
            .all(|(lhs, rhs)| fresh_layout_projection_matches(lhs, rhs))
}

fn prefix_fresh_value_status<'db>(
    mut status: FreshValueStatus<'db>,
    prefix: &[LayoutBackingProjection],
) -> FreshValueStatus<'db> {
    for statuses in status.roots.values_mut() {
        for root_status in statuses {
            root_status.carriers = root_status
                .carriers
                .iter()
                .map(|carrier| {
                    let mut prefixed = prefix.to_vec();
                    prefixed.extend_from_slice(carrier);
                    prefixed
                })
                .collect();
        }
    }
    status
}

fn project_fresh_value_status<'db>(
    status: &FreshValueStatus<'db>,
    projection: &[LayoutBackingProjection],
) -> FreshValueStatus<'db> {
    let mut projected = FreshValueStatus {
        roots: FreshRootStates::default(),
        complete: status.complete,
    };
    for (&root, statuses) in &status.roots {
        for root_status in statuses {
            let mut root_status = root_status.clone();
            root_status.carriers = root_status
                .carriers
                .iter()
                .filter_map(|carrier| {
                    if fresh_layout_path_is_prefix(projection, carrier) {
                        Some(carrier[projection.len()..].to_vec())
                    } else if fresh_layout_path_is_prefix(carrier, projection) {
                        Some(FreshCarrierPath::new())
                    } else {
                        None
                    }
                })
                .collect();
            if !root_status.carriers.is_empty() {
                merge_fresh_root_status(&mut projected.roots, root, root_status);
            }
        }
    }
    projected
}

fn replace_fresh_value_projection<'db>(
    status: &mut FreshValueStatus<'db>,
    projection: &[LayoutBackingProjection],
    replacement: FreshValueStatus<'db>,
) {
    if !projection.contains(&LayoutBackingProjection::Index(None)) {
        status.roots.retain(|_, statuses| {
            statuses.retain_mut(|root_status| {
                root_status.carriers.retain(|carrier| {
                    !fresh_layout_path_is_prefix(projection, carrier)
                        || projection.iter().copied().zip(carrier.iter().copied()).any(
                            |(written, held)| {
                                matches!(
                                    (written, held),
                                    (
                                        LayoutBackingProjection::Index(Some(_)),
                                        LayoutBackingProjection::Index(None)
                                    )
                                )
                            },
                        )
                });
                !root_status.carriers.is_empty()
            });
            !statuses.is_empty()
        });
    }
    status.complete &= replacement.complete;
    merge_fresh_value_status(status, prefix_fresh_value_status(replacement, projection));
}

fn cfg_backedges(successors: &CfgAdjacency) -> FxHashSet<(SBlockId, SBlockId)> {
    fn visit(
        block: SBlockId,
        successors: &CfgAdjacency,
        colors: &mut [u8],
        backedges: &mut FxHashSet<(SBlockId, SBlockId)>,
    ) {
        colors[block.index()] = 1;
        for &successor in &successors[block] {
            match colors[successor.index()] {
                0 => visit(successor, successors, colors, backedges),
                1 => {
                    backedges.insert((block, successor));
                }
                2 => {}
                _ => unreachable!(),
            }
        }
        colors[block.index()] = 2;
    }

    let block_count = successors.iter().count();
    let mut colors = vec![0; block_count];
    let mut backedges = FxHashSet::default();
    for idx in 0..block_count {
        if colors[idx] == 0 {
            visit(SBlockId::new(idx), successors, &mut colors, &mut backedges);
        }
    }
    backedges
}

fn cfg_cycle_blocks(successors: &CfgAdjacency) -> FxHashSet<SBlockId> {
    let mut cyclic = FxHashSet::default();
    for (start, _) in successors.iter() {
        let mut seen = FxHashSet::default();
        let mut pending = successors[start].iter().copied().collect::<VecDeque<_>>();
        while let Some(block) = pending.pop_front() {
            if block == start {
                cyclic.insert(start);
                break;
            }
            if seen.insert(block) {
                pending.extend(successors[block].iter().copied());
            }
        }
    }
    cyclic
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
    !return_borrow_results_in_ty(db, instance.normalized_result_ty(db)).is_empty()
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
    for result in return_borrow_results_in_ty(db, instance.normalized_result_ty(db)) {
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

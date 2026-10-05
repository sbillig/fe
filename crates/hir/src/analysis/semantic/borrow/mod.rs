//! Borrow checking of normalized semantic bodies.
//!
//! The checker is local to one body. Calls are checked against the callee's
//! signature: a returned capability borrows from a capability-carrying `self`
//! receiver, or else from every capability-carrying argument; a `mut`
//! referent that can hold capabilities may receive any of them. Raw memory is
//! a trust boundary: dereferencing raw pointers requires `unsafe` and is not
//! modeled. See `docs/architecture/borrow-checker.md`.
mod analysis;
mod carried;
mod check;
mod control;
mod place;

use std::{collections::VecDeque, fmt};

use common::diagnostics::CompleteDiagnostic;
use cranelift_entity::EntityRef;
use rustc_hash::FxHashSet;
use salsa::Update;

use self::analysis::Analysis;
pub use self::{
    carried::{Carried, carried_capabilities},
    control::{
        ExecutableBlock, ExecutableControlFlow, semantic_executable_control_flow,
        semantic_may_return,
    },
};
use crate::{
    analysis::{
        HirAnalysisDb,
        analysis_pass::ModuleAnalysisPass,
        diagnostics::{DiagnosticVoucher, SpannedHirAnalysisDb},
        semantic::{
            CallSiteProviderRefinement, SemOrigin, SemanticInstance,
            diagnostics::{
                BlockedSemanticBody, SemanticDiagnosticId, SemanticDiagnosticKind,
                SemanticNormalizationFailure,
            },
            get_or_build_semantic_instance, identity_semantic_instance_key,
            normalized::{
                NEffectArgValue, NExpr, NStatementKind, normalize_semantic_body,
                normalize_semantic_body_provisional,
            },
            provisional_provider_idx_for_requirement,
        },
        ty::{
            provider::ProviderAddressSpace,
            ty_check::{BodyOwner, EffectParamSite, EffectPassMode},
        },
    },
    hir_def::{ItemKind, TopLevelMod},
};

#[derive(Clone, Debug, PartialEq, Eq, Update)]
pub enum SemanticBorrowCheckResult<'db> {
    Ok,
    Blocked(BlockedSemanticBody<'db>),
    Err(SemanticDiagnosticId<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticAnalysisError<'db> {
    Blocked(BlockedSemanticBody<'db>),
    Diagnostic(CompleteDiagnostic),
}

impl SemanticAnalysisError<'_> {
    pub fn diagnostic(&self) -> Option<&CompleteDiagnostic> {
        match self {
            Self::Blocked(_) => None,
            Self::Diagnostic(diag) => Some(diag),
        }
    }
}

impl fmt::Display for SemanticAnalysisError<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blocked(blocked) => write!(
                formatter,
                "semantic body is blocked by upstream causes: {:?}",
                blocked.causes
            ),
            Self::Diagnostic(diag) => formatter.write_str(&diag.message),
        }
    }
}

#[salsa::tracked(return_ref)]
fn body_check<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowCheckResult<'db> {
    let artifacts = match normalize_semantic_body(db, instance) {
        Ok(artifacts) => artifacts,
        Err(SemanticNormalizationFailure::Blocked(blocked)) => {
            return SemanticBorrowCheckResult::Blocked(blocked);
        }
        Err(
            SemanticNormalizationFailure::Rejected(diag)
            | SemanticNormalizationFailure::InternalFailure(diag),
        ) => return SemanticBorrowCheckResult::Err(SemanticDiagnosticId::new(db, diag)),
    };
    let mut analysis = Analysis::new(db, instance, &artifacts.body, true);
    analysis.solve();
    match analysis.check() {
        Ok(()) => SemanticBorrowCheckResult::Ok,
        Err(diag) => SemanticBorrowCheckResult::Err(SemanticDiagnosticId::new(db, diag)),
    }
}

/// Whether a borrow derived from this method's `mut self` reaches an argument
/// that requires memory, so callers must supply a memory receiver. This is the
/// one body fact callers depend on: transport is a representation constraint,
/// not part of the borrow contract.
#[salsa::tracked(
    cycle_fn=requires_memory_receiver_cycle_recover,
    cycle_initial=requires_memory_receiver_cycle_initial
)]
pub(crate) fn requires_memory_receiver<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> bool {
    let Ok(artifacts) = normalize_semantic_body(db, instance) else {
        return false;
    };
    let mut analysis = Analysis::new(db, instance, &artifacts.body, true);
    analysis.solve();
    analysis.receiver_reaches_memory_argument()
}

fn requires_memory_receiver_cycle_initial<'db>(
    _: &'db dyn HirAnalysisDb,
    _: SemanticInstance<'db>,
) -> bool {
    false
}

fn requires_memory_receiver_cycle_recover<'db>(
    _: &'db dyn HirAnalysisDb,
    _: &bool,
    _: u32,
    _: SemanticInstance<'db>,
) -> salsa::CycleRecoveryAction<bool> {
    salsa::CycleRecoveryAction::Iterate
}

pub fn check_semantic_borrows<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<(), SemanticAnalysisError<'db>> {
    match body_check(db, instance) {
        SemanticBorrowCheckResult::Ok => Ok(()),
        SemanticBorrowCheckResult::Blocked(body) => {
            Err(SemanticAnalysisError::Blocked(body.clone()))
        }
        SemanticBorrowCheckResult::Err(diag) => {
            Err(SemanticAnalysisError::Diagnostic(diag.to_complete(db)))
        }
    }
}

/// The provider address spaces a provisional body passes to its callees'
/// effect parameters.
#[derive(Clone, Debug, PartialEq, Eq, Update)]
pub(crate) enum CallSiteRefinements<'db> {
    Refined(Vec<CallSiteProviderRefinement>),
    /// Upstream causes block the provisional body.
    Blocked,
    Rejected(SemanticDiagnosticId<'db>),
}

#[salsa::tracked(return_ref)]
pub(crate) fn provisional_call_site_provider_refinements<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> CallSiteRefinements<'db> {
    let body = match normalize_semantic_body_provisional(db, instance) {
        Ok(artifacts) => artifacts.body,
        Err(SemanticNormalizationFailure::Blocked(_)) => return CallSiteRefinements::Blocked,
        Err(
            SemanticNormalizationFailure::Rejected(diag)
            | SemanticNormalizationFailure::InternalFailure(diag),
        ) => return CallSiteRefinements::Rejected(SemanticDiagnosticId::new(db, diag)),
    };
    let mut analysis = Analysis::new(db, instance, &body, false);
    analysis.solve();
    let mut refinements = Vec::new();
    for block in &body.blocks {
        for statement in &block.statements {
            let NStatementKind::Define {
                expr:
                    NExpr::Call {
                        call_site,
                        callee,
                        effect_args,
                        ..
                    },
                ..
            } = &statement.kind
            else {
                continue;
            };
            for arg in effect_args {
                if matches!(arg.pass_mode, EffectPassMode::Unknown) {
                    continue;
                }
                let regions = match &arg.arg {
                    NEffectArgValue::Place(place) => analysis.resolve(place).regions,
                    NEffectArgValue::Value(value) => analysis.values[value.value.index()]
                        .iter()
                        .flat_map(|token| analysis.tokens[*token as usize].regions.clone())
                        .collect(),
                };
                let mut spaces = Vec::new();
                let mut symbolic = regions.is_empty();
                for region in &regions {
                    match analysis.space(region.base) {
                        Some(space) if !spaces.contains(&space) => spaces.push(space),
                        Some(_) => {}
                        None => symbolic = true,
                    }
                }
                let address_space = match spaces.as_slice() {
                    [] => arg.provider,
                    [space] if !symbolic => Some(*space),
                    [_] => None,
                    _ => {
                        spaces.sort_by_key(|space| address_space_rank(*space));
                        let origin = match arg.arg {
                            NEffectArgValue::Value(value) => {
                                value.origin.map_or(statement.origin, SemOrigin::Expr)
                            }
                            NEffectArgValue::Place(_) => statement.origin,
                        };
                        let diag = analysis.diag(
                            SemanticDiagnosticKind::ProviderProvenanceConflict,
                            origin,
                            format!(
                                "effect argument may come from multiple address spaces: {}",
                                spaces
                                    .iter()
                                    .map(|space| space.pretty())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        );
                        return CallSiteRefinements::Rejected(SemanticDiagnosticId::new(db, diag));
                    }
                };
                let Some(address_space) = address_space else {
                    continue;
                };
                let provider_idx = match callee.key.owner(db) {
                    BodyOwner::Func(func) => provisional_provider_idx_for_requirement(
                        db,
                        EffectParamSite::Func(func),
                        arg.binding_idx,
                    ),
                    BodyOwner::Const(_)
                    | BodyOwner::AnonConstBody { .. }
                    | BodyOwner::ContractInit { .. }
                    | BodyOwner::ContractRecvArm { .. } => None,
                };
                refinements.push(CallSiteProviderRefinement {
                    call_site: *call_site,
                    binding_idx: arg.binding_idx,
                    provider_idx,
                    address_space,
                });
            }
        }
    }
    CallSiteRefinements::Refined(refinements)
}

fn address_space_rank(space: ProviderAddressSpace) -> u8 {
    match space {
        ProviderAddressSpace::Memory => 0,
        ProviderAddressSpace::Storage => 1,
        ProviderAddressSpace::Transient => 2,
        ProviderAddressSpace::Calldata => 3,
        ProviderAddressSpace::Code => 4,
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
    let mut owners = Vec::new();
    for item in top_mod
        .all_items(db)
        .iter()
        .filter(|item| item.top_mod(db) == top_mod)
    {
        owners.extend(BodyOwner::const_predicates_of(db, *item));
        match item {
            ItemKind::Func(func) => owners.push(BodyOwner::Func(*func)),
            ItemKind::Const(const_) => owners.push(BodyOwner::Const(*const_)),
            ItemKind::Contract(contract) => {
                owners.push(BodyOwner::ContractInit {
                    contract: *contract,
                });
                for (recv_idx, recv) in contract.recvs(db).data(db).iter().enumerate() {
                    for arm_idx in 0..recv.arms.data(db).len() {
                        owners.push(BodyOwner::ContractRecvArm {
                            contract: *contract,
                            recv_idx: recv_idx as u32,
                            arm_idx: arm_idx as u32,
                        });
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
    // Generic bodies are checked with their type parameters, which carry no
    // borrows; the specializations they call are checked as concrete bodies.
    let mut pending: VecDeque<_> = owners
        .into_iter()
        .map(|owner| get_or_build_semantic_instance(db, identity_semantic_instance_key(db, owner)))
        .collect();
    let mut seen = FxHashSet::default();
    let mut seen_diags = FxHashSet::default();
    let mut diags: Vec<Box<dyn DiagnosticVoucher + 'db>> = Vec::new();
    while let Some(instance) = pending.pop_front() {
        if !seen.insert(instance) {
            continue;
        }
        match body_check(db, instance) {
            SemanticBorrowCheckResult::Ok => pending.extend(
                instance
                    .callees(db)
                    .iter()
                    .map(|callee| get_or_build_semantic_instance(db, callee.key)),
            ),
            SemanticBorrowCheckResult::Blocked(_) => {}
            SemanticBorrowCheckResult::Err(diag) => {
                if seen_diags.insert(*diag) {
                    diags.push(Box::new(*diag));
                }
            }
        }
    }
    diags
}

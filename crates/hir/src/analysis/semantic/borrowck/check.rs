use std::collections::VecDeque;

use common::diagnostics::{
    CompleteDiagnostic, DiagnosticPass, GlobalErrorCode, LabelStyle, Severity, Span, SubDiagnostic,
};
use cranelift_entity::EntityRef;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::{
        HirAnalysisDb,
        diagnostics::SpannedHirAnalysisDb,
        semantic::{
            NSLocal, SemOrigin, SemanticInstance, get_or_build_semantic_instance,
            identity_semantic_instance_key,
        },
        ty::{
            ty_check::{BodyOwner, ParamSite},
            ty_def::BorrowKind,
        },
    },
    hir_def::{Body, FuncParamMode, ItemKind, TopLevelMod},
    projection::{Aliasing, IndexSource, Projection},
    span::LazySpan,
};

use super::{
    ir::{
        BorrowDiagnosticId, BorrowInputRef, BorrowSummary, BorrowSummaryId, BorrowTransform,
        NBorrowRoot, NBorrowRootId, NEffectArgValue, NExpr, NLocalInterface, NOperand, NSPlace,
        NSPlaceRoot, NSProjectionPath, NSStmtKind, NSTerminatorKind, NormalizedBindingLowering,
        NormalizedSemanticBody, ReadMode, SemanticBorrowCheckResult, SemanticBorrowSummaryResult,
        local_has_runtime_move_semantics,
    },
    normalize::normalize_semantic_body,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LoanId(u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BorrowRoot<'db> {
    Param(u32),
    Local(crate::analysis::semantic::SLocalId),
    Provider(crate::semantic::ProviderBinding<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CanonPlace<'db> {
    root: BorrowRoot<'db>,
    proj: NSProjectionPath<'db>,
}

#[derive(Clone, Debug)]
struct Loan<'db> {
    kind: BorrowKind,
    targets: FxHashSet<CanonPlace<'db>>,
    parents: FxHashSet<LoanId>,
    origin: crate::analysis::semantic::SemOrigin<'db>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State {
    local_loans: FxHashMap<crate::analysis::semantic::SLocalId, FxHashSet<LoanId>>,
}

impl State {
    fn loans_in(&self, local: crate::analysis::semantic::SLocalId) -> FxHashSet<LoanId> {
        self.local_loans.get(&local).cloned().unwrap_or_default()
    }

    fn assign_loans(
        &mut self,
        local: crate::analysis::semantic::SLocalId,
        loans: FxHashSet<LoanId>,
    ) {
        if loans.is_empty() {
            self.local_loans.remove(&local);
        } else {
            self.local_loans.insert(local, loans);
        }
    }

    fn join_from(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (local, loans) in &other.local_loans {
            let entry = self.local_loans.entry(*local).or_default();
            let before = entry.len();
            entry.extend(loans.iter().copied());
            changed |= before != entry.len();
        }
        changed
    }
}

#[salsa::tracked(
    cycle_fn=semantic_borrow_summary_cycle_recover,
    cycle_initial=semantic_borrow_summary_cycle_initial
)]
fn semantic_borrow_summary_query<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowSummaryResult<'db> {
    match Borrowck::new(db, instance).and_then(Borrowck::borrow_summary) {
        Ok(summary) => SemanticBorrowSummaryResult::Ok(
            summary.map(|summary| BorrowSummaryId::new(db, summary)),
        ),
        Err(diag) => SemanticBorrowSummaryResult::Err(BorrowDiagnosticId::new(db, diag)),
    }
}

pub fn semantic_borrow_summary<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<Option<BorrowSummary<'db>>, CompleteDiagnostic> {
    match semantic_borrow_summary_query(db, instance) {
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
        SemanticBorrowCheckResult::Err(diag) => Err(diag.diag(db).clone()),
    }
}

#[salsa::tracked]
fn semantic_borrow_check_query<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowCheckResult<'db> {
    match Borrowck::new(db, instance).and_then(Borrowck::check) {
        Ok(()) => SemanticBorrowCheckResult::Ok,
        Err(diag) => SemanticBorrowCheckResult::Err(BorrowDiagnosticId::new(db, diag)),
    }
}

pub fn collect_semantic_borrow_diagnostics<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Vec<CompleteDiagnostic> {
    let mut diags = Vec::new();
    for item in top_mod.all_items(db) {
        match item {
            ItemKind::Func(func) => collect_owner(db, BodyOwner::Func(*func), &mut diags),
            ItemKind::Const(const_) => collect_owner(db, BodyOwner::Const(*const_), &mut diags),
            ItemKind::Contract(contract) => {
                collect_owner(
                    db,
                    BodyOwner::ContractInit {
                        contract: *contract,
                    },
                    &mut diags,
                );
                for (recv_idx, recv) in contract.recvs(db).data(db).iter().enumerate() {
                    for arm_idx in 0..recv.arms.data(db).len() {
                        collect_owner(
                            db,
                            BodyOwner::ContractRecvArm {
                                contract: *contract,
                                recv_idx: recv_idx as u32,
                                arm_idx: arm_idx as u32,
                            },
                            &mut diags,
                        );
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
            | ItemKind::Use(_)
            | ItemKind::TopMod(_)
            | ItemKind::Body(_) => {}
        }
    }
    diags
}

fn collect_owner<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    owner: BodyOwner<'db>,
    diags: &mut Vec<CompleteDiagnostic>,
) {
    let key = identity_semantic_instance_key(db, owner);
    let instance = get_or_build_semantic_instance(db, key);
    if let Err(diag) = check_semantic_borrows(db, instance) {
        diags.push(diag);
    }
}

pub fn verify_normalized_semantic_body<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
) -> Result<(), CompleteDiagnostic> {
    for (local_idx, local) in body.locals.iter().enumerate() {
        let local_id = crate::analysis::semantic::SLocalId::from_u32(local_idx as u32);
        let verify_rooted_place = |place: &NSPlace<'db>, label: &str| {
            if let Some(root) = place.root.borrow_root() {
                if body.root(root).is_none() {
                    return Err(normalized_body_internal_diag(
                        db,
                        instance,
                        body,
                        SemOrigin::Body(body.template_owner),
                        format!("{label} {} has missing borrow root", local_id.index()),
                    ));
                }
            } else if !matches!(place.root, super::ir::NSPlaceRoot::CarrierDerefLocal(_)) {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    SemOrigin::Body(body.template_owner),
                    format!("{label} {} has missing borrow root", local_id.index()),
                ));
            }
            Ok(())
        };
        match (&local.facts.interface, &local.lowering) {
            (NLocalInterface::Erased, NormalizedBindingLowering::Erased)
            | (NLocalInterface::DirectValue, NormalizedBindingLowering::ValueLocal { .. })
            | (
                NLocalInterface::PlaceBoundValue,
                NormalizedBindingLowering::PlaceBoundValue { .. },
            )
            | (
                NLocalInterface::PlaceCarrier | NLocalInterface::DirectCarrier,
                NormalizedBindingLowering::CarrierLocal { .. },
            ) => {}
            _ => {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    SemOrigin::Body(body.template_owner),
                    format!(
                        "normalized local {} has mismatched interface/lowering: {:?} vs {:?}",
                        local_id.index(),
                        local.facts.interface,
                        &local.lowering,
                    ),
                ));
            }
        }
        match &local.lowering {
            NormalizedBindingLowering::ValueLocal { place } => {
                verify_rooted_place(place, "value local")?;
            }
            NormalizedBindingLowering::PlaceBoundValue { place, .. } => {
                verify_rooted_place(place, "place-bound local")?;
            }
            NormalizedBindingLowering::CarrierLocal { root, .. } => {
                if let Some(root) = root
                    && body.root(*root).is_none()
                {
                    return Err(normalized_body_internal_diag(
                        db,
                        instance,
                        body,
                        SemOrigin::Body(body.template_owner),
                        format!("carrier local {} has missing borrow root", local_id.index()),
                    ));
                }
            }
            NormalizedBindingLowering::Erased => {}
        }
        if let Some(place) = local.snapshot_source_place() {
            verify_rooted_place(place, "snapshot source place for local")?;
        }
    }

    for block in &body.blocks {
        for stmt in &block.stmts {
            match &stmt.kind {
                NSStmtKind::Assign { dst, expr } => {
                    verify_local_exists(db, instance, body, stmt.origin, *dst)?;
                    verify_expr(db, instance, body, stmt.origin, expr)?;
                }
                NSStmtKind::Store { dst, src } => {
                    verify_place(db, instance, body, stmt.origin, dst)?;
                    verify_local_exists(db, instance, body, stmt.origin, *src)?;
                }
            }
        }
        verify_terminator(db, instance, body, &block.terminator)?;
    }
    Ok(())
}

fn verify_terminator<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
    term: &super::ir::NSTerminator<'db>,
) -> Result<(), CompleteDiagnostic> {
    match &term.kind {
        NSTerminatorKind::Goto(bb) => {
            if body.block(*bb).is_none() {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    term.origin,
                    format!("missing normalized block {}", bb.index()),
                ));
            }
        }
        NSTerminatorKind::Branch {
            cond,
            then_bb,
            else_bb,
        } => {
            verify_operand(db, instance, body, term.origin, *cond)?;
            if body.block(*then_bb).is_none() || body.block(*else_bb).is_none() {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    term.origin,
                    "branch target is missing".to_string(),
                ));
            }
        }
        NSTerminatorKind::MatchEnum {
            value,
            cases,
            default,
            ..
        } => {
            verify_operand(db, instance, body, term.origin, *value)?;
            if cases.iter().any(|(_, bb)| body.block(*bb).is_none())
                || default.is_some_and(|bb| body.block(bb).is_none())
            {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    term.origin,
                    "match target is missing".to_string(),
                ));
            }
        }
        NSTerminatorKind::Return(Some(value)) => {
            verify_operand(db, instance, body, term.origin, *value)?;
        }
        NSTerminatorKind::Return(None) => {}
    }
    Ok(())
}

fn verify_expr<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
    origin: SemOrigin<'db>,
    expr: &NExpr<'db>,
) -> Result<(), CompleteDiagnostic> {
    match expr {
        NExpr::Use(value)
        | NExpr::Unary { value, .. }
        | NExpr::Cast { value, .. }
        | NExpr::GetEnumTag { value }
        | NExpr::IsEnumVariant { value, .. }
        | NExpr::ExtractEnumField { value, .. } => {
            verify_operand(db, instance, body, origin, *value)
        }
        NExpr::Binary { lhs, rhs, .. } => {
            verify_operand(db, instance, body, origin, *lhs)?;
            verify_operand(db, instance, body, origin, *rhs)
        }
        NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
            for field in fields {
                verify_operand(db, instance, body, origin, *field)?;
            }
            Ok(())
        }
        NExpr::ReadPlace { place, mode } => {
            verify_place(db, instance, body, origin, place)?;
            if *mode == ReadMode::Move && !place_move_is_valid(body, place) {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    origin,
                    "move read is invalid for this normalized place".to_string(),
                ));
            }
            Ok(())
        }
        NExpr::Borrow { place, .. } => verify_place(db, instance, body, origin, place),
        NExpr::Call {
            args, effect_args, ..
        } => {
            for arg in args {
                verify_operand(db, instance, body, origin, *arg)?;
            }
            for effect_arg in effect_args {
                match &effect_arg.arg {
                    NEffectArgValue::Place(place) => {
                        verify_place(db, instance, body, origin, place)?
                    }
                    NEffectArgValue::Value(value) => {
                        verify_operand(db, instance, body, origin, *value)?
                    }
                }
            }
            Ok(())
        }
        NExpr::Const(_)
        | NExpr::CodeRegionRef { .. }
        | NExpr::CodeRegionOffset { .. }
        | NExpr::CodeRegionLen { .. } => Ok(()),
    }
}

fn verify_operand<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
    origin: SemOrigin<'db>,
    operand: NOperand,
) -> Result<(), CompleteDiagnostic> {
    let local = verify_local_exists(db, instance, body, origin, operand.local)?;
    if operand.mode == ReadMode::Move
        && !local_has_runtime_move_semantics(db, local, &body.borrow_roots)
    {
        return Err(normalized_body_internal_diag(
            db,
            instance,
            body,
            origin,
            format!(
                "move read is invalid for normalized local {}",
                operand.local.index()
            ),
        ));
    }
    Ok(())
}

fn verify_local_exists<'db, 'a>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &'a NormalizedSemanticBody<'db>,
    origin: SemOrigin<'db>,
    local: crate::analysis::semantic::SLocalId,
) -> Result<&'a NSLocal<'db>, CompleteDiagnostic> {
    body.local(local).ok_or_else(|| {
        normalized_body_internal_diag(
            db,
            instance,
            body,
            origin,
            format!("missing normalized local {}", local.index()),
        )
    })
}

fn verify_place<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
    origin: SemOrigin<'db>,
    place: &NSPlace<'db>,
) -> Result<(), CompleteDiagnostic> {
    match place.root {
        NSPlaceRoot::Root(root) => {
            if body.root(root).is_none() {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    origin,
                    format!("missing normalized borrow root {}", root.index()),
                ));
            }
        }
        NSPlaceRoot::CarrierDerefLocal(local) => {
            let local = verify_local_exists(db, instance, body, origin, local)?;
            if !matches!(
                local.lowering,
                NormalizedBindingLowering::CarrierLocal { .. }
            ) {
                return Err(normalized_body_internal_diag(
                    db,
                    instance,
                    body,
                    origin,
                    "carrier-deref place root does not reference a carrier local".to_string(),
                ));
            }
        }
    }
    for proj in place.path.iter() {
        if let Projection::Index(IndexSource::Dynamic(index)) = proj {
            verify_local_exists(db, instance, body, origin, *index)?;
        }
    }
    Ok(())
}

fn place_move_is_valid<'db>(body: &NormalizedSemanticBody<'db>, place: &NSPlace<'db>) -> bool {
    match place.root {
        NSPlaceRoot::Root(root) => match body.root(root) {
            Some(NBorrowRoot::Param { local, .. }) | Some(NBorrowRoot::LocalSlot { local }) => {
                body.local(*local).is_some_and(|local| {
                    matches!(
                        local.lowering,
                        NormalizedBindingLowering::ValueLocal { .. }
                            | NormalizedBindingLowering::PlaceBoundValue { .. }
                    )
                })
            }
            Some(NBorrowRoot::Provider { .. }) => false,
            None => false,
        },
        NSPlaceRoot::CarrierDerefLocal(_) => false,
    }
}

fn normalized_body_internal_diag<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
    origin: SemOrigin<'db>,
    message: String,
) -> CompleteDiagnostic {
    CompleteDiagnostic::new(
        Severity::Error,
        format!(
            "internal borrow checking error in `fn {}`",
            checker_name(db, instance)
        ),
        vec![SubDiagnostic::new(
            LabelStyle::Primary,
            message,
            span_for_origin_from_body(db, instance.key(db).owner(db).body(db), origin).or_else(
                || {
                    body.template_owner
                        .body(db)
                        .and_then(|hir_body| hir_body.span().resolve(db))
                },
            ),
        )],
        Vec::new(),
        GlobalErrorCode::new(DiagnosticPass::SemanticBorrowck, 4),
    )
}

fn normalize_error_to_diag<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    err: super::ir::SemanticNormalizeError<'db>,
) -> CompleteDiagnostic {
    let owner = instance.key(db).owner(db);
    let hir_body = owner.body(db);
    let (origin, message, span) = match err {
        super::ir::SemanticNormalizeError::MissingBorrowRoot { local } => {
            let message = if let Some(body) = hir_body
                && let Some(raw_local) = instance.body(db).local(local)
                && let Some(source) = raw_local.source
            {
                format!(
                    "cannot normalize borrow roots for `{}`",
                    source.pretty_name_in_body(db, body)
                )
            } else {
                format!("cannot normalize borrow roots for `%{}`", local.index())
            };
            let span = hir_body
                .and_then(|body| {
                    instance
                        .body(db)
                        .local(local)
                        .and_then(|local| local.source)
                        .and_then(|source| source.def_span_in_body(body).resolve(db))
                })
                .or_else(|| hir_body.and_then(|body| body.span().resolve(db)));
            (SemOrigin::Body(owner), message, span)
        }
        super::ir::SemanticNormalizeError::IllegalCarrierPlace { local, origin } => {
            let message = if let Some(body) = hir_body
                && let Some(raw_local) = instance.body(db).local(local)
                && let Some(source) = raw_local.source
            {
                format!(
                    "cannot normalize carrier-style place access for `{}`",
                    source.pretty_name_in_body(db, body)
                )
            } else {
                format!(
                    "cannot normalize carrier-style place access for `%{}`",
                    local.index()
                )
            };
            (
                origin,
                message,
                span_for_origin_from_body(db, hir_body, origin),
            )
        }
        super::ir::SemanticNormalizeError::LocalProvenanceCycle { local, .. } => (
            SemOrigin::Body(owner),
            format!(
                "detected a cycle while normalizing derived-place provenance for `%{}`",
                local.index()
            ),
            hir_body.and_then(|body| body.span().resolve(db)),
        ),
        super::ir::SemanticNormalizeError::NonPlaceDerivedValue { local, base, .. } => (
            SemOrigin::Body(owner),
            format!(
                "cannot normalize derived-place provenance for `%{}` from non-place base `%{}`",
                local.index(),
                base.index()
            ),
            hir_body.and_then(|body| body.span().resolve(db)),
        ),
    };
    let _ = origin;
    CompleteDiagnostic::new(
        Severity::Error,
        format!(
            "internal borrow checking error in `fn {}`",
            checker_name(db, instance)
        ),
        vec![SubDiagnostic::new(LabelStyle::Primary, message, span)],
        Vec::new(),
        GlobalErrorCode::new(DiagnosticPass::SemanticBorrowck, 4),
    )
}

fn span_for_origin_from_body<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    body: Option<Body<'db>>,
    origin: SemOrigin<'db>,
) -> Option<Span> {
    let body = body?;
    match origin {
        SemOrigin::Expr(expr) => expr.span(body).resolve(db),
        SemOrigin::Stmt(stmt) => stmt.span(body).resolve(db),
        SemOrigin::Body(owner) => owner.body(db).and_then(|body| body.span().resolve(db)),
        SemOrigin::Synthetic => None,
    }
}

struct Borrowck<'db> {
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: NormalizedSemanticBody<'db>,
    hir_body: Option<Body<'db>>,
    param_modes: Vec<FuncParamMode>,
    param_index_of_local: FxHashMap<crate::analysis::semantic::SLocalId, u32>,
    effect_input_of_root: FxHashMap<NBorrowRootId, u32>,
    loan_for_local: FxHashMap<crate::analysis::semantic::SLocalId, LoanId>,
    param_loan_for_local: FxHashMap<crate::analysis::semantic::SLocalId, LoanId>,
    loans: Vec<Loan<'db>>,
    entry_state: Vec<State>,
    moved_entry: Vec<FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>>,
    live_before: Vec<Vec<FxHashSet<crate::analysis::semantic::SLocalId>>>,
    live_before_term: Vec<FxHashSet<crate::analysis::semantic::SLocalId>>,
}

impl<'db> Borrowck<'db> {
    fn new(
        db: &'db dyn SpannedHirAnalysisDb,
        instance: SemanticInstance<'db>,
    ) -> Result<Self, CompleteDiagnostic> {
        let body = normalize_semantic_body(db, instance)
            .map_err(|err| normalize_error_to_diag(db, instance, err))?;
        verify_normalized_semantic_body(db, instance, &body)?;
        let owner = instance.key(db).owner(db);
        let param_modes = match owner {
            BodyOwner::Func(func) => func.params(db).map(|param| param.mode(db)).collect(),
            _ => Vec::new(),
        };
        let mut param_index_of_local = FxHashMap::default();
        let mut effect_input_of_root = FxHashMap::default();
        for root_id in 0..body.borrow_roots.len() {
            let root_id = NBorrowRootId::from_u32(root_id as u32);
            match body.root(root_id).expect("borrow root") {
                NBorrowRoot::Param { local, param_idx } => {
                    param_index_of_local.insert(*local, *param_idx);
                }
                NBorrowRoot::Provider { .. } | NBorrowRoot::LocalSlot { .. } => {}
            }
        }
        for local_id in 0..body.locals.len() {
            let local_id = crate::analysis::semantic::SLocalId::from_u32(local_id as u32);
            let Some(local) = body.local(local_id) else {
                continue;
            };
            if let Some(root) = local.lowering.root()
                && let Some(idx) = match local.source {
                    Some(crate::analysis::ty::ty_check::LocalBinding::EffectParam {
                        idx, ..
                    })
                    | Some(crate::analysis::ty::ty_check::LocalBinding::Param {
                        site: ParamSite::EffectField(_),
                        idx,
                        ..
                    }) => Some(idx as u32),
                    Some(crate::analysis::ty::ty_check::LocalBinding::Param { .. })
                    | Some(crate::analysis::ty::ty_check::LocalBinding::Local { .. })
                    | None => None,
                }
            {
                effect_input_of_root.insert(root, idx);
            }
        }
        let mut checker = Self {
            db,
            instance,
            hir_body: owner.body(db),
            body,
            param_modes,
            param_index_of_local,
            effect_input_of_root,
            loan_for_local: FxHashMap::default(),
            param_loan_for_local: FxHashMap::default(),
            loans: Vec::new(),
            entry_state: Vec::new(),
            moved_entry: Vec::new(),
            live_before: Vec::new(),
            live_before_term: Vec::new(),
        };
        checker.init_loans();
        Ok(checker)
    }

    fn borrow_summary(mut self) -> Result<Option<BorrowSummary<'db>>, CompleteDiagnostic> {
        let owner = self.instance.key(self.db).owner(self.db);
        let typed_body = self.instance.key(self.db).instantiate_typed_body(self.db);
        if typed_body.result_ty().as_borrow(self.db).is_none() || owner.body(self.db).is_none() {
            return Ok(None);
        }
        self.compute_entry_states();
        self.compute_loan_targets()?;
        self.compute_return_summary().map(Some)
    }

    fn check(mut self) -> Result<(), CompleteDiagnostic> {
        self.compute_entry_states();
        self.compute_loan_targets()?;
        self.compute_moved_states()?;
        self.compute_liveness();
        self.check_conflicts()?;
        if self
            .instance
            .key(self.db)
            .instantiate_typed_body(self.db)
            .result_ty()
            .as_borrow(self.db)
            .is_some()
        {
            let _ = self.compute_return_summary()?;
        }
        Ok(())
    }

    fn compute_liveness(&mut self) {
        self.live_before = self
            .body
            .blocks
            .iter()
            .map(|block| vec![FxHashSet::default(); block.stmts.len()])
            .collect();
        self.live_before_term = vec![FxHashSet::default(); self.body.blocks.len()];
        loop {
            let mut changed = false;
            for (bb_idx, block) in self.body.blocks.iter().enumerate().rev() {
                let mut live = self.terminator_uses(&block.terminator);
                for succ in self.successors(&block.terminator.kind) {
                    live.extend(self.block_live_in(succ));
                }
                if self.live_before_term[bb_idx] != live {
                    self.live_before_term[bb_idx] = live.clone();
                    changed = true;
                }
                for (stmt_idx, stmt) in block.stmts.iter().enumerate().rev() {
                    live = self.live_before_stmt(stmt, &live);
                    if self.live_before[bb_idx][stmt_idx] != live {
                        self.live_before[bb_idx][stmt_idx] = live.clone();
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn block_live_in(
        &self,
        bb: crate::analysis::semantic::SBlockId,
    ) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        self.body
            .block(bb)
            .and_then(|block| {
                block
                    .stmts
                    .is_empty()
                    .then(|| self.live_before_term[bb.index()].clone())
            })
            .or_else(|| {
                self.live_before
                    .get(bb.index())
                    .and_then(|live| live.first().cloned())
            })
            .unwrap_or_default()
    }

    fn live_before_stmt(
        &self,
        stmt: &super::ir::NSStmt<'db>,
        live_after: &FxHashSet<crate::analysis::semantic::SLocalId>,
    ) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        let mut live = live_after.clone();
        let uses = self.stmt_uses(stmt);
        match &stmt.kind {
            NSStmtKind::Assign { dst, .. } => {
                live.remove(dst);
                live.extend(uses);
            }
            NSStmtKind::Store { .. } => live.extend(uses),
        }
        live
    }

    fn init_loans(&mut self) {
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
                let loan = LoanId(self.loans.len() as u32);
                let mut targets = FxHashSet::default();
                targets.insert(CanonPlace {
                    root: BorrowRoot::Param(param_idx),
                    proj: NSProjectionPath::default(),
                });
                self.loans.push(Loan {
                    kind,
                    targets,
                    parents: FxHashSet::default(),
                    origin: crate::analysis::semantic::SemOrigin::Body(self.body.template_owner),
                });
                self.param_loan_for_local.insert(local_id, loan);
            }
        }

        for block in &self.body.blocks {
            for stmt in &block.stmts {
                let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
                    continue;
                };
                if self
                    .body
                    .local(*dst)
                    .is_some_and(|local| local.ty.as_borrow(self.db).is_some())
                    && matches!(expr, NExpr::Borrow { .. } | NExpr::Call { .. })
                {
                    let kind = self
                        .body
                        .local(*dst)
                        .and_then(|local| local.ty.as_borrow(self.db))
                        .map(|(kind, _)| kind)
                        .expect("borrow local");
                    let loan = LoanId(self.loans.len() as u32);
                    self.loan_for_local.insert(*dst, loan);
                    self.loans.push(Loan {
                        kind,
                        targets: FxHashSet::default(),
                        parents: FxHashSet::default(),
                        origin: stmt.origin,
                    });
                }
            }
        }
    }

    fn compute_entry_states(&mut self) {
        self.entry_state = vec![State::default(); self.body.blocks.len()];
        if self.body.blocks.is_empty() {
            return;
        }
        if let Some(entry) = self.entry_state.first_mut() {
            for (&local, &loan) in &self.param_loan_for_local {
                let mut set = FxHashSet::default();
                set.insert(loan);
                entry.assign_loans(local, set);
            }
        }
        let mut worklist = VecDeque::from([crate::analysis::semantic::SBlockId::from_u32(0)]);
        let mut queued = FxHashSet::from_iter([crate::analysis::semantic::SBlockId::from_u32(0)]);
        while let Some(bb) = worklist.pop_front() {
            queued.remove(&bb);
            let mut state = self.entry_state[bb.index()].clone();
            let block = &self.body.blocks[bb.index()];
            for stmt in &block.stmts {
                self.apply_stmt_state(&mut state, stmt);
            }
            for succ in self.successors(&block.terminator.kind) {
                let changed = self.entry_state[succ.index()].join_from(&state);
                if changed && queued.insert(succ) {
                    worklist.push_back(succ);
                }
            }
        }
    }

    fn compute_loan_targets(&mut self) -> Result<(), CompleteDiagnostic> {
        loop {
            let mut changed = false;
            let blocks = self.body.blocks.clone();
            for (bb_idx, block) in blocks.iter().enumerate() {
                let mut state = self.entry_state[bb_idx].clone();
                for stmt in &block.stmts {
                    self.update_loan_from_stmt(&state, stmt, &mut changed)?;
                    self.apply_stmt_state(&mut state, stmt);
                }
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }

    fn compute_moved_states(&mut self) -> Result<(), CompleteDiagnostic> {
        self.moved_entry = vec![FxHashMap::default(); self.body.blocks.len()];
        if self.body.blocks.is_empty() {
            return Ok(());
        }
        let mut worklist = VecDeque::from([crate::analysis::semantic::SBlockId::from_u32(0)]);
        let mut queued = FxHashSet::from_iter([crate::analysis::semantic::SBlockId::from_u32(0)]);
        while let Some(bb) = worklist.pop_front() {
            queued.remove(&bb);
            let mut state = self.entry_state[bb.index()].clone();
            let mut moved = self.moved_entry[bb.index()].clone();
            let block = &self.body.blocks[bb.index()];
            for stmt in &block.stmts {
                self.update_moved_for_stmt(&state, &mut moved, stmt)?;
                self.apply_stmt_state(&mut state, stmt);
            }
            for succ in self.successors(&block.terminator.kind) {
                let mut changed = false;
                let entry = &mut self.moved_entry[succ.index()];
                for (place, origin) in &moved {
                    changed |= entry.insert(place.clone(), *origin).is_none();
                }
                if changed && queued.insert(succ) {
                    worklist.push_back(succ);
                }
            }
        }
        Ok(())
    }

    fn check_conflicts(&self) -> Result<(), CompleteDiagnostic> {
        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            let mut state = self.entry_state[bb_idx].clone();
            let mut moved = self.moved_entry[bb_idx].clone();
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                self.check_stmt(&state, &moved, &self.live_before[bb_idx][stmt_idx], stmt)?;
                self.update_moved_for_stmt(&state, &mut moved, stmt)?;
                self.apply_stmt_state(&mut state, stmt);
            }
            self.check_terminator(
                &state,
                &moved,
                &self.live_before_term[bb_idx],
                &block.terminator,
            )?;
        }
        Ok(())
    }

    fn apply_stmt_state(&self, state: &mut State, stmt: &super::ir::NSStmt<'db>) {
        let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
            return;
        };
        let loans = match expr {
            NExpr::Use(src) => state.loans_in(src.local),
            NExpr::Borrow { .. } | NExpr::Call { .. } => self
                .loan_for_local
                .get(dst)
                .copied()
                .map(|loan| FxHashSet::from_iter([loan]))
                .unwrap_or_default(),
            _ => FxHashSet::default(),
        };
        state.assign_loans(*dst, loans);
    }

    fn update_loan_from_stmt(
        &mut self,
        state: &State,
        stmt: &super::ir::NSStmt<'db>,
        changed: &mut bool,
    ) -> Result<(), CompleteDiagnostic> {
        let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
            return Ok(());
        };
        let Some(&loan_id) = self.loan_for_local.get(dst) else {
            return Ok(());
        };
        match expr {
            NExpr::Borrow { place, .. } => {
                let targets = self.canonicalize_place(state, place, stmt.origin)?;
                let parents = self.mut_loans_for_place(state, place, stmt.origin)?;
                *changed |= self.extend_loan(loan_id, targets, parents);
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
                    return Ok(());
                };
                let mut targets = FxHashSet::default();
                let mut parents = FxHashSet::default();
                for transform in &summary {
                    match transform.input {
                        BorrowInputRef::Param(idx) => {
                            if let Some(arg) = args.get(idx as usize) {
                                for base in
                                    self.canonicalize_value_base(state, arg.local, stmt.origin)?
                                {
                                    targets.insert(CanonPlace {
                                        root: base.root,
                                        proj: base.proj.concat(&transform.proj),
                                    });
                                }
                                parents.extend(self.mut_loans_for_value(state, arg.local));
                            }
                        }
                        BorrowInputRef::EffectArg(idx) => {
                            let Some(effect_arg) = effect_args.get(idx as usize) else {
                                continue;
                            };
                            if matches!(
                                effect_arg.pass_mode,
                                crate::analysis::ty::ty_check::EffectPassMode::ByPlace
                            ) && let NEffectArgValue::Place(place) = &effect_arg.arg
                            {
                                for base in self.canonicalize_place(state, place, stmt.origin)? {
                                    targets.insert(CanonPlace {
                                        root: base.root,
                                        proj: base.proj.concat(&transform.proj),
                                    });
                                }
                                parents.extend(self.mut_loans_for_place(
                                    state,
                                    place,
                                    stmt.origin,
                                )?);
                            }
                        }
                    }
                }
                *changed |= self.extend_loan(loan_id, targets, parents);
            }
            _ => {}
        }
        Ok(())
    }

    fn extend_loan(
        &mut self,
        loan_id: LoanId,
        targets: FxHashSet<CanonPlace<'db>>,
        parents: FxHashSet<LoanId>,
    ) -> bool {
        let loan = &mut self.loans[loan_id.0 as usize];
        let before_targets = loan.targets.len();
        let before_parents = loan.parents.len();
        loan.targets.extend(targets);
        loan.parents.extend(parents);
        before_targets != loan.targets.len() || before_parents != loan.parents.len()
    }

    fn check_stmt(
        &self,
        state: &State,
        moved: &FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        live: &FxHashSet<crate::analysis::semantic::SLocalId>,
        stmt: &super::ir::NSStmt<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        let active = self.effective_loans(state, live);
        match &stmt.kind {
            NSStmtKind::Assign { expr, .. } => match expr {
                NExpr::ReadPlace { place, mode } => {
                    let targets = self.canonicalize_place(state, place, stmt.origin)?;
                    self.check_moved_overlap(
                        moved,
                        &targets,
                        stmt.origin,
                        "cannot use a value after it was moved",
                    )?;
                    if *mode == ReadMode::Move {
                        self.check_move_out(state, &active, place, &targets, stmt.origin)?;
                    }
                }
                NExpr::Borrow { place, kind, .. } => {
                    let targets = self.canonicalize_place(state, place, stmt.origin)?;
                    self.check_moved_overlap(
                        moved,
                        &targets,
                        stmt.origin,
                        "cannot borrow a moved value",
                    )?;
                    if let Some(conflict) = self.first_loan_conflict(&active, *kind, &targets) {
                        return Err(self.borrow_conflict_diag(
                            stmt.origin,
                            self.overlapping_loans_msg(conflict, *kind),
                            conflict,
                        ));
                    }
                }
                _ => self.check_expr_operands(state, moved, stmt.origin, expr)?,
            },
            NSStmtKind::Store { dst, .. } => {
                let targets = self.canonicalize_place(state, dst, stmt.origin)?;
                self.check_moved_parent(moved, &targets, stmt.origin)?;
            }
        }
        Ok(())
    }

    fn check_terminator(
        &self,
        state: &State,
        moved: &FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        live: &FxHashSet<crate::analysis::semantic::SLocalId>,
        term: &super::ir::NSTerminator<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        match &term.kind {
            NSTerminatorKind::Goto(_) => {}
            NSTerminatorKind::Branch { cond, .. }
            | NSTerminatorKind::MatchEnum { value: cond, .. }
            | NSTerminatorKind::Return(Some(cond)) => {
                let _ = live;
                self.check_operand(
                    state,
                    moved,
                    *cond,
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
            && state.loans_in(value.local).is_empty()
        {
            return Err(self.internal_diag(
                term.origin,
                "borrow return local has no tracked loan targets".to_string(),
            ));
        }
        Ok(())
    }

    fn compute_return_summary(&self) -> Result<BorrowSummary<'db>, CompleteDiagnostic> {
        let mut out = Vec::new();
        for (bb_idx, block) in self.body.blocks.iter().enumerate() {
            let NSTerminatorKind::Return(Some(value)) = block.terminator.kind else {
                continue;
            };
            let mut state = self.entry_state[bb_idx].clone();
            for stmt in &block.stmts {
                self.apply_stmt_state(&mut state, stmt);
            }
            for loan in state.loans_in(value.local) {
                for target in &self.loans[loan.0 as usize].targets {
                    for proj in target.proj.iter() {
                        if matches!(proj, Projection::Index(IndexSource::Dynamic(_))) {
                            return Err(self.invalid_return_diag(
                                block.terminator.origin,
                                "return borrows with dynamic indices are not supported".to_string(),
                            ));
                        }
                    }
                    match &target.root {
                        BorrowRoot::Param(idx) => {
                            let transform = BorrowTransform {
                                input: BorrowInputRef::Param(*idx),
                                proj: target.proj.clone(),
                            };
                            if !out.contains(&transform) {
                                out.push(transform);
                            }
                        }
                        BorrowRoot::Provider(binding) => {
                            let Some(idx) = self.effect_input_index_for_provider(binding.clone())
                            else {
                                return Err(self.invalid_return_diag(
                                    block.terminator.origin,
                                    "return borrows must be derived from explicit borrow inputs"
                                        .to_string(),
                                ));
                            };
                            let transform = BorrowTransform {
                                input: BorrowInputRef::EffectArg(idx),
                                proj: target.proj.clone(),
                            };
                            if !out.contains(&transform) {
                                out.push(transform);
                            }
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
        Ok(out)
    }

    fn effect_input_index_for_provider(
        &self,
        provider: crate::semantic::ProviderBinding<'db>,
    ) -> Option<u32> {
        self.effect_input_of_root
            .iter()
            .find_map(|(root, idx)| match self.body.root(*root) {
                Some(NBorrowRoot::Provider { binding }) if *binding == provider => Some(*idx),
                _ => None,
            })
    }

    fn canonicalize_value_base(
        &self,
        state: &State,
        local: crate::analysis::semantic::SLocalId,
        _origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<FxHashSet<CanonPlace<'db>>, CompleteDiagnostic> {
        if self
            .body
            .local(local)
            .is_some_and(|local| local.ty.as_borrow(self.db).is_some())
        {
            let mut out = FxHashSet::default();
            for loan in state.loans_in(local) {
                out.extend(self.loans[loan.0 as usize].targets.iter().cloned());
            }
            return Ok(out);
        }

        let Some(local_data) = self.body.local(local) else {
            return Ok(FxHashSet::default());
        };
        if let Some(place) = local_data.lowering.place() {
            return Ok(place
                .root
                .borrow_root()
                .and_then(|root| self.root_to_borrow_root(root))
                .into_iter()
                .map(|root| CanonPlace {
                    root,
                    proj: place.path.clone(),
                })
                .collect());
        }
        let root = match &local_data.lowering {
            NormalizedBindingLowering::CarrierLocal { root, provider, .. } => provider
                .clone()
                .map(BorrowRoot::Provider)
                .or_else(|| root.and_then(|root| self.root_to_borrow_root(root))),
            NormalizedBindingLowering::Erased => None,
            NormalizedBindingLowering::ValueLocal { .. }
            | NormalizedBindingLowering::PlaceBoundValue { .. } => unreachable!(),
        };
        Ok(root
            .into_iter()
            .map(|root| CanonPlace {
                root,
                proj: NSProjectionPath::default(),
            })
            .collect())
    }

    fn canonicalize_place(
        &self,
        state: &State,
        place: &NSPlace<'db>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<FxHashSet<CanonPlace<'db>>, CompleteDiagnostic> {
        match place.root {
            NSPlaceRoot::Root(root) => Ok(FxHashSet::from_iter([CanonPlace {
                root: self
                    .root_to_borrow_root(root)
                    .expect("normalized borrow root"),
                proj: place.path.clone(),
            }])),
            NSPlaceRoot::CarrierDerefLocal(local) => {
                let suffix = place.path.clone();
                let mut out = FxHashSet::default();
                let mut resolved = false;
                for loan in state.loans_in(local) {
                    resolved = true;
                    for target in &self.loans[loan.0 as usize].targets {
                        out.insert(CanonPlace {
                            root: target.root.clone(),
                            proj: target.proj.concat(&suffix),
                        });
                    }
                }
                if !resolved
                    && let Some(NormalizedBindingLowering::CarrierLocal { root, provider, .. }) =
                        self.body.local(local).map(|local| &local.lowering)
                {
                    if let Some(provider) = provider {
                        out.insert(CanonPlace {
                            root: BorrowRoot::Provider(provider.clone()),
                            proj: suffix.clone(),
                        });
                    } else if let Some(root) = root.and_then(|root| self.root_to_borrow_root(root))
                    {
                        out.insert(CanonPlace { root, proj: suffix });
                    }
                }
                if out.is_empty() {
                    return Err(self.internal_diag(
                        origin,
                        "cannot canonicalize carrier-rooted place".to_string(),
                    ));
                }
                Ok(out)
            }
        }
    }

    fn mut_loans_for_place(
        &self,
        state: &State,
        place: &NSPlace<'db>,
        _origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<FxHashSet<LoanId>, CompleteDiagnostic> {
        let loans = match place.root {
            NSPlaceRoot::CarrierDerefLocal(local) => state.loans_in(local),
            NSPlaceRoot::Root(_) => FxHashSet::default(),
        };
        Ok(loans
            .into_iter()
            .filter(|loan| self.loans[loan.0 as usize].kind == BorrowKind::Mut)
            .collect())
    }

    fn mut_loans_for_value(
        &self,
        state: &State,
        local: crate::analysis::semantic::SLocalId,
    ) -> FxHashSet<LoanId> {
        state
            .loans_in(local)
            .into_iter()
            .filter(|loan| self.loans[loan.0 as usize].kind == BorrowKind::Mut)
            .collect()
    }

    fn stmt_uses(
        &self,
        stmt: &super::ir::NSStmt<'db>,
    ) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        match &stmt.kind {
            NSStmtKind::Assign { expr, .. } => self.expr_uses(expr),
            NSStmtKind::Store { dst, src } => {
                let mut uses = self.place_uses(dst);
                uses.insert(*src);
                uses
            }
        }
    }

    fn terminator_uses(
        &self,
        term: &super::ir::NSTerminator<'db>,
    ) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        match &term.kind {
            NSTerminatorKind::Goto(_) | NSTerminatorKind::Return(None) => FxHashSet::default(),
            NSTerminatorKind::Branch { cond, .. }
            | NSTerminatorKind::MatchEnum { value: cond, .. }
            | NSTerminatorKind::Return(Some(cond)) => FxHashSet::from_iter([cond.local]),
        }
    }

    fn expr_uses(&self, expr: &NExpr<'db>) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        let mut uses = FxHashSet::default();
        match expr {
            NExpr::Use(value)
            | NExpr::Unary { value, .. }
            | NExpr::Cast { value, .. }
            | NExpr::GetEnumTag { value }
            | NExpr::IsEnumVariant { value, .. }
            | NExpr::ExtractEnumField { value, .. } => {
                uses.insert(value.local);
            }
            NExpr::Binary { lhs, rhs, .. } => {
                uses.insert(lhs.local);
                uses.insert(rhs.local);
            }
            NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
                uses.extend(fields.iter().map(|field| field.local));
            }
            NExpr::ReadPlace { place, .. } | NExpr::Borrow { place, .. } => {
                uses.extend(self.place_uses(place));
            }
            NExpr::Call {
                args, effect_args, ..
            } => {
                uses.extend(args.iter().map(|arg| arg.local));
                for effect_arg in effect_args {
                    match &effect_arg.arg {
                        NEffectArgValue::Place(place) => uses.extend(self.place_uses(place)),
                        NEffectArgValue::Value(value) => {
                            uses.insert(value.local);
                        }
                    }
                }
            }
            NExpr::Const(_)
            | NExpr::CodeRegionRef { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. } => {}
        }
        uses
    }

    fn place_uses(&self, place: &NSPlace<'db>) -> FxHashSet<crate::analysis::semantic::SLocalId> {
        let mut uses = FxHashSet::default();
        match place.root {
            NSPlaceRoot::Root(root) => match self.body.root(root) {
                Some(NBorrowRoot::Param { local, .. }) | Some(NBorrowRoot::LocalSlot { local }) => {
                    uses.insert(*local);
                }
                Some(NBorrowRoot::Provider { .. }) | None => {}
            },
            NSPlaceRoot::CarrierDerefLocal(local) => {
                uses.insert(local);
            }
        }
        uses.extend(place.path.iter().filter_map(|proj| match proj {
            Projection::Index(IndexSource::Dynamic(index)) => Some(*index),
            _ => None,
        }));
        uses
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
            .flat_map(|(_, loans)| loans.iter().copied())
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
        new_kind: BorrowKind,
        targets: &FxHashSet<CanonPlace<'db>>,
    ) -> Option<LoanId> {
        active.iter().copied().find(|loan| {
            let loan = &self.loans[loan.0 as usize];
            !matches!((loan.kind, new_kind), (BorrowKind::Ref, BorrowKind::Ref))
                && place_set_overlaps(&loan.targets, targets)
        })
    }

    fn check_move_out(
        &self,
        state: &State,
        active: &[LoanId],
        place: &NSPlace<'db>,
        targets: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        if matches!(place.root, NSPlaceRoot::CarrierDerefLocal(_)) {
            return Err(self.move_conflict_diag(
                origin,
                "cannot move out through a borrow handle".to_string(),
            ));
        }
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
        let _ = state;
        Ok(())
    }

    fn update_moved_for_stmt(
        &self,
        state: &State,
        moved: &mut FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        stmt: &super::ir::NSStmt<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => {
                if let Some(root) = self
                    .local_root(*dst)
                    .and_then(|root| self.root_to_borrow_root(root))
                {
                    moved.retain(|place, _| place.root != root);
                }
                if let NExpr::ReadPlace {
                    place,
                    mode: ReadMode::Move,
                } = expr
                {
                    for place in self.canonicalize_place(state, place, stmt.origin)? {
                        moved.insert(place, stmt.origin);
                    }
                }
                self.record_expr_moves(state, moved, stmt.origin, expr)?;
            }
            NSStmtKind::Store { dst, .. } => {
                let written = self.canonicalize_place(state, dst, stmt.origin)?;
                moved.retain(|place, _| {
                    !written.iter().any(|written| {
                        written.root == place.root && written.proj.is_prefix_of(&place.proj)
                    })
                });
            }
        }
        Ok(())
    }

    fn check_expr_operands(
        &self,
        state: &State,
        moved: &FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        expr: &NExpr<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        match expr {
            NExpr::Use(value)
            | NExpr::Unary { value, .. }
            | NExpr::Cast { value, .. }
            | NExpr::GetEnumTag { value }
            | NExpr::IsEnumVariant { value, .. }
            | NExpr::ExtractEnumField { value, .. } => self.check_operand(
                state,
                moved,
                *value,
                origin,
                "cannot use a value after it was moved",
            ),
            NExpr::Binary { lhs, rhs, .. } => {
                self.check_operand(
                    state,
                    moved,
                    *lhs,
                    origin,
                    "cannot use a value after it was moved",
                )?;
                self.check_operand(
                    state,
                    moved,
                    *rhs,
                    origin,
                    "cannot use a value after it was moved",
                )
            }
            NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
                for field in fields {
                    self.check_operand(
                        state,
                        moved,
                        *field,
                        origin,
                        "cannot use a value after it was moved",
                    )?;
                }
                Ok(())
            }
            NExpr::Call {
                args, effect_args, ..
            } => {
                for arg in args {
                    self.check_operand(
                        state,
                        moved,
                        *arg,
                        origin,
                        "cannot use a value after it was moved",
                    )?;
                }
                for effect_arg in effect_args {
                    if let NEffectArgValue::Value(value) = effect_arg.arg {
                        self.check_operand(
                            state,
                            moved,
                            value,
                            origin,
                            "cannot use a value after it was moved",
                        )?;
                    }
                }
                Ok(())
            }
            NExpr::ReadPlace { .. }
            | NExpr::Borrow { .. }
            | NExpr::Const(_)
            | NExpr::CodeRegionRef { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. } => Ok(()),
        }
    }

    fn check_operand(
        &self,
        state: &State,
        moved: &FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        operand: NOperand,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: &str,
    ) -> Result<(), CompleteDiagnostic> {
        let targets = self.canonicalize_value_base(state, operand.local, origin)?;
        if targets.is_empty() {
            return Ok(());
        }
        self.check_moved_overlap(moved, &targets, origin, message)
    }

    fn record_expr_moves(
        &self,
        state: &State,
        moved: &mut FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        expr: &NExpr<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        match expr {
            NExpr::Use(value)
            | NExpr::Unary { value, .. }
            | NExpr::Cast { value, .. }
            | NExpr::GetEnumTag { value }
            | NExpr::IsEnumVariant { value, .. }
            | NExpr::ExtractEnumField { value, .. } => {
                self.record_operand_move(state, moved, *value, origin)
            }
            NExpr::Binary { lhs, rhs, .. } => {
                self.record_operand_move(state, moved, *lhs, origin)?;
                self.record_operand_move(state, moved, *rhs, origin)
            }
            NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
                for field in fields {
                    self.record_operand_move(state, moved, *field, origin)?;
                }
                Ok(())
            }
            NExpr::Call {
                args, effect_args, ..
            } => {
                for arg in args {
                    self.record_operand_move(state, moved, *arg, origin)?;
                }
                for effect_arg in effect_args {
                    if let NEffectArgValue::Value(value) = effect_arg.arg {
                        self.record_operand_move(state, moved, value, origin)?;
                    }
                }
                Ok(())
            }
            NExpr::ReadPlace { .. }
            | NExpr::Borrow { .. }
            | NExpr::Const(_)
            | NExpr::CodeRegionRef { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. } => Ok(()),
        }
    }

    fn record_operand_move(
        &self,
        state: &State,
        moved: &mut FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        operand: NOperand,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        if operand.mode == ReadMode::Move && self.local_has_runtime_move_semantics(operand.local) {
            for place in self.canonicalize_value_base(state, operand.local, origin)? {
                moved.insert(place, origin);
            }
        }
        Ok(())
    }

    fn local_has_runtime_move_semantics(&self, local: crate::analysis::semantic::SLocalId) -> bool {
        self.body.local(local).is_some_and(|local| {
            local_has_runtime_move_semantics(self.db, local, &self.body.borrow_roots)
        })
    }

    fn check_moved_overlap(
        &self,
        moved: &FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        accessed: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: &str,
    ) -> Result<(), CompleteDiagnostic> {
        if let Some((_, moved_origin)) = moved.iter().find(|(moved, _)| {
            accessed
                .iter()
                .any(|accessed| places_overlap(moved, accessed))
        }) {
            let mut diag = self.move_conflict_diag(origin, message.to_string());
            self.push_secondary_origin(&mut diag, *moved_origin, "value is moved here".to_string());
            return Err(diag);
        }
        Ok(())
    }

    fn check_moved_parent(
        &self,
        moved: &FxHashMap<CanonPlace<'db>, crate::analysis::semantic::SemOrigin<'db>>,
        written: &FxHashSet<CanonPlace<'db>>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<(), CompleteDiagnostic> {
        if let Some((_, moved_origin)) = moved.iter().find(|(moved, _)| {
            written.iter().any(|written| {
                written.root == moved.root
                    && moved.proj.is_prefix_of(&written.proj)
                    && moved.proj != written.proj
            })
        }) {
            let mut diag =
                self.move_conflict_diag(origin, "cannot write through a moved value".to_string());
            self.push_secondary_origin(&mut diag, *moved_origin, "value is moved here".to_string());
            return Err(diag);
        }
        Ok(())
    }

    fn local_root(&self, local: crate::analysis::semantic::SLocalId) -> Option<NBorrowRootId> {
        self.body.local(local)?.lowering.root()
    }

    fn root_to_borrow_root(&self, root: NBorrowRootId) -> Option<BorrowRoot<'db>> {
        match self.body.root(root)? {
            NBorrowRoot::Param { param_idx, .. } => Some(BorrowRoot::Param(*param_idx)),
            NBorrowRoot::LocalSlot { local } => Some(BorrowRoot::Local(*local)),
            NBorrowRoot::Provider { binding } => Some(BorrowRoot::Provider(binding.clone())),
        }
    }

    fn successors(&self, term: &NSTerminatorKind<'db>) -> Vec<crate::analysis::semantic::SBlockId> {
        match term {
            NSTerminatorKind::Goto(bb) => vec![*bb],
            NSTerminatorKind::Branch {
                then_bb, else_bb, ..
            } => vec![*then_bb, *else_bb],
            NSTerminatorKind::MatchEnum { cases, default, .. } => {
                let mut out: Vec<_> = cases.iter().map(|(_, bb)| *bb).collect();
                if let Some(default) = default {
                    out.push(*default);
                }
                out
            }
            NSTerminatorKind::Return(_) => Vec::new(),
        }
    }

    fn borrow_conflict_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
        loan: LoanId,
    ) -> CompleteDiagnostic {
        let mut diag = self.diag(1, self.borrow_conflict_header(), origin, message);
        self.push_secondary_origin(
            &mut diag,
            self.loans[loan.0 as usize].origin,
            "borrow created here".to_string(),
        );
        diag
    }

    fn move_conflict_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> CompleteDiagnostic {
        self.diag(2, self.move_conflict_header(), origin, message)
    }

    fn invalid_return_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> CompleteDiagnostic {
        self.diag(3, self.invalid_return_header(), origin, message)
    }

    fn internal_diag(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> CompleteDiagnostic {
        self.diag(4, self.internal_error_header(), origin, message)
    }

    fn diag(
        &self,
        local_code: u16,
        header: String,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) -> CompleteDiagnostic {
        CompleteDiagnostic::new(
            Severity::Error,
            header,
            vec![SubDiagnostic::new(
                LabelStyle::Primary,
                message,
                self.span_for_origin(origin),
            )],
            Vec::new(),
            GlobalErrorCode::new(DiagnosticPass::SemanticBorrowck, local_code),
        )
    }

    fn push_secondary_origin(
        &self,
        diag: &mut CompleteDiagnostic,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        message: String,
    ) {
        diag.sub_diagnostics.push(SubDiagnostic::new(
            LabelStyle::Secondary,
            message,
            self.span_for_origin(origin),
        ));
    }

    fn span_for_origin(&self, origin: crate::analysis::semantic::SemOrigin<'db>) -> Option<Span> {
        let body = self.hir_body?;
        match origin {
            crate::analysis::semantic::SemOrigin::Expr(expr) => expr.span(body).resolve(self.db),
            crate::analysis::semantic::SemOrigin::Stmt(stmt) => stmt.span(body).resolve(self.db),
            crate::analysis::semantic::SemOrigin::Body(owner) => owner
                .body(self.db)
                .and_then(|body| body.span().resolve(self.db)),
            crate::analysis::semantic::SemOrigin::Synthetic => None,
        }
    }

    fn borrow_conflict_header(&self) -> String {
        format!(
            "borrow conflict in `fn {}`",
            checker_name(self.db, self.instance)
        )
    }

    fn move_conflict_header(&self) -> String {
        format!(
            "move conflict in `fn {}`",
            checker_name(self.db, self.instance)
        )
    }

    fn invalid_return_header(&self) -> String {
        format!(
            "invalid return borrow in `fn {}`",
            checker_name(self.db, self.instance)
        )
    }

    fn internal_error_header(&self) -> String {
        format!(
            "internal borrow checking error in `fn {}`",
            checker_name(self.db, self.instance)
        )
    }

    fn overlapping_loans_msg(&self, loan: LoanId, new_kind: BorrowKind) -> String {
        match (new_kind, self.loans[loan.0 as usize].kind) {
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

fn checker_name<'db>(db: &'db dyn HirAnalysisDb, instance: SemanticInstance<'db>) -> String {
    match instance.key(db).owner(db) {
        BodyOwner::Func(func) => match func.name(db) {
            crate::hir_def::Partial::Present(name) => name.data(db).to_string(),
            crate::hir_def::Partial::Absent => "<fn>".to_string(),
        },
        BodyOwner::Const(const_) => match const_.name(db) {
            crate::hir_def::Partial::Present(name) => name.data(db).to_string(),
            crate::hir_def::Partial::Absent => "<const>".to_string(),
        },
        BodyOwner::AnonConstBody { .. } => "<anon const>".to_string(),
        BodyOwner::ContractInit { contract } => format!(
            "{}::__init__",
            match contract.name(db) {
                crate::hir_def::Partial::Present(name) => name.data(db).to_string(),
                crate::hir_def::Partial::Absent => "<contract>".to_string(),
            }
        ),
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } => format!(
            "{}::recv[{recv_idx}][{arm_idx}]",
            match contract.name(db) {
                crate::hir_def::Partial::Present(name) => name.data(db).to_string(),
                crate::hir_def::Partial::Absent => "<contract>".to_string(),
            }
        ),
    }
}

fn semantic_borrow_summary_cycle_initial<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBorrowSummaryResult<'db> {
    let owner = instance.key(db).owner(db);
    let typed_body = instance.key(db).instantiate_typed_body(db);
    SemanticBorrowSummaryResult::Ok(
        (typed_body.result_ty().as_borrow(db).is_some() && owner.body(db).is_some())
            .then(|| BorrowSummaryId::new(db, Vec::new())),
    )
}

fn semantic_borrow_summary_cycle_recover<'db>(
    _db: &'db dyn SpannedHirAnalysisDb,
    _value: &SemanticBorrowSummaryResult<'db>,
    _count: u32,
    _instance: SemanticInstance<'db>,
) -> salsa::CycleRecoveryAction<SemanticBorrowSummaryResult<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

fn place_set_overlaps<'db>(
    lhs: &FxHashSet<CanonPlace<'db>>,
    rhs: &FxHashSet<CanonPlace<'db>>,
) -> bool {
    lhs.iter()
        .any(|lhs| rhs.iter().any(|rhs| places_overlap(lhs, rhs)))
}

fn places_overlap<'db>(lhs: &CanonPlace<'db>, rhs: &CanonPlace<'db>) -> bool {
    lhs.root == rhs.root && !matches!(lhs.proj.may_alias(&rhs.proj), Aliasing::No)
}

use common::diagnostics::{
    CompleteDiagnostic, DiagnosticPass, GlobalErrorCode, LabelStyle, Severity, Span, SubDiagnostic,
};
use cranelift_entity::EntityRef;

use crate::{
    analysis::{
        HirAnalysisDb,
        diagnostics::SpannedHirAnalysisDb,
        semantic::{NOperand, SemOrigin, SemanticInstance},
        ty::ty_check::BodyOwner,
    },
    hir_def::{Body, Partial},
    span::LazySpan,
};

use super::ir::{NormalizedSemanticBody, SemanticNormalizeError};

pub(super) fn operand_origin<'db>(operand: NOperand, fallback: SemOrigin<'db>) -> SemOrigin<'db> {
    operand.origin.map_or(fallback, SemOrigin::Expr)
}

pub(super) fn normalized_body_internal_diag<'db>(
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

pub(super) fn normalize_error_to_diag<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    instance: SemanticInstance<'db>,
    err: SemanticNormalizeError<'db>,
) -> CompleteDiagnostic {
    let owner = instance.key(db).owner(db);
    let hir_body = owner.body(db);
    let (origin, message, span) = match err {
        SemanticNormalizeError::MissingBorrowRoot { local } => {
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
        SemanticNormalizeError::IllegalCarrierPlace { local, origin } => {
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
        SemanticNormalizeError::LocalProvenanceCycle { local, .. } => (
            SemOrigin::Body(owner),
            format!(
                "detected a cycle while normalizing derived-place provenance for `%{}`",
                local.index()
            ),
            hir_body.and_then(|body| body.span().resolve(db)),
        ),
        SemanticNormalizeError::NonPlaceDerivedValue { local, base, .. } => (
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

pub(super) fn span_for_origin_from_body<'db>(
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

pub(super) fn checker_name<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> String {
    match instance.key(db).owner(db) {
        BodyOwner::Func(func) => match func.name(db) {
            Partial::Present(name) => name.data(db).to_string(),
            Partial::Absent => "<fn>".to_string(),
        },
        BodyOwner::Const(const_) => match const_.name(db) {
            Partial::Present(name) => name.data(db).to_string(),
            Partial::Absent => "<const>".to_string(),
        },
        BodyOwner::AnonConstBody { .. } => "<anon const>".to_string(),
        BodyOwner::ContractInit { contract } => format!(
            "{}::__init__",
            match contract.name(db) {
                Partial::Present(name) => name.data(db).to_string(),
                Partial::Absent => "<contract>".to_string(),
            }
        ),
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } => format!(
            "{}::recv[{recv_idx}][{arm_idx}]",
            match contract.name(db) {
                Partial::Present(name) => name.data(db).to_string(),
                Partial::Absent => "<contract>".to_string(),
            }
        ),
    }
}

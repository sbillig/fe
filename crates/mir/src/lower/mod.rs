//! MIR lowering entrypoints and shared builder scaffolding. Dispatches to submodules that handle
//! expression lowering, intrinsics, matches, aggregates (records/variants), layout, and contract
//! metadata.

use std::{error::Error, fmt};

use common::diagnostics::{CompleteDiagnostic, Severity, Span, cmp_complete_diagnostics};
use common::ingot::{Ingot, IngotKind};
use cranelift_entity::EntityRef;
use hir::analysis::{
    HirAnalysisDb,
    diagnostics::SpannedHirAnalysisDb,
    place::PlaceBase,
    ty::{
        adt_def::AdtRef,
        effects::EffectKeyKind,
        ty_check::{
            BindingSource, EffectArg, EffectParamSite, EffectPassMode, LocalBinding, ParamSite,
            PatBindingMode, RecordLike, ResolvedEffectArg, ReturnProvenance, TypedBody,
            check_func_body,
        },
        ty_def::{PrimTy, TyBase, TyData, TyId},
    },
};
use hir::hir_def::{
    ArithmeticMode, Attr, AttrArg, AttrArgValue, Body, CallableDef, Cond, CondId, Const, Expr,
    ExprId, Field, FieldIndex, Func, HirIngot, IdentId, InlineAttrErrorKind, ItemKind, LitKind,
    MatchArm, Partial, Pat, PatId, Stmt, StmtId, TopLevelMod, UnOp, VariantKind,
};

use crate::{
    capability_space::{
        PointerInfoConflict, pointer_leaf_infos_for_ty_with_default, pointer_leaf_paths_for_ty,
    },
    core_lib::CoreLib,
    ir::{
        AddressSpaceKind, BasicBlockId, BodyBuilder, CallOrigin, CallTargetRef, CodeRegionRef,
        ConstRegionId, ContractFunction, ContractFunctionKind, IntrinsicOp, LocalData, LocalId,
        LoopInfo, MirBody, MirFunction, MirInst, MirModule, MirProjection, MirProjectionPath,
        Place, PointerInfo, Rvalue, SwitchTarget, SwitchValue, SymbolSource, SyntheticValue,
        Terminator, ValueData, ValueId, ValueOrigin, ValueRepr,
    },
    monomorphize::monomorphize_functions,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rustc_hash::{FxHashMap, FxHashSet};

mod aggregates;
mod contract;
mod contracts;
mod diagnostics;
mod expr;
mod intrinsics;
mod match_lowering;
mod prepass;
mod variants;

pub(super) use contract::extract_contract_function;
use hir::span::LazySpan;

/// Errors that can occur while lowering HIR into MIR.
#[derive(Debug)]
pub enum MirLowerError {
    MissingBody {
        func_name: String,
    },
    AnalysisDiagnostics {
        func_name: String,
        diagnostics: String,
    },
    MirDiagnostics {
        func_name: String,
        diagnostics: String,
    },
    UnloweredHirExpr {
        func_name: String,
        expr: String,
    },
    Unsupported {
        func_name: String,
        message: String,
    },
}

impl fmt::Display for MirLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirLowerError::MissingBody { func_name } => {
                write!(f, "function `{func_name}` is missing a body")
            }
            MirLowerError::AnalysisDiagnostics {
                func_name: _,
                diagnostics,
            } => {
                write!(f, "{diagnostics}")
            }
            MirLowerError::MirDiagnostics {
                func_name: _,
                diagnostics,
            } => {
                write!(f, "{diagnostics}")
            }
            MirLowerError::UnloweredHirExpr { func_name, expr } => {
                write!(
                    f,
                    "unlowered HIR expression survived MIR lowering in `{func_name}`: {expr}"
                )
            }
            MirLowerError::Unsupported { func_name, message } => {
                write!(f, "unsupported while lowering `{func_name}`: {message}")
            }
        }
    }
}

impl Error for MirLowerError {}

pub type MirLowerResult<T> = Result<T, MirLowerError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirDiagnosticsMode {
    TemplatesOnly,
    CompilerParity,
}

#[derive(Debug, Default)]
pub struct MirDiagnosticsOutput {
    pub diagnostics: Vec<CompleteDiagnostic>,
    pub internal_errors: Vec<MirLowerError>,
}

fn sort_and_dedup_complete_diagnostics(diags: &mut Vec<CompleteDiagnostic>) {
    diags.sort_by(cmp_complete_diagnostics);
    diags.dedup();
}

fn collect_hir_error_diagnostics<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mods: impl IntoIterator<Item = TopLevelMod<'db>>,
) -> Vec<CompleteDiagnostic> {
    let mut pass_manager = hir::analysis::initialize_analysis_pass();
    let mut diags = Vec::new();
    for top_mod in top_mods {
        diags.extend(pass_manager.run_on_module(db, top_mod));
    }

    let mut complete_diags: Vec<_> = diags
        .into_iter()
        .map(|diag| diag.to_complete(db))
        .filter(|diag| diag.severity == Severity::Error)
        .collect();
    sort_and_dedup_complete_diagnostics(&mut complete_diags);
    complete_diags
}

fn collect_hir_error_diagnostics_for_top_mod<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Vec<CompleteDiagnostic> {
    collect_hir_error_diagnostics(db, std::iter::once(top_mod))
}

fn collect_hir_error_diagnostics_for_ingot<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    ingot: Ingot<'db>,
) -> Vec<CompleteDiagnostic> {
    let mut top_mods = ingot.all_modules(db).clone();
    for &(_, dep_ingot) in ingot.resolved_external_ingots(db) {
        top_mods.extend(dep_ingot.all_modules(db));
    }
    collect_hir_error_diagnostics(db, top_mods)
}

fn analysis_diagnostics_error(
    db: &dyn SpannedHirAnalysisDb,
    subject: String,
    diags: Vec<CompleteDiagnostic>,
) -> Option<MirLowerError> {
    (!diags.is_empty()).then(|| MirLowerError::AnalysisDiagnostics {
        func_name: subject,
        diagnostics: hir::analysis::diagnostics::format_diags(db, diags.iter()),
    })
}

fn invalid_hir_error_for_top_mod<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Option<MirLowerError> {
    let subject = top_mod
        .scope()
        .pretty_path(db)
        .unwrap_or_else(|| "<module>".to_string());
    analysis_diagnostics_error(
        db,
        subject,
        collect_hir_error_diagnostics_for_top_mod(db, top_mod),
    )
}

fn invalid_hir_error_for_ingot<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    ingot: Ingot<'db>,
) -> Option<MirLowerError> {
    let subject = ingot
        .module_tree(db)
        .root_data()
        .and_then(|root| root.top_mod.scope().pretty_path(db))
        .unwrap_or_else(|| "<ingot>".to_string());
    analysis_diagnostics_error(
        db,
        subject,
        collect_hir_error_diagnostics_for_ingot(db, ingot),
    )
}

fn collect_funcs_to_lower<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Vec<Func<'db>> {
    let mut funcs_to_lower = Vec::new();
    let mut seen = FxHashSet::default();

    let mut queue_func = |func: Func<'db>| {
        if seen.insert(func) {
            funcs_to_lower.push(func);
        }
    };

    // Skip associated functions here to avoid pulling in trait methods (which may refer to
    // abstract associated items) as MIR templates. Impl/impl-trait functions are queued below.
    for &func in top_mod.all_funcs(db) {
        if !func.is_associated_func(db) {
            queue_func(func);
        }
    }
    for &impl_block in top_mod.all_impls(db) {
        for func in impl_block.funcs(db) {
            queue_func(func);
        }
    }
    for &impl_trait in top_mod.all_impl_traits(db) {
        for func in impl_trait.methods(db) {
            queue_func(func);
        }
    }

    funcs_to_lower
}

fn mir_func_name<'db>(db: &'db dyn SpannedHirAnalysisDb, func: &MirFunction<'db>) -> String {
    match func.origin {
        crate::ir::MirFunctionOrigin::Hir(hir_func) => hir_func.pretty_print_signature(db),
        crate::ir::MirFunctionOrigin::Synthetic(_) => func.symbol_name.clone(),
    }
}

fn invalid_inline_attr_error<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    func: Func<'db>,
) -> Option<MirLowerError> {
    let kind = func.inline_attr_error(db)?;
    let func_name = func
        .name(db)
        .to_opt()
        .map(|ident| ident.data(db).to_string())
        .unwrap_or_else(|| "<anonymous>".to_string());

    let diagnostics = match kind {
        InlineAttrErrorKind::Duplicate => {
            "duplicate `#[inline]` attribute; functions support at most one inline hint"
                .to_string()
        }
        InlineAttrErrorKind::InvalidForm => {
            "invalid `#[inline]` attribute; expected `#[inline]`, `#[inline(always)]`, or `#[inline(never)]`"
                .to_string()
        }
    };

    Some(MirLowerError::AnalysisDiagnostics {
        func_name,
        diagnostics,
    })
}

fn runtime_field_projection_prefix<'db>(
    db: &'db dyn HirAnalysisDb,
    owner_ty: TyId<'db>,
    field_idx: usize,
) -> MirProjectionPath<'db> {
    if crate::repr::transparent_field0_inner_ty(db, owner_ty, field_idx).is_some() {
        MirProjectionPath::new()
    } else {
        MirProjectionPath::from_projection(MirProjection::Field(field_idx))
    }
}

fn binding_source_projection<'db>(
    db: &'db dyn HirAnalysisDb,
    typed_body: &TypedBody<'db>,
    source: &BindingSource,
) -> MirProjectionPath<'db> {
    let mut path = MirProjectionPath::new();
    let mut owner_ty = typed_body.expr_ty(db, source.init_expr);
    for field_idx in source.field_path.iter().copied() {
        path = path.concat(&runtime_field_projection_prefix(db, owner_ty, field_idx));
        let Some(next_ty) = owner_ty.field_types(db).get(field_idx).copied() else {
            break;
        };
        owner_ty = next_ty;
    }
    path
}

fn field_access_idx<'db>(
    db: &'db dyn HirAnalysisDb,
    owner_ty: TyId<'db>,
    field_index: FieldIndex<'db>,
) -> Option<usize> {
    let record_like = RecordLike::from_ty(owner_ty);
    match field_index {
        FieldIndex::Ident(label) => record_like.record_field_idx(db, label),
        FieldIndex::Index(integer) => integer.data(db).to_usize(),
    }
}

fn forwarded_return_leaf_source_from_expr<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    typed_body: &TypedBody<'db>,
    expr: ExprId,
    target_path: &MirProjectionPath<'db>,
    seen: &mut FxHashSet<Func<'db>>,
    visited_locals: &mut FxHashSet<PatId>,
) -> Option<(usize, MirProjectionPath<'db>)> {
    let Partial::Present(expr_data) = expr.data(db, body) else {
        return None;
    };

    match expr_data {
        Expr::Block(stmts) => {
            let tail = stmts.last()?;
            match tail.data(db, body) {
                Partial::Present(Stmt::Expr(tail_expr)) => forwarded_return_leaf_source_from_expr(
                    db,
                    body,
                    typed_body,
                    *tail_expr,
                    target_path,
                    seen,
                    visited_locals,
                ),
                Partial::Present(Stmt::Return(Some(return_expr))) => {
                    forwarded_return_leaf_source_from_expr(
                        db,
                        body,
                        typed_body,
                        *return_expr,
                        target_path,
                        seen,
                        visited_locals,
                    )
                }
                _ => None,
            }
        }
        Expr::If(_, then_expr, else_expr) => {
            let then_source = forwarded_return_leaf_source_from_expr(
                db,
                body,
                typed_body,
                *then_expr,
                target_path,
                seen,
                visited_locals,
            )?;
            let else_source = forwarded_return_leaf_source_from_expr(
                db,
                body,
                typed_body,
                *else_expr.as_ref()?,
                target_path,
                seen,
                visited_locals,
            )?;
            (then_source == else_source).then_some(then_source)
        }
        Expr::Match(_, arms) => {
            let Partial::Present(arms) = arms else {
                return None;
            };
            let mut sources = arms.iter().filter_map(|arm| {
                forwarded_return_leaf_source_from_expr(
                    db,
                    body,
                    typed_body,
                    arm.body,
                    target_path,
                    seen,
                    visited_locals,
                )
            });
            let first = sources.next()?;
            sources.all(|source| source == first).then_some(first)
        }
        Expr::With(_, with_body) => forwarded_return_leaf_source_from_expr(
            db,
            body,
            typed_body,
            *with_body,
            target_path,
            seen,
            visited_locals,
        ),
        Expr::Un(inner, UnOp::Mut | UnOp::Ref) => forwarded_return_leaf_source_from_expr(
            db,
            body,
            typed_body,
            *inner,
            target_path,
            seen,
            visited_locals,
        ),
        Expr::Path(_) => match typed_body.expr_binding(expr)? {
            binding @ LocalBinding::Param { idx, .. } => typed_body
                .path_expr_preserves_binding_ty(db, expr, binding)
                .then_some((idx, target_path.clone())),
            binding @ LocalBinding::Local { pat, .. } => {
                if !visited_locals.insert(pat) {
                    return None;
                }
                if !typed_body.path_expr_preserves_binding_ty(db, expr, binding) {
                    visited_locals.remove(&pat);
                    return None;
                }
                let source = typed_body.binding_source(db, binding).and_then(|source| {
                    let source_path =
                        binding_source_projection(db, typed_body, &source).concat(target_path);
                    forwarded_return_leaf_source_from_expr(
                        db,
                        body,
                        typed_body,
                        source.init_expr,
                        &source_path,
                        seen,
                        visited_locals,
                    )
                });
                visited_locals.remove(&pat);
                source
            }
            LocalBinding::EffectParam { .. } => None,
        },
        Expr::Field(owner, field) => {
            let owner_ty = typed_body.expr_ty(db, *owner);
            let field_idx = field_access_idx(db, owner_ty, field.to_opt()?)?;
            let prefixed_target =
                runtime_field_projection_prefix(db, owner_ty, field_idx).concat(target_path);
            forwarded_return_leaf_source_from_expr(
                db,
                body,
                typed_body,
                *owner,
                &prefixed_target,
                seen,
                visited_locals,
            )
        }
        Expr::Call(_, args) => {
            let callable = typed_body.callable_expr(expr)?;
            let CallableDef::Func(func) = callable.callable_def else {
                return None;
            };
            let (arg_idx, arg_path) = forwarded_return_leaf_sources_from_callable(
                db,
                func,
                typed_body.expr_ty(db, expr),
                seen,
            )?
            .into_iter()
            .find_map(|(leaf_path, arg_idx, arg_path)| {
                (leaf_path == *target_path).then_some((arg_idx, arg_path))
            })?;
            let arg_expr = args.get(arg_idx)?.expr;
            forwarded_return_leaf_source_from_expr(
                db,
                body,
                typed_body,
                arg_expr,
                &arg_path,
                seen,
                visited_locals,
            )
        }
        Expr::MethodCall(receiver, _, _, args) => {
            let callable = typed_body.callable_expr(expr)?;
            let CallableDef::Func(func) = callable.callable_def else {
                return None;
            };
            let (arg_idx, arg_path) = forwarded_return_leaf_sources_from_callable(
                db,
                func,
                typed_body.expr_ty(db, expr),
                seen,
            )?
            .into_iter()
            .find_map(|(leaf_path, arg_idx, arg_path)| {
                (leaf_path == *target_path).then_some((arg_idx, arg_path))
            })?;
            let arg_expr = if arg_idx == 0 {
                *receiver
            } else {
                args.get(arg_idx - 1)?.expr
            };
            forwarded_return_leaf_source_from_expr(
                db,
                body,
                typed_body,
                arg_expr,
                &arg_path,
                seen,
                visited_locals,
            )
        }
        Expr::RecordInit(_, fields) => {
            let owner_ty = typed_body.expr_ty(db, expr);
            fields.iter().find_map(|field| {
                let label = field.label_eagerly(db, body)?;
                let field_idx = field_access_idx(db, owner_ty, FieldIndex::Ident(label))?;
                let field_prefix = runtime_field_projection_prefix(db, owner_ty, field_idx);
                let suffix = crate::ir::projection_strip_prefix(target_path, &field_prefix)?;
                forwarded_return_leaf_source_from_expr(
                    db,
                    body,
                    typed_body,
                    field.expr,
                    &suffix,
                    seen,
                    visited_locals,
                )
            })
        }
        Expr::Tuple(items) => {
            let owner_ty = typed_body.expr_ty(db, expr);
            items.iter().enumerate().find_map(|(field_idx, item_expr)| {
                let field_prefix = runtime_field_projection_prefix(db, owner_ty, field_idx);
                let suffix = crate::ir::projection_strip_prefix(target_path, &field_prefix)?;
                forwarded_return_leaf_source_from_expr(
                    db,
                    body,
                    typed_body,
                    *item_expr,
                    &suffix,
                    seen,
                    visited_locals,
                )
            })
        }
        _ => None,
    }
}

pub(crate) fn forwarded_return_leaf_sources_from_callable<'db>(
    db: &'db dyn HirAnalysisDb,
    func: Func<'db>,
    dest_ty: TyId<'db>,
    seen: &mut FxHashSet<Func<'db>>,
) -> Option<Vec<(MirProjectionPath<'db>, usize, MirProjectionPath<'db>)>> {
    if !seen.insert(func) {
        return None;
    }

    let (diags, typed_body) = check_func_body(db, func);
    if !diags.is_empty() {
        seen.remove(&func);
        return None;
    }
    let body = typed_body.body()?;
    let root_expr = func.body(db)?.expr(db);
    let core = CoreLib::new(db, func.scope());
    let mut out = Vec::new();
    let mut visited_locals = FxHashSet::default();
    for leaf_path in pointer_leaf_paths_for_ty(db, &core, dest_ty) {
        if let Some((arg_idx, arg_path)) = forwarded_return_leaf_source_from_expr(
            db,
            body,
            typed_body,
            root_expr,
            &leaf_path,
            seen,
            &mut visited_locals,
        ) {
            out.push((leaf_path, arg_idx, arg_path));
        }
    }
    seen.remove(&func);
    (!out.is_empty()).then_some(out)
}

pub(crate) fn call_return_pointer_leaf_infos<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    values: &[ValueData<'db>],
    locals: &[LocalData<'db>],
    call: &CallOrigin<'db>,
    dest_ty: TyId<'db>,
) -> Vec<(MirProjectionPath<'db>, PointerInfo<'db>)> {
    let Some(CallTargetRef::Hir(hir_target)) = call.target.as_ref() else {
        return Vec::new();
    };
    let CallableDef::Func(func) = hir_target.callable_def else {
        return Vec::new();
    };
    let mut seen = FxHashSet::default();
    let Some(leaf_sources) =
        forwarded_return_leaf_sources_from_callable(db, func, dest_ty, &mut seen)
    else {
        return Vec::new();
    };

    let mut infos = Vec::new();
    for (dest_path, arg_idx, arg_path) in &leaf_sources {
        let Some(arg_value) = call.args.get(*arg_idx).copied() else {
            continue;
        };
        let source_infos =
            crate::repr::pointer_leaf_infos_for_value(db, core, values, locals, arg_value);
        let retag_info = |info: PointerInfo<'db>| {
            pointer_leaf_infos_for_ty_with_default(db, core, dest_ty, info.address_space)
                .into_iter()
                .find_map(|(path, typed_info)| (path == *dest_path).then_some(typed_info))
                .unwrap_or(info)
        };
        let mut matched = false;
        for (source_path, info) in source_infos {
            if source_path == *arg_path {
                infos.push((dest_path.clone(), retag_info(info)));
                matched = true;
            }
        }
        if !matched
            && arg_path.is_empty()
            && let Some(info) = crate::ir::try_value_pointer_info_in(values, locals, arg_value)
        {
            infos.push((dest_path.clone(), retag_info(info)));
        }
    }

    infos
}

fn callable_forwarded_param_is_immutable<'db>(
    db: &'db dyn HirAnalysisDb,
    func: hir::hir_def::Func<'db>,
    idx: usize,
) -> bool {
    func.params(db).nth(idx).is_some_and(|param| {
        !param.is_mut(db) && !crate::repr::ty_has_mut_capability(db, param.ty(db))
    })
}

fn symbol_source_for_ingot_kind(kind: IngotKind) -> SymbolSource {
    match kind {
        IngotKind::Core | IngotKind::Std => SymbolSource::Internal,
        IngotKind::StandAlone | IngotKind::Local | IngotKind::External => SymbolSource::User,
    }
}

fn run_borrow_checks_collect<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    functions: &[MirFunction<'db>],
    output: &mut MirDiagnosticsOutput,
) {
    match crate::analysis::borrowck::compute_borrow_summaries(db, functions) {
        Ok(borrow_summaries) => {
            for func in functions {
                if let Some(diag) =
                    crate::analysis::borrowck::check_borrows(db, func, &borrow_summaries)
                {
                    output.diagnostics.push(diag);
                }
            }
        }
        Err(err) => output.diagnostics.push(err.diagnostic.clone()),
    }
}

fn run_borrow_checks_or_error<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    functions: &[MirFunction<'db>],
) -> MirLowerResult<()> {
    let borrow_summaries = crate::analysis::borrowck::compute_borrow_summaries(db, functions)
        .map_err(|err| {
            let diagnostics = hir::analysis::diagnostics::format_diags(db, [&err.diagnostic]);
            MirLowerError::MirDiagnostics {
                func_name: err.func_name,
                diagnostics,
            }
        })?;

    for func in functions {
        if let Some(diag) = crate::analysis::borrowck::check_borrows(db, func, &borrow_summaries) {
            let diagnostics = hir::analysis::diagnostics::format_diags(db, [&diag]);
            return Err(MirLowerError::MirDiagnostics {
                func_name: mir_func_name(db, func),
                diagnostics,
            });
        }
    }
    Ok(())
}

pub fn collect_mir_diagnostics<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
    mode: MirDiagnosticsMode,
) -> MirDiagnosticsOutput {
    let mut output = MirDiagnosticsOutput::default();
    if !collect_hir_error_diagnostics_for_top_mod(db, top_mod).is_empty() {
        return output;
    }
    let mut templates = Vec::new();

    for func in collect_funcs_to_lower(db, top_mod) {
        if func.body(db).is_none() {
            continue;
        }
        if let Some(err) = invalid_inline_attr_error(db, func) {
            output.internal_errors.push(err);
            continue;
        }
        let (diags, typed_body) = check_func_body(db, func);
        if !diags.is_empty() {
            continue;
        }
        match lower_function(
            db,
            func,
            typed_body.clone(),
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ) {
            Ok(lowered) => templates.push(lowered),
            Err(err) => output.internal_errors.push(err),
        }
    }

    match contracts::lower_contract_templates(db, top_mod) {
        Ok(contract_templates) => templates.extend(contract_templates),
        Err(err) => output.internal_errors.push(err),
    }

    run_borrow_checks_collect(db, &templates, &mut output);

    if matches!(mode, MirDiagnosticsMode::TemplatesOnly) {
        return output;
    }

    let mut functions = match monomorphize_functions(db, templates) {
        Ok(functions) => functions,
        Err(err) => {
            output.internal_errors.push(err);
            return output;
        }
    };
    for func in &mut functions {
        crate::transform::canonicalize_transparent_newtypes(db, &mut func.body);
        crate::transform::insert_temp_binds(db, &mut func.body);
    }

    for func in &functions {
        if let Some(diag) = crate::analysis::noesc::check_noesc_escapes(db, func) {
            output.diagnostics.push(diag);
        }
    }

    run_borrow_checks_collect(db, &functions, &mut output);

    output
}

/// Field type and byte offset information used when lowering record/variant accesses.
pub(super) struct FieldAccessInfo<'db> {
    pub(super) field_ty: TyId<'db>,
    pub(super) field_idx: usize,
}

/// Lowers every function within the top-level module into MIR.
///
/// # Parameters
/// - `db`: HIR analysis database.
/// - `top_mod`: The module containing functions/impls to lower.
///
/// # Returns
/// A populated `MirModule` on success.
pub fn lower_module<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> MirLowerResult<MirModule<'db>> {
    if let Some(err) = invalid_hir_error_for_top_mod(db, top_mod) {
        return Err(err);
    }

    let mut templates = Vec::new();
    for func in collect_funcs_to_lower(db, top_mod) {
        if func.body(db).is_none() {
            continue;
        }
        if let Some(err) = invalid_inline_attr_error(db, func) {
            return Err(err);
        }
        let (diags, typed_body) = check_func_body(db, func);
        if !diags.is_empty() {
            let func_name = func
                .name(db)
                .to_opt()
                .map(|ident| ident.data(db).to_string())
                .unwrap_or_else(|| "<anonymous>".to_string());
            let rendered = diagnostics::format_func_body_diags(db, diags);
            return Err(MirLowerError::AnalysisDiagnostics {
                func_name,
                diagnostics: rendered,
            });
        }
        let lowered = lower_function(
            db,
            func,
            typed_body.clone(),
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        templates.push(lowered);
    }

    templates.extend(contracts::lower_contract_templates(db, top_mod)?);

    // Run MIR diagnostics on the generic templates as well as the monomorphized instances. This
    // ensures borrow/move errors are surfaced even when a generic function is never instantiated.
    run_borrow_checks_or_error(db, &templates)?;

    let mut functions = monomorphize_functions(db, templates)?;
    for func in &mut functions {
        crate::transform::canonicalize_transparent_newtypes(db, &mut func.body);
        crate::transform::insert_temp_binds(db, &mut func.body);
    }
    validate_lowered_mir_functions(db, &functions)?;
    for func in &functions {
        if let Some(diag) = crate::analysis::noesc::check_noesc_escapes(db, func) {
            let func_name = match func.origin {
                crate::ir::MirFunctionOrigin::Hir(hir_func) => hir_func.pretty_print_signature(db),
                crate::ir::MirFunctionOrigin::Synthetic(_) => func.symbol_name.clone(),
            };
            let diagnostics = hir::analysis::diagnostics::format_diags(db, [&diag]);
            return Err(MirLowerError::MirDiagnostics {
                func_name,
                diagnostics,
            });
        }
    }
    run_borrow_checks_or_error(db, &functions)?;

    // Lower semantic capability MIR into backend-neutral runtime representation MIR.
    let core = CoreLib::new(db, top_mod.scope());
    for func in &mut functions {
        crate::transform::lower_capability_to_repr(db, &core, &mut func.body);
        crate::transform::canonicalize_transparent_newtypes(db, &mut func.body);
        crate::transform::insert_temp_binds(db, &mut func.body);
        crate::transform::canonicalize_zero_sized(db, &mut func.body);
    }
    validate_lowered_mir_functions(db, &functions)?;
    let mut module = MirModule { top_mod, functions };
    crate::transform::normalize_runtime_abi(db, &mut module);
    crate::transform::eliminate_dead_erased_arg_materializations(db, &mut module);
    crate::transform::normalize_runtime_shapes(db, &mut module);
    validate_lowered_mir_functions(db, &module.functions)?;
    Ok(module)
}

/// Lowers every function within every top-level module of an ingot into MIR.
///
/// This is primarily useful for emitting artifacts (e.g. `fe build`) across a whole ingot: all
/// contracts and contract entrypoints in any source file should be considered part of the same
/// compilation scope.
pub fn lower_ingot<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    ingot: Ingot<'db>,
) -> MirLowerResult<MirModule<'db>> {
    if let Some(err) = invalid_hir_error_for_ingot(db, ingot) {
        return Err(err);
    }

    let mut templates = Vec::new();
    let mut funcs_to_lower = Vec::new();
    let mut seen = FxHashSet::default();

    let mut queue_func = |func: Func<'db>| {
        if seen.insert(func) {
            funcs_to_lower.push(func);
        }
    };

    for &top_mod in ingot.all_modules(db).iter() {
        // Skip associated functions here to avoid pulling in trait methods (which may refer to
        // abstract associated items) as MIR templates. Impl/impl-trait functions are queued below.
        for &func in top_mod.all_funcs(db) {
            if !func.is_associated_func(db) {
                queue_func(func);
            }
        }

        for &impl_block in top_mod.all_impls(db) {
            for func in impl_block.funcs(db) {
                queue_func(func);
            }
        }
        for &impl_trait in top_mod.all_impl_traits(db) {
            for func in impl_trait.methods(db) {
                queue_func(func);
            }
        }
    }

    for func in funcs_to_lower {
        if func.body(db).is_none() {
            continue;
        }
        if let Some(err) = invalid_inline_attr_error(db, func) {
            return Err(err);
        }
        let (diags, typed_body) = check_func_body(db, func);
        if !diags.is_empty() {
            let func_name = func
                .name(db)
                .to_opt()
                .map(|ident| ident.data(db).to_string())
                .unwrap_or_else(|| "<anonymous>".to_string());
            let rendered = diagnostics::format_func_body_diags(db, diags);
            return Err(MirLowerError::AnalysisDiagnostics {
                func_name,
                diagnostics: rendered,
            });
        }
        let lowered = lower_function(
            db,
            func,
            typed_body.clone(),
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        templates.push(lowered);
    }

    for &top_mod in ingot.all_modules(db).iter() {
        templates.extend(contracts::lower_contract_templates(db, top_mod)?);
    }

    // Also generate contract templates for contracts defined in dependency
    // ingots. This is needed so that `create2<SomeContract>` works when
    // `SomeContract` lives in a different ingot within the same workspace.
    // The TargetContext is created from the *current* ingot's root module so
    // that `std::evm::EvmTarget` etc. resolve correctly.
    let host_top_mod = ingot.root_mod(db);
    for &(dep_name, dep_ingot) in ingot.resolved_external_ingots(db).iter() {
        templates.extend(contracts::lower_dependency_contract_templates(
            db,
            host_top_mod,
            dep_ingot,
            dep_name.data(db),
        )?);
    }

    // Run MIR diagnostics on the generic templates as well as the monomorphized instances. This
    // ensures borrow/move errors are surfaced even when a generic function is never instantiated.
    run_borrow_checks_or_error(db, &templates)?;

    let mut functions = monomorphize_functions(db, templates)?;
    for func in &mut functions {
        crate::transform::canonicalize_transparent_newtypes(db, &mut func.body);
        crate::transform::insert_temp_binds(db, &mut func.body);
    }
    validate_lowered_mir_functions(db, &functions)?;

    for func in &functions {
        if let Some(diag) = crate::analysis::noesc::check_noesc_escapes(db, func) {
            let diagnostics = hir::analysis::diagnostics::format_diags(db, [&diag]);
            return Err(MirLowerError::MirDiagnostics {
                func_name: mir_func_name(db, func),
                diagnostics,
            });
        }
    }
    run_borrow_checks_or_error(db, &functions)?;

    // Lower semantic capability MIR into backend-neutral runtime representation MIR.
    let root_mod = ingot.root_mod(db);
    let core = CoreLib::new(db, root_mod.scope());
    for func in &mut functions {
        crate::transform::lower_capability_to_repr(db, &core, &mut func.body);
        crate::transform::canonicalize_transparent_newtypes(db, &mut func.body);
        crate::transform::insert_temp_binds(db, &mut func.body);
        crate::transform::canonicalize_zero_sized(db, &mut func.body);
    }
    validate_lowered_mir_functions(db, &functions)?;
    let mut module = MirModule {
        top_mod: root_mod,
        functions,
    };
    crate::transform::normalize_runtime_abi(db, &mut module);
    crate::transform::eliminate_dead_erased_arg_materializations(db, &mut module);
    crate::transform::normalize_runtime_shapes(db, &mut module);
    validate_lowered_mir_functions(db, &module.functions)?;
    Ok(module)
}

/// Lowers a single HIR function (with its typed body) into a MIR function template.
///
/// # Parameters
/// - `db`: HIR analysis database.
/// - `func`: Function definition to lower.
/// - `typed_body`: Type-checked function body.
///
/// # Returns
/// The lowered MIR function template or an error when the function is missing a body.
pub(crate) fn lower_function<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    func: Func<'db>,
    typed_body: TypedBody<'db>,
    receiver_space: Option<AddressSpaceKind>,
    generic_args: Vec<TyId<'db>>,
    effect_param_space_overrides: Vec<Option<AddressSpaceKind>>,
    param_capability_space_overrides: Vec<Vec<(MirProjectionPath<'db>, AddressSpaceKind)>>,
) -> MirLowerResult<MirFunction<'db>> {
    let symbol_name = func
        .name(db)
        .to_opt()
        .map(|ident| ident.data(db).to_string())
        .unwrap_or_else(|| "<anonymous>".into());
    let symbol_source = symbol_source_for_ingot_kind(func.top_mod(db).ingot(db).kind(db));
    let contract_function = extract_contract_function(db, func);

    let Some(body) = func.body(db) else {
        return Err(MirLowerError::MissingBody {
            func_name: symbol_name,
        });
    };

    let mut builder = MirBuilder::new_for_func(
        db,
        func,
        body,
        &typed_body,
        &generic_args,
        LoweringOverrides {
            receiver_space,
            effect_param_space_overrides: &effect_param_space_overrides,
            param_capability_space_overrides: &param_capability_space_overrides,
        },
    )?;
    let entry = builder.builder.entry_block();
    builder.move_to_block(entry);
    builder.lower_root(body.expr(db));
    builder.ensure_const_expr_values();
    if let Some(block) = builder.current_block() {
        let ret_ty = builder.return_ty;
        let returns_value = !builder.is_unit_ty(ret_ty) && !ret_ty.is_never(db);
        let source = builder.source_for_expr(body.expr(db));
        if returns_value {
            let ret_val = builder.ensure_value(body.expr(db));
            builder.set_terminator(
                block,
                Terminator::Return {
                    source,
                    value: Some(ret_val),
                },
            );
        } else {
            builder.set_terminator(
                block,
                Terminator::Return {
                    source,
                    value: None,
                },
            );
        }
    }
    let deferred_error = builder.deferred_error.take();
    let effect_param_provider_tys = builder.effect_param_provider_tys.clone();
    let ret_ty = builder.return_ty;
    let returns_value = !crate::layout::is_zero_sized_ty(db, ret_ty);
    let runtime_return_shape =
        crate::repr::runtime_return_shape_seed_for_ty(db, &builder.core, ret_ty);
    let mir_body = builder.finish();

    if let Some(err) = deferred_error {
        return Err(err);
    }

    // Generic functions are re-lowered from HIR during monomorphization, so their initial
    // templates are never codegen'd. Allow construction-time placeholders here.
    let is_uninstantiated_generic =
        generic_args.is_empty() && !CallableDef::Func(func).params(db).is_empty();
    if !is_uninstantiated_generic {
        validate_lowered_mir_body(db, &symbol_name, body, &mir_body)?;
    }

    // Note: `MirFunction` may be used as a generic template during monomorphization.
    // Monomorphic instances get a fully-instantiated + normalized `ret_ty` in the
    // monomorphizer; this is the declared return type.
    let runtime_abi = crate::ir::RuntimeAbi::source_shaped(
        mir_body.param_locals.len(),
        effect_param_provider_tys,
    );

    Ok(MirFunction {
        origin: crate::ir::MirFunctionOrigin::Hir(func),
        body: mir_body,
        typed_body: Some(typed_body),
        generic_args,
        ret_ty,
        returns_value,
        runtime_abi,
        runtime_return_shape,
        runtime_return_pointer_leaf_infos: Vec::new(),
        contract_function,
        inline_hint: func.inline_hint(db),
        symbol_name,
        symbol_source,
        receiver_space,
        defer_root: false,
    })
}

/// Stateful helper that incrementally constructs MIR while walking HIR.
pub(super) struct MirBuilder<'db, 'a> {
    pub(super) db: &'db dyn SpannedHirAnalysisDb,
    pub(super) hir_func: Option<Func<'db>>,
    pub(super) arithmetic_mode: ArithmeticMode,
    pub(super) body: Body<'db>,
    pub(super) typed_body: &'a TypedBody<'db>,
    pub(super) generic_args: &'a [TyId<'db>],
    pub(super) return_ty: TyId<'db>,
    pub(super) builder: BodyBuilder<'db>,
    pub(super) core: CoreLib<'db>,
    pub(super) loop_stack: Vec<LoopScope>,
    pub(super) const_cache: FxHashMap<Const<'db>, ValueId>,
    pub(super) const_array_region_cache: FxHashMap<Const<'db>, (TyId<'db>, ConstRegionId)>,
    pub(super) source_info_cache: FxHashMap<Span, crate::ir::SourceInfoId>,
    pub(super) pat_address_space: FxHashMap<PatId, AddressSpaceKind>,
    pub(super) binding_locals: FxHashMap<LocalBinding<'db>, LocalId>,
    pub(super) address_taken_locals: FxHashSet<LocalId>,
    pub(super) expr_lower_states: Vec<ExprLowerState>,
    /// For methods, the address space variant being lowered.
    pub(super) receiver_space: Option<AddressSpaceKind>,
    /// Address space for each effect parameter, indexed by effect param position.
    pub(super) effect_param_spaces: Vec<AddressSpaceKind>,
    /// Address space overrides for effect bindings not tied to a function effect list.
    pub(super) effect_binding_spaces: FxHashMap<LocalBinding<'db>, AddressSpaceKind>,
    /// Provider type metadata aligned to `MirBody::effect_param_locals`.
    pub(super) effect_param_provider_tys: Vec<Option<TyId<'db>>>,
    /// Capability-space overrides for function parameters, indexed by parameter position.
    pub(super) param_capability_space_overrides:
        Vec<Vec<(MirProjectionPath<'db>, AddressSpaceKind)>>,
    /// Deferred error from intrinsic lowering (e.g. `size_of` on an unsupported type).
    pub(super) deferred_error: Option<MirLowerError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExprLowerState {
    NotStarted,
    InProgress,
    Done,
}

/// Loop context capturing break/continue targets.
#[derive(Clone, Copy)]
pub(super) struct LoopScope {
    pub(super) continue_target: BasicBlockId,
    pub(super) break_target: BasicBlockId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EffectProviderInferenceRationale {
    ConcreteProviderTy,
    DeclaredEffectKeyProviderTy,
    ByRefProviderDefaultsToMemory,
    ContractFieldProviderTy,
    ByTempPlaceMemPtr,
    ForwardedEffectParamProviderTy,
    ForwardedEffectParamFallbackMemPtr,
    ContractFieldFallbackStorPtr,
    DefaultMemPtr,
    StorageDefault,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct InferredEffectProvider<'db> {
    pub(super) provider_ty: Option<TyId<'db>>,
    pub(super) address_space: AddressSpaceKind,
    pub(super) rationale: EffectProviderInferenceRationale,
}

#[derive(Debug, Clone, Copy)]
struct LoweringOverrides<'a, 'db> {
    receiver_space: Option<AddressSpaceKind>,
    effect_param_space_overrides: &'a [Option<AddressSpaceKind>],
    param_capability_space_overrides: &'a [Vec<(MirProjectionPath<'db>, AddressSpaceKind)>],
}

impl<'db, 'a> MirBuilder<'db, 'a> {
    /// Constructs a new builder for the given HIR body and typed information.
    ///
    /// # Parameters
    /// - `db`: HIR analysis database.
    /// - `body`: HIR body being lowered.
    /// - `typed_body`: Type-checked body information.
    ///
    /// # Returns
    /// A ready-to-use `MirBuilder` or an error if core helpers are missing.
    #[allow(clippy::too_many_arguments)]
    fn new(
        db: &'db dyn SpannedHirAnalysisDb,
        hir_func: Option<Func<'db>>,
        body: Body<'db>,
        typed_body: &'a TypedBody<'db>,
        generic_args: &'a [TyId<'db>],
        return_ty: TyId<'db>,
        receiver_space: Option<AddressSpaceKind>,
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
        param_capability_space_overrides: &[Vec<(MirProjectionPath<'db>, AddressSpaceKind)>],
    ) -> Result<Self, MirLowerError> {
        let core = CoreLib::new(db, body.scope());
        let arithmetic_mode = hir_func
            .map(|func| func.arithmetic_mode(db))
            .unwrap_or(ArithmeticMode::Checked);

        let mut builder = Self {
            db,
            hir_func,
            arithmetic_mode,
            body,
            typed_body,
            generic_args,
            return_ty,
            builder: BodyBuilder::new(),
            core,
            loop_stack: Vec::new(),
            const_cache: FxHashMap::default(),
            const_array_region_cache: FxHashMap::default(),
            source_info_cache: FxHashMap::default(),
            pat_address_space: FxHashMap::default(),
            binding_locals: FxHashMap::default(),
            address_taken_locals: FxHashSet::default(),
            expr_lower_states: vec![ExprLowerState::NotStarted; body.exprs(db).len()],
            receiver_space,
            effect_param_spaces: Vec::new(),
            effect_binding_spaces: FxHashMap::default(),
            effect_param_provider_tys: Vec::new(),
            param_capability_space_overrides: param_capability_space_overrides.to_vec(),
            deferred_error: None,
        };

        builder.effect_param_spaces = builder.compute_effect_param_spaces();
        for (idx, space) in effect_param_space_overrides.iter().enumerate() {
            if idx < builder.effect_param_spaces.len()
                && let Some(space) = *space
            {
                builder.effect_param_spaces[idx] = space;
            }
        }
        builder.seed_signature_locals();

        Ok(builder)
    }

    fn source_info_for_span(&mut self, span: Option<Span>) -> crate::ir::SourceInfoId {
        let Some(span) = span else {
            return crate::ir::SourceInfoId::SYNTHETIC;
        };
        if let Some(&id) = self.source_info_cache.get(&span) {
            return id;
        }
        let id = self.builder.body.alloc_source_info(Some(span.clone()));
        self.source_info_cache.insert(span, id);
        id
    }

    pub(super) fn expr_lower_state(&self, expr: ExprId) -> ExprLowerState {
        self.expr_lower_states[expr.index()]
    }

    pub(super) fn set_expr_lower_state(&mut self, expr: ExprId, state: ExprLowerState) {
        self.expr_lower_states[expr.index()] = state;
    }

    pub(super) fn set_local_address_space(&mut self, local: LocalId, space: AddressSpaceKind) {
        let local_data = &mut self.builder.body.locals[local.index()];
        crate::repr::set_declared_local_address_space(self.db, &self.core, local_data, space);
    }

    fn source_for_expr(&mut self, expr: ExprId) -> crate::ir::SourceInfoId {
        self.source_info_for_span(expr.span(self.body).resolve(self.db))
    }

    fn source_for_stmt(&mut self, stmt: StmtId) -> crate::ir::SourceInfoId {
        self.source_info_for_span(stmt.span(self.body).resolve(self.db))
    }

    fn source_for_pat(&mut self, pat: PatId) -> crate::ir::SourceInfoId {
        self.source_info_for_span(pat.span(self.body).resolve(self.db))
    }

    fn source_for_func_param(&mut self, func: Func<'db>, idx: usize) -> crate::ir::SourceInfoId {
        let span = func
            .params(self.db)
            .nth(idx)
            .and_then(|param| param.span().resolve(self.db));
        self.source_info_for_span(span)
    }

    fn new_for_func(
        db: &'db dyn SpannedHirAnalysisDb,
        func: Func<'db>,
        body: Body<'db>,
        typed_body: &'a TypedBody<'db>,
        generic_args: &'a [TyId<'db>],
        overrides: LoweringOverrides<'a, 'db>,
    ) -> Result<Self, MirLowerError> {
        let declared_return_ty = CallableDef::Func(func).ret_ty(db);
        let return_ty = hir::analysis::ty::normalize::normalize_ty(
            db,
            if generic_args.is_empty() {
                declared_return_ty.instantiate_identity()
            } else {
                declared_return_ty.instantiate(db, generic_args)
            },
            crate::ty::normalization_scope_for_args(db, func, generic_args),
            hir::analysis::ty::trait_resolution::PredicateListId::empty_list(db),
        );
        Self::new(
            db,
            Some(func),
            body,
            typed_body,
            generic_args,
            return_ty,
            overrides.receiver_space,
            overrides.effect_param_space_overrides,
            overrides.param_capability_space_overrides,
        )
    }

    fn new_for_body_owner(
        db: &'db dyn SpannedHirAnalysisDb,
        body: Body<'db>,
        typed_body: &'a TypedBody<'db>,
        generic_args: &'a [TyId<'db>],
        return_ty: TyId<'db>,
    ) -> Result<Self, MirLowerError> {
        Self::new(
            db,
            None,
            body,
            typed_body,
            generic_args,
            return_ty,
            None,
            &[],
            &[],
        )
    }

    fn seed_synthetic_param_local(
        &mut self,
        name: String,
        ty: TyId<'db>,
        is_mut: bool,
        binding: Option<LocalBinding<'db>>,
    ) -> LocalId {
        let address_space = AddressSpaceKind::Memory;
        let pointer_leaf_infos =
            pointer_leaf_infos_for_ty_with_default(self.db, &self.core, ty, address_space);
        let local = self.builder.body.alloc_local(LocalData {
            name,
            ty,
            is_mut,
            source: crate::ir::SourceInfoId::SYNTHETIC,
            address_space,
            pointer_leaf_infos,
            place_root_layout: crate::repr::declared_local_place_root_layout(
                self.db,
                &self.core,
                ty,
                AddressSpaceKind::Memory,
            ),
            const_backing: self.param_const_backing(ty, is_mut, address_space),
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        self.builder.body.param_locals.push(local);
        if let Some(binding) = binding {
            self.binding_locals.insert(binding, local);
        }
        local
    }

    fn seed_synthetic_effect_param_local(
        &mut self,
        name: String,
        binding: LocalBinding<'db>,
        address_space: AddressSpaceKind,
        provider_ty: Option<TyId<'db>>,
    ) -> LocalId {
        let ty = self.u256_ty();
        let pointer_leaf_infos = provider_ty
            .and_then(|provider_ty| {
                crate::repr::runtime_pointer_info_for_ty(
                    self.db,
                    &self.core,
                    provider_ty,
                    address_space,
                )
            })
            .map(|info| vec![(MirProjectionPath::new(), info)])
            .unwrap_or_default();
        let local = self.builder.body.alloc_local(LocalData {
            name,
            ty,
            is_mut: binding.is_mut(),
            source: crate::ir::SourceInfoId::SYNTHETIC,
            address_space,
            pointer_leaf_infos,
            place_root_layout: crate::repr::declared_local_place_root_layout(
                self.db,
                &self.core,
                ty,
                address_space,
            ),
            const_backing: crate::ir::LocalConstBacking::Runtime,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        self.builder.body.effect_param_locals.push(local);
        self.binding_locals.insert(binding, local);
        self.effect_binding_spaces.insert(binding, address_space);
        self.effect_param_provider_tys.push(provider_ty);
        local
    }

    fn param_const_backing(
        &self,
        ty: TyId<'db>,
        is_mut: bool,
        address_space: AddressSpaceKind,
    ) -> crate::ir::LocalConstBacking {
        if !is_mut && address_space == AddressSpaceKind::Code && self.ty_uses_const_ref_runtime(ty)
        {
            crate::ir::LocalConstBacking::Const
        } else {
            crate::ir::LocalConstBacking::Runtime
        }
    }

    fn compute_effect_param_spaces(&self) -> Vec<AddressSpaceKind> {
        let Some(func) = self.hir_func else {
            return Vec::new();
        };
        let provider_arg_idx_by_effect =
            hir::analysis::ty::effects::place_effect_provider_param_index_map(self.db, func);

        let mut spaces = vec![AddressSpaceKind::Storage; func.effect_params(self.db).count()];
        for effect in func.effect_params(self.db) {
            let inferred = self.infer_effect_provider_for_effect_param(
                func,
                effect.index(),
                provider_arg_idx_by_effect,
            );
            let _rationale = inferred.rationale;
            spaces[effect.index()] = inferred.address_space;
        }

        spaces
    }

    fn infer_effect_provider_from_provider_ty(
        &self,
        provider_ty: TyId<'db>,
        concrete_rationale: EffectProviderInferenceRationale,
    ) -> Option<InferredEffectProvider<'db>> {
        self.infer_effect_provider_from_provider_ty_in_space(provider_ty, concrete_rationale, None)
    }

    fn infer_effect_provider_from_provider_ty_in_space(
        &self,
        provider_ty: TyId<'db>,
        concrete_rationale: EffectProviderInferenceRationale,
        by_ref_space: Option<AddressSpaceKind>,
    ) -> Option<InferredEffectProvider<'db>> {
        if let Some(space) = self.raw_effect_space_for_provider_ty(provider_ty) {
            return Some(InferredEffectProvider {
                provider_ty: Some(provider_ty),
                address_space: space,
                rationale: concrete_rationale,
            });
        }

        if let Some(space) = self.effect_provider_space_for_provider_ty(provider_ty) {
            return Some(InferredEffectProvider {
                provider_ty: Some(provider_ty),
                address_space: space,
                rationale: concrete_rationale,
            });
        }

        // By-ref provider values are passed as pointers; default to memory so callers and
        // callees agree on the address space for projections.
        if matches!(
            crate::repr::repr_kind_for_ty(self.db, &self.core, provider_ty),
            crate::repr::ReprKind::Ref
        ) {
            return Some(InferredEffectProvider {
                provider_ty: Some(provider_ty),
                address_space: by_ref_space.unwrap_or(AddressSpaceKind::Memory),
                rationale: by_ref_space
                    .map(|_| concrete_rationale)
                    .unwrap_or(EffectProviderInferenceRationale::ByRefProviderDefaultsToMemory),
            });
        }

        None
    }

    fn raw_effect_space_for_provider_ty(&self, provider_ty: TyId<'db>) -> Option<AddressSpaceKind> {
        let scope = self.body.scope();
        let raw_mem =
            hir::analysis::ty::corelib::resolve_lib_type_path(self.db, scope, "std::evm::RawMem")?;
        if provider_ty == raw_mem {
            return Some(AddressSpaceKind::Memory);
        }

        let raw_storage = hir::analysis::ty::corelib::resolve_lib_type_path(
            self.db,
            scope,
            "std::evm::RawStorage",
        )?;
        (provider_ty == raw_storage).then_some(AddressSpaceKind::Storage)
    }

    fn is_core_dyn_string_ty(&self, ty: TyId<'db>) -> bool {
        let ty = ty
            .as_capability(self.db)
            .map(|(_, inner)| inner)
            .unwrap_or(ty);
        let base = ty.base_ty(self.db);
        let TyData::TyBase(TyBase::Adt(adt)) = base.data(self.db) else {
            return false;
        };
        let adt_ref = adt.adt_ref(self.db);
        let Some(name) = adt_ref.name(self.db) else {
            return false;
        };
        if name.data(self.db) != "DynString" {
            return false;
        }

        base.ingot(self.db)
            .is_some_and(|ingot| ingot.kind(self.db) == IngotKind::Core)
    }

    fn infer_effect_provider_for_effect_param(
        &self,
        func: Func<'db>,
        effect_idx: usize,
        provider_arg_idx_by_effect: &[Option<usize>],
    ) -> InferredEffectProvider<'db> {
        let assumptions = hir::analysis::ty::trait_resolution::PredicateListId::empty_list(self.db);
        if let Some(provider_arg_idx) = provider_arg_idx_by_effect
            .get(effect_idx)
            .copied()
            .flatten()
        {
            let provider_ty = self
                .generic_args
                .get(provider_arg_idx)
                .copied()
                .or_else(|| {
                    CallableDef::Func(func)
                        .params(self.db)
                        .get(provider_arg_idx)
                        .copied()
                });
            if let Some(provider_ty) = provider_ty
                && let Some(inferred) = self.infer_effect_provider_from_provider_ty(
                    provider_ty,
                    EffectProviderInferenceRationale::ConcreteProviderTy,
                )
            {
                return inferred;
            }
        }

        if let Some(effect) = func.effect_params(self.db).nth(effect_idx)
            && let Some(key_path) = effect.key_path(self.db)
            && let Some(provider_ty) =
                hir::analysis::ty::effects::resolve_normalized_type_effect_key(
                    self.db,
                    key_path,
                    func.scope(),
                    hir::analysis::ty::trait_resolution::PredicateListId::empty_list(self.db),
                )
            && let Some(inferred) = self.infer_effect_provider_from_provider_ty(
                provider_ty,
                EffectProviderInferenceRationale::DeclaredEffectKeyProviderTy,
            )
        {
            return inferred;
        }

        if let Some(provider_ty) = func
            .effect_params(self.db)
            .nth(effect_idx)
            .and_then(|effect| effect.key_path(self.db))
            .and_then(
                |key_path| match hir::analysis::name_resolution::path_resolver::resolve_path(
                    self.db,
                    key_path,
                    func.scope(),
                    assumptions,
                    false,
                )
                .ok()?
                {
                    hir::analysis::name_resolution::path_resolver::PathRes::Ty(ty)
                    | hir::analysis::name_resolution::path_resolver::PathRes::TyAlias(_, ty) => {
                        Some(ty)
                    }
                    _ => None,
                },
            )
            && let Some(inferred) = self.infer_effect_provider_from_provider_ty(
                provider_ty,
                EffectProviderInferenceRationale::ConcreteProviderTy,
            )
        {
            return inferred;
        }

        if let Some(provider_ty) =
            self.contract_field_provider_ty_for_effect_site(EffectParamSite::Func(func), effect_idx)
            && let Some(inferred) = self.infer_effect_provider_from_provider_ty_in_space(
                provider_ty,
                EffectProviderInferenceRationale::ContractFieldProviderTy,
                Some(AddressSpaceKind::Storage),
            )
        {
            return inferred;
        }

        InferredEffectProvider {
            provider_ty: None,
            address_space: AddressSpaceKind::Storage,
            rationale: EffectProviderInferenceRationale::StorageDefault,
        }
    }

    fn infer_effect_provider_or_fallback(
        &self,
        provider_ty: TyId<'db>,
        concrete_rationale: EffectProviderInferenceRationale,
        fallback_space: AddressSpaceKind,
        fallback_rationale: EffectProviderInferenceRationale,
    ) -> InferredEffectProvider<'db> {
        self.infer_effect_provider_from_provider_ty(provider_ty, concrete_rationale)
            .unwrap_or(InferredEffectProvider {
                provider_ty: Some(provider_ty),
                address_space: fallback_space,
                rationale: fallback_rationale,
            })
    }

    fn caller_effect_param_provider_ty(
        &self,
        binding: LocalBinding<'db>,
        caller_provider_arg_idx_by_effect: Option<&[Option<usize>]>,
    ) -> Option<TyId<'db>> {
        let LocalBinding::EffectParam { site, idx, .. } = binding else {
            return None;
        };
        let current_func = self.hir_func?;
        let EffectParamSite::Func(binding_func) = site else {
            return None;
        };
        if binding_func != current_func {
            return None;
        }

        let provider_idx = caller_provider_arg_idx_by_effect?
            .get(idx)
            .copied()
            .flatten()?;
        if let Some(concrete) = self.generic_args.get(provider_idx).copied() {
            return Some(concrete);
        }
        CallableDef::Func(current_func)
            .params(self.db)
            .get(provider_idx)
            .copied()
    }

    fn infer_effect_provider_for_resolved_arg(
        &self,
        resolved_arg: &ResolvedEffectArg<'db>,
        caller_provider_arg_idx_by_effect: Option<&[Option<usize>]>,
    ) -> Option<InferredEffectProvider<'db>> {
        if !matches!(resolved_arg.key_kind, EffectKeyKind::Type) {
            return None;
        }
        let target_ty = resolved_arg.instantiated_target_ty?;

        match resolved_arg.pass_mode {
            EffectPassMode::ByTempPlace => {
                let provider_ty = TyId::app(self.db, self.core.mem_ptr_ctor, target_ty);
                Some(self.infer_effect_provider_or_fallback(
                    provider_ty,
                    EffectProviderInferenceRationale::ByTempPlaceMemPtr,
                    AddressSpaceKind::Memory,
                    EffectProviderInferenceRationale::ByTempPlaceMemPtr,
                ))
            }
            EffectPassMode::ByPlace => {
                let EffectArg::Place(place) = &resolved_arg.arg else {
                    return None;
                };
                let PlaceBase::Binding(binding) = place.base;
                match binding {
                    binding @ LocalBinding::EffectParam { .. } => {
                        if let Some(provider_ty) = self.caller_effect_param_provider_ty(
                            binding,
                            caller_provider_arg_idx_by_effect,
                        ) {
                            return Some(
                                self.infer_effect_provider_from_provider_ty_in_space(
                                    provider_ty,
                                    EffectProviderInferenceRationale::ForwardedEffectParamProviderTy,
                                    Some(self.bound_local_root_address_space(&binding)),
                                )
                                .unwrap_or(InferredEffectProvider {
                                    provider_ty: Some(provider_ty),
                                    address_space: AddressSpaceKind::Storage,
                                    rationale: EffectProviderInferenceRationale::StorageDefault,
                                }),
                            );
                        }

                        let provider_ty = TyId::app(self.db, self.core.mem_ptr_ctor, target_ty);
                        Some(self.infer_effect_provider_or_fallback(
                            provider_ty,
                            EffectProviderInferenceRationale::ForwardedEffectParamFallbackMemPtr,
                            AddressSpaceKind::Memory,
                            EffectProviderInferenceRationale::ForwardedEffectParamFallbackMemPtr,
                        ))
                    }
                    LocalBinding::Param {
                        site: ParamSite::EffectField(effect_site),
                        idx,
                        ..
                    } => {
                        if let Some(provider_ty) =
                            self.contract_field_provider_ty_for_effect_site(effect_site, idx)
                        {
                            return Some(
                                self.infer_effect_provider_from_provider_ty_in_space(
                                    provider_ty,
                                    EffectProviderInferenceRationale::ContractFieldProviderTy,
                                    Some(self.bound_local_root_address_space(&binding)),
                                )
                                .unwrap_or(
                                    InferredEffectProvider {
                                        provider_ty: Some(provider_ty),
                                        address_space: AddressSpaceKind::Storage,
                                        rationale: EffectProviderInferenceRationale::StorageDefault,
                                    },
                                ),
                            );
                        }

                        let provider_ty = TyId::app(self.db, self.core.stor_ptr_ctor, target_ty);
                        Some(self.infer_effect_provider_or_fallback(
                            provider_ty,
                            EffectProviderInferenceRationale::ContractFieldFallbackStorPtr,
                            AddressSpaceKind::Storage,
                            EffectProviderInferenceRationale::ContractFieldFallbackStorPtr,
                        ))
                    }
                    _ => {
                        let provider_ty = TyId::app(self.db, self.core.mem_ptr_ctor, target_ty);
                        Some(self.infer_effect_provider_or_fallback(
                            provider_ty,
                            EffectProviderInferenceRationale::DefaultMemPtr,
                            AddressSpaceKind::Memory,
                            EffectProviderInferenceRationale::DefaultMemPtr,
                        ))
                    }
                }
            }
            _ => None,
        }
    }

    pub(super) fn contract_field_provider_ty_for_effect_site(
        &self,
        site: EffectParamSite<'db>,
        idx: usize,
    ) -> Option<TyId<'db>> {
        let contract = match site {
            EffectParamSite::Func(func) => {
                let ItemKind::Contract(contract) = func.scope().parent_item(self.db)? else {
                    return None;
                };
                contract
            }
            EffectParamSite::Contract(contract)
            | EffectParamSite::ContractInit { contract }
            | EffectParamSite::ContractRecvArm { contract, .. } => contract,
        };

        let key_path = match site {
            EffectParamSite::Func(func) => func
                .effect_params(self.db)
                .nth(idx)
                .and_then(|effect| effect.key_path(self.db))?,
            EffectParamSite::Contract(contract) => contract
                .effect_params(self.db)
                .nth(idx)
                .and_then(|effect| effect.key_path(self.db))?,
            EffectParamSite::ContractInit { contract } => contract
                .init(self.db)?
                .effects(self.db)
                .data(self.db)
                .get(idx)?
                .key_path
                .to_opt()?,
            EffectParamSite::ContractRecvArm {
                contract,
                recv_idx,
                arm_idx,
            } => contract
                .recv_arm(self.db, recv_idx as usize, arm_idx as usize)?
                .effects
                .data(self.db)
                .get(idx)?
                .key_path
                .to_opt()?,
        };

        if key_path.len(self.db) != 1 {
            return None;
        }
        let field_name = key_path.ident(self.db).to_opt()?;
        let field = contract.fields(self.db).get(&field_name)?;
        Some(if field.is_provider {
            field.declared_ty
        } else {
            TyId::app(self.db, self.core.stor_ptr_ctor, field.target_ty)
        })
    }

    /// Consumes the builder and returns the accumulated MIR body.
    ///
    /// # Returns
    /// The completed `MirBody`.
    fn finish(self) -> MirBody<'db> {
        let mut body = self.builder.build();
        body.pat_address_space = self.pat_address_space;
        body
    }

    /// Allocates and returns a fresh basic block.
    ///
    /// # Returns
    /// The identifier for the newly created block.
    fn alloc_block(&mut self) -> BasicBlockId {
        self.builder.make_block()
    }

    fn current_block(&self) -> Option<BasicBlockId> {
        self.builder.current_block()
    }

    fn move_to_block(&mut self, block: BasicBlockId) {
        self.builder.move_to_block(block);
    }

    /// Sets the terminator for the specified block.
    ///
    /// # Parameters
    /// - `block`: Target basic block.
    /// - `term`: Terminator to assign.
    fn set_terminator(&mut self, block: BasicBlockId, term: Terminator<'db>) {
        self.builder.set_block_terminator(block, term);
    }

    fn set_current_terminator(&mut self, term: Terminator<'db>) {
        self.builder.terminate_current(term);
    }

    fn goto(&mut self, target: BasicBlockId) {
        self.set_current_terminator(Terminator::Goto {
            source: crate::ir::SourceInfoId::SYNTHETIC,
            target,
        });
    }

    fn branch(&mut self, cond: ValueId, then_bb: BasicBlockId, else_bb: BasicBlockId) {
        let source = self.builder.body.value(cond).source;
        self.set_current_terminator(Terminator::Branch {
            source,
            cond,
            then_bb,
            else_bb,
        });
    }

    fn switch(&mut self, discr: ValueId, targets: Vec<SwitchTarget>, default: BasicBlockId) {
        let source = self.builder.body.value(discr).source;
        self.set_current_terminator(Terminator::Switch {
            source,
            discr,
            targets,
            default,
        });
    }

    pub(super) fn alloc_temp_local(&mut self, ty: TyId<'db>, is_mut: bool, hint: &str) -> LocalId {
        let idx = self.builder.body.locals.len();
        let name = format!("tmp_{hint}{idx}");
        let pointer_leaf_infos = pointer_leaf_infos_for_ty_with_default(
            self.db,
            &self.core,
            ty,
            AddressSpaceKind::Memory,
        );
        self.builder.body.alloc_local(LocalData {
            name,
            ty,
            is_mut,
            source: crate::ir::SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos,
            place_root_layout: crate::repr::declared_local_place_root_layout(
                self.db,
                &self.core,
                ty,
                AddressSpaceKind::Memory,
            ),
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        })
    }

    /// Appends an instruction to the given block.
    ///
    /// # Parameters
    /// - `block`: Block receiving the instruction.
    /// - `inst`: Instruction to append.
    fn push_inst(&mut self, block: BasicBlockId, inst: MirInst<'db>) {
        self.builder.push_inst_in(block, inst);
    }

    fn push_inst_here(&mut self, inst: MirInst<'db>) {
        if let Some(block) = self.current_block() {
            self.push_inst(block, inst);
        }
    }

    fn normalize_pointer_leaf_infos(
        &mut self,
        infos: Vec<(MirProjectionPath<'db>, PointerInfo<'db>)>,
    ) -> Vec<(MirProjectionPath<'db>, PointerInfo<'db>)> {
        match crate::capability_space::normalize_pointer_leaf_info_entries_in_context(
            self.db, infos,
        ) {
            Ok(normalized) => normalized,
            Err(conflict) => {
                self.defer_pointer_info_conflict(conflict);
                Vec::new()
            }
        }
    }

    fn defer_pointer_info_conflict(&mut self, conflict: PointerInfoConflict<'db>) {
        if self.deferred_error.is_some() {
            return;
        }
        let func_name = self
            .hir_func
            .map(|func| func.pretty_print_signature(self.db))
            .unwrap_or_else(|| "<body owner>".to_owned());
        self.deferred_error = Some(MirLowerError::Unsupported {
            func_name,
            message: format!(
                "conflicting pointer metadata for path `{:?}`: `{:?}` vs `{:?}`",
                conflict.path, conflict.existing, conflict.incoming
            ),
        });
    }

    fn pointer_leaf_infos_for_place(
        &mut self,
        place: &Place<'db>,
        target_ty: TyId<'db>,
    ) -> Vec<(MirProjectionPath<'db>, PointerInfo<'db>)> {
        let fallback = |place: &Place<'db>| self.place_address_space(place);
        match crate::repr::try_pointer_leaf_infos_for_place_with_fallback(
            self.db,
            &self.core,
            &self.builder.body.values,
            &self.builder.body.locals,
            place,
            target_ty,
            &fallback,
        ) {
            Ok(infos) => infos,
            Err(conflict) => {
                self.defer_pointer_info_conflict(conflict);
                Vec::new()
            }
        }
    }

    fn pointer_leaf_infos_for_value(
        &mut self,
        value: ValueId,
    ) -> Vec<(MirProjectionPath<'db>, PointerInfo<'db>)> {
        let fallback = |place: &Place<'db>| self.place_address_space(place);
        match crate::repr::try_pointer_leaf_infos_for_value_with_fallback(
            self.db,
            &self.core,
            &self.builder.body.values,
            &self.builder.body.locals,
            value,
            &fallback,
        ) {
            Ok(infos) => infos,
            Err(conflict) => {
                self.defer_pointer_info_conflict(conflict);
                Vec::new()
            }
        }
    }

    fn value_root_capability_space_hint(&self, value: ValueId) -> AddressSpaceKind {
        if let Some((local, projection)) =
            crate::ir::resolve_local_projection_root(&self.builder.body.values, value)
            && let Some(info) = crate::ir::lookup_local_pointer_leaf_info(
                &self.builder.body.locals,
                local,
                &projection,
            )
        {
            return info.address_space;
        }
        self.value_address_space_or_memory_fallback(value)
    }

    fn call_return_param_sources(&self, call: &CallOrigin<'db>) -> Option<Vec<usize>> {
        let CallTargetRef::Hir(hir_target) = call.target.as_ref()? else {
            return None;
        };
        let CallableDef::Func(func) = hir_target.callable_def else {
            return None;
        };
        let (diags, typed_body) = check_func_body(self.db, func);
        if !diags.is_empty() {
            return None;
        }
        let provenance = typed_body.return_provenance(self.db);
        match provenance {
            ReturnProvenance::ForwardedParams(mut indices) => {
                indices.sort_unstable();
                (!indices.is_empty()).then_some(indices)
            }
            ReturnProvenance::Fresh | ReturnProvenance::Unknown => None,
        }
    }

    fn immutable_call_return_param_sources(&self, call: &CallOrigin<'db>) -> Option<Vec<usize>> {
        let Some(CallTargetRef::Hir(hir_target)) = call.target.as_ref() else {
            return None;
        };
        let CallableDef::Func(func) = hir_target.callable_def else {
            return None;
        };
        let indices = self.call_return_param_sources(call)?;
        indices
            .iter()
            .copied()
            .all(|idx| callable_forwarded_param_is_immutable(self.db, func, idx))
            .then_some(indices)
    }

    fn defer_call_return_space_conflict(
        &mut self,
        call: &CallOrigin<'db>,
        existing: AddressSpaceKind,
        incoming: AddressSpaceKind,
    ) {
        if self.deferred_error.is_some() {
            return;
        }
        let func_name = self
            .hir_func
            .map(|func| func.pretty_print_signature(self.db))
            .unwrap_or_else(|| "<body owner>".to_owned());
        let callee = call
            .target
            .as_ref()
            .and_then(|target| match target {
                CallTargetRef::Hir(target) => {
                    if let CallableDef::Func(func) = target.callable_def {
                        return Some(func.pretty_print_signature(self.db));
                    }
                    None
                }
                CallTargetRef::Synthetic(id) => Some(format!("{id:?}")),
            })
            .unwrap_or_else(|| "<call target>".to_owned());
        self.deferred_error = Some(MirLowerError::Unsupported {
            func_name,
            message: format!(
                "call to `{callee}` can return a capability from conflicting address spaces (`{existing:?}` vs `{incoming:?}`)"
            ),
        });
    }

    fn call_return_space_hint_from_args(
        &mut self,
        call: &CallOrigin<'db>,
    ) -> Option<AddressSpaceKind> {
        if let Some(return_param_indices) = self.call_return_param_sources(call) {
            let mut space_hint = None;
            for idx in return_param_indices {
                let Some(arg_value) = call.args.get(idx).copied() else {
                    continue;
                };
                let space = self.value_root_capability_space_hint(arg_value);
                match space_hint {
                    None => space_hint = Some(space),
                    Some(prev) if prev == space => {}
                    Some(prev) => {
                        self.defer_call_return_space_conflict(call, prev, space);
                        return None;
                    }
                }
            }
            if space_hint.is_some() {
                return space_hint;
            }
        }

        None
    }

    fn pointer_leaf_infos_for_rvalue(
        &mut self,
        dest: LocalId,
        rvalue: &Rvalue<'db>,
    ) -> Vec<(MirProjectionPath<'db>, PointerInfo<'db>)> {
        let (dest_ty, dest_address_space) = {
            let dest_local = self.builder.body.local(dest);
            (dest_local.ty, dest_local.address_space)
        };
        match rvalue {
            Rvalue::Value(value) => self.pointer_leaf_infos_for_value(*value),
            Rvalue::Load { place } => self.pointer_leaf_infos_for_place(place, dest_ty),
            Rvalue::Call(call) => {
                let mut infos = pointer_leaf_infos_for_ty_with_default(
                    self.db,
                    &self.core,
                    dest_ty,
                    AddressSpaceKind::Memory,
                );
                infos.extend(call_return_pointer_leaf_infos(
                    self.db,
                    &self.core,
                    &self.builder.body.values,
                    &self.builder.body.locals,
                    call,
                    dest_ty,
                ));
                let declared_provider_space = self.effect_provider_space_for_provider_ty(dest_ty);
                if infos.is_empty()
                    && hir::analysis::ty::ty_is_noesc(self.db, dest_ty)
                    && let Some(info) = crate::repr::pointer_info_for_ty(
                        self.db,
                        &self.core,
                        dest_ty,
                        AddressSpaceKind::Memory,
                    )
                {
                    infos.push((MirProjectionPath::new(), info));
                }
                let call_space_hint = self.call_return_space_hint_from_args(call).or_else(|| {
                    call.receiver_space
                        .filter(|space| !matches!(space, AddressSpaceKind::Memory))
                });
                if declared_provider_space.is_none()
                    && let Some(space) = call_space_hint
                    && !matches!(space, AddressSpaceKind::Memory)
                {
                    if infos.is_empty()
                        && hir::analysis::ty::ty_is_noesc(self.db, dest_ty)
                        && let Some(info) =
                            crate::repr::pointer_info_for_ty(self.db, &self.core, dest_ty, space)
                    {
                        infos.push((MirProjectionPath::new(), info));
                    }
                    for (_, mapped_info) in &mut infos {
                        mapped_info.address_space = space;
                    }
                }
                if infos.is_empty()
                    && let Some(info) = crate::repr::pointer_info_for_ty(
                        self.db,
                        &self.core,
                        dest_ty,
                        dest_address_space,
                    )
                {
                    return vec![(MirProjectionPath::new(), info)];
                }
                infos
            }
            Rvalue::Intrinsic { .. } => {
                crate::repr::pointer_info_for_ty(self.db, &self.core, dest_ty, dest_address_space)
                    .map(|info| vec![(MirProjectionPath::new(), info)])
                    .unwrap_or_default()
            }
            Rvalue::Alloc { address_space } => vec![(
                MirProjectionPath::new(),
                PointerInfo {
                    address_space: *address_space,
                    target_ty: Some(dest_ty),
                },
            )],
            Rvalue::ZeroInit | Rvalue::ConstAggregate { .. } => Vec::new(),
        }
    }

    fn local_type_tracks_address_space(&self, ty: TyId<'db>) -> bool {
        crate::repr::pointer_info_for_ty(self.db, &self.core, ty, AddressSpaceKind::Memory)
            .is_some()
    }

    fn local_address_space_for_rvalue(
        &mut self,
        dest: LocalId,
        rvalue: &Rvalue<'db>,
        infos: &[(MirProjectionPath<'db>, PointerInfo<'db>)],
    ) -> Option<AddressSpaceKind> {
        let dest_ty = self.builder.body.local(dest).ty;
        if !self.local_type_tracks_address_space(dest_ty) {
            return None;
        }

        if let Some(space) = infos
            .iter()
            .find(|(path, _)| path.is_empty())
            .map(|(_, info)| info.address_space)
        {
            return Some(space);
        }

        match rvalue {
            Rvalue::Value(value) => crate::ir::try_value_address_space_in(
                &self.builder.body.values,
                &self.builder.body.locals,
                *value,
            ),
            Rvalue::Load { place } => Some(self.place_address_space(place)),
            Rvalue::Call(call) => self
                .call_return_space_hint_from_args(call)
                .or(call.receiver_space)
                .or(self.effect_provider_space_for_provider_ty(dest_ty)),
            Rvalue::Intrinsic { .. } => crate::repr::runtime_pointer_info_for_ty(
                self.db,
                &self.core,
                dest_ty,
                self.builder.body.local(dest).address_space,
            )
            .map(|info| info.address_space),
            Rvalue::Alloc { address_space } => Some(*address_space),
            Rvalue::ZeroInit | Rvalue::ConstAggregate { .. } => None,
        }
    }

    fn ty_uses_const_ref_runtime(&self, ty: TyId<'db>) -> bool {
        matches!(
            crate::repr::runtime_shape_for_ty(self.db, &self.core, ty, AddressSpaceKind::Code),
            crate::ir::RuntimeShape::ConstRef { .. }
        )
    }

    fn local_can_be_const_backed(&self, dest: LocalId) -> bool {
        let local = self.builder.body.local(dest);
        !crate::repr::local_is_semantically_mutable(self.db, local)
            && self.ty_uses_const_ref_runtime(local.ty)
    }

    fn value_is_const_backed(&self, value: ValueId) -> bool {
        let value_data = self.builder.body.value(value);
        match &value_data.origin {
            ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) => self
                .builder
                .body
                .locals
                .get(local.index())
                .is_some_and(|local| local.const_backing.is_const()),
            ValueOrigin::TransparentCast { value } => self.value_is_const_backed(*value),
            ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
                self.ty_uses_const_ref_runtime(value_data.ty)
                    && (self.value_is_const_backed(place.base)
                        || crate::ir::try_place_pointer_info_in(
                            &self.builder.body.values,
                            &self.builder.body.locals,
                            place,
                        )
                        .map(|info| info.address_space)
                        .or_else(|| {
                            crate::ir::try_place_address_space_in(
                                &self.builder.body.values,
                                &self.builder.body.locals,
                                place,
                            )
                        }) == Some(AddressSpaceKind::Code))
            }
            ValueOrigin::ConstRegion(region) => self
                .builder
                .body
                .const_regions
                .get(region.index())
                .is_some_and(|region| self.ty_uses_const_ref_runtime(region.ty)),
            _ => false,
        }
    }

    fn place_load_is_const_backed(&self, place: &Place<'db>, loaded_ty: TyId<'db>) -> bool {
        if !self.ty_uses_const_ref_runtime(loaded_ty) {
            return false;
        }

        let fallback = |place: &Place<'db>| self.place_address_space(place);
        let loaded_root_space = crate::repr::try_pointer_leaf_infos_for_place_with_fallback(
            self.db,
            &self.core,
            &self.builder.body.values,
            &self.builder.body.locals,
            place,
            loaded_ty,
            &fallback,
        )
        .ok()
        .and_then(|infos| {
            infos
                .into_iter()
                .find(|(path, _)| path.is_empty())
                .map(|(_, info)| info.address_space)
        });

        self.value_is_const_backed(place.base)
            || loaded_root_space == Some(AddressSpaceKind::Code)
            || crate::ir::try_place_pointer_info_in(
                &self.builder.body.values,
                &self.builder.body.locals,
                place,
            )
            .map(|info| info.address_space)
            .or_else(|| {
                crate::ir::try_place_address_space_in(
                    &self.builder.body.values,
                    &self.builder.body.locals,
                    place,
                )
            }) == Some(AddressSpaceKind::Code)
    }

    fn call_result_is_const_backed(&self, call: &CallOrigin<'db>) -> bool {
        let Some(CallTargetRef::Hir(hir_target)) = call.target.as_ref() else {
            return false;
        };
        let CallableDef::Func(_) = hir_target.callable_def else {
            return false;
        };

        self.immutable_call_return_param_sources(call)
            .is_some_and(|indices| {
                indices.iter().copied().all(|idx| {
                    call.args
                        .get(idx)
                        .is_some_and(|arg| self.value_is_const_backed(*arg))
                })
            })
    }

    fn rvalue_is_const_backed(&self, dest: LocalId, rvalue: &Rvalue<'db>) -> bool {
        match rvalue {
            Rvalue::ConstAggregate { ty, .. } => self.ty_uses_const_ref_runtime(*ty),
            Rvalue::Value(value) => self.value_is_const_backed(*value),
            Rvalue::Load { place } => {
                self.place_load_is_const_backed(place, self.builder.body.local(dest).ty)
            }
            Rvalue::Call(call) => self.call_result_is_const_backed(call),
            Rvalue::ZeroInit | Rvalue::Intrinsic { .. } | Rvalue::Alloc { .. } => false,
        }
    }

    fn assign_with_source(
        &mut self,
        source: crate::ir::SourceInfoId,
        dest: Option<LocalId>,
        rvalue: Rvalue<'db>,
    ) {
        if let Some(dest) = dest {
            let infos = self.pointer_leaf_infos_for_rvalue(dest, &rvalue);
            let infos = self.normalize_pointer_leaf_infos(infos);
            let address_space = self.local_address_space_for_rvalue(dest, &rvalue, &infos);
            let const_backed =
                self.local_can_be_const_backed(dest) && self.rvalue_is_const_backed(dest, &rvalue);
            if let Some(address_space) = address_space {
                self.set_local_address_space(dest, address_space);
            }

            {
                let local = &mut self.builder.body.locals[dest.index()];
                local.pointer_leaf_infos = infos;
                if const_backed {
                    if !matches!(local.const_backing, crate::ir::LocalConstBacking::Runtime) {
                        local.const_backing = crate::ir::LocalConstBacking::Const;
                    }
                } else {
                    local.const_backing = crate::ir::LocalConstBacking::Runtime;
                }
            }

            if const_backed {
                if !matches!(
                    self.builder.body.locals[dest.index()].const_backing,
                    crate::ir::LocalConstBacking::Runtime
                ) {
                    self.set_local_address_space(dest, AddressSpaceKind::Code);
                }
            } else {
                let local = &self.builder.body.locals[dest.index()];
                let local_ty = local.ty;
                if local.address_space == AddressSpaceKind::Code
                    && self.ty_uses_const_ref_runtime(local_ty)
                {
                    self.set_local_address_space(dest, AddressSpaceKind::Memory);
                }
            }

            if let Rvalue::Alloc { address_space } = rvalue {
                let ty = self.builder.body.local(dest).ty;
                self.builder.body.locals[dest.index()].place_root_layout =
                    crate::repr::allocated_local_place_root_layout(
                        self.db,
                        &self.core,
                        ty,
                        address_space,
                    );
            }
        }

        self.push_inst_here(MirInst::Assign {
            source,
            dest,
            rvalue,
        });
    }

    fn assign(&mut self, stmt: Option<StmtId>, dest: Option<LocalId>, rvalue: Rvalue<'db>) {
        let source = stmt
            .map(|stmt| self.source_for_stmt(stmt))
            .unwrap_or(crate::ir::SourceInfoId::SYNTHETIC);
        self.assign_with_source(source, dest, rvalue);
    }

    pub(super) fn refresh_value_pointer_info(&mut self, value: ValueId) {
        let info = crate::repr::infer_value_pointer_info(
            self.db,
            &self.core,
            &self.builder.body.values,
            &self.builder.body.locals,
            value,
        );
        self.builder.body.values[value.index()].pointer_info = info;
    }

    fn alloc_value(
        &mut self,
        ty: TyId<'db>,
        mut origin: ValueOrigin<'db>,
        repr: ValueRepr,
    ) -> ValueId {
        if let ValueOrigin::TransparentCast { value } = origin
            && self.builder.body.value(value).ty == ty
            && self.builder.body.value(value).repr == repr
        {
            origin = self.builder.body.value(value).origin.clone();
        }

        let place_pointer_info = match &origin {
            ValueOrigin::PlaceRef(place) => crate::repr::runtime_value_pointer_info_for_ty(
                self.db,
                &self.core,
                ty,
                self.place_address_space(place),
            ),
            ValueOrigin::MoveOut { place } => {
                self.builder
                    .body
                    .place_pointer_info(place)
                    .and_then(|info| {
                        crate::repr::runtime_value_pointer_info_for_ty(
                            self.db,
                            &self.core,
                            ty,
                            info.address_space,
                        )
                    })
            }
            _ => None,
        };
        let default_space = repr
            .address_space()
            .or_else(|| match &origin {
                ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) => {
                    crate::ir::lookup_local_pointer_leaf_info(
                        &self.builder.body.locals,
                        *local,
                        &MirProjectionPath::new(),
                    )
                    .map(|info| info.address_space)
                    .or_else(|| {
                        ty.as_capability(self.db)
                            .is_some()
                            .then_some(self.builder.body.local(*local).address_space)
                    })
                }
                ValueOrigin::TransparentCast { value } => crate::ir::try_value_address_space_in(
                    &self.builder.body.values,
                    &self.builder.body.locals,
                    *value,
                ),
                ValueOrigin::PlaceRef(_) => place_pointer_info.map(|info| info.address_space),
                ValueOrigin::MoveOut { place } => Some(self.place_address_space(place)),
                ValueOrigin::FieldPtr(field_ptr) => Some(field_ptr.addr_space),
                _ => None,
            })
            .unwrap_or(AddressSpaceKind::Memory);
        let pointer_info = place_pointer_info.or_else(|| {
            crate::repr::runtime_value_pointer_info_for_ty(self.db, &self.core, ty, default_space)
        });
        self.builder.body.alloc_value(ValueData {
            ty,
            origin,
            source: crate::ir::SourceInfoId::SYNTHETIC,
            repr,
            pointer_info,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        })
    }

    /// Determines the address space for a binding.
    ///
    /// # Parameters
    /// - `binding`: Binding metadata.
    ///
    /// # Returns
    /// The resolved address space kind.
    pub(super) fn address_space_for_binding(
        &self,
        binding: &LocalBinding<'db>,
    ) -> AddressSpaceKind {
        if let Some(&local) = self.binding_locals.get(binding) {
            return self.builder.body.local(local).address_space;
        }

        match binding {
            LocalBinding::EffectParam { site, idx, .. } => {
                if let Some(space) = self.effect_binding_spaces.get(binding).copied() {
                    return space;
                }
                if let Some(func) = self.hir_func
                    && matches!(site, EffectParamSite::Func(site_func) if *site_func == func)
                {
                    return self
                        .effect_param_spaces
                        .get(*idx)
                        .copied()
                        .unwrap_or(AddressSpaceKind::Storage);
                }
                AddressSpaceKind::Storage
            }
            LocalBinding::Local { pat, .. } => self
                .pat_address_space
                .get(pat)
                .copied()
                .unwrap_or(AddressSpaceKind::Memory),
            LocalBinding::Param { site, idx, .. } => match site {
                ParamSite::Func(_) => {
                    if *idx == 0 {
                        return self.receiver_space.unwrap_or(AddressSpaceKind::Memory);
                    }
                    AddressSpaceKind::Memory
                }
                ParamSite::ContractInit(_) => AddressSpaceKind::Memory,
                ParamSite::EffectField(effect_site) => match effect_site {
                    _ if self.effect_binding_spaces.contains_key(binding) => self
                        .effect_binding_spaces
                        .get(binding)
                        .copied()
                        .unwrap_or(AddressSpaceKind::Storage),
                    EffectParamSite::Func(effect_func)
                        if self.hir_func.is_some_and(|current| current == *effect_func) =>
                    {
                        self.effect_param_spaces
                            .get(*idx)
                            .copied()
                            .unwrap_or(AddressSpaceKind::Storage)
                    }
                    _ => AddressSpaceKind::Storage,
                },
            },
        }
    }

    pub(super) fn bound_local_root_address_space(
        &self,
        binding: &LocalBinding<'db>,
    ) -> AddressSpaceKind {
        self.binding_locals
            .get(binding)
            .and_then(|local| {
                crate::ir::lookup_local_pointer_leaf_info(
                    &self.builder.body.locals,
                    *local,
                    &MirProjectionPath::new(),
                )
                .map(|info| info.address_space)
                .or_else(|| {
                    self.builder
                        .body
                        .locals
                        .get(local.index())
                        .map(|local| local.address_space)
                })
            })
            .unwrap_or_else(|| self.address_space_for_binding(binding))
    }

    /// Computes the address space for an expression, defaulting to memory.
    ///
    /// # Parameters
    /// - `expr`: Expression id to inspect.
    ///
    /// # Returns
    /// The address space kind for the expression.
    pub(super) fn expr_address_space(&self, expr: ExprId) -> AddressSpaceKind {
        let prop = self.typed_body.expr_prop(self.db, expr);
        if let Some(binding) = prop.binding {
            self.bound_local_root_address_space(&binding)
        } else {
            AddressSpaceKind::Memory
        }
    }

    pub(super) fn u256_ty(&self) -> TyId<'db> {
        TyId::new(self.db, TyData::TyBase(TyBase::Prim(PrimTy::U256)))
    }

    /// Returns `true` when the given type is represented by-reference in MIR.
    ///
    /// Fe MIR represents user aggregates (structs/tuples/arrays/enums) as pointers into an address
    /// space. Effect pointer provider newtypes (`MemPtr`/`StorPtr`/`CalldataPtr`) are *not*
    /// represented by-reference: they are single-word values at runtime.
    pub(super) fn is_by_ref_ty(&self, ty: TyId<'db>) -> bool {
        matches!(
            crate::repr::repr_kind_for_ty(self.db, &self.core, ty),
            crate::repr::ReprKind::Ref
        )
    }

    pub(super) fn value_repr_for_expr(&self, expr: ExprId, ty: TyId<'db>) -> ValueRepr {
        if ty.as_capability(self.db).is_some() {
            if matches!(self.builder.body.stage, crate::ir::MirStage::Capability) {
                return ValueRepr::Word;
            }
            let space = self.expr_address_space(expr);
            return match crate::repr::repr_kind_for_ty(self.db, &self.core, ty) {
                crate::repr::ReprKind::Ptr(_) => ValueRepr::Ptr(space),
                crate::repr::ReprKind::Ref => ValueRepr::Ref(space),
                crate::repr::ReprKind::Zst | crate::repr::ReprKind::Word => ValueRepr::Word,
            };
        }

        match crate::repr::repr_kind_for_ty(self.db, &self.core, ty) {
            crate::repr::ReprKind::Ptr(space) => {
                if matches!(space, AddressSpaceKind::Memory)
                    && crate::repr::effect_provider_space_for_ty(self.db, &self.core, ty).is_none()
                    && crate::repr::pointer_info_for_ty(
                        self.db,
                        &self.core,
                        ty,
                        self.expr_address_space(expr),
                    )
                    .is_some()
                {
                    ValueRepr::Ptr(self.expr_address_space(expr))
                } else {
                    ValueRepr::Ptr(space)
                }
            }
            crate::repr::ReprKind::Ref => ValueRepr::Ref(self.expr_address_space(expr)),
            crate::repr::ReprKind::Zst | crate::repr::ReprKind::Word => ValueRepr::Word,
        }
    }

    pub(super) fn value_repr_for_ty(&self, ty: TyId<'db>, space: AddressSpaceKind) -> ValueRepr {
        if ty.as_capability(self.db).is_some() {
            if matches!(self.builder.body.stage, crate::ir::MirStage::Capability) {
                return ValueRepr::Word;
            }
            return match crate::repr::repr_kind_for_ty(self.db, &self.core, ty) {
                crate::repr::ReprKind::Ptr(_) => ValueRepr::Ptr(space),
                crate::repr::ReprKind::Ref => ValueRepr::Ref(space),
                crate::repr::ReprKind::Zst | crate::repr::ReprKind::Word => ValueRepr::Word,
            };
        }

        match crate::repr::repr_kind_for_ty(self.db, &self.core, ty) {
            crate::repr::ReprKind::Ptr(ptr_space) => {
                if matches!(ptr_space, AddressSpaceKind::Memory)
                    && crate::repr::effect_provider_space_for_ty(self.db, &self.core, ty).is_none()
                    && crate::repr::pointer_info_for_ty(self.db, &self.core, ty, space).is_some()
                {
                    ValueRepr::Ptr(space)
                } else {
                    ValueRepr::Ptr(ptr_space)
                }
            }
            crate::repr::ReprKind::Ref => ValueRepr::Ref(space),
            crate::repr::ReprKind::Zst | crate::repr::ReprKind::Word => ValueRepr::Word,
        }
    }

    fn project_tuple_elem_value(
        &mut self,
        tuple_value: ValueId,
        tuple_ty: TyId<'db>,
        field_idx: usize,
        field_ty: TyId<'db>,
        binding_mode: PatBindingMode,
    ) -> ValueId {
        let is_borrow_binding = matches!(binding_mode, PatBindingMode::ByBorrow);
        // Transparent newtype access: field 0 is a representation-preserving cast.
        if !is_borrow_binding
            && field_ty.as_capability(self.db).is_none()
            && crate::repr::transparent_field0_inner_ty(self.db, tuple_ty, field_idx).is_some()
        {
            let base_repr = self.builder.body.value(tuple_value).repr;
            if !base_repr.is_ref() {
                let space = base_repr
                    .address_space()
                    .unwrap_or(AddressSpaceKind::Memory);
                return self.alloc_value(
                    field_ty,
                    ValueOrigin::TransparentCast { value: tuple_value },
                    self.value_repr_for_ty(field_ty, space),
                );
            }
        }

        let place = if self.deref_target_ty(tuple_ty).is_some() {
            let mut place = self
                .place_from_derefable_value(tuple_value, tuple_ty)
                .expect("derefable tuple projection requires an address-backed place");
            place
                .projection
                .push(hir::projection::Projection::Field(field_idx));
            place
        } else {
            Place::new(
                tuple_value,
                MirProjectionPath::from_projection(hir::projection::Projection::Field(field_idx)),
            )
        };
        let base_space = self.place_address_space(&place);
        if is_borrow_binding {
            return self.alloc_value(
                field_ty,
                ValueOrigin::PlaceRef(place),
                self.value_repr_for_ty(field_ty, base_space),
            );
        }
        if self.is_by_ref_ty(field_ty) {
            return self.alloc_value(
                field_ty,
                ValueOrigin::PlaceRef(place),
                ValueRepr::Ref(base_space),
            );
        }
        let dest = self.alloc_temp_local(field_ty, false, "arg");
        self.set_local_address_space(dest, AddressSpaceKind::Memory);
        self.assign(None, Some(dest), Rvalue::Load { place });
        self.alloc_value(field_ty, ValueOrigin::Local(dest), ValueRepr::Word)
    }

    fn bind_pat_value(&mut self, pat: PatId, value: ValueId) {
        let Some(block) = self.current_block() else {
            return;
        };
        let Partial::Present(pat_data) = pat.data(self.db, self.body) else {
            return;
        };

        match pat_data {
            Pat::WildCard | Pat::Rest => {}
            Pat::Path(_, is_mut) => {
                let binding = self
                    .typed_body
                    .pat_binding(pat)
                    .unwrap_or(LocalBinding::Local {
                        pat,
                        is_mut: *is_mut,
                    });
                let Some(local) = self.local_for_binding(binding) else {
                    return;
                };
                self.move_to_block(block);
                self.assign(None, Some(local), Rvalue::Value(value));
                let pat_ty = self.typed_body.pat_ty(self.db, pat);
                let carries_space = self
                    .value_repr_for_ty(pat_ty, AddressSpaceKind::Memory)
                    .address_space()
                    .is_some()
                    || pat_ty.as_capability(self.db).is_some();
                if carries_space {
                    self.set_pat_address_space(pat, self.builder.body.local(local).address_space);
                }
            }
            Pat::Tuple(pats) | Pat::PathTuple(_, pats) => {
                let owner_ty = self.typed_body.pat_ty(self.db, pat);
                let owner_by_ref = self.is_by_ref_ty(owner_ty);
                let tuple_repr = self.builder.body.value(value).repr;
                for (idx, field_pat) in pats.iter().enumerate() {
                    if !owner_by_ref && idx != 0 {
                        continue;
                    }
                    let Partial::Present(field_pat_data) = field_pat.data(self.db, self.body)
                    else {
                        continue;
                    };
                    if matches!(field_pat_data, Pat::WildCard | Pat::Rest) {
                        continue;
                    }
                    let field_ty = self.typed_body.pat_ty(self.db, *field_pat);
                    let binding_mode = self
                        .typed_body
                        .pat_binding_mode(*field_pat)
                        .unwrap_or(PatBindingMode::ByValue);
                    let field_value = if owner_by_ref
                        || matches!(binding_mode, PatBindingMode::ByBorrow)
                    {
                        self.project_tuple_elem_value(value, owner_ty, idx, field_ty, binding_mode)
                    } else {
                        self.alloc_value(
                            field_ty,
                            ValueOrigin::TransparentCast { value },
                            tuple_repr,
                        )
                    };
                    self.bind_pat_value(*field_pat, field_value);
                    if self.current_block().is_none() {
                        break;
                    }
                }
            }
            Pat::Record(_, fields) => {
                let owner_ty = self.typed_body.pat_ty(self.db, pat);
                let owner_by_ref = self.is_by_ref_ty(owner_ty);
                let record_repr = self.builder.body.value(value).repr;
                let base_space = self.value_address_space(value);
                for field in fields {
                    let Some(label) = field.label(self.db, self.body) else {
                        continue;
                    };
                    let Some(info) = self.field_access_info(owner_ty, FieldIndex::Ident(label))
                    else {
                        continue;
                    };
                    if !owner_by_ref && info.field_idx != 0 {
                        continue;
                    }
                    let field_ty = self.typed_body.pat_ty(self.db, field.pat);
                    let binding_mode = self
                        .typed_body
                        .pat_binding_mode(field.pat)
                        .unwrap_or(PatBindingMode::ByValue);
                    let field_value =
                        if owner_by_ref || matches!(binding_mode, PatBindingMode::ByBorrow) {
                            if self
                                .value_repr_for_ty(field_ty, AddressSpaceKind::Memory)
                                .address_space()
                                .is_some()
                            {
                                self.set_pat_address_space(field.pat, base_space);
                            }
                            self.project_tuple_elem_value(
                                value,
                                owner_ty,
                                info.field_idx,
                                field_ty,
                                binding_mode,
                            )
                        } else {
                            self.alloc_value(
                                field_ty,
                                ValueOrigin::TransparentCast { value },
                                record_repr,
                            )
                        };
                    self.bind_pat_value(field.pat, field_value);
                    if self.current_block().is_none() {
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    fn seed_signature_locals(&mut self) {
        let Some(func) = self.hir_func else {
            return;
        };
        for (idx, param) in func.params(self.db).enumerate() {
            let source = self.source_info_for_span(param.span().resolve(self.db));
            let binding = self
                .typed_body
                .param_binding(idx)
                .unwrap_or(LocalBinding::Param {
                    site: ParamSite::Func(func),
                    idx,
                    mode: param.mode(self.db),
                    ty: param.ty(self.db),
                    is_mut: param.is_mut(self.db),
                });
            let name = param
                .name(self.db)
                .map(|ident| ident.data(self.db).to_string())
                .unwrap_or_else(|| format!("arg{idx}"));
            let ty = match binding {
                LocalBinding::Param { ty, .. } => ty,
                _ => param.ty(self.db),
            };
            let mut address_space = self.address_space_for_binding(&binding);
            let mut place_root_layout = crate::repr::declared_local_place_root_layout(
                self.db,
                &self.core,
                ty,
                address_space,
            );
            if address_space == AddressSpaceKind::Memory
                && place_root_layout.is_object_root()
                && (binding.is_mut() || crate::repr::ty_has_mut_capability(self.db, ty))
            {
                place_root_layout = crate::ir::LocalPlaceRootLayout::MemorySlot;
            }
            let mut pointer_leaf_infos =
                pointer_leaf_infos_for_ty_with_default(self.db, &self.core, ty, address_space);
            if let Some(overrides) = self.param_capability_space_overrides.get(idx) {
                let mut local = LocalData {
                    name: String::new(),
                    ty,
                    is_mut: binding.is_mut(),
                    source,
                    address_space,
                    pointer_leaf_infos,
                    place_root_layout,
                    const_backing: self.param_const_backing(ty, binding.is_mut(), address_space),
                    runtime_shape: crate::ir::RuntimeShape::Unresolved,
                };
                for (path, space) in overrides {
                    crate::repr::apply_param_capability_space_override(
                        self.db, &self.core, &mut local, path, *space,
                    );
                }
                address_space = local.address_space;
                place_root_layout = local.place_root_layout;
                pointer_leaf_infos = self.normalize_pointer_leaf_infos(local.pointer_leaf_infos);
            }
            let local = self.builder.body.alloc_local(LocalData {
                name,
                ty,
                is_mut: binding.is_mut(),
                source,
                address_space,
                pointer_leaf_infos,
                place_root_layout,
                const_backing: self.param_const_backing(ty, binding.is_mut(), address_space),
                runtime_shape: crate::ir::RuntimeShape::Unresolved,
            });
            self.builder.body.param_locals.push(local);
            self.binding_locals.insert(binding, local);
        }
        let effects_source = self.source_info_for_span(func.span().effects().resolve(self.db));
        let provider_arg_idx_by_effect =
            hir::analysis::ty::effects::place_effect_provider_param_index_map(self.db, func);
        for effect in func.effect_params(self.db) {
            let idx = effect.index();
            let Some(key_path) = effect.key_path(self.db) else {
                continue;
            };
            let binding = LocalBinding::EffectParam {
                site: EffectParamSite::Func(func),
                idx,
                key_path,
                is_mut: effect.is_mut(self.db),
            };
            let name = effect
                .name(self.db)
                .map(|ident| ident.data(self.db).to_string())
                .or_else(|| {
                    key_path
                        .ident(self.db)
                        .to_opt()
                        .map(|ident| ident.data(self.db).to_string())
                })
                .unwrap_or_else(|| format!("effect{idx}"));
            let address_space = self.address_space_for_binding(&binding);
            let inferred =
                self.infer_effect_provider_for_effect_param(func, idx, provider_arg_idx_by_effect);
            let pointer_leaf_infos = inferred
                .provider_ty
                .and_then(|provider_ty| {
                    crate::repr::runtime_pointer_info_for_ty(
                        self.db,
                        &self.core,
                        provider_ty,
                        address_space,
                    )
                })
                .map(|info| vec![(MirProjectionPath::new(), info)])
                .unwrap_or_default();
            let local = self.builder.body.alloc_local(LocalData {
                name,
                ty: self.u256_ty(),
                is_mut: binding.is_mut(),
                source: effects_source,
                address_space,
                pointer_leaf_infos,
                place_root_layout: crate::repr::declared_local_place_root_layout(
                    self.db,
                    &self.core,
                    self.u256_ty(),
                    address_space,
                ),
                const_backing: crate::ir::LocalConstBacking::Runtime,
                runtime_shape: crate::ir::RuntimeShape::Unresolved,
            });
            self.builder.body.effect_param_locals.push(local);
            self.binding_locals.insert(binding, local);
            self.effect_param_provider_tys.push(inferred.provider_ty);
        }
    }

    pub(super) fn local_for_binding(&mut self, binding: LocalBinding<'db>) -> Option<LocalId> {
        if let Some(&local) = self.binding_locals.get(&binding) {
            return Some(local);
        }
        let needs_effect_param_local = matches!(
            binding,
            LocalBinding::EffectParam {
                site: EffectParamSite::Contract(_)
                    | EffectParamSite::ContractInit { .. }
                    | EffectParamSite::ContractRecvArm { .. },
                ..
            }
        );
        if let LocalBinding::Param {
            site: ParamSite::EffectField(effect_site),
            idx,
            ..
        } = binding
            && let Some(current) = self.hir_func
            && matches!(effect_site, EffectParamSite::Func(func) if func == current)
            && let Some(&local) = self.builder.body.effect_param_locals.get(idx)
        {
            self.binding_locals.insert(binding, local);
            return Some(local);
        }
        let name = self.binding_name(binding)?;
        let (ty, is_mut) = match binding {
            LocalBinding::Local { pat, is_mut } => (self.typed_body.pat_ty(self.db, pat), is_mut),
            LocalBinding::Param { ty, is_mut, .. } => (ty, is_mut),
            LocalBinding::EffectParam { is_mut, .. } => (self.u256_ty(), is_mut),
        };
        let source = match &binding {
            LocalBinding::Local { pat, .. } => self.source_for_pat(*pat),
            LocalBinding::Param {
                site: ParamSite::Func(func),
                idx,
                ..
            } => self.source_for_func_param(*func, *idx),
            _ => crate::ir::SourceInfoId::SYNTHETIC,
        };
        let pointer_leaf_infos = pointer_leaf_infos_for_ty_with_default(
            self.db,
            &self.core,
            ty,
            self.address_space_for_binding(&binding),
        );
        let address_space = self.address_space_for_binding(&binding);
        let local = self.builder.body.alloc_local(LocalData {
            name,
            ty,
            is_mut,
            source,
            address_space,
            pointer_leaf_infos,
            place_root_layout: crate::repr::declared_local_place_root_layout(
                self.db,
                &self.core,
                ty,
                address_space,
            ),
            const_backing: if matches!(binding, LocalBinding::Param { .. }) {
                self.param_const_backing(ty, is_mut, address_space)
            } else {
                crate::ir::LocalConstBacking::Unknown
            },
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        if needs_effect_param_local {
            self.builder.body.effect_param_locals.push(local);
        }
        self.binding_locals.insert(binding, local);
        Some(local)
    }

    pub(super) fn binding_name(&self, binding: LocalBinding<'db>) -> Option<String> {
        match binding {
            LocalBinding::Local { pat, .. } => match pat.data(self.db, self.body) {
                Partial::Present(Pat::Path(path, _)) => path
                    .to_opt()
                    .and_then(|path| path.as_ident(self.db))
                    .map(|ident| ident.data(self.db).to_string()),
                _ => None,
            },
            LocalBinding::Param { site, idx, .. } => match site {
                ParamSite::Func(func) => func
                    .params(self.db)
                    .nth(idx)
                    .and_then(|param| param.name(self.db))
                    .map(|ident| ident.data(self.db).to_string())
                    .or_else(|| Some(format!("arg{idx}"))),
                ParamSite::ContractInit(contract) => contract
                    .init(self.db)?
                    .params(self.db)
                    .data(self.db)
                    .get(idx)
                    .and_then(|param| param.name())
                    .map(|ident| ident.data(self.db).to_string())
                    .or_else(|| Some(format!("arg{idx}"))),
                ParamSite::EffectField(effect_site) => {
                    let name = match effect_site {
                        EffectParamSite::Func(func) => func
                            .effect_params(self.db)
                            .nth(idx)
                            .and_then(|effect| effect.name(self.db)),
                        EffectParamSite::Contract(contract) => contract
                            .effects(self.db)
                            .data(self.db)
                            .get(idx)
                            .and_then(|effect| effect.name),
                        EffectParamSite::ContractInit { contract } => contract
                            .init(self.db)?
                            .effects(self.db)
                            .data(self.db)
                            .get(idx)
                            .and_then(|effect| effect.name),
                        EffectParamSite::ContractRecvArm {
                            contract,
                            recv_idx,
                            arm_idx,
                        } => contract
                            .recv_arm(self.db, recv_idx as usize, arm_idx as usize)?
                            .effects
                            .data(self.db)
                            .get(idx)
                            .and_then(|effect| effect.name),
                    };
                    name.map(|ident| ident.data(self.db).to_string())
                        .or_else(|| Some(format!("effect_field{idx}")))
                }
            },
            LocalBinding::EffectParam {
                site,
                idx,
                key_path,
                ..
            } => {
                let explicit = match site {
                    EffectParamSite::Func(func) => func
                        .effect_params(self.db)
                        .nth(idx)
                        .and_then(|effect| effect.name(self.db))
                        .map(|ident| ident.data(self.db).to_string()),
                    EffectParamSite::Contract(contract) => contract
                        .effects(self.db)
                        .data(self.db)
                        .get(idx)
                        .and_then(|effect| effect.name)
                        .map(|ident| ident.data(self.db).to_string()),
                    EffectParamSite::ContractInit { contract } => contract
                        .init(self.db)?
                        .effects(self.db)
                        .data(self.db)
                        .get(idx)
                        .and_then(|effect| effect.name)
                        .map(|ident| ident.data(self.db).to_string()),
                    EffectParamSite::ContractRecvArm {
                        contract,
                        recv_idx,
                        arm_idx,
                    } => contract
                        .recv_arm(self.db, recv_idx as usize, arm_idx as usize)?
                        .effects
                        .data(self.db)
                        .get(idx)
                        .and_then(|effect| effect.name)
                        .map(|ident| ident.data(self.db).to_string()),
                };
                explicit
                    .or_else(|| {
                        key_path
                            .ident(self.db)
                            .to_opt()
                            .map(|ident| ident.data(self.db).to_string())
                    })
                    .or_else(|| Some(format!("effect{idx}")))
            }
        }
    }

    pub(super) fn binding_value(&mut self, binding: LocalBinding<'db>) -> Option<ValueId> {
        let local = self.local_for_binding(binding)?;
        let value_id = self.alloc_value(self.u256_ty(), ValueOrigin::Local(local), ValueRepr::Word);
        Some(value_id)
    }

    pub(super) fn effect_provider_space_for_provider_ty(
        &self,
        provider_ty: TyId<'db>,
    ) -> Option<AddressSpaceKind> {
        crate::repr::effect_provider_space_for_ty(self.db, &self.core, provider_ty)
    }

    /// Determines the address space associated with a MIR value.
    ///
    /// This is used when lowering projections and effect arguments that need to know whether a
    /// pointer-like value is addressing memory or storage.
    pub(super) fn value_address_space(&self, value: ValueId) -> AddressSpaceKind {
        self.builder.body.value_address_space(value)
    }

    pub(super) fn place_address_space(&self, place: &Place<'db>) -> AddressSpaceKind {
        crate::repr::resolve_place(
            self.db,
            &self.core,
            &self.builder.body.values,
            &self.builder.body.locals,
            place,
        )
        .and_then(|resolved| {
            resolved
                .final_state()
                .pointer_info
                .map(|info| info.address_space)
                .or(resolved.final_state().location_address_space())
        })
        .unwrap_or_else(|| self.builder.body.place_address_space(place))
    }

    pub(super) fn value_address_space_or_memory_fallback(
        &self,
        value: ValueId,
    ) -> AddressSpaceKind {
        crate::ir::try_value_address_space_in(
            &self.builder.body.values,
            &self.builder.body.locals,
            value,
        )
        .unwrap_or(AddressSpaceKind::Memory)
    }

    pub(super) fn value_local_address_space_hint(
        &self,
        value: ValueId,
    ) -> Option<AddressSpaceKind> {
        let mut root = value;
        while let ValueOrigin::TransparentCast { value } = &self.builder.body.value(root).origin {
            root = *value;
        }

        match self.builder.body.value(root).origin {
            ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) => {
                Some(self.builder.body.local(local).address_space)
            }
            _ => None,
        }
    }

    pub(super) fn capability_binding_space_from_container(
        &self,
        container: ValueId,
    ) -> AddressSpaceKind {
        if let Some(space) = self.value_local_address_space_hint(container)
            && space != AddressSpaceKind::Memory
        {
            return space;
        }

        self.value_address_space_or_memory_fallback(container)
    }

    /// Associates a pattern with an address space.
    ///
    /// # Parameters
    /// - `pat`: Pattern id to annotate.
    /// - `space`: Address space kind to record.
    pub(super) fn set_pat_address_space(&mut self, pat: PatId, space: AddressSpaceKind) {
        self.pat_address_space.insert(pat, space);
        let locals_to_update: Vec<LocalId> = self
            .binding_locals
            .iter()
            .filter_map(|(binding, local)| match binding {
                LocalBinding::Local {
                    pat: binding_pat, ..
                } if *binding_pat == pat => Some(*local),
                _ => None,
            })
            .collect();
        for local in locals_to_update {
            self.set_local_address_space(local, space);
        }
    }
}

fn format_hir_expr_context(db: &dyn SpannedHirAnalysisDb, body: Body<'_>, expr: ExprId) -> String {
    let span = expr.span(body).resolve(db);
    let span_context = if let Some(span) = span {
        let path = span
            .file
            .path(db)
            .as_ref()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "<unknown file>".into());
        let start: usize = u32::from(span.range.start()) as usize;
        let text = span.file.text(db);
        let (mut line, mut col) = (1usize, 1usize);
        for byte in text.as_bytes().iter().take(start) {
            if *byte == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        format!("{path}:{line}:{col}")
    } else {
        "<no span>".into()
    };

    let expr_data = match expr.data(db, body) {
        Partial::Present(expr_data) => match expr_data {
            Expr::Path(path) => path
                .to_opt()
                .map(|path| format!("Path({})", path.pretty_print(db)))
                .unwrap_or_else(|| "Path(<absent>)".into()),
            Expr::Call(callee, args) => {
                let callee_data = match callee.data(db, body) {
                    Partial::Present(Expr::Path(path)) => path
                        .to_opt()
                        .map(|path| format!("Path({})", path.pretty_print(db)))
                        .unwrap_or_else(|| "Path(<absent>)".into()),
                    Partial::Present(other) => format!("{other:?}"),
                    Partial::Absent => "<absent>".into(),
                };
                format!("Call({callee:?} {callee_data}, {args:?})")
            }
            Expr::MethodCall(receiver, method, _, args) => {
                let method_name = method
                    .to_opt()
                    .map(|id| id.data(db).to_string())
                    .unwrap_or_else(|| "<absent>".into());
                format!("MethodCall({receiver:?}, {method_name}, {args:?})")
            }
            other => format!("{other:?}"),
        },
        Partial::Absent => "<absent>".into(),
    };

    format!("expr={expr:?} at {span_context}: {expr_data}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MirLoweringInvariantViolation {
    UnloweredExpr(ExprId),
    ControlFlowResult(ExprId),
}

impl MirLoweringInvariantViolation {
    fn expr(self) -> ExprId {
        match self {
            Self::UnloweredExpr(expr) | Self::ControlFlowResult(expr) => expr,
        }
    }
}

pub(super) fn validate_lowered_mir_body<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    func_name: &str,
    body: Body<'db>,
    mir_body: &MirBody<'db>,
) -> MirLowerResult<()> {
    let Some(violation) = first_lowering_invariant_violation_used_by_mir(mir_body) else {
        return Ok(());
    };
    let expr_context = format_hir_expr_context(db, body, violation.expr());
    Err(MirLowerError::UnloweredHirExpr {
        func_name: func_name.to_string(),
        expr: expr_context,
    })
}

fn validate_lowered_mir_functions<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    functions: &[MirFunction<'db>],
) -> MirLowerResult<()> {
    for func in functions {
        let Some(typed_body) = &func.typed_body else {
            continue;
        };
        let Some(body) = typed_body.body() else {
            continue;
        };
        validate_lowered_mir_body(db, &mir_func_name(db, func), body, &func.body)?;
    }
    Ok(())
}

fn first_lowering_invariant_violation_used_by_mir<'db>(
    body: &MirBody<'db>,
) -> Option<MirLoweringInvariantViolation> {
    let mut used_values: FxHashSet<ValueId> = FxHashSet::default();

    for block in &body.blocks {
        for inst in &block.insts {
            match inst {
                MirInst::Assign { rvalue, .. } => match rvalue {
                    crate::ir::Rvalue::ZeroInit => {}
                    crate::ir::Rvalue::Value(value) => {
                        used_values.insert(*value);
                    }
                    crate::ir::Rvalue::Call(call) => {
                        used_values.extend(call.args.iter().copied());
                        used_values.extend(call.effect_args.iter().copied());
                    }
                    crate::ir::Rvalue::Intrinsic { args, .. } => {
                        used_values.extend(args.iter().copied());
                    }
                    crate::ir::Rvalue::Load { place } => {
                        used_values.insert(place.base);
                        used_values.extend(dynamic_indices(&place.projection));
                    }
                    crate::ir::Rvalue::Alloc { .. } => {}
                    crate::ir::Rvalue::ConstAggregate { .. } => {}
                },
                MirInst::BindValue { value, .. } => {
                    used_values.insert(*value);
                }
                MirInst::Store { place, value, .. } => {
                    used_values.insert(place.base);
                    used_values.insert(*value);
                    used_values.extend(dynamic_indices(&place.projection));
                }
                MirInst::InitAggregate { place, inits, .. } => {
                    used_values.insert(place.base);
                    used_values.extend(dynamic_indices(&place.projection));
                    for (path, value) in inits {
                        used_values.extend(dynamic_indices(path));
                        used_values.insert(*value);
                    }
                }
                MirInst::SetDiscriminant { place, .. } => {
                    used_values.insert(place.base);
                    used_values.extend(dynamic_indices(&place.projection));
                }
            }
        }

        match &block.terminator {
            Terminator::Return {
                value: Some(value), ..
            } => {
                used_values.insert(*value);
            }
            Terminator::TerminatingCall { call, .. } => match call {
                crate::ir::TerminatingCall::Call(call) => {
                    used_values.extend(call.args.iter().copied());
                    used_values.extend(call.effect_args.iter().copied());
                }
                crate::ir::TerminatingCall::Intrinsic { args, .. } => {
                    used_values.extend(args.iter().copied());
                }
            },
            Terminator::Branch { cond, .. } => {
                used_values.insert(*cond);
            }
            Terminator::Switch { discr, .. } => {
                used_values.insert(*discr);
            }
            Terminator::Return { value: None, .. }
            | Terminator::Goto { .. }
            | Terminator::Unreachable { .. } => {}
        }
    }

    let mut worklist: Vec<ValueId> = used_values.into_iter().collect();
    let mut visited: FxHashSet<ValueId> = FxHashSet::default();

    while let Some(value_id) = worklist.pop() {
        if !visited.insert(value_id) {
            continue;
        }

        match &body.value(value_id).origin {
            ValueOrigin::Expr(expr) => {
                return Some(MirLoweringInvariantViolation::UnloweredExpr(*expr));
            }
            ValueOrigin::ControlFlowResult { expr } => {
                return Some(MirLoweringInvariantViolation::ControlFlowResult(*expr));
            }
            ValueOrigin::Unary { inner, .. } => worklist.push(*inner),
            ValueOrigin::Binary { lhs, rhs, .. } => {
                worklist.push(*lhs);
                worklist.push(*rhs);
            }
            ValueOrigin::TransparentCast { value } => worklist.push(*value),
            _ => {}
        }
    }

    None
}

fn dynamic_indices<'db, 'a>(
    path: &'a MirProjectionPath<'db>,
) -> impl Iterator<Item = ValueId> + 'a {
    path.iter().filter_map(|proj| match proj {
        hir::projection::Projection::Index(hir::projection::IndexSource::Dynamic(value_id)) => {
            Some(*value_id)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use cranelift_entity::EntityRef;
    use driver::DriverDataBase;
    use hir::analysis::ty::ty_def::TyId;

    use super::*;
    use crate::ir::{BodyBuilder, Rvalue, ValueOrigin, ValueRepr};

    #[test]
    fn lowering_invariant_detects_live_control_flow_results() {
        let db = DriverDataBase::default();
        let ty = TyId::unit(&db);
        let expr = ExprId::new(0);
        let mut builder = BodyBuilder::new();
        let value =
            builder.alloc_value(ty, ValueOrigin::ControlFlowResult { expr }, ValueRepr::Word);
        builder.assign(None, Rvalue::Value(value));
        let body = builder.build();

        assert_eq!(
            first_lowering_invariant_violation_used_by_mir(&body),
            Some(MirLoweringInvariantViolation::ControlFlowResult(expr))
        );
    }

    #[test]
    fn lowering_invariant_ignores_dead_placeholders() {
        let db = DriverDataBase::default();
        let ty = TyId::unit(&db);
        let expr = ExprId::new(0);
        let mut builder = BodyBuilder::new();
        let _ = builder.alloc_value(ty, ValueOrigin::Expr(expr), ValueRepr::Word);
        let body = builder.build();

        assert_eq!(first_lowering_invariant_violation_used_by_mir(&body), None);
    }
}

use crate::{
    HirDb,
    hir_def::{ItemKind, TopLevelMod},
    span::DynLazySpan,
};
use common::diagnostics::CompleteDiagnostic;
pub mod analysis_pass;
pub mod core_requirements;
pub mod diagnostics;
pub mod place;
pub mod semantic;

use self::analysis_pass::{
    AbiStructLowerPass, AnalysisPassManager, AttrMisusePass, ErrorLowerPass, EventLowerPass,
    MsgLowerPass, ParsingPass,
};
use self::diagnostics::SpannedHirAnalysisDb;
use self::name_resolution::ImportAnalysisPass;
use self::ty::{
    AdtDefAnalysisPass, BodyAnalysisPass, ContractAnalysisPass, DefConflictAnalysisPass,
    FuncAnalysisPass, ImplAnalysisPass, ImplTraitAnalysisPass, MsgAnalysisPass, TraitAnalysisPass,
    TypeAliasAnalysisPass,
    diagnostics::{TyDiagCollection, TyLowerDiag},
    normalize::NormalizationLimitReached,
    ty_check::check_func_body,
};

#[salsa::db]
pub trait HirAnalysisDb: HirDb {}

#[salsa::db]
impl<T> HirAnalysisDb for T where T: HirDb {}

pub mod name_resolution;
pub mod ty;

pub(crate) fn initialize_pre_contract_analysis_pass() -> AnalysisPassManager {
    let mut pass_manager = AnalysisPassManager::new();
    pass_manager.add_module_pass("Parsing", Box::new(ParsingPass {}));
    pass_manager.add_module_pass("AttrMisuse", Box::new(AttrMisusePass {}));
    pass_manager.add_module_pass("MsgLower", Box::new(MsgLowerPass {}));
    pass_manager.add_module_pass("EventLower", Box::new(EventLowerPass {}));
    pass_manager.add_module_pass("ErrorLower", Box::new(ErrorLowerPass {}));
    pass_manager.add_module_pass("AbiStructLower", Box::new(AbiStructLowerPass {}));
    pass_manager.add_module_pass("MsgAnalysis", Box::new(MsgAnalysisPass {}));
    pass_manager.add_module_pass("DefConflict", Box::new(DefConflictAnalysisPass {}));
    pass_manager.add_module_pass("Import", Box::new(ImportAnalysisPass {}));
    pass_manager.add_module_pass("AdtDef", Box::new(AdtDefAnalysisPass {}));
    pass_manager.add_module_pass("TypeAlias", Box::new(TypeAliasAnalysisPass {}));
    pass_manager.add_module_pass("Trait", Box::new(TraitAnalysisPass {}));
    pass_manager.add_module_pass("Impl", Box::new(ImplAnalysisPass {}));
    pass_manager.add_module_pass("ImplTrait", Box::new(ImplTraitAnalysisPass {}));
    pass_manager.add_module_pass("Func", Box::new(FuncAnalysisPass {}));
    pass_manager.add_module_pass("Body", Box::new(BodyAnalysisPass {}));
    pass_manager
}

pub fn initialize_analysis_pass() -> AnalysisPassManager {
    let mut pass_manager = initialize_pre_contract_analysis_pass();
    pass_manager.add_module_pass("Contract", Box::new(ContractAnalysisPass {}));
    pass_manager
}

/// The diagnostics of every analysis pass on `top_mod`, as one tracked
/// computation, so that [`analyze_module`] can collect what the passes
/// accumulate.
#[salsa::tracked(return_ref)]
fn module_diagnostics<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Vec<CompleteDiagnostic> {
    initialize_analysis_pass()
        .run_on_module(db, top_mod)
        .iter()
        .map(|diag| diag.to_complete(db))
        .collect()
}

/// The diagnostics of every analysis pass on `top_mod`.
///
/// A normalization that reaches a limit leaves a type that matches any type,
/// so the program must not be accepted: a limit reached anywhere in the
/// analysis is an error. Where no type that shows it was reported, the error is
/// given at the function whose body reached it, or else at the module's first
/// item.
pub fn analyze_module<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    top_mod: TopLevelMod<'db>,
) -> Vec<CompleteDiagnostic> {
    let mut diags = module_diagnostics(db, top_mod).clone();
    let limit = |span: DynLazySpan<'db>, reached: &NormalizationLimitReached| {
        TyDiagCollection::from(TyLowerDiag::TypeNormalizationLimitReached {
            span,
            projection: reached.projection.clone(),
        })
        .to_voucher()
        .to_complete(db)
    };
    let Some(reached) = module_diagnostics::accumulated::<NormalizationLimitReached>(db, top_mod)
        .first()
        .copied()
    else {
        return diags;
    };
    let limit_code = limit(top_mod.span().into(), reached).error_code;
    if diags.iter().any(|diag| diag.error_code == limit_code) {
        return diags;
    }
    let in_bodies: Vec<_> = top_mod
        .all_funcs(db)
        .iter()
        .filter_map(|&func| {
            let reached = check_func_body::accumulated::<NormalizationLimitReached>(db, func)
                .first()
                .copied()?;
            Some(limit(func.span().name().into(), reached))
        })
        .collect();
    if in_bodies.is_empty() {
        let span = top_mod
            .all_items(db)
            .iter()
            .filter(|item| !matches!(item, ItemKind::TopMod(_)))
            .find_map(|item| item.name_span())
            .unwrap_or_else(|| top_mod.span().into());
        diags.push(limit(span, reached));
    } else {
        diags.extend(in_bodies);
    }
    diags
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<'db, T> {
    pub data: T,
    pub span: DynLazySpan<'db>,
}

impl<'db, T> Spanned<'db, T> {
    pub fn new(data: T, span: DynLazySpan<'db>) -> Self {
        Self { data, span }
    }
}

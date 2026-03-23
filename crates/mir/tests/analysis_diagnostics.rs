use common::InputDb;
use driver::DriverDataBase;
use fe_mir::{MirDiagnosticsMode, MirLowerError, collect_mir_diagnostics, lower_module};
use url::Url;

#[test]
fn lower_module_reports_analysis_diagnostics_as_error() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///analysis_diagnostics.fe").unwrap();
    let src = r#"
pub fn mismatched_ret() -> bool {
    1
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);

    let err = lower_module(&db, top_mod).expect_err("analysis diagnostics should fail lowering");

    let MirLowerError::AnalysisDiagnostics {
        func_name,
        diagnostics,
    } = err
    else {
        panic!("expected AnalysisDiagnostics, got {err:?}");
    };

    assert!(!func_name.is_empty(), "func name is empty");
    assert!(diagnostics.contains("type mismatch"));
}

#[test]
fn collect_mir_diagnostics_short_circuits_on_invalid_hir() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///analysis_diagnostics_in_monomorphization.fe").unwrap();
    let src = r#"
fn foo(x: u8) -> u256 {
    x
}

#[test]
fn bar_test() {
    foo(42)
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);

    let output = collect_mir_diagnostics(&db, top_mod, MirDiagnosticsMode::CompilerParity);
    assert!(
        output.internal_errors.is_empty(),
        "expected invalid HIR to skip MIR entirely, got: {:?}",
        output.internal_errors
    );
    assert!(output.diagnostics.is_empty());
}

#[test]
fn lower_module_rejects_unsupported_const_path_match_patterns() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///unsupported_const_path_pattern.fe").unwrap();
    let src = r#"
const FOO: String<3> = "foo"

pub fn test(x: String<3>) -> u8 {
    match x {
        FOO => 0
        _ => 1
    }
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);

    let err = lower_module(&db, top_mod).expect_err("unsupported const-path pattern should fail");

    let MirLowerError::Unsupported { message, .. } = err else {
        panic!("expected Unsupported, got {err:?}");
    };
    assert!(
        message.contains("pattern"),
        "expected unsupported-pattern error, got `{message}`",
    );
}

#[test]
fn lower_module_rejects_unsupported_string_literal_match_patterns() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///unsupported_string_literal_pattern.fe").unwrap();
    let src = r#"
pub fn test(x: String<3>) -> u8 {
    match x {
        "foo" => 0
        _ => 1
    }
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);

    let err =
        lower_module(&db, top_mod).expect_err("unsupported string literal pattern should fail");

    let MirLowerError::Unsupported { message, .. } = err else {
        panic!("expected Unsupported, got {err:?}");
    };
    assert!(
        message.contains("pattern"),
        "expected unsupported-pattern error, got `{message}`",
    );
}

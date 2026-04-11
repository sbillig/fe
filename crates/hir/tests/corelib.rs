use common::InputDb;
use common::stdlib::{HasBuiltinCore, HasBuiltinStd};
use driver::{DriverDataBase, db::DiagnosticsCollection};
use fe_hir::analysis::ty::ty_check::ReturnProvenance;
use fe_hir::analysis::ty::{
    corelib::{resolve_core_trait, resolve_lib_func_path, resolve_lib_type_path},
    trait_resolution::{GoalSatisfiability, TraitSolveCx, is_goal_satisfiable},
    ty_check::check_func_body,
};
use fe_hir::hir_def::{Expr, LitKind, Partial};
use fe_hir::test_db::HirAnalysisTestDb;
use url::Url;

#[cfg(target_arch = "wasm32")]
use test_utils::url_utils::UrlExt;

#[test]
fn analyze_corelib() {
    let db = DriverDataBase::default();
    let core = db.builtin_core();
    assert!(
        core.files(&db).iter().next().is_some(),
        "builtin core ingot should not be empty"
    );

    let core_diags = db.run_on_ingot(core);
    assert_builtin_clean(&db, core_diags, "core");
}

#[test]
fn analyze_stdlib() {
    let db = DriverDataBase::default();
    let std_ingot = db.builtin_std();
    assert!(
        std_ingot.files(&db).iter().next().is_some(),
        "builtin std ingot should not be empty"
    );

    let std_diags = db.run_on_ingot(std_ingot);
    assert_builtin_clean(&db, std_diags, "std");
}

fn assert_builtin_clean(db: &DriverDataBase, diags: DiagnosticsCollection<'_>, name: &str) {
    if diags.is_empty() {
        return;
    }

    diags.emit(db);
    panic!(
        "expected no diagnostics for builtin {name}, but got:\n{}",
        diags.format_diags(db)
    );
}

#[test]
fn solver_proves_encode_for_fixed_bool_arrays() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///encode_fixed_bool_array_goal.fe").unwrap();
    let src = r#"
use core::abi::Encode
use std::abi::Sol

pub fn root(values: [bool; 5]) {
    values.encode_to_ptr(0)
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    assert!(
        diags.is_empty(),
        "unexpected diagnostics: {}",
        diags.format_diags(&db)
    );

    let func = top_mod.all_funcs(&db)[0];
    let body = func.body(&db).expect("root should have a body");
    let typed_body = &check_func_body(&db, func).1;
    let callable = body
        .exprs(&db)
        .keys()
        .find_map(|expr| {
            let callable = typed_body.callable_expr(expr)?;
            let name = callable.callable_def.name(&db)?;
            (name.data(&db) == "encode_to_ptr").then_some(callable)
        })
        .expect("encode_to_ptr callable should resolve during type checking");
    let inst = callable
        .trait_inst()
        .expect("encode_to_ptr should be a trait method");
    let solve_cx = TraitSolveCx::new(&db, func.scope());

    match is_goal_satisfiable(&db, solve_cx, inst) {
        GoalSatisfiability::Satisfied(_) => {}
        other => panic!(
            "expected `{}` to be satisfiable, got {other:?}",
            inst.pretty_print(&db, true)
        ),
    }
}

#[test]
fn string_literals_can_pick_up_dynstring_api_from_later_use() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///string_dynstring_api.fe").unwrap();
    let src = r#"
pub fn root() -> u8 {
    let text = "hello-dynamic-api"
    text.view().byte_at(0)
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    assert!(
        diags.is_empty(),
        "unexpected diagnostics: {}",
        diags.format_diags(&db)
    );

    let func = top_mod.all_funcs(&db)[0];
    let body = func.body(&db).expect("root should have a body");
    let typed_body = &check_func_body(&db, func).1;
    let literal_expr = body
        .exprs(&db)
        .keys()
        .find(|expr| {
            matches!(
                expr.data(&db, body),
                Partial::Present(Expr::Lit(LitKind::String(_)))
            )
        })
        .expect("string literal should be present");

    assert_eq!(
        typed_body.expr_ty(&db, literal_expr).pretty_print(&db),
        "DynString"
    );
}

#[test]
fn runtime_string_literals_can_infer_dynstring_from_return_type() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///string_runtime_return_dynstring.fe").unwrap();
    let src = r#"
use std::abi::Text

pub fn root() -> Text {
    let text = "hello-runtime-return"
    text
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    assert!(
        diags.is_empty(),
        "unexpected diagnostics: {}",
        diags.format_diags(&db)
    );

    let func = top_mod.all_funcs(&db)[0];
    let body = func.body(&db).expect("root should have a body");
    let typed_body = &check_func_body(&db, func).1;
    let literal_expr = body
        .exprs(&db)
        .keys()
        .find(|expr| {
            matches!(
                expr.data(&db, body),
                Partial::Present(Expr::Lit(LitKind::String(_)))
            )
        })
        .expect("string literal should be present");

    assert_eq!(
        typed_body.expr_ty(&db, literal_expr).pretty_print(&db),
        "DynString"
    );
}

#[test]
fn const_fn_plus1_body_stays_usize_in_isolation() {
    let mut db = HirAnalysisTestDb::default();
    let url = Url::parse("file:///const_fn_plus1_body_stays_usize_in_isolation.fe").unwrap();
    let src = r#"
const fn plus1(_ x: usize) -> usize {
    x + 1
}
"#;

    let file = db.new_stand_alone(url.path().trim_start_matches('/').into(), src);
    let (top_mod, _) = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    assert!(
        diags.is_empty(),
        "unexpected diagnostics: {}",
        fe_hir::test_db::format_diagnostics(&db, &diags)
    );

    let func = top_mod.all_funcs(&db)[0];
    assert!(
        resolve_core_trait(&db, func.scope(), &["ops", "Add"]).is_some(),
        "failed to resolve core::ops::Add"
    );
    let typed_body = &check_func_body(&db, func).1;
    let body = func.body(&db).expect("plus1 should have a body");

    assert_eq!(typed_body.result_ty().pretty_print(&db), "usize");
    assert_eq!(
        typed_body.expr_ty(&db, body.expr(&db)).pretty_print(&db),
        "usize"
    );
}

#[test]
fn standalone_root_exported_core_and_std_helpers_resolve() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "standalone_root_exported_core_and_std_helpers_resolve.fe".into(),
        "fn f() {}",
    );
    let (top_mod, _) = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    assert!(
        diags.is_empty(),
        "unexpected diagnostics: {}",
        fe_hir::test_db::format_diagnostics(&db, &diags)
    );

    let func = top_mod.all_funcs(&db)[0];
    assert!(resolve_core_trait(&db, func.scope(), &["EffectRef"]).is_some());
    assert!(resolve_lib_type_path(&db, func.scope(), "core::range::Range").is_some());
    assert!(resolve_lib_func_path(&db, func.scope(), "core::panic").is_some());
    assert!(resolve_lib_type_path(&db, func.scope(), "std::abi::Sol").is_some());
}

#[test]
fn runtime_sol_selector_still_accepts_string_literals() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///runtime_sol_selector.fe").unwrap();
    let src = r#"
use std::abi::sol

pub fn root() -> u32 {
    let signature = "ping()"
    let selector = sol(signature)
    selector
}
"#;

    let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
    let top_mod = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    assert!(
        diags.is_empty(),
        "unexpected diagnostics: {}",
        diags.format_diags(&db)
    );
}

#[test]
fn long_sol_signature_literals_remain_compile_time_only() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "long_sol_signature_literals_remain_compile_time_only.fe".into(),
        r#"use std::abi::sol

const SELECTOR: u32 = sol("transferFrom(address,address,uint256)")

pub fn selector() -> u32 {
    SELECTOR
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn implicit_ref_load_returns_are_not_treated_as_forwarded_params() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "implicit_ref_load_returns_are_not_treated_as_forwarded_params.fe".into(),
        r#"fn read_balance(x: ref u256) -> u256 {
    x
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let funcs = top_mod.all_funcs(&db);
    let [func] = funcs.as_slice() else {
        panic!("expected exactly one function");
    };

    let typed_body = check_func_body(&db, *func).1.clone();
    assert_eq!(typed_body.return_provenance(&db), ReturnProvenance::Fresh);
}

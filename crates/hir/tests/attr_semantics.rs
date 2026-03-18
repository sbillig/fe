use fe_hir::hir_def::{InlineAttrErrorKind, Partial, Stmt};
use fe_hir::test_db::HirAnalysisTestDb;

#[test]
fn expr_valued_inline_attr_stays_invalid() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "inline_attr_expr_value.fe".into(),
        r#"#[inline = 1 + 2]
fn bad() {}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let funcs = top_mod.all_funcs(&db);
    let [func] = funcs.as_slice() else {
        panic!("expected exactly one function");
    };

    assert_eq!(func.inline_hint(&db), None);
    assert_eq!(
        func.inline_attr_error(&db),
        Some(InlineAttrErrorKind::InvalidForm)
    );
}

#[test]
fn expr_valued_loop_unroll_attr_does_not_apply_hint() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "loop_unroll_expr_value.fe".into(),
        r#"fn bad(xs: [u256; 1]) {
    #[unroll = 1 + 2]
    for x in xs {}
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let funcs = top_mod.all_funcs(&db);
    let [func] = funcs.as_slice() else {
        panic!("expected exactly one function");
    };
    let body = func.body(&db).expect("expected function body");

    let unroll_hint = body.stmts(&db).iter().find_map(|(_, stmt)| match stmt {
        Partial::Present(Stmt::For(_, _, _, hint)) => Some(*hint),
        Partial::Present(_) | Partial::Absent => None,
    });

    assert_eq!(unroll_hint, Some(None));
}

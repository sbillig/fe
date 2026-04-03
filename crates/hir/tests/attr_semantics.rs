use fe_hir::test_db::HirAnalysisTestDb;
use fe_hir::{
    analysis::{
        semantic::{
            GenericSubst, ImplEnv, SExpr, SStmtKind, SemanticCodeRegionRef, SemanticInstanceKey,
            get_or_build_semantic_instance,
        },
        ty::ty_check::{BodyOwner, check_func_body},
    },
    hir_def::{InlineAttrErrorKind, ManualContractRootAttr, Partial, Stmt},
};

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
fn empty_inline_attr_parens_stay_invalid() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "inline_attr_empty_parens.fe".into(),
        r#"#[inline()]
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

#[test]
fn empty_loop_unroll_attr_parens_do_not_apply_hint() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "loop_unroll_empty_parens.fe".into(),
        r#"fn bad(xs: [u256; 1]) {
    #[unroll()]
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

#[test]
fn unroll_never_attr_prevents_unrolling() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "loop_unroll_never.fe".into(),
        r#"fn bad(xs: [u256; 1]) {
    #[unroll(never)]
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

    assert_eq!(unroll_hint, Some(Some(false)));
}

#[test]
fn empty_payable_attr_parens_do_not_mark_init_payable() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "payable_init_empty_parens.fe".into(),
        r#"contract C {
    #[payable()]
    init() {}
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let contracts = top_mod.all_contracts(&db);
    let [contract] = contracts.as_slice() else {
        panic!("expected exactly one contract");
    };
    let init = contract.init(&db).expect("expected contract init");

    assert!(!init.is_payable(&db));
}

#[test]
fn invalid_payable_init_attr_does_not_mark_init_payable() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "payable_init_invalid.fe".into(),
        r#"contract C {
    #[payable = 1]
    init() {}
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let contracts = top_mod.all_contracts(&db);
    let [contract] = contracts.as_slice() else {
        panic!("expected exactly one contract");
    };
    let init = contract.init(&db).expect("expected contract init");

    assert!(!init.is_payable(&db));
}

#[test]
fn malformed_payable_does_not_mask_valid_init_payable_attr() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "payable_init_mixed_validity.fe".into(),
        r#"contract C {
    #[payable(foo)]
    #[payable]
    init() {}
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let contracts = top_mod.all_contracts(&db);
    let [contract] = contracts.as_slice() else {
        panic!("expected exactly one contract");
    };
    let init = contract.init(&db).expect("expected contract init");

    assert!(init.is_payable(&db));
}

#[test]
fn invalid_payable_recv_arm_attr_does_not_mark_arm_payable() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "payable_recv_arm_invalid.fe".into(),
        r#"use std::abi::sol

msg M {
    #[selector = sol("ping()")]
    Ping,
}

contract C {
    recv M {
        #[payable(foo)]
        Ping {} {}
    }
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let contracts = top_mod.all_contracts(&db);
    let [contract] = contracts.as_slice() else {
        panic!("expected exactly one contract");
    };
    let arm = contract
        .recv_arm(&db, 0, 0)
        .expect("expected one recv arm at 0:0");

    assert!(!arm.is_payable(&db));
}

#[test]
fn manual_contract_root_attr_parses_valid_forms() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "manual_contract_root_attr.fe".into(),
        r#"use std::evm::Evm

#[contract_init(Coin)]
fn init() uses (evm: mut Evm) {}

#[contract_runtime(Coin)]
fn runtime() uses (evm: mut Evm) {}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let funcs = top_mod.all_funcs(&db);
    let init = funcs
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "init")
        })
        .expect("missing init");
    let runtime = funcs
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "runtime")
        })
        .expect("missing runtime");

    assert_eq!(
        init.manual_contract_root_attr(&db),
        Some(ManualContractRootAttr::Init {
            contract_name: fe_hir::hir_def::StringId::new(&db, "Coin".to_string()),
        })
    );
    assert_eq!(
        runtime.manual_contract_root_attr(&db),
        Some(ManualContractRootAttr::Runtime {
            contract_name: fe_hir::hir_def::StringId::new(&db, "Coin".to_string()),
        })
    );
}

#[test]
fn code_region_intrinsics_lower_to_semantic_code_region_exprs() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "manual_contract_code_region.fe".into(),
        r#"use std::evm::Evm

#[contract_init(Coin)]
fn init() uses (evm: mut Evm) {
    let len = evm.code_region_len(runtime)
    let off = evm.code_region_offset(runtime)
}

#[contract_runtime(Coin)]
fn runtime() uses (evm: mut Evm) {}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let funcs = top_mod.all_funcs(&db);
    let init = funcs
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "init")
        })
        .expect("missing init");
    let runtime = funcs
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "runtime")
        })
        .expect("missing runtime");

    let (diags, typed_body) = check_func_body(&db, init);
    assert!(diags.is_empty(), "{diags:#?}");
    let body = init.body(&db).expect("missing init body");
    let refs = body
        .exprs(&db)
        .keys()
        .filter_map(|expr| typed_body.code_region_ref(expr))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(refs.len(), 2);
    assert!(refs.iter().all(|region| matches!(
        region,
        SemanticCodeRegionRef::ManualContractRoot { func } if *func == runtime
    )));

    let instance = get_or_build_semantic_instance(
        &db,
        SemanticInstanceKey::new(
            &db,
            BodyOwner::Func(init),
            GenericSubst::empty(&db),
            ImplEnv::empty(&db, init.scope()),
        ),
    );
    let body = instance.body(&db);
    let exprs = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .filter_map(|stmt| match &stmt.kind {
            SStmtKind::Assign { expr, .. } => Some(expr),
            SStmtKind::Store { .. } => None,
        })
        .collect::<Vec<_>>();
    assert!(exprs.iter().any(|expr| matches!(
        expr,
        SExpr::CodeRegionLen { region: SemanticCodeRegionRef::ManualContractRoot { func } }
            if *func == runtime
    )));
    assert!(exprs.iter().any(|expr| matches!(
        expr,
        SExpr::CodeRegionOffset { region: SemanticCodeRegionRef::ManualContractRoot { func } }
            if *func == runtime
    )));
}

#[test]
fn direct_code_region_intrinsics_record_semantic_refs() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "manual_contract_code_region_direct.fe".into(),
        r#"use std::evm::Evm
use std::evm::intrinsic::{code_region_len, code_region_offset}

#[contract_init(Coin)]
fn init() uses (evm: mut Evm) {
    let len = code_region_len(runtime)
    let off = code_region_offset(runtime)
}

#[contract_runtime(Coin)]
fn runtime() uses (evm: mut Evm) {}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let funcs = top_mod.all_funcs(&db);
    let init = funcs
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "init")
        })
        .expect("missing init");
    let runtime = funcs
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "runtime")
        })
        .expect("missing runtime");

    let (diags, typed_body) = check_func_body(&db, init);
    assert!(diags.is_empty(), "{diags:#?}");
    let body = init.body(&db).expect("missing init body");
    let refs = body
        .exprs(&db)
        .keys()
        .filter_map(|expr| typed_body.code_region_ref(expr))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(refs.len(), 2);
    assert!(refs.iter().all(|region| matches!(
        region,
        SemanticCodeRegionRef::ManualContractRoot { func } if *func == runtime
    )));
}

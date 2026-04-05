use common::InputDb;
use driver::{DriverDataBase, MirDiagnosticsMode};
use hir::{
    analysis::semantic::{
        check_semantic_borrows, get_or_build_semantic_instance, identity_semantic_instance_key,
        normalize_semantic_body,
    },
    analysis::ty::ty_check::BodyOwner,
    hir_def::ItemKind,
};
use std::path::PathBuf;
use url::Url;

#[test]
fn code_region_fixture_has_no_semantic_borrow_diagnostics() {
    let mut db = DriverDataBase::default();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/code_region.fe");
    let file_url = Url::from_file_path(&fixture).expect("fixture path should be absolute");
    let content = std::fs::read_to_string(&fixture).expect("fixture should load");
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(content));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let diags = db.mir_diagnostics_for_top_mod(top_mod, MirDiagnosticsMode::CompilerParity);
    assert!(
        diags
            .iter()
            .all(|diag| !diag.message.contains("move conflict")),
        "{diags:#?}"
    );
    assert!(
        diags
            .iter()
            .all(|diag| !diag.message.contains("internal borrow checking error")),
        "{diags:#?}"
    );
}

#[test]
fn code_region_allocate_instance_has_no_semantic_borrow_diagnostics() {
    let mut db = DriverDataBase::default();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/code_region.fe");
    let file_url = Url::from_file_path(&fixture).expect("fixture path should be absolute");
    let content = std::fs::read_to_string(&fixture).expect("fixture should load");
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(content));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let allocate = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "allocate") =>
            {
                Some(func)
            }
            _ => None,
        })
        .expect("allocate fixture function");
    let key = identity_semantic_instance_key(&db, BodyOwner::Func(*allocate));
    let instance = get_or_build_semantic_instance(&db, key);
    if let Err(diag) = check_semantic_borrows(&db, instance) {
        panic!("{diag:#?}");
    }
}

#[test]
fn code_region_specialized_allocate_callee_has_no_semantic_borrow_diagnostics() {
    let mut db = DriverDataBase::default();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/code_region.fe");
    let file_url = Url::from_file_path(&fixture).expect("fixture path should be absolute");
    let content = std::fs::read_to_string(&fixture).expect("fixture should load");
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(content));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let init = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "init") =>
            {
                Some(func)
            }
            _ => None,
        })
        .expect("init fixture function");
    let init_instance = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(*init)),
    );
    let allocate = init_instance
        .callees(&db)
        .iter()
        .find_map(|callee| match callee.key.owner(&db) {
            BodyOwner::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "allocate") =>
            {
                Some(get_or_build_semantic_instance(&db, callee.key))
            }
            _ => None,
        })
        .expect("specialized allocate callee");
    let _ = normalize_semantic_body(&db, allocate).expect("normalized body");
    if let Err(diag) = check_semantic_borrows(&db, allocate) {
        panic!("{diag:#?}");
    }
}

#[test]
fn erc20_low_level_fixture_has_no_semantic_borrow_diagnostics() {
    let mut db = DriverDataBase::default();
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/erc20_low_level.fe");
    let file_url = Url::from_file_path(&fixture).expect("fixture path should be absolute");
    let content = std::fs::read_to_string(&fixture).expect("fixture should load");
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(content));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let diags = db.mir_diagnostics_for_top_mod(top_mod, MirDiagnosticsMode::CompilerParity);
    assert!(
        diags
            .iter()
            .all(|diag| !diag.message.contains("move conflict")),
        "{diags:#?}"
    );
    assert!(
        diags
            .iter()
            .all(|diag| !diag.message.contains("internal borrow checking error")),
        "{diags:#?}"
    );
}

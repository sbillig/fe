use common::InputDb;
use driver::DriverDataBase;
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
    let diags = db.mir_diagnostics_for_top_mod(top_mod);
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
    let diags = db.mir_diagnostics_for_top_mod(top_mod);
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

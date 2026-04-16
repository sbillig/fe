use common::InputDb;
use dir_test::{Fixture, dir_test};
use driver::DriverDataBase;
use fe_codegen::emit_module_yul;
use test_utils::snap_test;
use url::Url;

// NOTE: `dir_test` discovers fixtures at compile time; new fixture files will be picked up on a
// clean build (e.g. CI) or whenever this test target is recompiled.
#[dir_test(
    dir: "$CARGO_MANIFEST_DIR/tests/fixtures",
    glob: "*.fe"
)]
fn yul_snap(fixture: Fixture<&str>) {
    let mut db = DriverDataBase::default();
    let file_url = Url::from_file_path(fixture.path()).expect("fixture path should be absolute");
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(fixture.content().to_string()),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);

    let output = emit_module_yul(&db, top_mod).expect("Yul emission should succeed");

    snap_test!(output, fixture.path());
}

fn emit_inline_yul(path: &str, src: &str) -> String {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(path).expect("fixture path should be valid");
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(src.to_owned()));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    emit_module_yul(&db, top_mod).expect("Yul emission should succeed")
}

#[test]
fn narrow_signed_named_locals_are_canonicalized_on_yul_reads() {
    let yul = emit_inline_yul(
        "file:///narrow_signed_named_locals_are_canonicalized_on_yul_reads.fe",
        r#"
fn checked_add() -> i8 {
    let a: i8 = -100
    let b: i8 = -100
    let c: i8 = a + b
    return c
}

fn signed_lt() -> bool {
    let a: i8 = -1
    let b: i8 = 1
    return a < b
}
"#,
    );

    assert!(
        yul.contains("add(signextend(0, and("),
        "checked signed add should canonicalize named narrow signed operands before arithmetic:\n{yul}"
    );
    assert!(
        yul.contains("slt(signextend(0, and("),
        "signed comparisons should canonicalize named narrow signed operands before comparison:\n{yul}"
    );
}

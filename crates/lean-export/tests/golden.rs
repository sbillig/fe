use std::{env, fs, path::PathBuf};

use fe_lean_export::{Unsupported, export_contract, export_contract_with_runtime};
use hir::{analysis::HirAnalysisDb, hir_def::Contract, test_db::HirAnalysisTestDb};

type Export = for<'db> fn(&'db dyn HirAnalysisDb, Contract<'db>) -> Result<String, Unsupported>;

/// Exports the one contract of `tests/fixtures/<name>.fe` with `export` and
/// compares it with `tests/golden/<name>.lean`. `LEAN_EXPORT_BLESS=1`
/// rewrites the golden file.
fn check_with(name: &str, export: Export) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let source = fs::read_to_string(dir.join(format!("fixtures/{name}.fe"))).unwrap();
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(format!("{name}.fe").into(), &source);
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let [contract] = top_mod.all_contracts(&db).as_slice() else {
        panic!("expected one contract");
    };
    let lean = export(&db, *contract).expect("exported");
    let golden = dir.join(format!("golden/{name}.lean"));
    if env::var_os("LEAN_EXPORT_BLESS").is_some() {
        fs::write(&golden, &lean).unwrap();
    }
    assert_eq!(lean, fs::read_to_string(&golden).unwrap());
}

fn check(name: &str) {
    check_with(name, export_contract);
}

#[test]
fn scalar_store() {
    check("scalar_store");
}

#[test]
fn erc20() {
    check("erc20");
}

#[test]
fn packed_store() {
    check_with("packed_store", export_contract_with_runtime);
}

#[test]
fn helper_call() {
    check("helper_call");
}

#[test]
fn join() {
    check("join");
}

#[test]
fn local_struct() {
    check("local_struct");
}

#[test]
fn tuple() {
    check("tuple");
}

#[test]
fn param_slot() {
    check("param_slot");
}

#[test]
fn decode() {
    check("decode");
}

#[test]
fn encode() {
    check("encode");
}

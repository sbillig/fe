use std::{env, fs, path::PathBuf};

use fe_lean_export::{Export, Library, Unsupported, export_contract, export_contract_with_entries};
use hir::{analysis::HirAnalysisDb, hir_def::Contract, test_db::HirAnalysisTestDb};

type Exporter =
    for<'db> fn(&'db dyn HirAnalysisDb, Contract<'db>, &mut Library) -> Result<Export, Unsupported>;

/// The one contract of `tests/fixtures/<name>.fe` as `export` gives it
/// with `library`.
fn exported(name: &str, export: Exporter, library: &mut Library) -> Export {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let source = fs::read_to_string(dir.join(format!("fixtures/{name}.fe"))).unwrap();
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(format!("{name}.fe").into(), &source);
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let [contract] = top_mod.all_contracts(&db).as_slice() else {
        panic!("expected one contract");
    };
    export(&db, *contract, library).expect("exported")
}

/// Compares `lean` with `tests/golden/<name>.lean`. `LEAN_EXPORT_BLESS=1`
/// rewrites the golden file.
fn compare(name: &str, lean: &str) {
    let golden =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/golden/{name}.lean"));
    if env::var_os("LEAN_EXPORT_BLESS").is_some() {
        fs::write(&golden, lean).unwrap();
    }
    assert_eq!(lean, fs::read_to_string(&golden).unwrap());
}

/// Checks the fixture `name` exported alone: the library's definitions,
/// then its own.
fn check_with(name: &str, export: Exporter) {
    let mut library = Library::default();
    let own = exported(name, export, &mut library);
    compare(name, &format!("{}{}", library.text(), own.text));
}

fn check(name: &str) {
    check_with(name, export_contract);
}

/// The contracts with bytecode certificates share one library,
/// `golden/lib.lean`. Each golden file has the contract's own definitions,
/// after the names they call the library's by, for a library in a namespace
/// `Lib`.
#[test]
fn certified() {
    let mut library = Library::default();
    for name in ["scalar_store", "code_scalar", "packed_store", "erc20"] {
        let own = exported(name, export_contract_with_entries, &mut library);
        compare(name, &format!("{}{}", own.uses_command("Lib"), own.text));
    }
    compare("lib", library.text());
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

#[test]
fn ints() {
    check("ints");
}

#[test]
fn enums() {
    check("enums");
}

#[test]
fn loops() {
    check("loops");
}

#[test]
fn arrays() {
    check("arrays");
}

#[test]
fn state() {
    check("state");
}

#[test]
fn entries() {
    check("entries");
}

#[test]
fn wide() {
    check_with("wide", export_contract_with_entries);
}

#[test]
fn calldata_read() {
    check("calldata_read");
}

#[test]
fn immutables() {
    check_with("immutables", export_contract_with_entries);
}

#[test]
fn memory_access() {
    check("memory_access");
}

#[test]
fn projections() {
    check("projections");
}

#[test]
fn products() {
    check("products");
}

#[test]
fn drops() {
    check("drops");
}

#[test]
fn mutual() {
    check("mutual");
}

#[test]
fn closures() {
    check("closures");
}

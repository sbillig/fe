use std::{env, fs, path::PathBuf};

use fe_lean_export::{Library, export_contract, export_contract_with_entries};
use hir::test_db::HirAnalysisTestDb;

/// The fixtures in the order they are exported in, and whether with their
/// entries. The contracts with bytecode certificates come first.
const FIXTURES: &[(&str, bool)] = &[
    ("scalar_store", true),
    ("code_scalar", true),
    ("packed_store", true),
    ("erc20", true),
    ("helper_call", false),
    ("join", false),
    ("local_struct", false),
    ("tuple", false),
    ("param_slot", false),
    ("decode", false),
    ("encode", false),
    ("ints", false),
    ("enums", false),
    ("loops", false),
    ("arrays", false),
    ("state", false),
    ("entries", false),
    ("wide", true),
    ("calldata_read", false),
    ("immutables", true),
    ("memory_access", false),
    ("projections", false),
    ("products", false),
    ("drops", false),
    ("mutual", false),
    ("closures", false),
];

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

/// The one contract of each `tests/fixtures/<name>.fe`, exported with one
/// library, `golden/lib.lean`. Each golden file has the contract's own
/// definitions, after the names they call the library's by, for a library
/// in a namespace `Lib`.
#[test]
fn golden() {
    let mut library = Library::default();
    for &(name, entries) in FIXTURES {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.fe"));
        let source = fs::read_to_string(fixture).unwrap();
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(format!("{name}.fe").into(), &source);
        let (top_mod, _) = db.top_mod(file);
        db.assert_no_diags(top_mod);
        let [contract] = top_mod.all_contracts(&db).as_slice() else {
            panic!("expected one contract in {name}");
        };
        let export = if entries {
            export_contract_with_entries
        } else {
            export_contract
        };
        let own = export(&db, *contract, &mut library).expect("exported");
        compare(name, &format!("{}{}", own.uses_command("Lib"), own.text));
    }
    compare("lib", library.text());
}

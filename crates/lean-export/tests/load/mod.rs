use std::{env, fs, path::PathBuf};

use common::InputDb;
use driver::{DriverDataBase, init_ingot};
use hir::{
    analysis::HirAnalysisDb,
    hir_def::{Contract, HirIngot},
    test_db::HirAnalysisTestDb,
};
use url::Url;

/// The fixtures in the order they are exported in, and whether with their
/// entries. The contracts with bytecode certificates come first, and WETH9,
/// certified after the others, last.
pub const FIXTURES: &[(&str, bool)] = &[
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
    ("weth9", true),
    ("trust", true),
    ("ecrecover", true),
];

/// `f` of the last contract of `tests/fixtures/<name>.fe`, after those it
/// deploys, or of the one contract of the root module of the ingot
/// `tests/fixtures/<name>/`.
pub fn with_fixture<T>(
    name: &str,
    f: impl for<'db> FnOnce(&'db dyn HirAnalysisDb, Contract<'db>) -> T,
) -> T {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}"));
    if fixture.is_dir() {
        let mut db = DriverDataBase::default();
        let url = Url::from_directory_path(&fixture).unwrap();
        assert!(!init_ingot(&mut db, &url), "{name} failed to load");
        let ingot = db.workspace().containing_ingot(&db, url).unwrap();
        assert!(!db.run_on_ingot(ingot).has_errors(&db), "{name} has errors");
        let [contract] = ingot.root_mod(&db).all_contracts(&db).as_slice() else {
            panic!("expected one contract in the root module of {name}");
        };
        f(&db, *contract)
    } else {
        let source = fs::read_to_string(fixture.with_extension("fe")).unwrap();
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(format!("{name}.fe").into(), &source);
        let (top_mod, _) = db.top_mod(file);
        db.assert_no_diags(top_mod);
        let contract = *top_mod.all_contracts(&db).last().unwrap();
        f(&db, contract)
    }
}

/// Compares `text` with `tests/golden/<file>`. `LEAN_EXPORT_BLESS=1`
/// rewrites the golden file.
pub fn compare(file: &str, text: &str) {
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/golden/{file}"));
    if env::var_os("LEAN_EXPORT_BLESS").is_some() {
        fs::write(&golden, text).unwrap();
    }
    assert_eq!(text, fs::read_to_string(&golden).unwrap());
}

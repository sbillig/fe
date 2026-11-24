mod load;

use std::{env, fs};

use fe_lean_export::{Export, IngotExport, Library};
use load::{FIXTURES, compare, golden_path, with_fixture};

/// The last contract of each `tests/fixtures/<name>.fe`, after those it
/// deploys, or the contracts of the root module of the ingot
/// `tests/fixtures/<name>/`, exported with one library, `golden/lib.lean`.
/// The golden file of a contract has its own definitions, after the names
/// they call the library's by, for a library in a namespace `Lib`, and the
/// shared definitions of its ingot by, for those in a namespace `Shared`;
/// `.trust` has what they take on trust, and `.checks` the checks of their
/// constants, if they define any, and is absent otherwise. A fixture with one
/// contract has its files, `<name>.lean`; one with several has the shared
/// definitions in those and each contract's in `<name>_<contract>.lean`.
#[test]
fn golden() {
    let mut library = Library::default();
    let golden = |file: &str, export: &Export| {
        compare(
            &format!("{file}.lean"),
            &format!("{}{}", export.uses_command("Lib", "Shared"), export.text),
        );
        compare(&format!("{file}.trust"), &export.trust_report());
        let checks = format!("{file}.checks");
        if !export.checks.is_empty() {
            compare(&checks, &export.checks);
        } else {
            // A part whose constants all moved to the shared part keeps no
            // checks file.
            let golden = golden_path(&checks);
            if env::var_os("LEAN_EXPORT_BLESS").is_some() && golden.exists() {
                fs::remove_file(&golden).unwrap();
            }
            assert!(
                !golden.exists(),
                "golden/{checks} is left from checks now empty"
            );
        }
    };
    for &(name, entries) in FIXTURES {
        let (contracts, exports) = with_fixture(name, |db, contracts| {
            let mut ingot = IngotExport::new(db);
            for &contract in contracts {
                ingot
                    .export_contract(contract, entries, &mut library)
                    .expect("exported");
            }
            let names: Vec<_> = contracts
                .iter()
                .map(|contract| contract.name(db).unwrap().data(db).to_lowercase())
                .collect();
            (names, ingot.finish(&library))
        });
        if let [export] = exports.contracts.as_slice() {
            assert!(exports.shared.text.is_empty(), "{name} shares definitions");
            golden(name, export);
        } else {
            golden(name, &exports.shared);
            for (contract, export) in contracts.iter().zip(&exports.contracts) {
                golden(&format!("{name}_{contract}"), export);
            }
        }
    }
    compare("lib.lean", library.text());
    compare("lib.checks", library.checks());
}

/// A contract of a program the compiler rejects is not exported:
/// `refutable_let.fe` binds a constant's name with `let`, a refutable
/// pattern.
#[test]
fn rejects_programs_with_errors() {
    let mut library = Library::default();
    let error = with_fixture("refutable_let", |db, contracts| {
        IngotExport::new(db)
            .export_contract(contracts[0], true, &mut library)
            .unwrap_err()
    });
    assert!(error.0.contains("refutable pattern"), "{}", error.0);
}

/// A type with no values, or an enum with a variant that has none, is not
/// exported: each type the export defines is inhabited. `uninhabited.fe`
/// matches on a `Result` whose `Err` holds an enum with no variants.
#[test]
fn rejects_uninhabited_types() {
    let mut library = Library::default();
    let error = with_fixture("uninhabited", |db, contracts| {
        IngotExport::new(db)
            .export_contract(contracts[0], true, &mut library)
            .unwrap_err()
    });
    assert!(error.0.contains("uninhabited"), "{}", error.0);
}

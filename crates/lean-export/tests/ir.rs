mod load;

use std::fmt::Write;

use fe_lean_export::{export_ir, ir_text};
use load::{FIXTURES, compare, with_fixture};

/// The normalized IR of the last contract of `tests/fixtures/<name>.fe`, as
/// `golden/ir/<file>.json` has it, for the last component `file` of `name`.
fn golden(name: &str) {
    let json = with_fixture(name, |db, contracts| {
        export_ir(db, *contracts.last().unwrap()).expect("serialized")
    });
    let file = name.rsplit('/').next().unwrap();
    compare(&format!("ir/{file}.json"), &ir_text(&json));
}

#[test]
fn scalar_store() {
    golden("scalar_store");
}

#[test]
fn erc20() {
    golden("erc20");
}

#[test]
fn ints() {
    golden("ints");
}

#[test]
fn enums() {
    golden("enums");
}

#[test]
fn loops() {
    golden("loops");
}

#[test]
fn lane_holds() {
    golden("lane_holds");
}

#[test]
fn trust() {
    golden("trust");
}

#[test]
fn immutables() {
    golden("immutables");
}

#[test]
fn review2() {
    golden("ir/review2");
}

#[test]
fn review() {
    golden("ir/review");
}

/// Whether each contract of each fixture serializes, or the construct that
/// stops it, as `golden/ir/coverage.txt` has it.
#[test]
fn coverage() {
    let mut report = String::new();
    for &(name, _) in FIXTURES {
        with_fixture(name, |db, contracts| {
            for &contract in contracts {
                let contract_name = contract.name(db).unwrap().data(db);
                let outcome = match export_ir(db, contract) {
                    Ok(_) => "ok".to_string(),
                    Err(error) => error.0,
                };
                writeln!(report, "{name} {contract_name}: {outcome}").unwrap();
            }
        });
    }
    compare("ir/coverage.txt", &report);
}

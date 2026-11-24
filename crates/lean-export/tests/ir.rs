mod load;

use std::fmt::Write;

use load::{FIXTURES, compare, with_fixture};
use serde_json::json;
use spec_ir::{SpecIr, export_ir, ir_text, merge_library};

/// The fixtures with IR golden files, by `tests/fixtures` path.
const GOLDEN: &[&str] = &[
    "scalar_store",
    "erc20",
    "ints",
    "enums",
    "loops",
    "lane_holds",
    "trust",
    "immutables",
    "ir/review",
    "ir/review2",
    "ir/review3",
    "ir/review4",
    "ir/init_pattern",
];

/// The serialized IR of the last contract of `tests/fixtures/<name>.fe`.
fn serialize(name: &str) -> SpecIr {
    with_fixture(name, |db, contracts| {
        export_ir(db, *contracts.last().unwrap()).expect("serialized")
    })
}

/// The contract file of the last contract of `tests/fixtures/<name>.fe`, as
/// `golden/ir/<file>.json` has it, for the last component `file` of `name`.
fn golden(name: &str) {
    let file = name.rsplit('/').next().unwrap();
    compare(
        &format!("ir/{file}.json"),
        &ir_text(&serialize(name).contract),
    );
}

/// The library of all of the golden fixtures, merged, as
/// `golden/ir/library.json` has it: the library each of their contract
/// files is linked against.
#[test]
fn library() {
    let mut libraries = GOLDEN.iter().map(|name| serialize(name).library);
    let mut library = libraries.next().unwrap();
    for other in libraries {
        merge_library(&mut library, &other).expect("libraries merge");
    }
    compare("ir/library.json", &ir_text(&library));
}

/// Merging is independent of order, and refuses an entry that differs.
#[test]
fn merge() {
    let (a, b) = (
        serialize("scalar_store").library,
        serialize("erc20").library,
    );
    let (mut ab, mut ba) = (a.clone(), b.clone());
    merge_library(&mut ab, &b).unwrap();
    merge_library(&mut ba, &a).unwrap();
    assert_eq!(ab, ba);
    let mut changed = b.clone();
    let entry = changed["funcs"]
        .as_array_mut()
        .unwrap()
        .first_mut()
        .unwrap();
    let key = entry["key"].as_str().unwrap().to_string();
    entry["mayReturn"] = json!(!entry["mayReturn"].as_bool().unwrap());
    let mut merged = a.clone();
    merge_library(&mut merged, &b).unwrap();
    assert_eq!(
        merge_library(&mut merged, &changed),
        Err(format!("the libraries differ in `{key}`"))
    );
    let mut other_target = b;
    other_target["target"]["usizeBits"] = json!(64);
    assert!(merge_library(&mut ab, &other_target).is_err());
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
fn review4() {
    golden("ir/review4");
}

#[test]
fn review2() {
    golden("ir/review2");
}

#[test]
fn review3() {
    golden("ir/review3");
}

#[test]
fn init_pattern() {
    golden("ir/init_pattern");
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

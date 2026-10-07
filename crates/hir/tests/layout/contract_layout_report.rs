use crate::layout_test_support::parse_ok;
use fe_hir::{
    analysis::ty::ProviderAddressSpace,
    core::semantic::{
        ContractLayoutEntry, ContractLayoutEntryKind, ContractLayoutParameterOrigin,
        ContractLayoutValue,
    },
    test_db::{HirAnalysisTestDb, find_contract},
};

fn entry<'a, 'db>(
    db: &'db HirAnalysisTestDb,
    entries: &'a [ContractLayoutEntry<'db>],
    path: &str,
) -> &'a ContractLayoutEntry<'db> {
    entries
        .iter()
        .find(|entry| entry.path.display(db) == path)
        .unwrap_or_else(|| panic!("missing layout entry `{path}`: {entries:#?}"))
}

fn scalar_value(db: &HirAnalysisTestDb, entry: &ContractLayoutEntry<'_>) -> String {
    let ContractLayoutValue::Scalar(value) = entry.value else {
        panic!("expected scalar layout value: {entry:#?}");
    };
    value.data(db).to_string()
}

#[test]
fn report_preserves_inline_array_geometry_and_enum_overlays() {
    parse_ok!(
        db,
        top_mod,
        r#"
struct Leaf { left: u256, right: u8 }
enum Choice { Pair(Leaf), Unit }

contract C {
    mut values: [Leaf; 2],
    mut choice: Choice,
}
"#,
    );
    let contract = find_contract(&db, top_mod, "C");
    let report = contract.layout_report(&db).unwrap();

    for (path, base) in [("values[i0].left", "0"), ("values[i0].right", "1")] {
        let entry = entry(&db, &report.entries, path);
        let ContractLayoutValue::Indexed {
            base: actual,
            dimensions,
            strides,
            extent,
        } = &entry.value
        else {
            panic!("expected indexed inline entry: {entry:#?}");
        };
        assert_eq!(actual.data(&db).to_string(), base);
        assert_eq!(dimensions, &[2]);
        assert_eq!(strides, &[2]);
        assert_eq!(*extent, 3);
    }

    let tag = entry(&db, &report.entries, "choice.<tag>");
    assert_eq!(scalar_value(&db, tag), "4");
    assert_eq!(tag.kind, ContractLayoutEntryKind::EnumTag);
    assert_eq!(
        scalar_value(&db, entry(&db, &report.entries, "choice::Pair.0.left")),
        "5"
    );
    assert_eq!(
        scalar_value(&db, entry(&db, &report.entries, "choice::Pair.0.right")),
        "6"
    );
}

#[test]
fn report_labels_shared_parameters_and_mutually_exclusive_variants() {
    parse_ok!(
        db,
        top_mod,
        r#"
struct Slot<const ROOT: u256 = _> {}
struct Shared<const ROOT: u256 = _> {
    left: Slot<ROOT>,
    right: Slot<ROOT>,
}
struct Explicit<const LEFT: u256 = _, const RIGHT: u256 = _> {}
enum Choice {
    Named { value: Slot },
    Tuple(Slot),
}

contract C {
    mut shared: Shared,
    mut choice: Choice,
    mut explicit: Explicit<4, 4>,
}
"#,
    );
    let contract = find_contract(&db, top_mod, "C");
    let report = contract.layout_report(&db).unwrap();

    for path in ["shared.left.ROOT", "shared.right.ROOT"] {
        let entry = entry(&db, &report.entries, path);
        assert_eq!(scalar_value(&db, entry), "0", "{path}");
        assert_eq!(
            entry.kind,
            ContractLayoutEntryKind::Parameter(ContractLayoutParameterOrigin::Inferred)
        );
    }
    assert_eq!(
        scalar_value(&db, entry(&db, &report.entries, "choice.<tag>")),
        "1"
    );
    for path in ["choice::Named.value.ROOT", "choice::Tuple.0.ROOT"] {
        let entry = entry(&db, &report.entries, path);
        assert_eq!(scalar_value(&db, entry), "2", "{path}");
        assert_eq!(
            entry.kind,
            ContractLayoutEntryKind::Parameter(ContractLayoutParameterOrigin::Inferred)
        );
    }
    for path in ["explicit.LEFT", "explicit.RIGHT"] {
        let entry = entry(&db, &report.entries, path);
        assert_eq!(scalar_value(&db, entry), "4", "{path}");
        assert_eq!(
            entry.kind,
            ContractLayoutEntryKind::Parameter(ContractLayoutParameterOrigin::Explicit)
        );
    }
}

#[test]
fn report_numbers_storage_and_transient_fields_together() {
    parse_ok!(
        db,
        top_mod,
        r#"
use std::evm::TStorPtr

contract C {
    mut stored: u256,
    mut temporary: TStorPtr<u256>,
    mut later: u256,
    immutable: u256,
}
"#,
    );
    let contract = find_contract(&db, top_mod, "C");
    let report = contract.layout_report(&db).unwrap();
    for (path, space, value) in [
        ("stored", ProviderAddressSpace::Storage, "0"),
        ("temporary", ProviderAddressSpace::Transient, "1"),
        ("later", ProviderAddressSpace::Storage, "2"),
        ("immutable", ProviderAddressSpace::Code, "0"),
    ] {
        let entry = entry(&db, &report.entries, path);
        assert_eq!(entry.address_space, space, "{path}");
        assert_eq!(scalar_value(&db, entry), value, "{path}");
    }
}

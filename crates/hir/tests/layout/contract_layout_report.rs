use crate::layout_test_support::{parse_module, parse_ok};
use fe_hir::{
    analysis::{initialize_analysis_pass, ty::ProviderAddressSpace},
    core::semantic::{
        ContractFieldId, ContractLayoutEntry, ContractLayoutEntryKind, ContractLayoutError,
        ContractLayoutValue,
    },
    hir_def::IdentId,
    test_db::{HirAnalysisTestDb, find_contract, format_diagnostics},
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

#[test]
fn mutex_lock_cells_share_the_field_counter() {
    parse_ok!(
        db,
        top_mod,
        r#"
use std::evm::Mutex
use std::evm::effects::TStorPtr

contract C {
    t0: TStorPtr<bool>,
    t1: TStorPtr<bool>,
    mut m: Mutex<u256>,
    mut m2: Mutex<u256>,
    after: TStorPtr<bool>,
}
"#,
    );
    let contract = find_contract(&db, top_mod, "C");
    let layout = contract.storage_layout(&db);
    let fields = &layout
        .allocated
        .as_ref()
        .expect("contract layout should be allocated")
        .fields;
    // Storage and transient fields share one counter. A mutex's value takes
    // a storage slot and its lock, a transient cell, the next number.
    for (name, offset, count) in [
        ("t0", 0, 1),
        ("t1", 1, 1),
        ("m", 2, 2),
        ("m2", 4, 2),
        ("after", 6, 1),
    ] {
        let field = &fields[&IdentId::new(&db, name.to_string())];
        assert_eq!(field.slot_offset, offset, "{name}");
        assert_eq!(field.slot_count, count, "{name}");
    }
}

#[test]
fn layout_extent_overflow_rejects_without_panicking() {
    parse_module!(
        db,
        top_mod,
        r#"
contract C {
    huge: [u256; 576460752303423488],
}
"#,
    );
    let rendered = format_diagnostics(&db, &initialize_analysis_pass().run_on_module(&db, top_mod));
    assert!(
        rendered.contains("contract-field layout extent overflowed"),
        "{rendered}"
    );
    let contract = find_contract(&db, top_mod, "C");
    let result = contract.storage_layout(&db);
    assert!(result.allocated.is_none());
    assert_eq!(
        result.field_errors(&IdentId::new(&db, "huge".to_string())),
        Some([ContractLayoutError::LayoutExtentOverflow].as_slice())
    );
}

#[test]
fn duplicate_contract_field_names_block_layout_without_panicking() {
    parse_module!(
        db,
        top_mod,
        r#"
contract DuplicateFields {
    mut value: u256,
    mut value: u256,
}
"#,
    );
    let contract = find_contract(&db, top_mod, "DuplicateFields");
    let result = contract.storage_layout(&db);
    assert!(result.allocated.is_none());
    assert_eq!(result.field_results.len(), 2);
    for index in 0..2 {
        assert!(matches!(
            result.field_errors_for_id(ContractFieldId { contract, index }),
            Some([ContractLayoutError::InvalidFieldType])
        ));
    }
    let rendered = format_diagnostics(&db, &initialize_analysis_pass().run_on_module(&db, top_mod));
    assert!(
        rendered.contains("duplicate field name in contract `DuplicateFields`"),
        "unexpected diagnostics:\n{rendered}"
    );
}

#[test]
fn ambiguous_handle_impl_rejects_the_field() {
    parse_module!(
        db,
        top_mod,
        r#"
use core::effect_ref::{AddressSpace, EffectHandle}

trait MarkerA {}
trait MarkerB {}

struct Value {}

impl MarkerA for Value {}
impl MarkerB for Value {}

struct Ptr<T> { raw: u256 }

impl<T: MarkerA> EffectHandle for Ptr<T> {
    type Target = T

    const SPACE: AddressSpace = AddressSpace::Storage
    type Raw = u256

    fn raw(self) -> u256 {
        self.raw
    }
}

impl<T: MarkerB> EffectHandle for Ptr<T> {
    type Target = T

    const SPACE: AddressSpace = AddressSpace::Storage
    type Raw = u256

    fn raw(self) -> u256 {
        self.raw
    }
}

contract C {
    mut value: Ptr<Value>,
}
"#,
    );
    let contract = find_contract(&db, top_mod, "C");
    let errors = contract
        .storage_layout(&db)
        .field_errors(&IdentId::new(&db, "value".to_string()))
        .expect("field should be rejected");
    assert!(matches!(
        errors.first(),
        Some(ContractLayoutError::AmbiguousProviderLayout)
    ));
    assert!(contract.storage_layout(&db).allocated.is_none());
}

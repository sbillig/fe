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

#[test]
fn single_field_wrapper_slot_roots_keep_layout_for_field_projection() {
    let yul = emit_inline_yul(
        "file:///single_field_wrapper_slot_roots_keep_layout_for_field_projection.fe",
        r#"
use std::evm::CallData

fn read_base() -> u256 {
    let data = CallData::with_base(4)
    data.base
}
"#,
    );

    assert!(
        yul.contains("function $read_base()"),
        "single-field wrapper field reads should lower to Yul without losing slot layout:\n{yul}"
    );
}

#[test]
fn generated_yul_param_names_do_not_collide_with_function_symbols() {
    let yul = emit_inline_yul(
        "file:///generated_yul_param_names_do_not_collide_with_function_symbols.fe",
        r#"
use std::abi::sol
use std::abi::sol::{decode_bytes_view, decode_bytes_view_at, decode_string_view}
use std::evm::{CallData, Evm}

const BYTES_LEN_SELECTOR: u32 = sol("bytesLen(bytes)")
const SECOND_BYTES_LEN_SELECTOR: u32 = sol("secondBytesLen(bytes,bytes)")
const STRING_FIRST_SELECTOR: u32 = sol("stringFirst(string)")
const STRING_LEN_SELECTOR: u32 = sol("stringLen(string)")

#[contract_init(ViewHarness)]
fn init() uses (evm: mut Evm) {
    evm.create_contract(runtime)
}

#[contract_runtime(ViewHarness)]
fn runtime() uses (evm: mut Evm) {
    let sel = evm.selector()

    if sel == BYTES_LEN_SELECTOR {
        let view = decode_bytes_view(CallData::with_base(4))
        evm.mstore(addr: 0, value: view.len())
        evm.return_data(offset: 0, len: 32)
    }

    if sel == SECOND_BYTES_LEN_SELECTOR {
        let view = decode_bytes_view_at(CallData::with_base(4), base: 0, head_pos: 32)
        evm.mstore(addr: 0, value: view.len())
        evm.return_data(offset: 0, len: 32)
    }

    if sel == STRING_FIRST_SELECTOR {
        let view = decode_string_view(CallData::with_base(4))
        let first: u256 = if view.is_empty() { 0 } else { view.byte_at(0) as u256 }
        evm.mstore(addr: 0, value: first)
        evm.return_data(offset: 0, len: 32)
    }

    if sel == STRING_LEN_SELECTOR {
        let view = decode_string_view(CallData::with_base(4))
        evm.mstore(addr: 0, value: view.len())
        evm.return_data(offset: 0, len: 32)
    }

    evm.revert(offset: 0, len: 0)
}
"#,
    );

    assert!(
        yul.contains("function $input()"),
        "fixture should still emit the sibling $input helper that previously collided with $decode_bytes_view params:\n{yul}"
    );
    assert!(
        yul.contains("function $decode_bytes_view(p0)"),
        "generated Yul params should use the disjoint p{{idx}} namespace instead of semantic names:\n{yul}"
    );
    assert!(
        !yul.contains("function $decode_bytes_view($input)"),
        "generated Yul params must not reuse semantic names that can collide with function symbols:\n{yul}"
    );
}

#[test]
fn dynamic_view_lowering_does_not_reference_missing_allocate_helper() {
    let yul = emit_inline_yul(
        "file:///dynamic_view_lowering_does_not_reference_missing_allocate_helper.fe",
        r#"
use std::abi::sol
use std::abi::sol::decode_bytes_view_at
use std::evm::{CallData, Evm}

const SECOND_BYTES_LEN_SELECTOR: u32 = sol("secondBytesLen(bytes,bytes)")

#[contract_init(ViewHarness)]
fn init() uses (evm: mut Evm) {
    evm.create_contract(runtime)
}

#[contract_runtime(ViewHarness)]
fn runtime() uses (evm: mut Evm) {
    if evm.selector() == SECOND_BYTES_LEN_SELECTOR {
        let view = decode_bytes_view_at(CallData::with_base(4), base: 0, head_pos: 32)
        evm.mstore(addr: 0, value: view.len())
        evm.return_data(offset: 0, len: 32)
    }

    evm.revert(offset: 0, len: 0)
}
"#,
    );

    assert!(
        !yul.contains("allocate("),
        "dynamic view lowering should inline free-memory-pointer allocation instead of calling a missing helper:\n{yul}"
    );
    assert!(
        yul.contains("mstore(0x40, add("),
        "dynamic view lowering should still advance the free-memory pointer for temporary allocations:\n{yul}"
    );
}

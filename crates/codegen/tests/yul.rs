use common::InputDb;
use dir_test::{Fixture, dir_test};
use driver::DriverDataBase;
use fe_codegen::emit_module_yul;
use std::fs;
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

fn emit_fixture_yul(path: &str) -> String {
    let abs_path = format!("{}/tests/fixtures/{path}", env!("CARGO_MANIFEST_DIR"));
    let src = fs::read_to_string(&abs_path)
        .unwrap_or_else(|err| panic!("failed to read fixture `{abs_path}`: {err}"));
    let file_url = Url::from_file_path(&abs_path).expect("fixture path should be absolute");
    let mut db = DriverDataBase::default();
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(src.to_owned()));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    emit_module_yul(&db, top_mod).expect("Yul emission should succeed")
}

fn yul_function_body<'a>(yul: &'a str, name: &str) -> &'a str {
    let marker = format!("function ${name}(");
    let start = yul
        .find(&marker)
        .unwrap_or_else(|| panic!("missing function `{name}` in emitted Yul:\n{yul}"));
    let tail = &yul[start..];
    let end = tail.find("\n      function $").unwrap_or(tail.len());
    &tail[..end]
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
fn aggregate_ref_roots_keep_layout_for_field_projection() {
    let yul = emit_inline_yul(
        "file:///aggregate_ref_roots_keep_layout_for_field_projection.fe",
        r#"
use core::ops::{Add, AddAssign}

struct OnlyAssign {
    value: u8,
}

impl Copy for OnlyAssign {}

impl AddAssign for OnlyAssign {
    fn add_assign(mut self, _ other: own OnlyAssign) {
        self.value = self.value + other.value + 1
    }
}

struct Divergent {
    value: u8,
}

impl Copy for Divergent {}

impl Add for Divergent {
    fn add(own self, _ other: own Divergent) -> Divergent {
        Divergent { value: 99 }
    }
}

impl AddAssign for Divergent {
    fn add_assign(mut self, _ other: own Divergent) {
        self.value = 7
    }
}

fn use_add_assign() {
    let mut x = OnlyAssign { value: 1 }
    let y = OnlyAssign { value: 2 }
    x += y

    let mut a = Divergent { value: 1 }
    let b = Divergent { value: 2 }
    a += b
}
"#,
    );

    assert!(
        yul.contains("function $add_assign_1(") && yul.contains("function $add_assign_0("),
        "aggregate ref roots should keep pointer/layout shape through field projection lowering:\n{yul}"
    );
}

#[test]
fn compound_if_let_pointer_roots_lower_to_yul() {
    emit_inline_yul(
        "file:///compound_if_let_pointer_roots_lower_to_yul.fe",
        r#"
use std::evm::effects::assert

fn sum_compound_if_let(_ a: Option<Option<usize>>) -> usize {
    if let Option::Some(b) = a && let Option::Some(c) = b {
        c
    } else {
        0
    }
}

#[test]
fn test_sum_compound_if_let() {
    assert(sum_compound_if_let(Option::Some(Option::Some(7))) == 7)
    assert(sum_compound_if_let(Option::Some(Option::None)) == 0)
    assert(sum_compound_if_let(Option::None) == 0)
}
"#,
    );
}

#[test]
fn packed_byte_arrays_use_byte_loads_and_stores_in_yul() {
    let yul = emit_inline_yul(
        "file:///packed_byte_arrays_use_byte_loads_and_stores_in_yul.fe",
        r#"
fn mutate_u8() -> u8 {
    let mut xs: [u8; 4] = [0, 1, 2, 3]
    xs[1] = 9
    xs[1]
}
"#,
    );

    let body = yul_function_body(&yul, "mutate_u8");
    assert!(
        body.contains("mstore8(add("),
        "packed byte arrays should lower stores with mstore8:\n{body}"
    );
    assert!(
        body.contains("byte(0, mload(add("),
        "packed byte arrays should lower loads from the first byte of the loaded word:\n{body}"
    );
}

#[test]
fn memory_enum_tags_use_packed_byte_ops_in_yul() {
    let yul = emit_inline_yul(
        "file:///memory_enum_tags_use_packed_byte_ops_in_yul.fe",
        r#"
enum MyOption {
    None,
    Some(u256),
}

fn unwrap_some() -> u256 {
    let value = MyOption::Some(42)
    match value {
        MyOption::Some(inner) => inner,
        MyOption::None => 0,
    }
}
"#,
    );

    let body = yul_function_body(&yul, "unwrap_some");
    assert!(
        body.contains("mstore8("),
        "memory enum tag writes should use packed byte stores:\n{body}"
    );
    assert!(
        body.contains("byte(0, mload("),
        "memory enum tag reads should use packed byte loads:\n{body}"
    );
}

#[test]
fn materialized_fixed_array_init_args_do_not_keep_code_transport_in_yul() {
    let yul = emit_inline_yul(
        "file:///materialized_fixed_array_init_args_do_not_keep_code_transport_in_yul.fe",
        r#"
use std::evm::effects::assert

msg ContractMsg {
    #[selector = 0x01]
    Sum -> u256,
}

struct Store {
    sum: u256,
}

pub contract C {
    mut store: Store

    init(values: [u256; 4]) uses (mut store) {
        store.sum = values[0] + values[1] + values[2] + values[3]
    }

    recv ContractMsg {
        Sum -> u256 uses (store) {
            store.sum
        }
    }
}

#[test]
fn test_contract_init_fixed_array_arg() uses (evm: mut Evm) {
    let c = evm.create2<C>(value: 0, args: ([1, 2, 3, 4],), salt: 0)
    let sum: u256 = evm.call(addr: c, gas: 100000, value: 0, message: ContractMsg::Sum {})
    assert(sum == 10)
}
"#,
    );

    assert!(
        !yul.contains("function $create2_arg2_f0_code("),
        "fixed array init args copied into memory must not keep stale code transport on create2 specialization:\n{yul}"
    );
    assert!(
        !yul.contains("function $encode_payload_2_arg0_root_code("),
        "fixed array payload encoding after memory materialization must not keep stale code-root helpers:\n{yul}"
    );
}

#[test]
fn effect_handle_field_deref_loads_provider_words_before_storage_access() {
    let yul = emit_fixture_yul("effect_handle_field_deref.fe");
    let body = yul_function_body(&yul, "bump");

    assert!(
        !body.contains("sload(p0)"),
        "field deref through a stored effect handle should load the provider word before storage access:\n{body}"
    );
    assert!(
        body.contains("mload(p0)"),
        "field deref through a stored effect handle should read the handle out of the wrapper object:\n{body}"
    );
}

#[test]
fn nested_object_ref_fields_are_followed_before_payload_projection_in_yul() {
    let yul = emit_inline_yul(
        "file:///nested_object_ref_fields_are_followed_before_payload_projection_in_yul.fe",
        r#"
struct Slot {
    value: u256,
}

struct Reader {
    source: ref Slot,
    bias: u256,
}

fn sum_reader_once(_ r: ref Reader) -> u256 {
    r.source.value + r.bias
}

fn call() -> u256 {
    let slot = Slot { value: 7 }
    let reader = Reader {
        source: ref slot,
        bias: 3,
    }
    sum_reader_once(ref reader)
}
"#,
    );
    let body = yul_function_body(&yul, "sum_reader_once");

    assert!(
        body.contains("mload(p0)"),
        "nested object ref lowering should load the intermediate handle word from the outer object:\n{body}"
    );
    assert!(
        body.contains("mload(t"),
        "nested object ref lowering should follow the intermediate handle before reading its payload:\n{body}"
    );
}

#[test]
fn sol_decoder_keeps_byte_input_fields_aggregate_backed_in_yul() {
    let yul = emit_fixture_yul("effect_handle_field_deref.fe");
    let body = yul_function_body(&yul, "read_word_0");

    assert!(
        !body.contains("mload(p0)"),
        "decoder word reads should treat `self.cur.input` as an aggregate field reference, not load its first word:\n{body}"
    );
}

#[test]
fn single_field_wrapper_ctors_return_words_in_yul() {
    let yul = emit_inline_yul(
        "file:///single_field_wrapper_ctors_return_words_in_yul.fe",
        r#"
use std::evm::CallData

fn ctor() -> CallData {
    CallData::with_base(4)
}
"#,
    );

    assert!(
        yul.contains("function $ctor() -> ret {\n      let v2 := 0x04\n      let v0 := v2\n      let v3 := $with_base(v0)\n      let v1 := v3\n      ret := v1"),
        "single-field wrapper ctor callsites should treat the result as a word, not re-materialize an object:\n{yul}"
    );
    assert!(
        !yul.contains("function $with_base(p0) -> ret {\n      let r0 := mload(0x40)\n      if iszero(r0) {\n        r0 := 0x80\n      }\n      mstore(0x40, add(r0, 32))\n      mstore(r0, p0)\n      let t4 := mload(r0)\n      let v1 := t4\n      let t5 := mload(0x40)"),
        "single-field wrapper ctors must not allocate and return a second object wrapper:\n{yul}"
    );
    let body = yul_function_body(&yul, "with_base");
    assert!(
        body.contains("ret := p0")
            || body.contains("let v1 := p0\n      ret := v1")
            || body.contains("let v1 := p0\n      let v2 := v1\n      ret := v2")
            || body
                .contains("let v1 := p0\n      let v3 := v1\n      let v2 := v1\n      ret := v2"),
        "single-field wrapper ctor helpers should return the underlying word directly:\n{body}"
    );
    assert!(
        !body.contains("mload(p0)") && !body.contains("mload(v1)") && !body.contains("mload(v3)"),
        "single-field wrapper ctor helpers must not reinterpret the scalar arg as a memory pointer:\n{body}"
    );
}

#[test]
fn structured_yul_loops_allow_early_returns_alongside_normal_exit() {
    let _yul = emit_inline_yul(
        "file:///structured_yul_loops_allow_early_returns_alongside_normal_exit.fe",
        r#"
fn iterate(_ max_iter: u256) -> u256 {
    let mut i: u256 = 0
    while i < max_iter {
        if i == 3 {
            return i
        }
        i += 1
    }

    if i == max_iter {
        return i + 1
    }
    0
}
"#,
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
fn hidden_self_root_slots_can_project_fields_without_visible_value_classes() {
    let yul = emit_fixture_yul("erc20.fe");

    assert!(
        yul.contains("function $grant("),
        "erc20 fixture should still emit the AccessControl grant helper after Yul legalization:\n{yul}"
    );
    assert!(
        yul.contains("function $require("),
        "erc20 fixture should still emit the AccessControl require helper that uses the hidden self slot root:\n{yul}"
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

#[test]
fn create_contract_helpers_keep_scalar_params_value_backed() {
    let yul = emit_inline_yul(
        "file:///create_contract_helpers_keep_scalar_params_value_backed.fe",
        r#"
use std::evm::Evm

#[contract_init(CreateHarness)]
fn init() uses (evm: mut Evm) {
    evm.create_contract(runtime)
}

#[contract_runtime(CreateHarness)]
fn runtime() uses (evm: mut Evm) {
    evm.return_data(offset: 0, len: 0)
}
"#,
    );

    assert!(
        yul.contains("function $return_data_1(p0, p1) {\n      let v3 := p0\n      let v4 := p1\n      return(v3, v4)"),
        "create_contract return-data helpers should stay value-backed and forward scalar params without spilling through memory slots:\n{yul}"
    );
    assert!(
        !yul.contains("function $return_data_1(p0, p1) {\n      let r1 := mload(0x40)"),
        "create_contract return-data helpers must not allocate scalar root slots from mload(0x40):\n{yul}"
    );
}

#[test]
fn scalar_collapsible_struct_consts_stay_aggregate_in_yul_const_regions() {
    let _yul = emit_inline_yul(
        "file:///scalar_collapsible_struct_consts_stay_aggregate_in_yul_const_regions.fe",
        r#"
use std::evm::mem

struct Pair {
    a: Address,
    b: Address,
}

#[test]
fn test_alloc_after_struct() uses (evm: Evm) {
    let p = Pair {
        a: Address { inner: 0x1111111111111111111111111111111111111111 },
        b: Address { inner: 0x2222222222222222222222222222222222222222 },
    }

    let ptr = mem::alloc(32)

    assert(ptr < 0x100000)
    assert(p.a.inner == 0x1111111111111111111111111111111111111111)
    assert(p.b.inner == 0x2222222222222222222222222222222222222222)
}
"#,
    );
}

#[test]
fn scalar_collapsible_field_loads_materialize_words_in_yul() {
    let yul = emit_inline_yul(
        "file:///scalar_collapsible_field_loads_materialize_words_in_yul.fe",
        r#"
use std::abi::sol::decode_bytes_view
use std::evm::CallData

fn read_word() -> u256 {
    let view = decode_bytes_view(CallData::with_base(4))
    view.word_at(0)
}
"#,
    );

    let body = yul_function_body(&yul, "word_at_0");
    assert!(
        body.contains("mload(r0)"),
        "collapsed aggregate field loads should materialize the field word before forwarding it:\n{body}"
    );
    assert!(
        !body.contains("let v2 := r0"),
        "collapsed aggregate field loads must not forward the enclosing object root as a scalar receiver:\n{body}"
    );
}

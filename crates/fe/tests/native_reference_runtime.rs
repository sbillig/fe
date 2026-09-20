use std::{fmt::Write, path::PathBuf};

use codegen::{OptLevel, emit_module_sonatina_bytecode};
use common::{
    InputDb,
    file::File,
    stdlib::{BUILTIN_STD_BASE_URL, HasBuiltinCore, HasBuiltinStd},
};
use contract_harness::{ExecutionOptions, RuntimeInstance};
use driver::DriverDataBase;
use fe::bench_support::compile_fe_sonatina_bytecode;
use hir::analysis::{
    semantic::{get_or_build_semantic_instance, identity_semantic_instance_key},
    ty::ty_check::BodyOwner,
};
use mir::{
    MirDb, RefKind, RuntimeCarrier, RuntimeClass, RuntimeInstanceKey, build_runtime_package,
    get_or_build_runtime_instance, instance::RuntimeInstanceSource, verify_runtime_body,
    verify_runtime_package,
};
use test_utils::snap_test;
use url::Url;

fn trusted_source(source: String) -> (DriverDataBase, File) {
    let mut db = DriverDataBase::default();
    db.initialize_builtin_core();
    db.initialize_builtin_std();
    let base = Url::parse("test-effect-handle:/").unwrap();
    db.workspace().touch(
        &mut db,
        base.join("fe.toml").unwrap(),
        Some("[ingot]\nname = \"native-space-test\"\nversion = \"0.0.0\"\n".into()),
    );
    db.dependency_graph().add_dependency(
        &mut db,
        &base,
        &Url::parse(BUILTIN_STD_BASE_URL).unwrap(),
        "std".into(),
        Default::default(),
    );
    let file = db
        .workspace()
        .touch(&mut db, base.join("src/lib.fe").unwrap(), Some(source));
    (db, file)
}

fn native_space_source(space: &str, writable: bool) -> String {
    let kind = if writable { "mut" } else { "ref" };
    let effect_mode = if writable { "mut " } else { "" };
    let offset = if space == "Calldata" { 4 } else { 0 };
    let initialize = if matches!(space, "Storage" | "TransientStorage") {
        "value = Cell { prefix: 7, pair: [11, 23] }"
    } else {
        ""
    };
    let update = if writable { "(*slot).pair[1] += 1" } else { "" };
    format!(
        r#"
use core::ptr
use core::effect_ref::{{EffectHandle, EffectRef, EffectRefMut}}
struct Provider<T> {{ address: u256 }}
impl<T> Copy for Provider<T> {{}}
impl<T> EffectHandle for Provider<T> {{
    type Target = T
    const SPACE: core::effect_ref::AddressSpace = core::effect_ref::AddressSpace::{space}
    type Raw = u256
    fn raw(self) -> u256 {{ self.address }}
}}
impl<T> EffectRef<T> for Provider<T> {{}}
impl<T> EffectRefMut<T> for Provider<T> {{}}
struct Cell {{ prefix: u8, pair: [u8; 2] }}

#[inline(never)]
fn pass(slot: *{kind} Cell) -> {kind} Cell {{ *slot }}
#[inline(never)]
fn project(slot: *{kind} Cell) -> u8 {{ (*slot).pair[1] }}
#[inline(never)]
fn churn(seed: u256) -> *u256 {{
    let allocation = ptr::alloc<u256>()
    *allocation = seed
    allocation
}}
fn observe(seed: u256) -> u8 uses (value: {effect_mode}Cell) {{
    {initialize}
    let slot = ptr::alloc<{kind} Cell>()
    *slot = {kind} value
    {update}
    let carrier = pass(slot)
    let second = ptr::alloc<{kind} Cell>()
    *second = carrier
    let scratch = churn(seed)
    assert!(*scratch == seed)
    project(slot: second)
}}
msg NativeMsg {{
    #[selector = 1]
    Run {{ seed: u256 }} -> u8,
}}
pub contract NativeSpaceContract {{
    recv NativeMsg {{
        Run {{ seed }} -> u8 {{
            let mut provider = Provider<Cell> {{ address: {offset} }}
            with (provider) {{ observe(seed) }}
        }}
    }}
}}
"#
    )
}

#[test]
fn native_descriptors_preserve_provider_spaces_and_projected_returns() {
    for (space, writable) in [
        ("Storage", true),
        ("TransientStorage", true),
        ("Calldata", false),
        ("Code", false),
    ] {
        let (db, file) = trusted_source(native_space_source(space, writable));
        let module = db.top_mod(file);
        let diagnostics = db.run_on_top_mod(module);
        assert!(diagnostics.is_empty(), "{}", diagnostics.format_diags(&db));
        let mut artifacts = emit_module_sonatina_bytecode(&db, module, OptLevel::O2, None)
            .unwrap_or_else(|error| panic!("{space}: {error:?}"));
        let artifact = artifacts.remove("NativeSpaceContract").unwrap();
        let expected = if space == "Code" {
            artifact.runtime[2]
        } else if space == "Calldata" {
            33
        } else {
            24
        };
        let mut runtime = RuntimeInstance::deploy(&hex::encode(artifact.deploy)).expect("deploy");
        let mut calldata = vec![0u8; 36];
        calldata[3] = 1;
        calldata[4..7].copy_from_slice(&[11, 22, 33]);
        let result = runtime
            .call_raw(&calldata, ExecutionOptions::default())
            .unwrap_or_else(|error| panic!("{space}: {error:?}"));
        let mut expected_data = vec![0u8; 32];
        expected_data[31] = expected;
        assert_eq!(result.return_data, expected_data, "{space}");
    }
}

#[test]
fn readonly_native_descriptors_reject_mutable_access() {
    for space in ["Calldata", "Code"] {
        let (db, file) = trusted_source(native_space_source(space, true));
        let module = db.top_mod(file);
        let diagnostics = db.run_on_top_mod(module);
        assert!(diagnostics.is_empty(), "{}", diagnostics.format_diags(&db));
        let diagnostics = db.format_complete_diagnostics(&db.mir_diagnostics_for_top_mod(module));
        assert!(
            diagnostics.contains(&format!("cannot write to {}", space.to_lowercase())),
            "{space}: {diagnostics}"
        );
        let error = build_runtime_package(&db, module)
            .expect_err("concrete specialization must reject a readonly write")
            .to_string();
        assert!(error.contains("storage violation"), "{space}: {error}");
        assert!(!error.contains("internal"), "{error}");
    }
}

#[test]
fn native_transport_signature_and_body_are_query_order_independent() {
    let helper = r#"
struct Choice { first: ref u8, second: ref u8 }
fn choose(value: Choice, take_first: bool) -> ref u8 {
    if take_first { value.first } else { value.second }
}
"#;
    let caller = r#"
fn entry() -> u8 {
    let first: [u8; 2] = [11, 23]
    let second = core::ptr::alloc<[u8; 2]>()
    *second = [37, 41]
    let value = Choice { first: ref first[1], second: ref (*second)[0] }
    choose(value, take_first: true)
}
"#;
    for source in [format!("{helper}\n{caller}"), format!("{caller}\n{helper}")] {
        for signature_first in [true, false] {
            let (db, file) = trusted_source(source.clone());
            let module = db.top_mod(file);
            let diagnostics = db.run_on_top_mod(module);
            assert!(diagnostics.is_empty(), "{}", diagnostics.format_diags(&db));
            let function = module
                .all_funcs(&db)
                .iter()
                .copied()
                .find(|function| {
                    function
                        .name(&db)
                        .to_opt()
                        .is_some_and(|name| name.data(&db) == "entry")
                })
                .unwrap();
            let semantic = get_or_build_semantic_instance(
                &db,
                identity_semantic_instance_key(&db, BodyOwner::Func(function)),
            );
            let key =
                RuntimeInstanceKey::new(&db, RuntimeInstanceSource::Semantic(semantic), Vec::new());
            let instance = get_or_build_runtime_instance(&db, key);
            if signature_first {
                instance.interface_signature(&db);
            }
            let body = instance.body(&db);
            let signature = instance.interface_signature(&db);
            assert_eq!(body.signature, signature);
            assert!(body.locals.iter().any(|local| matches!(
                local.carrier,
                RuntimeCarrier::Value(RuntimeClass::Ref {
                    kind: RefKind::Native,
                    ..
                })
            )));
            let program: &dyn MirDb = &db;
            verify_runtime_body(&db, &program, &body).expect("body verifier");
            let package = build_runtime_package(&db, module).expect("runtime package");
            verify_runtime_package(&db, package).expect("release package verifier");
        }
    }
}

#[test]
#[allow(clippy::print_stdout)]
fn native_transport_records_gas_and_code_size() {
    let mut report = String::from(
        "O2; gas includes transaction/calldata costs\ncase deploy_bytes runtime_bytes first_gas second_gas\n",
    );
    for (name, helpers) in [
        (
            "static_view",
            r#"
#[inline(never)]
fn select(cell: Cell, take_first: bool) -> u8 {
    if take_first { cell.pair[0] } else { cell.pair[1] }
}
#[inline(never)]
fn measure(cell: Cell, take_first: bool) -> u8 { select(cell, take_first) }
"#,
        ),
        (
            "native_carriers",
            r#"
struct Holder { first: ref u8, second: ref u8 }
#[inline(never)]
fn select(holder: Holder, take_first: bool) -> ref u8 {
    if take_first { holder.first } else { holder.second }
}
#[inline(never)]
fn measure(cell: Cell, take_first: bool) -> u8 {
    let holder = Holder { first: ref cell.pair[0], second: ref cell.pair[1] }
    select(holder, take_first)
}
"#,
        ),
    ] {
        let source = format!(
            r#"
struct Cell {{ pair: [u8; 2] }}
{helpers}
msg BenchMsg {{
    #[selector = 1]
    Run {{ first: u8, second: u8, take_first: bool }} -> u8,
}}
pub contract NativeBench {{
    recv BenchMsg {{
        Run {{ first, second, take_first }} -> u8 {{
            let cell = Cell {{ pair: [first, second] }}
            measure(cell, take_first)
        }}
    }}
}}
"#
        );
        let artifact =
            compile_fe_sonatina_bytecode(&source, name, "NativeBench").expect("compile benchmark");
        let mut runtime =
            RuntimeInstance::deploy(&hex::encode(&artifact.deploy)).expect("deploy benchmark");
        let mut gas = Vec::new();
        for (take_first, expected) in [(true, 11), (false, 23)] {
            let mut calldata = vec![0u8; 100];
            calldata[3] = 1;
            calldata[35] = 11;
            calldata[67] = 23;
            calldata[99] = u8::from(take_first);
            let result = runtime
                .call_raw(&calldata, ExecutionOptions::default())
                .expect("benchmark call");
            let mut expected_data = vec![0u8; 32];
            expected_data[31] = expected;
            assert_eq!(result.return_data, expected_data, "{name} {take_first}");
            gas.push(result.gas_used);
        }
        writeln!(
            report,
            "{name} {} {} {} {}",
            artifact.deploy.len(),
            artifact.runtime.len(),
            gas[0],
            gas[1]
        )
        .unwrap();
    }
    println!("{report}");
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/native_reference_runtime_cost");
    snap_test!(report, path.to_str().unwrap());
}

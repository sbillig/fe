//! Runs the exporter over many contracts, for `scripts/audit_exports.sh` of
//! fe-equivm-proofs. Both tests are ignored: they read their input from the
//! environment.

use std::{collections::BTreeMap, env, fmt::Write, fs, panic, path::Path};

use codegen::{
    OptLevel, SonatinaTestOptions, TestSuiteOutput, emit_ingot_sonatina_bytecode,
    emit_ingot_sonatina_sidecar, emit_module_sonatina_bytecode, emit_module_sonatina_sidecar,
    emit_test_suite_sonatina,
};
use common::{InputDb, ingot::Ingot};
use cranelift_entity::EntityRef;
use driver::{DriverDataBase, init_ingot};
use fe_lean_export::{IngotExport, Library, Unsupported, export_ir, ir_text};
use hir::{
    analysis::{
        semantic::{
            NExpr, NStatementKind, SConst, SemanticInstance, SemanticInstanceKey,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{
            abi_ty::semantic_ty_to_abi_desc, corelib::resolve_lib_type_path, ty_check::BodyOwner,
            ty_def::TyId,
        },
    },
    hir_def::{Contract, HirIngot, TopLevelMod},
    test_db::HirAnalysisTestDb,
};
use num_bigint::BigUint;
use rustc_hash::FxHashSet;
use url::Url;

/// Exports every contract of the colon-separated files and ingot directories
/// in `LEAN_EXPORT_CORPUS`. Writes the exports of each file to
/// `LEAN_EXPORT_CORPUS_OUT/<file>.lean`, importing the modules
/// `LEAN_EXPORT_CORPUS_IMPORTS` names, each contract in a namespace so that
/// the file elaborates on its own, after the creation code of each contract
/// an export deploys and the definitions of the file's code that more than
/// one of its contracts reaches, in `Corpus.<file>_shared`, and prints the
/// construct each other contract stopped at as `BLOCK <ids>\t<reason>`.
///
/// Writes what each exported contract takes on trust to
/// `LEAN_EXPORT_CORPUS_OUT/<file>.trust`, each line after the contract's
/// namespace, and the checks of the constants the exports define to
/// `<file>.checks`, Lean commands that hold after the exports.
///
/// Writes the normalized IR of each exported contract, for fe-spec's
/// interpreter, to `LEAN_EXPORT_CORPUS_OUT/<file>_<index>.ir.json`, or the
/// construct the serializer stopped at to `<file>_<index>.ir.refused`.
///
/// A file that compiles ends in `Corpus.targets`, the creation and runtime
/// bytecode, model, interface and IR path of each exported contract. A call of
/// `Fe.EVM.Diff.main` on them and the object the file's tests compile to
/// runs each test and compares every call it makes on such a contract with
/// the contract's model. A call of `Fe.EVM.Diff.generated` does the same for
/// generated calls.
#[test]
#[ignore = "reads its input from the environment"]
fn export_corpus() {
    let out = env::var("LEAN_EXPORT_CORPUS_OUT").unwrap();
    let imports: String = env::var("LEAN_EXPORT_CORPUS_IMPORTS")
        .unwrap()
        .split_whitespace()
        .map(|module| format!("import {module}\n"))
        .collect();
    let mut blockers = BTreeMap::<String, Vec<String>>::new();
    let (mut exported, mut skipped, mut uncompiled, mut total) = (0, Vec::new(), Vec::new(), 0);
    let files = env::var("LEAN_EXPORT_CORPUS").unwrap();
    for path in files.split(':') {
        let name = file_name(path);
        // A file or ingot with errors is skipped. A panic of the exporter is
        // reported as what stopped the contract.
        let Some((db, url)) = analyze(path) else {
            skipped.push(name);
            continue;
        };
        let db = &db;
        let (ingot, modules) = modules(db, path, &url);
        let mut library = Library::default();
        let mut ingot_export = IngotExport::new(db);
        let (mut models, mut unexported) = (Vec::new(), Vec::new());
        let contracts = modules.iter().flat_map(|module| module.all_contracts(db));
        for (index, &contract) in contracts.enumerate() {
            total += 1;
            let id = format!("{name}_{index}");
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                ingot_export.export_contract(contract, true, &mut library)
            }))
            .unwrap_or_else(|_| Err(Unsupported("PANIC".to_string())));
            // The contract's IR for fe-spec's interpreter, or why it is not serialized,
            // which the Lean file reads when it runs.
            let ir = panic::catch_unwind(panic::AssertUnwindSafe(|| export_ir(db, contract)))
                .unwrap_or_else(|_| Err(Unsupported("PANIC".to_string())));
            match ir {
                Ok(json) => fs::write(format!("{out}/{id}.ir.json"), ir_text(&json)),
                Err(reason) => fs::write(format!("{out}/{id}.ir.refused"), reason.0),
            }
            .unwrap();
            let name = contract.name(db).unwrap().data(db).to_string();
            match result {
                Ok(()) => {
                    exported += 1;
                    let bytes = contract.code_address_space_slot_count(db) * 32;
                    models.push((
                        id.clone(),
                        name,
                        format!(
                            "some ⟨deploy Corpus.{id}.init_entry, entry Corpus.{id}.runtime, \
                             {bytes}⟩"
                        ),
                        interface(db, contract),
                    ));
                }
                // A contract the exporter stops at is compared with the interpreter only.
                Err(reason) => {
                    blockers.entry(reason.0).or_default().push(id.clone());
                    unexported.push((id, name, "none".to_string(), interface(db, contract)));
                }
            }
        }
        // The definitions the file's contracts share, then each contract's.
        let exports = ingot_export.finish(&library);
        let shared = format!("{name}_shared");
        let (mut text, mut trust, mut checks) = (String::new(), String::new(), String::new());
        let parts = [(shared.as_str(), &exports.shared)].into_iter().chain(
            models
                .iter()
                .map(|(id, ..)| id.as_str())
                .zip(&exports.contracts),
        );
        for (id, lean) in parts {
            let uses = lean.uses_command("Corpus.Lib", &format!("Corpus.{shared}"));
            write!(
                text,
                "\nnamespace Corpus.{id}\nsection\n{uses}\
                 set_option linter.unusedVariables false\n\n{}\nend\nend Corpus.{id}\n",
                lean.text
            )
            .unwrap();
            for item in &lean.trusted {
                writeln!(trust, "Corpus.{id} {item}").unwrap();
            }
            if !lean.checks.is_empty() {
                write!(
                    checks,
                    "\nnamespace Corpus.{id}\n{uses}{}end Corpus.{id}\n",
                    lean.checks
                )
                .unwrap();
            }
        }
        if models.is_empty() && unexported.is_empty() {
            continue;
        }
        let compiled = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let options = SonatinaTestOptions::default();
            let code = match ingot {
                Some(ingot) => emit_ingot_sonatina_bytecode(db, ingot, OptLevel::O1, None),
                None => emit_module_sonatina_bytecode(db, modules[0], OptLevel::O1, None),
            };
            let suite = emit_test_suite_sonatina(db, &modules, OptLevel::O1, options, None);
            Some((code.ok()?, suite.ok()?))
        }));
        // An export that deploys a contract names the contract's creation code.
        if let Ok(Some((code, suite))) = compiled {
            let creations: String = code
                .iter()
                .filter(|(contract, _)| text.contains(&format!("creation_{contract}")))
                .map(|(contract, code)| {
                    format!(
                        "\ndef creation_{contract} : ByteArray :=\n  Fe.EVM.Diff.hex \"{}\"\n",
                        hex(&code.deploy)
                    )
                })
                .collect();
            text.insert_str(0, &creations);
            let targets: Vec<_> = models
                .iter()
                .chain(&unexported)
                .map(|(id, contract, model, interface)| {
                    format!(
                        "{{ name := \"{contract}\", creation := Fe.EVM.Diff.hex \"{}\", \
                         runtime := Fe.EVM.Diff.hex \"{}\",\n     model := {model},\n     \
                         interface := {interface},\n     ir := \"{out}/{id}\", \
                         creations := Corpus.creations }}",
                        hex(&code[contract].deploy),
                        hex(&code[contract].runtime)
                    )
                })
                .collect();
            // The interpreter finds the creation code of a contract another deploys by
            // its name.
            let all_creations: Vec<_> = code
                .iter()
                .map(|(contract, code)| {
                    format!(
                        "(\"{contract}\", Fe.EVM.Diff.hex \"{}\")",
                        hex(&code.deploy)
                    )
                })
                .collect();
            write!(
                text,
                "\ndef Corpus.creations : List (String × ByteArray) :=\n  [{}]\n\n\
                 def Corpus.targets : List Fe.EVM.Diff.Target :=\n  [{}]\n",
                all_creations.join(",\n   "),
                targets.join(",\n   ")
            )
            .unwrap();
            if let Some(suite) = suite {
                write!(
                    text,
                    "\n#eval Fe.EVM.Diff.main \"{name}\" Corpus.targets\n  (Fe.EVM.Diff.hex \"{}\")\n  [{}]\n",
                    hex(&suite.bytecode),
                    tests(&suite)
                )
                .unwrap();
            }
            writeln!(
                text,
                "\n#eval Fe.EVM.Diff.generated \"{name}\" Corpus.targets"
            )
            .unwrap();
        } else {
            uncompiled.push(name);
        }
        // The file's contracts share the definitions of core and std. Lean
        // elaborates a loop that runs in the exit of loops before it deeper
        // than its default limit of 512 allows when they carry many
        // variables, as the four loops of fevm's secp256k1 `combination` do.
        // Mathlib's and Batteries' linters walk each definition's
        // elaboration recursively, and run out of stack on fevm's modexp; the
        // exported text has nothing for them to check.
        let header = format!(
            "{imports}set_option maxRecDepth 2048\nset_option linter.all false\n\
             open Ethereum Ethereum.EVM Fe Fe.EVM\n\n\
             namespace Corpus.Lib\nsection\n\
             set_option linter.unusedVariables false\n\n{}\nend\nend Corpus.Lib\n",
            library.text()
        );
        fs::write(format!("{out}/{name}.lean"), header + &text).unwrap();
        fs::write(format!("{out}/{name}.trust"), trust).unwrap();
        let library_checks = format!(
            "\nnamespace Corpus.Lib\n{}end Corpus.Lib\n",
            library.checks()
        );
        fs::write(format!("{out}/{name}.checks"), library_checks + &checks).unwrap();
    }
    println!(
        "contracts: {total}, exported: {exported}, files skipped: {skipped:?}, \
         not compiled: {uncompiled:?}"
    );
    for (reason, ids) in &blockers {
        let reason: String = reason.chars().take(400).collect();
        println!("BLOCK {}\t{reason}", ids.join(","));
    }
}

/// Compares the Sonatina IR of every contract of the colon-separated files and
/// ingot directories in `SONATINA_CORPUS` with its bytecode, for
/// `scripts/audit_sonatina.sh` of fe-equivm-proofs. Writes the sidecar of each
/// contract to `SONATINA_CORPUS_OUT/<file>.<contract>.sidecar.json`, and
/// `<file>.lean`, which runs the file's tests (`Fe.EVM.Diff.main`) and
/// generated calls (`Fe.EVM.Diff.generated`) with the IR of each contract as
/// its model.
/// Serializes the normalized IR of every contract of the colon-separated
/// files and ingot directories in `LEAN_EXPORT_CORPUS` to
/// `IR_CORPUS_OUT/<file>_<index>.json`, and prints the construct each
/// contract that does not serialize stopped at as `BLOCK <ids>\t<reason>`.
#[test]
#[ignore = "reads its input from the environment"]
fn ir_corpus() {
    let out = env::var("IR_CORPUS_OUT").unwrap();
    let mut blockers = BTreeMap::<String, Vec<String>>::new();
    let (mut serialized, mut skipped, mut total) = (0, Vec::new(), 0);
    let files = env::var("LEAN_EXPORT_CORPUS").unwrap();
    for path in files.split(':') {
        let name = file_name(path);
        let Some((db, url)) = analyze(path) else {
            skipped.push(name);
            continue;
        };
        let db = &db;
        let (_, modules) = modules(db, path, &url);
        let contracts = modules.iter().flat_map(|module| module.all_contracts(db));
        for (index, &contract) in contracts.enumerate() {
            total += 1;
            let id = format!("{name}_{index}");
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| export_ir(db, contract)))
                .unwrap_or_else(|_| Err(Unsupported("PANIC".to_string())));
            match result {
                Ok(json) => {
                    serialized += 1;
                    fs::write(format!("{out}/{id}.json"), ir_text(&json)).unwrap();
                }
                Err(reason) => blockers.entry(reason.0).or_default().push(id),
            }
        }
    }
    for (reason, ids) in &blockers {
        println!("BLOCK {}\t{reason}", ids.join(" "));
    }
    println!("skipped (errors): {}", skipped.join(" "));
    println!("contracts: {serialized} of {total} serialized");
}

#[test]
#[ignore = "reads its input from the environment"]
fn sonatina_corpus() {
    let out = env::var("SONATINA_CORPUS_OUT").unwrap();
    let files = env::var("SONATINA_CORPUS").unwrap();
    let (mut skipped, mut uncompiled) = (Vec::new(), Vec::new());
    let (mut same_bytes, mut other_bytes) = (0, Vec::new());
    for path in files.split(':') {
        let name = file_name(path);
        let Some((db, url)) = analyze(path) else {
            skipped.push(name);
            continue;
        };
        let db = &db;
        let (ingot, modules) = modules(db, path, &url);
        if modules
            .iter()
            .all(|module| module.all_contracts(db).is_empty())
        {
            continue;
        }
        let compiled = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let (sidecars, code) = match ingot {
                Some(ingot) => (
                    emit_ingot_sonatina_sidecar(db, ingot, OptLevel::O1, None),
                    emit_ingot_sonatina_bytecode(db, ingot, OptLevel::O1, None),
                ),
                None => (
                    emit_module_sonatina_sidecar(db, modules[0], OptLevel::O1, None),
                    emit_module_sonatina_bytecode(db, modules[0], OptLevel::O1, None),
                ),
            };
            let suite = emit_test_suite_sonatina(
                db,
                &modules,
                OptLevel::O1,
                SonatinaTestOptions::default(),
                None,
            );
            Some((sidecars.ok()?, code.ok()?, suite.ok()?))
        }));
        let Ok(Some((sidecars, code, suite))) = compiled else {
            uncompiled.push(name);
            continue;
        };
        // The sidecar's compile records the lowering; the bytecode's does not.
        for (contract, sidecar) in &sidecars {
            let bytes = |section: &str| {
                sidecar
                    .sections
                    .iter()
                    .find(|s| s.name == section)
                    .map(|s| s.bytecode.clone())
            };
            if code.get(contract).is_some_and(|code| {
                bytes("init") == Some(hex(&code.deploy))
                    && bytes("runtime") == Some(hex(&code.runtime))
            }) {
                same_bytes += 1;
            } else {
                other_bytes.push(format!("{name}::{contract}"));
            }
        }
        let mut targets = Vec::new();
        for &contract in modules.iter().flat_map(|module| module.all_contracts(db)) {
            let contract_name = contract.name(db).unwrap().data(db).to_string();
            let Some(sidecar) = sidecars.get(&contract_name) else {
                continue;
            };
            let file = format!("{out}/{name}.{contract_name}.sidecar.json");
            fs::write(&file, serde_json::to_string(sidecar).unwrap()).unwrap();
            targets.push(format!("(\"{file}\", {})", interface(db, contract)));
        }
        let mut text = format!(
            "import Fe.Sonatina.Diff\n\ndef targets : IO (List Fe.EVM.Diff.Target) :=\n  \
             Fe.Sonatina.targetsOf \"{name}\"\n    [{}]\n",
            targets.join(",\n     ")
        );
        if let Some(suite) = suite {
            write!(
                text,
                "\n#eval do\n  Fe.EVM.Diff.main \"{name}\" (← targets)\n    (Fe.EVM.Diff.hex \"{}\")\n    [{}]\n",
                hex(&suite.bytecode),
                tests(&suite)
            )
            .unwrap();
        }
        writeln!(
            text,
            "\n#eval do Fe.EVM.Diff.generated \"{name}\" (← targets)"
        )
        .unwrap();
        fs::write(format!("{out}/{name}.lean"), text).unwrap();
    }
    println!("files skipped: {skipped:?}, not compiled: {uncompiled:?}");
    println!(
        "bytes: {same_bytes} contracts the same with and without the lowering record, other: {other_bytes:?}"
    );
}

/// The name of a file or ingot directory, without `.fe`.
fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap().trim_end_matches(".fe")
}

/// The analyzed file or ingot at `path` and its URL, or `None` if it has
/// errors or the analysis panics.
fn analyze(path: &str) -> Option<(DriverDataBase, Url)> {
    panic::catch_unwind(|| {
        let mut db = DriverDataBase::default();
        let (url, has_errors) = if Path::new(path).is_dir() {
            let url = Url::from_directory_path(path).unwrap();
            let has_errors = init_ingot(&mut db, &url) || {
                let ingot = db.workspace().containing_ingot(&db, url.clone()).unwrap();
                db.run_on_ingot(ingot).has_errors(&db)
            };
            (url, has_errors)
        } else {
            let url = Url::from_file_path(path).unwrap();
            let source = fs::read_to_string(path).unwrap();
            let file = db.workspace().touch(&mut db, url.clone(), Some(source));
            (url, db.run_on_top_mod(db.top_mod(file)).has_errors(&db))
        };
        (!has_errors).then_some((db, url))
    })
    .ok()
    .flatten()
}

/// The ingot at `path`, if it is a directory, and its modules, or the module
/// of the file.
fn modules<'db>(
    db: &'db DriverDataBase,
    path: &str,
    url: &Url,
) -> (Option<Ingot<'db>>, Vec<TopLevelMod<'db>>) {
    if Path::new(path).is_dir() {
        let ingot = db.workspace().containing_ingot(db, url.clone()).unwrap();
        let mut modules = ingot.all_modules(db).to_vec();
        modules.sort_by_key(|module| module.name(db));
        (Some(ingot), modules)
    } else {
        (None, vec![db.top_mod(db.workspace().get(db, url).unwrap())])
    }
}

/// The `Fe.EVM.Diff.Interface` of a contract: the argument types generated
/// calls are made from. An arm whose types have no ABI form gets arbitrary
/// words.
fn interface(db: &DriverDataBase, contract: Contract<'_>) -> String {
    let sol = resolve_lib_type_path(db, contract.scope(), "std::abi::Sol").unwrap();
    let canonical = |ty| {
        semantic_ty_to_abi_desc(db, ty)
            .map(|desc| desc.canonical_type)
            .unwrap_or_default()
    };
    let arms: Vec<_> = contract
        .recv_views(db)
        .flat_map(|recv| recv.arms(db))
        .filter_map(|arm| {
            let info = arm.abi_info(db, sol);
            let payable = arm.arm(db).is_some_and(|arm| arm.is_payable(db));
            let inputs = canonical(info.args_ty);
            Some(format!(
                "⟨{}, \"{inputs}\", {payable}⟩",
                info.selector_value?
            ))
        })
        .collect();
    let init_args = contract.init_args_ty(db);
    let init = if init_args == TyId::unit(db) {
        "()".to_string()
    } else {
        canonical(init_args)
    };
    let init_payable = contract.init(db).is_some_and(|init| init.is_payable(db));
    format!(
        "{{ init := \"{init}\", initPayable := {init_payable}, arms := [{}] }}",
        arms.join(", ")
    )
}

/// The `Fe.EVM.Diff.Test`s of a test suite.
fn tests(suite: &TestSuiteOutput) -> String {
    let tests: Vec<_> = suite
        .tests
        .iter()
        .map(|test| {
            let balance =
                BigUint::from_bytes_be(test.initial_balance.as_deref().unwrap_or_default());
            format!(
                "{{ name := \"{}\", selector := {}, balance := {balance} }}",
                test.name, test.selector
            )
        })
        .collect();
    tests.join(",\n   ")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Prints the normalized body of every instance the contracts of the file
/// `LEAN_EXPORT_DUMP` reach.
#[test]
#[ignore = "reads its input from the environment"]
fn dump_ir() {
    let source = fs::read_to_string(env::var("LEAN_EXPORT_DUMP").unwrap()).unwrap();
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("dump.fe".into(), &source);
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let db = &db;
    let mut pending: Vec<SemanticInstanceKey> = Vec::new();
    for &contract in top_mod.all_contracts(db) {
        pending.push(identity_semantic_instance_key(
            db,
            BodyOwner::ContractInit { contract },
        ));
        for (recv_idx, recv) in contract.recv_views(db).enumerate() {
            for (arm_idx, _) in recv.arms(db).enumerate() {
                let owner = BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: recv_idx as u32,
                    arm_idx: arm_idx as u32,
                };
                pending.push(identity_semantic_instance_key(db, owner));
            }
        }
    }
    let mut seen = Vec::new();
    while let Some(key) = pending.pop() {
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let name = match key.owner(db) {
            // A method of an impl has no path, only its name.
            BodyOwner::Func(func, mode) => {
                let path = func.scope().pretty_path(db);
                let name = func.name(db).to_opt().map(|name| name.data(db).clone());
                format!("{} mode={mode:?}", path.or(name).unwrap_or_default())
            }
            other => format!("{other:?}"),
        };
        println!("=== {name} subst={:?}", key.subst(db).generic_args(db));
        let normalized = match normalize_semantic_body(db, SemanticInstance::new(db, key)) {
            Ok(normalized) => normalized,
            Err(error) => {
                println!("    no body: {error:?}");
                continue;
            }
        };
        let body = &normalized.body;
        for (index, root) in body.roots.iter().enumerate() {
            let ty = root.ty.pretty_print(db);
            println!("root {index}: {:?} ty={ty}", root.kind);
        }
        for (index, value) in body.values.iter().enumerate() {
            let ty = value.ty.pretty_print(db);
            println!("v{index}: {:?} ty={ty}", value.definition);
        }
        for (index, block) in body.blocks.iter().enumerate() {
            let params: Vec<_> = block.params.iter().map(|param| param.index()).collect();
            println!("block {index} params={params:?}");
            for statement in &block.statements {
                println!("  {:?}", statement.kind);
                if let NStatementKind::Define { expr, .. } = &statement.kind {
                    match expr {
                        NExpr::Call { callee, .. } => pending.push(callee.key),
                        NExpr::Const(SConst::Ref(constant)) => pending.push(constant.instance(db)),
                        _ => {}
                    }
                }
            }
            println!("  -> {:?}", block.terminator.kind);
        }
    }
}

/// The fixtures of `crates/fe/tests/fixtures/fe_test` whose bodies hold
/// copies of lanes.
const LANE_FIXTURES: &[&str] = &[
    "assoc_type_lanes",
    "clear_place",
    "contract_field_packing",
    "effect_aliasing_lanes",
    "forwarded_projection_spaces",
    "lane_holds_before_halts",
    "lane_pattern_writes",
    "lane_projection_spaces",
    "library_state_index",
    "product_params",
    "sol_array_lanes",
    "storage_lane_grants",
    "storage_packing",
    "storage_packing_value_types",
    "trait_effect_lane_writes",
];

/// The export holds copies of lanes, and stores them back before successful
/// halts, where lowering does (`mir::lane_halt_points`): in the same
/// instances, and at the same calls and halts. Checked on the fixtures that
/// hold lanes and this crate's `lane_holds`, or on the files and ingots in
/// `LEAN_EXPORT_LANES`, separated by `:`. A change to where either holds a
/// copy fails here rather than leave a model that differs from the bytecode.
#[test]
fn lane_holds() {
    let root = env!("CARGO_MANIFEST_DIR");
    let default = LANE_FIXTURES
        .iter()
        .map(|name| format!("{root}/../fe/tests/fixtures/fe_test/{name}.fe"))
        .chain([format!("{root}/tests/fixtures/lane_holds.fe")])
        .collect::<Vec<_>>()
        .join(":");
    let files = env::var("LEAN_EXPORT_LANES").unwrap_or(default);
    let mut differences = String::new();
    for path in files.split(':') {
        let (db, url) = analyze(path).unwrap_or_else(|| panic!("{path} does not compile"));
        let db = &db;
        let (_, modules) = modules(db, path, &url);
        let mut ingot = IngotExport::new(db);
        let mut library = Library::default();
        for contract in modules.iter().flat_map(|module| module.all_contracts(db)) {
            if let Err(Unsupported(reason)) = ingot.export_contract(*contract, true, &mut library) {
                writeln!(differences, "{path}: not exported: {reason}").unwrap();
            }
        }
        let exported = ingot.lane_report();
        let mut holders = FxHashSet::default();
        let mut halts = FxHashSet::default();
        for &module in &modules {
            let package = mir::build_runtime_package(db, module).unwrap();
            for function in package.functions(db) {
                let instance = function.instance(db);
                let Some(key) = instance.key(db).semantic(db) else {
                    continue;
                };
                let Some((holds, points)) = mir::lane_halt_points(db, instance) else {
                    continue;
                };
                let key = key.key(db);
                if holds {
                    holders.insert(key);
                }
                for callee in points {
                    let callee = callee.map(|callee| callee.key(db).semantic(db));
                    let callee = match callee {
                        Some(Some(callee)) => Some(callee.key(db)),
                        Some(None) => {
                            writeln!(
                                differences,
                                "{path}: {} halts in a synthetic callee",
                                describe(db, key)
                            )
                            .unwrap();
                            continue;
                        }
                        None => None,
                    };
                    halts.insert((key, callee));
                }
            }
        }
        eprintln!(
            "{path}: {} instances hold lane copies, {} points store them back",
            holders.len(),
            halts.len()
        );
        for key in holders.symmetric_difference(&exported.holders) {
            let side = if holders.contains(key) {
                "lowering only"
            } else {
                "export only"
            };
            writeln!(
                differences,
                "{path}: {} holds a lane copy: {side}",
                describe(db, *key)
            )
            .unwrap();
        }
        for (key, callee) in halts.symmetric_difference(&exported.halts) {
            let side = if halts.contains(&(*key, *callee)) {
                "lowering only"
            } else {
                "export only"
            };
            let at = callee.map_or("its own halt".to_string(), |callee| {
                format!("the call of {}", describe(db, callee))
            });
            writeln!(
                differences,
                "{path}: {} stores back copies before {at}: {side}",
                describe(db, *key)
            )
            .unwrap();
        }
    }
    assert!(differences.is_empty(), "{differences}");
}

/// The function or entry `key` is an instance of, with its types.
fn describe(db: &DriverDataBase, key: SemanticInstanceKey<'_>) -> String {
    let owner = match key.owner(db) {
        BodyOwner::Func(func, _) => func
            .name(db)
            .to_opt()
            .map_or("fn".to_string(), |name| name.data(db).to_string()),
        owner => format!("{owner:?}"),
    };
    let args: Vec<_> = key
        .subst(db)
        .generic_args(db)
        .iter()
        .map(|arg| arg.pretty_print(db).to_string())
        .collect();
    match args.is_empty() {
        true => owner,
        false => format!("{owner}<{}>", args.join(", ")),
    }
}

//! Runs the exporter over many contracts, for `scripts/audit_exports.sh` of
//! fe-equivm-proofs. Both tests are ignored: they read their input from the
//! environment.

use std::{collections::BTreeMap, env, fmt::Write, fs, panic, path::Path};

use codegen::{
    OptLevel, SonatinaTestOptions, emit_ingot_sonatina_bytecode, emit_module_sonatina_bytecode,
    emit_test_suite_sonatina,
};
use common::InputDb;
use cranelift_entity::EntityRef;
use driver::{DriverDataBase, init_ingot};
use fe_lean_export::{Library, Unsupported, export_contract_with_entries};
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
    hir_def::HirIngot,
    test_db::HirAnalysisTestDb,
};
use num_bigint::BigUint;
use url::Url;

/// Exports every contract of the colon-separated files and ingot directories
/// in `LEAN_EXPORT_CORPUS`. Writes the exports of each file to
/// `LEAN_EXPORT_CORPUS_OUT/<file>.lean`, importing the modules
/// `LEAN_EXPORT_CORPUS_IMPORTS` names, each contract in a namespace so that
/// the file elaborates on its own, after the creation code of each contract
/// an export deploys, and prints the construct each other contract stopped at
/// as `BLOCK <ids>\t<reason>`.
///
/// Writes what each exported contract takes on trust to
/// `LEAN_EXPORT_CORPUS_OUT/<file>.trust`, each line after the contract's
/// namespace.
///
/// A file that compiles ends in `Corpus.targets`, the creation and runtime
/// bytecode, model and interface of each exported contract. A call of
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
        let name = path.rsplit('/').next().unwrap().trim_end_matches(".fe");
        let is_ingot = Path::new(path).is_dir();
        // A file or ingot with errors is skipped. A panic of the exporter is
        // reported as what stopped the contract.
        let analyzed = panic::catch_unwind(|| {
            let mut db = DriverDataBase::default();
            let (url, has_errors) = if is_ingot {
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
        });
        let Ok(Some((db, url))) = analyzed else {
            skipped.push(name);
            continue;
        };
        let db = &db;
        let ingot = is_ingot.then(|| db.workspace().containing_ingot(db, url.clone()).unwrap());
        let modules = match ingot {
            Some(ingot) => {
                let mut modules = ingot.all_modules(db).to_vec();
                modules.sort_by_key(|module| module.name(db));
                modules
            }
            None => vec![db.top_mod(db.workspace().get(db, &url).unwrap())],
        };
        let mut text = String::new();
        let mut trust = String::new();
        let mut library = Library::default();
        let mut models = Vec::new();
        let contracts = modules.iter().flat_map(|module| module.all_contracts(db));
        for (index, &contract) in contracts.enumerate() {
            total += 1;
            let id = format!("{name}_{index}");
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                export_contract_with_entries(db, contract, &mut library)
            }))
            .unwrap_or_else(|_| Err(Unsupported("PANIC".to_string())));
            match result {
                Ok(lean) => {
                    exported += 1;
                    write!(
                        text,
                        "\nnamespace Corpus.{id}\nsection\n{}\
                         set_option linter.unusedVariables false\n\n{}\nend\nend Corpus.{id}\n",
                        lean.uses_command("Corpus.Lib"),
                        lean.text
                    )
                    .unwrap();
                    for item in &lean.trusted {
                        writeln!(trust, "Corpus.{id} {item}").unwrap();
                    }
                    let name = contract.name(db).unwrap().data(db).to_string();
                    // The argument types generated calls are made from. An
                    // arm whose types have no ABI form gets arbitrary words.
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
                    let bytes = contract.code_address_space_slot_count(db) * 32;
                    models.push((
                        name,
                        format!("⟨Corpus.{id}.init_entry, entry Corpus.{id}.runtime, {bytes}⟩"),
                        format!(
                            "{{ init := \"{init}\", initPayable := {init_payable}, arms := [{}] }}",
                            arms.join(", ")
                        ),
                    ));
                }
                Err(reason) => blockers.entry(reason.0).or_default().push(id),
            }
        }
        if models.is_empty() {
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
                .map(|(contract, model, interface)| {
                    format!(
                        "{{ name := \"{contract}\", creation := Fe.EVM.Diff.hex \"{}\", \
                         runtime := Fe.EVM.Diff.hex \"{}\",\n     model := {model},\n     \
                         interface := {interface} }}",
                        hex(&code[contract].deploy),
                        hex(&code[contract].runtime)
                    )
                })
                .collect();
            write!(
                text,
                "\ndef Corpus.targets : List Fe.EVM.Diff.Target :=\n  [{}]\n",
                targets.join(",\n   ")
            )
            .unwrap();
            if let Some(suite) = suite {
                let tests: Vec<_> = suite
                    .tests
                    .iter()
                    .map(|test| {
                        let balance = BigUint::from_bytes_be(
                            test.initial_balance.as_deref().unwrap_or_default(),
                        );
                        format!(
                            "{{ name := \"{}\", selector := {}, balance := {balance} }}",
                            test.name, test.selector
                        )
                    })
                    .collect();
                write!(
                    text,
                    "\n#eval Fe.EVM.Diff.main \"{name}\" Corpus.targets\n  (Fe.EVM.Diff.hex \"{}\")\n  [{}]\n",
                    hex(&suite.bytecode),
                    tests.join(",\n   ")
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
        // The file's contracts share the definitions of core and std.
        let header = format!(
            "{imports}open Ethereum Ethereum.EVM Fe Fe.EVM\n\nnamespace Corpus.Lib\nsection\n\
             set_option linter.unusedVariables false\n\n{}\nend\nend Corpus.Lib\n",
            library.text()
        );
        fs::write(format!("{out}/{name}.lean"), header + &text).unwrap();
        fs::write(format!("{out}/{name}.trust"), trust).unwrap();
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

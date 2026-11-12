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
use fe_lean_export::{Unsupported, export_contract_with_entries};
use hir::{
    analysis::{
        semantic::{
            NExpr, NStatementKind, SConst, SemanticInstance, SemanticInstanceKey,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::ty_check::BodyOwner,
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
/// A file with tests ends in a call of `Fe.EVM.Diff.main` on the creation
/// and runtime bytecode of its exported contracts and the object its tests
/// compile to, which runs each test and compares every call it makes on such
/// a contract with the contract's model.
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
        let mut models = Vec::new();
        let contracts = modules.iter().flat_map(|module| module.all_contracts(db));
        for (index, &contract) in contracts.enumerate() {
            total += 1;
            let id = format!("{name}_{index}");
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                export_contract_with_entries(db, contract)
            }))
            .unwrap_or_else(|_| Err(Unsupported("PANIC".to_string())));
            match result {
                Ok(lean) => {
                    exported += 1;
                    write!(
                        text,
                        "\nnamespace Corpus.{id}\nsection\n\
                         set_option linter.unusedVariables false\n\n{lean}\nend\nend Corpus.{id}\n"
                    )
                    .unwrap();
                    let name = contract.name(db).unwrap().data(db).to_string();
                    models.push((name, id, contract.code_address_space_slot_count(db) * 32));
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
        if let Ok(Some((code, _))) = &compiled {
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
        }
        match compiled {
            Ok(Some((code, Some(suite)))) => {
                let targets: Vec<_> = models
                    .iter()
                    .map(|(contract, id, bytes)| {
                        let creation = hex(&code[contract].deploy);
                        let runtime = hex(&code[contract].runtime);
                        format!(
                            "{{ name := \"{contract}\", creation := Fe.EVM.Diff.hex \"{creation}\", \
                             runtime := Fe.EVM.Diff.hex \"{runtime}\", \
                             model := ⟨Corpus.{id}.init_entry, entry Corpus.{id}.runtime, {bytes}⟩ }}"
                        )
                    })
                    .collect();
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
                    "\n#eval Fe.EVM.Diff.main \"{name}\"\n  [{}]\n  (Fe.EVM.Diff.hex \"{}\")\n  [{}]\n",
                    targets.join(",\n   "),
                    hex(&suite.bytecode),
                    tests.join(",\n   ")
                )
                .unwrap();
            }
            Ok(Some(_)) => {}
            _ => uncompiled.push(name),
        }
        let header = format!("{imports}open Ethereum Ethereum.EVM Fe Fe.EVM\n");
        fs::write(format!("{out}/{name}.lean"), header + &text).unwrap();
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

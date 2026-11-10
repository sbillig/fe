//! Runs the exporter over many contracts, for `scripts/audit_exports.sh` of
//! fe-equivm-proofs. Both tests are ignored: they read their input from the
//! environment.

use std::{collections::BTreeMap, env, fs, panic};

use cranelift_entity::EntityRef;
use fe_lean_export::export_contract_with_entries;
use hir::{
    analysis::{
        semantic::{
            NExpr, NStatementKind, SConst, SemanticInstance, SemanticInstanceKey,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::ty_check::BodyOwner,
    },
    test_db::HirAnalysisTestDb,
};

/// Exports every contract of the colon-separated files in
/// `LEAN_EXPORT_CORPUS`. Writes each export to
/// `LEAN_EXPORT_CORPUS_OUT/<file>_<n>.lean`, wrapped in a namespace so that it
/// elaborates on its own, and prints the construct each other contract
/// stopped at as `BLOCK <ids>\t<reason>`.
#[test]
#[ignore = "reads its input from the environment"]
fn export_corpus() {
    let out = env::var("LEAN_EXPORT_CORPUS_OUT").unwrap();
    let mut blockers = BTreeMap::<String, Vec<String>>::new();
    let (mut exported, mut skipped, mut total) = (0, Vec::new(), 0);
    let files = env::var("LEAN_EXPORT_CORPUS").unwrap();
    for path in files.split(':') {
        let source = fs::read_to_string(path).unwrap();
        let name = path.rsplit('/').next().unwrap().trim_end_matches(".fe");
        // A file with diagnostics is skipped.
        let results = panic::catch_unwind(|| {
            let mut db = HirAnalysisTestDb::default();
            let file = db.new_stand_alone("corpus.fe".into(), &source);
            let (top_mod, _) = db.top_mod(file);
            db.assert_no_diags(top_mod);
            let contracts = top_mod.all_contracts(&db);
            let export = |contract: &_| export_contract_with_entries(&db, *contract);
            contracts.iter().map(export).collect::<Vec<_>>()
        });
        let Ok(results) = results else {
            skipped.push(name);
            continue;
        };
        for (index, result) in results.into_iter().enumerate() {
            total += 1;
            let id = format!("{name}_{index}");
            match result {
                Ok(lean) => {
                    exported += 1;
                    let text = format!(
                        "import Fe.EVM.Storage\nopen Ethereum Ethereum.EVM Fe Fe.EVM\n\
                         namespace Corpus.{id}\nsection\n\
                         set_option linter.unusedVariables false\n\n{lean}\nend\nend Corpus.{id}\n"
                    );
                    fs::write(format!("{out}/{id}.lean"), text).unwrap();
                }
                Err(reason) => blockers.entry(reason.0).or_default().push(id),
            }
        }
    }
    println!("contracts: {total}, exported: {exported}, files skipped: {skipped:?}");
    for (reason, ids) in &blockers {
        let reason: String = reason.chars().take(400).collect();
        println!("BLOCK {}\t{reason}", ids.join(","));
    }
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
            BodyOwner::Func(func, mode) => {
                format!("{:?} mode={mode:?}", func.scope().pretty_path(db))
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

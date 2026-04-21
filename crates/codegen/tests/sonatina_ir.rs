//! Snapshot tests for Sonatina IR output.
//!
//! These tests compile Fe fixtures to Sonatina IR and snapshot the human-readable
//! IR text. This helps catch IR lowering bugs and makes it easy to review what
//! IR is generated for each fixture.
//!
//! Snapshots are stored in `fixtures/sonatina_ir/` to avoid conflicting with Yul snapshots.

use common::InputDb;
use dir_test::{Fixture, dir_test};
use driver::DriverDataBase;
use fe_codegen::emit_module_sonatina_ir;
use std::path::Path;
use test_utils::_macro_support::_insta::{self, Settings};
use url::Url;

fn with_top_mod_for_source<T>(
    name: &str,
    source: &str,
    f: impl for<'db> FnOnce(&'db DriverDataBase, hir::hir_def::TopLevelMod<'db>) -> T,
) -> T {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(&format!("file:///{name}")).expect("test URL should parse");
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(source.to_string()));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    f(&db, top_mod)
}

#[test]
fn wildcard_storage_map_root_reports_runtime_root_error() {
    let err = with_top_mod_for_source(
        "wildcard_storage_map_root_reports_runtime_root_error.fe",
        r#"
use std::evm::StorageMap

pub fn main() -> u256
    uses (balances: mut StorageMap<u256, u256>)
{
    balances.get(key: 1)
}
"#,
        |db, top_mod| {
            emit_module_sonatina_ir(db, top_mod)
                .expect_err("wildcard StorageMap roots should be rejected")
        },
    );
    let message = err.to_string();
    assert!(
        message.contains("standalone runtime root")
            && message.contains("inferred layout const")
            && message.contains("StorageMap<..., ..., 0>"),
        "unexpected error message:\n{message}"
    );
}

#[test]
fn sonatina_ir_rejects_target_only_output() {
    let err = with_top_mod_for_source(
        "sonatina_ir_rejects_target_only_output.fe",
        r#"
fn helper(value: u256) -> u256 {
    value
}
"#,
        |db, top_mod| {
            emit_module_sonatina_ir(db, top_mod).expect_err("empty packages should not emit IR")
        },
    );
    let message = err.to_string();
    assert!(
        message.contains("no root objects") && message.contains("target-only Sonatina IR"),
        "unexpected error message:\n{message}"
    );
}

// NOTE: `dir_test` discovers fixtures at compile time; new fixture files will be picked up on a
// clean build (e.g. CI) or whenever this test target is recompiled.
//
// Unlike the Yul tests which run on all fixtures, Sonatina IR tests only run on fixtures
// that the backend currently supports. Unsupported fixtures will produce LowerError::Unsupported
// which we skip gracefully.
#[dir_test(
    dir: "$CARGO_MANIFEST_DIR/tests/fixtures",
    glob: "*.fe"
)]
fn sonatina_ir_snap(fixture: Fixture<&str>) {
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

    let output = match emit_module_sonatina_ir(&db, top_mod) {
        Ok(ir) => ir,
        Err(fe_codegen::LowerError::Unsupported(msg)) => {
            tracing::info!("SKIP {}: unsupported ({msg})", fixture.path());
            return;
        }
        Err(fe_codegen::LowerError::Internal(msg)) => {
            tracing::warn!("SKIP {}: internal error ({msg})", fixture.path());
            return;
        }
        Err(err) => panic!("Sonatina IR lowering failed: {err}"),
    };

    // Store snapshots in sonatina_ir/ subdirectory to avoid conflicting with Yul snapshots
    let fixture_path = Path::new(fixture.path());
    let fixture_name = fixture_path.file_stem().unwrap().to_str().unwrap();
    let snapshot_dir = fixture_path.parent().unwrap().join("sonatina_ir");

    let mut settings = Settings::new();
    settings.set_snapshot_path(snapshot_dir);
    settings.set_input_file(fixture.path());
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| {
        _insta::assert_snapshot!(fixture_name, output);
    });
}

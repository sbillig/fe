//! Shared helpers for benchmark-style Fe/Solidity comparisons.
//!
//! The `fe bench` CLI and heavier differential tests have different execution
//! models. Keep compilation and small reporting helpers here so they can share
//! infrastructure without forcing stateful scenarios into the simple bench
//! fixture format.

use codegen::OptLevel;
use common::InputDb;
use driver::DriverDataBase;
use std::fmt;
use url::Url;

/// Compile Solidity source to deploy bytecode hex string.
pub fn compile_solidity(
    source: &str,
    contract_name: &str,
    optimize: bool,
    solc_path: Option<&str>,
) -> Result<String, String> {
    solc_runner::compile_solidity(contract_name, source, optimize, solc_path)
        .map(|bc| bc.bytecode)
        .map_err(|e| {
            let label = if optimize { "sol+opt" } else { "sol" };
            format!("{label} compile error: {}", e.0)
        })
}

/// Compile Fe source to bytecode via Sonatina backend. Returns deploy bytecode as raw bytes.
pub fn compile_fe_sonatina(
    fe_source: &str,
    name: &str,
    contract_name: &str,
) -> Result<Vec<u8>, String> {
    let contract_name_owned = contract_name.to_string();
    let name_owned = name.to_string();
    with_fe_ingot(fe_source, name, move |db, ingot| {
        let diags = db.run_on_ingot(ingot);
        if !diags.is_empty() {
            diags.emit(db);
            return Err(format!("fe/sonatina diagnostics for {name_owned}"));
        }

        let mut map = codegen::emit_ingot_sonatina_bytecode(
            db,
            ingot,
            OptLevel::O2,
            Some(&contract_name_owned),
        )
        .map_err(|err| format!("fe/sonatina emit error for {name_owned}: {err}"))?;
        map.remove(&contract_name_owned)
            .map(|bc| bc.deploy)
            .ok_or_else(|| {
                format!("fe/sonatina: no bytecode emitted for contract `{contract_name_owned}`")
            })
    })?
}

/// Format a gas value for tabular reports.
pub fn fmt_gas(gas: u64) -> String {
    gas.to_string()
}

/// Format a percentage gas delta as `(lhs - rhs) / rhs`.
pub fn fmt_delta_pct(lhs: u64, rhs: u64) -> String {
    if rhs == 0 {
        return "-".to_string();
    }
    let pct = ((lhs as f64 - rhs as f64) / rhs as f64) * 100.0;
    format!("{pct:+.1}%")
}

/// One row in a two-way gas comparison.
pub struct GasComparisonRow {
    pub label: String,
    pub left_gas: u64,
    pub right_gas: u64,
}

impl GasComparisonRow {
    pub fn new(label: impl Into<String>, left_gas: u64, right_gas: u64) -> Self {
        Self {
            label: label.into(),
            left_gas,
            right_gas,
        }
    }

    pub fn delta(&self) -> i64 {
        self.left_gas as i64 - self.right_gas as i64
    }

    pub fn delta_pct(&self) -> f64 {
        (self.delta() as f64 / self.right_gas as f64) * 100.0
    }
}

/// Print a compact two-column gas comparison table.
pub fn print_gas_comparison_table(
    rows: &[GasComparisonRow],
    label_header: &str,
    left_header: &str,
    right_header: &str,
    delta_header: &str,
) {
    println!(
        "{:<8} {:>12} {:>12} {:>12}",
        label_header, left_header, right_header, delta_header
    );
    println!("{}", "-".repeat(50));
    for row in rows {
        println!(
            "{:<8} {:>12} {:>12} {:>+12} ({:+.1}%)",
            row.label,
            row.left_gas,
            row.right_gas,
            row.delta(),
            row.delta_pct(),
        );
    }
}

impl fmt::Debug for GasComparisonRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GasComparisonRow")
            .field("label", &self.label)
            .field("left_gas", &self.left_gas)
            .field("right_gas", &self.right_gas)
            .field("delta", &self.delta())
            .field("delta_pct", &self.delta_pct())
            .finish()
    }
}

/// Set up a temp ingot and run a callback with `(db, ingot)`.
///
/// Writes `fe.toml` + `src/lib.fe` to a temp directory, then calls `init_ingot`
/// so that `std` and `core` are resolved correctly.
fn with_fe_ingot<T>(
    fe_source: &str,
    name: &str,
    f: impl for<'db> FnOnce(&'db DriverDataBase, hir::Ingot<'db>) -> T,
) -> Result<T, String> {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("fe_bench_{name}_"))
        .tempdir()
        .map_err(|e| format!("create temp ingot dir for {name}: {e}"))?;
    std::fs::create_dir_all(tmp.path().join("src"))
        .map_err(|e| format!("create temp ingot dir for {name}: {e}"))?;
    std::fs::write(
        tmp.path().join("fe.toml"),
        format!("[ingot]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
    )
    .map_err(|e| format!("write fe.toml for {name}: {e}"))?;
    std::fs::write(tmp.path().join("src").join("lib.fe"), fe_source)
        .map_err(|e| format!("write lib.fe for {name}: {e}"))?;

    let ingot_url = Url::from_directory_path(tmp.path())
        .map_err(|_| format!("non-utf8 temp ingot path for {name}"))?;
    let mut db = DriverDataBase::default();
    let had_errors = driver::init_ingot(&mut db, &ingot_url);
    if had_errors {
        return Err(format!("fe ingot init errors for {name}"));
    }

    let ingot = db
        .workspace()
        .containing_ingot(&db, ingot_url)
        .ok_or_else(|| format!("no containing ingot for {name}"))?;
    Ok(f(&db, ingot))
}

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
) -> Option<String> {
    match solc_runner::compile_solidity(contract_name, source, optimize, solc_path) {
        Ok(bc) => Some(bc.bytecode),
        Err(e) => {
            let label = if optimize { "sol+opt" } else { "sol" };
            eprintln!("  {label} compile error: {}", e.0);
            None
        }
    }
}

/// Compile Fe source to bytecode via Sonatina backend. Returns deploy bytecode as raw bytes.
pub fn compile_fe_sonatina(fe_source: &str, name: &str, contract_name: &str) -> Option<Vec<u8>> {
    let contract_name = contract_name.to_string();
    with_fe_ingot(fe_source, name, move |db, ingot| {
        let diags = db.run_on_ingot(ingot);
        if !diags.is_empty() {
            eprintln!("  fe/sonatina diagnostics for {name}:");
            diags.emit(db);
            return None;
        }

        match codegen::emit_ingot_sonatina_bytecode(db, ingot, OptLevel::O2, Some(&contract_name)) {
            Ok(mut map) => map.remove(&contract_name).map(|bc| bc.deploy),
            Err(err) => {
                eprintln!("  fe/sonatina emit error for {name}: {err}");
                None
            }
        }
    })
    .flatten()
}

/// Format an optional gas value for tabular reports.
pub fn fmt_gas(gas: Option<u64>) -> String {
    match gas {
        Some(g) => g.to_string(),
        None => "-".to_string(),
    }
}

/// Format a percentage gas delta as `(lhs - rhs) / rhs`.
pub fn fmt_delta_pct(lhs: Option<u64>, rhs: Option<u64>) -> String {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) if rhs > 0 => {
            let pct = ((lhs as f64 - rhs as f64) / rhs as f64) * 100.0;
            format!("{pct:+.1}%")
        }
        _ => "-".to_string(),
    }
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
) -> Option<T> {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("fe_bench_{name}_"))
        .tempdir()
        .ok()?;
    std::fs::create_dir_all(tmp.path().join("src")).ok()?;
    std::fs::write(
        tmp.path().join("fe.toml"),
        format!("[ingot]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
    )
    .ok()?;
    std::fs::write(tmp.path().join("src").join("lib.fe"), fe_source).ok()?;

    let ingot_url = Url::from_directory_path(tmp.path()).ok()?;
    let mut db = DriverDataBase::default();
    let had_errors = driver::init_ingot(&mut db, &ingot_url);
    if had_errors {
        eprintln!("  fe ingot init errors for {name}");
        return None;
    }

    let ingot = db.workspace().containing_ingot(&db, ingot_url)?;
    Some(f(&db, ingot))
}

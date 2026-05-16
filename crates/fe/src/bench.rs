//! Gas benchmarking: compares Fe (Sonatina) against Solidity.
//!
//! Discovers paired `.fe` / `.sol` fixture files with a `.toml` manifest,
//! compiles each backend, deploys and calls them, and reports per-function
//! gas consumption.

use std::fmt::Write as _;
use std::fs;

use crate::bench_support::{compile_fe_sonatina, compile_solidity, fmt_delta_pct, fmt_gas};
use camino::{Utf8Path, Utf8PathBuf};
use contract_harness::{ExecutionOptions, RuntimeInstance};
use ethers_core::abi::AbiParser;

/// A single benchmark fixture: paired Fe + Solidity sources with a call manifest.
struct BenchFixture {
    name: String,
    fe_source: String,
    sol_source: String,
    contract_name: String,
    calls: Vec<BenchCall>,
}

/// A function call to benchmark.
struct BenchCall {
    /// Solidity-style function signature, e.g. `"add(uint256,uint256)"`.
    signature: String,
    /// Hex-encoded argument values (no 0x prefix), each padded to 32 bytes.
    args: Vec<String>,
}

/// Gas measurement for a single function call across all backends.
struct BenchResult {
    fixture: String,
    function: String,
    fe_sonatina_gas: Option<u64>,
    sol_gas: Option<u64>,
    sol_opt_gas: Option<u64>,
}

/// TOML manifest deserialized from `<fixture>.toml`.
#[derive(serde::Deserialize)]
struct Manifest {
    contract: String,
    #[serde(default)]
    calls: Vec<ManifestCall>,
}

#[derive(serde::Deserialize)]
struct ManifestCall {
    function: String,
    #[serde(default)]
    args: Vec<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_benchmarks(
    path: &Utf8Path,
    filter: Option<&str>,
    solc: Option<&str>,
    output: Option<&Utf8Path>,
) -> Result<(), String> {
    let path = resolve_fixtures_dir(path)?;
    let fixtures = discover_fixtures(&path, filter)?;
    if fixtures.is_empty() {
        return Err(format!("no benchmark fixtures found in {path}"));
    }

    println!("Found {} benchmark fixture(s)\n", fixtures.len());

    let mut all_results: Vec<BenchResult> = Vec::new();

    for fixture in &fixtures {
        println!("--- {} ---", fixture.name);

        // 1. Compile Solidity (unoptimized + optimized)
        let sol_bytecode =
            compile_solidity(&fixture.sol_source, &fixture.contract_name, false, solc);
        let sol_opt_bytecode =
            compile_solidity(&fixture.sol_source, &fixture.contract_name, true, solc);

        // 2. Compile Fe via Sonatina backend
        let fe_sonatina_bytecode =
            compile_fe_sonatina(&fixture.fe_source, &fixture.name, &fixture.contract_name);

        // 3. Deploy all variants and measure gas per call
        for call in &fixture.calls {
            let calldata = match encode_calldata(&call.signature, &call.args) {
                Ok(cd) => cd,
                Err(err) => {
                    eprintln!("  skip {}: {err}", call.signature);
                    continue;
                }
            };

            let fe_sonatina_gas = measure_call_bytes(&fe_sonatina_bytecode, &calldata);
            let sol_gas = measure_call(&sol_bytecode, &calldata);
            let sol_opt_gas = measure_call(&sol_opt_bytecode, &calldata);

            let fn_name = call.signature.split('(').next().unwrap_or(&call.signature);
            all_results.push(BenchResult {
                fixture: fixture.name.clone(),
                function: fn_name.to_string(),
                fe_sonatina_gas,
                sol_gas,
                sol_opt_gas,
            });
        }
    }

    // Print results
    print_table(&all_results);

    // Write CSV if requested
    if let Some(out_dir) = output {
        write_csv(&all_results, out_dir)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Fixture discovery
// ---------------------------------------------------------------------------

/// If `path` doesn't exist, try common locations relative to the repo root.
fn resolve_fixtures_dir(path: &Utf8Path) -> Result<Utf8PathBuf, String> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // When running from the repo root, check crates/fe/<path>
    let under_crate = Utf8PathBuf::from("crates/fe").join(path);
    if under_crate.exists() {
        return Ok(under_crate);
    }

    Err(format!("fixtures directory does not exist: {path}"))
}

fn discover_fixtures(dir: &Utf8Path, filter: Option<&str>) -> Result<Vec<BenchFixture>, String> {
    if !dir.exists() {
        return Err(format!("fixtures directory does not exist: {dir}"));
    }

    let mut fixtures = Vec::new();
    let mut toml_files: Vec<_> = fs::read_dir(dir.as_std_path())
        .map_err(|e| format!("failed to read {dir}: {e}"))?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
        .collect();
    toml_files.sort_by_key(|e| e.file_name());

    for entry in toml_files {
        let path = Utf8PathBuf::from_path_buf(entry.path())
            .map_err(|p| format!("non-utf8 path: {}", p.display()))?;
        let stem = path
            .file_stem()
            .ok_or_else(|| format!("no file stem: {path}"))?;

        if let Some(f) = filter
            && !stem.contains(f)
        {
            continue;
        }

        let fe_path = dir.join(format!("{stem}.fe"));
        let sol_path = dir.join(format!("{stem}.sol"));

        if !fe_path.exists() {
            eprintln!("warning: skipping {stem} — missing {fe_path}");
            continue;
        }
        if !sol_path.exists() {
            eprintln!("warning: skipping {stem} — missing {sol_path}");
            continue;
        }

        let manifest_str =
            fs::read_to_string(path.as_std_path()).map_err(|e| format!("read {path}: {e}"))?;
        let manifest: Manifest =
            toml::from_str(&manifest_str).map_err(|e| format!("parse {path}: {e}"))?;

        let fe_source = fs::read_to_string(fe_path.as_std_path())
            .map_err(|e| format!("read {fe_path}: {e}"))?;
        let sol_source = fs::read_to_string(sol_path.as_std_path())
            .map_err(|e| format!("read {sol_path}: {e}"))?;

        let calls = manifest
            .calls
            .into_iter()
            .map(|c| BenchCall {
                signature: c.function,
                args: c.args,
            })
            .collect();

        fixtures.push(BenchFixture {
            name: stem.to_string(),
            fe_source,
            sol_source,
            contract_name: manifest.contract,
            calls,
        });
    }

    Ok(fixtures)
}

// ---------------------------------------------------------------------------
// Execution / measurement
// ---------------------------------------------------------------------------

/// Encode calldata from a function signature and string arguments.
///
/// Argument format depends on the ABI type: integers are decimal, addresses
/// are hex (`0x…`), bools are `true`/`false`.
fn encode_calldata(signature: &str, args: &[String]) -> Result<Vec<u8>, String> {
    let function = AbiParser::default()
        .parse_function(signature)
        .map_err(|e| format!("bad signature `{signature}`: {e}"))?;

    if args.len() != function.inputs.len() {
        return Err(format!(
            "signature `{signature}` expects {} arg(s), got {}",
            function.inputs.len(),
            args.len(),
        ));
    }

    let tokens: Vec<ethers_core::abi::Token> = args
        .iter()
        .zip(function.inputs.iter())
        .map(|(val, param)| parse_arg(val, &param.kind))
        .collect::<Result<Vec<_>, _>>()?;

    let encoded = function
        .encode_input(&tokens)
        .map_err(|e| format!("encode error: {e}"))?;
    Ok(encoded)
}

/// Parse a string argument into an ABI token based on the expected param type.
fn parse_arg(
    val: &str,
    kind: &ethers_core::abi::ParamType,
) -> Result<ethers_core::abi::Token, String> {
    use ethers_core::abi::{ParamType, Token};
    match kind {
        ParamType::Uint(bits) => {
            let n = ethers_core::types::U256::from_dec_str(val)
                .map_err(|e| format!("cannot parse `{val}` as uint: {e}"))?;
            if n.bits() > *bits {
                return Err(format!("`{val}` does not fit in uint{bits}"));
            }
            Ok(Token::Uint(n))
        }
        ParamType::Int(bits) => {
            let n = ethers_core::types::I256::from_dec_str(val)
                .map_err(|e| format!("cannot parse `{val}` as int: {e}"))?;
            if n.bits() as usize > *bits {
                return Err(format!("`{val}` does not fit in int{bits}"));
            }
            Ok(Token::Int(n.into_raw()))
        }
        ParamType::Bool => {
            let b: bool = val
                .parse()
                .map_err(|e| format!("cannot parse `{val}` as bool: {e}"))?;
            Ok(Token::Bool(b))
        }
        ParamType::Address => {
            let addr: ethers_core::types::Address = val
                .parse()
                .map_err(|e| format!("cannot parse `{val}` as address: {e}"))?;
            Ok(Token::Address(addr))
        }
        _ => Err(format!("unsupported param type: {kind}")),
    }
}

/// Deploy a contract from hex-encoded init bytecode and call it.
fn measure_call(bytecode_hex: &Option<String>, calldata: &[u8]) -> Option<u64> {
    let hex = bytecode_hex.as_ref()?;
    let mut instance = RuntimeInstance::deploy(hex).ok()?;
    let result = instance
        .call_raw(calldata, ExecutionOptions::default())
        .ok()?;
    Some(result.gas_used)
}

/// Deploy a contract from raw bytes and call it.
fn measure_call_bytes(bytecode: &Option<Vec<u8>>, calldata: &[u8]) -> Option<u64> {
    let bytes = bytecode.as_ref()?;
    let hex_str = hex::encode(bytes);
    let mut instance = RuntimeInstance::deploy(&hex_str).ok()?;
    let result = instance
        .call_raw(calldata, ExecutionOptions::default())
        .ok()?;
    Some(result.gas_used)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_table(results: &[BenchResult]) {
    if results.is_empty() {
        println!("No results.");
        return;
    }

    // Header
    println!(
        "{:<20} {:<12} {:>10} {:>10} {:>10} {:>10}",
        "Fixture", "Function", "Sonatina", "Sol", "Sol+O", "vs Sol+O"
    );
    println!("{}", "-".repeat(74));

    for r in results {
        let delta = fmt_delta_pct(r.fe_sonatina_gas, r.sol_opt_gas);
        println!(
            "{:<20} {:<12} {:>10} {:>10} {:>10} {:>10}",
            r.fixture,
            r.function,
            fmt_gas(r.fe_sonatina_gas),
            fmt_gas(r.sol_gas),
            fmt_gas(r.sol_opt_gas),
            delta,
        );
    }
}

fn write_csv(results: &[BenchResult], out_dir: &Utf8Path) -> Result<(), String> {
    fs::create_dir_all(out_dir.as_std_path()).map_err(|e| format!("create dir {out_dir}: {e}"))?;

    let path = out_dir.join("gas_benchmark.csv");
    let mut csv = String::new();
    writeln!(
        csv,
        "fixture,function,fe_sonatina,sol,sol_opt,delta_fe_sonatina_vs_sol_opt_pct"
    )
    .unwrap();

    for r in results {
        let delta = match (r.fe_sonatina_gas, r.sol_opt_gas) {
            (Some(fe), Some(sol)) if sol > 0 => {
                format!("{:.2}", ((fe as f64 - sol as f64) / sol as f64) * 100.0)
            }
            _ => String::new(),
        };
        writeln!(
            csv,
            "{},{},{},{},{},{}",
            r.fixture,
            r.function,
            fmt_gas(r.fe_sonatina_gas),
            fmt_gas(r.sol_gas),
            fmt_gas(r.sol_opt_gas),
            delta,
        )
        .unwrap();
    }

    fs::write(path.as_std_path(), &csv).map_err(|e| format!("write csv: {e}"))?;
    println!(
        "\nCSV report written to {}",
        out_dir.join("gas_benchmark.csv")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_arg;
    use ethers_core::{
        abi::{ParamType, Token},
        types::U256,
    };

    #[test]
    fn parse_signed_int_uses_full_word_twos_complement() {
        assert_eq!(
            parse_arg("-1", &ParamType::Int(256)).unwrap(),
            Token::Int(U256::MAX)
        );
        assert_eq!(
            parse_arg("-2", &ParamType::Int(256)).unwrap(),
            Token::Int(U256::MAX - U256::from(1u8))
        );
        assert_eq!(
            parse_arg("42", &ParamType::Int(256)).unwrap(),
            Token::Int(U256::from(42u8))
        );
    }

    #[test]
    fn parse_arg_rejects_values_outside_declared_width() {
        assert!(parse_arg("255", &ParamType::Uint(8)).is_ok());
        assert!(parse_arg("256", &ParamType::Uint(8)).is_err());
        assert!(parse_arg("127", &ParamType::Int(8)).is_ok());
        assert!(parse_arg("-128", &ParamType::Int(8)).is_ok());
        assert!(parse_arg("128", &ParamType::Int(8)).is_err());
        assert!(parse_arg("-129", &ParamType::Int(8)).is_err());
    }
}

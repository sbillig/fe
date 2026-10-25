use std::{fmt, fs, sync::Arc, time::Instant};

use camino::Utf8PathBuf;
use codegen::{OptLevel, emit_test_suite_wasm};
use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;
use wasmi::{Caller, Config, Engine, Linker, Module, Store};

use super::{
    CompiledTest, ReportContext, SingleTestJob, SuitePreparation, TestEmitSelection, TestOutcome,
    TestResult, default_test_emit_dir, emit_with_catch_unwind, test_emit_stem,
    write_test_emit_artifact,
};

/// The interpreter fuel each test may use, about one unit per instruction.
const WASM_TEST_FUEL: u64 = 20_000_000_000;
/// Interpreter call depth and value stack bytes, comparable to a native
/// thread's stack.
const WASM_TEST_RECURSION_DEPTH: usize = 1 << 16;
const WASM_TEST_VALUE_STACK_BYTES: usize = 128 << 20;
/// Output kept from `putchar`, beyond which it is truncated.
const WASM_TEST_OUTPUT_LIMIT: usize = 1 << 20;
const WASM_TEST_DIAGNOSTIC_LIMIT: usize = 4096;

/// A suite's compiled module, shared by its tests.
struct WasmSuite {
    engine: Engine,
    module: Module,
}

impl fmt::Debug for WasmSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmSuite").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub(super) struct WasmTestCase {
    name: String,
    export: String,
    suite: Arc<WasmSuite>,
}

#[derive(Default)]
struct HostState {
    output: Vec<u8>,
    truncated: bool,
}

impl WasmTestCase {
    /// Runs the test in a fresh instance. The host provides the C stream
    /// functions `std::io` imports: `getchar` reads end of input.
    pub(super) fn run(&self, report: Option<&ReportContext>) -> TestOutcome {
        let started = Instant::now();
        let mut store = Store::new(&self.suite.engine, HostState::default());
        let result = store
            .set_fuel(WASM_TEST_FUEL)
            .and_then(|()| {
                let mut linker = Linker::new(&self.suite.engine);
                linker.func_wrap(
                    "env",
                    "putchar",
                    |mut caller: Caller<'_, HostState>, c: i32| -> i32 {
                        let state = caller.data_mut();
                        if state.output.len() < WASM_TEST_OUTPUT_LIMIT {
                            state.output.push(c as u8);
                        } else {
                            state.truncated = true;
                        }
                        c & 0xff
                    },
                )?;
                linker.func_wrap("env", "getchar", || -> i32 { -1 })?;
                // No CPU clock is available, which a negative reading reports.
                linker.func_wrap("env", "clock", || -> i64 { -1 })?;
                let instance = linker.instantiate_and_start(&mut store, &self.suite.module)?;
                instance
                    .get_typed_func::<(), ()>(&store, &self.export)?
                    .call(&mut store, ())
            })
            .map_err(|err| err.to_string());
        let state = store.data();
        if let Some(report) = report {
            let dir = report
                .root_dir
                .join(format!("artifacts/wasm/{}", self.export));
            let _ = fs::create_dir_all(&dir);
            let _ = fs::write(dir.join("stdout.txt"), &state.output);
        }
        let error_message = result.err().map(|err| {
            let output = &state.output[..state.output.len().min(WASM_TEST_DIAGNOSTIC_LIMIT)];
            format!(
                "wasm test failed: {err}\nstdout:\n{}{}",
                String::from_utf8_lossy(output),
                if state.truncated || state.output.len() > WASM_TEST_DIAGNOSTIC_LIMIT {
                    "\n[output truncated]"
                } else {
                    ""
                }
            )
        });
        TestOutcome {
            result: TestResult {
                name: self.name.clone(),
                passed: error_message.is_none(),
                error_message,
            },
            logs: Vec::new(),
            trace: None,
            elapsed: started.elapsed(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_tests(
    db: &DriverDataBase,
    top_mods: Vec<TopLevelMod<'_>>,
    source_path: &Utf8PathBuf,
    suite: &str,
    suite_key: &str,
    filter: Option<&str>,
    opt_level: OptLevel,
    emit: TestEmitSelection,
    report: Option<&ReportContext>,
    output: &mut String,
) -> SuitePreparation {
    let mut artifact_output = String::new();
    let prepared = emit_with_catch_unwind(
        || -> Result<Vec<SingleTestJob>, String> {
            let out_dir = (!emit.is_empty())
                .then(|| default_test_emit_dir(source_path))
                .transpose()?;
            let Some(compiled) = emit_test_suite_wasm(
                db,
                &top_mods,
                opt_level,
                filter,
                emit.ir || report.is_some(),
                emit.rmir || report.is_some(),
            )
            .map_err(|err| err.to_string())?
            else {
                return Ok(Vec::new());
            };
            if let Some(out_dir) = &out_dir {
                let stem = format!("{}.wasm", test_emit_stem(suite_key));
                if emit.ir
                    && let Some(ir) = &compiled.ir
                {
                    write_test_emit_artifact(out_dir, &stem, "sona", ir, &mut artifact_output)?;
                }
                if emit.rmir
                    && let Some(rmir) = &compiled.rmir
                {
                    write_test_emit_artifact(out_dir, &stem, "rmir", rmir, &mut artifact_output)?;
                }
            }
            if let Some(report) = report {
                let dir = report.root_dir.join("artifacts/wasm");
                fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
                fs::write(dir.join("tests.wasm"), &compiled.wasm).map_err(|err| err.to_string())?;
                if let Some(ir) = &compiled.ir {
                    fs::write(dir.join("tests.sona"), ir).map_err(|err| err.to_string())?;
                }
                if let Some(rmir) = &compiled.rmir {
                    fs::write(dir.join("tests.rmir"), rmir).map_err(|err| err.to_string())?;
                }
            }
            let mut config = Config::default();
            config
                .consume_fuel(true)
                .set_max_recursion_depth(WASM_TEST_RECURSION_DEPTH)
                .set_max_stack_height(WASM_TEST_VALUE_STACK_BYTES);
            let engine = Engine::new(&config);
            let module = Module::new(&engine, &compiled.wasm)
                .map_err(|err| format!("invalid wasm test module: {err}"))?;
            let suite = Arc::new(WasmSuite { engine, module });
            Ok(compiled
                .tests
                .into_iter()
                .map(|test| SingleTestJob {
                    suite_key: suite_key.to_string(),
                    case: CompiledTest::Wasm(WasmTestCase {
                        name: test.name,
                        export: test.symbol,
                        suite: Arc::clone(&suite),
                    }),
                    evm_trace: None,
                    report_root: report.map(|ctx| ctx.root_dir.clone()),
                })
                .collect())
        },
        "wasm",
        suite,
        report,
        output,
    );
    output.push_str(&artifact_output);
    match prepared {
        Ok(single_jobs) => SuitePreparation {
            results: Vec::new(),
            single_jobs,
        },
        Err(results) => SuitePreparation {
            results,
            single_jobs: Vec::new(),
        },
    }
}

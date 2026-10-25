#![cfg(feature = "wasm")]

use std::{
    fs, mem,
    path::{Path, PathBuf},
    process::Command,
};

use tempfile::tempdir;
use wasmi::{Caller, Config, Engine, Linker, Module, Store};

const LEVELS: [&str; 3] = ["0", "1", "2"];

fn build(source: &Path, out: &Path, level: &str) -> PathBuf {
    let result = Command::new(env!("CARGO_BIN_EXE_fe"))
        .args(["build", "--backend", "wasm", "-O", level, "--out-dir"])
        .arg(out)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "wasm build failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let stem = source.file_stem().unwrap().to_str().unwrap();
    out.join(format!("{stem}.wasm"))
}

#[derive(Default)]
struct Host {
    input: Vec<u8>,
    output: Vec<u8>,
}

/// The result of running a module's `main`, or the trap that ended it.
struct Run {
    exit: Result<i32, String>,
    output: Vec<u8>,
}

fn run(module: &Path, input: &[u8]) -> Run {
    let mut config = Config::default();
    config.set_max_recursion_depth(1 << 16);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, fs::read(module).unwrap()).unwrap();
    let mut store = Store::new(
        &engine,
        Host {
            input: input.iter().rev().copied().collect(),
            output: Vec::new(),
        },
    );
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap(
            "env",
            "putchar",
            |mut caller: Caller<'_, Host>, c: i32| -> i32 {
                caller.data_mut().output.push(c as u8);
                c & 0xff
            },
        )
        .unwrap();
    linker
        .func_wrap("env", "getchar", |mut caller: Caller<'_, Host>| -> i32 {
            caller.data_mut().input.pop().map_or(-1, i32::from)
        })
        .unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let exit = instance
        .get_typed_func::<(), i32>(&store, "main")
        .unwrap()
        .call(&mut store, ())
        .map_err(|error| error.to_string());
    Run {
        exit,
        output: mem::take(&mut store.data_mut().output),
    }
}

fn exit_code(source: &str, level: &str) -> Result<i32, String> {
    let temp = tempdir().unwrap();
    let path = temp.path().join("program.fe");
    fs::write(&path, source).unwrap();
    run(&build(&path, temp.path(), level), &[]).exit
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn wasm_main_returns_its_exit_code() {
    for level in LEVELS {
        assert_eq!(exit_code("pub fn main() -> i32 { 42 }\n", level), Ok(42));
    }
}

#[test]
fn wasm_executes_representative_programs() {
    for name in [
        "arithmetic",
        "control_flow",
        "aggregates",
        "wide_integer",
        "generic_reachability",
        "entry_name_collision",
    ] {
        let source =
            fs::read_to_string(fixtures().join(format!("cli_output/native/{name}.fe"))).unwrap();
        for level in LEVELS {
            assert_eq!(exit_code(&source, level), Ok(0), "{name} at O{level}");
        }
    }
}

#[test]
fn wasm_values_are_destroyed_once_in_scope_order() {
    let temp = tempdir().unwrap();
    for level in LEVELS {
        let out = temp.path().join(format!("out-{level}"));
        let module = build(&fixtures().join("native/drop_order.fe"), &out, level);
        let result = run(&module, &[]);
        assert_eq!(result.exit, Ok(0), "O{level}");
        assert_eq!(
            String::from_utf8_lossy(&result.output),
            "a.\n.a\n0123\nSs\nN\nA,B\n1;2\ny\nx\nx\ny\noi\nABC\nZ-[-\nH-KLJI\n-l\n!r\nA-B-C--Z\nc-c\n-e-ox\n.-p\n+o\n-e\n-BOIA\nq-k\nkQ\n-Vv-\n-u\nU\n-GF\nu1u2u3u4guuuu\n1a\n-b\n-\n-0-1-.2\nba\n-B\nd\nk",
            "O{level}"
        );
    }
}

#[test]
fn wasm_owners_destroy_what_they_hold() {
    let temp = tempdir().unwrap();
    for level in LEVELS {
        let out = temp.path().join(format!("out-{level}"));
        let result = run(
            &build(&fixtures().join("native/owners.fe"), &out, level),
            &[],
        );
        assert_eq!(result.exit, Ok(0), "O{level}");
        assert_eq!(
            String::from_utf8_lossy(&result.output),
            "de-ab-c-fg\nxy-ABC\n-b\nu-U\n123\n",
            "O{level}"
        );
    }
}

#[test]
fn wasm_raw_references_and_projections_execute() {
    let temp = tempdir().unwrap();
    for name in [
        "empty_owners",
        "raw_scalar_reference_accessors",
        "raw_aggregate_reference_accessors",
        "returned_reference_projection",
    ] {
        for level in LEVELS {
            let out = temp.path().join(format!("{name}-{level}"));
            let module = build(&fixtures().join(format!("native/{name}.fe")), &out, level);
            assert_eq!(run(&module, &[]).exit, Ok(0), "{name} at O{level}");
        }
    }
}

#[test]
fn wasm_host_io_reads_and_writes_characters() {
    let source = r#"
use std::io::{Read, Write, host, read_char, write_char, writeln}
pub fn main() -> i32 {
    with (Read = host(), Write = host()) {
        writeln("ready")
        var sum: i32 = 0
        var c = read_char()
        while c >= 0 {
            write_char(c + 1)
            sum += c
            c = read_char()
        }
        sum
    }
}
"#;
    let temp = tempdir().unwrap();
    let path = temp.path().join("echo.fe");
    fs::write(&path, source).unwrap();
    for level in LEVELS {
        let out = temp.path().join(format!("out-{level}"));
        let result = run(&build(&path, &out, level), b"abc");
        assert_eq!(result.exit, Ok(0x61 + 0x62 + 0x63), "O{level}");
        assert_eq!(result.output, b"ready\nbcd", "O{level}");
    }
}

#[test]
fn wasm_unchecked_div_rem_match_every_width() {
    for bits in [8, 16, 32, 64, 128, 256] {
        let sign_bit = bits - 1;
        let source = format!(
            r#"
#[arithmetic(unchecked)]
fn signed(value: i{bits}, divisor: i{bits}, quotient: i{bits}, remainder: i{bits}) -> bool {{
    value / divisor == quotient && value % divisor == remainder
}}

#[arithmetic(unchecked)]
fn unsigned(value: u{bits}, divisor: u{bits}, quotient: u{bits}, remainder: u{bits}) -> bool {{
    value / divisor == quotient && value % divisor == remainder
}}

fn checked_rem(value: i{bits}, divisor: i{bits}) -> i{bits} {{
    value % divisor
}}

pub fn main() -> i32 {{
    let min: i{bits} = 1 << {sign_bit}
    if !signed(value: -17, divisor: 0, quotient: 0, remainder: 0) {{ return 1 }}
    if !unsigned(value: 17, divisor: 0, quotient: 0, remainder: 0) {{ return 2 }}
    if !signed(value: min, divisor: -1, quotient: min, remainder: 0) {{ return 3 }}
    if !signed(value: -17, divisor: 5, quotient: -3, remainder: -2) {{ return 4 }}
    if !signed(value: 17, divisor: -5, quotient: -3, remainder: 2) {{ return 5 }}
    if !unsigned(value: 17, divisor: 5, quotient: 3, remainder: 2) {{ return 6 }}
    if checked_rem(value: min, divisor: -1) != 0 {{ return 7 }}
    0
}}
"#
        );
        for level in LEVELS {
            assert_eq!(exit_code(&source, level), Ok(0), "{bits}-bit at O{level}");
        }
    }
}

#[test]
fn wasm_checked_arithmetic_traps() {
    let mut cases = Vec::new();
    for bits in [8, 128, 256] {
        let sign_bit = bits - 1;
        let min = format!("1 << {sign_bit}");
        for (prefix, op, value, rhs) in [
            ("u", "/", "17".to_string(), "0".to_string()),
            ("u", "%", "17".to_string(), "0".to_string()),
            ("i", "/", "-17".to_string(), "0".to_string()),
            ("i", "%", "-17".to_string(), "0".to_string()),
            ("i", "/", min.clone(), "-1".to_string()),
            ("u", "*", "1 << 7".to_string(), format!("1 << {}", bits - 7)),
            ("i", "-", min.clone(), "1".to_string()),
            ("u", "**", "2".to_string(), bits.to_string()),
            ("i", "**", "2".to_string(), sign_bit.to_string()),
            ("i", "**", "3".to_string(), "-1".to_string()),
        ] {
            cases.push((format!("{prefix}{bits}"), op, value, rhs));
        }
    }

    // `getchar` selects a case, so one module per level covers every case.
    let mut source = String::from("use std::io::{Read, host, read_char}\n");
    for (index, (ty, op, _, _)) in cases.iter().enumerate() {
        source.push_str(&format!(
            "fn calculate_{index}(value: {ty}, rhs: {ty}) -> {ty} {{ value {op} rhs }}\n"
        ));
    }
    source.push_str(
        "pub fn main() -> i32 {\n    let selector = with (Read = host()) { read_char() }\n",
    );
    for (index, (_, _, value, rhs)) in cases.iter().enumerate() {
        source.push_str(&format!(
            "    if selector == {index} {{ if calculate_{index}(value: {value}, rhs: {rhs}) == 0 {{ return 0 }} else {{ return 1 }} }}\n"
        ));
    }
    source.push_str("    97\n}\n");

    let temp = tempdir().unwrap();
    let path = temp.path().join("checked.fe");
    fs::write(&path, &source).unwrap();
    for level in ["0", "1"] {
        let module = build(&path, &temp.path().join(format!("out-{level}")), level);
        assert_eq!(run(&module, &[]).exit, Ok(97), "control at O{level}");
        for (index, (ty, op, value, rhs)) in cases.iter().enumerate() {
            assert!(
                run(&module, &[u8::try_from(index).unwrap()]).exit.is_err(),
                "checked {ty} {value} {op} {rhs} did not trap at O{level}"
            );
        }
    }
}

#[test]
fn wasm_trait_providers_keep_state() {
    for copy_impl in ["impl Copy for Counter {}", ""] {
        let source = format!(
            r#"
trait Tick {{ fn next(mut self) -> i32 }}
struct Counter {{ value: i32 }}
{copy_impl}
impl Tick for Counter {{
    fn next(mut self) -> i32 {{
        self.value += 1
        self.value
    }}
}}
fn counter() -> Counter {{ Counter {{ value: 0 }} }}
fn next() -> i32 uses (tick: mut Tick) {{ tick.next() }}
pub fn main() -> i32 {{
    with (Tick = counter()) {{ next() + next() }}
}}
"#
        );
        for level in LEVELS {
            assert_eq!(exit_code(&source, level), Ok(3), "{copy_impl}, O{level}");
        }
    }
    let source = fs::read_to_string(fixtures().join("fe_test/trait_provider_identity.fe")).unwrap();
    for level in LEVELS {
        assert_eq!(exit_code(&source, level), Ok(42), "O{level}");
    }
}

#[test]
fn wasm_test_runner_executes_host_fixtures() {
    for name in [
        "aggregate_object_copies",
        "const_aggregate_values",
        "drops",
        "fallible_allocation",
        "mem_drop_and_forget",
    ] {
        let fixture = fixtures().join(format!("fe_test/{name}.fe"));
        for level in LEVELS {
            let result = Command::new(env!("CARGO_BIN_EXE_fe"))
                .args(["test", "--backend", "wasm", "--jobs", "1", "-O", level])
                .arg(&fixture)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{name} at O{level}:\n{}",
                String::from_utf8_lossy(&result.stdout)
            );
        }
    }
}

#[test]
fn wasm_rejects_entries_with_arguments() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("args.fe");
    fs::write(
        &path,
        "pub fn main(argc: i32, argv: **u8) -> i32 { argc }\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_fe"))
        .args(["build", "--backend", "wasm", "--out-dir"])
        .arg(temp.path())
        .arg(&path)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("wasm `main` must have signature `pub fn main() -> i32`"),
        "{result:?}"
    );
}

#[test]
fn wasm_build_rejects_other_backends_and_emits() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("main.fe");
    fs::write(&path, "pub fn main() -> i32 { 0 }\n").unwrap();
    for (args, message) in [
        (
            &["--emit", "wasm"][..],
            "`--emit wasm` requires `--backend wasm`",
        ),
        (
            &["--backend", "wasm", "--emit", "bytecode"][..],
            "wasm backend only supports `--emit wasm` and `--emit ir`",
        ),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_fe"))
            .arg("build")
            .args(args)
            .arg("--out-dir")
            .arg(temp.path())
            .arg(&path)
            .output()
            .unwrap();
        assert!(!result.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(message),
            "{args:?}: {result:?}"
        );
    }
}

#[test]
fn wasm_allocations_beyond_the_address_space_fail() {
    let source = r#"
use core::Option
use core::ptr
use std::io::{Read, host, read_char}
fn fails(_ size: u256) -> bool {
    match ptr::try_alloc_bytes(size) {
        Option::Some(_) => false
        Option::None => true
    }
}
pub fn main() -> i32 {
    if with (Read = host()) { read_char() } >= 0 {
        let _ = ptr::alloc_bytes(1 << 32)
        return 3
    }
    if !fails(1 << 32) || !fails((1 << 32) + 16) { return 1 }
    if fails(64) { return 2 }
    0
}
"#;
    let temp = tempdir().unwrap();
    let path = temp.path().join("oversized.fe");
    fs::write(&path, source).unwrap();
    for level in LEVELS {
        let module = build(&path, &temp.path().join(format!("out-{level}")), level);
        assert_eq!(run(&module, &[]).exit, Ok(0), "O{level}");
        assert!(run(&module, b"x").exit.is_err(), "O{level}");
    }
}

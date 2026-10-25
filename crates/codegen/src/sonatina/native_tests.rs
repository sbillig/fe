use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;
use mir::{RuntimePackage, build_native_test_runtime_package, format_runtime_package};
use sonatina_codegen::Compile;
#[cfg(feature = "cranelift")]
use sonatina_codegen::isa::cranelift::CraneliftObjectBackend;
#[cfg(feature = "wasm")]
use sonatina_codegen::isa::wasm::WasmBackend;
#[cfg(feature = "wasm")]
use sonatina_ir::isa::wasm32::Wasm32;
use sonatina_ir::{Linkage, Module, ir_writer::ModuleWriter, isa::Isa};

#[cfg(feature = "cranelift")]
use super::create_native_isa;
use super::{
    LowerError, ensure_module_sonatina_ir_valid, format_backend_errors, lower_runtime,
    to_sonatina_opt_level,
};
use crate::{
    OptLevel,
    test_output::{TestRootMetadataError, runtime_test_root_metadata},
};

#[derive(Debug)]
pub struct NativeTestEntry {
    pub name: String,
    pub symbol: String,
}

/// One compiled object for every selected test in a suite.
#[cfg(feature = "cranelift")]
#[derive(Debug)]
pub struct NativeTestSuite {
    pub tests: Vec<NativeTestEntry>,
    pub object: Vec<u8>,
    pub ir: Option<String>,
    pub rmir: Option<String>,
}

/// One wasm module exporting every selected test in a suite.
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub struct WasmTestSuite {
    pub tests: Vec<NativeTestEntry>,
    pub wasm: Vec<u8>,
    pub ir: Option<String>,
    pub rmir: Option<String>,
}

#[cfg(feature = "cranelift")]
pub fn emit_test_suite_native(
    db: &DriverDataBase,
    top_mods: &[TopLevelMod<'_>],
    opt_level: OptLevel,
    filter: Option<&str>,
    emit_ir: bool,
    emit_rmir: bool,
) -> Result<Option<NativeTestSuite>, LowerError> {
    let Some((module, tests, package)) =
        lower_test_suite(db, top_mods, filter, &create_native_isa()?)?
    else {
        return Ok(None);
    };
    let mut compile = Compile::new(module, CraneliftObjectBackend::new())
        .with_opt_level(to_sonatina_opt_level(opt_level));
    compile.optimize();
    ensure_module_sonatina_ir_valid(compile.module())?;
    let ir = emit_ir.then(|| ModuleWriter::new(compile.module()).dump_string());
    let object = compile
        .compile()
        .map(|artifact| artifact.into_bytes())
        .map_err(|errors| LowerError::Internal(format_backend_errors(&errors)))?;
    Ok(Some(NativeTestSuite {
        tests,
        object,
        ir,
        rmir: emit_rmir.then(|| format_runtime_package(db, &package)),
    }))
}

#[cfg(feature = "wasm")]
pub fn emit_test_suite_wasm(
    db: &DriverDataBase,
    top_mods: &[TopLevelMod<'_>],
    opt_level: OptLevel,
    filter: Option<&str>,
    emit_ir: bool,
    emit_rmir: bool,
) -> Result<Option<WasmTestSuite>, LowerError> {
    let Some((module, tests, package)) =
        lower_test_suite(db, top_mods, filter, &Wasm32::default())?
    else {
        return Ok(None);
    };
    let mut compile =
        Compile::new(module, WasmBackend::new()).with_opt_level(to_sonatina_opt_level(opt_level));
    compile.optimize();
    ensure_module_sonatina_ir_valid(compile.module())?;
    let ir = emit_ir.then(|| ModuleWriter::new(compile.module()).dump_string());
    let wasm = compile
        .compile()
        .map(|artifact| artifact.into_bytes())
        .map_err(|errors| LowerError::Internal(format_backend_errors(&errors)))?;
    Ok(Some(WasmTestSuite {
        tests,
        wasm,
        ir,
        rmir: emit_rmir.then(|| format_runtime_package(db, &package)),
    }))
}

/// Lowers every selected test to a public, parameterless function, or
/// nothing when no test is selected.
fn lower_test_suite<'db, I>(
    db: &'db DriverDataBase,
    top_mods: &[TopLevelMod<'db>],
    filter: Option<&str>,
    isa: &I,
) -> Result<Option<(Module, Vec<NativeTestEntry>, RuntimePackage<'db>)>, LowerError>
where
    I: Isa,
    I::InstSet: lower_runtime::LoweringInstSet,
{
    let package = build_native_test_runtime_package(db, top_mods, filter)?;
    if package.root_objects(db).is_empty() {
        return Ok(None);
    }
    let mut roots = Vec::new();
    for &object in package.root_objects(db) {
        for section in object.sections(db) {
            let metadata = runtime_test_root_metadata(db, section.entry.owner(db), &section.name)
                .map_err(|err| match err {
                TestRootMetadataError::InvalidPackage(message) => LowerError::Internal(message),
                TestRootMetadataError::Unsupported(message) => LowerError::Unsupported(message),
            })?;
            if metadata.expected_revert.is_some() || metadata.initial_balance.is_some() {
                return Err(LowerError::Unsupported(format!(
                    "test `{}` cannot use EVM `should_revert` or `balance` attributes",
                    metadata.display_name
                )));
            }
            roots.push((metadata.display_name, section.entry.instance(db)));
        }
    }
    let (module, functions) =
        lower_runtime::compile_runtime_package_sonatina_for_isa(db, &package, isa, false, None)?;
    let mut tests = Vec::with_capacity(roots.len());
    for (name, instance) in roots {
        let entry = functions[&instance];
        let symbol = module.ctx.func_sig(entry, |signature| {
            if !signature.args().is_empty() || !signature.ret_tys().is_empty() {
                return Err(LowerError::Internal(format!(
                    "test wrapper `{name}` must have a void, parameterless signature"
                )));
            }
            Ok(signature.name().to_string())
        })?;
        module.ctx.update_func_linkage(entry, Linkage::Public);
        tests.push(NativeTestEntry { name, symbol });
    }
    Ok(Some((module, tests, package)))
}

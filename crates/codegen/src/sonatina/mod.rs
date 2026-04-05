mod lower_runtime;

use std::collections::BTreeMap;

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::{HirIngot, TopLevelMod};
use mir2::{RuntimePackage, build_runtime_package, build_test_runtime_package};
use sonatina_codegen::{
    isa::evm::EvmBackend,
    object::{CompileOptions, ObjectArtifact, compile_all_objects},
};
use sonatina_ir::{Module, ir_writer::ModuleWriter, isa::evm::Evm, module::ModuleCtx};
use sonatina_triple::{Architecture, EvmVersion, OperatingSystem, TargetTriple, Vendor};
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

use crate::{
    OptLevel, TargetDataLayout, TestMetadata, TestModuleOutput,
    test_output::{TestRootMetadataError, runtime_test_root_metadata},
};

#[derive(Debug)]
pub enum LowerError {
    RuntimeLower(mir2::LowerError),
    Unsupported(String),
    Internal(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::RuntimeLower(err) => write!(f, "{err}"),
            LowerError::Unsupported(message) => write!(f, "unsupported: {message}"),
            LowerError::Internal(message) => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for LowerError {}

impl From<mir2::LowerError> for LowerError {
    fn from(err: mir2::LowerError) -> Self {
        LowerError::RuntimeLower(err)
    }
}

#[derive(Debug, Clone)]
pub struct SonatinaContractBytecode {
    pub deploy: Vec<u8>,
    pub runtime: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SonatinaTestOptions {
    pub emit_observability: bool,
}

pub(crate) fn create_evm_isa() -> Evm {
    Evm::new(TargetTriple::new(
        Architecture::Evm,
        Vendor::Ethereum,
        OperatingSystem::Evm(EvmVersion::Osaka),
    ))
}

fn create_module_ctx() -> ModuleCtx {
    ModuleCtx::new(&create_evm_isa())
}

fn ensure_module_sonatina_ir_valid(module: &Module) -> Result<(), LowerError> {
    let report = verify_module(module, &VerifierConfig::for_level(VerificationLevel::Full));
    if report.has_errors() {
        return Err(LowerError::Internal(report.to_string()));
    }
    Ok(())
}

fn run_sonatina_optimization_pipeline(module: &mut Module, opt_level: OptLevel) {
    match opt_level {
        OptLevel::O0 => {}
        OptLevel::Os => sonatina_codegen::optim::Pipeline::size().run(module),
        OptLevel::O2 => sonatina_codegen::optim::Pipeline::speed().run(module),
    }
}

fn compile_all_runtime_objects(
    module: &Module,
    emit_observability: bool,
) -> Result<Vec<ObjectArtifact>, LowerError> {
    let mut options = CompileOptions::default();
    let mut verifier_cfg = VerifierConfig::for_level(VerificationLevel::Full);
    verifier_cfg.allow_detached_entities = true;
    options.verifier_cfg = verifier_cfg;
    options.emit_observability = emit_observability;
    compile_all_objects(module, &EvmBackend::new(create_evm_isa()), &options).map_err(|errors| {
        LowerError::Internal(
            errors
                .iter()
                .map(|error| format!("{error:?}"))
                .collect::<Vec<_>>()
                .join("; "),
        )
    })
}

fn section_name_for_runtime(name: &mir2::RuntimeSectionName) -> sonatina_ir::SectionName {
    match name {
        mir2::RuntimeSectionName::Init => "init".into(),
        mir2::RuntimeSectionName::Runtime => "runtime".into(),
        mir2::RuntimeSectionName::Main => "main".into(),
        mir2::RuntimeSectionName::Test(name) => format!("test_{name}").into(),
        mir2::RuntimeSectionName::CodeRegion(symbol) => format!("code_region_{symbol}").into(),
    }
}

fn wrap_as_init_code(runtime: &[u8]) -> Vec<u8> {
    fn push_u256(mut value: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        while value > 0 {
            bytes.push((value & 0xff) as u8);
            value >>= 8;
        }
        if bytes.is_empty() {
            bytes.push(0);
        }
        bytes.reverse();
        let mut out = Vec::with_capacity(1 + bytes.len());
        out.push(0x5f + bytes.len() as u8);
        out.extend(bytes);
        out
    }

    let len_push = push_u256(runtime.len());
    let mut init = Vec::with_capacity(32 + runtime.len());
    init.extend(len_push.clone());
    init.push(0x61);
    let off_pos = init.len();
    init.extend([0, 0]);
    init.extend([0x60, 0x00]);
    init.push(0x39);
    init.extend(len_push);
    init.extend([0x60, 0x00]);
    init.push(0xf3);
    let off = init.len();
    init[off_pos] = ((off >> 8) & 0xff) as u8;
    init[off_pos + 1] = (off & 0xff) as u8;
    init.extend_from_slice(runtime);
    init
}

pub fn compile_runtime_package_sonatina(
    db: &DriverDataBase,
    package: &RuntimePackage<'_>,
    layout: TargetDataLayout,
) -> Result<Module, LowerError> {
    lower_runtime::compile_runtime_package_sonatina(db, package, layout)
}

pub fn emit_runtime_package_sonatina_ir(
    db: &DriverDataBase,
    package: &RuntimePackage<'_>,
    layout: TargetDataLayout,
) -> Result<String, LowerError> {
    let module = compile_runtime_package_sonatina(db, package, layout)?;
    let mut writer = ModuleWriter::new(&module);
    Ok(writer.dump_string())
}

pub fn emit_runtime_package_sonatina_ir_optimized(
    db: &DriverDataBase,
    package: &RuntimePackage<'_>,
    layout: TargetDataLayout,
    opt_level: OptLevel,
) -> Result<String, LowerError> {
    let mut module = compile_runtime_package_sonatina(db, package, layout)?;
    ensure_module_sonatina_ir_valid(&module)?;
    run_sonatina_optimization_pipeline(&mut module, opt_level);
    ensure_module_sonatina_ir_valid(&module)?;
    let mut writer = ModuleWriter::new(&module);
    Ok(writer.dump_string())
}

pub fn emit_runtime_package_sonatina_bytecode(
    db: &DriverDataBase,
    package: &RuntimePackage<'_>,
    layout: TargetDataLayout,
    opt_level: OptLevel,
) -> Result<BTreeMap<String, SonatinaContractBytecode>, LowerError> {
    let mut module = compile_runtime_package_sonatina(db, package, layout)?;
    ensure_module_sonatina_ir_valid(&module)?;
    run_sonatina_optimization_pipeline(&mut module, opt_level);
    ensure_module_sonatina_ir_valid(&module)?;
    let artifacts = compile_all_runtime_objects(&module, false)?;
    let artifacts_by_name = artifacts
        .iter()
        .map(|artifact| (artifact.object.0.as_str(), artifact))
        .collect::<std::collections::HashMap<_, _>>();

    let mut out = BTreeMap::new();
    for object in package.root_objects(db) {
        let object_name = object.name(db);
        let artifact = artifacts_by_name
            .get(object_name.as_str())
            .copied()
            .ok_or_else(|| {
                LowerError::Internal(format!("compiled object `{object_name}` not found"))
            })?;
        let init = artifact
            .sections
            .get(&section_name_for_runtime(&mir2::RuntimeSectionName::Init));
        let runtime = artifact.sections.get(&section_name_for_runtime(
            &mir2::RuntimeSectionName::Runtime,
        ));
        let (deploy, runtime) = match (init, runtime) {
            (Some(init), Some(runtime)) => (init.bytes.clone(), runtime.bytes.clone()),
            _ => {
                let sections = object.sections(db);
                let section = sections.first().ok_or_else(|| {
                    LowerError::Internal(format!("root object `{object_name}` has no sections"))
                })?;
                let runtime = artifact
                    .sections
                    .get(&section_name_for_runtime(&section.name))
                    .ok_or_else(|| {
                        LowerError::Internal(format!(
                            "compiled object `{object_name}` is missing section `{:?}`",
                            section.name
                        ))
                    })?
                    .bytes
                    .clone();
                (wrap_as_init_code(&runtime), runtime)
            }
        };
        out.insert(
            object_name.clone(),
            SonatinaContractBytecode { deploy, runtime },
        );
    }
    Ok(out)
}

pub fn emit_module_sonatina_ir(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
) -> Result<String, LowerError> {
    let package = build_runtime_package(db, top_mod)?;
    emit_runtime_package_sonatina_ir(db, &package, crate::EVM_LAYOUT)
}

pub fn emit_module_sonatina_ir_optimized(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    opt_level: OptLevel,
    _contract: Option<&str>,
) -> Result<String, LowerError> {
    let package = build_runtime_package(db, top_mod)?;
    emit_runtime_package_sonatina_ir_optimized(db, &package, crate::EVM_LAYOUT, opt_level)
}

pub fn emit_ingot_sonatina_ir(db: &DriverDataBase, ingot: Ingot<'_>) -> Result<String, LowerError> {
    let mut modules = Vec::new();
    for &top_mod in ingot.all_modules(db) {
        modules.push(emit_module_sonatina_ir(db, top_mod)?);
    }
    Ok(modules.join("\n\n"))
}

pub fn emit_ingot_sonatina_ir_optimized(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    opt_level: OptLevel,
    _contract: Option<&str>,
) -> Result<String, LowerError> {
    let mut modules = Vec::new();
    for &top_mod in ingot.all_modules(db) {
        modules.push(emit_module_sonatina_ir_optimized(
            db, top_mod, opt_level, None,
        )?);
    }
    Ok(modules.join("\n\n"))
}

pub fn validate_module_sonatina_ir(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
) -> Result<String, LowerError> {
    let package = build_runtime_package(db, top_mod)?;
    compile_runtime_package_sonatina(db, &package, crate::EVM_LAYOUT)?;
    Ok("ok\n".to_string())
}

pub fn emit_module_sonatina_bytecode(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    opt_level: OptLevel,
    _contract: Option<&str>,
) -> Result<BTreeMap<String, SonatinaContractBytecode>, LowerError> {
    let package = build_runtime_package(db, top_mod)?;
    emit_runtime_package_sonatina_bytecode(db, &package, crate::EVM_LAYOUT, opt_level)
}

pub fn emit_ingot_sonatina_bytecode(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    opt_level: OptLevel,
    _contract: Option<&str>,
) -> Result<BTreeMap<String, SonatinaContractBytecode>, LowerError> {
    let mut outputs = BTreeMap::new();
    for &top_mod in ingot.all_modules(db) {
        for (name, bytecode) in emit_module_sonatina_bytecode(db, top_mod, opt_level, None)? {
            if outputs.insert(name.clone(), bytecode).is_some() {
                return Err(LowerError::Internal(format!(
                    "duplicate root object `{name}` across ingot modules"
                )));
            }
        }
    }
    Ok(outputs)
}

pub fn emit_test_module_sonatina(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    opt_level: OptLevel,
    options: SonatinaTestOptions,
    filter: Option<&str>,
) -> Result<TestModuleOutput, LowerError> {
    let package = build_test_runtime_package(db, top_mod, filter)?;
    let mut module = compile_runtime_package_sonatina(db, &package, crate::EVM_LAYOUT)?;
    ensure_module_sonatina_ir_valid(&module)?;
    run_sonatina_optimization_pipeline(&mut module, opt_level);
    ensure_module_sonatina_ir_valid(&module)?;
    let artifacts = compile_all_runtime_objects(&module, options.emit_observability)?;
    let artifacts_by_name = artifacts
        .iter()
        .map(|artifact| (artifact.object.0.as_str(), artifact))
        .collect::<std::collections::HashMap<_, _>>();

    let mut tests = Vec::new();
    for object in package.root_objects(db) {
        let sections = object.sections(db);
        let Some(section) = sections.first() else {
            continue;
        };
        let mir2::RuntimeSectionName::Test(_) = &section.name else {
            continue;
        };
        let artifact = artifacts_by_name
            .get(object.name(db).as_str())
            .copied()
            .ok_or_else(|| {
                LowerError::Internal(format!("compiled object `{}` not found", object.name(db)))
            })?;
        let runtime = artifact
            .sections
            .get(&section_name_for_runtime(&section.name))
            .ok_or_else(|| {
                LowerError::Internal(format!(
                    "compiled object `{}` missing test section",
                    object.name(db)
                ))
            })?;
        let metadata = runtime_test_root_metadata(db, &section.entry.owner(db), &section.name)
            .map_err(|err| match err {
                TestRootMetadataError::InvalidPackage(message) => LowerError::Internal(message),
                TestRootMetadataError::Unsupported(message) => LowerError::Unsupported(message),
            })?;
        tests.push(TestMetadata {
            display_name: metadata.display_name,
            hir_name: metadata.hir_name,
            symbol_name: section.entry.symbol(db).clone(),
            object_name: object.name(db).clone(),
            yul: String::new(),
            bytecode: wrap_as_init_code(&runtime.bytes),
            sonatina_observability_json: artifact.observability_json(),
            value_param_count: 0,
            effect_param_count: 0,
            init_bytecode: Vec::new(),
            expected_revert: metadata.expected_revert,
            initial_balance: metadata.initial_balance,
        });
    }
    Ok(TestModuleOutput { tests })
}

pub fn emit_test_ingot_sonatina(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    opt_level: OptLevel,
    options: SonatinaTestOptions,
    filter: Option<&str>,
) -> Result<TestModuleOutput, LowerError> {
    let mut top_mods = ingot.all_modules(db).to_vec();
    top_mods.sort_by(|left, right| left.name(db).cmp(&right.name(db)));

    let mut output = TestModuleOutput { tests: Vec::new() };
    for top_mod in top_mods {
        output.extend(emit_test_module_sonatina(
            db, top_mod, opt_level, options, filter,
        )?);
    }
    output.sort_tests();
    Ok(output)
}

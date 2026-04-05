mod doc;
mod emitter;
mod errors;
mod legalize;
mod state;

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::{HirIngot, TopLevelMod};
use mir2::{RuntimePackage, build_runtime_package, build_test_runtime_package};

use crate::{TargetDataLayout, TestModuleOutput};

pub use errors::YulError;

#[derive(Debug)]
pub enum EmitModuleError {
    RuntimeLower(mir2::LowerError),
    Yul(YulError),
}

impl std::fmt::Display for EmitModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitModuleError::RuntimeLower(err) => write!(f, "{err}"),
            EmitModuleError::Yul(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for EmitModuleError {}

impl From<mir2::LowerError> for EmitModuleError {
    fn from(err: mir2::LowerError) -> Self {
        EmitModuleError::RuntimeLower(err)
    }
}

impl From<YulError> for EmitModuleError {
    fn from(err: YulError) -> Self {
        EmitModuleError::Yul(err)
    }
}

pub fn emit_runtime_package_yul<'db>(
    db: &'db DriverDataBase,
    package: &RuntimePackage<'db>,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let package = legalize::legalize_runtime_package(db, package, layout)?;
    emitter::emit_runtime_package_yul(db, &package).map_err(Into::into)
}

pub fn emit_test_runtime_package_yul<'db>(
    db: &'db DriverDataBase,
    package: &RuntimePackage<'db>,
    layout: TargetDataLayout,
    filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    let package = legalize::legalize_runtime_package(db, package, layout)?;
    emitter::emit_test_runtime_package_yul(db, &package, filter).map_err(Into::into)
}

pub fn emit_module_yul(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
) -> Result<String, EmitModuleError> {
    emit_module_yul_with_layout(db, top_mod, crate::EVM_LAYOUT)
}

pub fn emit_module_yul_with_layout(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let package = build_runtime_package(db, top_mod)?;
    emit_runtime_package_yul(db, &package, layout)
}

pub fn emit_ingot_yul(db: &DriverDataBase, ingot: Ingot<'_>) -> Result<String, EmitModuleError> {
    emit_ingot_yul_with_layout(db, ingot, crate::EVM_LAYOUT)
}

pub fn emit_ingot_yul_with_layout(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let mut modules = Vec::new();
    for &top_mod in ingot.all_modules(db) {
        modules.push(emit_module_yul_with_layout(db, top_mod, layout)?);
    }
    Ok(modules.join("\n\n"))
}

pub fn emit_test_module_yul(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    emit_test_module_yul_with_layout(db, top_mod, filter, crate::EVM_LAYOUT)
}

pub fn emit_test_module_yul_with_layout(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    filter: Option<&str>,
    layout: TargetDataLayout,
) -> Result<TestModuleOutput, EmitModuleError> {
    let package = build_test_runtime_package(db, top_mod, filter)?;
    emit_test_runtime_package_yul(db, &package, layout, filter)
}

pub fn emit_test_ingot_yul(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    emit_test_ingot_yul_with_layout(db, ingot, filter, crate::EVM_LAYOUT)
}

pub fn emit_test_ingot_yul_with_layout(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    filter: Option<&str>,
    layout: TargetDataLayout,
) -> Result<TestModuleOutput, EmitModuleError> {
    let mut top_mods = ingot.all_modules(db).to_vec();
    top_mods.sort_by(|left, right| left.name(db).cmp(&right.name(db)));

    let mut output = TestModuleOutput { tests: Vec::new() };
    for top_mod in top_mods {
        output.extend(emit_test_module_yul_with_layout(
            db, top_mod, filter, layout,
        )?);
    }
    output.sort_tests();
    Ok(output)
}

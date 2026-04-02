use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;

use crate::{TargetDataLayout, TestModuleOutput};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YulError {
    Unsupported(String),
}

impl std::fmt::Display for YulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            YulError::Unsupported(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for YulError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitModuleError {
    Unsupported(String),
}

impl std::fmt::Display for EmitModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitModuleError::Unsupported(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for EmitModuleError {}

impl From<YulError> for EmitModuleError {
    fn from(err: YulError) -> Self {
        match err {
            YulError::Unsupported(message) => EmitModuleError::Unsupported(message),
        }
    }
}

fn unsupported() -> EmitModuleError {
    EmitModuleError::Unsupported(
        "the Yul backend has not been migrated to mir2 yet; only the Sonatina pipeline is supported"
            .to_string(),
    )
}

pub fn emit_module_yul(
    _db: &DriverDataBase,
    _top_mod: TopLevelMod<'_>,
) -> Result<String, EmitModuleError> {
    Err(unsupported())
}

pub fn emit_module_yul_with_layout(
    _db: &DriverDataBase,
    _top_mod: TopLevelMod<'_>,
    _layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    Err(unsupported())
}

pub fn emit_ingot_yul(_db: &DriverDataBase, _ingot: Ingot<'_>) -> Result<String, EmitModuleError> {
    Err(unsupported())
}

pub fn emit_ingot_yul_with_layout(
    _db: &DriverDataBase,
    _ingot: Ingot<'_>,
    _layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    Err(unsupported())
}

pub fn emit_test_module_yul(
    _db: &DriverDataBase,
    _top_mod: TopLevelMod<'_>,
    _filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    Err(unsupported())
}

pub fn emit_test_module_yul_with_layout(
    _db: &DriverDataBase,
    _top_mod: TopLevelMod<'_>,
    _filter: Option<&str>,
    _layout: TargetDataLayout,
) -> Result<TestModuleOutput, EmitModuleError> {
    Err(unsupported())
}

use std::fmt;

use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;

use crate::TargetDataLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    O0,
    Os,
    #[default]
    O2,
}

impl std::str::FromStr for OptLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(OptLevel::O0),
            "s" => Ok(OptLevel::Os),
            "1" | "2" => Ok(OptLevel::O2),
            _ => Err(format!(
                "unknown optimization level: {s} (expected '0', '1', '2', or 's')"
            )),
        }
    }
}

impl OptLevel {
    pub fn yul_optimize(&self) -> bool {
        !matches!(self, OptLevel::O0)
    }
}

impl fmt::Display for OptLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "0"),
            OptLevel::Os => write!(f, "s"),
            OptLevel::O2 => write!(f, "2"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BackendOutput {
    Yul { source: String, solc_optimize: bool },
    Bytecode(Vec<u8>),
}

impl BackendOutput {
    pub fn as_yul(&self) -> Option<&str> {
        match self {
            BackendOutput::Yul { source, .. } => Some(source),
            BackendOutput::Bytecode(_) => None,
        }
    }

    pub fn yul_solc_optimize(&self) -> Option<bool> {
        match self {
            BackendOutput::Yul { solc_optimize, .. } => Some(*solc_optimize),
            BackendOutput::Bytecode(_) => None,
        }
    }

    pub fn as_bytecode(&self) -> Option<&[u8]> {
        match self {
            BackendOutput::Bytecode(bytes) => Some(bytes),
            BackendOutput::Yul { .. } => None,
        }
    }

    pub fn into_yul(self) -> Option<String> {
        match self {
            BackendOutput::Yul { source, .. } => Some(source),
            BackendOutput::Bytecode(_) => None,
        }
    }

    pub fn into_bytecode(self) -> Option<Vec<u8>> {
        match self {
            BackendOutput::Bytecode(bytes) => Some(bytes),
            BackendOutput::Yul { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum BackendError {
    RuntimeLower(mir2::LowerError),
    Yul(crate::yul::YulError),
    Sonatina(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::RuntimeLower(err) => write!(f, "{err}"),
            BackendError::Yul(err) => write!(f, "{err}"),
            BackendError::Sonatina(message) => write!(f, "sonatina error: {message}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<mir2::LowerError> for BackendError {
    fn from(err: mir2::LowerError) -> Self {
        BackendError::RuntimeLower(err)
    }
}

impl From<crate::yul::YulError> for BackendError {
    fn from(err: crate::yul::YulError) -> Self {
        BackendError::Yul(err)
    }
}

impl From<crate::sonatina::LowerError> for BackendError {
    fn from(err: crate::sonatina::LowerError) -> Self {
        BackendError::Sonatina(err.to_string())
    }
}

impl From<crate::EmitModuleError> for BackendError {
    fn from(err: crate::EmitModuleError) -> Self {
        match err {
            crate::EmitModuleError::RuntimeLower(err) => BackendError::RuntimeLower(err),
            crate::EmitModuleError::Yul(err) => BackendError::Yul(err),
        }
    }
}

pub trait Backend {
    fn name(&self) -> &'static str;

    fn compile(
        &self,
        db: &DriverDataBase,
        top_mod: TopLevelMod<'_>,
        layout: TargetDataLayout,
        opt_level: OptLevel,
    ) -> Result<BackendOutput, BackendError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    Yul,
    #[default]
    Sonatina,
}

impl BackendKind {
    pub fn name(&self) -> &'static str {
        match self {
            BackendKind::Yul => "yul",
            BackendKind::Sonatina => "sonatina",
        }
    }

    pub fn create(&self) -> Box<dyn Backend> {
        match self {
            BackendKind::Yul => Box::new(YulBackend),
            BackendKind::Sonatina => Box::new(SonatinaBackend),
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "yul" => Ok(BackendKind::Yul),
            "sonatina" => Ok(BackendKind::Sonatina),
            _ => Err(format!(
                "unknown backend: {s} (expected 'yul' or 'sonatina')"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct YulBackend;

impl Backend for YulBackend {
    fn name(&self) -> &'static str {
        "yul"
    }

    fn compile(
        &self,
        db: &DriverDataBase,
        top_mod: TopLevelMod<'_>,
        layout: TargetDataLayout,
        opt_level: OptLevel,
    ) -> Result<BackendOutput, BackendError> {
        let yul = crate::emit_module_yul_with_layout(db, top_mod, layout)?;
        Ok(BackendOutput::Yul {
            source: yul,
            solc_optimize: opt_level.yul_optimize(),
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SonatinaBackend;

impl Backend for SonatinaBackend {
    fn name(&self) -> &'static str {
        "sonatina"
    }

    fn compile(
        &self,
        db: &DriverDataBase,
        top_mod: TopLevelMod<'_>,
        layout: TargetDataLayout,
        opt_level: OptLevel,
    ) -> Result<BackendOutput, BackendError> {
        let package = mir2::build_runtime_package(db, top_mod)?;
        let artifacts = crate::sonatina::emit_runtime_package_sonatina_bytecode(
            db, &package, layout, opt_level,
        )?;
        let object = package
            .primary_object(db)
            .or_else(|| package.root_objects(db).first().copied())
            .ok_or_else(|| BackendError::Sonatina("no root objects to compile".to_string()))?;
        let object_name = object.name(db).clone();
        let contract = artifacts.get(&object_name).ok_or_else(|| {
            BackendError::Sonatina(format!("missing bytecode for `{object_name}`"))
        })?;
        Ok(BackendOutput::Bytecode(contract.runtime.clone()))
    }
}

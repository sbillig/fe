use std::fmt;

use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;

use crate::TargetDataLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    O0,
    O1,
    Os,
    #[default]
    O2,
}

impl std::str::FromStr for OptLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(OptLevel::O0),
            "1" => Ok(OptLevel::O1),
            "s" => Ok(OptLevel::Os),
            "2" => Ok(OptLevel::O2),
            _ => Err(format!(
                "unknown optimization level: {s} (expected '0', '1', '2', or 's')"
            )),
        }
    }
}

impl fmt::Display for OptLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "0"),
            OptLevel::O1 => write!(f, "1"),
            OptLevel::Os => write!(f, "s"),
            OptLevel::O2 => write!(f, "2"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BackendOutput {
    Bytecode(Vec<u8>),
}

impl BackendOutput {
    pub fn as_bytecode(&self) -> Option<&[u8]> {
        match self {
            BackendOutput::Bytecode(bytes) => Some(bytes),
        }
    }

    pub fn into_bytecode(self) -> Option<Vec<u8>> {
        match self {
            BackendOutput::Bytecode(bytes) => Some(bytes),
        }
    }
}

#[derive(Debug)]
pub enum BackendError {
    RuntimeLower(mir::LowerError),
    Sonatina(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::RuntimeLower(err) => write!(f, "{err}"),
            BackendError::Sonatina(message) => write!(f, "sonatina error: {message}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<mir::LowerError> for BackendError {
    fn from(err: mir::LowerError) -> Self {
        BackendError::RuntimeLower(err)
    }
}

impl From<crate::sonatina::LowerError> for BackendError {
    fn from(err: crate::sonatina::LowerError) -> Self {
        BackendError::Sonatina(err.to_string())
    }
}

#[cfg(feature = "cranelift")]
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeBackend;

#[cfg(feature = "cranelift")]
impl Backend for NativeBackend {
    fn name(&self) -> &'static str {
        "native"
    }

    fn compile(
        &self,
        db: &DriverDataBase,
        top_mod: TopLevelMod<'_>,
        layout: TargetDataLayout,
        opt_level: OptLevel,
    ) -> Result<BackendOutput, BackendError> {
        let package = mir::build_runtime_package(db, top_mod)?;
        let module =
            crate::sonatina::compile_runtime_package_sonatina_native(db, &package, layout)?;

        let clif_backend = sonatina_codegen::isa::cranelift::CraneliftBackend::new();
        let compile = sonatina_codegen::Compile::new(module, clif_backend)
            .with_opt_level(crate::sonatina::to_sonatina_opt_level(opt_level));
        let artifact = compile.compile().map_err(|errs| {
            let msgs: Vec<_> = errs.iter().map(|e| format!("{e}")).collect();
            BackendError::Sonatina(msgs.join("; "))
        })?;

        // For now, return empty bytecode — the artifact is JIT-compiled native code.
        // The real output mechanism for native targets is TBD (execute directly,
        // write object file, etc.)
        Ok(BackendOutput::Bytecode(Vec::new()))
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
    #[default]
    Sonatina,
    #[cfg(feature = "cranelift")]
    Native,
}

impl BackendKind {
    pub fn name(&self) -> &'static str {
        match self {
            BackendKind::Sonatina => "sonatina",
            #[cfg(feature = "cranelift")]
            BackendKind::Native => "native",
        }
    }

    pub fn create(&self) -> Box<dyn Backend> {
        match self {
            BackendKind::Sonatina => Box::new(SonatinaBackend),
            #[cfg(feature = "cranelift")]
            BackendKind::Native => Box::new(NativeBackend),
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sonatina" => Ok(BackendKind::Sonatina),
            #[cfg(feature = "cranelift")]
            "native" => Ok(BackendKind::Native),
            _ => Err(format!("unknown backend: {s}")),
        }
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
        let package = mir::build_runtime_package(db, top_mod)?;
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

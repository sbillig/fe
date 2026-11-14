//! The sidecar of a compiled object: the Sonatina IR the EVM backend compiled it from, and the
//! bytes and symbol table of each section of the result, taken from the one compilation. A proof
//! that the bytecode implements the IR starts from it; nothing in it is trusted.

use std::collections::{BTreeMap, HashMap};

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;
use mir::{RuntimePackage, build_runtime_package};
use serde::Serialize;
use sonatina_codegen::object::{ObjectArtifact, SymbolId};
use sonatina_ir::{
    Module,
    inst::data::SymbolRef,
    ir_writer::{IrWrite, ModuleWriter},
    module::ModuleCtx,
};

use super::{
    IsolatedObjectCompiler, LowerError, select_ingot_runtime_packages,
    select_runtime_package_contract,
};
use crate::{OptLevel, runtime_package::ensure_runtime_package_has_roots};

#[derive(Debug, Clone, Serialize)]
pub struct SonatinaSidecar {
    pub object: String,
    /// The module the backend compiled, after optimization, in Sonatina's text format.
    pub ir: String,
    pub sections: Vec<SidecarSection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarSection {
    pub name: String,
    /// The section's bytes, in hex.
    pub bytecode: String,
    /// The symbols the section defines, by offset.
    pub symbols: Vec<SidecarSymbol>,
}

/// A symbol as the IR writes it (`%func`, `$global`, `&embed`, or `.` for the section itself),
/// and the bytes it takes in the section.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarSymbol {
    pub symbol: String,
    pub offset: u32,
    pub size: u32,
}

/// The optimized module, as text, and the context that names its symbols after the module
/// itself is consumed by compilation.
pub(super) struct PostOptModule {
    ir: String,
    ctx: ModuleCtx,
}

impl PostOptModule {
    pub(super) fn new(module: &Module) -> Self {
        Self {
            ir: ModuleWriter::new(module).dump_string(),
            ctx: module.ctx.clone(),
        }
    }

    pub(super) fn sidecar(self, object: &str, artifact: &ObjectArtifact) -> SonatinaSidecar {
        let sections = artifact
            .sections
            .iter()
            .map(|(name, section)| {
                let mut symbols: Vec<_> = section
                    .symtab
                    .iter()
                    .map(|(id, def)| {
                        let symbol = match id {
                            SymbolId::Func(func) => SymbolRef::Func(*func),
                            SymbolId::Global(global) => SymbolRef::Global(*global),
                            SymbolId::Embed(embed) => SymbolRef::Embed(embed.clone()),
                            SymbolId::CurrentSection => SymbolRef::CurrentSection,
                        };
                        let mut text = Vec::new();
                        symbol
                            .write(&mut text, &self.ctx)
                            .expect("in-memory write should not fail");
                        SidecarSymbol {
                            symbol: String::from_utf8(text).expect("symbols are UTF-8"),
                            offset: def.offset,
                            size: def.size,
                        }
                    })
                    .collect();
                symbols.sort_by(|a, b| (a.offset, &a.symbol).cmp(&(b.offset, &b.symbol)));
                SidecarSection {
                    name: name.0.to_string(),
                    bytecode: hex::encode(&section.bytes),
                    symbols,
                }
            })
            .collect();
        SonatinaSidecar {
            object: object.to_string(),
            ir: self.ir,
            sections,
        }
    }
}

/// The sidecar of each root object of `package`, compiled as its bytecode is.
fn emit_runtime_package_sonatina_sidecar(
    db: &DriverDataBase,
    package: &RuntimePackage<'_>,
    opt_level: OptLevel,
) -> Result<BTreeMap<String, SonatinaSidecar>, LowerError> {
    ensure_runtime_package_has_roots(db, package, "Sonatina sidecar")?;
    // Observability gives the symbol tables; it does not change the bytes.
    let mut compiler = IsolatedObjectCompiler {
        db,
        package,
        opt_level,
        emit_observability: true,
        compiled: HashMap::new(),
        trace_owner: None,
        trace_facts: BTreeMap::new(),
        sidecars: Some(HashMap::new()),
    };
    let roots: Vec<_> = package
        .root_objects(db)
        .iter()
        .map(|object| object.name(db).clone())
        .collect();
    compiler.compile(&roots)?;
    let mut sidecars = compiler.sidecars.unwrap_or_default();
    roots
        .into_iter()
        .map(|name| {
            let sidecar = sidecars.remove(&name).ok_or_else(|| {
                LowerError::Internal(format!("no sidecar for compiled object `{name}`"))
            })?;
            Ok((name, sidecar))
        })
        .collect()
}

pub fn emit_module_sonatina_sidecar(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    opt_level: OptLevel,
    contract: Option<&str>,
) -> Result<BTreeMap<String, SonatinaSidecar>, LowerError> {
    let package = build_runtime_package(db, top_mod)?;
    let package = select_runtime_package_contract(db, package, contract)?;
    emit_runtime_package_sonatina_sidecar(db, &package, opt_level)
}

pub fn emit_ingot_sonatina_sidecar(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    opt_level: OptLevel,
    contract: Option<&str>,
) -> Result<BTreeMap<String, SonatinaSidecar>, LowerError> {
    let mut sidecars = BTreeMap::new();
    for package in select_ingot_runtime_packages(db, ingot, contract)? {
        for (name, sidecar) in emit_runtime_package_sonatina_sidecar(db, &package, opt_level)? {
            if sidecars.insert(name.clone(), sidecar).is_some() {
                return Err(LowerError::Internal(format!(
                    "duplicate root object `{name}` across ingot modules"
                )));
            }
        }
    }
    Ok(sidecars)
}

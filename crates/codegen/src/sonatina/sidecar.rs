//! The sidecar of a compiled object: the Sonatina IR the EVM backend compiled it from, and the
//! bytes, symbol table and lowering record of each section of the result, taken from the one
//! compilation. A proof that the bytecode implements the IR starts from it; nothing in it is
//! trusted.
//!
//! In a lowering record values are numbered as `v<n>` and blocks as `block<n>` in the text of
//! the module they belong to: `sources` and `mallocs` name the source IR, everything else the
//! machine IR. Immediates are decimal strings of their unsigned value.

use std::collections::{BTreeMap, HashMap};

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;
use mir::{RuntimePackage, build_runtime_package};
use serde::Serialize;
use sonatina_codegen::object::{
    ObjectArtifact, SymbolId,
    lowering::{
        CodeLabel, MallocLowering, SectionLowering, SpillLocation, StableLowering, StackSlot,
    },
};
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

/// The version of the sidecar format, raised with each change to it.
pub const SIDECAR_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize)]
pub struct SonatinaSidecar {
    pub version: u32,
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
    pub lowering: Option<SidecarLowering>,
}

/// How the backend lowered the section (`SectionLowering` of Sonatina).
#[derive(Debug, Clone, Serialize)]
pub struct SidecarLowering {
    /// The section's module after the backend's IR pipeline, which machine lowering reads.
    pub source_ir: String,
    /// The machine module the section's code is emitted from.
    pub machine_ir: String,
    pub code_size: u32,
    pub arena_base: u32,
    pub dyn_base: u32,
    pub entry_free_ptr: Option<u32>,
    pub funcs: Vec<SidecarFunc>,
    pub units: Vec<SidecarUnit>,
    pub redirects: Vec<SidecarRedirect>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarFunc {
    pub name: String,
    pub offset: Option<u32>,
    pub blocks: Vec<SidecarBlock>,
    pub spills: Vec<SidecarSpill>,
    pub frame: SidecarFrame,
    pub mallocs: Vec<SidecarMalloc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarBlock {
    pub block: u32,
    pub sources: Vec<u32>,
    pub offset: Option<u32>,
    pub alias: Option<u32>,
    /// Top first.
    pub entry_stack: Vec<SidecarSlot>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarSlot {
    Value(u32),
    Imm(String),
    FuncReturn,
    CallReturn,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarSpill {
    pub value: u32,
    pub location: SidecarLocation,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarLocation {
    Memory(u32),
    Frame(u32),
    Rematerialized,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarFrame {
    pub stable: SidecarStable,
    pub scratch_words: u32,
    pub stable_words: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarStable {
    None,
    Abs { base_word: u32 },
    DynamicFrame,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarMalloc {
    pub value: u32,
    pub placement: SidecarPlacement,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarPlacement {
    Fixed {
        base: u32,
    },
    Heap {
        min_base: u32,
        exact_base: Option<u32>,
        update_free_ptr: bool,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarUnit {
    pub name: String,
    pub offset: u32,
    pub size: u32,
}

/// Code that went to `from` goes to `to`.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarRedirect {
    pub from: SidecarLabel,
    pub to: SidecarLabel,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarLabel {
    Block { func: String, block: u32 },
    Func(String),
    Unit(String),
    Inst,
}

impl From<&SectionLowering> for SidecarLowering {
    fn from(lowering: &SectionLowering) -> Self {
        let label = |label: &CodeLabel| match label {
            CodeLabel::Block { func, block } => SidecarLabel::Block {
                func: func.clone(),
                block: block.as_u32(),
            },
            CodeLabel::Func(func) => SidecarLabel::Func(func.clone()),
            CodeLabel::Unit(unit) => SidecarLabel::Unit(unit.clone()),
            CodeLabel::Inst => SidecarLabel::Inst,
        };
        let funcs = lowering
            .funcs
            .iter()
            .map(|func| SidecarFunc {
                name: func.name.clone(),
                offset: func.offset,
                blocks: func
                    .blocks
                    .iter()
                    .map(|block| SidecarBlock {
                        block: block.block.as_u32(),
                        sources: block.sources.iter().map(|source| source.as_u32()).collect(),
                        offset: block.offset,
                        alias: block.alias.map(|alias| alias.as_u32()),
                        entry_stack: block
                            .entry_stack
                            .iter()
                            .map(|slot| match slot {
                                StackSlot::Value(value) => SidecarSlot::Value(value.as_u32()),
                                StackSlot::Imm(imm) => {
                                    SidecarSlot::Imm(imm.as_i256().to_u256().to_string())
                                }
                                StackSlot::FuncReturn => SidecarSlot::FuncReturn,
                                StackSlot::CallReturn => SidecarSlot::CallReturn,
                            })
                            .collect(),
                    })
                    .collect(),
                spills: func
                    .spills
                    .iter()
                    .map(|(value, location)| SidecarSpill {
                        value: value.as_u32(),
                        location: match *location {
                            SpillLocation::Memory(addr) => SidecarLocation::Memory(addr),
                            SpillLocation::Frame(slot) => SidecarLocation::Frame(slot),
                            SpillLocation::Rematerialized => SidecarLocation::Rematerialized,
                        },
                    })
                    .collect(),
                frame: SidecarFrame {
                    stable: match func.frame.stable {
                        StableLowering::None => SidecarStable::None,
                        StableLowering::Abs { base_word } => SidecarStable::Abs { base_word },
                        StableLowering::DynamicFrame => SidecarStable::DynamicFrame,
                    },
                    scratch_words: func.frame.scratch_words,
                    stable_words: func.frame.stable_words,
                },
                mallocs: func
                    .mallocs
                    .iter()
                    .map(|(value, malloc)| SidecarMalloc {
                        value: value.as_u32(),
                        placement: match *malloc {
                            MallocLowering::Fixed { base } => SidecarPlacement::Fixed { base },
                            MallocLowering::Heap {
                                min_base,
                                exact_base,
                                update_free_ptr,
                            } => SidecarPlacement::Heap {
                                min_base,
                                exact_base,
                                update_free_ptr,
                            },
                        },
                    })
                    .collect(),
            })
            .collect();
        Self {
            source_ir: lowering.source_ir.clone(),
            machine_ir: lowering.machine_ir.clone(),
            code_size: lowering.code_size,
            arena_base: lowering.arena_base,
            dyn_base: lowering.dyn_base,
            entry_free_ptr: lowering.entry_free_ptr,
            funcs,
            units: lowering
                .units
                .iter()
                .map(|(name, offset, size)| SidecarUnit {
                    name: name.clone(),
                    offset: *offset,
                    size: *size,
                })
                .collect(),
            redirects: lowering
                .redirects
                .iter()
                .map(|(from, to)| SidecarRedirect {
                    from: label(from),
                    to: label(to),
                })
                .collect(),
        }
    }
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
                    lowering: section.lowering.as_ref().map(SidecarLowering::from),
                }
            })
            .collect();
        SonatinaSidecar {
            version: SIDECAR_VERSION,
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

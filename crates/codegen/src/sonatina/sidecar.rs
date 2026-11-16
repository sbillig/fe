//! The sidecar of a compiled object: the Sonatina IR the EVM backend compiled it from, and the
//! bytes, symbol table and lowering record of each section of the result, taken from the one
//! compilation. A proof that the bytecode implements the IR starts from it; nothing in it is
//! trusted.
//!
//! In a lowering record values are numbered as `v<n>` and blocks as `block<n>` in the text of
//! the module they belong to: `sources` and `mallocs` name the source IR, everything else the
//! machine IR. Immediates are decimal strings of their unsigned value. A point of a function's
//! code is `"entry"`, or `{"before": {"block", "index"}}`, `{"after": {"block", "index"}}` or
//! `{"case": {"block", "index", "case"}}`, where `index` is an instruction's position in its
//! block, phis included.

use std::collections::{BTreeMap, HashMap};

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;
use mir::{RuntimePackage, build_runtime_package};
use serde::Serialize;
use sonatina_codegen::object::{
    ObjectArtifact,
    lowering::{
        CodeLabel, CodePoint, DynSpInitLowering, MallocLowering, SectionLowering, SpillLocation,
        StableLowering, StackSlot,
    },
};

use super::{
    IsolatedObjectCompiler, LowerError, select_ingot_runtime_packages,
    select_runtime_package_contract,
};
use crate::{OptLevel, runtime_package::ensure_runtime_package_has_roots};

/// The version of the sidecar format, raised with each change to it.
pub const SIDECAR_VERSION: u32 = 4;

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
    /// The symbols the section defines, by offset, as the section's IR (`lowering.source_ir`)
    /// names them.
    pub symbols: Vec<SidecarSymbol>,
    pub lowering: SidecarLowering,
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
    /// The points at which the code stores the value into its location, in emission order:
    /// after its definition, at entry for an argument, and before the terminator of each
    /// predecessor (an edge block, for a critical edge) that gives a phi an operand other than
    /// the phi itself. The stack top holds the value stored: a phi's operand on its edge.
    pub stores: Vec<SidecarPoint>,
    /// The points at which the code reloads the value from its location onto the stack, in
    /// emission order. A function without code of its own has no points.
    pub reloads: Vec<SidecarPoint>,
}

/// A point of a function's machine IR at which its code runs the stack allocator's moves.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarPoint {
    /// Function entry, before the entry block's first instruction.
    Entry,
    /// Before the instruction's opcodes: its operands, and for a terminator the moves into its
    /// successors.
    Before { block: u32, index: u32 },
    /// After the instruction's opcodes, where its results are stored.
    After { block: u32, index: u32 },
    /// Before the comparison of case `case` of the `br_table` instruction.
    Case { block: u32, index: u32, case: u32 },
}

impl From<&CodePoint> for SidecarPoint {
    fn from(point: &CodePoint) -> Self {
        match *point {
            CodePoint::Entry => Self::Entry,
            CodePoint::Before { block, index } => Self::Before {
                block: block.as_u32(),
                index,
            },
            CodePoint::After { block, index } => Self::After {
                block: block.as_u32(),
                index,
            },
            CodePoint::Case { block, index, case } => Self::Case {
                block: block.as_u32(),
                index,
                case,
            },
        }
    }
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
    pub lazy_frame: bool,
    /// `always` or `checked` when the function's entry initializes the dynamic stack pointer.
    pub dyn_sp_entry_init: Option<String>,
    pub dyn_sp_init_calls: u32,
    /// The calls, as block and position in it, around which scratch objects are saved.
    pub call_preserves: Vec<(u32, u32)>,
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
                    .map(|spill| SidecarSpill {
                        value: spill.value.as_u32(),
                        location: match spill.location {
                            SpillLocation::Memory(addr) => SidecarLocation::Memory(addr),
                            SpillLocation::Frame(slot) => SidecarLocation::Frame(slot),
                            SpillLocation::Rematerialized => SidecarLocation::Rematerialized,
                        },
                        stores: spill.stores.iter().map(SidecarPoint::from).collect(),
                        reloads: spill.reloads.iter().map(SidecarPoint::from).collect(),
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
                    lazy_frame: func.frame.lazy_frame,
                    dyn_sp_entry_init: func.frame.dyn_sp_entry_init.map(|kind| {
                        match kind {
                            DynSpInitLowering::Always => "always",
                            DynSpInitLowering::Checked => "checked",
                        }
                        .to_string()
                    }),
                    dyn_sp_init_calls: func.frame.dyn_sp_init_calls,
                    call_preserves: func
                        .frame
                        .call_preserves
                        .iter()
                        .map(|(block, position)| (block.as_u32(), *position))
                        .collect(),
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

impl SonatinaSidecar {
    /// The sidecar of `object`, compiled to `artifact` from the module whose text is `ir`.
    pub(super) fn new(
        ir: String,
        object: &str,
        artifact: &ObjectArtifact,
    ) -> Result<Self, LowerError> {
        let sections = artifact
            .sections
            .iter()
            .map(|(name, section)| {
                let lowering = section.lowering.as_ref().ok_or_else(|| {
                    LowerError::Internal(format!(
                        "section `{}` of `{object}` has no lowering record",
                        name.0
                    ))
                })?;
                Ok(SidecarSection {
                    name: name.0.to_string(),
                    bytecode: hex::encode(&section.bytes),
                    symbols: lowering
                        .symbols
                        .iter()
                        .map(|(symbol, def)| SidecarSymbol {
                            symbol: symbol.clone(),
                            offset: def.offset,
                            size: def.size,
                        })
                        .collect(),
                    lowering: lowering.into(),
                })
            })
            .collect::<Result<_, LowerError>>()?;
        Ok(Self {
            version: SIDECAR_VERSION,
            object: object.to_string(),
            ir,
            sections,
        })
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

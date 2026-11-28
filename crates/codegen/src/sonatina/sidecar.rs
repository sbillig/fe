//! The sidecar of a compiled object: the Sonatina IR the EVM backend compiled it from, and the
//! bytes, symbol table and lowering record of each section of the result, taken from the one
//! compilation. A proof that the bytecode implements the IR starts from it; nothing in it is
//! trusted.
//!
//! In a lowering record values are numbered as `v<n>` and blocks as `block<n>` in the text of
//! the module they belong to: `sources`, `mallocs` and `allocas` name the source IR, everything
//! else the machine IR. Immediates and other 256-bit words are decimal strings of their unsigned value;
//! only the signed bounds of a range are signed. A point of a function's code is `"entry"`, or
//! `{"before": {"block", "index"}}`, `{"after": {"block", "index"}}` or
//! `{"case": {"block", "index", "case"}}`, where `index` is an instruction's position in its
//! block, phis included. In `machine_steps`, the values, blocks and instructions of a step's
//! facts are those of the function before the step, except the inserted phis of `phis` and
//! `value_phis`, which are of the function after it.
//!
//! Version 6 adds the placement record: each function's `lowered_blocks` and `allocas`
//! (location and size), the `rule` of each fixed malloc, `needs_dyn_sp_clamp` and
//! `free_ptr_floor` of each heap malloc, each function's `restores_free_ptr` and
//! `spill_reserve`, and the section's `fixed_slot_spills`. Frame words are counted from the
//! arena base: frame word `w` is the bytes at `arena_base + 32 * w`.

use std::collections::{BTreeMap, HashMap};

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::TopLevelMod;
use mir::{RuntimePackage, build_runtime_package};
use serde::Serialize;
use sonatina_codegen::{
    isa::evm::LateCleanupProfile,
    object::{
        ObjectArtifact,
        lowering::{
            CodeLabel, CodePoint, DynSpInitLowering, FixedMallocRule, FuncSnap, GvnOperand,
            GvnTerm, MachinePassKind, MachineSteps, MallocLowering, ObjLocLowering,
            SectionLowering, SpillLocation, StableLowering, StackSlot, StepFacts, StepKind,
            WordRange,
        },
    },
};
use sonatina_ir::{BlockId, Immediate};

use super::{
    IsolatedObjectCompiler, LowerError, select_ingot_runtime_packages,
    select_runtime_package_contract,
};
use crate::{OptLevel, runtime_package::ensure_runtime_package_has_roots};

/// The version of the sidecar format, raised with each change to it.
pub const SIDECAR_VERSION: u32 = 6;

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
    /// Some function spills to the fixed slots, the words at `0` and `0x20`.
    pub fixed_slot_spills: bool,
    pub funcs: Vec<SidecarFunc>,
    pub units: Vec<SidecarUnit>,
    pub redirects: Vec<SidecarRedirect>,
    /// How the machine stage turned the machine IR lowering built into `machine_ir`.
    pub machine_steps: SidecarMachineSteps,
}

/// Each step of the machine stage (`MachineSteps` of Sonatina): `lowered`, with the `out` of
/// each step's functions put in place in order, is `machine_ir`.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarMachineSteps {
    /// `off`, `speed` or `size`.
    pub profile: &'static str,
    /// Every function of the section as lowering built it, in the section's order.
    pub lowered: Vec<SidecarFuncSnap>,
    /// The passes in the order they ran.
    pub passes: Vec<SidecarMachinePass>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarMachinePass {
    /// The pass, as the snake case of `MachinePassKind`.
    pub kind: &'static str,
    /// The behavior the pass read of every function of the module, if it reads any.
    pub func_behavior: Option<Vec<SidecarFuncSummary>>,
    /// The steps that changed some function, in order.
    pub steps: Vec<SidecarMachineStep>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarFuncSummary {
    pub func: String,
    pub noreturn: bool,
    pub will_return: bool,
    pub reads: bool,
    pub writes: bool,
    pub mutates: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarMachineStep {
    /// The step, as the snake case of `StepKind`.
    pub kind: &'static str,
    /// The functions the step changed, in the section's order.
    pub funcs: Vec<SidecarFuncStep>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarFuncStep {
    /// The whole function after the step.
    pub out: SidecarFuncSnap,
    pub facts: SidecarStepFacts,
}

/// A function of the machine module at one point of the machine stage.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarFuncSnap {
    pub func: String,
    pub func_ref: u32,
    /// The function's IR text, values and blocks named by their ids.
    pub text: String,
    /// The instruction id of each line of each block of `text`, phis and terminators included.
    pub insts: Vec<Vec<u32>>,
}

/// What a step relied on, of the function before it. A value without a fact is unknown, except
/// that a value `demanded_bits` does not list has no bit demanded.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarStepFacts {
    /// `[value, known_zero, known_one]`.
    pub known_bits: Vec<(u32, String, String)>,
    /// `[value, mask]`.
    pub demanded_bits: Option<Vec<(u32, String)>>,
    pub sccp: Option<SidecarSccpFacts>,
    pub finite_loops: Vec<SidecarFiniteLoop>,
    pub gvn: Option<SidecarGvnFacts>,
    pub ranges: Option<SidecarRangeFacts>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarSccpFacts {
    /// `[value, constant]`.
    pub consts: Vec<(u32, String)>,
    /// `[block, slot]`, slots numbered in the order of the terminator's destinations.
    pub dead_edges: Vec<(u32, u32)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarFiniteLoop {
    pub header: u32,
    pub latch: u32,
    pub iv: u32,
    pub step: u32,
    pub exit: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarGvnFacts {
    /// `[block, slot]`.
    pub dead_edges: Vec<(u32, u32)>,
    /// `[value, representative]`.
    pub class: Vec<(u32, SidecarGvnOperand)>,
    /// `[block, slot, lhs, rhs]`: `lhs` equals `rhs` once the edge is taken.
    pub preds: Vec<(u32, u32, SidecarGvnOperand, SidecarGvnOperand)>,
    /// The phis the step inserted, of the function after it, in the order inserted.
    pub phis: Vec<SidecarInsertedPhi>,
    /// `[value, phi]`: the step replaced `value` with `phi`, one of `phis`.
    pub value_phis: Vec<(u32, u32)>,
}

/// A phi a `gvn_rewrite` step inserted at the top of `block`, with the term of the step input
/// it stands for there, if GVN's reasoning reduces to one.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarInsertedPhi {
    pub phi: u32,
    pub block: u32,
    pub term: Option<SidecarGvnTerm>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarGvnTerm {
    Value(u32),
    Imm(String),
    /// The operation of instruction `inst` of the step input, on `args` in place of its
    /// operands.
    Inst {
        inst: u32,
        args: Vec<SidecarGvnTerm>,
    },
}

impl From<&GvnTerm> for SidecarGvnTerm {
    fn from(term: &GvnTerm) -> Self {
        match term {
            GvnTerm::Value(value) => Self::Value(value.as_u32()),
            GvnTerm::Imm(imm) => Self::Imm(word(*imm)),
            GvnTerm::Inst { inst, args } => Self::Inst {
                inst: inst.as_u32(),
                args: args.iter().map(Self::from).collect(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarGvnOperand {
    Value(u32),
    Imm(String),
}

impl From<&GvnOperand> for SidecarGvnOperand {
    fn from(operand: &GvnOperand) -> Self {
        match *operand {
            GvnOperand::Value(value) => Self::Value(value.as_u32()),
            GvnOperand::Imm(imm) => Self::Imm(word(imm)),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarRangeFacts {
    pub unreachable: Vec<u32>,
    /// `[block, ranges]`, the ranges of values at the block's entry.
    pub entry: Vec<(u32, Vec<SidecarValueRange>)>,
    /// Instruction ids.
    pub guarded_subs: Vec<u32>,
}

/// The value is in both intervals, bounds included; `signed` bounds are signed decimals.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarValueRange {
    pub value: u32,
    pub unsigned: (String, String),
    pub signed: (String, String),
}

/// An immediate as a decimal string of its unsigned value.
fn word(imm: Immediate) -> String {
    imm.as_i256().to_u256().to_string()
}

impl From<&FuncSnap> for SidecarFuncSnap {
    fn from(snap: &FuncSnap) -> Self {
        Self {
            func: snap.func.clone(),
            func_ref: snap.func_ref.as_u32(),
            text: snap.text.clone(),
            insts: snap
                .insts
                .iter()
                .map(|block| block.iter().map(|inst| inst.as_u32()).collect())
                .collect(),
        }
    }
}

impl From<&StepFacts> for SidecarStepFacts {
    fn from(facts: &StepFacts) -> Self {
        let edges = |edges: &[(BlockId, u32)]| {
            edges
                .iter()
                .map(|&(block, slot)| (block.as_u32(), slot))
                .collect()
        };
        Self {
            known_bits: facts
                .known_bits
                .iter()
                .map(|(value, zero, one)| (value.as_u32(), zero.to_string(), one.to_string()))
                .collect(),
            demanded_bits: facts.demanded_bits.as_ref().map(|demanded| {
                demanded
                    .iter()
                    .map(|(value, mask)| (value.as_u32(), mask.to_string()))
                    .collect()
            }),
            sccp: facts.sccp.as_ref().map(|sccp| SidecarSccpFacts {
                consts: sccp
                    .consts
                    .iter()
                    .map(|&(value, imm)| (value.as_u32(), word(imm)))
                    .collect(),
                dead_edges: edges(&sccp.dead_edges),
            }),
            finite_loops: facts
                .finite_loops
                .iter()
                .map(|finite| SidecarFiniteLoop {
                    header: finite.header.as_u32(),
                    latch: finite.latch.as_u32(),
                    iv: finite.iv.as_u32(),
                    step: finite.step.as_u32(),
                    exit: finite.exit.as_u32(),
                })
                .collect(),
            gvn: facts.gvn.as_ref().map(|gvn| SidecarGvnFacts {
                dead_edges: edges(&gvn.dead_edges),
                class: gvn
                    .class
                    .iter()
                    .map(|(value, rep)| (value.as_u32(), rep.into()))
                    .collect(),
                preds: gvn
                    .preds
                    .iter()
                    .map(|(block, slot, lhs, rhs)| (block.as_u32(), *slot, lhs.into(), rhs.into()))
                    .collect(),
                phis: gvn
                    .phis
                    .iter()
                    .map(|inserted| SidecarInsertedPhi {
                        phi: inserted.phi.as_u32(),
                        block: inserted.block.as_u32(),
                        term: inserted.term.as_ref().map(SidecarGvnTerm::from),
                    })
                    .collect(),
                value_phis: gvn
                    .value_phis
                    .iter()
                    .map(|(value, phi)| (value.as_u32(), phi.as_u32()))
                    .collect(),
            }),
            ranges: facts.ranges.as_ref().map(|ranges| SidecarRangeFacts {
                unreachable: ranges
                    .unreachable
                    .iter()
                    .map(|block| block.as_u32())
                    .collect(),
                entry: ranges
                    .entry
                    .iter()
                    .map(|(block, env)| {
                        let env = env
                            .iter()
                            .map(|range| SidecarValueRange {
                                value: range.value.as_u32(),
                                unsigned: (
                                    range.unsigned.0.to_string(),
                                    range.unsigned.1.to_string(),
                                ),
                                signed: (range.signed.0.to_string(), range.signed.1.to_string()),
                            })
                            .collect();
                        (block.as_u32(), env)
                    })
                    .collect(),
                guarded_subs: ranges
                    .guarded_subs
                    .iter()
                    .map(|inst| inst.as_u32())
                    .collect(),
            }),
        }
    }
}

impl From<&MachineSteps> for SidecarMachineSteps {
    fn from(steps: &MachineSteps) -> Self {
        Self {
            profile: match steps.profile {
                LateCleanupProfile::Off => "off",
                LateCleanupProfile::Speed => "speed",
                LateCleanupProfile::Size => "size",
            },
            lowered: steps.lowered.iter().map(SidecarFuncSnap::from).collect(),
            passes: steps
                .passes
                .iter()
                .map(|pass| SidecarMachinePass {
                    kind: match pass.kind {
                        MachinePassKind::CfgCleanup => "cfg_cleanup",
                        MachinePassKind::BranchCanonicalize => "branch_canonicalize",
                        MachinePassKind::ScalarCanonicalize => "scalar_canonicalize",
                        MachinePassKind::KnownBitsSimplify => "known_bits_simplify",
                        MachinePassKind::Sccp => "sccp",
                        MachinePassKind::Gvn => "gvn",
                        MachinePassKind::Adce => "adce",
                        MachinePassKind::LowerSwitches => "lower_switches",
                        MachinePassKind::BranchConditions => "branch_conditions",
                        MachinePassKind::CoalesceCopies => "coalesce_copies",
                        MachinePassKind::EdgeSplit => "edge_split",
                    },
                    func_behavior: pass.func_behavior.as_ref().map(|summaries| {
                        summaries
                            .iter()
                            .map(|summary| SidecarFuncSummary {
                                func: summary.func.clone(),
                                noreturn: summary.noreturn,
                                will_return: summary.will_return,
                                reads: summary.reads,
                                writes: summary.writes,
                                mutates: summary.mutates,
                            })
                            .collect()
                    }),
                    steps: pass
                        .steps
                        .iter()
                        .map(|step| SidecarMachineStep {
                            kind: match step.kind {
                                StepKind::CfgCleanup => "cfg_cleanup",
                                StepKind::BranchCanonicalize => "branch_canonicalize",
                                StepKind::ScalarCanonicalize => "scalar_canonicalize",
                                StepKind::KnownBitsSimplify => "known_bits_simplify",
                                StepKind::SccpPrune => "sccp_prune",
                                StepKind::SccpFold => "sccp_fold",
                                StepKind::Adce => "adce",
                                StepKind::GvnPrune => "gvn_prune",
                                StepKind::GvnRewrite => "gvn_rewrite",
                                StepKind::LowerSwitches => "lower_switches",
                                StepKind::BranchConditions => "branch_conditions",
                                StepKind::CoalesceCopies => "coalesce_copies",
                                StepKind::EdgeSplit => "edge_split",
                            },
                            funcs: step
                                .funcs
                                .iter()
                                .map(|func| SidecarFuncStep {
                                    out: (&func.out).into(),
                                    facts: (&func.facts).into(),
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarFunc {
    pub name: String,
    pub offset: Option<u32>,
    pub blocks: Vec<SidecarBlock>,
    /// `[source block, lowered block]`: each block of the source IR, in layout order, with the
    /// block of `machine_steps.lowered` lowering made for it.
    pub lowered_blocks: Vec<(u32, u32)>,
    pub spills: Vec<SidecarSpill>,
    pub frame: SidecarFrame,
    pub mallocs: Vec<SidecarMalloc>,
    pub allocas: Vec<SidecarAlloca>,
    /// Each `return` stores back at `0x40` the free pointer the function loaded at entry.
    pub restores_free_ptr: bool,
    pub spill_reserve: SidecarSpillReserve,
}

/// An `alloca`'s `words` words at `loc`, its type's size rounded up to a word.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarAlloca {
    pub value: u32,
    pub loc: SidecarObjLoc,
    pub words: u32,
}

/// A frame word, counted from the arena base: `scratch_abs` (in the scratch overlay) and
/// `stable_abs` (in the stable frame) word `w` is at `arena_base + 32 * w`; `stable_frame` is a
/// word of the dynamic frame.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarObjLoc {
    ScratchAbs(u32),
    StableAbs(u32),
    StableFrame(u32),
}

/// The words reserved for the function's spills; `stable` is `None` without an `abs` stable
/// frame.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarSpillReserve {
    pub scratch: SidecarWordRange,
    pub stable: Option<SidecarWordRange>,
}

/// The frame words `[start, start + words)`, frame word `w` being the bytes at
/// `arena_base + 32 * w`.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarWordRange {
    pub start: u32,
    pub words: u32,
}

impl From<WordRange> for SidecarWordRange {
    fn from(range: WordRange) -> Self {
        Self {
            start: range.start,
            words: range.words,
        }
    }
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
    /// the phi itself, except where the edge would reload that operand from the phi's own
    /// location. The stack top holds the value stored, a phi's operand on its edge, and the
    /// store may take it off the stack.
    pub stores: Vec<SidecarPoint>,
    /// The points at which the code reloads the value from its location onto the stack, in
    /// emission order. A reload of a phi's operand from the phi's own location, for the phi's
    /// store on the edge, is left out with that store, so a value may have stores and no
    /// reloads. A function without code of its own has no points.
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
        /// `terminal`, `program`, `exact_entry` or `private_static`.
        rule: &'static str,
    },
    Heap {
        min_base: u32,
        exact_base: Option<u32>,
        update_free_ptr: bool,
        /// The base is also raised to the dynamic stack pointer at `0x80`.
        needs_dyn_sp_clamp: bool,
        /// A lower bound of the free pointer before the malloc; at least `min_base`, it drops
        /// the raise to `min_base`.
        free_ptr_floor: Option<u32>,
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
                                StackSlot::Imm(imm) => SidecarSlot::Imm(word(*imm)),
                                StackSlot::FuncReturn => SidecarSlot::FuncReturn,
                                StackSlot::CallReturn => SidecarSlot::CallReturn,
                            })
                            .collect(),
                    })
                    .collect(),
                lowered_blocks: func
                    .lowered_blocks
                    .iter()
                    .map(|(source, lowered)| (source.as_u32(), lowered.as_u32()))
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
                            MallocLowering::Fixed { base, rule } => SidecarPlacement::Fixed {
                                base,
                                rule: match rule {
                                    FixedMallocRule::Terminal => "terminal",
                                    FixedMallocRule::Program => "program",
                                    FixedMallocRule::ExactEntry => "exact_entry",
                                    FixedMallocRule::PrivateStatic => "private_static",
                                },
                            },
                            MallocLowering::Heap {
                                min_base,
                                exact_base,
                                update_free_ptr,
                                needs_dyn_sp_clamp,
                                free_ptr_floor,
                            } => SidecarPlacement::Heap {
                                min_base,
                                exact_base,
                                update_free_ptr,
                                needs_dyn_sp_clamp,
                                free_ptr_floor,
                            },
                        },
                    })
                    .collect(),
                allocas: func
                    .allocas
                    .iter()
                    .map(|(value, alloca)| SidecarAlloca {
                        value: value.as_u32(),
                        loc: match alloca.loc {
                            ObjLocLowering::ScratchAbs(word) => SidecarObjLoc::ScratchAbs(word),
                            ObjLocLowering::StableAbs(word) => SidecarObjLoc::StableAbs(word),
                            ObjLocLowering::StableFrame(word) => SidecarObjLoc::StableFrame(word),
                        },
                        words: alloca.words,
                    })
                    .collect(),
                restores_free_ptr: func.restores_free_ptr,
                spill_reserve: SidecarSpillReserve {
                    scratch: func.spill_reserve.scratch.into(),
                    stable: func.spill_reserve.stable.map(SidecarWordRange::from),
                },
            })
            .collect();
        Self {
            source_ir: lowering.source_ir.clone(),
            machine_ir: lowering.machine_ir.clone(),
            code_size: lowering.code_size,
            arena_base: lowering.arena_base,
            dyn_base: lowering.dyn_base,
            entry_free_ptr: lowering.entry_free_ptr,
            fixed_slot_spills: lowering.fixed_slot_spills,
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
            machine_steps: (&lowering.machine_steps).into(),
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

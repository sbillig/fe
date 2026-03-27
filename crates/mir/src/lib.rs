pub mod analysis;
mod capability_space;
pub mod const_data;
mod core_lib;
mod dedup;
pub mod fmt;
mod hash;
pub mod ir;
pub mod layout;
mod lower;
mod monomorphize;
pub mod repr;
mod transform;
mod ty;

pub use const_data::{
    ConstData, pack_inline_string_word, serialize_const_data_to_bytes,
    serialize_const_u8_array_bytes,
};
pub use core_lib::CoreLib;
pub use ir::{
    BasicBlockId, CallOrigin, CallTargetRef, CodeRegionRef, ConstRegion, ConstRegionId, LocalData,
    LocalId, LoopInfo, MirBackend, MirBody, MirFunction, MirInst, MirModule, MirProjection,
    MirProjectionPath, MirStage, Rvalue, SwitchTarget, SwitchValue, TerminatingCall, Terminator,
    ValueData, ValueId, ValueOrigin, ValueRepr,
};
pub use lower::{
    MirDiagnosticsMode, MirDiagnosticsOutput, MirLowerError, MirLowerResult,
    collect_mir_diagnostics, lower_ingot, lower_module,
};
pub use transform::prepare_module_for_evm_yul_codegen;

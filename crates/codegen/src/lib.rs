mod backend;
mod function_symbols;
mod layout;
mod runtime_package;
pub mod sonatina;
mod test_output;

#[cfg(feature = "cranelift")]
pub use backend::NativeBackend;
pub use backend::{Backend, BackendError, BackendKind, BackendOutput, OptLevel, SonatinaBackend};
pub use layout::{DISCRIMINANT_SIZE_BYTES, EVM_LAYOUT, TargetDataLayout, WORD_SIZE_BYTES};
pub use sonatina::{
    LowerError, SonatinaContractBytecode, SonatinaTestOptions, emit_ingot_sonatina_bytecode,
    emit_ingot_sonatina_ir, emit_ingot_sonatina_ir_optimized, emit_module_sonatina_bytecode,
    emit_module_sonatina_ir, emit_module_sonatina_ir_native, emit_module_sonatina_ir_optimized,
    emit_runtime_package_sonatina_ir, emit_runtime_package_sonatina_ir_optimized,
    emit_test_ingot_sonatina, emit_test_module_sonatina, validate_module_sonatina_ir,
};
#[cfg(feature = "cranelift")]
pub use sonatina::{
    NativeMainAbi, NativeObject, NativeTestMetadata, NativeTestModuleOutput, Sp1Elf,
    emit_ingot_native_ir, emit_ingot_native_object, emit_ingot_native_object_with_abi,
    emit_ingot_sp1_elf, emit_ingot_sp1_ir, emit_module_native_ir, emit_module_native_object,
    emit_module_native_object_with_abi, emit_module_sp1_elf, emit_module_sp1_ir,
    emit_test_ingot_native, emit_test_module_native,
};
pub use test_output::{ExpectedRevert, TestMetadata, TestModuleOutput, parse_expected_revert};

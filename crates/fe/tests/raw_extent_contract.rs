use contract_harness::{ExecutionOptions, HarnessError, RuntimeInstance};
use fe::bench_support::compile_fe_sonatina_bytecode;

#[test]
fn checked_slice_halts_before_accessing_an_invalid_complete_range() {
    let source = r#"
use core::ptr::FixedMemBuffer
msg RangeMsg {
    #[selector = 1]
    Check { len: u256 } -> u256,
}
pub contract RangeContract {
    recv RangeMsg {
        Check { len } -> u256 {
            let buffer = FixedMemBuffer<32>::alloc()
            buffer.span().slice(offset: 16, len).len()
        }
    }
}
"#;
    let bytecode = compile_fe_sonatina_bytecode(source, "raw_extent", "RangeContract")
        .expect("compile checked range contract");
    let mut runtime = RuntimeInstance::deploy(&hex::encode(bytecode.deploy)).expect("deploy");
    let mut calldata = vec![0u8; 36];
    calldata[3] = 1;
    calldata[35] = 16;
    let valid = runtime
        .call_raw(&calldata, ExecutionOptions::default())
        .expect("complete in-bounds range");
    assert_eq!(valid.return_data.len(), 32);
    assert_eq!(valid.return_data[31], 16);

    calldata[35] = 17;
    let error = runtime
        .call_raw(&calldata, ExecutionOptions::default())
        .expect_err("range crossing the allocation end must halt");
    // Bare core::panic is an EVM halt, distinct from the test macro's revert.
    assert!(
        matches!(&error, HarnessError::Halted { reason, .. } if format!("{reason:?}") == "InvalidFEOpcode"),
        "{error:?}"
    );
}

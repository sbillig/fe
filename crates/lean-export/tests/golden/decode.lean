export Lib (Evm_decode_args_tuple_u256_u256)

/-- The `init` of contract `Decoder`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Probe -> u256`, selector `0xb74af5a9` (`probe()`) -/
def recv_Probe : FeM UInt256 := do
  let args ← Evm_decode_args_tuple_u256_u256
  let v3 := args.fst
  let v4 := args.snd
  let v5 ← checkedAdd v3 v4
  return v5


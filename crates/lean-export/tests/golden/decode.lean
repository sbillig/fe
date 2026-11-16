export Lib (Evm_decode_args_tuple_u256_u256)

def init : FeM Unit := do
  return ()

def recv_Probe : FeM UInt256 := do
  let args ← Evm_decode_args_tuple_u256_u256
  let v3 := args.fst
  let v4 := args.snd
  let v5 ← checkedAdd v3 v4
  return v5


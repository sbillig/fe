export Lib (Evm_decode_args_tuple_u256_u256)

def init : FeM Unit := do
  return ()

def recv0_arm0 : FeM UInt256 := do
  let v1 ← Evm_decode_args_tuple_u256_u256
  let v2 := v1
  let v3 := v2.fst
  let v4 := v2.snd
  let v5 ← checkedAdd v3 v4
  return v5


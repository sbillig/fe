export Lib (calldata_ptr_Evm_array_u256_2)

def init : FeM Unit := do
  return ()

def second_array_u256_2 (e0 : (Vector UInt256 2)) : FeM UInt256 := do
  let mut e0 := e0
  let v0 := (UInt256.ofNat 1)
  checkIndex v0 2
  let v1 := e0[v0.toNat]!
  return v1

def recv_Second : FeM UInt256 := do
  let mut r1 : UInt256 := default
  let data ← calldata_ptr_Evm_array_u256_2 (UInt256.ofNat 4)
  r1 := data
  let v5 := r1
  let v6_e0 ← cdloadArray 32 (fun q => cdload q) v5 2
  let v6 ← second_array_u256_2 v6_e0
  return v6


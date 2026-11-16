export Lib (calldata_ptr_Evm_array_u256_2)

/-- The `init` of contract `Calldata`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `calldata_read::second<[u256; 2]>` -/
def second_array_u256_2 (words : (Vector UInt256 2)) : FeM UInt256 := do
  let mut words := words
  -- words[1]
  let v0 := (UInt256.ofNat 1)
  checkIndex v0 2
  return words[v0.toNat]!

/-- `Second { .. } -> u256`, selector `0xc7a76601` (`second(uint256,uint256)`) -/
def recv_Second : FeM UInt256 := do
  let mut r1 : UInt256 := default
  -- evm.calldata_ptr<[u256; 2]>(4)
  let data ← calldata_ptr_Evm_array_u256_2 (UInt256.ofNat 4)
  -- with (*data) { second() }
  r1 := data
  -- second()
  let v6_e0 ← cdloadArray 32 (fun q => cdload q) r1 2
  let v6 ← second_array_u256_2 v6_e0
  return v6


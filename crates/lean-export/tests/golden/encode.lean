export Lib (revert_u256)

/-- The `init` of contract `Encoder`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Probe { value } -> u256`, selector `0xdb082440` (`probe(uint256)`) -/
def recv_Probe (value : UInt256) : FeM UInt256 := do
  let v1 ← revert_u256 value
  let v1 : UInt256 := nomatch v1
  return v1


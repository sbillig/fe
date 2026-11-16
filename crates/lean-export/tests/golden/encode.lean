export Lib (revert_u256)

def init : FeM Unit := do
  return ()

def recv_Probe (v0 : UInt256) : FeM UInt256 := do
  let v1 ← revert_u256 v0
  let v1 : UInt256 := nomatch v1
  return v1


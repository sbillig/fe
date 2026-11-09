def init (v0 : UInt256) : FeM Unit := do
  let v1 := v0
  sstore (UInt256.ofNat 0) v1
  let v2 := ()
  return ()

def recv0_arm0 : FeM UInt256 := do
  let v0 ← sload (UInt256.ofNat 0)
  return v0

def recv0_arm1 (v0 : UInt256) : FeM Unit := do
  sstore (UInt256.ofNat 0) v0
  let v1 := ()
  return ()


def Field.stored : Place :=
  .storage (UInt256.ofNat 0)

def init (v0 : UInt256) : FeM Unit := do
  let v1 := v0
  store Field.stored v1
  let v2 := ()
  return ()

def recv0_arm0 : FeM UInt256 := do
  let v0 ← load Field.stored
  return v0

def recv0_arm1 (v0 : UInt256) : FeM Unit := do
  store Field.stored v0
  let v1 := ()
  return ()


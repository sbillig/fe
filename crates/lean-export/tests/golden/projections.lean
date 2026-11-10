def Field.cell : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Cell : Layout :=
  { storage := [0] }

def init : FeM Unit := do
  return ()

def slot (v0 : Place) : FeM Place := do
  let v1 := (v0.field Layout.Cell 0)
  return v1

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let v1 := Field.cell
  let v2 ← slot v1
  let v3 ← load v2
  let v4 ← checkedAdd v3 v0
  store v2 v4
  let v5 := ()
  let v6 ← load (Field.cell.field Layout.Cell 0)
  return v6

def get (v0 : Place) : FeM UInt256 := do
  let v1 ← load (v0.field Layout.Cell 0)
  return v1

def set (v0 : Place) (v1 : UInt256) : FeM Unit := do
  let v2 := v1
  store (v0.field Layout.Cell 0) v2
  let v3 := ()
  return ()

def cached (v0 : Place) : FeM (UInt256 × (UInt256 → FeM Unit)) := do
  let mut r0 : UInt256 := default
  let v1 := v0
  let v2 ← get v1
  let v3 := v2
  r0 := v3
  let slide : UInt256 → FeM Unit := fun g => do
    let mut r0 := r0
    r0 := g
    let v5 := r0
    let v6 := v5
    let v7 ← set v0 v6
    return ()
  return (r0, slide)

def recv0_arm1 (v0 : UInt256) : FeM UInt256 := do
  let mut v2 : UInt256 := default
  let v1 := Field.cell
  let (v2_g, v2_k) ← cached v1
  v2 := v2_g
  let v3 := v2
  let v4 ← checkedAdd v3 v0
  v2 := v4
  let _ ← v2_k v2
  let v5 := ()
  let v6 ← load (Field.cell.field Layout.Cell 0)
  return v6

def first (v0 : (UInt256 × UInt256)) : FeM (UInt256 × (UInt256 → FeM (UInt256 × UInt256))) := do
  let mut v0 := v0
  let slide : UInt256 → FeM (UInt256 × UInt256) := fun g => do
    let mut v0 := v0
    v0 := { v0 with fst := g }
    return v0
  return (v0.fst, slide)

def recv0_arm2 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : (UInt256 × UInt256) := default
  let mut v5 : UInt256 := default
  let v2 := (v0, v1)
  let v3 := v2
  r0 := v3
  let (v5_g, v5_k) ← first r0
  v5 := v5_g
  let v6 := v5
  let v7 ← checkedAdd v6 v1
  v5 := v7
  let v5_0 ← v5_k v5
  r0 := v5_0
  let v8 := ()
  let v9 := r0.fst
  return v9


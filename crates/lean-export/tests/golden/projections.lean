def Field.cell : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Cell : Layout :=
  { storage := [0] }

def init : FeM Unit := do
  return ()

def Cell_slot (self : Place) : FeM Place := do
  let v1 := (self.field Layout.Cell 0)
  return v1

def recv_Direct («by» : UInt256) : FeM UInt256 := do
  let v1 := Field.cell
  let v2 ← Cell_slot v1
  let v3 ← load v2
  let v4 ← checkedAdd v3 «by»
  store v2 v4
  let v5 := ()
  let v6 ← load (Field.cell.field Layout.Cell 0)
  return v6

def Cell_get (self : Place) : FeM UInt256 := do
  let v1 ← load (self.field Layout.Cell 0)
  return v1

def Cell_set (self : Place) (v_1 : UInt256) : FeM Unit := do
  let v_2 := v_1
  store (self.field Layout.Cell 0) v_2
  let v3 := ()
  return ()

def Cell_cached (self_0 : Place) : FeM (UInt256 × (UInt256 → FeM Unit)) := do
  let mut tmp : UInt256 := default
  let self_1 := self_0
  let v2 ← Cell_get self_1
  let tmp_3 := v2
  tmp := tmp_3
  let slide : UInt256 → FeM Unit := fun g => do
    let mut tmp := tmp
    tmp := g
    let tmp_5 := tmp
    let tmp_6 := tmp_5
    let v7 ← Cell_set self_0 tmp_6
    return ()
  return (tmp, slide)

def recv_Slide («by» : UInt256) : FeM UInt256 := do
  let mut v2 : UInt256 := default
  let v1 := Field.cell
  let (v2_g, v2_k) ← Cell_cached v1
  v2 := v2_g
  let v3 := v2
  let v4 ← checkedAdd v3 «by»
  v2 := v4
  let _ ← v2_k v2
  let v5 := ()
  let v6 ← load (Field.cell.field Layout.Cell 0)
  return v6

def first (pair : (UInt256 × UInt256)) : FeM (UInt256 × (UInt256 → FeM (UInt256 × UInt256))) := do
  let mut pair := pair
  let slide : UInt256 → FeM (UInt256 × UInt256) := fun g => do
    let mut pair := pair
    pair := { pair with fst := g }
    return pair
  return (pair.fst, slide)

def recv_Local (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut pair : (UInt256 × UInt256) := default
  let mut v5 : UInt256 := default
  let v2 := (a, b)
  let pair_3 := v2
  pair := pair_3
  let (v5_g, v5_k) ← first pair
  v5 := v5_g
  let v6 := v5
  let v7 ← checkedAdd v6 b
  v5 := v7
  let v5_0 ← v5_k v5
  pair := v5_0
  let v8 := ()
  let v9 := pair.fst
  return v9


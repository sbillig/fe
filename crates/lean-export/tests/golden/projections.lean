def Field.cell : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Cell : Layout :=
  { storage := [0] }

/-- `cell.v` -/
abbrev Field.cell.v : Place :=
  Field.cell.field Layout.Cell 0

/-- The `init` of contract `Projections`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Cell::slot` -/
def Cell_slot (self : Place) : FeM Place := do
  let v1 := (self.field Layout.Cell 0)
  return v1

/-- `Direct { by } -> u256`, selector `0x9bbc59f9` (`direct(uint256)`) -/
def recv_Direct («by» : UInt256) : FeM UInt256 := do
  let v1 := Field.cell
  let v2 ← Cell_slot v1
  let v3 ← load v2
  let v4 ← checkedAdd v3 «by»
  store v2 v4
  let v6 ← load Field.cell.v
  return v6

/-- `Cell::get` -/
def Cell_get (self : Place) : FeM UInt256 := do
  let v1 ← load (self.field Layout.Cell 0)
  return v1

/-- `Cell::set` -/
def Cell_set (self : Place) (v : UInt256) : FeM Unit := do
  store (self.field Layout.Cell 0) v
  return ()

/-- `Cell::cached` -/
def Cell_cached (self_0 : Place) : FeM (UInt256 × (UInt256 → FeM Unit)) := do
  let mut tmp : UInt256 := default
  let self_1 := self_0
  let tmp_2 ← Cell_get self_1
  tmp := tmp_2
  let slide : UInt256 → FeM Unit := fun g => do
    let mut tmp := tmp
    tmp := g
    let tmp_5 := tmp
    let _ ← Cell_set self_0 tmp_5
    return ()
  return (tmp, slide)

/-- `Slide { by } -> u256`, selector `0x235361d9` (`slide(uint256)`) -/
def recv_Slide («by» : UInt256) : FeM UInt256 := do
  let mut v2 : UInt256 := default
  let v1 := Field.cell
  let (v2_g, v2_k) ← Cell_cached v1
  v2 := v2_g
  let v3 := v2
  let v4 ← checkedAdd v3 «by»
  v2 := v4
  let _ ← v2_k v2
  let v6 ← load Field.cell.v
  return v6

/-- `projections::first` -/
def first (pair : (UInt256 × UInt256)) : FeM (UInt256 × (UInt256 → FeM (UInt256 × UInt256))) := do
  let mut pair := pair
  let slide : UInt256 → FeM (UInt256 × UInt256) := fun g => do
    let mut pair := pair
    pair := { pair with fst := g }
    return pair
  return (pair.fst, slide)

/-- `Local { a, b } -> u256`, selector `0xa60eff5e` (`local(uint256,uint256)`) -/
def recv_Local (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut pair : (UInt256 × UInt256) := default
  let mut v5 : UInt256 := default
  let pair_2 := (a, b)
  pair := pair_2
  let (v5_g, v5_k) ← first pair
  v5 := v5_g
  let v6 := v5
  let v7 ← checkedAdd v6 b
  v5 := v7
  let v5_0 ← v5_k v5
  pair := v5_0
  let v9 := pair.fst
  return v9


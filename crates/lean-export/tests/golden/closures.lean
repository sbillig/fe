export Lib (array_at_ref_u256_2 array_has_u256_2 array_next_u256_2 array_start_u256_2)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0] }

/-- `store.n` -/
abbrev Field.store.n : Place :=
  Field.store.field Layout.Store 0

/-- The `init` of contract `Closures`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- A closure in `closures::Closures::{fn_body}` -/
def closure (self : UInt256) (acc : UInt256) (x : UInt256) : FeM (UInt256 × UInt256) := do
  let mut self := self
  -- xs.fold(0, |acc, x| acc + x * by)
  let v5 ← checkedMul x self
  let v6 ← checkedAdd acc v5
  return (v6, self)

/-- `core::iter::Collection::fold<[u256; 2], u256, |own u256, u256| -> u256>` -/
def fold_array_u256_2_u256_own_u256_u256_u256 (self : (Vector UInt256 2)) (init_1 : UInt256) (f : UInt256) : FeM (UInt256 × UInt256) := do
  let mut acc : UInt256 := default
  let mut f := f
  -- var acc = init
  acc := init_1
  -- for ref x in self {
  let v5 ← array_start_u256_2 self
  let rec loop (acc : UInt256) (f : UInt256) (v3 : UInt256) : FeM (UInt256 × UInt256) := do
    let mut acc := acc
    let mut f := f
    -- for ref x in self {
    let v7 ← array_has_u256_2 self v3
    if v7 then
      let v8 ← array_at_ref_u256_2 self v3
      -- acc = f(acc, x)
      let (acc_13, acc_13_0) ← closure f acc v8
      f := acc_13_0
      acc := acc_13
      -- for ref x in self {
      let v16 ← array_next_u256_2 self v3
      loop acc f v16
    else
      return (acc, f)
  partial_fixpoint
  loop acc f v5

/-- `Scaled { a, b, by } -> u256`, selector `0x994a2044` (`scaled(uint256,uint256,uint256)`) -/
def recv_Scaled (a : UInt256) (b : UInt256) («by» : UInt256) : FeM UInt256 := do
  let mut xs : (Vector UInt256 2) := default
  let mut r1 : UInt256 := default
  -- let xs: [u256; 2] = [a, b]
  xs := (#v[a, b])
  -- xs.fold(0, |acc, x| acc + x * by)
  r1 := «by»
  let (v10, v10_0) ← fold_array_u256_2_u256_own_u256_u256_u256 xs (UInt256.ofNat 0) r1
  r1 := v10_0
  return v10

/-- `closures::bump` -/
def bump («by» : UInt256) : FeM Unit := do
  -- store.n += by
  let v2 ← load Field.store.n
  let v3 ← checkedAdd v2 «by»
  store Field.store.n v3
  return ()

/-- A closure in `closures::Closures::{fn_body}` -/
def closure_1 (self : Unit) (acc : UInt256) (x : UInt256) : FeM (UInt256 × Unit) := do
  let mut self := self
  -- bump(x)
  let _ ← bump x
  -- acc + store.n
  let v4 ← load Field.store.n
  let v5 ← checkedAdd acc v4
  return (v5, self)

/-- `core::iter::Collection::fold<[u256; 2], u256, |own u256, u256| -> u256>` -/
def fold_array_u256_2_u256_own_u256_u256_u256_1 (self : (Vector UInt256 2)) (init_1 : UInt256) (f : Unit) : FeM (UInt256 × Unit) := do
  let mut acc : UInt256 := default
  let mut f := f
  -- var acc = init
  acc := init_1
  -- for ref x in self {
  let v5 ← array_start_u256_2 self
  let rec loop (acc : UInt256) (f : Unit) (v3 : UInt256) : FeM (UInt256 × Unit) := do
    let mut acc := acc
    let mut f := f
    -- for ref x in self {
    let v7 ← array_has_u256_2 self v3
    if v7 then
      let v8 ← array_at_ref_u256_2 self v3
      -- acc = f(acc, x)
      let (acc_13, acc_13_0) ← closure_1 f acc v8
      f := acc_13_0
      acc := acc_13
      -- for ref x in self {
      let v16 ← array_next_u256_2 self v3
      loop acc f v16
    else
      return (acc, f)
  partial_fixpoint
  loop acc f v5

/-- `Tally { a, b } -> u256`, selector `0x97c3f4e7` (`tally(uint256,uint256)`) -/
def recv_Tally (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut xs : (Vector UInt256 2) := default
  let mut r2 : Unit := default
  -- let xs: [u256; 2] = [a, b]
  xs := (#v[a, b])
  -- xs.fold(0, |acc, x| {
  r2 := ()
  let (v9, v9_0) ← fold_array_u256_2_u256_own_u256_u256_u256_1 xs (UInt256.ofNat 0) r2
  r2 := v9_0
  return v9


export Lib (array_at_ref_u256_2 array_has_u256_2 array_next_u256_2 array_start_u256_2)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

structure Store where
  n : UInt256
deriving Inhabited

def load_Store (p : Place) : FeM Store := do
  let f0 ← load (p.offset 0)
  return (Store.mk f0)

def Layout.Store : Layout :=
  { storage := [0] }

/-- The `init` of contract `Closures`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- A closure in `closures::Closures::{fn_body}` -/
def closure (self : UInt256) (acc : UInt256) (x : UInt256) : FeM (UInt256 × UInt256) := do
  let mut self := self
  let v4 := self
  let v5 ← checkedMul x v4
  let v6 ← checkedAdd acc v5
  return (v6, self)

/-- `core::iter::Collection::fold<[u256; 2], (), (), u256, |own u256, u256| -> u256>` -/
def fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256 (self : (Vector UInt256 2)) (init_1 : UInt256) (f : UInt256) : FeM (UInt256 × UInt256) := do
  let mut acc : UInt256 := default
  let mut f := f
  acc := init_1
  let v5 ← array_start_u256_2 self
  let rec b1 (acc : UInt256) (f : UInt256) (v3 : UInt256) : FeM (UInt256 × UInt256) := do
    let mut acc := acc
    let mut f := f
    let v7 ← array_has_u256_2 self v3
    if v7 then
      let v8 ← array_at_ref_u256_2 self v3
      let acc_10 := acc
      let x_12 := v8
      let (acc_13, acc_13_0) ← closure f acc_10 x_12
      f := acc_13_0
      acc := acc_13
      let v16 ← array_next_u256_2 self v3
      b1 acc f v16
    else
      let acc_18 := acc
      return (acc_18, f)
  partial_fixpoint
  b1 acc f v5

/-- `Scaled { a, b, by } -> u256`, selector `0x994a2044` (`scaled(uint256,uint256,uint256)`) -/
def recv_Scaled (a : UInt256) (b : UInt256) («by» : UInt256) : FeM UInt256 := do
  let mut xs : (Vector UInt256 2) := default
  let mut r1 : UInt256 := default
  let xs_3 := #v[a, b]
  xs := xs_3
  let v6 := «by»
  r1 := v6
  let (v10, v10_0) ← fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256 xs (UInt256.ofNat 0) r1
  r1 := v10_0
  return v10

/-- `closures::bump<Store>` -/
def bump_Store («by» : UInt256) : FeM Unit := do
  let v2 ← load (Field.store.field Layout.Store 0)
  let v3 ← checkedAdd v2 «by»
  store (Field.store.field Layout.Store 0) v3
  return ()

/-- A closure in `closures::Closures::{fn_body}` -/
def closure_1 (self : Unit) (acc : UInt256) (x : UInt256) : FeM (UInt256 × Unit) := do
  let mut self := self
  let _ ← bump_Store x
  let v4 ← load (Field.store.field Layout.Store 0)
  let v5 ← checkedAdd acc v4
  return (v5, self)

/-- `core::iter::Collection::fold<[u256; 2], (), (), u256, |own u256, u256| -> u256>` -/
def fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256_1 (self : (Vector UInt256 2)) (init_1 : UInt256) (f : Unit) : FeM (UInt256 × Unit) := do
  let mut acc : UInt256 := default
  let mut f := f
  acc := init_1
  let v5 ← array_start_u256_2 self
  let rec b1 (acc : UInt256) (f : Unit) (v3 : UInt256) : FeM (UInt256 × Unit) := do
    let mut acc := acc
    let mut f := f
    let v7 ← array_has_u256_2 self v3
    if v7 then
      let v8 ← array_at_ref_u256_2 self v3
      let acc_10 := acc
      let x_12 := v8
      let store_1 ← load_Store Field.store
      let (acc_14, acc_14_0) ← closure_1 f acc_10 x_12
      f := acc_14_0
      acc := acc_14
      let v17 ← array_next_u256_2 self v3
      b1 acc f v17
    else
      let acc_19 := acc
      return (acc_19, f)
  partial_fixpoint
  b1 acc f v5

/-- `Tally { a, b } -> u256`, selector `0x97c3f4e7` (`tally(uint256,uint256)`) -/
def recv_Tally (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut xs : (Vector UInt256 2) := default
  let mut r2 : Unit := default
  let xs_2 := #v[a, b]
  xs := xs_2
  let v5 := ()
  r2 := v5
  let (v9, v9_0) ← fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256_1 xs (UInt256.ofNat 0) r2
  r2 := v9_0
  return v9


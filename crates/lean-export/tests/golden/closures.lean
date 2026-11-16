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

def init : FeM Unit := do
  return ()

def closure (self : UInt256) (acc : UInt256) (x_2 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut self := self
  let x_3 := x_2
  let v4 := self
  let v5 ← checkedMul x_3 v4
  let v6 ← checkedAdd acc v5
  return (v6, self)

def fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256 (self : (Vector UInt256 2)) (init_1 : UInt256) (f : UInt256) : FeM (UInt256 × UInt256) := do
  let mut acc : UInt256 := default
  let mut f := f
  let init_4 := init_1
  acc := init_4
  let v5 ← array_start_u256_2 self
  let v6 := v5
  let rec b1 (acc : UInt256) (f : UInt256) (v3 : UInt256) : FeM (UInt256 × UInt256) := do
    let mut acc := acc
    let mut f := f
    let v7 ← array_has_u256_2 self v3
    if v7 then
      let v8 ← array_at_ref_u256_2 self v3
      let acc_10 := acc
      let acc_11 := acc_10
      let x_12 := v8
      let (v13, v13_0) ← closure f acc_11 x_12
      f := v13_0
      let acc_14 := v13
      acc := acc_14
      let v15 := ()
      let v16 ← array_next_u256_2 self v3
      let v17 := v16
      b1 acc f v17
    else
      let acc_18 := acc
      return (acc_18, f)
  partial_fixpoint
  b1 acc f v6

def recv_Scaled (a : UInt256) (b : UInt256) («by» : UInt256) : FeM UInt256 := do
  let mut xs : (Vector UInt256 2) := default
  let mut r1 : UInt256 := default
  let v3 := #v[a, b]
  let xs_4 := v3
  xs := xs_4
  let v5 := (UInt256.ofNat 0)
  let v6 := «by»
  let v7 := v6
  r1 := v7
  let (v10, v10_0) ← fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256 xs v5 r1
  r1 := v10_0
  return v10

def bump_Store (by_0 : UInt256) : FeM Unit := do
  let by_1 := by_0
  let v2 ← load (Field.store.field Layout.Store 0)
  let v3 ← checkedAdd v2 by_1
  store (Field.store.field Layout.Store 0) v3
  let v4 := ()
  return ()

def closure_1 (self : Unit) (acc : UInt256) (x : UInt256) : FeM (UInt256 × Unit) := do
  let mut self := self
  let v3 ← bump_Store x
  let v4 ← load (Field.store.field Layout.Store 0)
  let v5 ← checkedAdd acc v4
  return (v5, self)

def fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256_1 (self : (Vector UInt256 2)) (init_1 : UInt256) (f : Unit) : FeM (UInt256 × Unit) := do
  let mut acc : UInt256 := default
  let mut f := f
  let init_4 := init_1
  acc := init_4
  let v5 ← array_start_u256_2 self
  let v6 := v5
  let rec b1 (acc : UInt256) (f : Unit) (v3 : UInt256) : FeM (UInt256 × Unit) := do
    let mut acc := acc
    let mut f := f
    let v7 ← array_has_u256_2 self v3
    if v7 then
      let v8 ← array_at_ref_u256_2 self v3
      let acc_10 := acc
      let acc_11 := acc_10
      let x_12 := v8
      let store_1 ← load_Store Field.store
      let (v14, v14_0) ← closure_1 f acc_11 x_12
      f := v14_0
      let acc_15 := v14
      acc := acc_15
      let v16 := ()
      let v17 ← array_next_u256_2 self v3
      let v18 := v17
      b1 acc f v18
    else
      let acc_19 := acc
      return (acc_19, f)
  partial_fixpoint
  b1 acc f v6

def recv_Tally (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut xs : (Vector UInt256 2) := default
  let mut r2 : Unit := default
  let v2 := #v[a, b]
  let xs_3 := v2
  xs := xs_3
  let v4 := (UInt256.ofNat 0)
  let v5 := ()
  let v6 := v5
  r2 := v6
  let (v9, v9_0) ← fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256_1 xs v4 r2
  r2 := v9_0
  return v9


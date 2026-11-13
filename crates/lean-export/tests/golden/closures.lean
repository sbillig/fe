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

def closure (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut v0 := v0
  let v3 := v2
  let v4 := v0
  let v5 ← checkedMul v3 v4
  let v6 ← checkedAdd v1 v5
  return (v6, v0)

def fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256 (v0 : (Vector UInt256 2)) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut r0 : UInt256 := default
  let mut v2 := v2
  let v4 := v1
  r0 := v4
  let v5 ← array_start_u256_2 v0
  let v6 := v5
  let rec b1 (r0 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM (UInt256 × UInt256) := do
    let mut r0 := r0
    let mut v2 := v2
    let v7 ← array_has_u256_2 v0 v3
    if v7 then
      let v8 ← array_at_ref_u256_2 v0 v3
      let v10 := r0
      let v11 := v10
      let v12 := v8
      let (v13, v13_0) ← closure v2 v11 v12
      v2 := v13_0
      let v14 := v13
      r0 := v14
      let v15 := ()
      let v16 ← array_next_u256_2 v0 v3
      let v17 := v16
      b1 r0 v2 v17
    else
      let v18 := r0
      return (v18, v2)
  partial_fixpoint
  b1 r0 v2 v6

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let mut r0 : (Vector UInt256 2) := default
  let mut r1 : UInt256 := default
  let v3 := #v[v0, v1]
  let v4 := v3
  r0 := v4
  let v5 := (UInt256.ofNat 0)
  let v6 := v2
  let v7 := v6
  r1 := v7
  let (v10, v10_0) ← fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256 r0 v5 r1
  r1 := v10_0
  return v10

def bump_Store (v0 : UInt256) : FeM Unit := do
  let v1 := v0
  let v2 ← load (Field.store.field Layout.Store 0)
  let v3 ← checkedAdd v2 v1
  store (Field.store.field Layout.Store 0) v3
  let v4 := ()
  return ()

def closure_1 (v0 : Unit) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × Unit) := do
  let mut v0 := v0
  let v3 ← bump_Store v2
  let v4 ← load (Field.store.field Layout.Store 0)
  let v5 ← checkedAdd v1 v4
  return (v5, v0)

def fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256_1 (v0 : (Vector UInt256 2)) (v1 : UInt256) (v2 : Unit) : FeM (UInt256 × Unit) := do
  let mut r1 : UInt256 := default
  let mut v2 := v2
  let v4 := v1
  r1 := v4
  let v5 ← array_start_u256_2 v0
  let v6 := v5
  let rec b1 (r1 : UInt256) (v2 : Unit) (v3 : UInt256) : FeM (UInt256 × Unit) := do
    let mut r1 := r1
    let mut v2 := v2
    let v7 ← array_has_u256_2 v0 v3
    if v7 then
      let v8 ← array_at_ref_u256_2 v0 v3
      let v10 := r1
      let v11 := v10
      let v12 := v8
      let v13 ← load_Store Field.store
      let (v14, v14_0) ← closure_1 v2 v11 v12
      v2 := v14_0
      let v15 := v14
      r1 := v15
      let v16 := ()
      let v17 ← array_next_u256_2 v0 v3
      let v18 := v17
      b1 r1 v2 v18
    else
      let v19 := r1
      return (v19, v2)
  partial_fixpoint
  b1 r1 v2 v6

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r1 : (Vector UInt256 2) := default
  let mut r2 : Unit := default
  let v2 := #v[v0, v1]
  let v3 := v2
  r1 := v3
  let v4 := (UInt256.ofNat 0)
  let v5 := ()
  let v6 := v5
  r2 := v6
  let (v9, v9_0) ← fold_array_u256_2_tuple_tuple_u256_own_u256_u256_u256_1 r1 v4 r2
  r2 := v9_0
  return v9


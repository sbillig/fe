export Lib (StorageMap_locate_u256_u256)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

def init : FeM Unit := do
  return ()

def add_Store (v0 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 10)
  let v4 := decide (v3.toNat < v2.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    let v13 := v1
    store (Field.store.field Layout.Store 0) v13
    let v14 := ()
    let v15 := (UInt256.ofNat 1)
    let v16 := v0
    let l0 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot v16
    let v17 ← load (Place.mk .storage l0.slot)
    let v18 ← checkedAdd v17 v15
    let l1 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot v16
    store (Place.mk .storage l1.slot) v18
    let v19 := ()
    return v13
  if v4 then
    let v5 ← load (Field.store.field Layout.Store 0)
    let v6 := (UInt256.ofNat 10)
    let v7 ← checkedAdd v5 v6
    let v8 := v7
    b3 v8
  else
    let v9 ← load (Field.store.field Layout.Store 0)
    let v10 := v0
    let v11 ← checkedAdd v9 v10
    let v12 := v11
    b3 v12

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← add_Store v0
  return v1


export Lib (StorageMap_locate_u256_u256)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

def init : FeM Unit := do
  return ()

def add_Store (v0 : UInt256) : FeM UInt256 := do
  let v1 ← load (Field.store.field Layout.Store 0)
  let v2 := v0
  let v3 ← checkedAdd v1 v2
  let v4 := v3
  store (Field.store.field Layout.Store 0) v4
  let v5 := ()
  let v6 := (UInt256.ofNat 1)
  let v7 := v0
  let l0 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot v7
  let v8 ← load (Place.mk .storage l0.slot)
  let v9 ← checkedAdd v8 v6
  let l1 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot v7
  store (Place.mk .storage l1.slot) v9
  let v10 := ()
  return v4

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← add_Store v0
  return v1


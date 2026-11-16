export Lib (StorageMap_locate_u256_u256)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

def init : FeM Unit := do
  return ()

def add_Store (by_0 : UInt256) : FeM UInt256 := do
  let v1 ← load (Field.store.field Layout.Store 0)
  let by_2 := by_0
  let v3 ← checkedAdd v1 by_2
  let next := v3
  store (Field.store.field Layout.Store 0) next
  let v5 := ()
  let v6 := (UInt256.ofNat 1)
  let by_7 := by_0
  let l0 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot by_7
  let v8 ← load (Place.mk .storage l0.slot)
  let v9 ← checkedAdd v8 v6
  let l1 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot by_7
  store (Place.mk .storage l1.slot) v9
  let v10 := ()
  return next

def recv_Bump («by» : UInt256) : FeM UInt256 := do
  let v1 ← add_Store «by»
  return v1


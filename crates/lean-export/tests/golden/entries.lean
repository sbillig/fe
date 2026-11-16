export Lib (StorageMap_locate_bool_u256 StorageMap_locate_tuple_u256_u256_bool)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

def init : FeM Unit := do
  return ()

def recv_Vote (voter : UInt256) (proposal : UInt256) : FeM Bool := do
  let v2 := (voter, proposal)
  let l0 ← StorageMap_locate_tuple_u256_u256_bool (Field.store.field Layout.Store 0) (Field.store.field Layout.Store 0).slot v2
  let v3 ← loadBool (Place.mk .storage l0.slot)
  let before := v3
  let v5 := true
  let v6 := (voter, proposal)
  let l1 ← StorageMap_locate_tuple_u256_u256_bool (Field.store.field Layout.Store 0) (Field.store.field Layout.Store 0).slot v6
  storeBool (Place.mk .storage l1.slot) v5
  let v7 := ()
  return before

def recv_Flag (on : Bool) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  let l0 ← StorageMap_locate_bool_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot on
  let v2 ← load (Place.mk .storage l0.slot)
  let v3 ← checkedAdd v2 v1
  let l1 ← StorageMap_locate_bool_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot on
  store (Place.mk .storage l1.slot) v3
  let v4 := ()
  let l2 ← StorageMap_locate_bool_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot on
  let v5 ← load (Place.mk .storage l2.slot)
  return v5


export Lib (StorageMap_locate_bool_u256 StorageMap_locate_tuple_u256_u256_bool)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

/-- `store.voted` -/
abbrev Field.store.voted : Place :=
  Field.store.field Layout.Store 0

/-- `store.flags` -/
abbrev Field.store.flags : Place :=
  Field.store.field Layout.Store 1

/-- The `init` of contract `Entries`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Vote { voter, proposal } -> bool`, selector `0xb384abef` (`vote(uint256,uint256)`) -/
def recv_Vote (voter : UInt256) (proposal : UInt256) : FeM Bool := do
  let v2 := (voter, proposal)
  let l0 ← StorageMap_locate_tuple_u256_u256_bool Field.store.voted Field.store.voted.slot v2
  let before ← loadBool (Place.mk .storage l0.slot)
  let v6 := (voter, proposal)
  let l1 ← StorageMap_locate_tuple_u256_u256_bool Field.store.voted Field.store.voted.slot v6
  storeBool (Place.mk .storage l1.slot) true
  return before

/-- `Flag { on } -> u256`, selector `0xa92a4c3b` (`flag(bool)`) -/
def recv_Flag (on : Bool) : FeM UInt256 := do
  let l0 ← StorageMap_locate_bool_u256 Field.store.flags Field.store.flags.slot on
  let v2 ← load (Place.mk .storage l0.slot)
  let v3 ← checkedAdd v2 (UInt256.ofNat 1)
  let l1 ← StorageMap_locate_bool_u256 Field.store.flags Field.store.flags.slot on
  store (Place.mk .storage l1.slot) v3
  let l2 ← StorageMap_locate_bool_u256 Field.store.flags Field.store.flags.slot on
  let v5 ← load (Place.mk .storage l2.slot)
  return v5


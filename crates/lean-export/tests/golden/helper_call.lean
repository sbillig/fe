export Lib (StorageMap_locate_u256_u256)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

/-- `store.total` -/
abbrev Field.store.total : Place :=
  Field.store.field Layout.Store 0

/-- `store.hits` -/
abbrev Field.store.hits : Place :=
  Field.store.field Layout.Store 1

/-- The `init` of contract `Counter`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `helper_call::add<Store>` -/
def add_Store («by» : UInt256) : FeM UInt256 := do
  let v1 ← load Field.store.total
  let next ← checkedAdd v1 «by»
  store Field.store.total next
  let hits_by ← StorageMap_locate_u256_u256 Field.store.hits Field.store.hits.slot «by»
  let v8 ← load (Place.mk .storage hits_by.slot)
  let v9 ← checkedAdd v8 (UInt256.ofNat 1)
  let hits_by_1 ← StorageMap_locate_u256_u256 Field.store.hits Field.store.hits.slot «by»
  store (Place.mk .storage hits_by_1.slot) v9
  return next

/-- `Bump { by } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump («by» : UInt256) : FeM UInt256 := do
  let v1 ← add_Store «by»
  return v1


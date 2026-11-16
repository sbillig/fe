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

/-- `join::add<Store>` -/
def add_Store («by» : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun next => do
    -- store.total = next
    store Field.store.total next
    -- store.hits[by] += 1
    let hits_by ← StorageMap_locate_u256_u256 Field.store.hits Field.store.hits.slot «by»
    let v17 ← load (Place.mk .storage hits_by.slot)
    let v18 ← checkedAdd v17 (UInt256.ofNat 1)
    let hits_by_1 ← StorageMap_locate_u256_u256 Field.store.hits Field.store.hits.slot «by»
    store (Place.mk .storage hits_by_1.slot) v18
    return next
  -- let next = if by > 10 { store.total + 10 } else { store.total + by }
  if decide ((UInt256.ofNat 10).toNat < «by».toNat) then
    -- store.total + 10
    let v5 ← load Field.store.total
    let v7 ← checkedAdd v5 (UInt256.ofNat 10)
    join v7
  else
    -- store.total + by
    let v9 ← load Field.store.total
    let v11 ← checkedAdd v9 «by»
    join v11

/-- `Bump { by } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump («by» : UInt256) : FeM UInt256 := do
  -- add(by)
  let v1 ← add_Store «by»
  return v1


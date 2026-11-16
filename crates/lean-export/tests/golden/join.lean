export Lib (StorageMap_locate_u256_u256)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

/-- The `init` of contract `Counter`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `join::add<Store>` -/
def add_Store («by» : UInt256) : FeM UInt256 := do
  let v4 := decide ((UInt256.ofNat 10).toNat < «by».toNat)
  let b3 : UInt256 → FeM UInt256 := fun next => do
    store (Field.store.field Layout.Store 0) next
    let l0 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot «by»
    let v17 ← load (Place.mk .storage l0.slot)
    let v18 ← checkedAdd v17 (UInt256.ofNat 1)
    let l1 ← StorageMap_locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot «by»
    store (Place.mk .storage l1.slot) v18
    return next
  if v4 then
    let v5 ← load (Field.store.field Layout.Store 0)
    let v7 ← checkedAdd v5 (UInt256.ofNat 10)
    b3 v7
  else
    let v9 ← load (Field.store.field Layout.Store 0)
    let v11 ← checkedAdd v9 «by»
    b3 v11

/-- `Bump { by } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump («by» : UInt256) : FeM UInt256 := do
  let v1 ← add_Store «by»
  return v1


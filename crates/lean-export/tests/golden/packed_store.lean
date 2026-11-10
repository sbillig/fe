def Field.pack : Place :=
  .storage (UInt256.ofNat 0)

def Layout.StoragePackedArray_8 : Layout :=
  { storage := [0] }

def init : FeM Unit := do
  return ()

def LANES_PER_SLOT_8 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 256)
  let v1 := (UInt256.ofNat 8)
  let v2 ← checkedDiv v0 v1
  return v2

def LANE_MASK_8 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := (UInt256.ofNat 8)
  let v2 := v0 <<< v1
  let v3 := (UInt256.ofNat 1)
  let v4 ← checkedSub v2 v3
  return v4

def get_8 (v0 : Place) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 ← LANES_PER_SLOT_8
  let v4 ← checkedRem v2 v3
  let v5 := (UInt256.ofNat 8)
  let v6 := v4 * v5
  let v7 := v6
  let v8 := v1
  let v9 ← LANES_PER_SLOT_8
  let v10 ← checkedDiv v8 v9
  let v11 ← load ((v0.field Layout.StoragePackedArray_8 0).entry v10)
  let v12 := v11 >>> v7
  let v13 ← LANE_MASK_8
  let v14 := v12 &&& v13
  return v14

def recv0_arm0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← get_8 v2 v0
  return v3

def recv0_arm1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← get_8 v2 v0
  return v3

def U256_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)
  return v0

def set_8 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := v1
  let v4 ← LANES_PER_SLOT_8
  let v5 ← checkedRem v3 v4
  let v6 := (UInt256.ofNat 8)
  let v7 := v5 * v6
  let v8 := v7
  let v9 ← LANE_MASK_8
  let v10 := v9 <<< v8
  let v11 := v10
  let v12 := v1
  let v13 ← LANES_PER_SLOT_8
  let v14 ← checkedDiv v12 v13
  let v15 := ((v0.field Layout.StoragePackedArray_8 0).entry v14)
  let v16 := v15
  let v17 ← load v16
  let v18 ← U256_MASK
  let v19 := v18 ^^^ v11
  let v20 := v17 &&& v19
  let v21 := v2
  let v22 ← LANE_MASK_8
  let v23 := v21 &&& v22
  let v24 := v23 <<< v8
  let v25 := v20 ||| v24
  store v16 v25
  let v26 := ()
  return ()

def recv0_arm2 (v0 : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 0)
  let v2 := Field.pack
  let v3 ← set_8 v2 v1 v0
  return ()

def recv0_arm3 (v0 : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 1)
  let v2 := Field.pack
  let v3 ← set_8 v2 v1 v0
  return ()

def recv0_arm4 (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := Field.pack
  let v4 ← set_8 v3 v2 v0
  let v5 := (UInt256.ofNat 1)
  let v6 := Field.pack
  let v7 ← set_8 v6 v5 v1
  return ()


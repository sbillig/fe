def init (v0 : UInt256) : FeM Unit := do
  let v2 ← caller
  let v3 := v2
  let v4 := v0
  sstore (mapSlot v3 (UInt256.ofNat 0)) v4
  let v5 := ()
  let v6 := v0
  sstore (UInt256.ofNat 2) v6
  let v7 := ()
  let v8 := UInt256.ofNat 0
  let v9 := v0
  let v12 ← emitTransfer v8 v3 v9
  return ()

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← caller
  let v4 := v3
  let v5 ← sload (mapSlot v4 (UInt256.ofNat 0))
  let v6 := decide (v1.toNat ≤ v5.toNat)
  if v6 then
    let v7 := ()
    let v8 ← sload (mapSlot v4 (UInt256.ofNat 0))
    let v9 ← checkedSub v8 v1
    sstore (mapSlot v4 (UInt256.ofNat 0)) v9
    let v10 := ()
    let v11 ← sload (mapSlot v0 (UInt256.ofNat 0))
    let v12 ← checkedAdd v11 v1
    sstore (mapSlot v0 (UInt256.ofNat 0)) v12
    let v13 := ()
    let v16 ← emitTransfer v4 v0 v1
    let v17 := true
    return v17
  else
    M.revert (assertData "ERC20: insufficient balance")

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← caller
  let v4 := v3
  sstore (mapSlot v0 (mapSlot v4 (UInt256.ofNat 1))) v1
  let v5 := ()
  let v8 ← emitApproval v4 v0 v1
  let v9 := true
  return v9

def recv0_arm2 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Bool := do
  let v4 ← caller
  let v5 := v4
  let v6 ← sload (mapSlot v5 (mapSlot v0 (UInt256.ofNat 1)))
  let v7 := v6
  let v8 := decide (v2.toNat ≤ v7.toNat)
  if v8 then
    let v9 := ()
    let v10 ← sload (mapSlot v0 (UInt256.ofNat 0))
    let v11 := decide (v2.toNat ≤ v10.toNat)
    if v11 then
      let v12 := ()
      let v13 ← checkedSub v7 v2
      sstore (mapSlot v5 (mapSlot v0 (UInt256.ofNat 1))) v13
      let v14 := ()
      let v15 ← sload (mapSlot v0 (UInt256.ofNat 0))
      let v16 ← checkedSub v15 v2
      sstore (mapSlot v0 (UInt256.ofNat 0)) v16
      let v17 := ()
      let v18 ← sload (mapSlot v1 (UInt256.ofNat 0))
      let v19 ← checkedAdd v18 v2
      sstore (mapSlot v1 (UInt256.ofNat 0)) v19
      let v20 := ()
      let v23 ← emitTransfer v0 v1 v2
      let v24 := true
      return v24
    else
      M.revert (assertData "ERC20: insufficient balance")
  else
    M.revert (assertData "ERC20: insufficient allowance")

def recv0_arm3 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← sload (mapSlot v0 (UInt256.ofNat 0))
  return v1

def recv0_arm4 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← sload (mapSlot v1 (mapSlot v0 (UInt256.ofNat 1)))
  return v2

def recv0_arm5 : FeM UInt256 := do
  let v0 ← sload (UInt256.ofNat 2)
  return v0


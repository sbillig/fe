def init : FeM Unit := do
  return ()

mutual
def left (v0 : UInt256) : FeM UInt256 := do
  let v3 := v0
  let v4 := (UInt256.ofNat 0)
  let v5 := decide (v3 = v4)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7
  else
    let v8 := v0
    let v9 := (UInt256.ofNat 2)
    let v10 ← checkedRem v8 v9
    let v11 := (UInt256.ofNat 1)
    let v12 := decide (v10 = v11)
    let b6 : UInt256 → FeM UInt256 := fun v2 => do
      let v23 := v2
      b3 v23
    if v12 then
      let v13 := v0
      let v14 := (UInt256.ofNat 1)
      let v15 ← checkedSub v13 v14
      let v16 ← step v15
      let v17 := v16
      b6 v17
    else
      let v18 := v0
      let v19 := (UInt256.ofNat 1)
      let v20 ← checkedSub v18 v19
      let v21 ← right v20
      let v22 := v21
      b6 v22
partial_fixpoint

def step (v0 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (v2 = v3)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 := (UInt256.ofNat 1)
    let v6 := v5
    b3 v6
  else
    let v7 := v0
    let v8 := (UInt256.ofNat 1)
    let v9 ← checkedSub v7 v8
    let v10 ← left v9
    let v11 := v10
    b3 v11
partial_fixpoint

def right (v0 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (v2 = v3)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 := (UInt256.ofNat 2)
    let v6 := v5
    b3 v6
  else
    let v7 := v0
    let v8 := (UInt256.ofNat 1)
    let v9 ← checkedSub v7 v8
    let v10 ← step v9
    let v11 := v10
    b3 v11
partial_fixpoint

end

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← left v0
  return v1


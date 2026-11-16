def init : FeM Unit := do
  return ()

mutual
def left (n_0 : UInt256) : FeM UInt256 := do
  let n_3 := n_0
  let v4 := (UInt256.ofNat 0)
  let v5 := decide (n_3 = v4)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7
  else
    let n_8 := n_0
    let v9 := (UInt256.ofNat 2)
    let v10 ← checkedRem n_8 v9
    let v11 := (UInt256.ofNat 1)
    let v12 := decide (v10 = v11)
    let b6 : UInt256 → FeM UInt256 := fun v2 => do
      let v23 := v2
      b3 v23
    if v12 then
      let n_13 := n_0
      let v14 := (UInt256.ofNat 1)
      let v15 ← checkedSub n_13 v14
      let v16 ← step v15
      let v17 := v16
      b6 v17
    else
      let n_18 := n_0
      let v19 := (UInt256.ofNat 1)
      let v20 ← checkedSub n_18 v19
      let v21 ← right v20
      let v22 := v21
      b6 v22
partial_fixpoint

def step (n_0 : UInt256) : FeM UInt256 := do
  let n_2 := n_0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (n_2 = v3)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 := (UInt256.ofNat 1)
    let v6 := v5
    b3 v6
  else
    let n_7 := n_0
    let v8 := (UInt256.ofNat 1)
    let v9 ← checkedSub n_7 v8
    let v10 ← left v9
    let v11 := v10
    b3 v11
partial_fixpoint

def right (n_0 : UInt256) : FeM UInt256 := do
  let n_2 := n_0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (n_2 = v3)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 := (UInt256.ofNat 2)
    let v6 := v5
    b3 v6
  else
    let n_7 := n_0
    let v8 := (UInt256.ofNat 1)
    let v9 ← checkedSub n_7 v8
    let v10 ← step v9
    let v11 := v10
    b3 v11
partial_fixpoint

end

def recv_Walk (n : UInt256) : FeM UInt256 := do
  let v1 ← left n
  return v1


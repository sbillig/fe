def init : FeM Unit := do
  return ()

mutual
def left (n : UInt256) : FeM UInt256 := do
  let v5 := decide (n = (UInt256.ofNat 0))
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    b3 (UInt256.ofNat 0)
  else
    let v10 ← checkedRem n (UInt256.ofNat 2)
    let v12 := decide (v10 = (UInt256.ofNat 1))
    let b6 : UInt256 → FeM UInt256 := fun v2 => do
      b3 v2
    if v12 then
      let v15 ← checkedSub n (UInt256.ofNat 1)
      let v16 ← step v15
      b6 v16
    else
      let v20 ← checkedSub n (UInt256.ofNat 1)
      let v21 ← right v20
      b6 v21
partial_fixpoint

def step (n : UInt256) : FeM UInt256 := do
  let v4 := decide (n = (UInt256.ofNat 0))
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    b3 (UInt256.ofNat 1)
  else
    let v9 ← checkedSub n (UInt256.ofNat 1)
    let v10 ← left v9
    b3 v10
partial_fixpoint

def right (n : UInt256) : FeM UInt256 := do
  let v4 := decide (n = (UInt256.ofNat 0))
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    b3 (UInt256.ofNat 2)
  else
    let v9 ← checkedSub n (UInt256.ofNat 1)
    let v10 ← step v9
    b3 v10
partial_fixpoint

end

def recv_Walk (n : UInt256) : FeM UInt256 := do
  let v1 ← left n
  return v1


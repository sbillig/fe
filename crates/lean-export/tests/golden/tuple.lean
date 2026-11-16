def init : FeM Unit := do
  return ()

def swap (pair : (UInt256 × UInt256 × Bool)) : FeM (UInt256 × UInt256 × Bool) := do
  let v1 := pair.snd.fst
  let v2 := pair.fst
  let v3 := pair.snd.snd
  let v4 := (v1, v2, v3)
  return v4

def recv_Swap (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut pair : (UInt256 × UInt256 × Bool) := default
  let mut r1 : (UInt256 × UInt256 × Bool) := default
  let v4 := (a, b, true)
  r1 := v4
  let pair_6 ← swap r1
  pair := pair_6
  let v8 := pair.snd.fst
  let v10 ← checkedAdd v8 (UInt256.ofNat 1)
  pair := { pair with snd := { pair.snd with fst := v10 } }
  let v12 := pair.snd.snd
  let b3 : (UInt256 × UInt256 × Bool) → (UInt256 × UInt256 × Bool) → UInt256 → FeM UInt256 := fun pair r1 v2 => do
    let mut pair := pair
    let mut r1 := r1
    return v2
  if v12 then
    let v13 := pair.fst
    let v14 := pair.snd.fst
    let v15 ← checkedSub v13 v14
    b3 pair r1 v15
  else
    b3 pair r1 (UInt256.ofNat 0)


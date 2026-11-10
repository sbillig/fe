def init : FeM Unit := do
  return ()

def swap (v0 : (UInt256 × UInt256 × Bool)) : FeM (UInt256 × UInt256 × Bool) := do
  let v1 := v0.snd.fst
  let v2 := v0.fst
  let v3 := v0.snd.snd
  return (v1, v2, v3)

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : (UInt256 × UInt256 × Bool) := default
  let mut r1 : (UInt256 × UInt256 × Bool) := default
  let v3 := true
  r1 := (v0, v1, v3)
  let v6 ← swap r1
  let v7 := v6
  r0 := v7
  let v8 := r0.snd.fst
  let v9 := (UInt256.ofNat 1)
  let v10 ← checkedAdd v8 v9
  r0 := { r0 with snd := { r0.snd with fst := v10 } }
  let v11 := ()
  let v12 := r0.snd.snd
  let b3 : (UInt256 × UInt256 × Bool) → (UInt256 × UInt256 × Bool) → UInt256 → FeM UInt256 := fun r0 r1 v2 => do
    let mut r0 := r0
    let mut r1 := r1
    return v2
  if v12 then
    let v13 := r0.fst
    let v14 := r0.snd.fst
    let v15 ← checkedSub v13 v14
    let v16 := v15
    b3 r0 r1 v16
  else
    let v17 := (UInt256.ofNat 0)
    let v18 := v17
    b3 r0 r1 v18


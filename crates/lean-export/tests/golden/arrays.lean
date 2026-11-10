def init : FeM Unit := do
  return ()

def bump (v0 : (Vector UInt256 4)) (v1 : UInt256) (v2 : UInt256) : FeM (Unit × (Vector UInt256 4)) := do
  let mut v0 := v0
  let v3 := v2
  let v4 := v1
  checkIndex v4 4
  let v5 := v0[v4.toNat]!
  let v6 ← checkedAdd v5 v3
  checkIndex v4 4
  v0 := (v0.set! v4.toNat v6)
  let v7 := ()
  return ((), v0)

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : (Vector UInt256 4) := default
  let v2 := Vector.replicate 4 v1
  let v3 := v2
  r0 := v3
  let v4 := (UInt256.ofNat 7)
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 4
  r0 := (r0.set! v5.toNat v4)
  let v6 := ()
  let v8 := (UInt256.ofNat 1)
  let (v9, v9_0) ← bump r0 v0 v8
  r0 := v9_0
  checkIndex v0 4
  let v10 := r0[v0.toNat]!
  let v11 := (UInt256.ofNat 3)
  checkIndex v11 4
  let v12 := r0[v11.toNat]!
  let v13 ← checkedAdd v10 v12
  return v13

def WEIGHTS : FeM (Vector UInt256 3) := do
  let v0 := (UInt256.ofNat 1)
  let v1 := (UInt256.ofNat 10)
  let v2 := (UInt256.ofNat 100)
  let v3 := #v[v0, v1, v2]
  return v3

def recv0_arm1 (v0 : (Vector UInt256 3)) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let v2 := v1
  r0 := v2
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  r1 := v4
  let rec b1 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let v5 := r1
    let v6 := v5
    let v7 := (UInt256.ofNat 3)
    let v8 := decide (v6.toNat < v7.toNat)
    if v8 then
      let v9 := r1
      checkIndex v9 3
      let v10 := v0[v9.toNat]!
      let v11 ← WEIGHTS
      let v12 := r1
      checkIndex v12 3
      let v13 := v11[v12.toNat]!
      let v14 ← checkedMul v10 v13
      let v15 := r0
      let v16 ← checkedAdd v15 v14
      let v17 := v16
      r0 := v17
      let v18 := ()
      let v19 := (UInt256.ofNat 1)
      let v20 := r1
      let v21 ← checkedAdd v20 v19
      let v22 := v21
      r1 := v22
      let v23 := ()
      b1 r0 r1
    else
      let v24 := r0
      return v24
  partial_fixpoint
  b1 r0 r1


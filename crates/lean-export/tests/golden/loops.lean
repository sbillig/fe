inductive Option_u256 where
  | None
  | Some (f0 : UInt256)
deriving Inhabited

def Option_u256.tag : Option_u256 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_u256.Some_0 : Option_u256 → UInt256
  | .Some f0 => f0
  | _ => default

def Option_u256.set_Some_0 (x : Option_u256) (v : UInt256) : Option_u256 :=
  match x with
  | .Some f0 => .Some v
  | x => x

def init : FeM Unit := do
  return ()

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
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
    let v7 := decide (v6.toNat < v0.toNat)
    if v7 then
      let v8 := (UInt256.ofNat 1)
      let v9 := r1
      let v10 ← checkedAdd v9 v8
      let v11 := v10
      r1 := v11
      let v12 := ()
      let v13 := r1
      let v14 := v13
      let v15 := r0
      let v16 ← checkedAdd v15 v14
      let v17 := v16
      r0 := v17
      let v18 := ()
      b1 r0 r1
    else
      let v19 := r0
      return v19
  partial_fixpoint
  b1 r0 r1

def step (v0 : UInt256) : FeM Option_u256 := do
  let v3 := v0
  let v4 := (UInt256.ofNat 1)
  let v5 := decide (v3 = v4)
  let b3 : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if v5 then
    let v6 := Option_u256.None
    let v7 := v6
    b3 v7
  else
    let v8 := v0
    let v9 := (UInt256.ofNat 2)
    let v10 ← checkedRem v8 v9
    let v11 := (UInt256.ofNat 0)
    let v12 := decide (v10 = v11)
    let b6 : Option_u256 → FeM Option_u256 := fun v2 => do
      let v25 := v2
      b3 v25
    if v12 then
      let v13 := v0
      let v14 := (UInt256.ofNat 2)
      let v15 ← checkedDiv v13 v14
      let v16 := Option_u256.Some v15
      let v17 := v16
      b6 v17
    else
      let v18 := (UInt256.ofNat 3)
      let v19 := v0
      let v20 ← checkedMul v18 v19
      let v21 := (UInt256.ofNat 1)
      let v22 ← checkedAdd v20 v21
      let v23 := Option_u256.Some v22
      let v24 := v23
      b6 v24

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v2 := v0
  r0 := v2
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  r1 := v4
  let rec b1 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let v5 := r1
    let v6 := v5
    let v7 := decide (v6 = v1)
    if v7 then
      return v1
    else
      let v8 := ()
      let v9 := v8
      let v10 := r0
      let v11 := v10
      let v12 ← step v11
      match (← pure v12) with
      | .Some .. =>
        let v13 := v12.Some_0
        let v14 := v13
        let v15 := v14
        r0 := v15
        let v16 := ()
        let v17 := (UInt256.ofNat 1)
        let v18 := r1
        let v19 ← checkedAdd v18 v17
        let v20 := v19
        r1 := v20
        let v21 := ()
        let v22 := v21
        b1 r0 r1
      | .None =>
        let v23 := r1
        return v23
  partial_fixpoint
  b1 r0 r1


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

inductive Shape where
  | Empty
  | Square (f0 : UInt256)
  | Rect (f0 : UInt256) (f1 : UInt256)
deriving Inhabited

def Shape.tag : Shape → UInt256
  | .Empty => UInt256.ofNat 0
  | .Square .. => UInt256.ofNat 1
  | .Rect .. => UInt256.ofNat 2

def Shape.Square_0 : Shape → UInt256
  | .Square f0 => f0
  | _ => default

def Shape.set_Square_0 (x : Shape) (v : UInt256) : Shape :=
  match x with
  | .Square f0 => .Square v
  | x => x

def Shape.Rect_0 : Shape → UInt256
  | .Rect f0 f1 => f0
  | _ => default

def Shape.set_Rect_0 (x : Shape) (v : UInt256) : Shape :=
  match x with
  | .Rect f0 f1 => .Rect v f1
  | x => x

def Shape.Rect_1 : Shape → UInt256
  | .Rect f0 f1 => f1
  | _ => default

def Shape.set_Rect_1 (x : Shape) (v : UInt256) : Shape :=
  match x with
  | .Rect f0 f1 => .Rect f0 v
  | x => x

def init : FeM Unit := do
  return ()

def shape (v0 : UInt256) (v1 : UInt256) : FeM Shape := do
  let v4 := v0
  let v5 := (UInt256.ofNat 0)
  let v6 := decide (v4 = v5)
  let b3 : Shape → FeM Shape := fun v2 => do
    return v2
  if v6 then
    let v7 := Shape.Empty
    let v8 := v7
    b3 v8
  else
    let v9 := v0
    let v10 := v1
    let v11 := decide (v9 = v10)
    let b6 : Shape → FeM Shape := fun v3 => do
      let v19 := v3
      b3 v19
    if v11 then
      let v12 := v0
      let v13 := Shape.Square v12
      let v14 := v13
      b6 v14
    else
      let v15 := v0
      let v16 := v1
      let v17 := Shape.Rect v15 v16
      let v18 := v17
      b6 v18

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v3 ← shape v0 v1
  let b1 : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  match v3 with
  | .Empty =>
    let v4 := (UInt256.ofNat 0)
    let v5 := v4
    b1 v5
  | .Square .. =>
    let v6 := v3.Square_0
    let v7 := v6
    let v8 ← checkedMul v7 v7
    let v9 := v8
    b1 v9
  | .Rect .. =>
    let v10 := v3.Rect_0
    let v11 := v10
    let v12 := v3.Rect_1
    let v13 := v12
    let v14 ← checkedMul v11 v13
    let v15 := v14
    b1 v15

def halve (v0 : UInt256) : FeM Option_u256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 2)
  let v4 ← checkedRem v2 v3
  let v5 := (UInt256.ofNat 0)
  let v6 := decide (v4 = v5)
  let b3 : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if v6 then
    let v7 := v0
    let v8 := (UInt256.ofNat 2)
    let v9 ← checkedDiv v7 v8
    let v10 := Option_u256.Some v9
    let v11 := v10
    b3 v11
  else
    let v12 := Option_u256.None
    let v13 := v12
    b3 v13

def recv0_arm1 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← halve v0
  let b1 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  match v2 with
  | .Some .. =>
    let v3 := v2.Some_0
    let v4 := v3
    let v5 := v4
    b1 v5
  | _ =>
    let v6 := v0
    b1 v6

def recv0_arm2 (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : Shape := default
  let v2 := Shape.Square v0
  let v3 := v2
  r0 := v3
  let b3 : Shape → FeM UInt256 := fun r0 => do
    let mut r0 := r0
    let v13 := r0
    let b6 : Shape → UInt256 → FeM UInt256 := fun r0 v1 => do
      let mut r0 := r0
      return v1
    match v13 with
    | .Square .. =>
      let v14 := r0.Square_0
      let v15 := v14
      b6 r0 v15
    | _ =>
      let v16 := (UInt256.ofNat 0)
      let v17 := v16
      b6 r0 v17
  match r0 with
  | .Square .. =>
    let v6 := (UInt256.ofNat 1)
    let v7 := r0.Square_0
    let v8 ← checkedAdd v7 v6
    r0 := (r0.set_Square_0 v8)
    let v9 := ()
    let v10 := v9
    b3 r0
  | _ =>
    let v11 := ()
    let v12 := v11
    b3 r0


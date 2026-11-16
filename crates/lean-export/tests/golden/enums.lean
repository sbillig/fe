export Lib (Option_u256 Option_u256.None Option_u256.Some)

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

def shape (width_0 : UInt256) (height_1 : UInt256) : FeM Shape := do
  let width_4 := width_0
  let v5 := (UInt256.ofNat 0)
  let v6 := decide (width_4 = v5)
  let b3 : Shape → FeM Shape := fun v2 => do
    return v2
  if v6 then
    let v7 := Shape.Empty
    let v8 := v7
    b3 v8
  else
    let width_9 := width_0
    let height_10 := height_1
    let v11 := decide (width_9 = height_10)
    let b6 : Shape → FeM Shape := fun v3 => do
      let v19 := v3
      b3 v19
    if v11 then
      let width_12 := width_0
      let v13 := Shape.Square width_12
      let v14 := v13
      b6 v14
    else
      let width_15 := width_0
      let height_16 := height_1
      let v17 := Shape.Rect width_15 height_16
      let v18 := v17
      b6 v18

def recv_Area (width_0 : UInt256) (height_1 : UInt256) : FeM UInt256 := do
  let v3 ← shape width_0 height_1
  let b1 : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  match v3 with
  | .Empty =>
    let v4 := (UInt256.ofNat 0)
    let v5 := v4
    b1 v5
  | .Square .. =>
    let v6 := v3.Square_0
    let side := v6
    let v8 ← checkedMul side side
    let v9 := v8
    b1 v9
  | .Rect .. =>
    let v10 := v3.Rect_0
    let width_11 := v10
    let v12 := v3.Rect_1
    let height_13 := v12
    let v14 ← checkedMul width_11 height_13
    let v15 := v14
    b1 v15

def halve (value_0 : UInt256) : FeM Option_u256 := do
  let value_2 := value_0
  let v3 := (UInt256.ofNat 2)
  let v4 ← checkedRem value_2 v3
  let v5 := (UInt256.ofNat 0)
  let v6 := decide (v4 = v5)
  let b3 : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if v6 then
    let value_7 := value_0
    let v8 := (UInt256.ofNat 2)
    let v9 ← checkedDiv value_7 v8
    let v10 := Option_u256.Some v9
    let v11 := v10
    b3 v11
  else
    let v12 := Option_u256.None
    let v13 := v12
    b3 v13

def recv_Half (value_0 : UInt256) : FeM UInt256 := do
  let v2 ← halve value_0
  let b1 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  match v2 with
  | .Some .. =>
    let v3 := v2.Some_0
    let half_4 := v3
    let half_5 := half_4
    b1 half_5
  | _ =>
    let value_6 := value_0
    b1 value_6

def recv_Grow (side_0 : UInt256) : FeM UInt256 := do
  let mut current : Shape := default
  let v2 := Shape.Square side_0
  let current_3 := v2
  current := current_3
  let b3 : Shape → FeM UInt256 := fun current => do
    let mut current := current
    let v13 := current
    let b6 : Shape → UInt256 → FeM UInt256 := fun current v1 => do
      let mut current := current
      return v1
    match v13 with
    | .Square .. =>
      let side_14 := current.Square_0
      let side_15 := side_14
      b6 current side_15
    | _ =>
      let v16 := (UInt256.ofNat 0)
      let v17 := v16
      b6 current v17
  match current with
  | .Square .. =>
    let v6 := (UInt256.ofNat 1)
    let v7 := current.Square_0
    let v8 ← checkedAdd v7 v6
    current := (current.set_Square_0 v8)
    let v9 := ()
    let v10 := v9
    b3 current
  | _ =>
    let v11 := ()
    let v12 := v11
    b3 current


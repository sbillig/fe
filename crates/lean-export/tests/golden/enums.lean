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

/-- The `init` of contract `Enums`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `enums::shape` -/
def shape (width : UInt256) (height : UInt256) : FeM Shape := do
  let v6 := decide (width = (UInt256.ofNat 0))
  let join : Shape → FeM Shape := fun v2 => do
    return v2
  if v6 then
    let v7 := Shape.Empty
    join v7
  else
    let v11 := decide (width = height)
    let join_1 : Shape → FeM Shape := fun v3 => do
      join v3
    if v11 then
      let v13 := Shape.Square width
      join_1 v13
    else
      let v17 := Shape.Rect width height
      join_1 v17

/-- `Area { width, height } -> u256`, selector `0x15529a52` (`area(uint256,uint256)`) -/
def recv_Area (width_0 : UInt256) (height_1 : UInt256) : FeM UInt256 := do
  let v3 ← shape width_0 height_1
  let join : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  match v3 with
  | .Empty =>
    join (UInt256.ofNat 0)
  | .Square .. =>
    let side := v3.Square_0
    let v8 ← checkedMul side side
    join v8
  | .Rect .. =>
    let width_10 := v3.Rect_0
    let height_12 := v3.Rect_1
    let v14 ← checkedMul width_10 height_12
    join v14

/-- `enums::halve` -/
def halve (value : UInt256) : FeM Option_u256 := do
  let v4 ← checkedRem value (UInt256.ofNat 2)
  let v6 := decide (v4 = (UInt256.ofNat 0))
  let join : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if v6 then
    let v9 ← checkedDiv value (UInt256.ofNat 2)
    let v10 := Option_u256.Some v9
    join v10
  else
    let v12 := Option_u256.None
    join v12

/-- `Half { value } -> u256`, selector `0xdee6088b` (`half(uint256)`) -/
def recv_Half (value : UInt256) : FeM UInt256 := do
  let v2 ← halve value
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  match v2 with
  | .Some .. =>
    let half := v2.Some_0
    join half
  | _ =>
    join value

/-- `Grow { side } -> u256`, selector `0x28531f06` (`grow(uint256)`) -/
def recv_Grow (side_0 : UInt256) : FeM UInt256 := do
  let mut current : Shape := default
  let current_2 := Shape.Square side_0
  current := current_2
  let join : Shape → FeM UInt256 := fun current => do
    let mut current := current
    let v13 := current
    let join_1 : Shape → UInt256 → FeM UInt256 := fun current v1 => do
      let mut current := current
      return v1
    match v13 with
    | .Square .. =>
      let side_14 := current.Square_0
      join_1 current side_14
    | _ =>
      join_1 current (UInt256.ofNat 0)
  match current with
  | .Square .. =>
    let v7 := current.Square_0
    let v8 ← checkedAdd v7 (UInt256.ofNat 1)
    current := (current.set_Square_0 v8)
    join current
  | _ =>
    join current


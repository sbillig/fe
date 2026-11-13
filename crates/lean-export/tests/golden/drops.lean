structure Guard where
  n : UInt256
deriving Inhabited

inductive Option_Guard where
  | None
  | Some (f0 : Guard)
deriving Inhabited

def Option_Guard.tag : Option_Guard → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Guard.Some_0 : Option_Guard → Guard
  | .Some f0 => f0
  | _ => default

def Option_Guard.set_Some_0 (x : Option_Guard) (v : Guard) : Option_Guard :=
  match x with
  | .Some f0 => .Some v
  | x => x

structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def Field.pair : Place :=
  .storage (UInt256.ofNat 0)

def store_Pair (p : Place) (v : Pair) : FeM Unit := do
  store (p.offset 0) v.a
  store (p.offset 1) v.b

def Layout.Pair : Layout :=
  { storage := [0, 1] }

def init : FeM Unit := do
  return ()

def Guard_drop (v0 : Guard) : FeM Unit := do
  let v1 := ()
  return ()

def take (v0 : Guard) : FeM UInt256 := do
  let v1 := v0.n
  let _ ← Guard_drop v0
  return v1

def recv0_arm0 (v0 : Bool) : FeM UInt256 := do
  let mut f0 := true
  let v2 := (UInt256.ofNat 1)
  let v3 := (Guard.mk v2)
  let v4 := v3
  let b3 : Bool → UInt256 → FeM UInt256 := fun f0 v1 => do
    let mut f0 := f0
    if f0 then
      let _ ← Guard_drop v4
      pure ()
    f0 := false
    return v1
  if v0 then
    let v5 := (UInt256.ofNat 0)
    let v6 := v5
    b3 f0 v6
  else
    let v7 := v4
    f0 := false
    let v8 ← take v7
    let v9 := v8
    b3 f0 v9

def Option_is_some_Guard (v0 : Option_Guard) : FeM Bool := do
  let v2 := v0
  let b1 : Bool → FeM Bool := fun v1 => do
    return v1
  match v2 with
  | .Some .. =>
    let v3 := true
    let v4 := v3
    b1 v4
  | .None =>
    let v5 := false
    let v6 := v5
    b1 v6

def recv0_arm1 (v0 : Bool) : FeM UInt256 := do
  let mut r0 : Option_Guard := default
  let b3 : Option_Guard → Option_Guard → FeM UInt256 := fun r0 v1 => do
    let mut r0 := r0
    let v8 := v1
    r0 := v8
    let v10 ← Option_is_some_Guard r0
    let v11 := Bool.toUInt256 v10
    match r0 with
    | .Some .. =>
      let _ ← Guard_drop r0.Some_0
      pure ()
    | _ => pure ()
    return v11
  if v0 then
    let v2 := (UInt256.ofNat 2)
    let v3 := (Guard.mk v2)
    let v4 := Option_Guard.Some v3
    let v5 := v4
    b3 r0 v5
  else
    let v6 := Option_Guard.None
    let v7 := v6
    b3 r0 v7

def recv0_arm2 (v0 : UInt256) : FeM UInt256 := do
  let v1 := (Pair.mk v0 v0)
  store_Pair Field.pair v1
  let v2 := ()
  let v3 ← load (Field.pair.field Layout.Pair 0)
  return v3


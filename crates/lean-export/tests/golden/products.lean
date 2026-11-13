inductive Option_mut_u8 where
  | None
  | Some (f0 : UInt256)
deriving Inhabited

def Option_mut_u8.tag : Option_mut_u8 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_mut_u8.Some_0 : Option_mut_u8 → UInt256
  | .Some f0 => f0
  | _ => default

def Option_mut_u8.set_Some_0 (x : Option_mut_u8) (v : UInt256) : Option_mut_u8 :=
  match x with
  | .Some f0 => .Some v
  | x => x

def Field.pair : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Pair : Layout :=
  { storage := [0, 1] }

structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def Field.small : Place :=
  .storage (UInt256.ofNat 2)

def Layout.Small : Layout :=
  { storage := [0, 0] }

def init : FeM Unit := do
  return ()

def add_to (v0 : Place) (v1 : UInt256) : FeM Unit := do
  let v2 := (v0, v1)
  let v3 := v2.fst
  let v4 := v3
  let v6 := v1
  let v7 ← load v4
  let v8 ← checkedAdd v7 v6
  store v4 v8
  let v9 := ()
  return ()

def add_to_1 (v0 : UInt256) (v1 : Place) : FeM (Unit × UInt256) := do
  let mut v0 := v0
  let v2 := (v0, v1)
  let v5 := v2.snd
  let v6 ← load v5
  let v7 := v0
  let v8 ← checkedAdd v7 v6
  v0 := v8
  let v9 := ()
  return ((), v0)

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let mut r1 : UInt256 := default
  let v1 := v0
  r1 := v1
  let v2 := (Field.pair.field Layout.Pair 0)
  let v4 ← add_to v2 r1
  let v6 := (Field.pair.field Layout.Pair 0)
  let (v7, v7_0) ← add_to_1 r1 v6
  r1 := v7_0
  let v8 := r1
  return v8

def Pair_at (v0 : Place) : FeM (UInt256 × Place) := do
  let v1 ← load (v0.field Layout.Pair 0)
  let v2 := (v0.field Layout.Pair 1)
  let v3 := (v1, v2)
  return (v1, v2)

def bump (v0 : UInt256) (v1 : Place) : FeM UInt256 := do
  let v2 := (v0, v1)
  let v3 := v2.fst
  let v4 := v2.snd
  let v5 := v4
  let v6 ← load v5
  let v7 ← checkedAdd v6 v3
  store v5 v7
  let v8 := ()
  let v9 ← load v5
  return v9

def Pair_at_1 (v0 : Pair) : FeM ((UInt256 × UInt256) × ((UInt256 × UInt256) → FeM Pair)) := do
  let mut v0 := v0
  let v1 := v0.a
  let v3 := (v1, v0.b)
  let slide : (UInt256 × UInt256) → FeM Pair := fun g => do
    let mut v0 := v0
    v0 := { v0 with b := g.snd }
    return v0
  return ((v1, v0.b), slide)

def bump_1 (v0 : UInt256) (v1 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut v1 := v1
  let v2 := (v0, v1)
  let v3 := v2.fst
  let v6 := v1
  let v7 ← checkedAdd v6 v3
  v1 := v7
  let v8 := ()
  let v9 := v1
  return (v9, v1)

def recv0_arm1 (v0 : UInt256) : FeM UInt256 := do
  let mut r1 : Pair := default
  let mut v12 : (UInt256 × UInt256) := default
  let v1 := Field.pair
  let v2 ← Pair_at v1
  let v3 := v2.fst
  let v4 := v2.snd
  let v5 := v4
  let v6 ← bump v3 v5
  let v7 := v6
  let v8 := (UInt256.ofNat 1)
  let v9 := (Pair.mk v0 v8)
  let v10 := v9
  r1 := v10
  let (v12_g, v12_k) ← Pair_at_1 r1
  v12 := v12_g
  let v13 := v12.fst
  let (v16, v16_0) ← bump_1 v13 v12.snd
  v12 := { v12 with snd := v16_0 }
  let v12_0 ← v12_k v12
  r1 := v12_0
  let v17 ← checkedAdd v7 v16
  return v17

def Small_maybe (v0 : Place) (v1 : Bool) : FeM (Option_mut_u8 × (Option_mut_u8 → FeM Unit)) := do
  let mut h2 : UInt256 := default
  if v1 then
    h2 := (← loadIntAt (.u 8) (v0.field Layout.Small 1) ⟨1, 1, true⟩)
    let v3 := Option_mut_u8.Some h2
    let slide : Option_mut_u8 → FeM Unit := fun g => do
      let mut h2 := h2
      h2 := g.Some_0
      storeIntAt (.u 8) (v0.field Layout.Small 1) ⟨1, 1, true⟩ h2
      return ()
    return ((Option_mut_u8.Some h2), slide)
  else
    let v4 := Option_mut_u8.None
    let slide : Option_mut_u8 → FeM Unit := fun g => do
      let mut h2 := h2
      return ()
    return (Option_mut_u8.None, slide)

def recv0_arm2 (v0 : Bool) : FeM UInt256 := do
  let mut v2 : Option_mut_u8 := default
  let v1 := Field.small
  let (v2_g, v2_k) ← Small_maybe v1 v0
  v2 := v2_g
  let v3 := v2
  let b3 : Option_mut_u8 → FeM UInt256 := fun v2 => do
    let mut v2 := v2
    let v13 ← loadIntAt (.u 8) (Field.small.field Layout.Small 1) ⟨1, 1, true⟩
    let v14 := v13
    return v14
  match v3 with
  | .Some .. =>
    let v6 := (UInt256.ofNat 40)
    let v7 := v2.Some_0
    let v8 ← IntTy.add (.u 8) v7 v6
    v2 := (v2.set_Some_0 v8)
    let _ ← v2_k v2
    let v9 := ()
    let v10 := v9
    b3 v2
  | _ =>
    let _ ← v2_k v2
    let v11 := ()
    let v12 := v11
    b3 v2


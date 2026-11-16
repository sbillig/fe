export Lib (Option_mut_u8 Option_mut_u8.None Option_mut_u8.Some)

def Field.pair : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Pair : Layout :=
  { storage := [0, 1] }

/-- `pair.a` -/
abbrev Field.pair.a : Place :=
  Field.pair.field Layout.Pair 0

structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def Field.small : Place :=
  .storage (UInt256.ofNat 2)

def Layout.Small : Layout :=
  { storage := [0, 0] }

/-- `small.b` -/
abbrev Field.small.b : Place :=
  Field.small.field Layout.Small 1

/-- The `init` of contract `Products`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `products::add_to` -/
def add_to (v0 : Place) (v1 : UInt256) : FeM Unit := do
  let v2 := (v0, v1)
  let a := v2.fst
  let v7 ← load a
  let v8 ← checkedAdd v7 v1
  store a v8
  return ()

/-- `products::add_to` -/
def add_to_1 (v0 : UInt256) (v1 : Place) : FeM (Unit × UInt256) := do
  let mut v0 := v0
  let v2 := (v0, v1)
  let b ← load v2.snd
  let v7 := v0
  let v8 ← checkedAdd v7 b
  v0 := v8
  return ((), v0)

/-- `Parts { by } -> u256`, selector `0xc9eb4662` (`parts(uint256)`) -/
def recv_Parts («by» : UInt256) : FeM UInt256 := do
  let mut «local» : UInt256 := default
  «local» := «by»
  let v2 := Field.pair.a
  let _ ← add_to v2 «local»
  let v6 := Field.pair.a
  let (v7, v7_0) ← add_to_1 «local» v6
  «local» := v7_0
  let local_8 := «local»
  return local_8

/-- `Pair::at` -/
def Pair_at (self : Place) : FeM (UInt256 × Place) := do
  let v1 ← load (self.field Layout.Pair 0)
  let v2 := (self.field Layout.Pair 1)
  return (v1, v2)

/-- `products::bump` -/
def bump (v0 : UInt256) (v1 : Place) : FeM UInt256 := do
  let v2 := (v0, v1)
  let x := v2.snd
  let v6 ← load x
  let v7 ← checkedAdd v6 v2.fst
  store x v7
  let v9 ← load x
  return v9

/-- `Pair::at` -/
def Pair_at_1 (self : Pair) : FeM ((UInt256 × UInt256) × ((UInt256 × UInt256) → FeM Pair)) := do
  let mut self := self
  let v1 := self.a
  let v3 := (v1, self.b)
  let slide : (UInt256 × UInt256) → FeM Pair := fun g => do
    let mut self := self
    self := { self with b := g.snd }
    return self
  return ((v1, self.b), slide)

/-- `products::bump` -/
def bump_1 (v0 : UInt256) (v1 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut v1 := v1
  let v2 := (v0, v1)
  let v6 := v1
  let v7 ← checkedAdd v6 v2.fst
  v1 := v7
  let v9 := v1
  return (v9, v1)

/-- `Granted { by } -> u256`, selector `0x139fda34` (`granted(uint256)`) -/
def recv_Granted («by» : UInt256) : FeM UInt256 := do
  let mut «local» : Pair := default
  let mut v12 : (UInt256 × UInt256) := default
  let v1 := Field.pair
  let v2 ← Pair_at v1
  let stored ← bump v2.fst v2.snd
  «local» := (Pair.mk «by» (UInt256.ofNat 1))
  let (v12_g, v12_k) ← Pair_at_1 «local»
  v12 := v12_g
  let v13 := v12.fst
  let (v16, v16_0) ← bump_1 v13 v12.snd
  v12 := { v12 with snd := v16_0 }
  let v12_0 ← v12_k v12
  «local» := v12_0
  let v17 ← checkedAdd stored v16
  return v17

/-- `Small::maybe` -/
def Small_maybe (self : Place) (second : Bool) : FeM (Option_mut_u8 × (Option_mut_u8 → FeM Unit)) := do
  let mut h2 : UInt256 := default
  if second then
    h2 := (← loadIntAt (.u 8) (self.field Layout.Small 1) ⟨1, 1, true⟩)
    let v3 := Option_mut_u8.Some h2
    let slide : Option_mut_u8 → FeM Unit := fun g => do
      let mut h2 := h2
      h2 := g.Some_0
      storeIntAt (.u 8) (self.field Layout.Small 1) ⟨1, 1, true⟩ h2
      return ()
    return ((Option_mut_u8.Some h2), slide)
  else
    let slide : Option_mut_u8 → FeM Unit := fun g => do
      let mut h2 := h2
      return ()
    return (Option_mut_u8.None, slide)

/-- `Lane { second } -> u256`, selector `0x2a160f7a` (`lane(bool)`) -/
def recv_Lane (second : Bool) : FeM UInt256 := do
  let mut v2 : Option_mut_u8 := default
  let v1 := Field.small
  let (v2_g, v2_k) ← Small_maybe v1 second
  v2 := v2_g
  let v3 := v2
  let join : Option_mut_u8 → FeM UInt256 := fun v2 => do
    let mut v2 := v2
    let v13 ← loadIntAt (.u 8) Field.small.b ⟨1, 1, true⟩
    return v13
  match v3 with
  | .Some .. =>
    let v7 := v2.Some_0
    let v8 ← IntTy.add (.u 8) v7 (UInt256.ofNat 40)
    v2 := (v2.set_Some_0 v8)
    let _ ← v2_k v2
    join v2
  | _ =>
    let _ ← v2_k v2
    join v2


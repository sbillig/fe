structure Guard where
  n : UInt256
deriving Inhabited

inductive Option_Guard where
  | None
  | Some (f0 : Guard)

instance : Inhabited Option_Guard := ⟨Option_Guard.None⟩

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

/-- `pair.a` -/
abbrev Field.pair.a : Place :=
  Field.pair.field Layout.Pair 0

/-- The `init` of contract `Drops`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `<Guard as Drop>::drop` -/
def Guard_drop (self : Guard) : FeM Unit := do
  return ()

/-- `drops::take` -/
def take (guard : Guard) : FeM UInt256 := do
  let _ ← Guard_drop guard
  return guard.n

/-- `Moved { keep } -> u256`, selector `0x4f4c6300` (`moved(bool)`) -/
def recv_Moved (keep : Bool) : FeM UInt256 := do
  let mut f0 := true
  -- let guard = Guard { n: 1 }
  let guard := (Guard.mk (UInt256.ofNat 1))
  let join : Bool → UInt256 → FeM UInt256 := fun f0 v1 => do
    let mut f0 := f0
    if f0 then
      let _ ← Guard_drop guard
      pure ()
    f0 := false
    return v1
  -- if keep {
  if keep then
    join f0 (UInt256.ofNat 0)
  else
    f0 := false
    -- take(guard)
    let v8 ← take guard
    join f0 v8

/-- `Option<Guard>::is_some` -/
def Option_is_some_Guard (self : Option_Guard) : FeM Bool := do
  let join : Bool → FeM Bool := fun v1 => do
    return v1
  -- match self {
  match self with
  | .Some .. =>
    join true
  | .None =>
    join false

/-- `Payload { some } -> u256`, selector `0xacf03ddb` (`payload(bool)`) -/
def recv_Payload (some : Bool) : FeM UInt256 := do
  let mut guard : Option_Guard := default
  let join : Option_Guard → Option_Guard → FeM UInt256 := fun guard guard_1 => do
    let mut guard := guard
    -- let guard = if some {
    guard := guard_1
    -- guard.is_some() as u256
    let v10 ← Option_is_some_Guard guard
    match guard with
    | .Some .. =>
      let _ ← Guard_drop guard.Some_0
      pure ()
    | _ => pure ()
    return Bool.toUInt256 v10
  -- let guard = if some {
  if some then
    -- Option::Some(Guard { n: 2 })
    let v3 := (Guard.mk (UInt256.ofNat 2))
    let v4 := Option_Guard.Some v3
    join guard v4
  else
    -- Option::None
    let v6 := Option_Guard.None
    join guard v6

/-- `Stored { n } -> u256`, selector `0x8c396222` (`stored(uint256)`) -/
def recv_Stored (n : UInt256) : FeM UInt256 := do
  -- pair = Pair { a: n, b: n }
  store_Pair Field.pair (Pair.mk n n)
  -- pair.a
  let v3 ← load Field.pair.a
  return v3


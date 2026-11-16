export Lib (Layout.TSlot_bool)

def Field.packed : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Packed : Layout :=
  { storage := [0, 0, 0, 0, 1] }

/-- `packed.delta` -/
abbrev Field.packed.delta : Place :=
  Field.packed.field Layout.Packed 2

structure Packed where
  flag : Bool
  small : UInt256
  delta : UInt256
  owner : UInt256
  total : UInt256
deriving Inhabited

def load_Packed (p : Place) : FeM Packed := do
  let f0 ← loadBoolAt (p.offset 0) ⟨0, 1, true⟩
  let f1 ← loadIntAt (.u 8) (p.offset 0) ⟨1, 1, true⟩
  let f2 ← loadIntAt (.i 16) (p.offset 0) ⟨2, 2, true⟩
  let f3 ← loadIntAt (.u 256) (p.offset 0) ⟨4, 20, true⟩
  let f4 ← load (p.offset 1)
  return (Packed.mk f0 f1 f2 f3 f4)

def store_Packed (p : Place) (v : Packed) : FeM Unit := do
  let b0 ← Lane.pack ⟨0, 1, true⟩ (.u 8) (Bool.toUInt256 v.flag)
  let b1 ← Lane.pack ⟨1, 1, true⟩ (.u 8) v.small
  let b2 ← Lane.pack ⟨2, 2, true⟩ (.i 16) v.delta
  let b3 ← Lane.pack ⟨4, 20, true⟩ (.u 256) v.owner
  store (p.offset 0) (packed [(0, b0), (1, b1), (2, b2), (4, b3)])
  store (p.offset 1) v.total

def Field.phase : Place :=
  .storage (UInt256.ofNat 2)

inductive Phase where
  | Idle
  | Running (f0 : UInt256) (f1 : UInt256)
  | Done (f0 : Bool)
deriving Inhabited

def Phase.tag : Phase → UInt256
  | .Idle => UInt256.ofNat 0
  | .Running .. => UInt256.ofNat 1
  | .Done .. => UInt256.ofNat 2

def Phase.Running_0 : Phase → UInt256
  | .Running f0 f1 => f0
  | _ => default

def Phase.set_Running_0 (x : Phase) (v : UInt256) : Phase :=
  match x with
  | .Running f0 f1 => .Running v f1
  | x => x

def Phase.Running_1 : Phase → UInt256
  | .Running f0 f1 => f1
  | _ => default

def Phase.set_Running_1 (x : Phase) (v : UInt256) : Phase :=
  match x with
  | .Running f0 f1 => .Running f0 v
  | x => x

def Phase.Done_0 : Phase → Bool
  | .Done f0 => f0
  | _ => default

def Phase.set_Done_0 (x : Phase) (v : Bool) : Phase :=
  match x with
  | .Done f0 => .Done v
  | x => x

def load_Phase (p : Place) : FeM Phase := do
  match (← loadTag 8 3 p) with
  | 0 => do
    return .Idle
  | 1 => do
    let f0 ← loadInt (.u 64) (p.offset 1)
    let f1 ← load (p.offset 2)
    return .Running f0 f1
  | _ => do
    let f0 ← loadBool (p.offset 1)
    return .Done f0

def store_Phase (p : Place) (v : Phase) : FeM Unit := do
  match v with
  | .Idle => do
    store p (UInt256.ofNat 0)
  | .Running f0 f1 => do
    store p (UInt256.ofNat 1)
    storeInt (.u 64) (p.offset 1) f0
    store (p.offset 2) f1
  | .Done f0 => do
    store p (UInt256.ofNat 2)
    storeBool (p.offset 1) f0

def Field.levels : Place :=
  .storage (UInt256.ofNat 5)

def Field.narrow : Place :=
  .storage (UInt256.ofNat 8)

def Field.lock : Place :=
  .storage (UInt256.ofNat 0)

def Field.alone : Place :=
  .storage (UInt256.ofNat 8)

/-- The `init` of contract `State`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `SetDelta { delta } -> i16`, selector `0x36899e92` (`setDelta(int16)`) -/
def recv_SetDelta (delta : UInt256) : FeM UInt256 := do
  let old ← loadIntAt (.i 16) Field.packed.delta ⟨2, 2, true⟩
  storeIntAt (.i 16) Field.packed.delta ⟨2, 2, true⟩ delta
  return old

/-- `Replace { flag, small, delta, owner, total } -> u256`, selector `0x1e84534a` (`replace(bool,uint8,int16,address,uint256)`) -/
def recv_Replace (flag : Bool) (small : UInt256) (delta : UInt256) (owner : UInt256) (total : UInt256) : FeM UInt256 := do
  let packed_1 ← load_Packed Field.packed
  let v7 := (Packed.mk flag small delta owner total)
  store_Packed Field.packed v7
  let v9 := packed_1.total
  return v9

/-- `Advance { by } -> u256`, selector `0x71d6ddd6` (`advance(uint256)`) -/
def recv_Advance («by» : UInt256) : FeM UInt256 := do
  let v3 ← load_Phase Field.phase
  let b1 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  match v3 with
  | .Idle =>
    let v5 := Phase.Running (UInt256.ofNat 1) «by»
    store_Phase Field.phase v5
    b1 (UInt256.ofNat 0)
  | .Running .. =>
    let count ← loadInt (.u 64) (Field.phase.offset 1)
    let sum ← load (Field.phase.offset 2)
    let v11 := decide («by».toNat < sum.toNat)
    let v12 := Phase.Done v11
    store_Phase Field.phase v12
    b1 sum
  | .Done .. =>
    let ok ← loadBool (Field.phase.offset 1)
    let b7 : UInt256 → FeM UInt256 := fun v2 => do
      b1 v2
    if ok then
      b7 (UInt256.ofNat 1)
    else
      b7 (UInt256.ofNat 2)

/-- `Mark { at, level } -> u8`, selector `0x0a774e1e` (`mark(uint256,uint8)`) -/
def recv_Mark («at» : UInt256) (level : UInt256) : FeM UInt256 := do
  checkIndex «at» 3
  let old ← loadInt (.u 8) (Field.levels.index 1 «at»)
  checkIndex «at» 3
  storeInt (.u 8) (Field.levels.index 1 «at») level
  storeIntAt (.u 8) Field.narrow ⟨1, 1, true⟩ old
  let narrow ← loadIntAt (.u 8) Field.narrow ⟨1, 1, true⟩
  return narrow

/-- `Lock -> bool`, selector `0xf83d08ba` (`lock()`) -/
def recv_Lock : FeM Bool := do
  let was ← loadBoolAt ((Field.lock.pin .transient).field Layout.TSlot_bool 0) ⟨0, 1, false⟩
  storeBoolAt ((Field.lock.pin .transient).field Layout.TSlot_bool 0) ⟨0, 1, false⟩ true
  storeBoolAt Field.alone ⟨0, 1, true⟩ was
  let alone ← loadBoolAt Field.alone ⟨0, 1, true⟩
  return alone


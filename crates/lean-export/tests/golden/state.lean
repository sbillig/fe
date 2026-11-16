export Lib (Layout.TSlot_bool)

def Field.packed : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Packed : Layout :=
  { storage := [0, 0, 0, 0, 1] }

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

def init : FeM Unit := do
  return ()

def recv_SetDelta (delta : UInt256) : FeM UInt256 := do
  let v1 ← loadIntAt (.i 16) (Field.packed.field Layout.Packed 2) ⟨2, 2, true⟩
  let old := v1
  storeIntAt (.i 16) (Field.packed.field Layout.Packed 2) ⟨2, 2, true⟩ delta
  let v3 := ()
  return old

def recv_Replace (flag : Bool) (small : UInt256) (delta : UInt256) (owner : UInt256) (total : UInt256) : FeM UInt256 := do
  let packed_5 ← load_Packed Field.packed
  let packed_6 := packed_5
  let v7 := (Packed.mk flag small delta owner total)
  store_Packed Field.packed v7
  let v8 := ()
  let v9 := packed_6.total
  return v9

def recv_Advance («by» : UInt256) : FeM UInt256 := do
  let v3 ← load_Phase Field.phase
  let b1 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  match v3 with
  | .Idle =>
    let v4 := (UInt256.ofNat 1)
    let v5 := Phase.Running v4 «by»
    store_Phase Field.phase v5
    let v6 := ()
    let v7 := (UInt256.ofNat 0)
    let v8 := v7
    b1 v8
  | .Running .. =>
    let count ← loadInt (.u 64) (Field.phase.offset 1)
    let sum_10 ← load (Field.phase.offset 2)
    let v11 := decide («by».toNat < sum_10.toNat)
    let v12 := Phase.Done v11
    store_Phase Field.phase v12
    let v13 := ()
    let sum_14 := sum_10
    b1 sum_14
  | .Done .. =>
    let ok ← loadBool (Field.phase.offset 1)
    let b7 : UInt256 → FeM UInt256 := fun v2 => do
      let v20 := v2
      b1 v20
    if ok then
      let v16 := (UInt256.ofNat 1)
      let v17 := v16
      b7 v17
    else
      let v18 := (UInt256.ofNat 2)
      let v19 := v18
      b7 v19

def recv_Mark («at» : UInt256) (level : UInt256) : FeM UInt256 := do
  checkIndex «at» 3
  let v2 ← loadInt (.u 8) (Field.levels.index 1 «at»)
  let old := v2
  checkIndex «at» 3
  storeInt (.u 8) (Field.levels.index 1 «at») level
  let v4 := ()
  storeIntAt (.u 8) Field.narrow ⟨1, 1, true⟩ old
  let v5 := ()
  let narrow ← loadIntAt (.u 8) Field.narrow ⟨1, 1, true⟩
  return narrow

def recv_Lock : FeM Bool := do
  let v0 ← loadBoolAt ((Field.lock.pin .transient).field Layout.TSlot_bool 0) ⟨0, 1, false⟩
  let was := v0
  let v2 := true
  storeBoolAt ((Field.lock.pin .transient).field Layout.TSlot_bool 0) ⟨0, 1, false⟩ v2
  let v3 := ()
  storeBoolAt Field.alone ⟨0, 1, true⟩ was
  let v4 := ()
  let alone ← loadBoolAt Field.alone ⟨0, 1, true⟩
  return alone


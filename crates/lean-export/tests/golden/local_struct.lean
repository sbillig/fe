structure Cursor where
  pos : UInt256
  base : UInt256
deriving Inhabited

structure Span where
  start : Cursor
  len : UInt256
deriving Inhabited

def Field.saved : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Cursor : Layout :=
  { storage := [0, 1] }

def init : FeM Unit := do
  return ()

def advance (v0 : Cursor) (v1 : UInt256) : FeM (UInt256 × Cursor) := do
  let mut v0 := v0
  let v2 := v0.pos
  let v3 := v2
  let v4 := v1
  let v5 ← checkedAdd v3 v4
  v0 := { v0 with pos := v5 }
  let v6 := ()
  return (v3, v0)

def skip (v0 : Cursor) (v1 : UInt256) : FeM (Unit × Cursor) := do
  let mut v0 := v0
  let v2 := v1
  let v3 := (UInt256.ofNat 32)
  let v4 := decide (v3.toNat < v2.toNat)
  let b3 : Cursor → FeM (Unit × Cursor) := fun v0 => do
    let mut v0 := v0
    let v10 := v0.pos
    v0 := { v0 with base := v10 }
    let v11 := ()
    return ((), v0)
  if v4 then
    let v5 := (UInt256.ofNat 32)
    let (v6, v6_0) ← advance v0 v5
    v0 := v6_0
    let v7 := v6
    b3 v0
  else
    let (v8, v8_0) ← advance v0 v1
    v0 := v8_0
    let v9 := v8
    b3 v0

def «end» (v0 : Cursor) : FeM UInt256 := do
  let v1 := v0.pos
  let v2 := v0.base
  let v3 ← checkedAdd v1 v2
  return v3

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : Cursor := default
  let mut r1 : Span := default
  let v1 := (UInt256.ofNat 0)
  let v2 := (UInt256.ofNat 4)
  let v3 := Cursor.mk v1 v2
  let v4 := v3
  r0 := v4
  let (v6, v6_0) ← advance r0 v0
  r0 := v6_0
  let v7 := v6
  let (v9, v9_0) ← skip r0 v0
  r0 := v9_0
  let v10 := r0
  let v11 := v10
  let v12 := Span.mk v11 v0
  let v13 := v12
  r1 := v13
  let v14 := r1.len
  let (v16, v16_0) ← advance r1.start v14
  r1 := { r1 with start := v16_0 }
  let v17 := v16
  let v18 ← checkedAdd v7 v17
  let v21 ← «end» r1.start
  let v22 ← checkedAdd v18 v21
  return v22

def advance_1 (v0 : Place) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← load (v0.field Layout.Cursor 0)
  let v3 := v2
  let v4 := v1
  let v5 ← checkedAdd v3 v4
  store (v0.field Layout.Cursor 0) v5
  let v6 := ()
  return v3

def recv0_arm1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := Field.saved
  let v2 ← advance_1 v1 v0
  return v2


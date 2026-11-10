structure Cursor where
  pos : UInt256
  base : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def make (v0 : UInt256) : FeM Cursor := do
  let v1 := v0
  let v2 := (UInt256.ofNat 4)
  let v3 := Cursor.mk v1 v2
  return v3

def «end» (v0 : Cursor) : FeM UInt256 := do
  let v1 := v0.pos
  let v2 := v0.base
  let v3 ← checkedAdd v1 v2
  return v3

def advance (v0 : Cursor) (v1 : UInt256) : FeM (UInt256 × Cursor) := do
  let mut v0 := v0
  let v2 := v0.pos
  let v3 := v2
  let v4 := v1
  let v5 ← checkedAdd v3 v4
  v0 := { v0 with pos := v5 }
  let v6 := ()
  return (v3, v0)

def c_end (v0 : Cursor) : FeM UInt256 := do
  let v1 ← «end» v0
  return v1

def consume (r0 : Cursor) (v0 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  let mut r1 : Cursor := default
  let (v2, v2_0) ← advance r0 v0
  r0 := v2_0
  let v3 ← make v0
  r1 := v3
  let v5 ← c_end r1
  let v6 ← checkedAdd v2 v5
  return v6

def bump (r0 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  let v0 := (UInt256.ofNat 1)
  let v1 := r0
  let v2 ← checkedAdd v1 v0
  let v3 := v2
  r0 := v3
  let v4 := ()
  let v5 := r0
  return v5

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : Cursor := default
  let v1 ← make v0
  r0 := v1
  let v3 ← «end» r0
  let v4 := (UInt256.ofNat 1)
  let v5 ← make v4
  let v6 ← consume v5 v0
  let v7 ← checkedAdd v3 v6
  let v8 ← bump v0
  let v9 ← checkedAdd v7 v8
  return v9


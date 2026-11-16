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

def Cursor_advance (self : Cursor) (by_1 : UInt256) : FeM (UInt256 × Cursor) := do
  let mut self := self
  let v2 := self.pos
  let «at» := v2
  let by_4 := by_1
  let v5 ← checkedAdd «at» by_4
  self := { self with pos := v5 }
  let v6 := ()
  return («at», self)

def Cursor_skip (self : Cursor) (by_1 : UInt256) : FeM (Unit × Cursor) := do
  let mut self := self
  let by_2 := by_1
  let v3 := (UInt256.ofNat 32)
  let v4 := decide (v3.toNat < by_2.toNat)
  let b3 : Cursor → FeM (Unit × Cursor) := fun self => do
    let mut self := self
    let v10 := self.pos
    self := { self with base := v10 }
    let v11 := ()
    return ((), self)
  if v4 then
    let v5 := (UInt256.ofNat 32)
    let (v6, v6_0) ← Cursor_advance self v5
    self := v6_0
    let v7 := v6
    b3 self
  else
    let (v8, v8_0) ← Cursor_advance self by_1
    self := v8_0
    let v9 := v8
    b3 self

def Cursor_end (self : Cursor) : FeM UInt256 := do
  let v1 := self.pos
  let v2 := self.base
  let v3 ← checkedAdd v1 v2
  return v3

def recv_Sum (step : UInt256) : FeM UInt256 := do
  let mut c : Cursor := default
  let mut s : Span := default
  let v1 := (UInt256.ofNat 0)
  let v2 := (UInt256.ofNat 4)
  let v3 := (Cursor.mk v1 v2)
  let c_4 := v3
  c := c_4
  let (v6, v6_0) ← Cursor_advance c step
  c := v6_0
  let a := v6
  let (v9, v9_0) ← Cursor_skip c step
  c := v9_0
  let c_10 := c
  let c_11 := c_10
  let v12 := (Span.mk c_11 step)
  let s_13 := v12
  s := s_13
  let v14 := s.len
  let (v16, v16_0) ← Cursor_advance s.start v14
  s := { s with start := v16_0 }
  let b := v16
  let v18 ← checkedAdd a b
  let v21 ← Cursor_end s.start
  let v22 ← checkedAdd v18 v21
  return v22

def Cursor_advance_1 (self : Place) (by_1 : UInt256) : FeM UInt256 := do
  let v2 ← load (self.field Layout.Cursor 0)
  let «at» := v2
  let by_4 := by_1
  let v5 ← checkedAdd «at» by_4
  store (self.field Layout.Cursor 0) v5
  let v6 := ()
  return «at»

def recv_Bump (step : UInt256) : FeM UInt256 := do
  let v1 := Field.saved
  let v2 ← Cursor_advance_1 v1 step
  return v2


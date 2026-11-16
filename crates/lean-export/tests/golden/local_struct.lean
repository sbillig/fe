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

/-- The `init` of contract `Walker`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Cursor::advance` -/
def Cursor_advance (self : Cursor) («by» : UInt256) : FeM (UInt256 × Cursor) := do
  let mut self := self
  let «at» := self.pos
  let v5 ← checkedAdd «at» «by»
  self := { self with pos := v5 }
  return («at», self)

/-- `Cursor::skip` -/
def Cursor_skip (self : Cursor) («by» : UInt256) : FeM (Unit × Cursor) := do
  let mut self := self
  let v4 := decide ((UInt256.ofNat 32).toNat < «by».toNat)
  let join : Cursor → FeM (Unit × Cursor) := fun self => do
    let mut self := self
    let v10 := self.pos
    self := { self with base := v10 }
    return ((), self)
  if v4 then
    let (v6, v6_0) ← Cursor_advance self (UInt256.ofNat 32)
    self := v6_0
    join self
  else
    let (v8, v8_0) ← Cursor_advance self «by»
    self := v8_0
    join self

/-- `Cursor::end` -/
def Cursor_end (self : Cursor) : FeM UInt256 := do
  let v1 := self.pos
  let v2 := self.base
  let v3 ← checkedAdd v1 v2
  return v3

/-- `Sum { step } -> u256`, selector `0x188b85b4` (`sum(uint256)`) -/
def recv_Sum (step : UInt256) : FeM UInt256 := do
  let mut c : Cursor := default
  let mut s : Span := default
  let c_3 := (Cursor.mk (UInt256.ofNat 0) (UInt256.ofNat 4))
  c := c_3
  let (a, a_0) ← Cursor_advance c step
  c := a_0
  let (v9, v9_0) ← Cursor_skip c step
  c := v9_0
  let c_10 := c
  let s_12 := (Span.mk c_10 step)
  s := s_12
  let v14 := s.len
  let (b, b_0) ← Cursor_advance s.start v14
  s := { s with start := b_0 }
  let v18 ← checkedAdd a b
  let v21 ← Cursor_end s.start
  let v22 ← checkedAdd v18 v21
  return v22

/-- `Cursor::advance` -/
def Cursor_advance_1 (self : Place) («by» : UInt256) : FeM UInt256 := do
  let «at» ← load (self.field Layout.Cursor 0)
  let v5 ← checkedAdd «at» «by»
  store (self.field Layout.Cursor 0) v5
  return «at»

/-- `Bump { step } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump (step : UInt256) : FeM UInt256 := do
  let v1 := Field.saved
  let v2 ← Cursor_advance_1 v1 step
  return v2


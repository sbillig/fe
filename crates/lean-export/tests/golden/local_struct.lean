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
  -- let at = self.pos
  let «at» := self.pos
  -- self.pos = at + by
  let v5 ← checkedAdd «at» «by»
  self := { self with pos := v5 }
  return («at», self)

/-- `Cursor::skip` -/
def Cursor_skip (self : Cursor) («by» : UInt256) : FeM (Unit × Cursor) := do
  let mut self := self
  let join : Cursor → FeM (Unit × Cursor) := fun self => do
    let mut self := self
    -- self.base = self.pos
    self := { self with base := self.pos }
    return ((), self)
  -- if by > 32 {
  if decide ((UInt256.ofNat 32).toNat < «by».toNat) then
    -- self.advance(by: 32)
    let (v6, v6_0) ← Cursor_advance self (UInt256.ofNat 32)
    self := v6_0
    join self
  else
    -- self.advance(by)
    let (v8, v8_0) ← Cursor_advance self «by»
    self := v8_0
    join self

/-- `Cursor::end` -/
def Cursor_end (self : Cursor) : FeM UInt256 := do
  -- self.pos + self.base
  let v3 ← checkedAdd self.pos self.base
  return v3

/-- `Sum { step } -> u256`, selector `0x188b85b4` (`sum(uint256)`) -/
def recv_Sum (step : UInt256) : FeM UInt256 := do
  let mut c : Cursor := default
  let mut s : Span := default
  -- var c = Cursor { pos: 0, base: 4 }
  c := (Cursor.mk (UInt256.ofNat 0) (UInt256.ofNat 4))
  -- let a = c.advance(by: step)
  let (a, a_0) ← Cursor_advance c step
  c := a_0
  -- c.skip(by: step)
  let (v9, v9_0) ← Cursor_skip c step
  c := v9_0
  -- var s = Span { start: c, len: step }
  s := (Span.mk c step)
  -- let b = s.start.advance(by: s.len)
  let v14 := s.len
  let (b, b_0) ← Cursor_advance s.start v14
  s := { s with start := b_0 }
  -- a + b + s.start.end()
  let v18 ← checkedAdd a b
  let v21 ← Cursor_end s.start
  let v22 ← checkedAdd v18 v21
  return v22

/-- `Cursor::advance` -/
def Cursor_advance_1 (self : Place) («by» : UInt256) : FeM UInt256 := do
  -- let at = self.pos
  let «at» ← load (self.field Layout.Cursor 0)
  -- self.pos = at + by
  let v5 ← checkedAdd «at» «by»
  store (self.field Layout.Cursor 0) v5
  return «at»

/-- `Bump { step } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump (step : UInt256) : FeM UInt256 := do
  -- saved.advance(by: step)
  let v1 := Field.saved
  let v2 ← Cursor_advance_1 v1 step
  return v2


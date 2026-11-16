structure Cursor where
  pos : UInt256
  base : UInt256
deriving Inhabited

/-- The `init` of contract `Walker`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `param_slot::make` -/
def make (pos : UInt256) : FeM Cursor := do
  let v3 := (Cursor.mk pos (UInt256.ofNat 4))
  return v3

/-- `Cursor::end` -/
def Cursor_end (self : Cursor) : FeM UInt256 := do
  let v1 := self.pos
  let v2 := self.base
  let v3 ← checkedAdd v1 v2
  return v3

/-- `Cursor::advance` -/
def Cursor_advance (self : Cursor) («by» : UInt256) : FeM (UInt256 × Cursor) := do
  let mut self := self
  let «at» := self.pos
  let v5 ← checkedAdd «at» «by»
  self := { self with pos := v5 }
  return («at», self)

/-- `param_slot::c_end` -/
def c_end (c : Cursor) : FeM UInt256 := do
  let v1 ← Cursor_end c
  return v1

/-- `param_slot::consume` -/
def consume (c : Cursor) («by» : UInt256) : FeM UInt256 := do
  let mut c := c
  let mut r1 : Cursor := default
  let (v2, v2_0) ← Cursor_advance c «by»
  c := v2_0
  let v3 ← make «by»
  r1 := v3
  let v5 ← c_end r1
  let v6 ← checkedAdd v2 v5
  return v6

/-- `param_slot::bump` -/
def bump («by» : UInt256) : FeM UInt256 := do
  let mut «by» := «by»
  let by_1 := «by»
  let by_2 ← checkedAdd by_1 (UInt256.ofNat 1)
  «by» := by_2
  let by_5 := «by»
  return by_5

/-- `Sum { step } -> u256`, selector `0x188b85b4` (`sum(uint256)`) -/
def recv_Sum (step : UInt256) : FeM UInt256 := do
  let mut r0 : Cursor := default
  let v1 ← make step
  r0 := v1
  let v3 ← Cursor_end r0
  let v5 ← make (UInt256.ofNat 1)
  let v6 ← consume v5 step
  let v7 ← checkedAdd v3 v6
  let v8 ← bump step
  let v9 ← checkedAdd v7 v8
  return v9


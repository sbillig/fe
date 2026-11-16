structure Cursor where
  pos : UInt256
  base : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def make (pos_0 : UInt256) : FeM Cursor := do
  let pos_1 := pos_0
  let v2 := (UInt256.ofNat 4)
  let v3 := (Cursor.mk pos_1 v2)
  return v3

def Cursor_end (self : Cursor) : FeM UInt256 := do
  let v1 := self.pos
  let v2 := self.base
  let v3 ← checkedAdd v1 v2
  return v3

def Cursor_advance (self : Cursor) (by_1 : UInt256) : FeM (UInt256 × Cursor) := do
  let mut self := self
  let v2 := self.pos
  let «at» := v2
  let by_4 := by_1
  let v5 ← checkedAdd «at» by_4
  self := { self with pos := v5 }
  let v6 := ()
  return («at», self)

def c_end (c : Cursor) : FeM UInt256 := do
  let v1 ← Cursor_end c
  return v1

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

def bump («by» : UInt256) : FeM UInt256 := do
  let mut «by» := «by»
  let v0 := (UInt256.ofNat 1)
  let by_1 := «by»
  let v2 ← checkedAdd by_1 v0
  let by_3 := v2
  «by» := by_3
  let v4 := ()
  let by_5 := «by»
  return by_5

def recv_Sum (step : UInt256) : FeM UInt256 := do
  let mut r0 : Cursor := default
  let v1 ← make step
  r0 := v1
  let v3 ← Cursor_end r0
  let v4 := (UInt256.ofNat 1)
  let v5 ← make v4
  let v6 ← consume v5 step
  let v7 ← checkedAdd v3 v6
  let v8 ← bump step
  let v9 ← checkedAdd v7 v8
  return v9


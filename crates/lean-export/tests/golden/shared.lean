structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

/-- `lib::SCALE` -/
def SCALE : FeM UInt256 := do
  return UInt256.ofNat 3

/-- `lib::scaled` -/
def scaled (x : UInt256) : FeM UInt256 := do
  let v2 ← SCALE
  let v3 ← checkedMul x v2
  return v3

/-- `lib::sum` -/
def sum (pair : Pair) : FeM UInt256 := do
  let v2 ← scaled pair.a
  let v4 ← checkedAdd v2 pair.b
  return v4


structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def SCALE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 3)
  return v0

def scaled (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 ← SCALE
  let v3 ← checkedMul v1 v2
  return v3

def sum (v0 : Pair) : FeM UInt256 := do
  let v1 := v0.a
  let v2 ← scaled v1
  let v3 := v0.b
  let v4 ← checkedAdd v2 v3
  return v4


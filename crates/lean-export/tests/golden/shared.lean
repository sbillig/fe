structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def SCALE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 3)
  return v0

def scaled (x_0 : UInt256) : FeM UInt256 := do
  let x_1 := x_0
  let v2 ← SCALE
  let v3 ← checkedMul x_1 v2
  return v3

def sum (pair : Pair) : FeM UInt256 := do
  let v1 := pair.a
  let v2 ← scaled v1
  let v3 := pair.b
  let v4 ← checkedAdd v2 v3
  return v4


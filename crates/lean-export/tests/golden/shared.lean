structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def SCALE : FeM UInt256 := do
  return (UInt256.ofNat 3)

def scaled (x : UInt256) : FeM UInt256 := do
  let v2 ← SCALE
  let v3 ← checkedMul x v2
  return v3

def sum (pair : Pair) : FeM UInt256 := do
  let v1 := pair.a
  let v2 ← scaled v1
  let v3 := pair.b
  let v4 ← checkedAdd v2 v3
  return v4


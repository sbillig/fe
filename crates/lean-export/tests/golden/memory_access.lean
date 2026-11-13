export Lib (alloc_bytes)

def init : FeM Unit := do
  return ()

def cast_u8_Cell (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def alloc_Cell : FeM UInt256 := do
  let v0 := (UInt256.ofNat 33)
  let v1 ← alloc_bytes v0
  let v2 ← cast_u8_Cell v1
  return v2

def word_of (v0 : UInt256) : FeM UInt256 := do
  let v1 := (v0 + UInt256.ofNat 1)
  return v1

def add (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 := v1
  let v3 ← mload v0
  let v4 ← checkedAdd v3 v2
  mstore v0 v4
  let v5 := ()
  return ()

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← alloc_Cell
  let v2 := v1
  let v3 ← word_of v2
  let v4 := v3
  let v5 := v4
  let v6 := (UInt256.ofNat 1)
  mstore v5 v6
  let v7 := ()
  let v8 := v5
  let v9 ← add v8 v0
  let v10 ← mload v5
  return v10


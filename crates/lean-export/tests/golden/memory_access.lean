export Lib (alloc_bytes)

def init : FeM Unit := do
  return ()

def cast_u8_Cell (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let v2 := ptr_1
  return v2

def alloc_Cell : FeM UInt256 := do
  let v0 := (UInt256.ofNat 33)
  let v1 ← alloc_bytes v0
  let v2 ← cast_u8_Cell v1
  return v2

def word_of (pointer : UInt256) : FeM UInt256 := do
  let v1 := (pointer + UInt256.ofNat 1)
  return v1

def add (x : UInt256) (by_1 : UInt256) : FeM Unit := do
  let by_2 := by_1
  let v3 ← mload x
  let v4 ← checkedAdd v3 by_2
  mstore x v4
  let v5 := ()
  return ()

def recv_Bump («by» : UInt256) : FeM UInt256 := do
  let v1 ← alloc_Cell
  let pointer := v1
  let v3 ← word_of pointer
  let v4 := v3
  let word := v4
  let v6 := (UInt256.ofNat 1)
  mstore word v6
  let v7 := ()
  let v8 := word
  let v9 ← add v8 «by»
  let v10 ← mload word
  return v10


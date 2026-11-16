export Lib (alloc_bytes)

def init : FeM Unit := do
  return ()

def cast_u8_Cell (ptr : UInt256) : FeM UInt256 := do
  let v2 := ptr
  return v2

def alloc_Cell : FeM UInt256 := do
  let v1 ← alloc_bytes (UInt256.ofNat 33)
  let v2 ← cast_u8_Cell v1
  return v2

def word_of (pointer : UInt256) : FeM UInt256 := do
  let v1 := (pointer + UInt256.ofNat 1)
  return v1

def add (x : UInt256) («by» : UInt256) : FeM Unit := do
  let v3 ← mload x
  let v4 ← checkedAdd v3 «by»
  mstore x v4
  return ()

def recv_Bump («by» : UInt256) : FeM UInt256 := do
  let pointer ← alloc_Cell
  let v3 ← word_of pointer
  let v4 := v3
  let word := v4
  mstore word (UInt256.ofNat 1)
  let v8 := word
  let _ ← add v8 «by»
  let v10 ← mload word
  return v10


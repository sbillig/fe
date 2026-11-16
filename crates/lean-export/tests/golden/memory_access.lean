export Lib (alloc_bytes)

/-- The `init` of contract `Memory`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `core::ptr::cast<u8, Cell>` -/
def cast_u8_Cell (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `core::ptr::alloc<Cell>` -/
def alloc_Cell : FeM UInt256 := do
  -- cast<u8, T>(alloc_bytes(size_of<T>()))
  let v1 ← alloc_bytes (UInt256.ofNat 33)
  let v2 ← cast_u8_Cell v1
  return v2

/-- `memory_access::word_of` -/
def word_of (pointer : UInt256) : FeM UInt256 := do
  -- mut (*pointer).word
  let v1 := (pointer + UInt256.ofNat 1)
  return v1

/-- `memory_access::add` -/
def add (x : UInt256) («by» : UInt256) : FeM Unit := do
  -- x += by
  let v3 ← mload x
  let v4 ← checkedAdd v3 «by»
  mstore x v4
  return ()

/-- `Bump { by } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump («by» : UInt256) : FeM UInt256 := do
  -- let pointer = ptr::alloc<Cell>()
  let pointer ← alloc_Cell
  -- let word = mut word_of(pointer)
  let v3 ← word_of pointer
  let v4 := v3
  let word := v4
  -- word = 1
  mstore word (UInt256.ofNat 1)
  -- add(mut word, by)
  let v8 := word
  let _ ← add v8 «by»
  -- word
  let v10 ← mload word
  return v10


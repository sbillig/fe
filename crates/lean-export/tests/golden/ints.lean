export Lib (Address_word Evm_address Evm_block_number Evm_origin Evm_value Wei Wei_as_wei downcast_truncate_i256_i8 downcast_unchecked_u256_i256)

/-- The `init` of contract `Ints`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Narrow { a, b } -> u8`, selector `0x741ef9d5` (`narrow(uint8,uint8)`) -/
def recv_Narrow (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let sum ← IntTy.add (.u 8) a b
  let v5 := sum >>> (UInt256.ofNat 1)
  let low := v5 &&& (UInt256.ofNat 15)
  let v9 ← IntTy.sub (.u 8) sum low
  let v11 ← IntTy.mul (.u 8) v9 (UInt256.ofNat 2)
  let v12 ← checkedDiv v11 b
  let v14 ← checkedRem v12 (UInt256.ofNat 7)
  return v14

/-- `ints::wrapping` -/
def wrapping (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let v4 := IntTy.mulW (.i 32) a b
  let v6 := IntTy.shl (.i 32) v4 (UInt256.ofNat 3)
  return v6

/-- `Signed { a, b } -> i32`, selector `0x6effafad` (`signed(int32,int32)`) -/
def recv_Signed (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let v2 := UInt256.sltBool a b
  if v2 then
    let v3 ← IntTy.neg (.i 32) a
    return v3
  else
    let v6 ← wrapping a b
    let v7 ← IntTy.div (.i 32) a b
    let v8 ← IntTy.add (.i 32) v6 v7
    let v9 ← IntTy.rem (.i 32) a b
    let v10 ← IntTy.sub (.i 32) v8 v9
    let v12 ← IntTy.add (.i 32) v10 (IntTy.wrap (.i 256) (-5))
    return v12

/-- `Mixed { a, b } -> i256`, selector `0x728ac8e2` (`mixed(int256,uint16)`) -/
def recv_Mixed (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let wide := b
  let small ← downcast_truncate_i256_i8 a
  let bits ← downcast_unchecked_u256_i256 wide
  let v9 := IntTy.shr (.i 256) a (UInt256.ofNat 2)
  let v11 ← IntTy.pow (.i 256) v9 (UInt256.ofNat 2)
  let v13 ← IntTy.add (.i 256) v11 small
  let v14 ← IntTy.add (.i 256) v13 bits
  let v15 := IntTy.not (.i 256) a
  let v16 ← IntTy.add (.i 256) v14 v15
  return v16

/-- `Context -> u256`, selector `0xd0496d6a` (`context()`) -/
def recv_Context : FeM UInt256 := do
  let mut r1 : UInt256 := default
  let mut r2 : Wei := default
  let mut r3 : UInt256 := default
  let v1 ← Evm_address
  r1 := v1
  let v2 := r1
  let v3 ← Address_word v2
  let v5 ← Evm_value
  r2 := v5
  let v6 := r2
  let v7 ← Wei_as_wei v6
  let v8 := v3 ^^^ v7
  let v10 ← Evm_origin
  r3 := v10
  let v11 := r3
  let v12 ← Address_word v11
  let v13 := v8 ^^^ v12
  let v15 ← Evm_block_number
  let v16 := v13 ^^^ v15
  return v16


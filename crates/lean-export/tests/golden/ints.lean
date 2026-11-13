export Lib (Address_word Evm_address Evm_block_number Evm_origin Evm_value Wei Wei_as_wei downcast_truncate_i256_i8 downcast_unchecked_u256_i256)

def init : FeM Unit := do
  return ()

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← IntTy.add (.u 8) v0 v1
  let v3 := v2
  let v4 := (UInt256.ofNat 1)
  let v5 := v3 >>> v4
  let v6 := (UInt256.ofNat 15)
  let v7 := v5 &&& v6
  let v8 := v7
  let v9 ← IntTy.sub (.u 8) v3 v8
  let v10 := (UInt256.ofNat 2)
  let v11 ← IntTy.mul (.u 8) v9 v10
  let v12 ← checkedDiv v11 v1
  let v13 := (UInt256.ofNat 7)
  let v14 ← checkedRem v12 v13
  return v14

def wrapping (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := v1
  let v4 := IntTy.mulW (.i 32) v2 v3
  let v5 := (UInt256.ofNat 3)
  let v6 := IntTy.shl (.i 32) v4 v5
  return v6

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := UInt256.sltBool v0 v1
  if v2 then
    let v3 ← IntTy.neg (.i 32) v0
    return v3
  else
    let v4 := ()
    let v5 := v4
    let v6 ← wrapping v0 v1
    let v7 ← IntTy.div (.i 32) v0 v1
    let v8 ← IntTy.add (.i 32) v6 v7
    let v9 ← IntTy.rem (.i 32) v0 v1
    let v10 ← IntTy.sub (.i 32) v8 v9
    let v11 := (IntTy.wrap (.i 256) (-5))
    let v12 ← IntTy.add (.i 32) v10 v11
    return v12

def recv0_arm2 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 := v2
  let v4 ← downcast_truncate_i256_i8 v0
  let v5 := v4
  let v6 ← downcast_unchecked_u256_i256 v3
  let v7 := v6
  let v8 := (UInt256.ofNat 2)
  let v9 := IntTy.shr (.i 256) v0 v8
  let v10 := (UInt256.ofNat 2)
  let v11 ← IntTy.pow (.i 256) v9 v10
  let v12 := v5
  let v13 ← IntTy.add (.i 256) v11 v12
  let v14 ← IntTy.add (.i 256) v13 v7
  let v15 := IntTy.not (.i 256) v0
  let v16 ← IntTy.add (.i 256) v14 v15
  return v16

def recv0_arm3 : FeM UInt256 := do
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


structure Wei where
  amount : UInt256
deriving Inhabited

def to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def IS_SIGNED : FeM Bool := do
  let v0 := true
  return v0

def U8_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 255)
  return v0

def MASK : FeM UInt256 := do
  let v0 ← U8_MASK
  return v0

def I8_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 128)
  return v0

def SIGN_BIT : FeM UInt256 := do
  let v0 ← I8_SIGN_BIT
  return v0

def U256_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)
  return v0

def sign_extend (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v4 := v0
  let v5 := v1
  let v6 := v4 &&& v5
  let v7 := v6
  let v8 := v2
  let v9 := v7 &&& v8
  let v10 := (UInt256.ofNat 0)
  let v11 := decide (v9 ≠ v10)
  let b3 : UInt256 → FeM UInt256 := fun v3 => do
    return v3
  if v11 then
    let v12 ← U256_MASK
    let v13 := v1
    let v14 := v12 ^^^ v13
    let v15 := v7 ||| v14
    let v16 := v15
    b3 v16
  else
    let v17 := v7
    b3 v17

def from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 8) v0
  return v1

def downcast_truncate_i256_i8 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← to_word v0
  let v3 := v2
  let v4 ← IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← MASK
    let v6 ← SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← MASK
    let v11 := v3 &&& v10
    let v12 ← from_word v11
    let v13 := v12
    b3 v13

def from_word_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def downcast_unchecked_u256_i256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← to_word v0
  let v2 := v1
  let v3 ← from_word_1 v2
  return v3

def address : FeM UInt256 := do
  let v1 ← selfAddress
  let v2 := v1
  return v2

def word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def wei (v0 : UInt256) : FeM Wei := do
  let v1 := v0
  let v2 := (Wei.mk v1)
  return v2

def value : FeM Wei := do
  let v1 ← callvalue
  let v2 ← wei v1
  return v2

def as_wei (v0 : Wei) : FeM UInt256 := do
  let v1 := v0.amount
  return v1

def origin_1 : FeM UInt256 := do
  let v1 ← origin
  let v2 := v1
  return v2

def block_number : FeM UInt256 := do
  let v1 ← blockNumber
  return v1

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
  let v1 ← address
  r1 := v1
  let v2 := r1
  let v3 ← word v2
  let v5 ← value
  r2 := v5
  let v6 := r2
  let v7 ← as_wei v6
  let v8 := v3 ^^^ v7
  let v10 ← origin_1
  r3 := v10
  let v11 := r3
  let v12 ← word v11
  let v13 := v8 ^^^ v12
  let v15 ← block_number
  let v16 := v13 ^^^ v15
  return v16


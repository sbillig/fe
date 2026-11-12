structure CallData where
  base : UInt256
deriving Inhabited

def new : FeM CallData := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (CallData.mk v0)
  return v1

def input : FeM CallData := do
  let v1 ← new
  return v1

def to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def U64_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

def SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
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
  let v1 := truncate 64 v0
  return v1

def downcast_truncate_u256_u64 (v0 : UInt256) : FeM UInt256 := do
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

def size (v0 : CallData) : FeM UInt256 := do
  let v2 ← calldatasize
  let v3 := v2
  let v4 := v0.base
  let v5 := decide (v3.toNat < v4.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7
  else
    let v8 := v0.base
    let v9 ← checkedSub v3 v8
    let v10 ← downcast_truncate_u256_u64 v9
    let v11 := v10
    b3 v11

def len_CallData (v0 : CallData) : FeM UInt256 := do
  let v1 ← size v0
  let v2 := v1
  return v2

def IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def abi_field_head_size_u256 : FeM UInt256 := do
  let v0 ← IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← HEAD_SIZE
    return v4

def HEAD_SIZE_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def decode_error : FeM Empty := do
  let v0 ← revertEmpty
  nomatch v0

def checked_frame_end_Sol (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 := v0
  let v4 := v2
  let v5 := decide (v4.toNat < v3.toNat)
  let b1 : FeM UInt256 := do
    let v16 ← decode_error
    nomatch v16
  if v5 then
    b1
  else
    let v6 := v1
    let v7 := v2
    let v8 := v0
    let v9 := v7 - v8
    let v10 := decide (v9.toNat < v6.toNat)
    if v10 then
      b1
    else
      let v11 := ()
      let v12 := v11
      let v13 := v0
      let v14 := v1
      let v15 := v13 + v14
      return v15

def word_at (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0.base
  let v3 := v1
  let v4 := v2 + v3
  let v5 ← calldataload v4
  return v5

def checked_tail_Sol (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  let v5 := ~~~v4
  let v6 := v0
  let v7 := v5 - v6
  let v8 := decide (v7.toNat < v2.toNat)
  if v8 then
    let v9 ← decode_error
    nomatch v9
  else
    let v10 := ()
    let v11 := v10
    let v12 := v0
    let v13 := v1
    let v14 := v12 + v13
    return v14

def decode_from_Sol_CallData (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← word_at v0 v1
  return v2

def decode_from_bounded_u256_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← decode_from_Sol_CallData v0 v1
  return v5

def decode_field_from_prechecked_head_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← IS_DYNAMIC
  if v4 then
    let v5 ← word_at v0 v2
    let v6 := v5
    let v7 ← checked_tail_Sol v1 v6
    let v8 := v7
    let v9 ← decode_from_bounded_u256_Sol_CallData v0 v8 v3
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← decode_from_Sol_CallData v0 v2
    return v12

def decode_from_prechecked_head_Sol_u256_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut r0 : UInt256 := default
  let v3 := v1
  let v4 := v3
  r0 := v4
  let v5 := r0
  let v6 := v5
  let v7 ← decode_field_from_prechecked_head_Sol_u256_CallData v0 v1 v6 v2
  let v8 := v7
  let v9 ← abi_field_head_size_u256
  let v10 := r0
  let v11 ← checkedAdd v10 v9
  let v12 := v11
  r0 := v12
  let v13 := ()
  let v14 := r0
  let v15 := v14
  let v16 ← decode_field_from_prechecked_head_Sol_u256_CallData v0 v1 v15 v2
  let v17 := v16
  let v18 := (v8, v17)
  return v18

def decode_frame_from_Sol_u256_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← HEAD_SIZE_u256_u256
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← decode_from_prechecked_head_Sol_u256_u256_CallData v0 v1 v2
  return v5

def decode_input_at_u256_u256_CallData (v0 : CallData) (v1 : UInt256) : FeM (UInt256 × UInt256) := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_frame_from_Sol_u256_u256_CallData v0 v1 v3
  return v4

def decode_args_u256_u256 : FeM (UInt256 × UInt256) := do
  let mut r0 : CallData := default
  let v1 ← input
  r0 := v1
  let v2 := (UInt256.ofNat 4)
  let v3 := v2
  let v4 := r0
  let v5 ← decode_input_at_u256_u256_CallData v4 v3
  return v5

def init : FeM Unit := do
  return ()

def recv0_arm0 : FeM UInt256 := do
  let v1 ← decode_args_u256_u256
  let v2 := v1
  let v3 := v2.fst
  let v4 := v2.snd
  let v5 ← checkedAdd v3 v4
  return v5


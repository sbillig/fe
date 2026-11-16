export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)
export Shared (Pair Pair.mk sum)

structure Right where
  x : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def right (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : Pair := default
  let v1 := (UInt256.ofNat 1)
  let v2 := v0
  let v3 := (Pair.mk v1 v2)
  r0 := v3
  let v5 ← sum r0
  let v6 := (UInt256.ofNat 2)
  let v7 ← checkedMul v5 v6
  return v7

def recv_Right (v0 : UInt256) : FeM UInt256 := do
  let v1 ← right v0
  return v1

def Right_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Right_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Right := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Right.mk v5)
  return v6

def decode_from_prechecked_head_Right_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Right := do
  let v3 ← Right_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Right : FeM Right := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Right_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Right_Sol_CallData v11 v5 v8
  return v12

def recv_Right_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Right
  let result ← recv_Right message.x
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0xad5f32c7 then
      recv_Right_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


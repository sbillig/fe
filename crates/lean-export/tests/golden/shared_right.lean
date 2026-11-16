export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)
export Shared (Pair Pair.mk sum)

structure Right where
  x : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def right (x_0 : UInt256) : FeM UInt256 := do
  let mut r0 : Pair := default
  let v1 := (UInt256.ofNat 1)
  let x_2 := x_0
  let v3 := (Pair.mk v1 x_2)
  r0 := v3
  let v5 ← sum r0
  let v6 := (UInt256.ofNat 2)
  let v7 ← checkedMul v5 v6
  return v7

def recv_Right (x : UInt256) : FeM UInt256 := do
  let v1 ← right x
  return v1

def Right_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Right_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Right := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Right.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Right_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Right := do
  let v3 ← Right_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Right : FeM Right := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Right_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Right_Sol_CallData input_11 base len
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


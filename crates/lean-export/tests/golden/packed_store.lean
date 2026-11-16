export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head StoragePackedArray_get_8 StoragePackedArray_set_8 abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)

def Field.pack : Place :=
  .storage (UInt256.ofNat 0)

structure Get0 where
deriving Inhabited

structure Get1 where
deriving Inhabited

structure Set0 where
  value : UInt256
deriving Inhabited

structure Set1 where
  value : UInt256
deriving Inhabited

structure SetBoth where
  first : UInt256
  second : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def recv_Get0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← StoragePackedArray_get_8 v2 v0
  return v3

def recv_Get1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← StoragePackedArray_get_8 v2 v0
  return v3

def recv_Set0 (value : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 0)
  let v2 := Field.pack
  let v3 ← StoragePackedArray_set_8 v2 v1 value
  return ()

def recv_Set1 (value : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 1)
  let v2 := Field.pack
  let v3 ← StoragePackedArray_set_8 v2 v1 value
  return ()

def recv_SetBoth (first : UInt256) (second : UInt256) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := Field.pack
  let v4 ← StoragePackedArray_set_8 v3 v2 first
  let v5 := (UInt256.ofNat 1)
  let v6 := Field.pack
  let v7 ← StoragePackedArray_set_8 v6 v5 second
  return ()

def Get0_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Get0_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Get0 := do
  let v2 := (Get0.mk)
  return v2

def decode_from_prechecked_head_Get0_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get0 := do
  let v3 ← Get0_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Get0 : FeM Get0 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Get0_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Get0_Sol_CallData input_11 base len
  return v12

def recv_Get0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get0
  let result ← recv_Get0
  Evm_return_value_Sol_u256 result

def Get1_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Get1_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Get1 := do
  let v2 := (Get1.mk)
  return v2

def decode_from_prechecked_head_Get1_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get1 := do
  let v3 ← Get1_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Get1 : FeM Get1 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Get1_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Get1_Sol_CallData input_11 base len
  return v12

def recv_Get1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get1
  let result ← recv_Get1
  Evm_return_value_Sol_u256 result

def Set0_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Set0_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Set0 := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Set0.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Set0_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Set0 := do
  let v3 ← Set0_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Set0 : FeM Set0 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Set0_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Set0_Sol_CallData input_11 base len
  return v12

def recv_Set0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set0
  let _ ← recv_Set0 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def Set1_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Set1_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Set1 := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Set1.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Set1_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Set1 := do
  let v3 ← Set1_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Set1 : FeM Set1 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Set1_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Set1_Sol_CallData input_11 base len
  return v12

def recv_Set1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set1
  let _ ← recv_Set1 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def SetBoth_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def SetBoth_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM SetBoth := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (SetBoth.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_SetBoth_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM SetBoth := do
  let v3 ← SetBoth_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_SetBoth : FeM SetBoth := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← SetBoth_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_SetBoth_Sol_CallData input_11 base len
  return v12

def recv_SetBoth_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_SetBoth
  let _ ← recv_SetBoth message.first message.second
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x054c1a75 then
      recv_Get1_entry
    else if selector = UInt256.ofNat 0x90bb7b44 then
      recv_Get0_entry
    else if selector = UInt256.ofNat 0x92cf1ef5 then
      recv_SetBoth_entry
    else if selector = UInt256.ofNat 0x9bc39d3c then
      recv_Set1_entry
    else if selector = UInt256.ofNat 0xf26feec9 then
      recv_Set0_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


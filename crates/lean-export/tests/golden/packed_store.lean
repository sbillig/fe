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

def recv0_arm0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← StoragePackedArray_get_8 v2 v0
  return v3

def recv0_arm1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← StoragePackedArray_get_8 v2 v0
  return v3

def recv0_arm2 (v0 : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 0)
  let v2 := Field.pack
  let v3 ← StoragePackedArray_set_8 v2 v1 v0
  return ()

def recv0_arm3 (v0 : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 1)
  let v2 := Field.pack
  let v3 ← StoragePackedArray_set_8 v2 v1 v0
  return ()

def recv0_arm4 (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := Field.pack
  let v4 ← StoragePackedArray_set_8 v3 v2 v0
  let v5 := (UInt256.ofNat 1)
  let v6 := Field.pack
  let v7 ← StoragePackedArray_set_8 v6 v5 v1
  return ()

def Get0_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Get0_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get0 := do
  let v2 := (Get0.mk)
  return v2

def decode_from_prechecked_head_Get0_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Get0 := do
  let v3 ← Get0_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Get0 : FeM Get0 := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Get0_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Get0_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get0
  let result ← recv0_arm0
  Evm_return_value_Sol_u256 result

def Get1_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Get1_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get1 := do
  let v2 := (Get1.mk)
  return v2

def decode_from_prechecked_head_Get1_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Get1 := do
  let v3 ← Get1_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Get1 : FeM Get1 := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Get1_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Get1_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get1
  let result ← recv0_arm1
  Evm_return_value_Sol_u256 result

def Set0_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Set0_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Set0 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Set0.mk v5)
  return v6

def decode_from_prechecked_head_Set0_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Set0 := do
  let v3 ← Set0_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Set0 : FeM Set0 := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Set0_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Set0_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set0
  let _ ← recv0_arm2 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def Set1_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Set1_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Set1 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Set1.mk v5)
  return v6

def decode_from_prechecked_head_Set1_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Set1 := do
  let v3 ← Set1_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Set1 : FeM Set1 := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Set1_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Set1_Sol_CallData v11 v5 v8
  return v12

def recv0_arm3_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set1
  let _ ← recv0_arm3 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def SetBoth_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def SetBoth_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM SetBoth := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (SetBoth.mk v5 v10)
  return v11

def decode_from_prechecked_head_SetBoth_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM SetBoth := do
  let v3 ← SetBoth_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_SetBoth : FeM SetBoth := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← SetBoth_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_SetBoth_Sol_CallData v11 v5 v8
  return v12

def recv0_arm4_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_SetBoth
  let _ ← recv0_arm4 message.first message.second
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x054c1a75 then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0x90bb7b44 then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0x92cf1ef5 then
      recv0_arm4_entry
    else if selector = UInt256.ofNat 0x9bc39d3c then
      recv0_arm3_entry
    else if selector = UInt256.ofNat 0xf26feec9 then
      recv0_arm2_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


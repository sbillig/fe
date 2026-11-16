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

/-- The `init` of contract `FePackedStore`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Get0 -> u256`, selector `0x90bb7b44` (`get0()`) -/
def recv_Get0 : FeM UInt256 := do
  let v1 := Field.pack
  let v3 ← StoragePackedArray_get_8 v1 (UInt256.ofNat 0)
  return v3

/-- `Get1 -> u256`, selector `0x054c1a75` (`get1()`) -/
def recv_Get1 : FeM UInt256 := do
  let v1 := Field.pack
  let v3 ← StoragePackedArray_get_8 v1 (UInt256.ofNat 1)
  return v3

/-- `Set0 { value }`, selector `0xf26feec9` (`set0(uint256)`) -/
def recv_Set0 (value : UInt256) : FeM Unit := do
  let v2 := Field.pack
  let _ ← StoragePackedArray_set_8 v2 (UInt256.ofNat 0) value
  return ()

/-- `Set1 { value }`, selector `0x9bc39d3c` (`set1(uint256)`) -/
def recv_Set1 (value : UInt256) : FeM Unit := do
  let v2 := Field.pack
  let _ ← StoragePackedArray_set_8 v2 (UInt256.ofNat 1) value
  return ()

/-- `SetBoth { first, second }`, selector `0x92cf1ef5` (`setBoth(uint256,uint256)`) -/
def recv_SetBoth (first : UInt256) (second : UInt256) : FeM Unit := do
  let v3 := Field.pack
  let _ ← StoragePackedArray_set_8 v3 (UInt256.ofNat 0) first
  let v6 := Field.pack
  let _ ← StoragePackedArray_set_8 v6 (UInt256.ofNat 1) second
  return ()

/-- `<Get0 as AbiSize>::HEAD_SIZE` -/
def Get0_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Get0 as Decode>::decode_from<CallData>` -/
def Get0_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get0 := do
  return Get0.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Get0, Sol, CallData>` -/
def decode_from_prechecked_head_Get0_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get0 := do
  let v3 ← Get0_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Get0>` -/
def decode_runtime_args_Evm_Sol_Get0 : FeM Get0 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Get0_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Get0_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Get0`, for `packed_store::FePackedStoreMsg::Get0` -/
def recv_Get0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get0
  let result ← recv_Get0
  Evm_return_value_Sol_u256 result

/-- `<Get1 as AbiSize>::HEAD_SIZE` -/
def Get1_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Get1 as Decode>::decode_from<CallData>` -/
def Get1_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get1 := do
  return Get1.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Get1, Sol, CallData>` -/
def decode_from_prechecked_head_Get1_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get1 := do
  let v3 ← Get1_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Get1>` -/
def decode_runtime_args_Evm_Sol_Get1 : FeM Get1 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Get1_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Get1_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Get1`, for `packed_store::FePackedStoreMsg::Get1` -/
def recv_Get1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get1
  let result ← recv_Get1
  Evm_return_value_Sol_u256 result

/-- `<Set0 as AbiSize>::HEAD_SIZE` -/
def Set0_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Set0 as Decode>::decode_from<CallData>` -/
def Set0_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Set0 := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Set0.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Set0, Sol, CallData>` -/
def decode_from_prechecked_head_Set0_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Set0 := do
  let v3 ← Set0_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Set0>` -/
def decode_runtime_args_Evm_Sol_Set0 : FeM Set0 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Set0_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Set0_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Set0`, for `packed_store::FePackedStoreMsg::Set0` -/
def recv_Set0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set0
  let _ ← recv_Set0 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Set1 as AbiSize>::HEAD_SIZE` -/
def Set1_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Set1 as Decode>::decode_from<CallData>` -/
def Set1_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Set1 := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Set1.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Set1, Sol, CallData>` -/
def decode_from_prechecked_head_Set1_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Set1 := do
  let v3 ← Set1_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Set1>` -/
def decode_runtime_args_Evm_Sol_Set1 : FeM Set1 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Set1_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Set1_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Set1`, for `packed_store::FePackedStoreMsg::Set1` -/
def recv_Set1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set1
  let _ ← recv_Set1 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<SetBoth as AbiSize>::HEAD_SIZE` -/
def SetBoth_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<SetBoth as Decode>::decode_from<CallData>` -/
def SetBoth_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM SetBoth := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v1 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v2
  return SetBoth.mk v4 v9

/-- `core::abi::Decode::decode_from_prechecked_head<SetBoth, Sol, CallData>` -/
def decode_from_prechecked_head_SetBoth_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM SetBoth := do
  let v3 ← SetBoth_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, SetBoth>` -/
def decode_runtime_args_Evm_Sol_SetBoth : FeM SetBoth := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← SetBoth_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_SetBoth_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_SetBoth`, for `packed_store::FePackedStoreMsg::SetBoth` -/
def recv_SetBoth_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_SetBoth
  let _ ← recv_SetBoth message.first message.second
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `FePackedStore`: the entry of the arm the call's selector picks -/
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

/-- The deployment entry of contract `FePackedStore`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


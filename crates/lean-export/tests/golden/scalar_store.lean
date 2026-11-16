export Lib (CallData Evm_input Evm_return_value_Sol_u256 MemSlice_u8.mk Sol_SELECTOR_SIZE Sol_decoder_new Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8)

def Field.stored : Place :=
  .storage (UInt256.ofNat 0)

structure Get where
deriving Inhabited

structure Set where
  value : UInt256
deriving Inhabited

def init (initial : UInt256) : FeM Unit := do
  store Field.stored initial
  return ()

def recv_Get : FeM UInt256 := do
  let stored ← load Field.stored
  return stored

def recv_Set (value : UInt256) : FeM Unit := do
  store Field.stored value
  return ()

def Get_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 0)

def Get_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Get := do
  let v2 := (Get.mk)
  return v2

def decode_from_prechecked_head_Get_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get := do
  let v3 ← Get_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Get : FeM Get := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Get_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Get_Sol_CallData input_11 base len
  return v12

def recv_Get_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get
  let result ← recv_Get
  Evm_return_value_Sol_u256 result

def Set_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

def Set_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Set := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v6 := (Set.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Set_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Set := do
  let v3 ← Set_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Set : FeM Set := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Set_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Set_Sol_CallData input_11 base len
  return v12

def recv_Set_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set
  let _ ← recv_Set message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x60fe47b1 then
      recv_Set_entry
    else if selector = UInt256.ofNat 0x6d4ce63c then
      recv_Get_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← Sol_decoder_new
  let (decoded, _) ← tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let _ ← init decoded
  return ByteArray.empty


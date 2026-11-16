export Lib (CallData Evm_input Evm_return_value_Sol_u256 MemSlice_u8.mk Sol_SELECTOR_SIZE Sol_decoder_new Sol_validate_runtime_head len_CallData tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8)

structure Get where
deriving Inhabited

def init (initial_0 : UInt256) (e0 : UInt256) : FeM (Unit × UInt256) := do
  let mut e0 := e0
  let initial_1 := initial_0
  e0 := initial_1
  let v2 := ()
  return ((), e0)

def recv_Get : FeM UInt256 := do
  let e0 ← cload (← immutableAt 1 0)
  let value := e0
  return value

def Get_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Get_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Get := do
  let v2 := (Get.mk)
  return v2

def decode_from_prechecked_head_Get_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get := do
  let v3 ← Get_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Get : FeM Get := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Get_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Get_Sol_CallData input_11 base len
  return v12

def recv_Get_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get
  let result ← recv_Get
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x6d4ce63c then
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
  let words ← alloc (UInt256.ofNat 32)
  let (_, e0) ← init decoded default
  mstore (words + UInt256.ofNat (0 * 32)) e0
  memoryBytes words (UInt256.ofNat 32)


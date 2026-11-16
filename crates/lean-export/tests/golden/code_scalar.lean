export Lib (CallData Evm_input Evm_return_value_Sol_u256 MemSlice_u8.mk Sol_SELECTOR_SIZE Sol_decoder_new Sol_validate_runtime_head len_CallData tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8)

structure Get where
deriving Inhabited

/-- `init(initial: u256)` of contract `FeCodeScalar` -/
def init (initial : UInt256) (value : UInt256) : FeM (Unit × UInt256) := do
  let mut value := value
  -- value = initial
  value := initial
  return ((), value)

/-- `Get -> u256`, selector `0x6d4ce63c` (`get()`) -/
def recv_Get : FeM UInt256 := do
  let value ← cload (← immutableAt 1 0)
  return value

/-- `<Get as AbiSize>::HEAD_SIZE` -/
def Get_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Get as Decode>::decode_from<CallData>` -/
def Get_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get := do
  return Get.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Get, Sol, CallData>` -/
def decode_from_prechecked_head_Get_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Get := do
  -- Self::decode_from(input, pos)
  let v3 ← Get_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Get>` -/
def decode_runtime_args_Evm_Sol_Get : FeM Get := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Get_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Get_Sol_CallData input base len
  return v12

/-- The entry of `recv_Get`, for `code_scalar::FeCodeScalarMsg::Get` -/
def recv_Get_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get
  let result ← recv_Get
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `FeCodeScalar`: the entry of the arm the call's selector picks -/
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

/-- The deployment entry of contract `FeCodeScalar`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← Sol_decoder_new
  let (decoded, _) ← tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let words ← alloc (UInt256.ofNat 32)
  let (_, value) ← init decoded default
  mstore (words + UInt256.ofNat (0 * 32)) value
  memoryBytes words (UInt256.ofNat 32)


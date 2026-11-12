export Lib (CallData MemSlice_u8.mk SELECTOR_SIZE decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder_new input len_CallData return_value_Sol_u256 size tail validate_runtime_head)

structure Get where
deriving Inhabited

def init (v0 : UInt256) (e0 : UInt256) : FeM (Unit × UInt256) := do
  let mut e0 := e0
  let v1 := v0
  e0 := v1
  let v2 := ()
  return ((), e0)

def recv0_arm0 : FeM UInt256 := do
  let e0 ← cload (← immutableAt 1 0)
  let v0 := e0
  return v0

def HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get := do
  let v2 := (Get.mk)
  return v2

def decode_from_prechecked_head_Get_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Get := do
  let v3 ← decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Get : FeM Get := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Get_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get
  let result ← recv0_arm0
  return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x6d4ce63c then
      recv0_arm0_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← decoder_new
  let (decoded, _) ← decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let words ← alloc (UInt256.ofNat 32)
  let (_, e0) ← init decoded default
  mstore (words + UInt256.ofNat (0 * 32)) e0
  memoryBytes words (UInt256.ofNat 32)


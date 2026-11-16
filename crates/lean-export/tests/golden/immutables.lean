export Lib (CallData Evm_input Evm_return_value_Sol_u256 MemSlice_u8.mk Sol_SELECTOR_SIZE Sol_decoder_new Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8)

structure Config where
  small : UInt256
  on : Bool
  limit : UInt256
deriving Inhabited

def Field.count : Place :=
  .storage (UInt256.ofNat 0)

def cload_Config (p : UInt256) : FeM Config := do
  let f0 ← cloadInt (.u 8) (p + UInt256.ofNat 0)
  let f1 ← cloadBool (p + UInt256.ofNat 1)
  let f2 ← cload (p + UInt256.ofNat 2)
  return (Config.mk f0 f1 f2)

structure Limit where
deriving Inhabited

structure Sum where
deriving Inhabited

structure Bump where
  «by» : UInt256
deriving Inhabited

def mstore_Config (p : UInt256) (v : Config) : FeM Unit := do
  mstoreInt (.u 8) (p + UInt256.ofNat 0) v.small
  mstoreBool (p + UInt256.ofNat 1) v.on
  mstore (p + UInt256.ofNat 2) v.limit

/-- `immutables::set_small<Config>` -/
def set_small_Config (v : UInt256) (config : Config) : FeM (Unit × Config) := do
  let mut config := config
  config := { config with small := v }
  return ((), config)

/-- `init(limit: u256)` of contract `Immutables` -/
def init (limit : UInt256) (config : Config) (words : (Vector UInt256 2)) : FeM (Unit × Config × (Vector UInt256 2)) := do
  let mut config := config
  let mut words := words
  config := (Config.mk (UInt256.ofNat 0) true limit)
  let (v7, v7_0) ← set_small_Config (UInt256.ofNat 7) config
  config := v7_0
  words := (#v[limit, (UInt256.ofNat 2)])
  store Field.count (UInt256.ofNat 1)
  return ((), config, words)

/-- `Limit -> u256`, selector `0xa4d66daf` (`limit()`) -/
def recv_Limit : FeM UInt256 := do
  let config ← cload_Config (← immutableAt 4 0)
  let join : UInt256 → FeM UInt256 := fun v0 => do
    return v0
  if config.on then
    let v5 ← checkedAdd config.limit config.small
    join v5
  else
    join (UInt256.ofNat 0)

/-- `Sum -> u256`, selector `0x853255cc` (`sum()`) -/
def recv_Sum : FeM UInt256 := do
  let words ← cloadArray 32 (fun q => cload q) (← immutableAt 4 2) 2
  let v0 := (UInt256.ofNat 0)
  checkIndex v0 2
  let v2 := (UInt256.ofNat 1)
  checkIndex v2 2
  let v4 ← checkedAdd words[v0.toNat]! words[v2.toNat]!
  return v4

/-- `immutables::capped<Config>` -/
def capped_Config («by» : UInt256) : FeM UInt256 := do
  let config ← cload_Config (← immutableAt 4 0)
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if decide (config.limit.toNat < «by».toNat) then
    join config.limit
  else
    join «by»

/-- `Bump { by } -> u256`, selector `0xb20eb4c4` (`bump(uint256)`) -/
def recv_Bump («by» : UInt256) : FeM UInt256 := do
  let config ← cload_Config (← immutableAt 4 0)
  let v1 ← capped_Config «by»
  let count_2 ← load Field.count
  let v3 ← checkedAdd count_2 v1
  store Field.count v3
  let count_5 ← load Field.count
  return count_5

/-- `<Limit as AbiSize>::HEAD_SIZE` -/
def Limit_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Limit as Decode>::decode_from<CallData>` -/
def Limit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Limit := do
  return Limit.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Limit, Sol, CallData>` -/
def decode_from_prechecked_head_Limit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Limit := do
  let v3 ← Limit_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Limit>` -/
def decode_runtime_args_Evm_Sol_Limit : FeM Limit := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let len ← len_CallData input
  let v9 ← Limit_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let v12 ← decode_from_prechecked_head_Limit_Sol_CallData input base len
  return v12

/-- The entry of `recv_Limit`, for `immutables::ImmutablesMsg::Limit` -/
def recv_Limit_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Limit
  let result ← recv_Limit
  Evm_return_value_Sol_u256 result

/-- `<Sum as AbiSize>::HEAD_SIZE` -/
def Sum_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Sum as Decode>::decode_from<CallData>` -/
def Sum_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Sum := do
  return Sum.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Sum, Sol, CallData>` -/
def decode_from_prechecked_head_Sum_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Sum := do
  let v3 ← Sum_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Sum>` -/
def decode_runtime_args_Evm_Sol_Sum : FeM Sum := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let len ← len_CallData input
  let v9 ← Sum_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let v12 ← decode_from_prechecked_head_Sum_Sol_CallData input base len
  return v12

/-- The entry of `recv_Sum`, for `immutables::ImmutablesMsg::Sum` -/
def recv_Sum_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Sum
  let result ← recv_Sum
  Evm_return_value_Sol_u256 result

/-- `<Bump as AbiSize>::HEAD_SIZE` -/
def Bump_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Bump as Decode>::decode_from<CallData>` -/
def Bump_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Bump := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Bump.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Bump, Sol, CallData>` -/
def decode_from_prechecked_head_Bump_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Bump := do
  let v3 ← Bump_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Bump>` -/
def decode_runtime_args_Evm_Sol_Bump : FeM Bump := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let len ← len_CallData input
  let v9 ← Bump_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let v12 ← decode_from_prechecked_head_Bump_Sol_CallData input base len
  return v12

/-- The entry of `recv_Bump`, for `immutables::ImmutablesMsg::Bump` -/
def recv_Bump_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Bump
  let result ← recv_Bump message.«by»
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Immutables`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x853255cc then
      recv_Sum_entry
    else if selector = UInt256.ofNat 0xa4d66daf then
      recv_Limit_entry
    else if selector = UInt256.ofNat 0xb20eb4c4 then
      recv_Bump_entry
    else
      revertEmpty

/-- The deployment entry of contract `Immutables`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← Sol_decoder_new
  let (decoded, _) ← tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let words ← alloc (UInt256.ofNat 128)
  let (_, config, words_1) ← init decoded default default
  mstore_Config (words + UInt256.ofNat (0 * 32)) config
  mstoreArray 32 (fun q x => mstore q x) (words + UInt256.ofNat (2 * 32)) words_1
  memoryBytes words (UInt256.ofNat 128)


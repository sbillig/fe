export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData revert_u256)

structure Fail where
  value : UInt256
deriving Inhabited

structure One where
deriving Inhabited

/-- The `init` of contract `Diverge`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Fail { value } -> u256`, selector `0x132e4f3c` (`fail(uint256)`) -/
def recv_Fail (value : UInt256) : FeM Empty := do
  -- revert(value)
  let v1 ← revert_u256 value
  nomatch v1

/-- `One -> u256`, selector `0x901717d1` (`one()`) -/
def recv_One : FeM UInt256 := do
  return UInt256.ofNat 1

/-- `<Fail as AbiSize>::HEAD_SIZE` -/
def Fail_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Fail as Decode>::decode_from<CallData>` -/
def Fail_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Fail := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Fail.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Fail, Sol, CallData>` -/
def decode_from_prechecked_head_Fail_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Fail := do
  -- Self::decode_from(input, pos)
  let v3 ← Fail_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Fail>` -/
def decode_runtime_args_Evm_Sol_Fail : FeM Fail := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Fail_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Fail_Sol_CallData input base len
  return v12

/-- The entry of `recv_Fail`, for `diverging_arm::DivergeMsg::Fail` -/
def recv_Fail_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Fail
  let result ← recv_Fail message.value
  nomatch result

/-- `<One as AbiSize>::HEAD_SIZE` -/
def One_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<One as Decode>::decode_from<CallData>` -/
def One_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM One := do
  return One.mk

/-- `core::abi::Decode::decode_from_prechecked_head<One, Sol, CallData>` -/
def decode_from_prechecked_head_One_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM One := do
  -- Self::decode_from(input, pos)
  let v3 ← One_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, One>` -/
def decode_runtime_args_Evm_Sol_One : FeM One := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← One_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_One_Sol_CallData input base len
  return v12

/-- The entry of `recv_One`, for `diverging_arm::DivergeMsg::One` -/
def recv_One_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_One
  let result ← recv_One
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Diverge`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x132e4f3c then
      recv_Fail_entry
    else if selector = UInt256.ofNat 0x901717d1 then
      recv_One_entry
    else
      revertEmpty

/-- The deployment entry of contract `Diverge`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_bool abi_field_head_size_u256 decode_msg_field_from_Sol_bool_CallData decode_msg_field_from_Sol_u256_CallData len_CallData revert_u256)

structure Pick where
  c : Bool
  v : UInt256
deriving Inhabited

/-- The `init` of contract `NeverSlot`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `never_slot::pick` -/
def pick (c : Bool) (v : UInt256) : FeM UInt256 := do
  -- if c {
  if c then
    -- var x = revert(v)
    let v4 ← revert_u256 v
    nomatch v4
  else
    return v

/-- `Pick { c, v } -> u256`, selector `0x3998e1d2` (`pick(bool,uint256)`) -/
def recv_Pick (c : Bool) (v : UInt256) : FeM UInt256 := do
  -- pick(c, v)
  let v2 ← pick c v
  return v2

/-- `<Pick as AbiSize>::HEAD_SIZE` -/
def Pick_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_bool
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<Pick as Decode>::decode_from<CallData>` -/
def Pick_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Pick := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_bool_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_bool
  let v8 ← checkedAdd v1 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v2
  return Pick.mk v4 v9

/-- `core::abi::Decode::decode_from_prechecked_head<Pick, Sol, CallData>` -/
def decode_from_prechecked_head_Pick_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Pick := do
  -- Self::decode_from(input, pos)
  let v3 ← Pick_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Pick>` -/
def decode_runtime_args_Evm_Sol_Pick : FeM Pick := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Pick_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Pick_Sol_CallData input base len
  return v12

/-- The entry of `recv_Pick`, for `never_slot::SlotMsg::Pick` -/
def recv_Pick_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Pick
  let result ← recv_Pick message.c message.v
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `NeverSlot`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x3998e1d2 then
      recv_Pick_entry
    else
      revertEmpty

/-- The deployment entry of contract `NeverSlot`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


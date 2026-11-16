export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)
export Shared (Pair Pair.mk sum)

structure Right where
  x : UInt256
deriving Inhabited

/-- The `init` of contract `Right`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::right` -/
def right (x : UInt256) : FeM UInt256 := do
  let mut r0 : Pair := default
  r0 := (Pair.mk (UInt256.ofNat 1) x)
  let v5 ← sum r0
  let v7 ← checkedMul v5 (UInt256.ofNat 2)
  return v7

/-- `Right { x } -> u256`, selector `0xad5f32c7` (`right(uint256)`) -/
def recv_Right (x : UInt256) : FeM UInt256 := do
  let v1 ← right x
  return v1

/-- `<Right as AbiSize>::HEAD_SIZE` -/
def Right_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Right as Decode>::decode_from<CallData>` -/
def Right_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Right := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Right.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Right, Sol, CallData>` -/
def decode_from_prechecked_head_Right_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Right := do
  let v3 ← Right_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Right>` -/
def decode_runtime_args_Evm_Sol_Right : FeM Right := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Right_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Right_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Right`, for `lib::RightMsg::Right` -/
def recv_Right_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Right
  let result ← recv_Right message.x
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Right`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0xad5f32c7 then
      recv_Right_entry
    else
      revertEmpty

/-- The deployment entry of contract `Right`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


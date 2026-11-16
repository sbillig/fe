export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)
export Shared (Pair Pair.mk sum)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0] }

/-- `store.total` -/
abbrev Field.store.total : Place :=
  Field.store.field Layout.Store 0

structure Left where
  x : UInt256
deriving Inhabited

/-- The `init` of contract `Left`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::left` -/
def left (x : UInt256) : FeM UInt256 := do
  let mut r0 : Pair := default
  -- sum(Pair { a: x, b: 1 })
  r0 := (Pair.mk x (UInt256.ofNat 1))
  let v5 ← sum r0
  return v5

/-- `lib::bump` -/
def bump («by» : UInt256) : FeM UInt256 := do
  -- store.total = store.total + left(by)
  let v1 ← load Field.store.total
  let v2 ← left «by»
  let v3 ← checkedAdd v1 v2
  store Field.store.total v3
  -- store.total
  let v5 ← load Field.store.total
  return v5

/-- `Left { x } -> u256`, selector `0x1bb95617` (`left(uint256)`) -/
def recv_Left (x : UInt256) : FeM UInt256 := do
  -- bump(x)
  let v1 ← bump x
  return v1

/-- `<Left as AbiSize>::HEAD_SIZE` -/
def Left_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Left as Decode>::decode_from<CallData>` -/
def Left_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Left := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Left.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Left, Sol, CallData>` -/
def decode_from_prechecked_head_Left_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Left := do
  -- Self::decode_from(input, pos)
  let v3 ← Left_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Left>` -/
def decode_runtime_args_Evm_Sol_Left : FeM Left := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Left_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Left_Sol_CallData input base len
  return v12

/-- The entry of `recv_Left`, for `lib::LeftMsg::Left` -/
def recv_Left_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Left
  let result ← recv_Left message.x
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Left`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x1bb95617 then
      recv_Left_entry
    else
      revertEmpty

/-- The deployment entry of contract `Left`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


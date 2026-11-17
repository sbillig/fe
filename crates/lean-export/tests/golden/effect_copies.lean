export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)

def Field.count : Place :=
  .storage (UInt256.ofNat 0)

def Field.pair : Place :=
  .storage (UInt256.ofNat 1)

def Layout.Pair : Layout :=
  { storage := [0, 1] }

/-- `pair.b` -/
abbrev Field.pair.b : Place :=
  Field.pair.field Layout.Pair 1

structure Get where
deriving Inhabited

structure Local where
  v : UInt256
deriving Inhabited

structure Both where
  k : UInt256
deriving Inhabited

/-- The `init` of contract `Copies`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `effect_copies::read` -/
def read (x : UInt256) : FeM UInt256 := do
  return x

/-- `effect_copies::read_twice` -/
def read_twice (x : UInt256) : FeM UInt256 := do
  let mut x := x
  -- read() + x
  let v0 ← read x
  let v2 ← checkedAdd v0 x
  return v2

/-- `Get -> u256`, selector `0x6d4ce63c` (`get()`) -/
def recv_Get : FeM UInt256 := do
  -- read_twice()
  let v0_e0 ← load Field.count
  let v0 ← read_twice v0_e0
  return v0

/-- `effect_copies::read_through_ref` -/
def read_through_ref (x : UInt256) : FeM UInt256 := do
  let mut x := x
  -- r + 1
  let v2 := x
  let v4 ← checkedAdd v2 (UInt256.ofNat 1)
  return v4

/-- `Local { v } -> u256`, selector `0x183d1e0c` (`local(uint256)`) -/
def recv_Local (v : UInt256) : FeM UInt256 := do
  let mut v := v
  -- read_through_ref()
  let v0 ← read_through_ref v
  return v0

/-- `effect_copies::scaled` -/
def scaled (k : UInt256) (x : UInt256) : FeM UInt256 := do
  let mut x := x
  -- k * x + read()
  let v3 ← checkedMul k x
  let v4 ← read x
  let v5 ← checkedAdd v3 v4
  return v5

/-- `Both { k } -> u256`, selector `0x75de2d6b` (`both(uint256)`) -/
def recv_Both (k : UInt256) : FeM UInt256 := do
  -- scaled(k)
  let v1_e0 ← load Field.pair.b
  let v1 ← scaled k v1_e0
  return v1

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

/-- The entry of `recv_Get`, for `effect_copies::CopiesMsg::Get` -/
def recv_Get_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get
  let result ← recv_Get
  Evm_return_value_Sol_u256 result

/-- `<Local as AbiSize>::HEAD_SIZE` -/
def Local_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Local as Decode>::decode_from<CallData>` -/
def Local_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Local := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Local.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Local, Sol, CallData>` -/
def decode_from_prechecked_head_Local_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Local := do
  -- Self::decode_from(input, pos)
  let v3 ← Local_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Local>` -/
def decode_runtime_args_Evm_Sol_Local : FeM Local := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Local_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Local_Sol_CallData input base len
  return v12

/-- The entry of `recv_Local`, for `effect_copies::CopiesMsg::Local` -/
def recv_Local_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Local
  let result ← recv_Local message.v
  Evm_return_value_Sol_u256 result

/-- `<Both as AbiSize>::HEAD_SIZE` -/
def Both_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Both as Decode>::decode_from<CallData>` -/
def Both_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Both := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Both.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Both, Sol, CallData>` -/
def decode_from_prechecked_head_Both_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Both := do
  -- Self::decode_from(input, pos)
  let v3 ← Both_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Both>` -/
def decode_runtime_args_Evm_Sol_Both : FeM Both := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Both_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Both_Sol_CallData input base len
  return v12

/-- The entry of `recv_Both`, for `effect_copies::CopiesMsg::Both` -/
def recv_Both_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Both
  let result ← recv_Both message.k
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Copies`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x183d1e0c then
      recv_Local_entry
    else if selector = UInt256.ofNat 0x6d4ce63c then
      recv_Get_entry
    else if selector = UInt256.ofNat 0x75de2d6b then
      recv_Both_entry
    else
      revertEmpty

/-- The deployment entry of contract `Copies`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


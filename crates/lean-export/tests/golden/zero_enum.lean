export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head len_CallData)

inductive Slot where
  | Payload (f0 : UInt256)
  | Vacant

instance : Inhabited Slot := ⟨Slot.Payload default⟩

def Slot.tag : Slot → UInt256
  | .Payload .. => UInt256.ofNat 0
  | .Vacant => UInt256.ofNat 1

def Slot.Payload_0 : Slot → UInt256
  | .Payload f0 => f0
  | _ => default

def Slot.set_Payload_0 (x : Slot) (v : UInt256) : Slot :=
  match x with
  | .Payload f0 => .Payload v
  | x => x

structure Cleared where
deriving Inhabited

/-- The `init` of contract `ZeroEnum`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `zero_enum::payload` -/
def payload (slot : Slot) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- match slot {
  match slot with
  | .Payload .. =>
    let v5 ← checkedAdd slot.Payload_0 (UInt256.ofNat 1)
    join v5
  | .Vacant =>
    join (UInt256.ofNat 0)

/-- `Cleared -> u256`, selector `0xe8588c2f` (`cleared()`) -/
def recv_Cleared : FeM UInt256 := do
  let mut slot : Slot := default
  -- var slot = Slot::Vacant
  slot := Slot.Vacant
  -- clear(mut slot)
  slot := default
  let v3 := ()
  -- payload(slot)
  let v6 ← payload slot
  return v6

/-- `<Cleared as AbiSize>::HEAD_SIZE` -/
def Cleared_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Cleared as Decode>::decode_from<CallData>` -/
def Cleared_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Cleared := do
  return Cleared.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Cleared, Sol, CallData>` -/
def decode_from_prechecked_head_Cleared_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Cleared := do
  -- Self::decode_from(input, pos)
  let v3 ← Cleared_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Cleared>` -/
def decode_runtime_args_Evm_Sol_Cleared : FeM Cleared := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Cleared_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Cleared_Sol_CallData input base len
  return v12

/-- The entry of `recv_Cleared`, for `zero_enum::ZeroMsg::Cleared` -/
def recv_Cleared_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Cleared
  let result ← recv_Cleared
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `ZeroEnum`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0xe8588c2f then
      recv_Cleared_entry
    else
      revertEmpty

/-- The deployment entry of contract `ZeroEnum`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


export Lib (CallData Evm_input Sol_SELECTOR_SIZE Sol_validate_runtime_head alloc_u256 len_CallData null_)

structure Read where
deriving Inhabited

structure Borrow where
deriving Inhabited

/-- The `init` of contract `UninhabitedAccess`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `core::ptr::cast<u256, Mixed>` -/
def cast_u256_Mixed (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `uninhabited_access::read_mixed_field` -/
def read_mixed_field : FeM Empty := do
  -- let buf = ptr::alloc<u256>()
  let buf ← alloc_u256
  -- *buf = 42
  mstore buf (UInt256.ofNat 42)
  -- let p: *Mixed = ptr::cast<u256, Mixed>(buf)
  let p ← cast_u256_Mixed buf
  M.panic 0x01

/-- `Read -> u256`, selector `0x57de26a4` (`read()`) -/
def recv_Read : FeM Empty := do
  -- read_mixed_field()
  let v0 ← read_mixed_field
  let v0 : UInt256 := nomatch v0
  M.panic 0x01

/-- `uninhabited_access::borrow_never` -/
def borrow_never : FeM Empty := do
  -- let p: *! = ptr::null<!>()
  let p ← null_
  M.panic 0x01

/-- `Borrow -> u256`, selector `0xe68d3569` (`borrow()`) -/
def recv_Borrow : FeM Empty := do
  -- borrow_never()
  let v0 ← borrow_never
  let v0 : UInt256 := nomatch v0
  M.panic 0x01

/-- `<Read as AbiSize>::HEAD_SIZE` -/
def Read_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Read as Decode>::decode_from<CallData>` -/
def Read_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Read := do
  return Read.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Read, Sol, CallData>` -/
def decode_from_prechecked_head_Read_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Read := do
  -- Self::decode_from(input, pos)
  let v3 ← Read_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Read>` -/
def decode_runtime_args_Evm_Sol_Read : FeM Read := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Read_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Read_Sol_CallData input base len
  return v12

/-- The entry of `recv_Read`, for `uninhabited_access::AccessMsg::Read` -/
def recv_Read_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Read
  let result ← recv_Read
  nomatch result

/-- `<Borrow as AbiSize>::HEAD_SIZE` -/
def Borrow_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Borrow as Decode>::decode_from<CallData>` -/
def Borrow_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Borrow := do
  return Borrow.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Borrow, Sol, CallData>` -/
def decode_from_prechecked_head_Borrow_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Borrow := do
  -- Self::decode_from(input, pos)
  let v3 ← Borrow_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Borrow>` -/
def decode_runtime_args_Evm_Sol_Borrow : FeM Borrow := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Borrow_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Borrow_Sol_CallData input base len
  return v12

/-- The entry of `recv_Borrow`, for `uninhabited_access::AccessMsg::Borrow` -/
def recv_Borrow_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Borrow
  let result ← recv_Borrow
  nomatch result

/-- The runtime entry of contract `UninhabitedAccess`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x57de26a4 then
      recv_Read_entry
    else if selector = UInt256.ofNat 0xe68d3569 then
      recv_Borrow_entry
    else
      revertEmpty

/-- The deployment entry of contract `UninhabitedAccess`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


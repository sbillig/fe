export Lib (AbiFrame AbiFrame_from_raw_parts CallData CallOutcome CallOutcome_decode_u256 Evm_input Evm_return_value_Sol_Address Evm_return_value_Sol_u256 Location Location.mk MemBuffer MemBuffer_drop MemBuffer_empty MemBuffer_len MemBuffer_new_uninit MemBuffer_ptr MemBuffer_set_len MemBuffer_span_ref RawCallOutcome Sol_SELECTOR_SIZE Sol_validate_runtime_head Wei Wei_zero abi_field_head_size_Address abi_field_head_size_u256 bubble_last_revert copy copy_call_outcome create2_raw_Evm decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData downcast_unchecked_u256_usize encode_abi_payload_tuple keccak_words_2 len_CallData raw_call_Evm read_transient_word_Evm storage_at_mut_Evm_u256 tuple_HEAD_SIZE tuple_IS_DYNAMIC zero_bytes)

structure Get where
deriving Inhabited

def Field.words : Place :=
  .storage (UInt256.ofNat 0)

structure Deploy where
deriving Inhabited

structure Ask where
  target : UInt256
deriving Inhabited

structure Poke where
  key : UInt256
deriving Inhabited

/-- The `init` of contract `Trust`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `std::evm::effects::contract_init_code<Child>` -/
def contract_init_code_Child (args : Unit) : FeM MemBuffer := do
  let mut init_code_r0 : MemBuffer := default
  let mut encoded : MemBuffer := default
  let mut init_code_r2 : MemBuffer := default
  -- let init_len = C::init_code_len()
  let init_len ← initCodeLen creation_Child
  -- let init_off = C::init_code_offset()
  let init_off ← initCodeOffset creation_Child
  -- if !C::InitArgs::IS_DYNAMIC && C::InitArgs::HEAD_SIZE == 0 {
  let v6 ← tuple_IS_DYNAMIC
  let join : MemBuffer → MemBuffer → MemBuffer → MemBuffer → FeM MemBuffer := fun init_code_r0 encoded init_code_r2 v1 => do
    let mut init_code_r0 := init_code_r0
    let mut encoded := encoded
    let mut init_code_r2 := init_code_r2
    return v1
  let join_1 : MemBuffer → MemBuffer → MemBuffer → FeM MemBuffer := fun init_code_r0 encoded init_code_r2 => do
    let mut init_code_r0 := init_code_r0
    let mut encoded := encoded
    let mut init_code_r2 := init_code_r2
    -- let encoded = encode_abi_payload<C::InitArgs>(args)
    let encoded_20 ← encode_abi_payload_tuple args
    encoded := encoded_20
    -- let code_len: usize = init_len.downcast_unchecked()
    let code_len ← downcast_unchecked_u256_usize init_len
    -- MemBuffer::new_uninit(code_len + encoded.len())
    let v25 ← MemBuffer_len encoded
    let v26 ← checkedAdd code_len v25
    let init_code_27 ← MemBuffer_new_uninit v26
    -- let init_code = unsafe { MemBuffer::new_uninit(code_len + encoded.len()) }
    init_code_r2 := init_code_27
    -- ops::codecopy(dest: init_code.ptr(), offset: init_off, len: init_len)
    let v30 ← MemBuffer_ptr init_code_r2
    let _ ← codecopy v30 init_off init_len
    -- ptr::copy(
    let v33 ← MemBuffer_ptr init_code_r2
    let v36 ← MemBuffer_span_ref encoded
    let _ ← copy (v33 + code_len) v36
    let _ ← MemBuffer_drop encoded
    -- if !C::InitArgs::IS_DYNAMIC && C::InitArgs::HEAD_SIZE == 0 {
    let init_code_39 := init_code_r2
    join init_code_r0 encoded init_code_r2 init_code_39
  if !v6 then
    let v8 ← tuple_HEAD_SIZE
    if decide (v8 = (UInt256.ofNat 0)) then
      -- MemBuffer::new_uninit(init_len.downcast_unchecked())
      let v11 ← downcast_unchecked_u256_usize init_len
      let init_code_12 ← MemBuffer_new_uninit v11
      -- let init_code = unsafe { MemBuffer::new_uninit(init_len.downcast_unchecked()) }
      init_code_r0 := init_code_12
      -- ops::codecopy(dest: init_code.ptr(), offset: init_off, len: init_len)
      let v15 ← MemBuffer_ptr init_code_r0
      let _ ← codecopy v15 init_off init_len
      -- if !C::InitArgs::IS_DYNAMIC && C::InitArgs::HEAD_SIZE == 0 {
      let init_code_17 := init_code_r0
      join init_code_r0 encoded init_code_r2 init_code_17
    else
      join_1 init_code_r0 encoded init_code_r2
  else
    join_1 init_code_r0 encoded init_code_r2

/-- `std::evm::effects::Create::create2<Evm, Child>` -/
def create2_Evm_Child (value : Wei) (args : Unit) (salt : UInt256) : FeM UInt256 := do
  let mut init_code : MemBuffer := default
  -- let init_code = contract_init_code<C>(args)
  let init_code_4 ← contract_init_code_Child args
  init_code := init_code_4
  -- let addr = self.create2_raw(value: value, init_code: init_code.span(), salt)
  let v7 ← MemBuffer_span_ref init_code
  let addr ← create2_raw_Evm value v7 salt
  -- if addr.inner == 0 {
  if decide (addr = (UInt256.ofNat 0)) then
    -- bubble_last_revert()
    let v14 ← bubble_last_revert
    nomatch v14
  else
    let _ ← MemBuffer_drop init_code
    return addr

/-- `Deploy -> Address`, selector `0x00000010` -/
def recv_Deploy : FeM UInt256 := do
  let mut r1 : Wei := default
  -- create.create2<Child>(value: Wei::zero(), args: (), salt: 0)
  let v0 ← Wei_zero
  r1 := v0
  let v5 ← create2_Evm_Child r1 () (UInt256.ofNat 0)
  return v5

/-- `<Get as MsgVariant>::SELECTOR` -/
def Get_SELECTOR : FeM UInt256 := do
  return UInt256.ofNat 1

/-- `<Get as AbiSize>::HEAD_SIZE` -/
def Get_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Get as AbiSize>::payload_size` -/
def Get_payload_size (self : Get) : FeM UInt256 := do
  let v1 ← Get_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<Get>` -/
def abi_payload_size_Get (value : Get) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← Get_payload_size value
  return v1

/-- `<Get as Encode>::encode` -/
def Get_encode (self : Get) (v1 : AbiFrame) : FeM Unit := do
  return ()

/-- `std::evm::effects::encode_calldata<Get>` -/
def encode_calldata_Get (selector : UInt256) (message : Get) : FeM MemBuffer := do
  let mut message := message
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let selector_size = size_of<Sol::Selector>()
  let selector_size := (UInt256.ofNat 4)
  -- let payload_size = abi_payload_size(message)
  let payload_size ← abi_payload_size_Get message
  -- let len = selector_size + payload_size.downcast_unchecked()
  let v7 ← downcast_unchecked_u256_usize payload_size
  let len := selector_size + v7
  -- let capacity = if M::HEAD_SIZE < 28 && len < 32 { 32 } else { len }
  let v10 ← Get_HEAD_SIZE
  let join : Get → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun message buffer r2 capacity => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    -- MemBuffer::new_uninit(capacity)
    let buffer_19 ← MemBuffer_new_uninit capacity
    -- var buffer = unsafe { MemBuffer::new_uninit(capacity) }
    buffer := buffer_19
    -- let payload = ptr::offset_bytes(buffer.ptr(), selector_size)
    let v22 ← MemBuffer_ptr buffer
    let payload := v22 + selector_size
    -- buffer.set_len(len)
    let (v26, v26_0) ← MemBuffer_set_len buffer len
    buffer := v26_0
    -- ops::mstore(addr: buffer.ptr(), value: (selector as u256) << 224)
    let v28 ← MemBuffer_ptr buffer
    let _ ← mstore v28 (selector <<< (UInt256.ofNat 224))
    -- ptr::zero_bytes(payload, payload_size.downcast_unchecked())
    let v34 ← downcast_unchecked_u256_usize payload_size
    let _ ← zero_bytes payload v34
    -- message.encode(AbiFrame::from_raw_parts(ptr: payload, len: payload_size))
    let v36 ← AbiFrame_from_raw_parts payload payload_size
    r2 := v36
    let _ ← Get_encode message r2
    return buffer
  let join_1 : Get → MemBuffer → AbiFrame → FeM MemBuffer := fun message buffer r2 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    join message buffer r2 len
  if decide (v10.toNat < (UInt256.ofNat 28).toNat) then
    if decide (len.toNat < (UInt256.ofNat 32).toNat) then
      join message buffer r2 (UInt256.ofNat 32)
    else
      join_1 message buffer r2
  else
    join_1 message buffer r2

/-- `std::evm::effects::Call::try_call<Evm, Get>` -/
def try_call_Evm_Get (addr : UInt256) (gas_1 : UInt256) (value : Wei) (message : Get) : FeM CallOutcome := do
  let mut addr := addr
  let mut args : MemBuffer := default
  let mut ret : MemBuffer := default
  let mut «raw» : RawCallOutcome := default
  -- let args = encode_calldata(<M as MsgVariant<Sol>>::SELECTOR, message)
  let v4 ← Get_SELECTOR
  let args_6 ← encode_calldata_Get v4 message
  args := args_6
  -- var ret = MemBuffer::empty()
  let ret_8 ← MemBuffer_empty
  ret := ret_8
  -- let raw = self.raw_call(addr, gas, value, args: args.span(), ret: mut ret)
  let v11 ← MemBuffer_span_ref args
  let addr_13 := addr
  let (raw_15, raw_15_0) ← raw_call_Evm addr_13 gas_1 value v11 ret
  ret := raw_15_0
  «raw» := raw_15
  -- copy_call_outcome(raw)
  let v18 ← copy_call_outcome «raw»
  let _ ← MemBuffer_drop ret
  let _ ← MemBuffer_drop args
  return v18

/-- `std::evm::effects::Call::call<Evm, Get>` -/
def call_Evm_Get (addr : UInt256) (gas_1 : UInt256) (value : Wei) (message : Get) : FeM UInt256 := do
  -- self.try_call(addr, gas, value, message).decode()
  let v6 ← try_call_Evm_Get addr gas_1 value message
  let v7 ← CallOutcome_decode_u256 v6
  return v7

/-- `Address::call<Evm, Get>` -/
def Address_call_Evm_Get (self : UInt256) (message : Get) : FeM UInt256 := do
  let mut r1 : Wei := default
  -- call.call(addr: self, gas: ops::gas(), value: Wei::zero(), message)
  let v3 ← gas
  let v4 ← Wei_zero
  r1 := v4
  let v8 ← call_Evm_Get self v3 r1 message
  return v8

/-- `Ask { target } -> u256`, selector `0x00000011` -/
def recv_Ask (target : UInt256) : FeM UInt256 := do
  let mut target := target
  -- target.call(ChildMsg::Get {})
  let v2 ← Address_call_Evm_Get target (Get.mk)
  return v2

/-- `<Words as StateIndex>::locate` -/
def Words_locate (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let mut r0 : (Vector UInt256 2) := default
  -- Location { slot: keccak_words([key, root]), offset: 0 }
  r0 := (#v[key, root])
  let v7 ← keccak_words_2 r0
  return Location.mk v7 (UInt256.ofNat 0)

/-- `trust::word_at<Evm>` -/
def word_at_Evm (slot : UInt256) : FeM Place := do
  -- storage_at(slot)
  let v1 ← storage_at_mut_Evm_u256 slot
  return v1

/-- `trust::spare<Evm>` -/
def spare_Evm (slot : UInt256) : FeM Place := do
  -- mut word_at(slot)
  let v1 ← word_at_Evm slot
  let v2 := v1
  return v2

/-- `Poke { key } -> u256`, selector `0x00000012` -/
def recv_Poke (key : UInt256) : FeM UInt256 := do
  -- words[key] = 1
  let words_key ← Words_locate Field.words Field.words.slot key
  store (Place.mk .storage words_key.slot) (UInt256.ofNat 1)
  -- let before = words[key] + read_transient_word(key)
  let words_key_1 ← Words_locate Field.words Field.words.slot key
  let v3 ← load (Place.mk .storage words_key_1.slot)
  let v4 ← read_transient_word_Evm key
  let before ← checkedAdd v3 v4
  -- let word = mut spare(key + 1)
  let v8 ← checkedAdd key (UInt256.ofNat 1)
  let v9 ← spare_Evm v8
  let v10 := v9
  let word := v10
  -- word = 2
  store word (UInt256.ofNat 2)
  -- before + word
  let v14 ← load word
  let v15 ← checkedAdd before v14
  return v15

/-- `<Deploy as AbiSize>::HEAD_SIZE` -/
def Deploy_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Deploy as Decode>::decode_from<CallData>` -/
def Deploy_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Deploy := do
  return Deploy.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Deploy, Sol, CallData>` -/
def decode_from_prechecked_head_Deploy_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Deploy := do
  -- Self::decode_from(input, pos)
  let v3 ← Deploy_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Deploy>` -/
def decode_runtime_args_Evm_Sol_Deploy : FeM Deploy := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Deploy_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Deploy_Sol_CallData input base len
  return v12

/-- The entry of `recv_Deploy`, for `trust::TrustMsg::Deploy` -/
def recv_Deploy_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Deploy
  let result ← recv_Deploy
  Evm_return_value_Sol_Address result

/-- `<Ask as AbiSize>::HEAD_SIZE` -/
def Ask_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Ask as Decode>::decode_from<CallData>` -/
def Ask_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Ask := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  return Ask.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Ask, Sol, CallData>` -/
def decode_from_prechecked_head_Ask_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Ask := do
  -- Self::decode_from(input, pos)
  let v3 ← Ask_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Ask>` -/
def decode_runtime_args_Evm_Sol_Ask : FeM Ask := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Ask_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Ask_Sol_CallData input base len
  return v12

/-- The entry of `recv_Ask`, for `trust::TrustMsg::Ask` -/
def recv_Ask_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Ask
  let result ← recv_Ask message.target
  Evm_return_value_Sol_u256 result

/-- `<Poke as AbiSize>::HEAD_SIZE` -/
def Poke_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Poke as Decode>::decode_from<CallData>` -/
def Poke_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Poke := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Poke.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Poke, Sol, CallData>` -/
def decode_from_prechecked_head_Poke_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Poke := do
  -- Self::decode_from(input, pos)
  let v3 ← Poke_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Poke>` -/
def decode_runtime_args_Evm_Sol_Poke : FeM Poke := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Poke_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Poke_Sol_CallData input base len
  return v12

/-- The entry of `recv_Poke`, for `trust::TrustMsg::Poke` -/
def recv_Poke_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Poke
  let result ← recv_Poke message.key
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Trust`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x00000010 then
      recv_Deploy_entry
    else if selector = UInt256.ofNat 0x00000011 then
      recv_Ask_entry
    else if selector = UInt256.ofNat 0x00000012 then
      recv_Poke_entry
    else
      revertEmpty

/-- The deployment entry of contract `Trust`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


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

def init : FeM Unit := do
  return ()

def contract_init_code_Child (args : Unit) : FeM MemBuffer := do
  let mut init_code_r0 : MemBuffer := default
  let mut encoded : MemBuffer := default
  let mut init_code_r2 : MemBuffer := default
  let v2 ← initCodeLen creation_Child
  let init_len := v2
  let v4 ← initCodeOffset creation_Child
  let init_off := v4
  let v6 ← tuple_IS_DYNAMIC
  let v7 := !v6
  let b3 : MemBuffer → MemBuffer → MemBuffer → MemBuffer → FeM MemBuffer := fun init_code_r0 encoded init_code_r2 v1 => do
    let mut init_code_r0 := init_code_r0
    let mut encoded := encoded
    let mut init_code_r2 := init_code_r2
    return v1
  let b2 : MemBuffer → MemBuffer → MemBuffer → FeM MemBuffer := fun init_code_r0 encoded init_code_r2 => do
    let mut init_code_r0 := init_code_r0
    let mut encoded := encoded
    let mut init_code_r2 := init_code_r2
    let v20 ← encode_abi_payload_tuple args
    let encoded_21 := v20
    encoded := encoded_21
    let v22 ← downcast_unchecked_u256_usize init_len
    let code_len := v22
    let v25 ← MemBuffer_len encoded
    let v26 ← checkedAdd code_len v25
    let v27 ← MemBuffer_new_uninit v26
    let init_code_28 := v27
    init_code_r2 := init_code_28
    let v30 ← MemBuffer_ptr init_code_r2
    let v31 ← codecopy v30 init_off init_len
    let v33 ← MemBuffer_ptr init_code_r2
    let v34 := v33 + code_len
    let v36 ← MemBuffer_span_ref encoded
    let v38 ← copy v34 v36
    let _ ← MemBuffer_drop encoded
    let init_code_39 := init_code_r2
    let init_code_40 := init_code_39
    let init_code_41 := init_code_40
    b3 init_code_r0 encoded init_code_r2 init_code_41
  if v7 then
    let v8 ← tuple_HEAD_SIZE
    let v9 := (UInt256.ofNat 0)
    let v10 := decide (v8 = v9)
    if v10 then
      let v11 ← downcast_unchecked_u256_usize init_len
      let v12 ← MemBuffer_new_uninit v11
      let init_code_13 := v12
      init_code_r0 := init_code_13
      let v15 ← MemBuffer_ptr init_code_r0
      let v16 ← codecopy v15 init_off init_len
      let init_code_17 := init_code_r0
      let init_code_18 := init_code_17
      let init_code_19 := init_code_18
      b3 init_code_r0 encoded init_code_r2 init_code_19
    else
      b2 init_code_r0 encoded init_code_r2
  else
    b2 init_code_r0 encoded init_code_r2

def create2_Evm_Child (value : Wei) (args : Unit) (salt : UInt256) : FeM UInt256 := do
  let mut init_code : MemBuffer := default
  let v4 ← contract_init_code_Child args
  let init_code_5 := v4
  init_code := init_code_5
  let v7 ← MemBuffer_span_ref init_code
  let v9 ← create2_raw_Evm value v7 salt
  let addr := v9
  let v11 := addr
  let v12 := (UInt256.ofNat 0)
  let v13 := decide (v11 = v12)
  if v13 then
    let v14 ← bubble_last_revert
    nomatch v14
  else
    let v15 := ()
    let v16 := v15
    let _ ← MemBuffer_drop init_code
    return addr

def recv_Deploy : FeM UInt256 := do
  let mut r1 : Wei := default
  let v0 ← Wei_zero
  r1 := v0
  let v1 := ()
  let v2 := (UInt256.ofNat 0)
  let v4 := r1
  let v5 ← create2_Evm_Child v4 v1 v2
  return v5

def Get_SELECTOR : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  return v0

def Get_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Get_payload_size (self : Get) : FeM UInt256 := do
  let v1 ← Get_HEAD_SIZE
  return v1

def abi_payload_size_Get (value : Get) : FeM UInt256 := do
  let v1 ← Get_payload_size value
  return v1

def Get_encode (self : Get) (__fe_out : AbiFrame) : FeM Unit := do
  let v2 := ()
  return ()

def encode_calldata_Get (selector_0 : UInt256) (message : Get) : FeM MemBuffer := do
  let mut message := message
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v2 := (UInt256.ofNat 4)
  let selector_size := v2
  let v5 ← abi_payload_size_Get message
  let payload_size := v5
  let v7 ← downcast_unchecked_u256_usize payload_size
  let v8 := selector_size + v7
  let len_9 := v8
  let v10 ← Get_HEAD_SIZE
  let v11 := (UInt256.ofNat 28)
  let v12 := decide (v10.toNat < v11.toNat)
  let b3 : Get → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun message buffer r2 v1 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    let capacity := v1
    let v19 ← MemBuffer_new_uninit capacity
    let buffer_20 := v19
    buffer := buffer_20
    let v22 ← MemBuffer_ptr buffer
    let v23 := v22 + selector_size
    let payload := v23
    let (v26, v26_0) ← MemBuffer_set_len buffer len_9
    buffer := v26_0
    let v28 ← MemBuffer_ptr buffer
    let selector_29 := selector_0
    let v30 := selector_29
    let v31 := (UInt256.ofNat 224)
    let v32 := v30 <<< v31
    let v33 ← mstore v28 v32
    let v34 ← downcast_unchecked_u256_usize payload_size
    let v35 ← zero_bytes payload v34
    let v36 ← AbiFrame_from_raw_parts payload payload_size
    r2 := v36
    let v39 ← Get_encode message r2
    let buffer_40 := buffer
    return buffer_40
  let b2 : Get → MemBuffer → AbiFrame → FeM MemBuffer := fun message buffer r2 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    let len_17 := len_9
    b3 message buffer r2 len_17
  if v12 then
    let v13 := (UInt256.ofNat 32)
    let v14 := decide (len_9.toNat < v13.toNat)
    if v14 then
      let v15 := (UInt256.ofNat 32)
      let v16 := v15
      b3 message buffer r2 v16
    else
      b2 message buffer r2
  else
    b2 message buffer r2

def try_call_Evm_Get (addr : UInt256) (gas_1 : UInt256) (value : Wei) (message_3 : Get) : FeM CallOutcome := do
  let mut addr := addr
  let mut args : MemBuffer := default
  let mut ret : MemBuffer := default
  let mut raw : RawCallOutcome := default
  let v4 ← Get_SELECTOR
  let message_5 := message_3
  let v6 ← encode_calldata_Get v4 message_5
  let args_7 := v6
  args := args_7
  let v8 ← MemBuffer_empty
  let ret_9 := v8
  ret := ret_9
  let v11 ← MemBuffer_span_ref args
  let addr_13 := addr
  let (v15, v15_0) ← raw_call_Evm addr_13 gas_1 value v11 ret
  ret := v15_0
  let raw_16 := v15
  raw := raw_16
  let v18 ← copy_call_outcome raw
  let _ ← MemBuffer_drop ret
  let _ ← MemBuffer_drop args
  return v18

def call_Evm_Get (addr : UInt256) (gas_1 : UInt256) (value : Wei) (message_4 : Get) : FeM UInt256 := do
  let message_5 := message_4
  let v6 ← try_call_Evm_Get addr gas_1 value message_5
  let v7 ← CallOutcome_decode_u256 v6
  return v7

def Address_call_Evm_Get (self : UInt256) (message_1 : Get) : FeM UInt256 := do
  let mut r1 : Wei := default
  let v2 := self
  let v3 ← gas
  let v4 ← Wei_zero
  r1 := v4
  let message_5 := message_1
  let v7 := r1
  let v8 ← call_Evm_Get v2 v3 v7 message_5
  return v8

def recv_Ask (target : UInt256) : FeM UInt256 := do
  let mut target := target
  let v0 := (Get.mk)
  let target_1 := target
  let v2 ← Address_call_Evm_Get target_1 v0
  return v2

def Words_locate (self : Place) (root_1 : UInt256) (key_2 : UInt256) : FeM Location := do
  let mut r0 : (Vector UInt256 2) := default
  let key_3 := key_2
  let root_4 := root_1
  let v5 := #v[key_3, root_4]
  r0 := v5
  let v7 ← keccak_words_2 r0
  let v8 := (UInt256.ofNat 0)
  let v9 := (Location.mk v7 v8)
  return v9

def word_at_Evm (slot : UInt256) : FeM Place := do
  let v1 ← storage_at_mut_Evm_u256 slot
  return v1

def spare_Evm (slot : UInt256) : FeM Place := do
  let v1 ← word_at_Evm slot
  let v2 := v1
  return v2

def recv_Poke (key : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  let l0 ← Words_locate Field.words Field.words.slot key
  store (Place.mk .storage l0.slot) v1
  let v2 := ()
  let l1 ← Words_locate Field.words Field.words.slot key
  let v3 ← load (Place.mk .storage l1.slot)
  let v4 ← read_transient_word_Evm key
  let v5 ← checkedAdd v3 v4
  let before := v5
  let v7 := (UInt256.ofNat 1)
  let v8 ← checkedAdd key v7
  let v9 ← spare_Evm v8
  let v10 := v9
  let word := v10
  let v12 := (UInt256.ofNat 2)
  store word v12
  let v13 := ()
  let v14 ← load word
  let v15 ← checkedAdd before v14
  return v15

def Deploy_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Deploy_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Deploy := do
  let v2 := (Deploy.mk)
  return v2

def decode_from_prechecked_head_Deploy_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Deploy := do
  let v3 ← Deploy_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Deploy : FeM Deploy := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Deploy_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Deploy_Sol_CallData input_11 base len
  return v12

def recv_Deploy_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Deploy
  let result ← recv_Deploy
  Evm_return_value_Sol_Address result

def Ask_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  return v2

def Ask_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Ask := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Ask.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Ask_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Ask := do
  let v3 ← Ask_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Ask : FeM Ask := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Ask_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Ask_Sol_CallData input_11 base len
  return v12

def recv_Ask_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Ask
  let result ← recv_Ask message.target
  Evm_return_value_Sol_u256 result

def Poke_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Poke_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Poke := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Poke.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Poke_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Poke := do
  let v3 ← Poke_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Poke : FeM Poke := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Poke_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Poke_Sol_CallData input_11 base len
  return v12

def recv_Poke_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Poke
  let result ← recv_Poke message.key
  Evm_return_value_Sol_u256 result

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

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


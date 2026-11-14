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

def contract_init_code_Child (v0 : Unit) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : MemBuffer := default
  let mut r2 : MemBuffer := default
  let v2 ← initCodeLen creation_Child
  let v3 := v2
  let v4 ← initCodeOffset creation_Child
  let v5 := v4
  let v6 ← tuple_IS_DYNAMIC
  let v7 := !v6
  let b3 : MemBuffer → MemBuffer → MemBuffer → MemBuffer → FeM MemBuffer := fun r0 r1 r2 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    return v1
  let b2 : MemBuffer → MemBuffer → MemBuffer → FeM MemBuffer := fun r0 r1 r2 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v20 ← encode_abi_payload_tuple v0
    let v21 := v20
    r1 := v21
    let v22 ← downcast_unchecked_u256_usize v3
    let v23 := v22
    let v25 ← MemBuffer_len r1
    let v26 ← checkedAdd v23 v25
    let v27 ← MemBuffer_new_uninit v26
    let v28 := v27
    r2 := v28
    let v30 ← MemBuffer_ptr r2
    let v31 ← codecopy v30 v5 v3
    let v33 ← MemBuffer_ptr r2
    let v34 := v33 + v23
    let v36 ← MemBuffer_span_ref r1
    let v38 ← copy v34 v36
    let _ ← MemBuffer_drop r1
    let v39 := r2
    let v40 := v39
    let v41 := v40
    b3 r0 r1 r2 v41
  if v7 then
    let v8 ← tuple_HEAD_SIZE
    let v9 := (UInt256.ofNat 0)
    let v10 := decide (v8 = v9)
    if v10 then
      let v11 ← downcast_unchecked_u256_usize v3
      let v12 ← MemBuffer_new_uninit v11
      let v13 := v12
      r0 := v13
      let v15 ← MemBuffer_ptr r0
      let v16 ← codecopy v15 v5 v3
      let v17 := r0
      let v18 := v17
      let v19 := v18
      b3 r0 r1 r2 v19
    else
      b2 r0 r1 r2
  else
    b2 r0 r1 r2

def create2_Evm_Child (v1 : Wei) (v2 : Unit) (v3 : UInt256) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v4 ← contract_init_code_Child v2
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_span_ref r0
  let v9 ← create2_raw_Evm v1 v7 v3
  let v10 := v9
  let v11 := v10
  let v12 := (UInt256.ofNat 0)
  let v13 := decide (v11 = v12)
  if v13 then
    let v14 ← bubble_last_revert
    nomatch v14
  else
    let v15 := ()
    let v16 := v15
    let _ ← MemBuffer_drop r0
    return v10

def recv0_arm0 : FeM UInt256 := do
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

def Get_payload_size (v0 : Get) : FeM UInt256 := do
  let v1 ← Get_HEAD_SIZE
  return v1

def abi_payload_size_Get (v0 : Get) : FeM UInt256 := do
  let v1 ← Get_payload_size v0
  return v1

def Get_encode (v0 : Get) (v1 : AbiFrame) : FeM Unit := do
  let v2 := ()
  return ()

def encode_calldata_Get (v0 : UInt256) (r0 : Get) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v2 := (UInt256.ofNat 4)
  let v3 := v2
  let v5 ← abi_payload_size_Get r0
  let v6 := v5
  let v7 ← downcast_unchecked_u256_usize v6
  let v8 := v3 + v7
  let v9 := v8
  let v10 ← Get_HEAD_SIZE
  let v11 := (UInt256.ofNat 28)
  let v12 := decide (v10.toNat < v11.toNat)
  let b3 : Get → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun r0 r1 r2 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v18 := v1
    let v19 ← MemBuffer_new_uninit v18
    let v20 := v19
    r1 := v20
    let v22 ← MemBuffer_ptr r1
    let v23 := v22 + v3
    let v24 := v23
    let (v26, v26_0) ← MemBuffer_set_len r1 v9
    r1 := v26_0
    let v28 ← MemBuffer_ptr r1
    let v29 := v0
    let v30 := v29
    let v31 := (UInt256.ofNat 224)
    let v32 := v30 <<< v31
    let v33 ← mstore v28 v32
    let v34 ← downcast_unchecked_u256_usize v6
    let v35 ← zero_bytes v24 v34
    let v36 ← AbiFrame_from_raw_parts v24 v6
    r2 := v36
    let v39 ← Get_encode r0 r2
    let v40 := r1
    return v40
  let b2 : Get → MemBuffer → AbiFrame → FeM MemBuffer := fun r0 r1 r2 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v17 := v9
    b3 r0 r1 r2 v17
  if v12 then
    let v13 := (UInt256.ofNat 32)
    let v14 := decide (v9.toNat < v13.toNat)
    if v14 then
      let v15 := (UInt256.ofNat 32)
      let v16 := v15
      b3 r0 r1 r2 v16
    else
      b2 r0 r1 r2
  else
    b2 r0 r1 r2

def try_call_Evm_Get (r0 : UInt256) (v1 : UInt256) (v2 : Wei) (v3 : Get) : FeM CallOutcome := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : MemBuffer := default
  let mut r3 : RawCallOutcome := default
  let v4 ← Get_SELECTOR
  let v5 := v3
  let v6 ← encode_calldata_Get v4 v5
  let v7 := v6
  r1 := v7
  let v8 ← MemBuffer_empty
  let v9 := v8
  r2 := v9
  let v11 ← MemBuffer_span_ref r1
  let v13 := r0
  let (v15, v15_0) ← raw_call_Evm v13 v1 v2 v11 r2
  r2 := v15_0
  let v16 := v15
  r3 := v16
  let v18 ← copy_call_outcome r3
  let _ ← MemBuffer_drop r2
  let _ ← MemBuffer_drop r1
  return v18

def call_Evm_Get (v1 : UInt256) (v2 : UInt256) (v3 : Wei) (v4 : Get) : FeM UInt256 := do
  let v5 := v4
  let v6 ← try_call_Evm_Get v1 v2 v3 v5
  let v7 ← CallOutcome_decode_u256 v6
  return v7

def Address_call_Evm_Get (v0 : UInt256) (v1 : Get) : FeM UInt256 := do
  let mut r1 : Wei := default
  let v2 := v0
  let v3 ← gas
  let v4 ← Wei_zero
  r1 := v4
  let v5 := v1
  let v7 := r1
  let v8 ← call_Evm_Get v2 v3 v7 v5
  return v8

def recv0_arm1 (r0 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  let v0 := (Get.mk)
  let v1 := r0
  let v2 ← Address_call_Evm_Get v1 v0
  return v2

def Words_locate (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Location := do
  let mut r0 : (Vector UInt256 2) := default
  let v3 := v2
  let v4 := v1
  let v5 := #v[v3, v4]
  r0 := v5
  let v7 ← keccak_words_2 r0
  let v8 := (UInt256.ofNat 0)
  let v9 := (Location.mk v7 v8)
  return v9

def word_at_Evm (v0 : UInt256) : FeM Place := do
  let v1 ← storage_at_mut_Evm_u256 v0
  return v1

def spare_Evm (v0 : UInt256) : FeM Place := do
  let v1 ← word_at_Evm v0
  let v2 := v1
  return v2

def recv0_arm2 (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  let l0 ← Words_locate Field.words Field.words.slot v0
  store (Place.mk .storage l0.slot) v1
  let v2 := ()
  let l1 ← Words_locate Field.words Field.words.slot v0
  let v3 ← load (Place.mk .storage l1.slot)
  let v4 ← read_transient_word_Evm v0
  let v5 ← checkedAdd v3 v4
  let v6 := v5
  let v7 := (UInt256.ofNat 1)
  let v8 ← checkedAdd v0 v7
  let v9 ← spare_Evm v8
  let v10 := v9
  let v11 := v10
  let v12 := (UInt256.ofNat 2)
  store v11 v12
  let v13 := ()
  let v14 ← load v11
  let v15 ← checkedAdd v6 v14
  return v15

def Deploy_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Deploy_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Deploy := do
  let v2 := (Deploy.mk)
  return v2

def decode_from_prechecked_head_Deploy_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Deploy := do
  let v3 ← Deploy_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Deploy : FeM Deploy := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Deploy_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Deploy_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Deploy
  let result ← recv0_arm0
  Evm_return_value_Sol_Address result

def Ask_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  return v2

def Ask_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Ask := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Ask.mk v5)
  return v6

def decode_from_prechecked_head_Ask_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Ask := do
  let v3 ← Ask_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Ask : FeM Ask := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Ask_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Ask_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Ask
  let result ← recv0_arm1 message.target
  Evm_return_value_Sol_u256 result

def Poke_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Poke_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Poke := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Poke.mk v5)
  return v6

def decode_from_prechecked_head_Poke_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Poke := do
  let v3 ← Poke_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Poke : FeM Poke := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Poke_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Poke_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Poke
  let result ← recv0_arm2 message.key
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x00000010 then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0x00000011 then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0x00000012 then
      recv0_arm2_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


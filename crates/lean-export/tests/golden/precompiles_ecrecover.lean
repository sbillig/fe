export Lib (CallData Evm_input Evm_return_value_Sol_tuple_bool_u256_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)
export Shared (N recover)

structure Recover where
  hash : UInt256
  recovery_id : UInt256
  r : UInt256
  s : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def recv_Recover (hash : UInt256) (recovery_id : UInt256) (r : UInt256) (s : UInt256) : FeM (Bool × UInt256 × UInt256) := do
  let v5 := (UInt256.ofNat 1)
  let v6 := decide (v5.toNat < recovery_id.toNat)
  let b1 : FeM (Bool × UInt256 × UInt256) := do
    let v30 := false
    let v31 := (UInt256.ofNat 0)
    let v32 := (UInt256.ofNat 0)
    let v33 := (v30, v31, v32)
    return v33
  if v6 then
    b1
  else
    let v7 := (UInt256.ofNat 0)
    let v8 := decide (r = v7)
    if v8 then
      b1
    else
      let v9 ← N
      let v10 := decide (v9.toNat ≤ r.toNat)
      if v10 then
        b1
      else
        let v11 := (UInt256.ofNat 0)
        let v12 := decide (s = v11)
        if v12 then
          b1
        else
          let v13 ← N
          let v14 := decide (v13.toNat ≤ s.toNat)
          if v14 then
            b1
          else
            let v15 := ()
            let v16 := v15
            let v17 ← recover hash recovery_id r s
            let b8 : (Bool × UInt256 × UInt256) → FeM (Bool × UInt256 × UInt256) := fun v4 => do
              return v4
            match v17 with
            | .Some .. =>
              let v18 := v17.Some_0
              let key := v18
              let v20 := true
              let v21 := key.x
              let v22 := key.y
              let v23 := (v20, v21, v22)
              let v24 := v23
              b8 v24
            | .None =>
              let v25 := false
              let v26 := (UInt256.ofNat 0)
              let v27 := (UInt256.ofNat 0)
              let v28 := (v25, v26, v27)
              let v29 := v28
              b8 v29

def Recover_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  return v8

def Recover_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Recover := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let __fe_msg_decode_base_11 := __fe_msg_decode_base_1
  let v12 ← abi_field_head_size_u256
  let v13 ← checkedAdd __fe_msg_decode_base_11 v12
  let v14 ← abi_field_head_size_u256
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v15 __fe_msg_decode_input_len
  let __fe_field_2 := v16
  let __fe_msg_decode_base_18 := __fe_msg_decode_base_1
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd __fe_msg_decode_base_18 v19
  let v21 ← abi_field_head_size_u256
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_u256
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v24 __fe_msg_decode_input_len
  let __fe_field_3 := v25
  let v27 := (Recover.mk __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3)
  return v27

def decode_from_prechecked_head_Recover_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Recover := do
  let v3 ← Recover_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Recover : FeM Recover := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Recover_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Recover_Sol_CallData input_11 base len
  return v12

def recv_Recover_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Recover
  let result ← recv_Recover message.hash message.recovery_id message.r message.s
  Evm_return_value_Sol_tuple_bool_u256_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x2aa6d351 then
      recv_Recover_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


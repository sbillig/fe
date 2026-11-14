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

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM (Bool × UInt256 × UInt256) := do
  let v5 := (UInt256.ofNat 1)
  let v6 := decide (v5.toNat < v1.toNat)
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
    let v8 := decide (v2 = v7)
    if v8 then
      b1
    else
      let v9 ← N
      let v10 := decide (v9.toNat ≤ v2.toNat)
      if v10 then
        b1
      else
        let v11 := (UInt256.ofNat 0)
        let v12 := decide (v3 = v11)
        if v12 then
          b1
        else
          let v13 ← N
          let v14 := decide (v13.toNat ≤ v3.toNat)
          if v14 then
            b1
          else
            let v15 := ()
            let v16 := v15
            let v17 ← recover v0 v1 v2 v3
            let b8 : (Bool × UInt256 × UInt256) → FeM (Bool × UInt256 × UInt256) := fun v4 => do
              return v4
            match v17 with
            | .Some .. =>
              let v18 := v17.Some_0
              let v19 := v18
              let v20 := true
              let v21 := v19.x
              let v22 := v19.y
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

def Recover_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Recover := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := v1
  let v12 ← abi_field_head_size_u256
  let v13 ← checkedAdd v11 v12
  let v14 ← abi_field_head_size_u256
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v3
  let v17 := v16
  let v18 := v1
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd v18 v19
  let v21 ← abi_field_head_size_u256
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_u256
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v24 v3
  let v26 := v25
  let v27 := (Recover.mk v5 v10 v17 v26)
  return v27

def decode_from_prechecked_head_Recover_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Recover := do
  let v3 ← Recover_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Recover : FeM Recover := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Recover_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Recover_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Recover
  let result ← recv0_arm0 message.hash message.recovery_id message.r message.s
  Evm_return_value_Sol_tuple_bool_u256_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x2aa6d351 then
      recv0_arm0_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


export Lib (CallData Evm_input Evm_return_value_Sol_tuple_bool_u256_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)
export Shared (N recover)

structure Recover where
  hash : UInt256
  recovery_id : UInt256
  r : UInt256
  s : UInt256
deriving Inhabited

/-- The `init` of contract `Ecrecover`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Recover { hash, recovery_id, r, s } -> (bool, u256, u256)`, selector `0x2aa6d351` (`recover(uint256,uint256,uint256,uint256)`) -/
def recv_Recover (hash : UInt256) (recovery_id : UInt256) (r : UInt256) (s : UInt256) : FeM (Bool × UInt256 × UInt256) := do
  let v6 := decide ((UInt256.ofNat 1).toNat < recovery_id.toNat)
  let join : FeM (Bool × UInt256 × UInt256) := do
    return (false, (UInt256.ofNat 0), (UInt256.ofNat 0))
  if v6 then
    join
  else
    if decide (r = (UInt256.ofNat 0)) then
      join
    else
      let v9 ← N
      if decide (v9.toNat ≤ r.toNat) then
        join
      else
        if decide (s = (UInt256.ofNat 0)) then
          join
        else
          let v13 ← N
          if decide (v13.toNat ≤ s.toNat) then
            join
          else
            let v17 ← recover hash recovery_id r s
            let join_1 : (Bool × UInt256 × UInt256) → FeM (Bool × UInt256 × UInt256) := fun v4 => do
              return v4
            match v17 with
            | .Some .. =>
              let key := v17.Some_0
              join_1 (true, key.x, key.y)
            | .None =>
              join_1 (false, (UInt256.ofNat 0), (UInt256.ofNat 0))

/-- `<Recover as AbiSize>::HEAD_SIZE` -/
def Recover_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  return v8

/-- `<Recover as Decode>::decode_from<CallData>` -/
def Recover_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Recover := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v1 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v2
  let v12 ← abi_field_head_size_u256
  let v13 ← checkedAdd v1 v12
  let v14 ← abi_field_head_size_u256
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v2
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd v1 v19
  let v21 ← abi_field_head_size_u256
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_u256
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v24 v2
  return Recover.mk v4 v9 v16 v25

/-- `core::abi::Decode::decode_from_prechecked_head<Recover, Sol, CallData>` -/
def decode_from_prechecked_head_Recover_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Recover := do
  let v3 ← Recover_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Recover>` -/
def decode_runtime_args_Evm_Sol_Recover : FeM Recover := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Recover_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Recover_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Recover`, for `lib::EcrecoverMsg::Recover` -/
def recv_Recover_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Recover
  let result ← recv_Recover message.hash message.recovery_id message.r message.s
  Evm_return_value_Sol_tuple_bool_u256_u256 result

/-- The runtime entry of contract `Ecrecover`: the entry of the arm the call's selector picks -/
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

/-- The deployment entry of contract `Ecrecover`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


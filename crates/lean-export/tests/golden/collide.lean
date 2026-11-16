export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)

def Field.count : Place :=
  .storage (UInt256.ofNat 0)

structure Run where
  x : UInt256
deriving Inhabited

structure Spin where
  n : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def double (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := (UInt256.ofNat 2)
  let v3_1 ← checkedMul v1 v2
  return v3_1

def v3 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := (UInt256.ofNat 3)
  let v3_1 ← checkedAdd v1 v2
  return v3_1

def limit : FeM UInt256 := do
  let v0 := (UInt256.ofNat 10)
  return v0

def size : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4)
  return v0

def recv_CollideMsg_Run (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 2)
  let v2 ← checkedMul v0 v1
  let v3_1 := v2
  let v4 := (UInt256.ofNat 1)
  let v5 ← checkedAdd v3_1 v4
  let v6 := v5
  let v7 ← load Field.count
  let v8 ← checkedAdd v6 v7
  let v9 := v8
  let v10 := (UInt256.ofNat 1)
  let v11 ← checkedAdd v9 v10
  let v12 := v11
  let v13 := v12
  store Field.count v13
  let v14 := ()
  let v15 ← double v13
  let v16 ← v3 v15
  let v17 ← limit
  let v18 ← checkedAdd v16 v17
  let v19 ← size
  let v20 ← checkedAdd v18 v19
  let v21 ← load Field.count
  let v22 ← checkedAdd v20 v21
  return v22

def b1 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3_1 := v1
  let v4 ← checkedAdd v2 v3_1
  return v4

def recv_Spin (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let v2 := v1
  r0 := v2
  let v3_1 := (UInt256.ofNat 0)
  let v4 := v3_1
  r1 := v4
  let rec b1_1 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let v5 := r1
    let v6 := v5
    let v7 := decide (v6.toNat < v0.toNat)
    if v7 then
      let v8 := (UInt256.ofNat 1)
      let v9 := r1
      let v10 ← checkedAdd v9 v8
      let v11 := v10
      r1 := v11
      let v12 := ()
      let v13 := r0
      let v14 := v13
      let v15 := r1
      let v16 := v15
      let v17 ← b1 v14 v16
      let v18 := v17
      r0 := v18
      let v19 := ()
      b1_1 r0 r1
    else
      let v20 := r0
      return v20
  partial_fixpoint
  b1_1 r0 r1

def recv_OtherMsg_Run (v0 : UInt256) : FeM UInt256 := do
  return v0

def Run_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Run_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Run := do
  let v2 ← len_CallData v0
  let v3_1 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3_1
  let v5 := v4
  let v6 := (Run.mk v5)
  return v6

def decode_from_prechecked_head_Run_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Run := do
  let v3_1 ← Run_decode_from_CallData v0 v1
  return v3_1

def decode_runtime_args_Evm_Sol_Run : FeM Run := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3_1 := v2
  r0 := v3_1
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Run_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Run_Sol_CallData v11 v5 v8
  return v12

def recv_CollideMsg_Run_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Run
  let result ← recv_CollideMsg_Run message.x
  Evm_return_value_Sol_u256 result

def Spin_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Spin_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Spin := do
  let v2 ← len_CallData v0
  let v3_1 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3_1
  let v5 := v4
  let v6 := (Spin.mk v5)
  return v6

def decode_from_prechecked_head_Spin_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Spin := do
  let v3_1 ← Spin_decode_from_CallData v0 v1
  return v3_1

def decode_runtime_args_Evm_Sol_Spin : FeM Spin := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3_1 := v2
  r0 := v3_1
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Spin_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Spin_Sol_CallData v11 v5 v8
  return v12

def recv_Spin_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Spin
  let result ← recv_Spin message.n
  Evm_return_value_Sol_u256 result

def recv_OtherMsg_Run_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Run
  let result ← recv_OtherMsg_Run message.x
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size_1 ← calldatasize
  if size_1.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0xa444f5e9 then
      recv_CollideMsg_Run_entry
    else if selector = UInt256.ofNat 0xa5b6ea8f then
      recv_Spin_entry
    else if selector = UInt256.ofNat 0xeb034fa7 then
      recv_OtherMsg_Run_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


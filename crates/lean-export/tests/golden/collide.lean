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

def double (x_0 : UInt256) : FeM UInt256 := do
  let x_1 := x_0
  let v2 := (UInt256.ofNat 2)
  let v3_1 ← checkedMul x_1 v2
  return v3_1

def v3 (x_0 : UInt256) : FeM UInt256 := do
  let x_1 := x_0
  let v2 := (UInt256.ofNat 3)
  let v3_1 ← checkedAdd x_1 v2
  return v3_1

def limit : FeM UInt256 := do
  let v0 := (UInt256.ofNat 10)
  return v0

def size : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4)
  return v0

def recv_CollideMsg_Run (x : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 2)
  let v2 ← checkedMul x v1
  let limit_1 := v2
  let v4 := (UInt256.ofNat 1)
  let v5 ← checkedAdd limit_1 v4
  let double_1 := v5
  let count_7 ← load Field.count
  let v8 ← checkedAdd double_1 count_7
  let load_1 := v8
  let v10 := (UInt256.ofNat 1)
  let v11 ← checkedAdd load_1 v10
  let store_12 := v11
  let store_13 := store_12
  store Field.count store_13
  let v14 := ()
  let v15 ← double store_13
  let v16 ← v3 v15
  let v17 ← limit
  let v18 ← checkedAdd v16 v17
  let v19 ← size
  let v20 ← checkedAdd v18 v19
  let count_21 ← load Field.count
  let v22 ← checkedAdd v20 count_21
  return v22

def b1 (total_0 : UInt256) (i_1 : UInt256) : FeM UInt256 := do
  let total_2 := total_0
  let i_3 := i_1
  let v4 ← checkedAdd total_2 i_3
  return v4

def recv_Spin (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let total_2 := v1
  total := total_2
  let v3_1 := (UInt256.ofNat 0)
  let i_4 := v3_1
  i := i_4
  let rec b1_1 (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    let i_5 := i
    let i_6 := i_5
    let v7 := decide (i_6.toNat < n.toNat)
    if v7 then
      let v8 := (UInt256.ofNat 1)
      let i_9 := i
      let v10 ← checkedAdd i_9 v8
      let i_11 := v10
      i := i_11
      let v12 := ()
      let total_13 := total
      let total_14 := total_13
      let i_15 := i
      let i_16 := i_15
      let v17 ← b1 total_14 i_16
      let total_18 := v17
      total := total_18
      let v19 := ()
      b1_1 total i
    else
      let total_20 := total
      return total_20
  partial_fixpoint
  b1_1 total i

def recv_OtherMsg_Run (x : UInt256) : FeM UInt256 := do
  return x

def Run_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Run_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Run := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Run.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Run_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Run := do
  let v3_1 ← Run_decode_from_CallData input pos
  return v3_1

def decode_runtime_args_Evm_Sol_Run : FeM Run := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Run_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Run_Sol_CallData input_11 base len
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

def Spin_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Spin := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Spin.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Spin_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Spin := do
  let v3_1 ← Spin_decode_from_CallData input pos
  return v3_1

def decode_runtime_args_Evm_Sol_Spin : FeM Spin := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Spin_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Spin_Sol_CallData input_11 base len
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


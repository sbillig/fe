export Lib (CallData MemBuffer MemSlice_u8.mk SELECTOR_SIZE abi_field_head_size_Address abi_field_head_size_u256 as_topic caller_1 decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder_new drop encode_abi_payload_u256 input len_CallData locate_Address_StorageMap_Address_u256 locate_Address_u256 log3 return_value_Sol_bool return_value_Sol_u256 size span_ref tail validate_runtime_head zero)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.TokenStore : Layout :=
  { storage := [0, 1, 2] }

structure Transfer where
  «from» : UInt256
  «to» : UInt256
  value : UInt256
deriving Inhabited

structure Approval where
  owner : UInt256
  spender : UInt256
  value : UInt256
deriving Inhabited

structure Transfer_1 where
  «to» : UInt256
  value : UInt256
deriving Inhabited

structure Approve where
  spender : UInt256
  value : UInt256
deriving Inhabited

structure TransferFrom where
  «from» : UInt256
  «to» : UInt256
  value : UInt256
deriving Inhabited

structure BalanceOf where
  owner : UInt256
deriving Inhabited

structure Allowance where
  owner : UInt256
  spender : UInt256
deriving Inhabited

structure TotalSupply where
deriving Inhabited

def emit_Evm (v0 : Transfer) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := v0.value
  let v3 := v2
  let v4 ← encode_abi_payload_u256 v3
  r0 := v4
  let v6 ← span_ref r0
  let v7 := (UInt256.ofNat 100389287136786176327247604509743168900146139575972864366142685224231313322991) -- a constant, as the compiler evaluates it
  let v8 := v0.«from»
  r1 := v8
  let v9 := r1
  let v10 ← as_topic v9
  let v11 := v0.«to»
  r2 := v11
  let v12 := r2
  let v13 ← as_topic v12
  let v15 ← log3 v6 v7 v10 v13
  let _ ← drop r0
  return ()

def emit_Evm_Transfer (v1 : Transfer) : FeM Unit := do
  let v3 := v1
  let v4 ← emit_Evm v3
  return ()

def init (v0 : UInt256) : FeM Unit := do
  let v2 ← caller_1
  let v3 := v2
  let v4 := v0
  let l0 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v3
  store (Place.mk .storage l0.slot) v4
  let v5 := ()
  let v6 := v0
  store (Field.store.field Layout.TokenStore 2) v6
  let v7 := ()
  let v8 ← zero
  let v9 := v0
  let v10 := (Transfer.mk v8 v3 v9)
  let v12 ← emit_Evm_Transfer v10
  return ()

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← caller_1
  let v4 := v3
  let l0 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v4
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 := decide (v1.toNat ≤ v5.toNat)
  if v6 then
    let v7 := ()
    let l1 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v4
    let v8 ← load (Place.mk .storage l1.slot)
    let v9 ← checkedSub v8 v1
    let l2 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v4
    store (Place.mk .storage l2.slot) v9
    let v10 := ()
    let l3 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v0
    let v11 ← load (Place.mk .storage l3.slot)
    let v12 ← checkedAdd v11 v1
    let l4 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v0
    store (Place.mk .storage l4.slot) v12
    let v13 := ()
    let v14 := (Transfer.mk v4 v0 v1)
    let v16 ← emit_Evm_Transfer v14
    let v17 := true
    return v17
  else
    M.revert (assertData "ERC20: insufficient balance")

def emit_Evm_1 (v0 : Approval) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := v0.value
  let v3 := v2
  let v4 ← encode_abi_payload_u256 v3
  r0 := v4
  let v6 ← span_ref r0
  let v7 := (UInt256.ofNat 63486140976153616755203102783360879283472101686154884697241723088393386309925) -- a constant, as the compiler evaluates it
  let v8 := v0.owner
  r1 := v8
  let v9 := r1
  let v10 ← as_topic v9
  let v11 := v0.spender
  r2 := v11
  let v12 := r2
  let v13 ← as_topic v12
  let v15 ← log3 v6 v7 v10 v13
  let _ ← drop r0
  return ()

def emit_Evm_Approval (v1 : Approval) : FeM Unit := do
  let v3 := v1
  let v4 ← emit_Evm_1 v3
  return ()

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← caller_1
  let v4 := v3
  let l0 ← locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot v4
  let l1 ← locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot v0
  store (Place.mk .storage l1.slot) v1
  let v5 := ()
  let v6 := (Approval.mk v4 v0 v1)
  let v8 ← emit_Evm_Approval v6
  let v9 := true
  return v9

def recv0_arm2 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Bool := do
  let v4 ← caller_1
  let v5 := v4
  let l0 ← locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot v0
  let l1 ← locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot v5
  let v6 ← load (Place.mk .storage l1.slot)
  let v7 := v6
  let v8 := decide (v2.toNat ≤ v7.toNat)
  if v8 then
    let v9 := ()
    let l2 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v0
    let v10 ← load (Place.mk .storage l2.slot)
    let v11 := decide (v2.toNat ≤ v10.toNat)
    if v11 then
      let v12 := ()
      let v13 ← checkedSub v7 v2
      let l3 ← locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot v0
      let l4 ← locate_Address_u256 (Place.mk .storage l3.slot) (Place.mk .storage l3.slot).slot v5
      store (Place.mk .storage l4.slot) v13
      let v14 := ()
      let l5 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v0
      let v15 ← load (Place.mk .storage l5.slot)
      let v16 ← checkedSub v15 v2
      let l6 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v0
      store (Place.mk .storage l6.slot) v16
      let v17 := ()
      let l7 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v1
      let v18 ← load (Place.mk .storage l7.slot)
      let v19 ← checkedAdd v18 v2
      let l8 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v1
      store (Place.mk .storage l8.slot) v19
      let v20 := ()
      let v21 := (Transfer.mk v0 v1 v2)
      let v23 ← emit_Evm_Transfer v21
      let v24 := true
      return v24
    else
      M.revert (assertData "ERC20: insufficient balance")
  else
    M.revert (assertData "ERC20: insufficient allowance")

def recv0_arm3 (v0 : UInt256) : FeM UInt256 := do
  let l0 ← locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot v0
  let v1 ← load (Place.mk .storage l0.slot)
  return v1

def recv0_arm4 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let l0 ← locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot v0
  let l1 ← locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot v1
  let v2 ← load (Place.mk .storage l1.slot)
  return v2

def recv0_arm5 : FeM UInt256 := do
  let v0 ← load (Field.store.field Layout.TokenStore 2)
  return v0

def HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Transfer_1 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Transfer_1.mk v5 v10)
  return v11

def decode_from_prechecked_head_Transfer_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Transfer_1 := do
  let v3 ← decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv0_arm0 message.«to» message.value
  return_value_Sol_bool result

def decode_from_CallData_1 (v0 : CallData) (v1 : UInt256) : FeM Approve := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Approve.mk v5 v10)
  return v11

def decode_from_prechecked_head_Approve_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Approve := do
  let v3 ← decode_from_CallData_1 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv0_arm1 message.spender message.value
  return_value_Sol_bool result

def HEAD_SIZE_2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  return v6

def decode_from_CallData_2 (v0 : CallData) (v1 : UInt256) : FeM TransferFrom := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := v1
  let v12 ← abi_field_head_size_Address
  let v13 ← checkedAdd v11 v12
  let v14 ← abi_field_head_size_Address
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v3
  let v17 := v16
  let v18 := (TransferFrom.mk v5 v10 v17)
  return v18

def decode_from_prechecked_head_TransferFrom_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM TransferFrom := do
  let v3 ← decode_from_CallData_2 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_2
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv0_arm2 message.«from» message.«to» message.value
  return_value_Sol_bool result

def HEAD_SIZE_3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  return v2

def decode_from_CallData_3 (v0 : CallData) (v1 : UInt256) : FeM BalanceOf := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (BalanceOf.mk v5)
  return v6

def decode_from_prechecked_head_BalanceOf_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM BalanceOf := do
  let v3 ← decode_from_CallData_3 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_3
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData v11 v5 v8
  return v12

def recv0_arm3_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv0_arm3 message.owner
  return_value_Sol_u256 result

def HEAD_SIZE_4 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  return v4

def decode_from_CallData_4 (v0 : CallData) (v1 : UInt256) : FeM Allowance := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Allowance.mk v5 v10)
  return v11

def decode_from_prechecked_head_Allowance_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Allowance := do
  let v3 ← decode_from_CallData_4 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_4
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData v11 v5 v8
  return v12

def recv0_arm4_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv0_arm4 message.owner message.spender
  return_value_Sol_u256 result

def HEAD_SIZE_5 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def decode_from_CallData_5 (v0 : CallData) (v1 : UInt256) : FeM TotalSupply := do
  let v2 := (TotalSupply.mk)
  return v2

def decode_from_prechecked_head_TotalSupply_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM TotalSupply := do
  let v3 ← decode_from_CallData_5 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_5
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData v11 v5 v8
  return v12

def recv0_arm5_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv0_arm5
  return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x095ea7b3 then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0x18160ddd then
      recv0_arm5_entry
    else if selector = UInt256.ofNat 0x23b872dd then
      recv0_arm2_entry
    else if selector = UInt256.ofNat 0x70a08231 then
      recv0_arm3_entry
    else if selector = UInt256.ofNat 0xa9059cbb then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0xdd62ed3e then
      recv0_arm4_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← decoder_new
  let (decoded, _) ← decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let _ ← init decoded
  return ByteArray.empty


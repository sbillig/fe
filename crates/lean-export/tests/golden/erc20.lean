export Lib (Address_as_topic Address_zero CallData Evm_caller Evm_input Evm_log3 Evm_return_value_Sol_bool Evm_return_value_Sol_u256 MemBuffer MemBuffer_drop MemBuffer_span_ref MemSlice_u8.mk Sol_SELECTOR_SIZE Sol_decoder_new Sol_validate_runtime_head StorageMap_locate_Address_StorageMap_Address_u256 StorageMap_locate_Address_u256 abi_field_head_size_Address abi_field_head_size_u256 decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData encode_abi_payload_tuple_u256 len_CallData tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8)

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

def Transfer_emit_Evm (self : Transfer) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := self.value
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 100389287136786176327247604509743168900146139575972864366142685224231313322991) -- a constant, as the compiler evaluates it
  let v8 := self.«from»
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v11 := self.«to»
  r2 := v11
  let v12 := r2
  let v13 ← Address_as_topic v12
  let v15 ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Transfer (event_1 : Transfer) : FeM Unit := do
  let event_3 := event_1
  let v4 ← Transfer_emit_Evm event_3
  return ()

def init (initial_supply_0 : UInt256) : FeM Unit := do
  let v2 ← Evm_caller
  let sender := v2
  let initial_supply_4 := initial_supply_0
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
  store (Place.mk .storage l0.slot) initial_supply_4
  let v5 := ()
  let initial_supply_6 := initial_supply_0
  store (Field.store.field Layout.TokenStore 2) initial_supply_6
  let v7 := ()
  let v8 ← Address_zero
  let initial_supply_9 := initial_supply_0
  let v10 := (Transfer.mk v8 sender initial_supply_9)
  let v12 ← emit_Evm_Transfer v10
  return ()

def recv_Transfer («to» : UInt256) (value : UInt256) : FeM Bool := do
  let v3 ← Evm_caller
  let sender := v3
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 := decide (value.toNat ≤ v5.toNat)
  if v6 then
    let v7 := ()
    let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
    let v8 ← load (Place.mk .storage l1.slot)
    let v9 ← checkedSub v8 value
    let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
    store (Place.mk .storage l2.slot) v9
    let v10 := ()
    let l3 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
    let v11 ← load (Place.mk .storage l3.slot)
    let v12 ← checkedAdd v11 value
    let l4 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
    store (Place.mk .storage l4.slot) v12
    let v13 := ()
    let v14 := (Transfer.mk sender «to» value)
    let v16 ← emit_Evm_Transfer v14
    let v17 := true
    return v17
  else
    M.revert (assertData "ERC20: insufficient balance")

def Approval_emit_Evm (self : Approval) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := self.value
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 63486140976153616755203102783360879283472101686154884697241723088393386309925) -- a constant, as the compiler evaluates it
  let v8 := self.owner
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v11 := self.spender
  r2 := v11
  let v12 := r2
  let v13 ← Address_as_topic v12
  let v15 ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Approval (event_1 : Approval) : FeM Unit := do
  let event_3 := event_1
  let v4 ← Approval_emit_Evm event_3
  return ()

def recv_Approve (spender : UInt256) (value : UInt256) : FeM Bool := do
  let v3 ← Evm_caller
  let owner := v3
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot owner
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot spender
  store (Place.mk .storage l1.slot) value
  let v5 := ()
  let v6 := (Approval.mk owner spender value)
  let v8 ← emit_Evm_Approval v6
  let v9 := true
  return v9

def recv_TransferFrom («from» : UInt256) («to» : UInt256) (value : UInt256) : FeM Bool := do
  let v4 ← Evm_caller
  let spender := v4
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot «from»
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot spender
  let v6 ← load (Place.mk .storage l1.slot)
  let current_allowance := v6
  let v8 := decide (value.toNat ≤ current_allowance.toNat)
  if v8 then
    let v9 := ()
    let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «from»
    let v10 ← load (Place.mk .storage l2.slot)
    let v11 := decide (value.toNat ≤ v10.toNat)
    if v11 then
      let v12 := ()
      let v13 ← checkedSub current_allowance value
      let l3 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot «from»
      let l4 ← StorageMap_locate_Address_u256 (Place.mk .storage l3.slot) (Place.mk .storage l3.slot).slot spender
      store (Place.mk .storage l4.slot) v13
      let v14 := ()
      let l5 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «from»
      let v15 ← load (Place.mk .storage l5.slot)
      let v16 ← checkedSub v15 value
      let l6 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «from»
      store (Place.mk .storage l6.slot) v16
      let v17 := ()
      let l7 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
      let v18 ← load (Place.mk .storage l7.slot)
      let v19 ← checkedAdd v18 value
      let l8 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
      store (Place.mk .storage l8.slot) v19
      let v20 := ()
      let v21 := (Transfer.mk «from» «to» value)
      let v23 ← emit_Evm_Transfer v21
      let v24 := true
      return v24
    else
      M.revert (assertData "ERC20: insufficient balance")
  else
    M.revert (assertData "ERC20: insufficient allowance")

def recv_BalanceOf (owner : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot owner
  let v1 ← load (Place.mk .storage l0.slot)
  return v1

def recv_Allowance (owner : UInt256) (spender : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot owner
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot spender
  let v2 ← load (Place.mk .storage l1.slot)
  return v2

def recv_TotalSupply : FeM UInt256 := do
  let v0 ← load (Field.store.field Layout.TokenStore 2)
  return v0

def Transfer_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def Transfer_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Transfer_1 := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Transfer_1.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Transfer_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Transfer_1 := do
  let v3 ← Transfer_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Transfer_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData input_11 base len
  return v12

def recv_Transfer_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv_Transfer message.«to» message.value
  Evm_return_value_Sol_bool result

def Approve_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def Approve_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Approve := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Approve.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Approve_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Approve := do
  let v3 ← Approve_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Approve_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData input_11 base len
  return v12

def recv_Approve_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv_Approve message.spender message.value
  Evm_return_value_Sol_bool result

def TransferFrom_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  return v6

def TransferFrom_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM TransferFrom := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let __fe_msg_decode_base_11 := __fe_msg_decode_base_1
  let v12 ← abi_field_head_size_Address
  let v13 ← checkedAdd __fe_msg_decode_base_11 v12
  let v14 ← abi_field_head_size_Address
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v15 __fe_msg_decode_input_len
  let __fe_field_2 := v16
  let v18 := (TransferFrom.mk __fe_field_0 __fe_field_1 __fe_field_2)
  return v18

def decode_from_prechecked_head_TransferFrom_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TransferFrom := do
  let v3 ← TransferFrom_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← TransferFrom_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData input_11 base len
  return v12

def recv_TransferFrom_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv_TransferFrom message.«from» message.«to» message.value
  Evm_return_value_Sol_bool result

def BalanceOf_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  return v2

def BalanceOf_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM BalanceOf := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (BalanceOf.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_BalanceOf_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM BalanceOf := do
  let v3 ← BalanceOf_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← BalanceOf_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData input_11 base len
  return v12

def recv_BalanceOf_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv_BalanceOf message.owner
  Evm_return_value_Sol_u256 result

def Allowance_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  return v4

def Allowance_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Allowance := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Allowance.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Allowance_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Allowance := do
  let v3 ← Allowance_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Allowance_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData input_11 base len
  return v12

def recv_Allowance_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv_Allowance message.owner message.spender
  Evm_return_value_Sol_u256 result

def TotalSupply_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def TotalSupply_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM TotalSupply := do
  let v2 := (TotalSupply.mk)
  return v2

def decode_from_prechecked_head_TotalSupply_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TotalSupply := do
  let v3 ← TotalSupply_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← TotalSupply_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData input_11 base len
  return v12

def recv_TotalSupply_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv_TotalSupply
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x095ea7b3 then
      recv_Approve_entry
    else if selector = UInt256.ofNat 0x18160ddd then
      recv_TotalSupply_entry
    else if selector = UInt256.ofNat 0x23b872dd then
      recv_TransferFrom_entry
    else if selector = UInt256.ofNat 0x70a08231 then
      recv_BalanceOf_entry
    else if selector = UInt256.ofNat 0xa9059cbb then
      recv_Transfer_entry
    else if selector = UInt256.ofNat 0xdd62ed3e then
      recv_Allowance_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← Sol_decoder_new
  let (decoded, _) ← tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let _ ← init decoded
  return ByteArray.empty


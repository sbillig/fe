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

/-- `<Transfer as Event>::emit<Evm>` -/
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
  let _ ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, Transfer>` -/
def emit_Evm_Transfer (event : Transfer) : FeM Unit := do
  let _ ← Transfer_emit_Evm event
  return ()

/-- `init(initial_supply: u256)` of contract `FeERC20` -/
def init (initial_supply : UInt256) : FeM Unit := do
  let sender ← Evm_caller
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
  store (Place.mk .storage l0.slot) initial_supply
  store (Field.store.field Layout.TokenStore 2) initial_supply
  let v8 ← Address_zero
  let v10 := (Transfer.mk v8 sender initial_supply)
  let _ ← emit_Evm_Transfer v10
  return ()

/-- `Transfer { to, value } -> bool`, selector `0xa9059cbb` (`transfer(address,uint256)`) -/
def recv_Transfer («to» : UInt256) (value : UInt256) : FeM Bool := do
  let sender ← Evm_caller
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 := decide (value.toNat ≤ v5.toNat)
  if v6 then
    let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
    let v8 ← load (Place.mk .storage l1.slot)
    let v9 ← checkedSub v8 value
    let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot sender
    store (Place.mk .storage l2.slot) v9
    let l3 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
    let v11 ← load (Place.mk .storage l3.slot)
    let v12 ← checkedAdd v11 value
    let l4 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
    store (Place.mk .storage l4.slot) v12
    let v14 := (Transfer.mk sender «to» value)
    let _ ← emit_Evm_Transfer v14
    return true
  else
    M.revert (assertData "ERC20: insufficient balance")

/-- `<Approval as Event>::emit<Evm>` -/
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
  let _ ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, Approval>` -/
def emit_Evm_Approval (event : Approval) : FeM Unit := do
  let _ ← Approval_emit_Evm event
  return ()

/-- `Approve { spender, value } -> bool`, selector `0x095ea7b3` (`approve(address,uint256)`) -/
def recv_Approve (spender : UInt256) (value : UInt256) : FeM Bool := do
  let owner ← Evm_caller
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot owner
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot spender
  store (Place.mk .storage l1.slot) value
  let v6 := (Approval.mk owner spender value)
  let _ ← emit_Evm_Approval v6
  return true

/-- `TransferFrom { from, to, value } -> bool`, selector `0x23b872dd` (`transferFrom(address,address,uint256)`) -/
def recv_TransferFrom («from» : UInt256) («to» : UInt256) (value : UInt256) : FeM Bool := do
  let spender ← Evm_caller
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot «from»
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot spender
  let current_allowance ← load (Place.mk .storage l1.slot)
  let v8 := decide (value.toNat ≤ current_allowance.toNat)
  if v8 then
    let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «from»
    let v10 ← load (Place.mk .storage l2.slot)
    let v11 := decide (value.toNat ≤ v10.toNat)
    if v11 then
      let v13 ← checkedSub current_allowance value
      let l3 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot «from»
      let l4 ← StorageMap_locate_Address_u256 (Place.mk .storage l3.slot) (Place.mk .storage l3.slot).slot spender
      store (Place.mk .storage l4.slot) v13
      let l5 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «from»
      let v15 ← load (Place.mk .storage l5.slot)
      let v16 ← checkedSub v15 value
      let l6 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «from»
      store (Place.mk .storage l6.slot) v16
      let l7 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
      let v18 ← load (Place.mk .storage l7.slot)
      let v19 ← checkedAdd v18 value
      let l8 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot «to»
      store (Place.mk .storage l8.slot) v19
      let v21 := (Transfer.mk «from» «to» value)
      let _ ← emit_Evm_Transfer v21
      return true
    else
      M.revert (assertData "ERC20: insufficient balance")
  else
    M.revert (assertData "ERC20: insufficient allowance")

/-- `BalanceOf { owner } -> u256`, selector `0x70a08231` (`balanceOf(address)`) -/
def recv_BalanceOf (owner : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.TokenStore 0) (Field.store.field Layout.TokenStore 0).slot owner
  let v1 ← load (Place.mk .storage l0.slot)
  return v1

/-- `Allowance { owner, spender } -> u256`, selector `0xdd62ed3e` (`allowance(address,address)`) -/
def recv_Allowance (owner : UInt256) (spender : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.TokenStore 1) (Field.store.field Layout.TokenStore 1).slot owner
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot spender
  let v2 ← load (Place.mk .storage l1.slot)
  return v2

/-- `TotalSupply -> u256`, selector `0x18160ddd` (`totalSupply()`) -/
def recv_TotalSupply : FeM UInt256 := do
  let v0 ← load (Field.store.field Layout.TokenStore 2)
  return v0

/-- `<Transfer as AbiSize>::HEAD_SIZE` -/
def Transfer_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<Transfer as Decode>::decode_from<CallData>` -/
def Transfer_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Transfer_1 := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base v7
  let __fe_field_1 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v8 __fe_msg_decode_input_len
  let v11 := (Transfer_1.mk __fe_field_0 __fe_field_1)
  return v11

/-- `core::abi::Decode::decode_from_prechecked_head<Transfer, Sol, CallData>` -/
def decode_from_prechecked_head_Transfer_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Transfer_1 := do
  let v3 ← Transfer_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Transfer>` -/
def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Transfer_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Transfer`, for `erc20::Erc20Msg::Transfer` -/
def recv_Transfer_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv_Transfer message.«to» message.value
  Evm_return_value_Sol_bool result

/-- `<Approve as AbiSize>::HEAD_SIZE` -/
def Approve_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<Approve as Decode>::decode_from<CallData>` -/
def Approve_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Approve := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base v7
  let __fe_field_1 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v8 __fe_msg_decode_input_len
  let v11 := (Approve.mk __fe_field_0 __fe_field_1)
  return v11

/-- `core::abi::Decode::decode_from_prechecked_head<Approve, Sol, CallData>` -/
def decode_from_prechecked_head_Approve_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Approve := do
  let v3 ← Approve_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Approve>` -/
def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Approve_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Approve`, for `erc20::Erc20Msg::Approve` -/
def recv_Approve_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv_Approve message.spender message.value
  Evm_return_value_Sol_bool result

/-- `<TransferFrom as AbiSize>::HEAD_SIZE` -/
def TransferFrom_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  return v6

/-- `<TransferFrom as Decode>::decode_from<CallData>` -/
def TransferFrom_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM TransferFrom := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base v7
  let __fe_field_1 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base v8 __fe_msg_decode_input_len
  let v12 ← abi_field_head_size_Address
  let v13 ← checkedAdd __fe_msg_decode_base v12
  let v14 ← abi_field_head_size_Address
  let v15 ← checkedAdd v13 v14
  let __fe_field_2 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v15 __fe_msg_decode_input_len
  let v18 := (TransferFrom.mk __fe_field_0 __fe_field_1 __fe_field_2)
  return v18

/-- `core::abi::Decode::decode_from_prechecked_head<TransferFrom, Sol, CallData>` -/
def decode_from_prechecked_head_TransferFrom_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TransferFrom := do
  let v3 ← TransferFrom_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, TransferFrom>` -/
def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← TransferFrom_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_TransferFrom`, for `erc20::Erc20Msg::TransferFrom` -/
def recv_TransferFrom_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv_TransferFrom message.«from» message.«to» message.value
  Evm_return_value_Sol_bool result

/-- `<BalanceOf as AbiSize>::HEAD_SIZE` -/
def BalanceOf_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<BalanceOf as Decode>::decode_from<CallData>` -/
def BalanceOf_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM BalanceOf := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v6 := (BalanceOf.mk __fe_field_0)
  return v6

/-- `core::abi::Decode::decode_from_prechecked_head<BalanceOf, Sol, CallData>` -/
def decode_from_prechecked_head_BalanceOf_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM BalanceOf := do
  let v3 ← BalanceOf_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, BalanceOf>` -/
def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← BalanceOf_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_BalanceOf`, for `erc20::Erc20Msg::BalanceOf` -/
def recv_BalanceOf_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv_BalanceOf message.owner
  Evm_return_value_Sol_u256 result

/-- `<Allowance as AbiSize>::HEAD_SIZE` -/
def Allowance_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<Allowance as Decode>::decode_from<CallData>` -/
def Allowance_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Allowance := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base v7
  let __fe_field_1 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base v8 __fe_msg_decode_input_len
  let v11 := (Allowance.mk __fe_field_0 __fe_field_1)
  return v11

/-- `core::abi::Decode::decode_from_prechecked_head<Allowance, Sol, CallData>` -/
def decode_from_prechecked_head_Allowance_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Allowance := do
  let v3 ← Allowance_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Allowance>` -/
def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Allowance_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Allowance`, for `erc20::Erc20Msg::Allowance` -/
def recv_Allowance_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv_Allowance message.owner message.spender
  Evm_return_value_Sol_u256 result

/-- `<TotalSupply as AbiSize>::HEAD_SIZE` -/
def TotalSupply_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<TotalSupply as Decode>::decode_from<CallData>` -/
def TotalSupply_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM TotalSupply := do
  let v2 := (TotalSupply.mk)
  return v2

/-- `core::abi::Decode::decode_from_prechecked_head<TotalSupply, Sol, CallData>` -/
def decode_from_prechecked_head_TotalSupply_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TotalSupply := do
  let v3 ← TotalSupply_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, TotalSupply>` -/
def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← TotalSupply_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_TotalSupply`, for `erc20::Erc20Msg::TotalSupply` -/
def recv_TotalSupply_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv_TotalSupply
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `FeERC20`: the entry of the arm the call's selector picks -/
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

/-- The deployment entry of contract `FeERC20`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← Sol_decoder_new
  let (decoded, _) ← tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let _ ← init decoded
  return ByteArray.empty


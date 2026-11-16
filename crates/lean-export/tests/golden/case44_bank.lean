export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head StorageMap_locate_Address_u256 abi_field_head_size_Address abi_field_head_size_u256 decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData len_CallData)

def Field.ledger : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Ledger : Layout :=
  { storage := [0, 1] }

structure Credit where
  who : UInt256
  amount : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def credit (m : Place) (who : UInt256) (amount_2 : UInt256) : FeM Unit := do
  let amount_3 := amount_2
  let v4 := who
  let l0 ← StorageMap_locate_Address_u256 m m.slot v4
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 ← checkedAdd v5 amount_3
  let l1 ← StorageMap_locate_Address_u256 m m.slot v4
  store (Place.mk .storage l1.slot) v6
  let v7 := ()
  return ()

def recv_Credit (who : UInt256) (amount : UInt256) : FeM UInt256 := do
  let mut who := who
  let v1 := (Field.ledger.field Layout.Ledger 0)
  let who_2 := who
  let v3 ← credit v1 who_2 amount
  let v4 := (Field.ledger.field Layout.Ledger 0)
  let who_5 := who
  let v6 ← credit v4 who_5 amount
  let v7 ← load (Field.ledger.field Layout.Ledger 1)
  let v8 ← checkedAdd v7 amount
  store (Field.ledger.field Layout.Ledger 1) v8
  let v9 := ()
  let who_10 := who
  let l0 ← StorageMap_locate_Address_u256 (Field.ledger.field Layout.Ledger 0) (Field.ledger.field Layout.Ledger 0).slot who_10
  let v11 ← load (Place.mk .storage l0.slot)
  let v12 ← load (Field.ledger.field Layout.Ledger 1)
  let v13 ← checkedAdd v11 v12
  return v13

def Credit_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def Credit_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Credit := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Credit.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Credit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Credit := do
  let v3 ← Credit_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Credit : FeM Credit := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Credit_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Credit_Sol_CallData input_11 base len
  return v12

def recv_Credit_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Credit
  let result ← recv_Credit message.who message.amount
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x00000001 then
      recv_Credit_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


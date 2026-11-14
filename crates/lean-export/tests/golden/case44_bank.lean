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

def credit (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := v2
  let v4 := v1
  let l0 ← StorageMap_locate_Address_u256 v0 v0.slot v4
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 ← checkedAdd v5 v3
  let l1 ← StorageMap_locate_Address_u256 v0 v0.slot v4
  store (Place.mk .storage l1.slot) v6
  let v7 := ()
  return ()

def recv0_arm0 (r0 : UInt256) (v0 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  let v1 := (Field.ledger.field Layout.Ledger 0)
  let v2 := r0
  let v3 ← credit v1 v2 v0
  let v4 := (Field.ledger.field Layout.Ledger 0)
  let v5 := r0
  let v6 ← credit v4 v5 v0
  let v7 ← load (Field.ledger.field Layout.Ledger 1)
  let v8 ← checkedAdd v7 v0
  store (Field.ledger.field Layout.Ledger 1) v8
  let v9 := ()
  let v10 := r0
  let l0 ← StorageMap_locate_Address_u256 (Field.ledger.field Layout.Ledger 0) (Field.ledger.field Layout.Ledger 0).slot v10
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

def Credit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Credit := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Credit.mk v5 v10)
  return v11

def decode_from_prechecked_head_Credit_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Credit := do
  let v3 ← Credit_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Credit : FeM Credit := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Credit_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Credit_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Credit
  let result ← recv0_arm0 message.who message.amount
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x00000001 then
      recv0_arm0_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


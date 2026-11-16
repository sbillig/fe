export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head StorageMap_locate_Address_u256 abi_field_head_size_Address abi_field_head_size_u256 decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData len_CallData)

def Field.ledger : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Ledger : Layout :=
  { storage := [0, 1] }

/-- `ledger.balances` -/
abbrev Field.ledger.balances : Place :=
  Field.ledger.field Layout.Ledger 0

/-- `ledger.total` -/
abbrev Field.ledger.total : Place :=
  Field.ledger.field Layout.Ledger 1

structure Credit where
  who : UInt256
  amount : UInt256
deriving Inhabited

/-- The `init` of contract `Case44Bank`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `case44_bank::credit` -/
def credit (m : Place) (who : UInt256) (amount : UInt256) : FeM Unit := do
  -- m[who] += amount
  let v4 := who
  let l0 ← StorageMap_locate_Address_u256 m m.slot v4
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 ← checkedAdd v5 amount
  let l1 ← StorageMap_locate_Address_u256 m m.slot v4
  store (Place.mk .storage l1.slot) v6
  return ()

/-- `Credit { who, amount } -> u256`, selector `0x00000001` -/
def recv_Credit (who : UInt256) (amount : UInt256) : FeM UInt256 := do
  let mut who := who
  -- credit(mut ledger.balances, who, amount)
  let v1 := Field.ledger.balances
  let _ ← credit v1 who amount
  -- credit(mut ledger.balances, who, amount)
  let v4 := Field.ledger.balances
  let _ ← credit v4 who amount
  -- ledger.total += amount
  let v7 ← load Field.ledger.total
  let v8 ← checkedAdd v7 amount
  store Field.ledger.total v8
  -- ledger.balances[who] + ledger.total
  let balances_who_10 ← StorageMap_locate_Address_u256 Field.ledger.balances Field.ledger.balances.slot who
  let v11 ← load (Place.mk .storage balances_who_10.slot)
  let v12 ← load Field.ledger.total
  let v13 ← checkedAdd v11 v12
  return v13

/-- `<Credit as AbiSize>::HEAD_SIZE` -/
def Credit_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<Credit as Decode>::decode_from<CallData>` -/
def Credit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Credit := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v1 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v2
  return Credit.mk v4 v9

/-- `core::abi::Decode::decode_from_prechecked_head<Credit, Sol, CallData>` -/
def decode_from_prechecked_head_Credit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Credit := do
  -- Self::decode_from(input, pos)
  let v3 ← Credit_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Credit>` -/
def decode_runtime_args_Evm_Sol_Credit : FeM Credit := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Credit_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Credit_Sol_CallData input base len
  return v12

/-- The entry of `recv_Credit`, for `case44_bank::LedgerMsg::Credit` -/
def recv_Credit_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Credit
  let result ← recv_Credit message.who message.amount
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Case44Bank`: the entry of the arm the call's selector picks -/
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

/-- The deployment entry of contract `Case44Bank`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


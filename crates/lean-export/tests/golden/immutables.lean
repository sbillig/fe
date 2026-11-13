export Lib (CallData Evm_input Evm_return_value_Sol_u256 MemSlice_u8.mk Sol_SELECTOR_SIZE Sol_decoder_new Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8)

structure Config where
  small : UInt256
  on : Bool
  limit : UInt256
deriving Inhabited

def Field.count : Place :=
  .storage (UInt256.ofNat 0)

def cload_Config (p : UInt256) : FeM Config := do
  let f0 ← cloadInt (.u 8) (p + UInt256.ofNat 0)
  let f1 ← cloadBool (p + UInt256.ofNat 1)
  let f2 ← cload (p + UInt256.ofNat 2)
  return (Config.mk f0 f1 f2)

structure Limit where
deriving Inhabited

structure Sum where
deriving Inhabited

structure Bump where
  «by» : UInt256
deriving Inhabited

def mstore_Config (p : UInt256) (v : Config) : FeM Unit := do
  mstoreInt (.u 8) (p + UInt256.ofNat 0) v.small
  mstoreBool (p + UInt256.ofNat 1) v.on
  mstore (p + UInt256.ofNat 2) v.limit

def set_small_Config (v0 : UInt256) (e0 : Config) : FeM (Unit × Config) := do
  let mut e0 := e0
  let v1 := v0
  e0 := { e0 with small := v1 }
  let v2 := ()
  return ((), e0)

def init (v0 : UInt256) (e0 : Config) (e1 : (Vector UInt256 2)) : FeM (Unit × Config × (Vector UInt256 2)) := do
  let mut e0 := e0
  let mut e1 := e1
  let v1 := (UInt256.ofNat 0)
  let v2 := true
  let v3 := v0
  let v4 := (Config.mk v1 v2 v3)
  e0 := v4
  let v5 := ()
  let v6 := (UInt256.ofNat 7)
  let (v7, v7_0) ← set_small_Config v6 e0
  e0 := v7_0
  let v8 := v0
  let v9 := (UInt256.ofNat 2)
  let v10 := #v[v8, v9]
  e1 := v10
  let v11 := ()
  let v12 := (UInt256.ofNat 1)
  store Field.count v12
  let v13 := ()
  return ((), e0, e1)

def recv0_arm0 : FeM UInt256 := do
  let e0 ← cload_Config (← immutableAt 4 0)
  let v1 := e0.on
  let b3 : UInt256 → FeM UInt256 := fun v0 => do
    return v0
  if v1 then
    let v2 := e0.limit
    let v3 := e0.small
    let v4 := v3
    let v5 ← checkedAdd v2 v4
    let v6 := v5
    b3 v6
  else
    let v7 := (UInt256.ofNat 0)
    let v8 := v7
    b3 v8

def recv0_arm1 : FeM UInt256 := do
  let e1 ← cloadArray 32 (fun q => cload q) (← immutableAt 4 2) 2
  let v0 := (UInt256.ofNat 0)
  checkIndex v0 2
  let v1 := e1[v0.toNat]!
  let v2 := (UInt256.ofNat 1)
  checkIndex v2 2
  let v3 := e1[v2.toNat]!
  let v4 ← checkedAdd v1 v3
  return v4

def capped_Config (v0 : UInt256) : FeM UInt256 := do
  let e0 ← cload_Config (← immutableAt 4 0)
  let v2 := v0
  let v3 := e0.limit
  let v4 := decide (v3.toNat < v2.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 := e0.limit
    let v6 := v5
    b3 v6
  else
    let v7 := v0
    let v8 := v7
    b3 v8

def recv0_arm2 (v0 : UInt256) : FeM UInt256 := do
  let e0 ← cload_Config (← immutableAt 4 0)
  let v1 ← capped_Config v0
  let v2 ← load Field.count
  let v3 ← checkedAdd v2 v1
  store Field.count v3
  let v4 := ()
  let v5 ← load Field.count
  return v5

def Limit_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Limit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Limit := do
  let v2 := (Limit.mk)
  return v2

def decode_from_prechecked_head_Limit_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Limit := do
  let v3 ← Limit_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Limit : FeM Limit := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Limit_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Limit_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Limit
  let result ← recv0_arm0
  Evm_return_value_Sol_u256 result

def Sum_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Sum_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Sum := do
  let v2 := (Sum.mk)
  return v2

def decode_from_prechecked_head_Sum_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Sum := do
  let v3 ← Sum_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Sum : FeM Sum := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Sum_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Sum_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Sum
  let result ← recv0_arm1
  Evm_return_value_Sol_u256 result

def Bump_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Bump_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Bump := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Bump.mk v5)
  return v6

def decode_from_prechecked_head_Bump_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Bump := do
  let v3 ← Bump_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Bump : FeM Bump := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Bump_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Bump_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Bump
  let result ← recv0_arm2 message.«by»
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x853255cc then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0xa4d66daf then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0xb20eb4c4 then
      recv0_arm2_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← Sol_decoder_new
  let (decoded, _) ← tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  let words ← alloc (UInt256.ofNat 128)
  let (_, e0, e1) ← init decoded default default
  mstore_Config (words + UInt256.ofNat (0 * 32)) e0
  mstoreArray 32 (fun q x => mstore q x) (words + UInt256.ofNat (2 * 32)) e1
  memoryBytes words (UInt256.ofNat 128)


export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_tuple_u256_u256 abi_field_head_size_u256 decode_msg_field_from_Sol_tuple_u256_u256_CallData decode_msg_field_from_Sol_u256_CallData len_CallData)

def Field.count : Place :=
  .storage (UInt256.ofNat 0)

structure Run where
  x : UInt256
deriving Inhabited

structure Spin where
  n : UInt256
deriving Inhabited

structure Pairs where
  t : (UInt256 × UInt256)
deriving Inhabited

/-- The `init` of contract `Collide`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `collide::double` -/
def double (x : UInt256) : FeM UInt256 := do
  let v3_1 ← checkedMul x (UInt256.ofNat 2)
  return v3_1

/-- `collide::v3` -/
def v3 (x : UInt256) : FeM UInt256 := do
  let v3_1 ← checkedAdd x (UInt256.ofNat 3)
  return v3_1

/-- `collide::bounds::limit` -/
def limit : FeM UInt256 := do
  return UInt256.ofNat 10

/-- `collide::size` -/
def size : FeM UInt256 := do
  return UInt256.ofNat 4

/-- `Run { x } -> u256`, selector `0xa444f5e9` (`run(uint256)`) -/
def recv_CollideMsg_Run (x : UInt256) : FeM UInt256 := do
  let limit_1 ← checkedMul x (UInt256.ofNat 2)
  let double_1 ← checkedAdd limit_1 (UInt256.ofNat 1)
  let count_7 ← load Field.count
  let load_1 ← checkedAdd double_1 count_7
  let store_1 ← checkedAdd load_1 (UInt256.ofNat 1)
  store Field.count store_1
  let v15 ← double store_1
  let v16 ← v3 v15
  let v17 ← limit
  let v18 ← checkedAdd v16 v17
  let v19 ← size
  let v20 ← checkedAdd v18 v19
  let count_21 ← load Field.count
  let v22 ← checkedAdd v20 count_21
  return v22

/-- `collide::b1` -/
def b1 (total : UInt256) (i : UInt256) : FeM UInt256 := do
  let v4 ← checkedAdd total i
  return v4

/-- `Spin { n } -> u256`, selector `0xa5b6ea8f` (`spin(uint256)`) -/
def recv_Spin (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  total := (UInt256.ofNat 0)
  i := (UInt256.ofNat 0)
  let rec loop (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    if decide (i.toNat < n.toNat) then
      let i_10 ← checkedAdd i (UInt256.ofNat 1)
      i := i_10
      let total_17 ← b1 total i
      total := total_17
      loop total i
    else
      return total
  partial_fixpoint
  loop total i

/-- `Run { x } -> u256`, selector `0xeb034fa7` (`rerun(uint256)`) -/
def recv_OtherMsg_Run (x : UInt256) : FeM UInt256 := do
  return x

/-- `Pairs { t: (a, b) } -> u256`, selector `0x6a530d0d` (`pairs((uint256,uint256))`) -/
def recv_Pairs (v0 : (UInt256 × UInt256)) : FeM UInt256 := do
  let v5 ← checkedAdd v0.fst v0.snd
  return v5

/-- `<Run as AbiSize>::HEAD_SIZE` -/
def Run_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Run as Decode>::decode_from<CallData>` -/
def Run_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Run := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Run.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Run, Sol, CallData>` -/
def decode_from_prechecked_head_Run_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Run := do
  let v3_1 ← Run_decode_from_CallData input pos
  return v3_1

/-- `core::contracts::decode_runtime_args<Evm, Sol, Run>` -/
def decode_runtime_args_Evm_Sol_Run : FeM Run := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let len ← len_CallData input
  let v9 ← Run_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let v12 ← decode_from_prechecked_head_Run_Sol_CallData input base len
  return v12

/-- The entry of `recv_CollideMsg_Run`, for `collide::CollideMsg::Run` -/
def recv_CollideMsg_Run_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Run
  let result ← recv_CollideMsg_Run message.x
  Evm_return_value_Sol_u256 result

/-- `<Spin as AbiSize>::HEAD_SIZE` -/
def Spin_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Spin as Decode>::decode_from<CallData>` -/
def Spin_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Spin := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Spin.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Spin, Sol, CallData>` -/
def decode_from_prechecked_head_Spin_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Spin := do
  let v3_1 ← Spin_decode_from_CallData input pos
  return v3_1

/-- `core::contracts::decode_runtime_args<Evm, Sol, Spin>` -/
def decode_runtime_args_Evm_Sol_Spin : FeM Spin := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let len ← len_CallData input
  let v9 ← Spin_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let v12 ← decode_from_prechecked_head_Spin_Sol_CallData input base len
  return v12

/-- The entry of `recv_Spin`, for `collide::CollideMsg::Spin` -/
def recv_Spin_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Spin
  let result ← recv_Spin message.n
  Evm_return_value_Sol_u256 result

/-- The entry of `recv_OtherMsg_Run`, for `collide::OtherMsg::Run` -/
def recv_OtherMsg_Run_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Run
  let result ← recv_OtherMsg_Run message.x
  Evm_return_value_Sol_u256 result

/-- `<Pairs as AbiSize>::HEAD_SIZE` -/
def Pairs_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_tuple_u256_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Pairs as Decode>::decode_from<CallData>` -/
def Pairs_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Pairs := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_tuple_u256_u256_CallData v0 v1 v1 v2
  return Pairs.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Pairs, Sol, CallData>` -/
def decode_from_prechecked_head_Pairs_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Pairs := do
  let v3_1 ← Pairs_decode_from_CallData input pos
  return v3_1

/-- `core::contracts::decode_runtime_args<Evm, Sol, Pairs>` -/
def decode_runtime_args_Evm_Sol_Pairs : FeM Pairs := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let len ← len_CallData input
  let v9 ← Pairs_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let v12 ← decode_from_prechecked_head_Pairs_Sol_CallData input base len
  return v12

/-- The entry of `recv_Pairs`, for `collide::OtherMsg::Pairs` -/
def recv_Pairs_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Pairs
  let result ← recv_Pairs message.t
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Collide`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size_1 ← calldatasize
  if size_1.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x6a530d0d then
      recv_Pairs_entry
    else if selector = UInt256.ofNat 0xa444f5e9 then
      recv_CollideMsg_Run_entry
    else if selector = UInt256.ofNat 0xa5b6ea8f then
      recv_Spin_entry
    else if selector = UInt256.ofNat 0xeb034fa7 then
      recv_OtherMsg_Run_entry
    else
      revertEmpty

/-- The deployment entry of contract `Collide`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


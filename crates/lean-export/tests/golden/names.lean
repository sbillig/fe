export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)

structure Pair where
  a : UInt256
  b : UInt256
deriving Inhabited

def Field.pair : Place :=
  .storage (UInt256.ofNat 0)

def store_Pair (p : Place) (v : Pair) : FeM Unit := do
  store (p.offset 0) v.a
  store (p.offset 1) v.b

structure Prop_1 where
  flag : Bool
  n : UInt256
deriving Inhabited

def Field.prop : Place :=
  .storage (UInt256.ofNat 2)

def store_Prop (p : Place) (v : Prop_1) : FeM Unit := do
  let b0 ← Lane.pack ⟨0, 1, false⟩ (.u 8) (Bool.toUInt256 v.flag)
  store (p.offset 0) (packed [(0, b0)])
  store (p.offset 1) v.n

def load_Pair_1 (p : Place) : FeM Pair := do
  let f0 ← load (p.offset 0)
  let f1 ← load (p.offset 1)
  return (Pair.mk f0 f1)

def load_Prop (p : Place) : FeM Prop_1 := do
  let f0 ← loadBoolAt (p.offset 0) ⟨0, 1, false⟩
  let f1 ← load (p.offset 1)
  return (Prop_1.mk f0 f1)

structure Run where
  x : UInt256
deriving Inhabited

/-- The `init` of contract `Names`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `names::load_Pair` -/
def load_Pair (x : UInt256) : FeM UInt256 := do
  let v3 ← checkedAdd x (UInt256.ofNat 1)
  return v3

/-- `names::long_k1` -/
def long_k1 (x : UInt256) : FeM UInt256 := do
  let v3 ← checkedAdd x (UInt256.ofNat 2)
  return v3

def long_k9 (v216 : UInt256) : FeM UInt256 := do
  let a43 ← checkedAdd v216 (UInt256.ofNat 43)
  let v221 ← checkedMul a43 (UInt256.ofNat 3)
  let a44 ← checkedAdd v221 (UInt256.ofNat 44)
  return a44

def long_k8 (v191 : UInt256) : FeM UInt256 := do
  let a38 ← checkedAdd v191 (UInt256.ofNat 38)
  let v196 ← checkedMul a38 (UInt256.ofNat 3)
  let a39 ← checkedAdd v196 (UInt256.ofNat 39)
  let v201 ← checkedMul a39 (UInt256.ofNat 3)
  let a40 ← checkedAdd v201 (UInt256.ofNat 40)
  let v206 ← checkedMul a40 (UInt256.ofNat 3)
  let a41 ← checkedAdd v206 (UInt256.ofNat 41)
  let v211 ← checkedMul a41 (UInt256.ofNat 3)
  let a42 ← checkedAdd v211 (UInt256.ofNat 42)
  let v216 ← checkedMul a42 (UInt256.ofNat 3)
  long_k9 v216

def long_k7 (a33 : UInt256) : FeM UInt256 := do
  let v171 ← checkedMul a33 (UInt256.ofNat 3)
  let a34 ← checkedAdd v171 (UInt256.ofNat 34)
  let v176 ← checkedMul a34 (UInt256.ofNat 3)
  let a35 ← checkedAdd v176 (UInt256.ofNat 35)
  let v181 ← checkedMul a35 (UInt256.ofNat 3)
  let a36 ← checkedAdd v181 (UInt256.ofNat 36)
  let v186 ← checkedMul a36 (UInt256.ofNat 3)
  let a37 ← checkedAdd v186 (UInt256.ofNat 37)
  let v191 ← checkedMul a37 (UInt256.ofNat 3)
  long_k8 v191

def long_k6 (a28 : UInt256) : FeM UInt256 := do
  let v146 ← checkedMul a28 (UInt256.ofNat 3)
  let a29 ← checkedAdd v146 (UInt256.ofNat 29)
  let v151 ← checkedMul a29 (UInt256.ofNat 3)
  let a30 ← checkedAdd v151 (UInt256.ofNat 30)
  let v156 ← checkedMul a30 (UInt256.ofNat 3)
  let a31 ← checkedAdd v156 (UInt256.ofNat 31)
  let v161 ← checkedMul a31 (UInt256.ofNat 3)
  let a32 ← checkedAdd v161 (UInt256.ofNat 32)
  let v166 ← checkedMul a32 (UInt256.ofNat 3)
  let a33 ← checkedAdd v166 (UInt256.ofNat 33)
  long_k7 a33

def long_k5 (a23 : UInt256) : FeM UInt256 := do
  let v121 ← checkedMul a23 (UInt256.ofNat 3)
  let a24 ← checkedAdd v121 (UInt256.ofNat 24)
  let v126 ← checkedMul a24 (UInt256.ofNat 3)
  let a25 ← checkedAdd v126 (UInt256.ofNat 25)
  let v131 ← checkedMul a25 (UInt256.ofNat 3)
  let a26 ← checkedAdd v131 (UInt256.ofNat 26)
  let v136 ← checkedMul a26 (UInt256.ofNat 3)
  let a27 ← checkedAdd v136 (UInt256.ofNat 27)
  let v141 ← checkedMul a27 (UInt256.ofNat 3)
  let a28 ← checkedAdd v141 (UInt256.ofNat 28)
  long_k6 a28

def long_k4 (v96 : UInt256) : FeM UInt256 := do
  let a19 ← checkedAdd v96 (UInt256.ofNat 19)
  let v101 ← checkedMul a19 (UInt256.ofNat 3)
  let a20 ← checkedAdd v101 (UInt256.ofNat 20)
  let v106 ← checkedMul a20 (UInt256.ofNat 3)
  let a21 ← checkedAdd v106 (UInt256.ofNat 21)
  let v111 ← checkedMul a21 (UInt256.ofNat 3)
  let a22 ← checkedAdd v111 (UInt256.ofNat 22)
  let v116 ← checkedMul a22 (UInt256.ofNat 3)
  let a23 ← checkedAdd v116 (UInt256.ofNat 23)
  long_k5 a23

def long_k3 (v71 : UInt256) : FeM UInt256 := do
  let a14 ← checkedAdd v71 (UInt256.ofNat 14)
  let v76 ← checkedMul a14 (UInt256.ofNat 3)
  let a15 ← checkedAdd v76 (UInt256.ofNat 15)
  let v81 ← checkedMul a15 (UInt256.ofNat 3)
  let a16 ← checkedAdd v81 (UInt256.ofNat 16)
  let v86 ← checkedMul a16 (UInt256.ofNat 3)
  let a17 ← checkedAdd v86 (UInt256.ofNat 17)
  let v91 ← checkedMul a17 (UInt256.ofNat 3)
  let a18 ← checkedAdd v91 (UInt256.ofNat 18)
  let v96 ← checkedMul a18 (UInt256.ofNat 3)
  long_k4 v96

def long_k2 (a9 : UInt256) : FeM UInt256 := do
  let v51 ← checkedMul a9 (UInt256.ofNat 3)
  let a10 ← checkedAdd v51 (UInt256.ofNat 10)
  let v56 ← checkedMul a10 (UInt256.ofNat 3)
  let a11 ← checkedAdd v56 (UInt256.ofNat 11)
  let v61 ← checkedMul a11 (UInt256.ofNat 3)
  let a12 ← checkedAdd v61 (UInt256.ofNat 12)
  let v66 ← checkedMul a12 (UInt256.ofNat 3)
  let a13 ← checkedAdd v66 (UInt256.ofNat 13)
  let v71 ← checkedMul a13 (UInt256.ofNat 3)
  long_k3 v71

def long_k1_1 (a4 : UInt256) : FeM UInt256 := do
  let v26 ← checkedMul a4 (UInt256.ofNat 3)
  let a5 ← checkedAdd v26 (UInt256.ofNat 5)
  let v31 ← checkedMul a5 (UInt256.ofNat 3)
  let a6 ← checkedAdd v31 (UInt256.ofNat 6)
  let v36 ← checkedMul a6 (UInt256.ofNat 3)
  let a7 ← checkedAdd v36 (UInt256.ofNat 7)
  let v41 ← checkedMul a7 (UInt256.ofNat 3)
  let a8 ← checkedAdd v41 (UInt256.ofNat 8)
  let v46 ← checkedMul a8 (UInt256.ofNat 3)
  let a9 ← checkedAdd v46 (UInt256.ofNat 9)
  long_k2 a9

/-- `names::long` -/
def long (x : UInt256) : FeM UInt256 := do
  let a0 ← checkedAdd x (UInt256.ofNat 1)
  let v6 ← checkedMul a0 (UInt256.ofNat 3)
  let a1 ← checkedAdd v6 (UInt256.ofNat 1)
  let v11 ← checkedMul a1 (UInt256.ofNat 3)
  let a2 ← checkedAdd v11 (UInt256.ofNat 2)
  let v16 ← checkedMul a2 (UInt256.ofNat 3)
  let a3 ← checkedAdd v16 (UInt256.ofNat 3)
  let v21 ← checkedMul a3 (UInt256.ofNat 3)
  let a4 ← checkedAdd v21 (UInt256.ofNat 4)
  long_k1_1 a4

/-- `Run { x } -> u256`, selector `0xa444f5e9` (`run(uint256)`) -/
def recv_Run (x : UInt256) : FeM UInt256 := do
  let v1 ← load_Pair x
  let v2 := (Pair.mk x v1)
  store_Pair Field.pair v2
  let v5 ← long_k1 x
  let v6 := (Prop_1.mk true v5)
  store_Prop Field.prop v6
  let pair ← load_Pair_1 Field.pair
  let prop ← load_Prop Field.prop
  let v12 := pair.a
  let v13 := pair.b
  let v14 ← checkedAdd v12 v13
  let v15 := prop.n
  let v16 ← checkedAdd v14 v15
  let v17 ← long x
  let v18 ← checkedAdd v16 v17
  return v18

/-- `<Run as AbiSize>::HEAD_SIZE` -/
def Run_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Run as Decode>::decode_from<CallData>` -/
def Run_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Run := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v6 := (Run.mk __fe_field_0)
  return v6

/-- `core::abi::Decode::decode_from_prechecked_head<Run, Sol, CallData>` -/
def decode_from_prechecked_head_Run_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Run := do
  let v3 ← Run_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Run>` -/
def decode_runtime_args_Evm_Sol_Run : FeM Run := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Run_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Run_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Run`, for `names::NamesMsg::Run` -/
def recv_Run_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Run
  let result ← recv_Run message.x
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Names`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0xa444f5e9 then
      recv_Run_entry
    else
      revertEmpty

/-- The deployment entry of contract `Names`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


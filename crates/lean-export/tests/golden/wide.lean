export Lib (CallData Evm_input Evm_return_value_Sol_u256 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 decode_msg_field_from_Sol_u256_CallData len_CallData)

structure Sum where
  a : UInt256
  b : UInt256
  c : UInt256
  d : UInt256
  e : UInt256
  f : UInt256
  g : UInt256
  h : UInt256
  i : UInt256
  j : UInt256
deriving Inhabited

/-- The `init` of contract `Wide`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Sum { a, b, c, d, e, f, g, h, i, j } -> u256`, selector `0x460e9674` (`sum(uint256,uint256,uint256,uint256,uint256,uint256,uint256,uint256,uint256,uint256)`) -/
def recv_Sum (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (e : UInt256) (f : UInt256) (g : UInt256) (h : UInt256) (i : UInt256) (j : UInt256) : FeM UInt256 := do
  let v10 ← checkedAdd a b
  let v11 ← checkedAdd v10 c
  let v12 ← checkedAdd v11 d
  let v13 ← checkedAdd v12 e
  let v14 ← checkedAdd v13 f
  let v15 ← checkedAdd v14 g
  let v16 ← checkedAdd v15 h
  let v17 ← checkedAdd v16 i
  let v18 ← checkedAdd v17 j
  return v18

/-- `<Sum as AbiSize>::HEAD_SIZE` -/
def Sum_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  let v9 ← abi_field_head_size_u256
  let v10 ← checkedAdd v8 v9
  let v11 ← abi_field_head_size_u256
  let v12 ← checkedAdd v10 v11
  let v13 ← abi_field_head_size_u256
  let v14 ← checkedAdd v12 v13
  let v15 ← abi_field_head_size_u256
  let v16 ← checkedAdd v14 v15
  let v17 ← abi_field_head_size_u256
  let v18 ← checkedAdd v16 v17
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd v18 v19
  return v20

def Sum_decode_from_CallData_k5 (__fe_field_0 : UInt256) (__fe_field_1 : UInt256) (__fe_field_2 : UInt256) (__fe_field_3 : UInt256) (__fe_field_4 : UInt256) (__fe_field_5 : UInt256) (__fe_field_6 : UInt256) (__fe_field_7 : UInt256) (__fe_field_8 : UInt256) (__fe_field_9 : UInt256) : FeM Sum := do
  let v123 := (Sum.mk __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3 __fe_field_4 __fe_field_5 __fe_field_6 __fe_field_7 __fe_field_8 __fe_field_9)
  return v123

def Sum_decode_from_CallData_k4 (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) (__fe_msg_decode_input_len : UInt256) (__fe_field_0 : UInt256) (__fe_field_1 : UInt256) (__fe_field_2 : UInt256) (__fe_field_3 : UInt256) (__fe_field_4 : UInt256) (__fe_field_5 : UInt256) (__fe_field_6 : UInt256) (__fe_field_7 : UInt256) (v97 : UInt256) : FeM Sum := do
  let v98 ← abi_field_head_size_u256
  let v99 ← checkedAdd v97 v98
  let __fe_field_8 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v99 __fe_msg_decode_input_len
  let v103 ← abi_field_head_size_u256
  let v104 ← checkedAdd __fe_msg_decode_base v103
  let v105 ← abi_field_head_size_u256
  let v106 ← checkedAdd v104 v105
  let v107 ← abi_field_head_size_u256
  let v108 ← checkedAdd v106 v107
  let v109 ← abi_field_head_size_u256
  let v110 ← checkedAdd v108 v109
  let v111 ← abi_field_head_size_u256
  let v112 ← checkedAdd v110 v111
  let v113 ← abi_field_head_size_u256
  let v114 ← checkedAdd v112 v113
  let v115 ← abi_field_head_size_u256
  let v116 ← checkedAdd v114 v115
  let v117 ← abi_field_head_size_u256
  let v118 ← checkedAdd v116 v117
  let v119 ← abi_field_head_size_u256
  let v120 ← checkedAdd v118 v119
  let __fe_field_9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v120 __fe_msg_decode_input_len
  Sum_decode_from_CallData_k5 __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3 __fe_field_4 __fe_field_5 __fe_field_6 __fe_field_7 __fe_field_8 __fe_field_9

def Sum_decode_from_CallData_k3 (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) (__fe_msg_decode_input_len : UInt256) (__fe_field_0 : UInt256) (__fe_field_1 : UInt256) (__fe_field_2 : UInt256) (__fe_field_3 : UInt256) (__fe_field_4 : UInt256) (__fe_field_5 : UInt256) (__fe_field_6 : UInt256) (v72 : UInt256) (v73 : UInt256) : FeM Sum := do
  let v74 ← checkedAdd v72 v73
  let v75 ← abi_field_head_size_u256
  let v76 ← checkedAdd v74 v75
  let v77 ← abi_field_head_size_u256
  let v78 ← checkedAdd v76 v77
  let v79 ← abi_field_head_size_u256
  let v80 ← checkedAdd v78 v79
  let __fe_field_7 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v80 __fe_msg_decode_input_len
  let v84 ← abi_field_head_size_u256
  let v85 ← checkedAdd __fe_msg_decode_base v84
  let v86 ← abi_field_head_size_u256
  let v87 ← checkedAdd v85 v86
  let v88 ← abi_field_head_size_u256
  let v89 ← checkedAdd v87 v88
  let v90 ← abi_field_head_size_u256
  let v91 ← checkedAdd v89 v90
  let v92 ← abi_field_head_size_u256
  let v93 ← checkedAdd v91 v92
  let v94 ← abi_field_head_size_u256
  let v95 ← checkedAdd v93 v94
  let v96 ← abi_field_head_size_u256
  let v97 ← checkedAdd v95 v96
  Sum_decode_from_CallData_k4 __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_input_len __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3 __fe_field_4 __fe_field_5 __fe_field_6 __fe_field_7 v97

def Sum_decode_from_CallData_k2 (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) (__fe_msg_decode_input_len : UInt256) (__fe_field_0 : UInt256) (__fe_field_1 : UInt256) (__fe_field_2 : UInt256) (__fe_field_3 : UInt256) (__fe_field_4 : UInt256) (__fe_field_5 : UInt256) : FeM Sum := do
  let v52 ← abi_field_head_size_u256
  let v53 ← checkedAdd __fe_msg_decode_base v52
  let v54 ← abi_field_head_size_u256
  let v55 ← checkedAdd v53 v54
  let v56 ← abi_field_head_size_u256
  let v57 ← checkedAdd v55 v56
  let v58 ← abi_field_head_size_u256
  let v59 ← checkedAdd v57 v58
  let v60 ← abi_field_head_size_u256
  let v61 ← checkedAdd v59 v60
  let v62 ← abi_field_head_size_u256
  let v63 ← checkedAdd v61 v62
  let __fe_field_6 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v63 __fe_msg_decode_input_len
  let v67 ← abi_field_head_size_u256
  let v68 ← checkedAdd __fe_msg_decode_base v67
  let v69 ← abi_field_head_size_u256
  let v70 ← checkedAdd v68 v69
  let v71 ← abi_field_head_size_u256
  let v72 ← checkedAdd v70 v71
  let v73 ← abi_field_head_size_u256
  Sum_decode_from_CallData_k3 __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_input_len __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3 __fe_field_4 __fe_field_5 __fe_field_6 v72 v73

def Sum_decode_from_CallData_k1 (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) (__fe_msg_decode_input_len : UInt256) (__fe_field_0 : UInt256) (__fe_field_1 : UInt256) (__fe_field_2 : UInt256) (__fe_field_3 : UInt256) : FeM Sum := do
  let v28 ← abi_field_head_size_u256
  let v29 ← checkedAdd __fe_msg_decode_base v28
  let v30 ← abi_field_head_size_u256
  let v31 ← checkedAdd v29 v30
  let v32 ← abi_field_head_size_u256
  let v33 ← checkedAdd v31 v32
  let v34 ← abi_field_head_size_u256
  let v35 ← checkedAdd v33 v34
  let __fe_field_4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v35 __fe_msg_decode_input_len
  let v39 ← abi_field_head_size_u256
  let v40 ← checkedAdd __fe_msg_decode_base v39
  let v41 ← abi_field_head_size_u256
  let v42 ← checkedAdd v40 v41
  let v43 ← abi_field_head_size_u256
  let v44 ← checkedAdd v42 v43
  let v45 ← abi_field_head_size_u256
  let v46 ← checkedAdd v44 v45
  let v47 ← abi_field_head_size_u256
  let v48 ← checkedAdd v46 v47
  let __fe_field_5 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v48 __fe_msg_decode_input_len
  Sum_decode_from_CallData_k2 __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_input_len __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3 __fe_field_4 __fe_field_5

/-- `<Sum as Decode>::decode_from<CallData>` -/
def Sum_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Sum := do
  let __fe_msg_decode_input_len ← len_CallData __fe_msg_decode_input
  let __fe_field_0 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd __fe_msg_decode_base v7
  let __fe_field_1 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v8 __fe_msg_decode_input_len
  let v12 ← abi_field_head_size_u256
  let v13 ← checkedAdd __fe_msg_decode_base v12
  let v14 ← abi_field_head_size_u256
  let v15 ← checkedAdd v13 v14
  let __fe_field_2 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v15 __fe_msg_decode_input_len
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd __fe_msg_decode_base v19
  let v21 ← abi_field_head_size_u256
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_u256
  let v24 ← checkedAdd v22 v23
  let __fe_field_3 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base v24 __fe_msg_decode_input_len
  Sum_decode_from_CallData_k1 __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_input_len __fe_field_0 __fe_field_1 __fe_field_2 __fe_field_3

/-- `core::abi::Decode::decode_from_prechecked_head<Sum, Sol, CallData>` -/
def decode_from_prechecked_head_Sum_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Sum := do
  let v3 ← Sum_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Sum>` -/
def decode_runtime_args_Evm_Sol_Sum : FeM Sum := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Sum_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Sum_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Sum`, for `wide::WideMsg::Sum` -/
def recv_Sum_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Sum
  let result ← recv_Sum message.a message.b message.c message.d message.e message.f message.g message.h message.i message.j
  Evm_return_value_Sol_u256 result

/-- The runtime entry of contract `Wide`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x460e9674 then
      recv_Sum_entry
    else
      revertEmpty

/-- The deployment entry of contract `Wide`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


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

def init : FeM Unit := do
  return ()

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) : FeM UInt256 := do
  let v10 ← checkedAdd v0 v1
  let v11 ← checkedAdd v10 v2
  let v12 ← checkedAdd v11 v3
  let v13 ← checkedAdd v12 v4
  let v14 ← checkedAdd v13 v5
  let v15 ← checkedAdd v14 v6
  let v16 ← checkedAdd v15 v7
  let v17 ← checkedAdd v16 v8
  let v18 ← checkedAdd v17 v9
  return v18

def Sum_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
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

def Sum_decode_from_CallData_k5 (v5 : UInt256) (v10 : UInt256) (v17 : UInt256) (v26 : UInt256) (v37 : UInt256) (v50 : UInt256) (v65 : UInt256) (v82 : UInt256) (v101 : UInt256) (v121 : UInt256) : FeM Sum := do
  let v122 := v121
  let v123 := (Sum.mk v5 v10 v17 v26 v37 v50 v65 v82 v101 v122)
  return v123

def Sum_decode_from_CallData_k4 (v0 : CallData) (v1 : UInt256) (v3 : UInt256) (v5 : UInt256) (v10 : UInt256) (v17 : UInt256) (v26 : UInt256) (v37 : UInt256) (v50 : UInt256) (v65 : UInt256) (v82 : UInt256) (v97 : UInt256) : FeM Sum := do
  let v98 ← abi_field_head_size_u256
  let v99 ← checkedAdd v97 v98
  let v100 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v99 v3
  let v101 := v100
  let v102 := v1
  let v103 ← abi_field_head_size_u256
  let v104 ← checkedAdd v102 v103
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
  let v121 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v120 v3
  Sum_decode_from_CallData_k5 v5 v10 v17 v26 v37 v50 v65 v82 v101 v121

def Sum_decode_from_CallData_k3 (v0 : CallData) (v1 : UInt256) (v3 : UInt256) (v5 : UInt256) (v10 : UInt256) (v17 : UInt256) (v26 : UInt256) (v37 : UInt256) (v50 : UInt256) (v65 : UInt256) (v72 : UInt256) (v73 : UInt256) : FeM Sum := do
  let v74 ← checkedAdd v72 v73
  let v75 ← abi_field_head_size_u256
  let v76 ← checkedAdd v74 v75
  let v77 ← abi_field_head_size_u256
  let v78 ← checkedAdd v76 v77
  let v79 ← abi_field_head_size_u256
  let v80 ← checkedAdd v78 v79
  let v81 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v80 v3
  let v82 := v81
  let v83 := v1
  let v84 ← abi_field_head_size_u256
  let v85 ← checkedAdd v83 v84
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
  Sum_decode_from_CallData_k4 v0 v1 v3 v5 v10 v17 v26 v37 v50 v65 v82 v97

def Sum_decode_from_CallData_k2 (v0 : CallData) (v1 : UInt256) (v3 : UInt256) (v5 : UInt256) (v10 : UInt256) (v17 : UInt256) (v26 : UInt256) (v37 : UInt256) (v49 : UInt256) : FeM Sum := do
  let v50 := v49
  let v51 := v1
  let v52 ← abi_field_head_size_u256
  let v53 ← checkedAdd v51 v52
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
  let v64 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v63 v3
  let v65 := v64
  let v66 := v1
  let v67 ← abi_field_head_size_u256
  let v68 ← checkedAdd v66 v67
  let v69 ← abi_field_head_size_u256
  let v70 ← checkedAdd v68 v69
  let v71 ← abi_field_head_size_u256
  let v72 ← checkedAdd v70 v71
  let v73 ← abi_field_head_size_u256
  Sum_decode_from_CallData_k3 v0 v1 v3 v5 v10 v17 v26 v37 v50 v65 v72 v73

def Sum_decode_from_CallData_k1 (v0 : CallData) (v1 : UInt256) (v3 : UInt256) (v5 : UInt256) (v10 : UInt256) (v17 : UInt256) (v25 : UInt256) : FeM Sum := do
  let v26 := v25
  let v27 := v1
  let v28 ← abi_field_head_size_u256
  let v29 ← checkedAdd v27 v28
  let v30 ← abi_field_head_size_u256
  let v31 ← checkedAdd v29 v30
  let v32 ← abi_field_head_size_u256
  let v33 ← checkedAdd v31 v32
  let v34 ← abi_field_head_size_u256
  let v35 ← checkedAdd v33 v34
  let v36 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v35 v3
  let v37 := v36
  let v38 := v1
  let v39 ← abi_field_head_size_u256
  let v40 ← checkedAdd v38 v39
  let v41 ← abi_field_head_size_u256
  let v42 ← checkedAdd v40 v41
  let v43 ← abi_field_head_size_u256
  let v44 ← checkedAdd v42 v43
  let v45 ← abi_field_head_size_u256
  let v46 ← checkedAdd v44 v45
  let v47 ← abi_field_head_size_u256
  let v48 ← checkedAdd v46 v47
  let v49 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v48 v3
  Sum_decode_from_CallData_k2 v0 v1 v3 v5 v10 v17 v26 v37 v49

def Sum_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Sum := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := v1
  let v12 ← abi_field_head_size_u256
  let v13 ← checkedAdd v11 v12
  let v14 ← abi_field_head_size_u256
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v3
  let v17 := v16
  let v18 := v1
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd v18 v19
  let v21 ← abi_field_head_size_u256
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_u256
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v24 v3
  Sum_decode_from_CallData_k1 v0 v1 v3 v5 v10 v17 v25

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

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Sum
  let result ← recv0_arm0 message.a message.b message.c message.d message.e message.f message.g message.h message.i message.j
  Evm_return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x460e9674 then
      recv0_arm0_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


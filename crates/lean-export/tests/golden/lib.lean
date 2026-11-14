structure CallData where
  base : UInt256
deriving Inhabited

structure FixedMemBuffer_32 where
  ptr : UInt256
deriving Inhabited

structure FixedMemBuffer_64 where
  ptr : UInt256
deriving Inhabited

structure MemBuffer where
  ptr : UInt256
  len : UInt256
  capacity : UInt256
deriving Inhabited

structure AbiFrame where
  ptr : UInt256
  len : UInt256
deriving Inhabited

structure MemSlice_u8 where
  ptr : UInt256
  len : UInt256
deriving Inhabited

structure SolDecoder where
  pos : UInt256
  base : UInt256
deriving Inhabited

def CallData_new : FeM CallData := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (CallData.mk v0)
  return v1

def Evm_input : FeM CallData := do
  let v1 ← CallData_new
  return v1

def Sol_SELECTOR_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4)
  return v0

def u256_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def u64_IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def U64_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def u64_MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

def u64_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def U256_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)
  return v0

def sign_extend (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v4 := v0
  let v5 := v1
  let v6 := v4 &&& v5
  let v7 := v6
  let v8 := v2
  let v9 := v7 &&& v8
  let v10 := (UInt256.ofNat 0)
  let v11 := decide (v9 ≠ v10)
  let b3 : UInt256 → FeM UInt256 := fun v3 => do
    return v3
  if v11 then
    let v12 ← U256_MASK
    let v13 := v1
    let v14 := v12 ^^^ v13
    let v15 := v7 ||| v14
    let v16 := v15
    b3 v16
  else
    let v17 := v7
    b3 v17

def u64_from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := truncate 64 v0
  return v1

def downcast_truncate_u256_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← u256_to_word v0
  let v3 := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := v3 &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def CallData_size (v0 : CallData) : FeM UInt256 := do
  let v2 ← calldatasize
  let v3 := v2
  let v4 := v0.base
  let v5 := decide (v3.toNat < v4.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7
  else
    let v8 := v0.base
    let v9 ← checkedSub v3 v8
    let v10 ← downcast_truncate_u256_u64 v9
    let v11 := v10
    b3 v11

def len_CallData (v0 : CallData) : FeM UInt256 := do
  let v1 ← CallData_size v0
  let v2 := v1
  return v2

def i256_from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def Sol_validate_runtime_head (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := v2
  let v4 := (UInt256.ofNat 0)
  let v5 := decide (v3 ≠ v4)
  let b3 : FeM Unit := do
    return ()
  if v5 then
    let v6 := v0
    let v7 := v1
    let v8 := v6 - v7
    let v9 ← i256_from_word v8
    let v10 := v9
    let v11 ← i256_from_word v2
    let v12 := UInt256.sltBool v10 v11
    if v12 then
      let v13 ← revertEmpty
      nomatch v13
    else
      let v14 := ()
      let v15 := v14
      let v16 := v15
      b3
  else
    let v17 := ()
    let v18 := v17
    b3

def u256_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def u256_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def u64_max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def u64_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def usize_IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def usize_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  let v2 := v1
  return v2

def usize_from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def I256_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 57896044618658097711785492504343953926634992332820282019728792003956564819968)
  return v0

def usize_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def downcast_saturate_u64_usize (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let v8 ← u64_to_word v0
  let v9 := v8
  let v10 ← usize_IS_SIGNED
  let v11 := !v10
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun r0 v1 => do
    let mut r0 := r0
    return v1
  if v11 then
    let v12 := v9
    r0 := v12
    let v13 ← u64_IS_SIGNED
    let b6 : UInt256 → FeM UInt256 := fun r0 => do
      let mut r0 := r0
      let v24 := r0
      let v25 := v24
      let v26 ← usize_MASK
      let v27 := decide (v26.toNat < v25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 v2 => do
        let mut r0 := r0
        let v35 := v2
        b3 r0 v35
      if v27 then
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        let v30 := v29
        b10 r0 v30
      else
        let v31 := r0
        let v32 := v31
        let v33 ← usize_from_word v32
        let v34 := v33
        b10 r0 v34
    let b5 : UInt256 → FeM UInt256 := fun r0 => do
      let mut r0 := r0
      let v22 := ()
      let v23 := v22
      b6 r0
    if v13 then
      let v14 ← I256_SIGN_BIT
      let v15 := v9 &&& v14
      let v16 := (UInt256.ofNat 0)
      let v17 := decide (v15 ≠ v16)
      if v17 then
        let v18 := (UInt256.ofNat 0)
        let v19 := v18
        r0 := v19
        let v20 := ()
        let v21 := v20
        b6 r0
      else
        b5 r0
    else
      b5 r0
  else
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let v39 ← sign_extend v36 v37 v38
    let v40 := v39
    let v41 ← usize_SIGN_BIT
    let v42 := (UInt256.ofNat 1)
    let v43 ← checkedSub v41 v42
    let v44 := v43
    let v45 := v40
    let v46 := v45
    let v47 := v44
    let v48 := v47
    let v49 ← u64_IS_SIGNED
    let b13 : UInt256 → UInt256 → FeM UInt256 := fun r0 v3 => do
      let mut r0 := r0
      let v77 := v3
      b3 r0 v77
    if v49 then
      let v50 := v9
      let v51 := v50
      let v52 := UInt256.sltBool v51 v46
      let b16 : UInt256 → UInt256 → FeM UInt256 := fun r0 v4 => do
        let mut r0 := r0
        let v61 := v4
        b13 r0 v61
      if v52 then
        let v53 ← usize_from_word v40
        let v54 := v53
        b16 r0 v54
      else
        let v55 := UInt256.sgtBool v51 v48
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun r0 v5 => do
          let mut r0 := r0
          let v60 := v5
          b16 r0 v60
        if v55 then
          let v56 ← usize_from_word v44
          let v57 := v56
          b19 r0 v57
        else
          let v58 ← usize_from_word v9
          let v59 := v58
          b19 r0 v59
    else
      let v62 ← I256_SIGN_BIT
      let v63 := v9 &&& v62
      let v64 := (UInt256.ofNat 0)
      let v65 := decide (v63 ≠ v64)
      let b22 : UInt256 → UInt256 → FeM UInt256 := fun r0 v6 => do
        let mut r0 := r0
        let v76 := v6
        b13 r0 v76
      if v65 then
        let v66 ← usize_from_word v44
        let v67 := v66
        b22 r0 v67
      else
        let v68 := v9
        let v69 := v68
        let v70 := UInt256.sgtBool v69 v48
        let b25 : UInt256 → UInt256 → FeM UInt256 := fun r0 v7 => do
          let mut r0 := r0
          let v75 := v7
          b22 r0 v75
        if v70 then
          let v71 ← usize_from_word v44
          let v72 := v71
          b25 r0 v72
        else
          let v73 ← usize_from_word v9
          let v74 := v73
          b25 r0 v74

def MAX_ALLOCATION : FeM UInt256 := do
  let v0 ← u64_max
  let v1 ← downcast_saturate_u64_usize v0
  return v1

def PANIC_OUT_OF_MEMORY : FeM UInt256 := do
  let v0 := (UInt256.ofNat 65)
  return v0

def alloc_bytes (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 ← MAX_ALLOCATION
  let v3 := decide (v2.toNat < v1.toNat)
  if v3 then
    let v4 ← PANIC_OUT_OF_MEMORY
    let v5 ← panicCode v4
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← alloc v0
    return v8

def FixedMemBuffer_alloc_uninit_32 : FeM FixedMemBuffer_32 := do
  let v0 := (UInt256.ofNat 32)
  let v1 ← alloc_bytes v0
  let v2 := (FixedMemBuffer_32.mk v1)
  return v2

def zero_bytes (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 ← zeroMem v0 v1
  return ()

def FixedMemBuffer_alloc_32 : FeM FixedMemBuffer_32 := do
  let v0 ← FixedMemBuffer_alloc_uninit_32
  let v1 := v0
  let v2 := v1.ptr
  let v3 := (UInt256.ofNat 32)
  let v4 ← zero_bytes v2 v3
  return v1

def FixedMemBuffer_ptr_32 (v0 : FixedMemBuffer_32) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def AbiFrame_from_raw_parts (v0 : UInt256) (v1 : UInt256) : FeM AbiFrame := do
  let v2 := v0
  let v3 := v1
  let v4 := (AbiFrame.mk v2 v3)
  return v4

def PANIC_INDEX_OUT_OF_BOUNDS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 50)
  return v0

def index_out_of_bounds : FeM Empty := do
  let v0 ← PANIC_INDEX_OUT_OF_BOUNDS
  let v1 ← panicCode v0
  nomatch v1

def word_to_usize (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def ptr_offset (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize v1
  let v3 := v0 + v2
  return v3

def AbiFrame_slice (v0 : AbiFrame) (v1 : UInt256) (v2 : UInt256) : FeM AbiFrame := do
  let v3 := v1
  let v4 := v0.len
  let v5 := decide (v4.toNat < v3.toNat)
  let b1 : FeM AbiFrame := do
    let v17 ← index_out_of_bounds
    nomatch v17
  if v5 then
    b1
  else
    let v6 := v2
    let v7 := v0.len
    let v8 := v1
    let v9 := v7 - v8
    let v10 := decide (v9.toNat < v6.toNat)
    if v10 then
      b1
    else
      let v11 := ()
      let v12 := v11
      let v13 := v0.ptr
      let v14 ← ptr_offset v13 v1
      let v15 := v2
      let v16 := (AbiFrame.mk v14 v15)
      return v16

def cast_u8_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def store_word (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 := v1
  let v3 ← cast_u8_u256 v0
  mstore v3 v2
  let v4 := ()
  return ()

def AbiFrame_store_word (v0 : AbiFrame) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := (UInt256.ofNat 32)
  let v4 ← AbiFrame_slice v0 v1 v3
  let v5 := v4
  let v6 := v5.ptr
  let v7 ← store_word v6 v2
  return ()

def u256_encode_Sol (v0 : UInt256) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← AbiFrame_store_word v1 v2 v0
  return ()

def downcast_unchecked_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let v1 ← u256_to_word v0
  let v2 := v1
  let v3 ← usize_from_word v2
  return v3

def MemSlice_from_raw_parts_ref_u8 (v0 : UInt256) (v1 : UInt256) : FeM MemSlice_u8 := do
  let mut r0 : MemSlice_u8 := default
  let v2 := (MemSlice_u8.mk v0 v1)
  let v3 := v2
  r0 := v3
  return r0

def MemSlice_ptr_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def MemSlice_len_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := v0.len
  return v1

def Evm_return_data (v1 : MemSlice_u8) : FeM Empty := do
  let v2 ← MemSlice_ptr_u8 v1
  let v3 ← MemSlice_len_u8 v1
  let v4 := v3
  let v5 ← returnData v2 v4
  nomatch v5

def FixedMemBuffer_alloc_uninit_64 : FeM FixedMemBuffer_64 := do
  let v0 := (UInt256.ofNat 64)
  let v1 ← alloc_bytes v0
  let v2 := (FixedMemBuffer_64.mk v1)
  return v2

def FixedMemBuffer_alloc_64 : FeM FixedMemBuffer_64 := do
  let v0 ← FixedMemBuffer_alloc_uninit_64
  let v1 := v0
  let v2 := v1.ptr
  let v3 := (UInt256.ofNat 64)
  let v4 ← zero_bytes v2 v3
  return v1

def FixedMemBuffer_ptr_64 (v0 : FixedMemBuffer_64) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def MemBuffer_new_uninit (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes v0
  let v2 := v0
  let v3 := v0
  let v4 := (MemBuffer.mk v1 v2 v3)
  return v4

def MemBuffer_new (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← MemBuffer_new_uninit v0
  let v2 := v1
  let v3 := v2.ptr
  let v4 ← zero_bytes v3 v0
  return v2

def MemBuffer_ptr (v0 : MemBuffer) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def MemBuffer_span_ref (v0 : MemBuffer) : FeM MemSlice_u8 := do
  let v1 := v0.ptr
  let v2 := v0.len
  let v3 ← MemSlice_from_raw_parts_ref_u8 v1 v2
  return v3

def const_u256 : FeM Bool := do
  let v0 := false
  return v0

def payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← u256_HEAD_SIZE
  return v1

def abi_payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 v0
  return v1

def usize_max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  return v1

def allocation_len (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 ← usize_max
  let v3 := v2
  let v4 := decide (v3.toNat < v1.toNat)
  if v4 then
    let v5 ← PANIC_OUT_OF_MEMORY
    let v6 ← panicCode v5
    nomatch v6
  else
    let v7 := ()
    let v8 := v7
    let v9 ← word_to_usize v0
    return v9

def encode_alloc_Sol_u256 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_u256 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_ptr r0
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r1 := v8
  let v10 ← u256_encode_Sol v0 r1
  let v11 := r0
  return v11

def abi_field_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 v0
  let v2 := v1
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    return v2

def abi_single_root_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_u256 v0
  return v1

def abi_field_head_size_u256 : FeM UInt256 := do
  let v0 ← u256_IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← u256_HEAD_SIZE
    return v4

def abi_single_root_head_size_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

def AbiFrame_tail (v0 : AbiFrame) (v1 : UInt256) : FeM AbiFrame := do
  let v2 := v1
  let v3 := v0.len
  let v4 := decide (v3.toNat < v2.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := v0.ptr
    let v9 ← ptr_offset v8 v1
    let v10 := v0.len
    let v11 := v1
    let v12 := v10 - v11
    let v13 := (AbiFrame.mk v9 v12)
    return v13

def AbiFrame_len (v0 : AbiFrame) : FeM UInt256 := do
  let v1 := v0.len
  return v1

def encode_field_Sol_u256 (v0 : UInt256) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_u256 v0
    let v5 := v4
    let v6 ← AbiFrame_len v1
    let v8 ← AbiFrame_len r0
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice r0 v11 v5
    r1 := v13
    let v15 ← u256_encode_Sol v0 r1
    let v17 ← AbiFrame_tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← u256_HEAD_SIZE
    let v21 ← AbiFrame_slice v1 v2 v20
    r2 := v21
    let v23 ← u256_encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_u256 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_single_root_size_u256 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_ptr r0
  let v8 ← AbiFrame_from_raw_parts v7 v2
  let v9 := v8
  r1 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_u256
  let v13 ← AbiFrame_tail r1 v11
  let v15 ← encode_field_Sol_u256 v0 r1 v10 v13
  let v16 := r0
  return v16

def encode_return_alloc_Sol_u256 (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← const_u256
  if v1 then
    let v2 ← encode_alloc_Sol_u256 v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_u256 v0
    return v5

def Evm_return_bytes (v1 : MemSlice_u8) : FeM Empty := do
  let v2 ← Evm_return_data v1
  nomatch v2

def Evm_return_value_Sol_u256 (v1 : UInt256) : FeM Empty := do
  let mut r0 : FixedMemBuffer_32 := default
  let mut r1 : FixedMemBuffer_64 := default
  let mut r2 : MemBuffer := default
  let mut r3 : MemBuffer := default
  let mut r4 : AbiFrame := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let v2 ← u256_IS_DYNAMIC
  let v3 := !v2
  if v3 then
    let v4 ← u256_HEAD_SIZE
    let v5 := (UInt256.ofNat 32)
    let v6 := decide (v4.toNat ≤ v5.toNat)
    if v6 then
      let v7 ← FixedMemBuffer_alloc_32
      let v8 := v7
      r0 := v8
      let v10 ← FixedMemBuffer_ptr_32 r0
      let v11 ← u256_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let v14 ← u256_encode_Sol v1 r4
      let v16 ← FixedMemBuffer_ptr_32 r0
      let v17 ← u256_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← u256_HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let v27 := v26
        r1 := v27
        let v29 ← FixedMemBuffer_ptr_64 r1
        let v30 ← u256_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let v33 ← u256_encode_Sol v1 r5
        let v35 ← FixedMemBuffer_ptr_64 r1
        let v36 ← u256_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← u256_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← MemBuffer_new v43
        let v45 := v44
        r2 := v45
        let v47 ← MemBuffer_ptr r2
        let v48 ← u256_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let v51 ← u256_encode_Sol v1 r6
        let v53 ← MemBuffer_span_ref r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_u256 v1
    let v58 := v57
    r3 := v58
    let v60 ← MemBuffer_span_ref r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

def CallData_word_at (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0.base
  let v3 := v1
  let v4 := v2 + v3
  let v5 ← calldataload v4
  return v5

def Sol_decode_error : FeM Empty := do
  let v0 ← revertEmpty
  nomatch v0

def checked_tail_Sol (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  let v5 := ~~~v4
  let v6 := v0
  let v7 := v5 - v6
  let v8 := decide (v7.toNat < v2.toNat)
  if v8 then
    let v9 ← Sol_decode_error
    nomatch v9
  else
    let v10 := ()
    let v11 := v10
    let v12 := v0
    let v13 := v1
    let v14 := v12 + v13
    return v14

def checked_frame_end_Sol (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 := v0
  let v4 := v2
  let v5 := decide (v4.toNat < v3.toNat)
  let b1 : FeM UInt256 := do
    let v16 ← Sol_decode_error
    nomatch v16
  if v5 then
    b1
  else
    let v6 := v1
    let v7 := v2
    let v8 := v0
    let v9 := v7 - v8
    let v10 := decide (v9.toNat < v6.toNat)
    if v10 then
      b1
    else
      let v11 := ()
      let v12 := v11
      let v13 := v0
      let v14 := v1
      let v15 := v13 + v14
      return v15

def u256_decode_from_Sol_CallData (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← CallData_word_at v0 v1
  return v2

def decode_from_bounded_u256_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← u256_decode_from_Sol_CallData v0 v1
  return v5

def decode_field_from_prechecked_head_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← CallData_word_at v0 v2
    let v6 := v5
    let v7 ← checked_tail_Sol v1 v6
    let v8 := v7
    let v9 ← decode_from_bounded_u256_Sol_CallData v0 v8 v3
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← u256_decode_from_Sol_CallData v0 v2
    return v12

def decode_field_from_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v4 ← CallData_word_at v0 v2
    let v5 := v4
    let v6 ← checked_tail_Sol v1 v5
    let v7 ← u256_decode_from_Sol_CallData v0 v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← u256_decode_from_Sol_CallData v0 v2
    return v10

def decode_msg_field_from_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_u256_CallData v0 v1 v2 v3
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_u256_CallData v0 v1 v2
    return v8

def SolDecoder_with_base (v0 : UInt256) : FeM SolDecoder := do
  let v1 := v0
  let v2 := v0
  let v3 := (SolDecoder.mk v1 v2)
  return v3

def Sol_decoder_new : FeM SolDecoder := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← SolDecoder_with_base v0
  return v1

def usize_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def downcast_truncate_usize_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← usize_to_word v0
  let v3 := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := v3 &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def MemSlice_u8_size (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_len_u8 v0
  let v2 ← downcast_truncate_usize_u64 v1
  return v2

def len_MemSlice_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_u8_size v0
  let v2 := v1
  return v2

def MemSlice_u8_word_at (v0 : MemSlice_u8) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_len_u8 v0
  let v3 := v2
  let v4 := v3
  let v5 := v1
  let v6 := decide (v4.toNat ≤ v5.toNat)
  if v6 then
    let v7 := (UInt256.ofNat 0)
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 := v1
    let v11 := v4 - v10
    let v12 ← word_to_usize v11
    let v13 := v12
    let v14 ← MemSlice_ptr_u8 v0
    let v15 ← word_to_usize v1
    let v16 := v14 + v15
    let v17 ← readPaddedWord v16 v13
    return v17

def SolDecoder_read_word_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let v2 := v0.pos
  let v3 := (UInt256.ofNat 32)
  let v4 ← checkedAdd v2 v3
  let v5 := v4
  let v6 := v0.pos
  let v7 := decide (v5.toNat < v6.toNat)
  let b1 : SolDecoder → FeM (UInt256 × SolDecoder) := fun v0 => do
    let mut v0 := v0
    let v16 ← revertEmpty
    nomatch v16
  if v7 then
    b1 v0
  else
    let v8 ← len_MemSlice_u8 v1
    let v9 := decide (v8.toNat < v5.toNat)
    if v9 then
      b1 v0
    else
      let v10 := ()
      let v11 := v10
      let v12 := v0.pos
      let v13 ← MemSlice_u8_word_at v1 v12
      let v14 := v13
      v0 := { v0 with pos := v5 }
      let v15 := ()
      return (v14, v0)

def SolDecoder_base (v0 : SolDecoder) : FeM UInt256 := do
  let v1 := v0.base
  return v1

def SolDecoder_pos (v0 : SolDecoder) : FeM UInt256 := do
  let v1 := v0.pos
  return v1

def SolDecoder_set_base (v0 : SolDecoder) (v1 : UInt256) : FeM (Unit × SolDecoder) := do
  let mut v0 := v0
  let v2 := v1
  v0 := { v0 with base := v2 }
  let v3 := ()
  return ((), v0)

def SolDecoder_set_pos (v0 : SolDecoder) (v1 : UInt256) : FeM (Unit × SolDecoder) := do
  let mut v0 := v0
  let v2 := v1
  v0 := { v0 with pos := v2 }
  let v3 := ()
  return ((), v0)

def u256_decode_payload_Sol_SolDecoder_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let (v2, v2_0) ← SolDecoder_read_word_MemSlice_u8 v0 v1
  v0 := v2_0
  return (v2, v0)

def decode_field_Sol_u256_SolDecoder_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let v2 ← u256_IS_DYNAMIC
  if v2 then
    let (v3, v3_0) ← SolDecoder_read_word_MemSlice_u8 v0 v1
    v0 := v3_0
    let v4 := v3
    let v6 ← SolDecoder_base v0
    let v7 := v6
    let v9 ← SolDecoder_pos v0
    let v10 := v9
    let v11 ← checked_tail_Sol v7 v4
    let (v12, v12_0) ← SolDecoder_set_base v0 v11
    v0 := v12_0
    let v14 ← SolDecoder_base v0
    let (v15, v15_0) ← SolDecoder_set_pos v0 v14
    v0 := v15_0
    let (v17, v17_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 v0 v1
    v0 := v17_0
    let v18 := v17
    let (v19, v19_0) ← SolDecoder_set_base v0 v7
    v0 := v19_0
    let (v20, v20_0) ← SolDecoder_set_pos v0 v10
    v0 := v20_0
    return (v18, v0)
  else
    let v21 := ()
    let v22 := v21
    let (v24, v24_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 v0 v1
    v0 := v24_0
    return (v24, v0)

def tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let (v3, v3_0) ← decode_field_Sol_u256_SolDecoder_MemSlice_u8 v0 v1
  v0 := v3_0
  let v4 := v3
  let v5 := v4
  return (v5, v0)

def Layout.StoragePackedArray_8 : Layout :=
  { storage := [0] }

structure Location where
  slot : UInt256
  offset : UInt256
deriving Inhabited

def StoragePackedArray_LANES_PER_SLOT_8 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 256)
  let v1 := (UInt256.ofNat 8)
  let v2 ← checkedDiv v0 v1
  return v2

def u256_encoded_len (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def u256_IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def downcast_saturate_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let v8 ← u256_to_word v0
  let v9 := v8
  let v10 ← usize_IS_SIGNED
  let v11 := !v10
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun r0 v1 => do
    let mut r0 := r0
    return v1
  if v11 then
    let v12 := v9
    r0 := v12
    let v13 ← u256_IS_SIGNED
    let b6 : UInt256 → FeM UInt256 := fun r0 => do
      let mut r0 := r0
      let v24 := r0
      let v25 := v24
      let v26 ← usize_MASK
      let v27 := decide (v26.toNat < v25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 v2 => do
        let mut r0 := r0
        let v35 := v2
        b3 r0 v35
      if v27 then
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        let v30 := v29
        b10 r0 v30
      else
        let v31 := r0
        let v32 := v31
        let v33 ← usize_from_word v32
        let v34 := v33
        b10 r0 v34
    let b5 : UInt256 → FeM UInt256 := fun r0 => do
      let mut r0 := r0
      let v22 := ()
      let v23 := v22
      b6 r0
    if v13 then
      let v14 ← I256_SIGN_BIT
      let v15 := v9 &&& v14
      let v16 := (UInt256.ofNat 0)
      let v17 := decide (v15 ≠ v16)
      if v17 then
        let v18 := (UInt256.ofNat 0)
        let v19 := v18
        r0 := v19
        let v20 := ()
        let v21 := v20
        b6 r0
      else
        b5 r0
    else
      b5 r0
  else
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let v39 ← sign_extend v36 v37 v38
    let v40 := v39
    let v41 ← usize_SIGN_BIT
    let v42 := (UInt256.ofNat 1)
    let v43 ← checkedSub v41 v42
    let v44 := v43
    let v45 := v40
    let v46 := v45
    let v47 := v44
    let v48 := v47
    let v49 ← u256_IS_SIGNED
    let b13 : UInt256 → UInt256 → FeM UInt256 := fun r0 v3 => do
      let mut r0 := r0
      let v77 := v3
      b3 r0 v77
    if v49 then
      let v50 := v9
      let v51 := v50
      let v52 := UInt256.sltBool v51 v46
      let b16 : UInt256 → UInt256 → FeM UInt256 := fun r0 v4 => do
        let mut r0 := r0
        let v61 := v4
        b13 r0 v61
      if v52 then
        let v53 ← usize_from_word v40
        let v54 := v53
        b16 r0 v54
      else
        let v55 := UInt256.sgtBool v51 v48
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun r0 v5 => do
          let mut r0 := r0
          let v60 := v5
          b16 r0 v60
        if v55 then
          let v56 ← usize_from_word v44
          let v57 := v56
          b19 r0 v57
        else
          let v58 ← usize_from_word v9
          let v59 := v58
          b19 r0 v59
    else
      let v62 ← I256_SIGN_BIT
      let v63 := v9 &&& v62
      let v64 := (UInt256.ofNat 0)
      let v65 := decide (v63 ≠ v64)
      let b22 : UInt256 → UInt256 → FeM UInt256 := fun r0 v6 => do
        let mut r0 := r0
        let v76 := v6
        b13 r0 v76
      if v65 then
        let v66 ← usize_from_word v44
        let v67 := v66
        b22 r0 v67
      else
        let v68 := v9
        let v69 := v68
        let v70 := UInt256.sgtBool v69 v48
        let b25 : UInt256 → UInt256 → FeM UInt256 := fun r0 v7 => do
          let mut r0 := r0
          let v75 := v7
          b22 r0 v75
        if v70 then
          let v71 ← usize_from_word v44
          let v72 := v71
          b25 r0 v72
        else
          let v73 ← usize_from_word v9
          let v74 := v73
          b25 r0 v74

def u256_write_key (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 ← mstore v0 v1
  return ()

def keccak256_1 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_ptr_u8 v0
  let v2 ← MemSlice_len_u8 v0
  let v3 := v2
  let v4 ← keccak256 v1 v3
  return v4

def cast_u8_u8 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def dealloc_u8 (v0 : UInt256) : FeM Unit := do
  let v1 ← cast_u8_u8 v0
  let v2 := ()
  return ()

def MemBuffer_drop (v0 : MemBuffer) : FeM Unit := do
  let v1 := v0.ptr
  let v2 ← dealloc_u8 v1
  return ()

def mapping_slot_u256 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v2 ← u256_encoded_len v1
  let v3 := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd v3 v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let v8 := v7
  r0 := v8
  let v10 ← MemBuffer_ptr r0
  let v11 := v10
  let v12 ← u256_write_key v11 v1
  let v13 ← downcast_unchecked_u256_usize v3
  let v14 := v11 + v13
  let v15 ← mstore v14 v0
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd v3 v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 v11 v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop r0
  return v21

def StorageMap_locate_u256_u256 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Location := do
  let v3 ← mapping_slot_u256 v1 v2
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def StoragePackedArray_LANE_MASK_8 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := (UInt256.ofNat 8)
  let v2 := v0 <<< v1
  let v3 := (UInt256.ofNat 1)
  let v4 ← checkedSub v2 v3
  return v4

def StoragePackedArray_get_8 (v0 : Place) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 ← StoragePackedArray_LANES_PER_SLOT_8
  let v4 ← checkedRem v2 v3
  let v5 := (UInt256.ofNat 8)
  let v6 := v4 * v5
  let v7 := v6
  let v8 := v1
  let v9 ← StoragePackedArray_LANES_PER_SLOT_8
  let v10 ← checkedDiv v8 v9
  let l0 ← StorageMap_locate_u256_u256 (v0.field Layout.StoragePackedArray_8 0) (v0.field Layout.StoragePackedArray_8 0).slot v10
  let v11 ← load (Place.mk .storage l0.slot)
  let v12 := v11 >>> v7
  let v13 ← StoragePackedArray_LANE_MASK_8
  let v14 := v12 &&& v13
  return v14

def StoragePackedArray_set_8 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := v1
  let v4 ← StoragePackedArray_LANES_PER_SLOT_8
  let v5 ← checkedRem v3 v4
  let v6 := (UInt256.ofNat 8)
  let v7 := v5 * v6
  let v8 := v7
  let v9 ← StoragePackedArray_LANE_MASK_8
  let v10 := v9 <<< v8
  let v11 := v10
  let v12 := v1
  let v13 ← StoragePackedArray_LANES_PER_SLOT_8
  let v14 ← checkedDiv v12 v13
  let l0 ← StorageMap_locate_u256_u256 (v0.field Layout.StoragePackedArray_8 0) (v0.field Layout.StoragePackedArray_8 0).slot v14
  let v15 := (Place.mk .storage l0.slot)
  let v16 := v15
  let v17 ← load v16
  let v18 ← U256_MASK
  let v19 := v18 ^^^ v11
  let v20 := v17 &&& v19
  let v21 := v2
  let v22 ← StoragePackedArray_LANE_MASK_8
  let v23 := v21 &&& v22
  let v24 := v23 <<< v8
  let v25 := v20 ||| v24
  store v16 v25
  let v26 := ()
  return ()

def Evm_caller : FeM UInt256 := do
  let v1 ← caller
  let v2 := v1
  return v2

def Address_encoded_len (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def Address_write_key (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 := v1
  let v3 ← u256_write_key v0 v2
  return ()

def mapping_slot_Address (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v2 ← Address_encoded_len v1
  let v3 := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd v3 v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let v8 := v7
  r0 := v8
  let v10 ← MemBuffer_ptr r0
  let v11 := v10
  let v12 ← Address_write_key v11 v1
  let v13 ← downcast_unchecked_u256_usize v3
  let v14 := v11 + v13
  let v15 ← mstore v14 v0
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd v3 v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 v11 v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop r0
  return v21

def StorageMap_locate_Address_u256 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Location := do
  let v3 ← mapping_slot_Address v1 v2
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def Address_zero : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := v0
  return v1

def tuple_HEAD_SIZE_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

def dynamic_payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← u256_IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_u256 v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 := (UInt256.ofNat 0)
    return v5

def tuple_payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE_u256
  let v2 := v0
  let v3 ← dynamic_payload_size_u256 v2
  let v4 := v1 + v3
  return v4

def abi_payload_size_tuple_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← tuple_payload_size_u256 v0
  return v1

def tuple_encode_Sol_u256 (v0 : UInt256) (v1 : AbiFrame) : FeM Unit := do
  let v3 := (UInt256.ofNat 0)
  let v4 ← tuple_HEAD_SIZE_u256
  let v5 ← AbiFrame_tail v1 v4
  let v6 := v0
  let v7 ← encode_field_Sol_u256 v6 v1 v3 v5
  return ()

def encode_alloc_Sol_tuple_u256 (r0 : UInt256) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_tuple_u256 r0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r2 := v8
  let v11 ← tuple_encode_Sol_u256 r0 r2
  let v12 := r1
  return v12

def encode_abi_payload_tuple_u256 (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_tuple_u256 v0
  return v1

def Address_as_topic (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def Evm_log3 (v1 : MemSlice_u8) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) : FeM Unit := do
  let v5 ← MemSlice_ptr_u8 v1
  let v6 ← MemSlice_len_u8 v1
  let v7 := v6
  let v8 ← logData v5 v7 #[v2, v3, v4]
  return ()

def StorageMap_locate_Address_StorageMap_Address_u256 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Location := do
  let v3 ← mapping_slot_Address v1 v2
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def Address_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def Address_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def abi_field_head_size_Address : FeM UInt256 := do
  let v0 ← Address_IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← Address_HEAD_SIZE
    return v4

def Address_decode_from_Sol_CallData (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← CallData_word_at v0 v1
  let v3 := v2
  let v4 := (UInt256.ofNat 160)
  let v5 := v3 >>> v4
  let v6 := (UInt256.ofNat 0)
  let v7 := decide (v5 ≠ v6)
  if v7 then
    let v8 ← revertEmpty
    nomatch v8
  else
    let v9 := ()
    let v10 := v9
    let v11 := v3
    return v11

def decode_from_bounded_Address_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← Address_HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← Address_decode_from_Sol_CallData v0 v1
  return v5

def decode_field_from_prechecked_head_Sol_Address_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    let v5 ← CallData_word_at v0 v2
    let v6 := v5
    let v7 ← checked_tail_Sol v1 v6
    let v8 := v7
    let v9 ← decode_from_bounded_Address_Sol_CallData v0 v8 v3
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← Address_decode_from_Sol_CallData v0 v2
    return v12

def decode_field_from_Sol_Address_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v4 ← CallData_word_at v0 v2
    let v5 := v4
    let v6 ← checked_tail_Sol v1 v5
    let v7 ← Address_decode_from_Sol_CallData v0 v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← Address_decode_from_Sol_CallData v0 v2
    return v10

def decode_msg_field_from_Sol_Address_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_Address_CallData v0 v1 v2 v3
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_Address_CallData v0 v1 v2
    return v8

def bool_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def bool_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def bool_encode_Sol (v0 : Bool) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := v0
  let v4 := Bool.toUInt256 v3
  let v5 ← AbiFrame_store_word v1 v2 v4
  return ()

def const_bool : FeM Bool := do
  let v0 := false
  return v0

def payload_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← bool_HEAD_SIZE
  return v1

def abi_payload_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← payload_size_bool v0
  return v1

def encode_alloc_Sol_bool (v0 : Bool) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_bool v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_ptr r0
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r1 := v8
  let v10 ← bool_encode_Sol v0 r1
  let v11 := r0
  return v11

def abi_field_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← payload_size_bool v0
  let v2 := v1
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    return v2

def abi_single_root_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← abi_field_size_bool v0
  return v1

def abi_field_head_size_bool : FeM UInt256 := do
  let v0 ← bool_IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← bool_HEAD_SIZE
    return v4

def abi_single_root_head_size_bool : FeM UInt256 := do
  let v0 ← abi_field_head_size_bool
  return v0

def encode_field_Sol_bool (v0 : Bool) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_bool v0
    let v5 := v4
    let v6 ← AbiFrame_len v1
    let v8 ← AbiFrame_len r0
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice r0 v11 v5
    r1 := v13
    let v15 ← bool_encode_Sol v0 r1
    let v17 ← AbiFrame_tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← bool_HEAD_SIZE
    let v21 ← AbiFrame_slice v1 v2 v20
    r2 := v21
    let v23 ← bool_encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_bool (v0 : Bool) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_single_root_size_bool v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_ptr r0
  let v8 ← AbiFrame_from_raw_parts v7 v2
  let v9 := v8
  r1 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_bool
  let v13 ← AbiFrame_tail r1 v11
  let v15 ← encode_field_Sol_bool v0 r1 v10 v13
  let v16 := r0
  return v16

def encode_return_alloc_Sol_bool (v0 : Bool) : FeM MemBuffer := do
  let v1 ← const_bool
  if v1 then
    let v2 ← encode_alloc_Sol_bool v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_bool v0
    return v5

def Evm_return_value_Sol_bool (v1 : Bool) : FeM Empty := do
  let mut r0 : FixedMemBuffer_32 := default
  let mut r1 : FixedMemBuffer_64 := default
  let mut r2 : MemBuffer := default
  let mut r3 : MemBuffer := default
  let mut r4 : AbiFrame := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let v2 ← bool_IS_DYNAMIC
  let v3 := !v2
  if v3 then
    let v4 ← bool_HEAD_SIZE
    let v5 := (UInt256.ofNat 32)
    let v6 := decide (v4.toNat ≤ v5.toNat)
    if v6 then
      let v7 ← FixedMemBuffer_alloc_32
      let v8 := v7
      r0 := v8
      let v10 ← FixedMemBuffer_ptr_32 r0
      let v11 ← bool_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let v14 ← bool_encode_Sol v1 r4
      let v16 ← FixedMemBuffer_ptr_32 r0
      let v17 ← bool_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← bool_HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let v27 := v26
        r1 := v27
        let v29 ← FixedMemBuffer_ptr_64 r1
        let v30 ← bool_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let v33 ← bool_encode_Sol v1 r5
        let v35 ← FixedMemBuffer_ptr_64 r1
        let v36 ← bool_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← bool_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← MemBuffer_new v43
        let v45 := v44
        r2 := v45
        let v47 ← MemBuffer_ptr r2
        let v48 ← bool_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let v51 ← bool_encode_Sol v1 r6
        let v53 ← MemBuffer_span_ref r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_bool v1
    let v58 := v57
    r3 := v58
    let v60 ← MemBuffer_span_ref r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

def tuple_HEAD_SIZE_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def tuple_decode_from_prechecked_head_Sol_u256_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let mut r0 : UInt256 := default
  let v3 := v1
  let v4 := v3
  r0 := v4
  let v5 := r0
  let v6 := v5
  let v7 ← decode_field_from_prechecked_head_Sol_u256_CallData v0 v1 v6 v2
  let v8 := v7
  let v9 ← abi_field_head_size_u256
  let v10 := r0
  let v11 ← checkedAdd v10 v9
  let v12 := v11
  r0 := v12
  let v13 := ()
  let v14 := r0
  let v15 := v14
  let v16 ← decode_field_from_prechecked_head_Sol_u256_CallData v0 v1 v15 v2
  let v17 := v16
  let v18 := (v8, v17)
  return v18

def decode_frame_from_Sol_tuple_u256_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← tuple_HEAD_SIZE_u256_u256
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← tuple_decode_from_prechecked_head_Sol_u256_u256_CallData v0 v1 v2
  return v5

def decode_input_at_tuple_u256_u256_CallData (v0 : CallData) (v1 : UInt256) : FeM (UInt256 × UInt256) := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_frame_from_Sol_tuple_u256_u256_CallData v0 v1 v3
  return v4

def Evm_decode_args_tuple_u256_u256 : FeM (UInt256 × UInt256) := do
  let mut r0 : CallData := default
  let v1 ← Evm_input
  r0 := v1
  let v2 := (UInt256.ofNat 4)
  let v3 := v2
  let v4 := r0
  let v5 ← decode_input_at_tuple_u256_u256_CallData v4 v3
  return v5

def MemBuffer_len (v0 : MemBuffer) : FeM UInt256 := do
  let v1 := v0.len
  return v1

def revert_u256 (v0 : UInt256) : FeM Empty := do
  let mut r0 : MemBuffer := default
  let v1 ← encode_single_root_alloc_Sol_u256 v0
  let v2 := v1
  r0 := v2
  let v4 ← MemBuffer_ptr r0
  let v6 ← MemBuffer_len r0
  let v7 := v6
  let v8 ← revertData v4 v7
  nomatch v8

structure Wei where
  amount : UInt256
deriving Inhabited

def i256_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def i8_IS_SIGNED : FeM Bool := do
  let v0 := true
  return v0

def U8_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 255)
  return v0

def i8_MASK : FeM UInt256 := do
  let v0 ← U8_MASK
  return v0

def I8_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 128)
  return v0

def i8_SIGN_BIT : FeM UInt256 := do
  let v0 ← I8_SIGN_BIT
  return v0

def i8_from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 8) v0
  return v1

def downcast_truncate_i256_i8 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← i256_to_word v0
  let v3 := v2
  let v4 ← i8_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i8_MASK
    let v6 ← i8_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← i8_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← i8_MASK
    let v11 := v3 &&& v10
    let v12 ← i8_from_word v11
    let v13 := v12
    b3 v13

def downcast_unchecked_u256_i256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← u256_to_word v0
  let v2 := v1
  let v3 ← i256_from_word v2
  return v3

def Evm_address : FeM UInt256 := do
  let v1 ← selfAddress
  let v2 := v1
  return v2

def Address_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def wei (v0 : UInt256) : FeM Wei := do
  let v1 := v0
  let v2 := (Wei.mk v1)
  return v2

def Evm_value : FeM Wei := do
  let v1 ← callvalue
  let v2 ← wei v1
  return v2

def Wei_as_wei (v0 : Wei) : FeM UInt256 := do
  let v1 := v0.amount
  return v1

def Evm_origin : FeM UInt256 := do
  let v1 ← origin
  let v2 := v1
  return v2

def Evm_block_number : FeM UInt256 := do
  let v1 ← blockNumber
  return v1

inductive Option_u256 where
  | None
  | Some (f0 : UInt256)
deriving Inhabited

def Option_u256.tag : Option_u256 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_u256.Some_0 : Option_u256 → UInt256
  | .Some f0 => f0
  | _ => default

def Option_u256.set_Some_0 (x : Option_u256) (v : UInt256) : Option_u256 :=
  match x with
  | .Some f0 => .Some v
  | x => x

def Layout.TSlot_bool : Layout :=
  { storage := [0] }

def tuple_encoded_len_u256_u256 (v0 : (UInt256 × UInt256)) : FeM UInt256 := do
  let v1 := v0.fst
  let v2 ← u256_encoded_len v1
  let v3 := v0.snd
  let v4 ← u256_encoded_len v3
  let v5 ← checkedAdd v2 v4
  return v5

def tuple_write_key_u256_u256 (v0 : UInt256) (v1 : (UInt256 × UInt256)) : FeM Unit := do
  let v2 := v1.fst
  let v3 ← u256_encoded_len v2
  let v4 := v3
  let v5 := v1.fst
  let v6 ← u256_write_key v0 v5
  let v7 ← downcast_unchecked_u256_usize v4
  let v8 := v0 + v7
  let v9 := v1.snd
  let v10 ← u256_write_key v8 v9
  return ()

def mapping_slot_tuple_u256_u256 (v0 : UInt256) (v1 : (UInt256 × UInt256)) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v2 ← tuple_encoded_len_u256_u256 v1
  let v3 := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd v3 v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let v8 := v7
  r0 := v8
  let v10 ← MemBuffer_ptr r0
  let v11 := v10
  let v12 ← tuple_write_key_u256_u256 v11 v1
  let v13 ← downcast_unchecked_u256_usize v3
  let v14 := v11 + v13
  let v15 ← mstore v14 v0
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd v3 v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 v11 v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop r0
  return v21

def StorageMap_locate_tuple_u256_u256_bool (v0 : Place) (v1 : UInt256) (v2 : (UInt256 × UInt256)) : FeM Location := do
  let v3 ← mapping_slot_tuple_u256_u256 v1 v2
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def bool_encoded_len (v0 : Bool) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def bool_write_key (v0 : UInt256) (v1 : Bool) : FeM Unit := do
  let v3 := v1
  let b3 : UInt256 → FeM Unit := fun v2 => do
    let v8 ← u256_write_key v0 v2
    return ()
  if v3 then
    let v4 := (UInt256.ofNat 1)
    let v5 := v4
    b3 v5
  else
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7

def mapping_slot_bool (v0 : UInt256) (v1 : Bool) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v2 ← bool_encoded_len v1
  let v3 := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd v3 v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let v8 := v7
  r0 := v8
  let v10 ← MemBuffer_ptr r0
  let v11 := v10
  let v12 ← bool_write_key v11 v1
  let v13 ← downcast_unchecked_u256_usize v3
  let v14 := v11 + v13
  let v15 ← mstore v14 v0
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd v3 v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 v11 v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop r0
  return v21

def StorageMap_locate_bool_u256 (v0 : Place) (v1 : UInt256) (v2 : Bool) : FeM Location := do
  let v3 ← mapping_slot_bool v1 v2
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def calldata_ptr_Evm_array_u256_2 (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 := v2
  return v3

inductive Option_mut_u8 where
  | None
  | Some (f0 : UInt256)
deriving Inhabited

def Option_mut_u8.tag : Option_mut_u8 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_mut_u8.Some_0 : Option_mut_u8 → UInt256
  | .Some f0 => f0
  | _ => default

def Option_mut_u8.set_Some_0 (x : Option_mut_u8) (v : UInt256) : Option_mut_u8 :=
  match x with
  | .Some f0 => .Some v
  | x => x

def array_start_u256_2 (v0 : (Vector UInt256 2)) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def array_has_u256_2 (v0 : (Vector UInt256 2)) (v1 : UInt256) : FeM Bool := do
  let v2 := v1
  let v3 := (UInt256.ofNat 2)
  let v4 := decide (v2.toNat < v3.toNat)
  return v4

def array_at_ref_u256_2 (v0 : (Vector UInt256 2)) (v1 : UInt256) : FeM UInt256 := do
  checkIndex v1 2
  return v0[v1.toNat]!

def usize_wrapping_add (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0 + v1
  return v2

def array_next_u256_2 (v0 : (Vector UInt256 2)) (v1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let v3 := v1
  let v4 ← usize_wrapping_add v3 v2
  return v4

structure DynString where
  data : UInt256
  len : UInt256
  size : UInt256
deriving Inhabited

structure Panic where
  code : UInt256
deriving Inhabited

structure SolWord where
  word : UInt256
deriving Inhabited

def Layout.SolSlot : Layout :=
  { storage := [0] }

structure RawCallOutcome where
  success : Bool
  returndata_len : UInt256
deriving Inhabited

def DynString_span_ref (v0 : DynString) : FeM MemSlice_u8 := do
  let v1 := v0.data
  let v2 := (UInt256.ofNat 32)
  let v3 := v1 + v2
  let v4 := v0.len
  let v5 ← MemSlice_from_raw_parts_ref_u8 v3 v4
  return v5

def PANIC_STORAGE_ENCODING : FeM UInt256 := do
  let v0 := (UInt256.ofNat 34)
  return v0

def Panic_from (v0 : UInt256) : FeM Panic := do
  let v1 := v0
  let v2 := (Panic.mk v1)
  return v2

def Panic_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Panic_payload_size (v0 : Panic) : FeM UInt256 := do
  let v1 ← Panic_HEAD_SIZE
  let v2 := v0.code
  let v3 ← dynamic_payload_size_u256 v2
  let v4 ← checkedAdd v1 v3
  return v4

def abi_payload_size_Panic (v0 : Panic) : FeM UInt256 := do
  let v1 ← Panic_payload_size v0
  return v1

def MemBuffer_set_len (v0 : MemBuffer) (v1 : UInt256) : FeM (Unit × MemBuffer) := do
  let mut v0 := v0
  let v2 := v1
  let v3 := v0.capacity
  let v4 := decide (v3.toNat < v2.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := v1
    v0 := { v0 with len := v8 }
    let v9 := ()
    return ((), v0)

def Panic_encode (v0 : Panic) (v1 : AbiFrame) : FeM Unit := do
  let mut r0 : AbiFrame := default
  let mut r1 : UInt256 := default
  let v2 ← Panic_HEAD_SIZE
  let v3 ← AbiFrame_tail v1 v2
  let v4 := v3
  r0 := v4
  let v5 := (UInt256.ofNat 0)
  let v6 := v5
  r1 := v6
  let v7 := v0.code
  let v8 := r1
  let v9 := v8
  let v10 := r0
  let v11 := v10
  let v12 ← encode_field_Sol_u256 v7 v1 v9 v11
  return ()

def encode_calldata_Panic (v0 : UInt256) (r0 : Panic) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v2 := (UInt256.ofNat 4)
  let v3 := v2
  let v5 ← abi_payload_size_Panic r0
  let v6 := v5
  let v7 ← downcast_unchecked_u256_usize v6
  let v8 := v3 + v7
  let v9 := v8
  let v10 ← Panic_HEAD_SIZE
  let v11 := (UInt256.ofNat 28)
  let v12 := decide (v10.toNat < v11.toNat)
  let b3 : Panic → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun r0 r1 r2 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v18 := v1
    let v19 ← MemBuffer_new_uninit v18
    let v20 := v19
    r1 := v20
    let v22 ← MemBuffer_ptr r1
    let v23 := v22 + v3
    let v24 := v23
    let (v26, v26_0) ← MemBuffer_set_len r1 v9
    r1 := v26_0
    let v28 ← MemBuffer_ptr r1
    let v29 := v0
    let v30 := v29
    let v31 := (UInt256.ofNat 224)
    let v32 := v30 <<< v31
    let v33 ← mstore v28 v32
    let v34 ← downcast_unchecked_u256_usize v6
    let v35 ← zero_bytes v24 v34
    let v36 ← AbiFrame_from_raw_parts v24 v6
    r2 := v36
    let v39 ← Panic_encode r0 r2
    let v40 := r1
    return v40
  let b2 : Panic → MemBuffer → AbiFrame → FeM MemBuffer := fun r0 r1 r2 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v17 := v9
    b3 r0 r1 r2 v17
  if v12 then
    let v13 := (UInt256.ofNat 32)
    let v14 := decide (v9.toNat < v13.toNat)
    if v14 then
      let v15 := (UInt256.ofNat 32)
      let v16 := v15
      b3 r0 r1 r2 v16
    else
      b2 r0 r1 r2
  else
    b2 r0 r1 r2

def encode_error_Panic (v0 : Panic) : FeM MemBuffer := do
  let v1 := (UInt256.ofNat 1313373041) -- a constant, as the compiler evaluates it
  let v2 := v0
  let v3 ← encode_calldata_Panic v1 v2
  return v3

def revert_error_Panic (v0 : Panic) : FeM Empty := do
  let mut r0 : MemBuffer := default
  let v1 := v0
  let v2 ← encode_error_Panic v1
  let v3 := v2
  r0 := v3
  let v5 ← MemBuffer_ptr r0
  let v7 ← MemBuffer_len r0
  let v8 := v7
  let v9 ← revertData v5 v8
  nomatch v9

def keccak_words_1 (v0 : (Vector UInt256 1)) : FeM UInt256 := do
  let v1 := keccakWords v0
  return v1

def array_data_slot (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : (Vector UInt256 1) := default
  let v1 := v0
  let v2 := #v[v1]
  r0 := v2
  let v4 ← keccak_words_1 r0
  return v4

def u256_wrapping_add (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0 + v1
  return v2

def u256_max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)
  return v0

def DynString_drop (v0 : DynString) : FeM Unit := do
  let v1 := v0.data
  let v2 ← dealloc_u8 v1
  return ()

def u8_BITS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 8)
  return v0

def field_shift (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 32)
  let v4 := decide (v3.toNat ≤ v2.toNat)
  let b1 : FeM UInt256 := do
    let v17 ← revertEmpty
    nomatch v17
  if v4 then
    b1
  else
    let v5 := v1
    let v6 := (UInt256.ofNat 256)
    let v7 := v0
    let v8 := (UInt256.ofNat 8)
    let v9 ← checkedMul v7 v8
    let v10 ← checkedSub v6 v9
    let v11 := decide (v10.toNat < v5.toNat)
    if v11 then
      b1
    else
      let v12 := ()
      let v13 := v12
      let v14 := v0
      let v15 := (UInt256.ofNat 8)
      let v16 ← checkedMul v14 v15
      return v16

def u8_to_packed (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def field_mask (v0 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 256)
  let v4 := decide (v3.toNat ≤ v2.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u256_max
    let v6 := v5
    b3 v6
  else
    let v7 := (UInt256.ofNat 1)
    let v8 := v7
    let v9 := v0
    let v10 := v8 <<< v9
    let v11 := (UInt256.ofNat 1)
    let v12 ← checkedSub v10 v11
    let v13 := v12
    b3 v13

def SolWord_set_u8 (v0 : SolWord) (v1 : UInt256) (v2 : UInt256) : FeM (Unit × SolWord) := do
  let mut v0 := v0
  let v3 ← u8_BITS
  let v4 ← field_shift v1 v3
  let v5 := v4
  let v6 ← u8_BITS
  let v7 ← field_mask v6
  let v8 := v7 <<< v5
  let v9 := v8
  let v10 := v0.word
  let v11 := ~~~v9
  let v12 := v10 &&& v11
  let v13 ← u8_to_packed v2
  let v14 := v13 <<< v5
  let v15 := v14 &&& v9
  let v16 := v12 ||| v15
  v0 := { v0 with word := v16 }
  let v17 := ()
  return ((), v0)

def round_up_words (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := (UInt256.ofNat 0)
  let v3 := decide (v1 = v2)
  if v3 then
    let v4 := (UInt256.ofNat 0)
    return v4
  else
    let v5 := ()
    let v6 := v5
    let v7 := v0
    let v8 := (UInt256.ofNat 1)
    let v9 ← checkedSub v7 v8
    let v10 := (UInt256.ofNat 32)
    let v11 ← checkedDiv v9 v10
    let v12 := (UInt256.ofNat 1)
    let v13 ← checkedAdd v11 v12
    let v14 := (UInt256.ofNat 32)
    let v15 ← checkedMul v13 v14
    return v15

def mem_word_at (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize v1
  let v3 := v0 + v2
  let v4 ← cast_u8_u256 v3
  let v5 ← mload v4
  let v6 := v5
  return v6

def DynString_from_validated_abi_tail (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM DynString := do
  let v3 := v0
  let v4 := v1
  let v5 := v2
  let v6 := (DynString.mk v3 v4 v5)
  return v6

def DynString_from_abi_tail (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM DynString := do
  let v3 := (UInt256.ofNat 32)
  let v4 := v1
  let v5 := v4
  let v6 ← round_up_words v5
  let v7 ← checkedAdd v3 v6
  let v8 := v7
  let v9 := v2
  let v10 := decide (v9 ≠ v8)
  let b1 : FeM DynString := do
    let v19 ← trap
    nomatch v19
  if v10 then
    b1
  else
    let v11 := (UInt256.ofNat 0)
    let v12 ← mem_word_at v0 v11
    let v13 := v1
    let v14 := v13
    let v15 := decide (v12 ≠ v14)
    if v15 then
      b1
    else
      let v16 := ()
      let v17 := v16
      let v18 ← DynString_from_validated_abi_tail v0 v1 v2
      return v18

def Evm_selfbalance : FeM Wei := do
  let v1 ← selfbalance
  let v2 ← wei v1
  return v2

def Address_eq (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v2 := v0
  let v3 := v1
  let v4 := decide (v2 = v3)
  return v4

def ne_Address_Address (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v2 ← Address_eq v0 v1
  let v3 := !v2
  return v3

def MemBuffer_with_capacity (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes v0
  let v2 := (UInt256.ofNat 0)
  let v3 := v0
  let v4 := (MemBuffer.mk v1 v2 v3)
  return v4

def MemBuffer_empty : FeM MemBuffer := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← MemBuffer_with_capacity v0
  return v1

def MemSlice_empty_u8 : FeM MemSlice_u8 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← alloc_bytes v0
  let v2 ← cast_u8_u8 v1
  let v3 := (UInt256.ofNat 0)
  let v4 ← MemSlice_from_raw_parts_ref_u8 v2 v3
  return v4

def MemBuffer_capacity (v0 : MemBuffer) : FeM UInt256 := do
  let v1 := v0.capacity
  return v1

def finish_raw_call (v0 : Bool) (v1 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut v1 := v1
  let v3 ← returndatasize
  let v4 := v3
  let v6 ← MemBuffer_capacity v1
  let v7 := v6
  let v8 := decide (v4.toNat < v7.toNat)
  let b3 : MemBuffer → UInt256 → FeM (RawCallOutcome × MemBuffer) := fun v1 v2 => do
    let mut v1 := v1
    let v14 := v2
    let (v15, v15_0) ← MemBuffer_set_len v1 v14
    v1 := v15_0
    let v16 := v0
    let v17 := (RawCallOutcome.mk v16 v4)
    return (v17, v1)
  if v8 then
    let v9 ← downcast_unchecked_u256_usize v4
    let v10 := v9
    b3 v1 v10
  else
    let v12 ← MemBuffer_capacity v1
    let v13 := v12
    b3 v1 v13

def raw_call_Evm (v1 : UInt256) (v2 : UInt256) (v3 : Wei) (v4 : MemSlice_u8) (v5 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut v5 := v5
  let v6 := v1
  let v7 ← Wei_as_wei v3
  let v8 ← MemSlice_ptr_u8 v4
  let v9 ← MemSlice_len_u8 v4
  let v10 := v9
  let v12 ← MemBuffer_ptr v5
  let v14 ← MemBuffer_capacity v5
  let v15 := v14
  let v16 ← Fe.EVM.call v2 v6 v7 v8 v10 v12 v15
  let v17 := (UInt256.ofNat 0)
  let v18 := decide (v16 ≠ v17)
  let v19 := v18
  let (v21, v21_0) ← finish_raw_call v19 v5
  v5 := v21_0
  return (v21, v5)

def send_value_Evm (r0 : UInt256) (v1 : UInt256) (v2 : Wei) : FeM RawCallOutcome := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let v3 ← MemBuffer_empty
  let v4 := v3
  r1 := v4
  let v5 ← MemSlice_empty_u8
  let v7 := r0
  let (v9, v9_0) ← raw_call_Evm v7 v1 v2 v5 r1
  r1 := v9_0
  let _ ← MemBuffer_drop r1
  return v9

def RawCallOutcome_success (v0 : RawCallOutcome) : FeM Bool := do
  let v1 := v0.success
  return v1

def Evm_log2 (v1 : MemSlice_u8) (v2 : UInt256) (v3 : UInt256) : FeM Unit := do
  let v4 ← MemSlice_ptr_u8 v1
  let v5 ← MemSlice_len_u8 v1
  let v6 := v5
  let v7 ← logData v4 v6 #[v2, v3]
  return ()

def u8_IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def u8_MASK : FeM UInt256 := do
  let v0 ← U8_MASK
  return v0

def u8_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def u8_from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := truncate 8 v0
  return v1

def downcast_truncate_u256_u8 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← u256_to_word v0
  let v3 := v2
  let v4 ← u8_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← u8_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u8_MASK
    let v11 := v3 &&& v10
    let v12 ← u8_from_word v11
    let v13 := v12
    b3 v13

def u8_from_packed (v0 : UInt256) : FeM UInt256 := do
  let v1 ← downcast_truncate_u256_u8 v0
  return v1

def SolWord_get_u8 (v0 : SolWord) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← u8_BITS
  let v3 ← field_shift v1 v2
  let v4 := v3
  let v5 := v0.word
  let v6 := v5 >>> v4
  let v7 ← u8_BITS
  let v8 ← field_mask v7
  let v9 := v6 &&& v8
  let v10 ← u8_from_packed v9
  return v10

def DynString_IS_DYNAMIC : FeM Bool := do
  let v0 := true
  return v0

def DynString_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def DynString_encoded_span (v0 : DynString) : FeM MemSlice_u8 := do
  let v1 := v0.data
  let v2 := v0.size
  let v3 ← word_to_usize v2
  let v4 ← MemSlice_from_raw_parts_ref_u8 v1 v3
  return v4

def copy_raw (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 ← mcopy v0 v1 v2
  return ()

def copy (v0 : UInt256) (v1 : MemSlice_u8) : FeM Unit := do
  let v2 ← MemSlice_ptr_u8 v1
  let v3 ← MemSlice_len_u8 v1
  let v4 ← copy_raw v0 v2 v3
  return ()

def AbiFrame_copy_from (v0 : AbiFrame) (v1 : UInt256) (v2 : MemSlice_u8) : FeM Unit := do
  let v3 ← MemSlice_len_u8 v2
  let v4 := v3
  let v5 ← AbiFrame_slice v0 v1 v4
  let v6 := v5
  let v7 := v6.ptr
  let v8 ← copy v7 v2
  return ()

def DynString_encode_Sol (v0 : DynString) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← DynString_encoded_span v0
  let v5 ← AbiFrame_copy_from v1 v2 v3
  return ()

def const_DynString : FeM Bool := do
  let v0 := false
  return v0

def DynString_encoded_size (v0 : DynString) : FeM UInt256 := do
  let v1 := v0.size
  return v1

def DynString_payload_size (v0 : DynString) : FeM UInt256 := do
  let v1 ← DynString_encoded_size v0
  return v1

def abi_payload_size_DynString (v0 : DynString) : FeM UInt256 := do
  let v1 ← DynString_payload_size v0
  return v1

def encode_alloc_Sol_DynString (r0 : DynString) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_DynString r0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r2 := v8
  let v11 ← DynString_encode_Sol r0 r2
  let _ ← DynString_drop r0
  let v12 := r1
  return v12

def abi_field_size_DynString (v0 : DynString) : FeM UInt256 := do
  let v1 ← DynString_payload_size v0
  let v2 := v1
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    return v2

def abi_single_root_size_DynString (v0 : DynString) : FeM UInt256 := do
  let v1 ← abi_field_size_DynString v0
  return v1

def abi_field_head_size_DynString : FeM UInt256 := do
  let v0 ← DynString_IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← DynString_HEAD_SIZE
    return v4

def abi_single_root_head_size_DynString : FeM UInt256 := do
  let v0 ← abi_field_head_size_DynString
  return v0

def encode_field_Sol_DynString (v0 : DynString) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    let v4 ← DynString_payload_size v0
    let v5 := v4
    let v6 ← AbiFrame_len v1
    let v8 ← AbiFrame_len r0
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice r0 v11 v5
    r1 := v13
    let v15 ← DynString_encode_Sol v0 r1
    let v17 ← AbiFrame_tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← DynString_HEAD_SIZE
    let v21 ← AbiFrame_slice v1 v2 v20
    r2 := v21
    let v23 ← DynString_encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_DynString (r0 : DynString) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_single_root_size_DynString r0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  let v9 := v8
  r2 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_DynString
  let v13 ← AbiFrame_tail r2 v11
  let v16 ← encode_field_Sol_DynString r0 r2 v10 v13
  let _ ← DynString_drop r0
  let v17 := r1
  return v17

def encode_return_alloc_Sol_DynString (v0 : DynString) : FeM MemBuffer := do
  let v1 ← const_DynString
  if v1 then
    let v2 := v0
    let v3 ← encode_alloc_Sol_DynString v2
    return v3
  else
    let v4 := ()
    let v5 := v4
    let v6 := v0
    let v7 ← encode_single_root_alloc_Sol_DynString v6
    return v7

def Evm_return_value_Sol_DynString (r0 : DynString) : FeM Empty := do
  let mut r0 := r0
  let mut r1 : FixedMemBuffer_32 := default
  let mut r2 : FixedMemBuffer_64 := default
  let mut r3 : MemBuffer := default
  let mut r4 : MemBuffer := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let mut r7 : AbiFrame := default
  let v1 ← DynString_IS_DYNAMIC
  let v2 := !v1
  if v2 then
    let v3 ← DynString_HEAD_SIZE
    let v4 := (UInt256.ofNat 32)
    let v5 := decide (v3.toNat ≤ v4.toNat)
    if v5 then
      let v6 ← FixedMemBuffer_alloc_32
      let v7 := v6
      r1 := v7
      let v9 ← FixedMemBuffer_ptr_32 r1
      let v10 ← DynString_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let v14 ← DynString_encode_Sol r0 r5
      let v16 ← FixedMemBuffer_ptr_32 r1
      let v17 ← DynString_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← DynString_HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let v27 := v26
        r2 := v27
        let v29 ← FixedMemBuffer_ptr_64 r2
        let v30 ← DynString_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let v34 ← DynString_encode_Sol r0 r6
        let v36 ← FixedMemBuffer_ptr_64 r2
        let v37 ← DynString_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← DynString_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let v46 := v45
        r3 := v46
        let v48 ← MemBuffer_ptr r3
        let v49 ← DynString_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let v53 ← DynString_encode_Sol r0 r7
        let v55 ← MemBuffer_span_ref r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let v59 := r0
    let v60 := v59
    let v61 ← encode_return_alloc_Sol_DynString v60
    let v62 := v61
    r4 := v62
    let v64 ← MemBuffer_span_ref r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

def u8_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def u8_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def u8_encode_Sol (v0 : UInt256) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := v0
  let v4 := v3
  let v5 ← AbiFrame_store_word v1 v2 v4
  return ()

def const_u8 : FeM Bool := do
  let v0 := false
  return v0

def payload_size_u8 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← u8_HEAD_SIZE
  return v1

def abi_payload_size_u8 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u8 v0
  return v1

def encode_alloc_Sol_u8 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_u8 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_ptr r0
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r1 := v8
  let v10 ← u8_encode_Sol v0 r1
  let v11 := r0
  return v11

def abi_field_size_u8 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u8 v0
  let v2 := v1
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    return v2

def abi_single_root_size_u8 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_u8 v0
  return v1

def abi_field_head_size_u8 : FeM UInt256 := do
  let v0 ← u8_IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← u8_HEAD_SIZE
    return v4

def abi_single_root_head_size_u8 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u8
  return v0

def encode_field_Sol_u8 (v0 : UInt256) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_u8 v0
    let v5 := v4
    let v6 ← AbiFrame_len v1
    let v8 ← AbiFrame_len r0
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice r0 v11 v5
    r1 := v13
    let v15 ← u8_encode_Sol v0 r1
    let v17 ← AbiFrame_tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← u8_HEAD_SIZE
    let v21 ← AbiFrame_slice v1 v2 v20
    r2 := v21
    let v23 ← u8_encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_u8 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_single_root_size_u8 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r0 := v5
  let v7 ← MemBuffer_ptr r0
  let v8 ← AbiFrame_from_raw_parts v7 v2
  let v9 := v8
  r1 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_u8
  let v13 ← AbiFrame_tail r1 v11
  let v15 ← encode_field_Sol_u8 v0 r1 v10 v13
  let v16 := r0
  return v16

def encode_return_alloc_Sol_u8 (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← const_u8
  if v1 then
    let v2 ← encode_alloc_Sol_u8 v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_u8 v0
    return v5

def Evm_return_value_Sol_u8 (v1 : UInt256) : FeM Empty := do
  let mut r0 : FixedMemBuffer_32 := default
  let mut r1 : FixedMemBuffer_64 := default
  let mut r2 : MemBuffer := default
  let mut r3 : MemBuffer := default
  let mut r4 : AbiFrame := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let v2 ← u8_IS_DYNAMIC
  let v3 := !v2
  if v3 then
    let v4 ← u8_HEAD_SIZE
    let v5 := (UInt256.ofNat 32)
    let v6 := decide (v4.toNat ≤ v5.toNat)
    if v6 then
      let v7 ← FixedMemBuffer_alloc_32
      let v8 := v7
      r0 := v8
      let v10 ← FixedMemBuffer_ptr_32 r0
      let v11 ← u8_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let v14 ← u8_encode_Sol v1 r4
      let v16 ← FixedMemBuffer_ptr_32 r0
      let v17 ← u8_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← u8_HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let v27 := v26
        r1 := v27
        let v29 ← FixedMemBuffer_ptr_64 r1
        let v30 ← u8_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let v33 ← u8_encode_Sol v1 r5
        let v35 ← FixedMemBuffer_ptr_64 r1
        let v36 ← u8_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← u8_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← MemBuffer_new v43
        let v45 := v44
        r2 := v45
        let v47 ← MemBuffer_ptr r2
        let v48 ← u8_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let v51 ← u8_encode_Sol v1 r6
        let v53 ← MemBuffer_span_ref r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_u8 v1
    let v58 := v57
    r3 := v58
    let v60 ← MemBuffer_span_ref r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

structure CallOutcome where
  success : Bool
  returndata : MemBuffer
deriving Inhabited

def Wei_zero : FeM Wei := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (Wei.mk v0)
  return v1

def tuple_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def tuple_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def payload_size_tuple (v0 : Unit) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE
  return v1

def abi_payload_size_tuple (v0 : Unit) : FeM UInt256 := do
  let v1 ← payload_size_tuple v0
  return v1

def tuple_encode_Sol (v0 : Unit) (v1 : AbiFrame) : FeM Unit := do
  let v2 := ()
  return ()

def encode_alloc_Sol_tuple (r0 : Unit) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v0 := r0
  let v1 ← abi_payload_size_tuple v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r2 := v8
  let v9 := r0
  let v11 ← tuple_encode_Sol v9 r2
  let v12 := r1
  return v12

def encode_abi_payload_tuple (v0 : Unit) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_tuple v0
  return v1

def create2_raw_Evm (v1 : Wei) (v2 : MemSlice_u8) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← Wei_as_wei v1
  let v5 ← MemSlice_ptr_u8 v2
  let v6 ← MemSlice_len_u8 v2
  let v7 := v6
  let v8 ← create2 v4 v5 v7 v3
  let v9 := v8
  return v9

def copy_returndata : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let v0 ← returndatasize
  let v1 := v0
  let v2 ← downcast_unchecked_u256_usize v1
  let v3 ← MemBuffer_new_uninit v2
  let v4 := v3
  r0 := v4
  let v6 ← MemBuffer_ptr r0
  let v7 := (UInt256.ofNat 0)
  let v8 ← returndatacopy v6 v7 v1
  let v9 := r0
  return v9

def bubble_last_revert : FeM Empty := do
  let mut r0 : MemBuffer := default
  let v0 ← copy_returndata
  let v1 := v0
  r0 := v1
  let v3 ← MemBuffer_ptr r0
  let v5 ← MemBuffer_len r0
  let v6 := v5
  let v7 ← revertData v3 v6
  nomatch v7

def copy_call_outcome (v0 : RawCallOutcome) : FeM CallOutcome := do
  let v1 ← RawCallOutcome_success v0
  let v2 ← copy_returndata
  let v3 := (CallOutcome.mk v1 v2)
  return v3

def u256_decode_from_Sol_MemSlice_u8 (v0 : MemSlice_u8) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_u8_word_at v0 v1
  return v2

def decode_from_prechecked_head_u256_Sol_MemSlice_u8 (v0 : MemSlice_u8) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← u256_decode_from_Sol_MemSlice_u8 v0 v1
  return v3

def decode_frame_from_Sol_u256_MemSlice_u8 (v0 : MemSlice_u8) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← decode_from_prechecked_head_u256_Sol_MemSlice_u8 v0 v1 v2
  return v5

def decode_from_bounded_u256_Sol_MemSlice_u8 (v0 : MemSlice_u8) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← u256_decode_from_Sol_MemSlice_u8 v0 v1
  return v5

def decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 (v0 : MemSlice_u8) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← MemSlice_u8_word_at v0 v2
    let v6 := v5
    let v7 ← checked_tail_Sol v1 v6
    let v8 := v7
    let v9 ← decode_from_bounded_u256_Sol_MemSlice_u8 v0 v8 v3
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← u256_decode_from_Sol_MemSlice_u8 v0 v2
    return v12

def decode_output_at_u256_MemSlice_u8 (v0 : MemSlice_u8) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← len_MemSlice_u8 v0
  let v3 := v2
  let v4 ← const_u256
  if v4 then
    let v5 ← decode_frame_from_Sol_u256_MemSlice_u8 v0 v1 v3
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← abi_single_root_head_size_u256
    let v9 ← checked_frame_end_Sol v1 v8 v3
    let v10 ← decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 v0 v1 v1 v3
    return v10

def decode_output_u256_MemSlice_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  let v2 ← decode_output_at_u256_MemSlice_u8 v0 v1
  return v2

def decode_returndata_u256 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← decode_output_u256_MemSlice_u8 v0
  return v1

def CallOutcome_decode_u256 (v0 : CallOutcome) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v1 := v0.success
  let v2 := v1
  let v3 := v0.returndata
  let v4 := v3
  r0 := v4
  let v5 := !v2
  if v5 then
    let v7 ← MemBuffer_ptr r0
    let v9 ← MemBuffer_len r0
    let v10 := v9
    let v11 ← revertData v7 v10
    nomatch v11
  else
    let v12 := ()
    let v13 := v12
    let v15 ← MemBuffer_span_ref r0
    let v17 ← decode_returndata_u256 v15
    let _ ← MemBuffer_drop r0
    return v17

def keccak_words_2 (v0 : (Vector UInt256 2)) : FeM UInt256 := do
  let v1 := keccakWords v0
  return v1

def read_transient_word_Evm (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  let v3 ← load (Place.mk .transient v2)
  return v3

def storage_at_mut_Evm_u256 (v0 : UInt256) : FeM Place := do
  let v1 := v0
  let v2 := (Place.mk .storage v1)
  return v2

def Address_encode_Sol (v0 : UInt256) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := v0
  let v4 ← AbiFrame_store_word v1 v2 v3
  return ()

def const_Address : FeM Bool := do
  let v0 := false
  return v0

def payload_size_Address (v0 : UInt256) : FeM UInt256 := do
  let v1 ← Address_HEAD_SIZE
  return v1

def abi_payload_size_Address (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_Address v0
  return v1

def encode_alloc_Sol_Address (r0 : UInt256) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v0 := r0
  let v1 ← abi_payload_size_Address v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r2 := v8
  let v9 := r0
  let v11 ← Address_encode_Sol v9 r2
  let v12 := r1
  return v12

def abi_field_size_Address (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_Address v0
  let v2 := v1
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    return v2

def abi_single_root_size_Address (v0 : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_Address v0
  return v1

def abi_single_root_head_size_Address : FeM UInt256 := do
  let v0 ← abi_field_head_size_Address
  return v0

def encode_field_Sol_Address (v0 : UInt256) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_Address v0
    let v5 := v4
    let v6 ← AbiFrame_len v1
    let v8 ← AbiFrame_len r0
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice r0 v11 v5
    r1 := v13
    let v15 ← Address_encode_Sol v0 r1
    let v17 ← AbiFrame_tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← Address_HEAD_SIZE
    let v21 ← AbiFrame_slice v1 v2 v20
    r2 := v21
    let v23 ← Address_encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_Address (r0 : UInt256) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v0 := r0
  let v1 ← abi_single_root_size_Address v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  let v9 := v8
  r2 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_Address
  let v13 ← AbiFrame_tail r2 v11
  let v14 := r0
  let v16 ← encode_field_Sol_Address v14 r2 v10 v13
  let v17 := r1
  return v17

def encode_return_alloc_Sol_Address (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← const_Address
  if v1 then
    let v2 ← encode_alloc_Sol_Address v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_Address v0
    return v5

def Evm_return_value_Sol_Address (r0 : UInt256) : FeM Empty := do
  let mut r0 := r0
  let mut r1 : FixedMemBuffer_32 := default
  let mut r2 : FixedMemBuffer_64 := default
  let mut r3 : MemBuffer := default
  let mut r4 : MemBuffer := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let mut r7 : AbiFrame := default
  let v1 ← Address_IS_DYNAMIC
  let v2 := !v1
  if v2 then
    let v3 ← Address_HEAD_SIZE
    let v4 := (UInt256.ofNat 32)
    let v5 := decide (v3.toNat ≤ v4.toNat)
    if v5 then
      let v6 ← FixedMemBuffer_alloc_32
      let v7 := v6
      r1 := v7
      let v9 ← FixedMemBuffer_ptr_32 r1
      let v10 ← Address_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let v12 := r0
      let v14 ← Address_encode_Sol v12 r5
      let v16 ← FixedMemBuffer_ptr_32 r1
      let v17 ← Address_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← Address_HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let v27 := v26
        r2 := v27
        let v29 ← FixedMemBuffer_ptr_64 r2
        let v30 ← Address_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let v32 := r0
        let v34 ← Address_encode_Sol v32 r6
        let v36 ← FixedMemBuffer_ptr_64 r2
        let v37 ← Address_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← Address_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let v46 := v45
        r3 := v46
        let v48 ← MemBuffer_ptr r3
        let v49 ← Address_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let v51 := r0
        let v53 ← Address_encode_Sol v51 r7
        let v55 ← MemBuffer_span_ref r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let v59 := r0
    let v60 ← encode_return_alloc_Sol_Address v59
    let v61 := v60
    r4 := v61
    let v63 ← MemBuffer_span_ref r4
    let v66 ← Evm_return_bytes v63
    nomatch v66

structure Range_Known_1_Known_4 where
  start : Unit
  «end» : Unit
deriving Inhabited

structure Range_Known_1_Known_16 where
  start : Unit
  «end» : Unit
deriving Inhabited

def assert (v0 : Bool) : FeM Unit := do
  let v1 := v0
  let v2 := !v1
  if v2 then
    let v3 ← trap
    nomatch v3
  else
    let v4 := ()
    let v5 := v4
    return ()

def u128_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def downcast_truncate_u128_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← u128_to_word v0
  let v3 := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := v3 &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def u64_wrapping_mul (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := IntTy.mulW (.u 64) v0 v1
  return v2

def u128_wrapping_sub (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := IntTy.subW (.u 128) v0 v1
  return v2

def i64_IS_SIGNED : FeM Bool := do
  let v0 := true
  return v0

def i64_MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

def I64_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 9223372036854775808)
  return v0

def i64_SIGN_BIT : FeM UInt256 := do
  let v0 ← I64_SIGN_BIT
  return v0

def i64_from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 64) v0
  return v1

def downcast_truncate_u64_i64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← u64_to_word v0
  let v3 := v2
  let v4 ← i64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← i64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← i64_MASK
    let v11 := v3 &&& v10
    let v12 ← i64_from_word v11
    let v13 := v12
    b3 v13

def i64_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  let v3 := v2
  return v3

def downcast_truncate_i64_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← i64_to_word v0
  let v3 := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := v3 &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def i128_to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  let v3 := v2
  return v3

def downcast_truncate_i128_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← i128_to_word v0
  let v3 := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := v3 &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def downcast_truncate_i128_i64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← i128_to_word v0
  let v3 := v2
  let v4 ← i64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← i64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← i64_MASK
    let v11 := v3 &&& v10
    let v12 ← i64_from_word v11
    let v13 := v12
    b3 v13

def array_start_usize_3 (v0 : (Vector UInt256 3)) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def array_has_usize_3 (v0 : (Vector UInt256 3)) (v1 : UInt256) : FeM Bool := do
  let v2 := v1
  let v3 := (UInt256.ofNat 3)
  let v4 := decide (v2.toNat < v3.toNat)
  return v4

def array_at_ref_usize_3 (v0 : (Vector UInt256 3)) (v1 : UInt256) : FeM UInt256 := do
  checkIndex v1 3
  return v0[v1.toNat]!

def Range_start_1_4 (v0 : Range_Known_1_Known_4) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  return v1

def range_has (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Bool := do
  let v4 := v1
  let v5 := v0
  let v6 := decide (v4.toNat ≤ v5.toNat)
  let b3 : Bool → FeM Bool := fun v3 => do
    return v3
  let b2 : FeM Bool := do
    let v12 := false
    let v13 := v12
    b3 v13
  if v6 then
    let v7 := v0
    let v8 := v2
    let v9 := decide (v7.toNat < v8.toNat)
    if v9 then
      let v10 := true
      let v11 := v10
      b3 v11
    else
      b2
  else
    b2

def Range_has_1_4 (v0 : Range_Known_1_Known_4) (v1 : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 1)
  let v3 := (UInt256.ofNat 4)
  let v4 ← range_has v1 v2 v3
  return v4

def Range_at_1_4 (v0 : Range_Known_1_Known_4) (r0 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  return r0

def Range_next_1_4 (v0 : Range_Known_1_Known_4) (v1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let v3 := v1
  let v4 ← usize_wrapping_add v3 v2
  return v4

def array_next_usize_3 (v0 : (Vector UInt256 3)) (v1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let v3 := v1
  let v4 ← usize_wrapping_add v3 v2
  return v4

def Range_start_1_16 (v0 : Range_Known_1_Known_16) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  return v1

def Range_has_1_16 (v0 : Range_Known_1_Known_16) (v1 : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 1)
  let v3 := (UInt256.ofNat 16)
  let v4 ← range_has v1 v2 v3
  return v4

def Range_at_1_16 (v0 : Range_Known_1_Known_16) (r0 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  return r0

def Range_next_1_16 (v0 : Range_Known_1_Known_16) (v1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let v3 := v1
  let v4 ← usize_wrapping_add v3 v2
  return v4

def downcast_truncate_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let v2 ← u256_to_word v0
  let v3 := v2
  let v4 ← usize_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← usize_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← usize_MASK
    let v11 := v3 &&& v10
    let v12 ← usize_from_word v11
    let v13 := v12
    b3 v13

def tuple_IS_DYNAMIC_bool_u256_u256 : FeM Bool := do
  let v1 ← bool_IS_DYNAMIC
  let b3 : Bool → FeM Bool := fun v0 => do
    return v0
  let b1 : FeM Bool := do
    let v6 := true
    let v7 := v6
    b3 v7
  if v1 then
    b1
  else
    let v2 ← u256_IS_DYNAMIC
    if v2 then
      b1
    else
      let v3 ← u256_IS_DYNAMIC
      if v3 then
        b1
      else
        let v4 := false
        let v5 := v4
        b3 v5

def tuple_HEAD_SIZE_bool_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_bool
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def tuple_encode_Sol_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) (v1 : AbiFrame) : FeM Unit := do
  let mut r0 : AbiFrame := default
  let v5 ← tuple_HEAD_SIZE_bool_u256_u256
  let v6 ← AbiFrame_tail v1 v5
  let v7 := v6
  r0 := v7
  let v8 ← abi_field_head_size_bool
  let v9 := v8
  let v10 ← abi_field_head_size_u256
  let v11 := v9 + v10
  let v12 := v11
  let v13 := (UInt256.ofNat 0)
  let v14 := r0
  let v15 := v14
  let v16 := v0.fst
  let v17 ← encode_field_Sol_bool v16 v1 v13 v15
  let v18 := v17
  r0 := v18
  let v19 := ()
  let v20 := r0
  let v21 := v20
  let v22 := v0.snd.fst
  let v23 ← encode_field_Sol_u256 v22 v1 v9 v21
  let v24 := v23
  r0 := v24
  let v25 := ()
  let v26 := r0
  let v27 := v26
  let v28 := v0.snd.snd
  let v29 ← encode_field_Sol_u256 v28 v1 v12 v27
  return ()

def tuple_IS_PARAM_LIST_bool_u256_u256 : FeM Bool := do
  let v0 := true
  return v0

def dynamic_payload_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← bool_IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_bool v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 := (UInt256.ofNat 0)
    return v5

def tuple_payload_size_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE_bool_u256_u256
  let v2 := v0.fst
  let v3 ← dynamic_payload_size_bool v2
  let v4 := v1 + v3
  let v5 := v0.snd.fst
  let v6 ← dynamic_payload_size_u256 v5
  let v7 := v4 + v6
  let v8 := v0.snd.snd
  let v9 ← dynamic_payload_size_u256 v8
  let v10 := v7 + v9
  return v10

def abi_payload_size_tuple_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_bool_u256_u256 v0
  return v1

def encode_alloc_Sol_tuple_bool_u256_u256 (r0 : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_tuple_bool_u256_u256 r0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  r2 := v8
  let v11 ← tuple_encode_Sol_bool_u256_u256 r0 r2
  let v12 := r1
  return v12

def abi_field_size_tuple_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_bool_u256_u256 v0
  let v2 := v1
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    return v2

def abi_single_root_size_tuple_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← abi_field_size_tuple_bool_u256_u256 v0
  return v1

def abi_field_head_size_tuple_bool_u256_u256 : FeM UInt256 := do
  let v0 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← tuple_HEAD_SIZE_bool_u256_u256
    return v4

def abi_single_root_head_size_tuple_bool_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_tuple_bool_u256_u256
  return v0

def encode_field_Sol_tuple_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    let v4 ← tuple_payload_size_bool_u256_u256 v0
    let v5 := v4
    let v6 ← AbiFrame_len v1
    let v8 ← AbiFrame_len r0
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice r0 v11 v5
    r1 := v13
    let v15 ← tuple_encode_Sol_bool_u256_u256 v0 r1
    let v17 ← AbiFrame_tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← tuple_HEAD_SIZE_bool_u256_u256
    let v21 ← AbiFrame_slice v1 v2 v20
    r2 := v21
    let v23 ← tuple_encode_Sol_bool_u256_u256 v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_tuple_bool_u256_u256 (r0 : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_single_root_size_tuple_bool_u256_u256 r0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 ← MemBuffer_ptr r1
  let v8 ← AbiFrame_from_raw_parts v7 v2
  let v9 := v8
  r2 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_tuple_bool_u256_u256
  let v13 ← AbiFrame_tail r2 v11
  let v16 ← encode_field_Sol_tuple_bool_u256_u256 r0 r2 v10 v13
  let v17 := r1
  return v17

def encode_return_alloc_Sol_tuple_bool_u256_u256 (v0 : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let v1 ← tuple_IS_PARAM_LIST_bool_u256_u256
  if v1 then
    let v2 ← encode_alloc_Sol_tuple_bool_u256_u256 v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_tuple_bool_u256_u256 v0
    return v5

def Evm_return_value_Sol_tuple_bool_u256_u256 (r0 : (Bool × UInt256 × UInt256)) : FeM Empty := do
  let mut r0 := r0
  let mut r1 : FixedMemBuffer_32 := default
  let mut r2 : FixedMemBuffer_64 := default
  let mut r3 : MemBuffer := default
  let mut r4 : MemBuffer := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let mut r7 : AbiFrame := default
  let v1 ← tuple_IS_DYNAMIC_bool_u256_u256
  let v2 := !v1
  if v2 then
    let v3 ← tuple_HEAD_SIZE_bool_u256_u256
    let v4 := (UInt256.ofNat 32)
    let v5 := decide (v3.toNat ≤ v4.toNat)
    if v5 then
      let v6 ← FixedMemBuffer_alloc_32
      let v7 := v6
      r1 := v7
      let v9 ← FixedMemBuffer_ptr_32 r1
      let v10 ← tuple_HEAD_SIZE_bool_u256_u256
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let v14 ← tuple_encode_Sol_bool_u256_u256 r0 r5
      let v16 ← FixedMemBuffer_ptr_32 r1
      let v17 ← tuple_HEAD_SIZE_bool_u256_u256
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← tuple_HEAD_SIZE_bool_u256_u256
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let v27 := v26
        r2 := v27
        let v29 ← FixedMemBuffer_ptr_64 r2
        let v30 ← tuple_HEAD_SIZE_bool_u256_u256
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let v34 ← tuple_encode_Sol_bool_u256_u256 r0 r6
        let v36 ← FixedMemBuffer_ptr_64 r2
        let v37 ← tuple_HEAD_SIZE_bool_u256_u256
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← tuple_HEAD_SIZE_bool_u256_u256
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let v46 := v45
        r3 := v46
        let v48 ← MemBuffer_ptr r3
        let v49 ← tuple_HEAD_SIZE_bool_u256_u256
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let v53 ← tuple_encode_Sol_bool_u256_u256 r0 r7
        let v55 ← MemBuffer_span_ref r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let v59 := r0
    let v60 ← encode_return_alloc_Sol_tuple_bool_u256_u256 v59
    let v61 := v60
    r4 := v61
    let v63 ← MemBuffer_span_ref r4
    let v66 ← Evm_return_bytes v63
    nomatch v66


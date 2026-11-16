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

def u256_to_word (self_0 : UInt256) : FeM UInt256 := do
  let self_1 := self_0
  return self_1

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

def sign_extend (word_0 : UInt256) (mask_1 : UInt256) (sign_bit_2 : UInt256) : FeM UInt256 := do
  let word_4 := word_0
  let mask_5 := mask_1
  let v6 := word_4 &&& mask_5
  let value_7 := v6
  let sign_bit_8 := sign_bit_2
  let v9 := value_7 &&& sign_bit_8
  let v10 := (UInt256.ofNat 0)
  let v11 := decide (v9 ≠ v10)
  let b3 : UInt256 → FeM UInt256 := fun v3 => do
    return v3
  if v11 then
    let v12 ← U256_MASK
    let mask_13 := mask_1
    let v14 := v12 ^^^ mask_13
    let v15 := value_7 ||| v14
    let v16 := v15
    b3 v16
  else
    let value_17 := value_7
    b3 value_17

def u64_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := truncate 64 word
  return v1

def downcast_truncate_u256_u64 (self : UInt256) : FeM UInt256 := do
  let v2 ← u256_to_word self
  let word := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def CallData_size (self : CallData) : FeM UInt256 := do
  let v2 ← calldatasize
  let total := v2
  let v4 := self.base
  let v5 := decide (total.toNat < v4.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7
  else
    let v8 := self.base
    let v9 ← checkedSub total v8
    let v10 ← downcast_truncate_u256_u64 v9
    let v11 := v10
    b3 v11

def len_CallData (self : CallData) : FeM UInt256 := do
  let v1 ← CallData_size self
  let v2 := v1
  return v2

def i256_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := word
  return v1

def Sol_validate_runtime_head (input_len_0 : UInt256) (base_1 : UInt256) (head_size_2 : UInt256) : FeM Unit := do
  let head_size_3 := head_size_2
  let v4 := (UInt256.ofNat 0)
  let v5 := decide (head_size_3 ≠ v4)
  let b3 : FeM Unit := do
    return ()
  if v5 then
    let input_len_6 := input_len_0
    let base_7 := base_1
    let v8 := input_len_6 - base_7
    let v9 ← i256_from_word v8
    let args_len := v9
    let v11 ← i256_from_word head_size_2
    let v12 := UInt256.sltBool args_len v11
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

def u64_to_word (self_0 : UInt256) : FeM UInt256 := do
  let self_1 := self_0
  let v2 := self_1
  return v2

def usize_IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def usize_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  let v2 := v1
  return v2

def usize_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := word
  return v1

def I256_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 57896044618658097711785492504343953926634992332820282019728792003956564819968)
  return v0

def usize_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def downcast_saturate_u64_usize (self : UInt256) : FeM UInt256 := do
  let mut unsigned_word : UInt256 := default
  let v8 ← u64_to_word self
  let word_9 := v8
  let v10 ← usize_IS_SIGNED
  let v11 := !v10
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v1 => do
    let mut unsigned_word := unsigned_word
    return v1
  if v11 then
    let word_12 := word_9
    unsigned_word := word_12
    let v13 ← u64_IS_SIGNED
    let b6 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      let unsigned_word_24 := unsigned_word
      let unsigned_word_25 := unsigned_word_24
      let v26 ← usize_MASK
      let v27 := decide (v26.toNat < unsigned_word_25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v2 => do
        let mut unsigned_word := unsigned_word
        let v35 := v2
        b3 unsigned_word v35
      if v27 then
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        let v30 := v29
        b10 unsigned_word v30
      else
        let unsigned_word_31 := unsigned_word
        let unsigned_word_32 := unsigned_word_31
        let v33 ← usize_from_word unsigned_word_32
        let v34 := v33
        b10 unsigned_word v34
    let b5 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      let v22 := ()
      let v23 := v22
      b6 unsigned_word
    if v13 then
      let v14 ← I256_SIGN_BIT
      let v15 := word_9 &&& v14
      let v16 := (UInt256.ofNat 0)
      let v17 := decide (v15 ≠ v16)
      if v17 then
        let v18 := (UInt256.ofNat 0)
        let unsigned_word_19 := v18
        unsigned_word := unsigned_word_19
        let v20 := ()
        let v21 := v20
        b6 unsigned_word
      else
        b5 unsigned_word
    else
      b5 unsigned_word
  else
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let v39 ← sign_extend v36 v37 v38
    let t_min_word := v39
    let v41 ← usize_SIGN_BIT
    let v42 := (UInt256.ofNat 1)
    let v43 ← checkedSub v41 v42
    let t_max_word := v43
    let v45 := t_min_word
    let t_min := v45
    let v47 := t_max_word
    let t_max := v47
    let v49 ← u64_IS_SIGNED
    let b13 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v3 => do
      let mut unsigned_word := unsigned_word
      let v77 := v3
      b3 unsigned_word v77
    if v49 then
      let v50 := word_9
      let x_51 := v50
      let v52 := UInt256.sltBool x_51 t_min
      let b16 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v4 => do
        let mut unsigned_word := unsigned_word
        let v61 := v4
        b13 unsigned_word v61
      if v52 then
        let v53 ← usize_from_word t_min_word
        let v54 := v53
        b16 unsigned_word v54
      else
        let v55 := UInt256.sgtBool x_51 t_max
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v5 => do
          let mut unsigned_word := unsigned_word
          let v60 := v5
          b16 unsigned_word v60
        if v55 then
          let v56 ← usize_from_word t_max_word
          let v57 := v56
          b19 unsigned_word v57
        else
          let v58 ← usize_from_word word_9
          let v59 := v58
          b19 unsigned_word v59
    else
      let v62 ← I256_SIGN_BIT
      let v63 := word_9 &&& v62
      let v64 := (UInt256.ofNat 0)
      let v65 := decide (v63 ≠ v64)
      let b22 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v6 => do
        let mut unsigned_word := unsigned_word
        let v76 := v6
        b13 unsigned_word v76
      if v65 then
        let v66 ← usize_from_word t_max_word
        let v67 := v66
        b22 unsigned_word v67
      else
        let v68 := word_9
        let x_69 := v68
        let v70 := UInt256.sgtBool x_69 t_max
        let b25 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v7 => do
          let mut unsigned_word := unsigned_word
          let v75 := v7
          b22 unsigned_word v75
        if v70 then
          let v71 ← usize_from_word t_max_word
          let v72 := v71
          b25 unsigned_word v72
        else
          let v73 ← usize_from_word word_9
          let v74 := v73
          b25 unsigned_word v74

def MAX_ALLOCATION : FeM UInt256 := do
  let v0 ← u64_max
  let v1 ← downcast_saturate_u64_usize v0
  return v1

def PANIC_OUT_OF_MEMORY : FeM UInt256 := do
  let v0 := (UInt256.ofNat 65)
  return v0

def alloc_bytes (len_0 : UInt256) : FeM UInt256 := do
  let len_1 := len_0
  let v2 ← MAX_ALLOCATION
  let v3 := decide (v2.toNat < len_1.toNat)
  if v3 then
    let v4 ← PANIC_OUT_OF_MEMORY
    let v5 ← panicCode v4
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← alloc len_0
    return v8

def FixedMemBuffer_alloc_uninit_32 : FeM FixedMemBuffer_32 := do
  let v0 := (UInt256.ofNat 32)
  let v1 ← alloc_bytes v0
  let v2 := (FixedMemBuffer_32.mk v1)
  return v2

def zero_bytes (ptr : UInt256) (len : UInt256) : FeM Unit := do
  let v2 ← zeroMem ptr len
  return ()

def FixedMemBuffer_alloc_32 : FeM FixedMemBuffer_32 := do
  let v0 ← FixedMemBuffer_alloc_uninit_32
  let buffer := v0
  let v2 := buffer.ptr
  let v3 := (UInt256.ofNat 32)
  let v4 ← zero_bytes v2 v3
  return buffer

def FixedMemBuffer_ptr_32 (self : FixedMemBuffer_32) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

def AbiFrame_from_raw_parts (ptr_0 : UInt256) (len_1 : UInt256) : FeM AbiFrame := do
  let ptr_2 := ptr_0
  let len_3 := len_1
  let v4 := (AbiFrame.mk ptr_2 len_3)
  return v4

def PANIC_INDEX_OUT_OF_BOUNDS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 50)
  return v0

def index_out_of_bounds : FeM Empty := do
  let v0 ← PANIC_INDEX_OUT_OF_BOUNDS
  let v1 ← panicCode v0
  nomatch v1

def word_to_usize (word : UInt256) : FeM UInt256 := do
  let v1 := word
  return v1

def ptr_offset (ptr : UInt256) (offset : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize offset
  let v3 := ptr + v2
  return v3

def AbiFrame_slice (self : AbiFrame) (offset_1 : UInt256) (len_2 : UInt256) : FeM AbiFrame := do
  let offset_3 := offset_1
  let v4 := self.len
  let v5 := decide (v4.toNat < offset_3.toNat)
  let b1 : FeM AbiFrame := do
    let v17 ← index_out_of_bounds
    nomatch v17
  if v5 then
    b1
  else
    let len_6 := len_2
    let v7 := self.len
    let offset_8 := offset_1
    let v9 := v7 - offset_8
    let v10 := decide (v9.toNat < len_6.toNat)
    if v10 then
      b1
    else
      let v11 := ()
      let v12 := v11
      let v13 := self.ptr
      let v14 ← ptr_offset v13 offset_1
      let len_15 := len_2
      let v16 := (AbiFrame.mk v14 len_15)
      return v16

def cast_u8_u256 (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let v2 := ptr_1
  return v2

def store_word (ptr : UInt256) (value : UInt256) : FeM Unit := do
  let v2 := value
  let v3 ← cast_u8_u256 ptr
  mstore v3 v2
  let v4 := ()
  return ()

def AbiFrame_store_word (self : AbiFrame) («at» : UInt256) (value : UInt256) : FeM Unit := do
  let v3 := (UInt256.ofNat 32)
  let v4 ← AbiFrame_slice self «at» v3
  let word := v4
  let v6 := word.ptr
  let v7 ← store_word v6 value
  return ()

def u256_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← AbiFrame_store_word out v2 self
  return ()

def downcast_unchecked_u256_usize (self : UInt256) : FeM UInt256 := do
  let v1 ← u256_to_word self
  let word := v1
  let v3 ← usize_from_word word
  return v3

def MemSlice_from_raw_parts_ref_u8 (ptr : UInt256) (len : UInt256) : FeM MemSlice_u8 := do
  let mut slice : MemSlice_u8 := default
  let v2 := (MemSlice_u8.mk ptr len)
  let slice_3 := v2
  slice := slice_3
  return slice

def MemSlice_ptr_u8 (self : MemSlice_u8) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

def MemSlice_len_u8 (self : MemSlice_u8) : FeM UInt256 := do
  let v1 := self.len
  return v1

def Evm_return_data (data : MemSlice_u8) : FeM Empty := do
  let v2 ← MemSlice_ptr_u8 data
  let v3 ← MemSlice_len_u8 data
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
  let buffer := v0
  let v2 := buffer.ptr
  let v3 := (UInt256.ofNat 64)
  let v4 ← zero_bytes v2 v3
  return buffer

def FixedMemBuffer_ptr_64 (self : FixedMemBuffer_64) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

def MemBuffer_new_uninit (len_0 : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes len_0
  let len_2 := len_0
  let len_3 := len_0
  let v4 := (MemBuffer.mk v1 len_2 len_3)
  return v4

def MemBuffer_new (len : UInt256) : FeM MemBuffer := do
  let v1 ← MemBuffer_new_uninit len
  let buffer := v1
  let v3 := buffer.ptr
  let v4 ← zero_bytes v3 len
  return buffer

def MemBuffer_ptr (self : MemBuffer) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

def MemBuffer_span_ref (self : MemBuffer) : FeM MemSlice_u8 := do
  let v1 := self.ptr
  let v2 := self.len
  let v3 ← MemSlice_from_raw_parts_ref_u8 v1 v2
  return v3

def const_u256 : FeM Bool := do
  let v0 := false
  return v0

def payload_size_u256 (self : UInt256) : FeM UInt256 := do
  let v1 ← u256_HEAD_SIZE
  return v1

def abi_payload_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 value
  return v1

def usize_max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  return v1

def allocation_len (len_0 : UInt256) : FeM UInt256 := do
  let len_1 := len_0
  let v2 ← usize_max
  let v3 := v2
  let v4 := decide (v3.toNat < len_1.toNat)
  if v4 then
    let v5 ← PANIC_OUT_OF_MEMORY
    let v6 ← panicCode v5
    nomatch v6
  else
    let v7 := ()
    let v8 := v7
    let v9 ← word_to_usize len_0
    return v9

def encode_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_u256 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  let v10 ← u256_encode_Sol value r1
  let buffer_11 := buffer
  return buffer_11

def abi_field_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 value
  let payload_size := v1
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_u256 value
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

def AbiFrame_tail (self : AbiFrame) (n_1 : UInt256) : FeM AbiFrame := do
  let n_2 := n_1
  let v3 := self.len
  let v4 := decide (v3.toNat < n_2.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := self.ptr
    let v9 ← ptr_offset v8 n_1
    let v10 := self.len
    let n_11 := n_1
    let v12 := v10 - n_11
    let v13 := (AbiFrame.mk v9 v12)
    return v13

def AbiFrame_len (self : AbiFrame) : FeM UInt256 := do
  let v1 := self.len
  return v1

def encode_field_Sol_u256 (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_u256 value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← u256_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← u256_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← u256_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def encode_single_root_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let v1 ← abi_single_root_size_u256 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_u256
  let v13 ← AbiFrame_tail out v11
  let v15 ← encode_field_Sol_u256 value out v10 v13
  let buffer_16 := buffer
  return buffer_16

def encode_return_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let v1 ← const_u256
  if v1 then
    let v2 ← encode_alloc_Sol_u256 value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_u256 value
    return v5

def Evm_return_bytes (data : MemSlice_u8) : FeM Empty := do
  let v2 ← Evm_return_data data
  nomatch v2

def Evm_return_value_Sol_u256 (value : UInt256) : FeM Empty := do
  let mut out_r0 : FixedMemBuffer_32 := default
  let mut out_r1 : FixedMemBuffer_64 := default
  let mut out_r2 : MemBuffer := default
  let mut out_r3 : MemBuffer := default
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
      let out_8 := v7
      out_r0 := out_8
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← u256_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let v14 ← u256_encode_Sol value r4
      let v16 ← FixedMemBuffer_ptr_32 out_r0
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
        let out_27 := v26
        out_r1 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← u256_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let v33 ← u256_encode_Sol value r5
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← u256_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← u256_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← MemBuffer_new v43
        let out_45 := v44
        out_r2 := out_45
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← u256_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let v51 ← u256_encode_Sol value r6
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_u256 value
    let out_58 := v57
    out_r3 := out_58
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

def CallData_word_at (self : CallData) (byte_offset_1 : UInt256) : FeM UInt256 := do
  let v2 := self.base
  let byte_offset_3 := byte_offset_1
  let v4 := v2 + byte_offset_3
  let v5 ← calldataload v4
  return v5

def Sol_decode_error : FeM Empty := do
  let v0 ← revertEmpty
  nomatch v0

def checked_tail_Sol (base_0 : UInt256) (tail_rel_1 : UInt256) : FeM UInt256 := do
  let tail_rel_2 := tail_rel_1
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  let v5 := ~~~v4
  let base_6 := base_0
  let v7 := v5 - base_6
  let v8 := decide (v7.toNat < tail_rel_2.toNat)
  if v8 then
    let v9 ← Sol_decode_error
    nomatch v9
  else
    let v10 := ()
    let v11 := v10
    let base_12 := base_0
    let tail_rel_13 := tail_rel_1
    let v14 := base_12 + tail_rel_13
    return v14

def checked_frame_end_Sol (pos_0 : UInt256) (head_size_1 : UInt256) (input_len_2 : UInt256) : FeM UInt256 := do
  let pos_3 := pos_0
  let input_len_4 := input_len_2
  let v5 := decide (input_len_4.toNat < pos_3.toNat)
  let b1 : FeM UInt256 := do
    let v16 ← Sol_decode_error
    nomatch v16
  if v5 then
    b1
  else
    let head_size_6 := head_size_1
    let input_len_7 := input_len_2
    let pos_8 := pos_0
    let v9 := input_len_7 - pos_8
    let v10 := decide (v9.toNat < head_size_6.toNat)
    if v10 then
      b1
    else
      let v11 := ()
      let v12 := v11
      let pos_13 := pos_0
      let head_size_14 := head_size_1
      let v15 := pos_13 + head_size_14
      return v15

def u256_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM UInt256 := do
  let v2 ← CallData_word_at input pos
  return v2

def decode_from_bounded_u256_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← u256_decode_from_Sol_CallData input pos
  return v5

def decode_field_from_prechecked_head_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← CallData_word_at input head_pos
    let tail_rel := v5
    let v7 ← checked_tail_Sol base tail_rel
    let tail_at := v7
    let v9 ← decode_from_bounded_u256_Sol_CallData input tail_at input_len
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← u256_decode_from_Sol_CallData input head_pos
    return v12

def decode_field_from_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM UInt256 := do
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v4 ← CallData_word_at input head_pos
    let tail_rel := v4
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← u256_decode_from_Sol_CallData input v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← u256_decode_from_Sol_CallData input head_pos
    return v10

def decode_msg_field_from_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_u256_CallData input base head_pos input_len
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_u256_CallData input base head_pos
    return v8

def SolDecoder_with_base (base_0 : UInt256) : FeM SolDecoder := do
  let base_1 := base_0
  let base_2 := base_0
  let v3 := (SolDecoder.mk base_1 base_2)
  return v3

def Sol_decoder_new : FeM SolDecoder := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← SolDecoder_with_base v0
  return v1

def usize_to_word (self_0 : UInt256) : FeM UInt256 := do
  let self_1 := self_0
  let v2 := self_1
  return v2

def downcast_truncate_usize_u64 (self : UInt256) : FeM UInt256 := do
  let v2 ← usize_to_word self
  let word := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def MemSlice_u8_size (self : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_len_u8 self
  let v2 ← downcast_truncate_usize_u64 v1
  return v2

def len_MemSlice_u8 (self : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_u8_size self
  let v2 := v1
  return v2

def MemSlice_u8_word_at (self : MemSlice_u8) (byte_offset_1 : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_len_u8 self
  let v3 := v2
  let len := v3
  let byte_offset_5 := byte_offset_1
  let v6 := decide (len.toNat ≤ byte_offset_5.toNat)
  if v6 then
    let v7 := (UInt256.ofNat 0)
    return v7
  else
    let v8 := ()
    let v9 := v8
    let byte_offset_10 := byte_offset_1
    let v11 := len - byte_offset_10
    let v12 ← word_to_usize v11
    let available := v12
    let v14 ← MemSlice_ptr_u8 self
    let v15 ← word_to_usize byte_offset_1
    let v16 := v14 + v15
    let v17 ← readPaddedWord v16 available
    return v17

def SolDecoder_read_word_MemSlice_u8 (self : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut self := self
  let v2 := self.pos
  let v3 := (UInt256.ofNat 32)
  let v4 ← checkedAdd v2 v3
  let new_pos := v4
  let v6 := self.pos
  let v7 := decide (new_pos.toNat < v6.toNat)
  let b1 : SolDecoder → FeM (UInt256 × SolDecoder) := fun self => do
    let mut self := self
    let v16 ← revertEmpty
    nomatch v16
  if v7 then
    b1 self
  else
    let v8 ← len_MemSlice_u8 input
    let v9 := decide (v8.toNat < new_pos.toNat)
    if v9 then
      b1 self
    else
      let v10 := ()
      let v11 := v10
      let v12 := self.pos
      let v13 ← MemSlice_u8_word_at input v12
      let w := v13
      self := { self with pos := new_pos }
      let v15 := ()
      return (w, self)

def SolDecoder_base (self : SolDecoder) : FeM UInt256 := do
  let v1 := self.base
  return v1

def SolDecoder_pos (self : SolDecoder) : FeM UInt256 := do
  let v1 := self.pos
  return v1

def SolDecoder_set_base (self : SolDecoder) (new_base_1 : UInt256) : FeM (Unit × SolDecoder) := do
  let mut self := self
  let new_base_2 := new_base_1
  self := { self with base := new_base_2 }
  let v3 := ()
  return ((), self)

def SolDecoder_set_pos (self : SolDecoder) (new_pos_1 : UInt256) : FeM (Unit × SolDecoder) := do
  let mut self := self
  let new_pos_2 := new_pos_1
  self := { self with pos := new_pos_2 }
  let v3 := ()
  return ((), self)

def u256_decode_payload_Sol_SolDecoder_MemSlice_u8 (d : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d := d
  let (v2, v2_0) ← SolDecoder_read_word_MemSlice_u8 d input
  d := v2_0
  return (v2, d)

def decode_field_Sol_u256_SolDecoder_MemSlice_u8 (d_0 : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d_0 := d_0
  let v2 ← u256_IS_DYNAMIC
  if v2 then
    let (v3, v3_0) ← SolDecoder_read_word_MemSlice_u8 d_0 input
    d_0 := v3_0
    let tail_rel := v3
    let v6 ← SolDecoder_base d_0
    let parent_base := v6
    let v9 ← SolDecoder_pos d_0
    let parent_pos := v9
    let v11 ← checked_tail_Sol parent_base tail_rel
    let (v12, v12_0) ← SolDecoder_set_base d_0 v11
    d_0 := v12_0
    let v14 ← SolDecoder_base d_0
    let (v15, v15_0) ← SolDecoder_set_pos d_0 v14
    d_0 := v15_0
    let (v17, v17_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 d_0 input
    d_0 := v17_0
    let value := v17
    let (v19, v19_0) ← SolDecoder_set_base d_0 parent_base
    d_0 := v19_0
    let (v20, v20_0) ← SolDecoder_set_pos d_0 parent_pos
    d_0 := v20_0
    return (value, d_0)
  else
    let v21 := ()
    let v22 := v21
    let (v24, v24_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 d_0 input
    d_0 := v24_0
    return (v24, d_0)

def tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 (d : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d := d
  let (v3, v3_0) ← decode_field_Sol_u256_SolDecoder_MemSlice_u8 d input
  d := v3_0
  let v0 := v3
  let v5 := v0
  return (v5, d)

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

def u256_encoded_len (self : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def u256_IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def downcast_saturate_u256_usize (self : UInt256) : FeM UInt256 := do
  let mut unsigned_word : UInt256 := default
  let v8 ← u256_to_word self
  let word_9 := v8
  let v10 ← usize_IS_SIGNED
  let v11 := !v10
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v1 => do
    let mut unsigned_word := unsigned_word
    return v1
  if v11 then
    let word_12 := word_9
    unsigned_word := word_12
    let v13 ← u256_IS_SIGNED
    let b6 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      let unsigned_word_24 := unsigned_word
      let unsigned_word_25 := unsigned_word_24
      let v26 ← usize_MASK
      let v27 := decide (v26.toNat < unsigned_word_25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v2 => do
        let mut unsigned_word := unsigned_word
        let v35 := v2
        b3 unsigned_word v35
      if v27 then
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        let v30 := v29
        b10 unsigned_word v30
      else
        let unsigned_word_31 := unsigned_word
        let unsigned_word_32 := unsigned_word_31
        let v33 ← usize_from_word unsigned_word_32
        let v34 := v33
        b10 unsigned_word v34
    let b5 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      let v22 := ()
      let v23 := v22
      b6 unsigned_word
    if v13 then
      let v14 ← I256_SIGN_BIT
      let v15 := word_9 &&& v14
      let v16 := (UInt256.ofNat 0)
      let v17 := decide (v15 ≠ v16)
      if v17 then
        let v18 := (UInt256.ofNat 0)
        let unsigned_word_19 := v18
        unsigned_word := unsigned_word_19
        let v20 := ()
        let v21 := v20
        b6 unsigned_word
      else
        b5 unsigned_word
    else
      b5 unsigned_word
  else
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let v39 ← sign_extend v36 v37 v38
    let t_min_word := v39
    let v41 ← usize_SIGN_BIT
    let v42 := (UInt256.ofNat 1)
    let v43 ← checkedSub v41 v42
    let t_max_word := v43
    let v45 := t_min_word
    let t_min := v45
    let v47 := t_max_word
    let t_max := v47
    let v49 ← u256_IS_SIGNED
    let b13 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v3 => do
      let mut unsigned_word := unsigned_word
      let v77 := v3
      b3 unsigned_word v77
    if v49 then
      let v50 := word_9
      let x_51 := v50
      let v52 := UInt256.sltBool x_51 t_min
      let b16 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v4 => do
        let mut unsigned_word := unsigned_word
        let v61 := v4
        b13 unsigned_word v61
      if v52 then
        let v53 ← usize_from_word t_min_word
        let v54 := v53
        b16 unsigned_word v54
      else
        let v55 := UInt256.sgtBool x_51 t_max
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v5 => do
          let mut unsigned_word := unsigned_word
          let v60 := v5
          b16 unsigned_word v60
        if v55 then
          let v56 ← usize_from_word t_max_word
          let v57 := v56
          b19 unsigned_word v57
        else
          let v58 ← usize_from_word word_9
          let v59 := v58
          b19 unsigned_word v59
    else
      let v62 ← I256_SIGN_BIT
      let v63 := word_9 &&& v62
      let v64 := (UInt256.ofNat 0)
      let v65 := decide (v63 ≠ v64)
      let b22 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v6 => do
        let mut unsigned_word := unsigned_word
        let v76 := v6
        b13 unsigned_word v76
      if v65 then
        let v66 ← usize_from_word t_max_word
        let v67 := v66
        b22 unsigned_word v67
      else
        let v68 := word_9
        let x_69 := v68
        let v70 := UInt256.sgtBool x_69 t_max
        let b25 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v7 => do
          let mut unsigned_word := unsigned_word
          let v75 := v7
          b22 unsigned_word v75
        if v70 then
          let v71 ← usize_from_word t_max_word
          let v72 := v71
          b25 unsigned_word v72
        else
          let v73 ← usize_from_word word_9
          let v74 := v73
          b25 unsigned_word v74

def u256_write_key (ptr : UInt256) (self : UInt256) : FeM Unit := do
  let v2 ← mstore ptr self
  return ()

def keccak256_1 (data : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_ptr_u8 data
  let v2 ← MemSlice_len_u8 data
  let v3 := v2
  let v4 ← keccak256 v1 v3
  return v4

def cast_u8_u8 (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let ptr_2 := ptr_1
  return ptr_2

def dealloc_u8 (ptr : UInt256) : FeM Unit := do
  let v1 ← cast_u8_u8 ptr
  let v2 := ()
  return ()

def MemBuffer_drop (self : MemBuffer) : FeM Unit := do
  let v1 := self.ptr
  let v2 ← dealloc_u8 v1
  return ()

def mapping_slot_u256 (root : UInt256) (key : UInt256) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let v2 ← u256_encoded_len key
  let key_len := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd key_len v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let buffer_8 := v7
  buffer := buffer_8
  let v10 ← MemBuffer_ptr buffer
  let scratch := v10
  let v12 ← u256_write_key scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let v15 ← mstore v14 root
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd key_len v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

def StorageMap_locate_u256_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let v3 ← mapping_slot_u256 root key
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

def StoragePackedArray_get_8 (self : Place) (i_1 : UInt256) : FeM UInt256 := do
  let i_2 := i_1
  let v3 ← StoragePackedArray_LANES_PER_SLOT_8
  let v4 ← checkedRem i_2 v3
  let v5 := (UInt256.ofNat 8)
  let v6 := v4 * v5
  let bit_pos := v6
  let i_8 := i_1
  let v9 ← StoragePackedArray_LANES_PER_SLOT_8
  let v10 ← checkedDiv i_8 v9
  let l0 ← StorageMap_locate_u256_u256 (self.field Layout.StoragePackedArray_8 0) (self.field Layout.StoragePackedArray_8 0).slot v10
  let v11 ← load (Place.mk .storage l0.slot)
  let v12 := v11 >>> bit_pos
  let v13 ← StoragePackedArray_LANE_MASK_8
  let v14 := v12 &&& v13
  return v14

def StoragePackedArray_set_8 (self : Place) (i_1 : UInt256) (value_2 : UInt256) : FeM Unit := do
  let i_3 := i_1
  let v4 ← StoragePackedArray_LANES_PER_SLOT_8
  let v5 ← checkedRem i_3 v4
  let v6 := (UInt256.ofNat 8)
  let v7 := v5 * v6
  let bit_pos := v7
  let v9 ← StoragePackedArray_LANE_MASK_8
  let v10 := v9 <<< bit_pos
  let lane_mask_shifted := v10
  let i_12 := i_1
  let v13 ← StoragePackedArray_LANES_PER_SLOT_8
  let v14 ← checkedDiv i_12 v13
  let l0 ← StorageMap_locate_u256_u256 (self.field Layout.StoragePackedArray_8 0) (self.field Layout.StoragePackedArray_8 0).slot v14
  let v15 := (Place.mk .storage l0.slot)
  let word := v15
  let v17 ← load word
  let v18 ← U256_MASK
  let v19 := v18 ^^^ lane_mask_shifted
  let v20 := v17 &&& v19
  let value_21 := value_2
  let v22 ← StoragePackedArray_LANE_MASK_8
  let v23 := value_21 &&& v22
  let v24 := v23 <<< bit_pos
  let v25 := v20 ||| v24
  store word v25
  let v26 := ()
  return ()

def Evm_caller : FeM UInt256 := do
  let v1 ← caller
  let v2 := v1
  return v2

def Address_encoded_len (self : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def Address_write_key (ptr : UInt256) (self : UInt256) : FeM Unit := do
  let v2 := self
  let v3 ← u256_write_key ptr v2
  return ()

def mapping_slot_Address (root : UInt256) (key : UInt256) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let v2 ← Address_encoded_len key
  let key_len := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd key_len v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let buffer_8 := v7
  buffer := buffer_8
  let v10 ← MemBuffer_ptr buffer
  let scratch := v10
  let v12 ← Address_write_key scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let v15 ← mstore v14 root
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd key_len v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

def StorageMap_locate_Address_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let v3 ← mapping_slot_Address root key
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

def dynamic_payload_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← u256_IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_u256 value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 := (UInt256.ofNat 0)
    return v5

def tuple_payload_size_u256 (self : UInt256) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE_u256
  let v2 := self
  let v3 ← dynamic_payload_size_u256 v2
  let v4 := v1 + v3
  return v4

def abi_payload_size_tuple_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← tuple_payload_size_u256 value
  return v1

def tuple_encode_Sol_u256 (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let v3 := (UInt256.ofNat 0)
  let v4 ← tuple_HEAD_SIZE_u256
  let v5 ← AbiFrame_tail out v4
  let v0_6 := self
  let v7 ← encode_field_Sol_u256 v0_6 out v3 v5
  return ()

def encode_alloc_Sol_tuple_u256 (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_tuple_u256 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let v11 ← tuple_encode_Sol_u256 value r2
  let buffer_12 := buffer
  return buffer_12

def encode_abi_payload_tuple_u256 (value : UInt256) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_tuple_u256 value
  return v1

def Address_as_topic (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

def Evm_log3 (data : MemSlice_u8) (topic0 : UInt256) (topic1 : UInt256) (topic2 : UInt256) : FeM Unit := do
  let v5 ← MemSlice_ptr_u8 data
  let v6 ← MemSlice_len_u8 data
  let v7 := v6
  let v8 ← logData v5 v7 #[topic0, topic1, topic2]
  return ()

def StorageMap_locate_Address_StorageMap_Address_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let v3 ← mapping_slot_Address root key
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

def Address_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM UInt256 := do
  let v2 ← CallData_word_at input pos
  let word := v2
  let v4 := (UInt256.ofNat 160)
  let v5 := word >>> v4
  let v6 := (UInt256.ofNat 0)
  let v7 := decide (v5 ≠ v6)
  if v7 then
    let v8 ← revertEmpty
    nomatch v8
  else
    let v9 := ()
    let v10 := v9
    let v11 := word
    return v11

def decode_from_bounded_Address_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← Address_HEAD_SIZE
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← Address_decode_from_Sol_CallData input pos
  return v5

def decode_field_from_prechecked_head_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    let v5 ← CallData_word_at input head_pos
    let tail_rel := v5
    let v7 ← checked_tail_Sol base tail_rel
    let tail_at := v7
    let v9 ← decode_from_bounded_Address_Sol_CallData input tail_at input_len
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← Address_decode_from_Sol_CallData input head_pos
    return v12

def decode_field_from_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM UInt256 := do
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v4 ← CallData_word_at input head_pos
    let tail_rel := v4
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← Address_decode_from_Sol_CallData input v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← Address_decode_from_Sol_CallData input head_pos
    return v10

def decode_msg_field_from_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_Address_CallData input base head_pos input_len
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_Address_CallData input base head_pos
    return v8

def bool_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def bool_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def bool_encode_Sol (self_0 : Bool) (out : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let self_3 := self_0
  let v4 := Bool.toUInt256 self_3
  let v5 ← AbiFrame_store_word out v2 v4
  return ()

def const_bool : FeM Bool := do
  let v0 := false
  return v0

def payload_size_bool (self : Bool) : FeM UInt256 := do
  let v1 ← bool_HEAD_SIZE
  return v1

def abi_payload_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← payload_size_bool value
  return v1

def encode_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_bool value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  let v10 ← bool_encode_Sol value r1
  let buffer_11 := buffer
  return buffer_11

def abi_field_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← payload_size_bool value
  let payload_size := v1
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← abi_field_size_bool value
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

def encode_field_Sol_bool (value : Bool) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_bool value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← bool_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← bool_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← bool_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def encode_single_root_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let v1 ← abi_single_root_size_bool value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_bool
  let v13 ← AbiFrame_tail out v11
  let v15 ← encode_field_Sol_bool value out v10 v13
  let buffer_16 := buffer
  return buffer_16

def encode_return_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let v1 ← const_bool
  if v1 then
    let v2 ← encode_alloc_Sol_bool value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_bool value
    return v5

def Evm_return_value_Sol_bool (value : Bool) : FeM Empty := do
  let mut out_r0 : FixedMemBuffer_32 := default
  let mut out_r1 : FixedMemBuffer_64 := default
  let mut out_r2 : MemBuffer := default
  let mut out_r3 : MemBuffer := default
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
      let out_8 := v7
      out_r0 := out_8
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← bool_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let v14 ← bool_encode_Sol value r4
      let v16 ← FixedMemBuffer_ptr_32 out_r0
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
        let out_27 := v26
        out_r1 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← bool_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let v33 ← bool_encode_Sol value r5
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← bool_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← bool_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← MemBuffer_new v43
        let out_45 := v44
        out_r2 := out_45
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← bool_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let v51 ← bool_encode_Sol value r6
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_bool value
    let out_58 := v57
    out_r3 := out_58
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

def tuple_HEAD_SIZE_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def tuple_decode_from_prechecked_head_Sol_u256_u256_CallData (input : CallData) (pos_1 : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let mut head_pos : UInt256 := default
  let pos_3 := pos_1
  let pos_4 := pos_3
  head_pos := pos_4
  let head_pos_5 := head_pos
  let head_pos_6 := head_pos_5
  let v7 ← decode_field_from_prechecked_head_Sol_u256_CallData input pos_1 head_pos_6 input_len
  let v0 := v7
  let v9 ← abi_field_head_size_u256
  let head_pos_10 := head_pos
  let v11 ← checkedAdd head_pos_10 v9
  let head_pos_12 := v11
  head_pos := head_pos_12
  let v13 := ()
  let head_pos_14 := head_pos
  let head_pos_15 := head_pos_14
  let v16 ← decode_field_from_prechecked_head_Sol_u256_CallData input pos_1 head_pos_15 input_len
  let v1 := v16
  let v18 := (v0, v1)
  return v18

def decode_frame_from_Sol_tuple_u256_u256_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← tuple_HEAD_SIZE_u256_u256
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← tuple_decode_from_prechecked_head_Sol_u256_u256_CallData input pos input_len
  return v5

def decode_input_at_tuple_u256_u256_CallData (input : CallData) (base : UInt256) : FeM (UInt256 × UInt256) := do
  let v2 ← len_CallData input
  let input_len := v2
  let v4 ← decode_frame_from_Sol_tuple_u256_u256_CallData input base input_len
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

def MemBuffer_len (self : MemBuffer) : FeM UInt256 := do
  let v1 := self.len
  return v1

def revert_u256 (value : UInt256) : FeM Empty := do
  let mut result : MemBuffer := default
  let v1 ← encode_single_root_alloc_Sol_u256 value
  let result_2 := v1
  result := result_2
  let v4 ← MemBuffer_ptr result
  let v6 ← MemBuffer_len result
  let v7 := v6
  let v8 ← revertData v4 v7
  nomatch v8

structure Wei where
  amount : UInt256
deriving Inhabited

def i256_to_word (self : UInt256) : FeM UInt256 := do
  let v1 := self
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

def i8_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 8) word
  return v1

def downcast_truncate_i256_i8 (self : UInt256) : FeM UInt256 := do
  let v2 ← i256_to_word self
  let word := v2
  let v4 ← i8_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i8_MASK
    let v6 ← i8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i8_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← i8_MASK
    let v11 := word &&& v10
    let v12 ← i8_from_word v11
    let v13 := v12
    b3 v13

def downcast_unchecked_u256_i256 (self : UInt256) : FeM UInt256 := do
  let v1 ← u256_to_word self
  let word := v1
  let v3 ← i256_from_word word
  return v3

def Evm_address : FeM UInt256 := do
  let v1 ← selfAddress
  let v2 := v1
  return v2

def Address_word (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

def wei (n_0 : UInt256) : FeM Wei := do
  let n_1 := n_0
  let v2 := (Wei.mk n_1)
  return v2

def Evm_value : FeM Wei := do
  let v1 ← callvalue
  let v2 ← wei v1
  return v2

def Wei_as_wei (self : Wei) : FeM UInt256 := do
  let v1 := self.amount
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

def tuple_encoded_len_u256_u256 (self : (UInt256 × UInt256)) : FeM UInt256 := do
  let v1 := self.fst
  let v2 ← u256_encoded_len v1
  let v3 := self.snd
  let v4 ← u256_encoded_len v3
  let v5 ← checkedAdd v2 v4
  return v5

def tuple_write_key_u256_u256 (ptr : UInt256) (self : (UInt256 × UInt256)) : FeM Unit := do
  let v2 := self.fst
  let v3 ← u256_encoded_len v2
  let a_len := v3
  let v5 := self.fst
  let v6 ← u256_write_key ptr v5
  let v7 ← downcast_unchecked_u256_usize a_len
  let v8 := ptr + v7
  let v9 := self.snd
  let v10 ← u256_write_key v8 v9
  return ()

def mapping_slot_tuple_u256_u256 (root : UInt256) (key : (UInt256 × UInt256)) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let v2 ← tuple_encoded_len_u256_u256 key
  let key_len := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd key_len v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let buffer_8 := v7
  buffer := buffer_8
  let v10 ← MemBuffer_ptr buffer
  let scratch := v10
  let v12 ← tuple_write_key_u256_u256 scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let v15 ← mstore v14 root
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd key_len v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

def StorageMap_locate_tuple_u256_u256_bool (self : Place) (root : UInt256) (key : (UInt256 × UInt256)) : FeM Location := do
  let v3 ← mapping_slot_tuple_u256_u256 root key
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def bool_encoded_len (self : Bool) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def bool_write_key (ptr : UInt256) (self_1 : Bool) : FeM Unit := do
  let self_3 := self_1
  let b3 : UInt256 → FeM Unit := fun v2 => do
    let v8 ← u256_write_key ptr v2
    return ()
  if self_3 then
    let v4 := (UInt256.ofNat 1)
    let v5 := v4
    b3 v5
  else
    let v6 := (UInt256.ofNat 0)
    let v7 := v6
    b3 v7

def mapping_slot_bool (root : UInt256) (key : Bool) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let v2 ← bool_encoded_len key
  let key_len := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd key_len v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← MemBuffer_new_uninit v6
  let buffer_8 := v7
  buffer := buffer_8
  let v10 ← MemBuffer_ptr buffer
  let scratch := v10
  let v12 ← bool_write_key scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let v15 ← mstore v14 root
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd key_len v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

def StorageMap_locate_bool_u256 (self : Place) (root : UInt256) (key : Bool) : FeM Location := do
  let v3 ← mapping_slot_bool root key
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def calldata_ptr_Evm_array_u256_2 (offset_1 : UInt256) : FeM UInt256 := do
  let offset_2 := offset_1
  let v3 := offset_2
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

def array_start_u256_2 (self : (Vector UInt256 2)) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def array_has_u256_2 (self : (Vector UInt256 2)) (c_1 : UInt256) : FeM Bool := do
  let c_2 := c_1
  let v3 := (UInt256.ofNat 2)
  let v4 := decide (c_2.toNat < v3.toNat)
  return v4

def array_at_ref_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM UInt256 := do
  checkIndex c 2
  return self[c.toNat]!

def usize_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self + other
  return v2

def array_next_u256_2 (self : (Vector UInt256 2)) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
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

def DynString_span_ref (self : DynString) : FeM MemSlice_u8 := do
  let v1 := self.data
  let v2 := (UInt256.ofNat 32)
  let v3 := v1 + v2
  let v4 := self.len
  let v5 ← MemSlice_from_raw_parts_ref_u8 v3 v4
  return v5

def PANIC_STORAGE_ENCODING : FeM UInt256 := do
  let v0 := (UInt256.ofNat 34)
  return v0

def Panic_from (code_0 : UInt256) : FeM Panic := do
  let code_1 := code_0
  let v2 := (Panic.mk code_1)
  return v2

def Panic_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def Panic_payload_size (self : Panic) : FeM UInt256 := do
  let v1 ← Panic_HEAD_SIZE
  let v2 := self.code
  let v3 ← dynamic_payload_size_u256 v2
  let v4 ← checkedAdd v1 v3
  return v4

def abi_payload_size_Panic (value : Panic) : FeM UInt256 := do
  let v1 ← Panic_payload_size value
  return v1

def MemBuffer_set_len (self : MemBuffer) (len_1 : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let len_2 := len_1
  let v3 := self.capacity
  let v4 := decide (v3.toNat < len_2.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let len_8 := len_1
    self := { self with len := len_8 }
    let v9 := ()
    return ((), self)

def Panic_encode (self : Panic) (__fe_out : AbiFrame) : FeM Unit := do
  let mut __fe_tails : AbiFrame := default
  let mut __fe_head_pos : UInt256 := default
  let v2 ← Panic_HEAD_SIZE
  let v3 ← AbiFrame_tail __fe_out v2
  let __fe_tails_4 := v3
  __fe_tails := __fe_tails_4
  let v5 := (UInt256.ofNat 0)
  let __fe_head_pos_6 := v5
  __fe_head_pos := __fe_head_pos_6
  let v7 := self.code
  let __fe_head_pos_8 := __fe_head_pos
  let __fe_head_pos_9 := __fe_head_pos_8
  let __fe_tails_10 := __fe_tails
  let __fe_tails_11 := __fe_tails_10
  let v12 ← encode_field_Sol_u256 v7 __fe_out __fe_head_pos_9 __fe_tails_11
  return ()

def encode_calldata_Panic (selector_0 : UInt256) (message : Panic) : FeM MemBuffer := do
  let mut message := message
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v2 := (UInt256.ofNat 4)
  let selector_size := v2
  let v5 ← abi_payload_size_Panic message
  let payload_size := v5
  let v7 ← downcast_unchecked_u256_usize payload_size
  let v8 := selector_size + v7
  let len_9 := v8
  let v10 ← Panic_HEAD_SIZE
  let v11 := (UInt256.ofNat 28)
  let v12 := decide (v10.toNat < v11.toNat)
  let b3 : Panic → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun message buffer r2 v1 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    let capacity := v1
    let v19 ← MemBuffer_new_uninit capacity
    let buffer_20 := v19
    buffer := buffer_20
    let v22 ← MemBuffer_ptr buffer
    let v23 := v22 + selector_size
    let payload := v23
    let (v26, v26_0) ← MemBuffer_set_len buffer len_9
    buffer := v26_0
    let v28 ← MemBuffer_ptr buffer
    let selector_29 := selector_0
    let v30 := selector_29
    let v31 := (UInt256.ofNat 224)
    let v32 := v30 <<< v31
    let v33 ← mstore v28 v32
    let v34 ← downcast_unchecked_u256_usize payload_size
    let v35 ← zero_bytes payload v34
    let v36 ← AbiFrame_from_raw_parts payload payload_size
    r2 := v36
    let v39 ← Panic_encode message r2
    let buffer_40 := buffer
    return buffer_40
  let b2 : Panic → MemBuffer → AbiFrame → FeM MemBuffer := fun message buffer r2 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    let len_17 := len_9
    b3 message buffer r2 len_17
  if v12 then
    let v13 := (UInt256.ofNat 32)
    let v14 := decide (len_9.toNat < v13.toNat)
    if v14 then
      let v15 := (UInt256.ofNat 32)
      let v16 := v15
      b3 message buffer r2 v16
    else
      b2 message buffer r2
  else
    b2 message buffer r2

def encode_error_Panic (error_0 : Panic) : FeM MemBuffer := do
  let v1 := (UInt256.ofNat 1313373041) -- a constant, as the compiler evaluates it
  let error_2 := error_0
  let v3 ← encode_calldata_Panic v1 error_2
  return v3

def revert_error_Panic (error_0 : Panic) : FeM Empty := do
  let mut buffer : MemBuffer := default
  let error_1 := error_0
  let v2 ← encode_error_Panic error_1
  let buffer_3 := v2
  buffer := buffer_3
  let v5 ← MemBuffer_ptr buffer
  let v7 ← MemBuffer_len buffer
  let v8 := v7
  let v9 ← revertData v5 v8
  nomatch v9

def keccak_words_1 (words : (Vector UInt256 1)) : FeM UInt256 := do
  let v1 := keccakWords words
  return v1

def array_data_slot (root_0 : UInt256) : FeM UInt256 := do
  let mut r0 : (Vector UInt256 1) := default
  let root_1 := root_0
  let v2 := #v[root_1]
  r0 := v2
  let v4 ← keccak_words_1 r0
  return v4

def u256_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self + other
  return v2

def u256_max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)
  return v0

def DynString_drop (self : DynString) : FeM Unit := do
  let v1 := self.data
  let v2 ← dealloc_u8 v1
  return ()

def u8_BITS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 8)
  return v0

def field_shift (offset_0 : UInt256) (bits_1 : UInt256) : FeM UInt256 := do
  let offset_2 := offset_0
  let v3 := (UInt256.ofNat 32)
  let v4 := decide (v3.toNat ≤ offset_2.toNat)
  let b1 : FeM UInt256 := do
    let v17 ← revertEmpty
    nomatch v17
  if v4 then
    b1
  else
    let bits_5 := bits_1
    let v6 := (UInt256.ofNat 256)
    let offset_7 := offset_0
    let v8 := (UInt256.ofNat 8)
    let v9 ← checkedMul offset_7 v8
    let v10 ← checkedSub v6 v9
    let v11 := decide (v10.toNat < bits_5.toNat)
    if v11 then
      b1
    else
      let v12 := ()
      let v13 := v12
      let offset_14 := offset_0
      let v15 := (UInt256.ofNat 8)
      let v16 ← checkedMul offset_14 v15
      return v16

def u8_to_packed (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

def field_mask (bits_0 : UInt256) : FeM UInt256 := do
  let bits_2 := bits_0
  let v3 := (UInt256.ofNat 256)
  let v4 := decide (v3.toNat ≤ bits_2.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u256_max
    let v6 := v5
    b3 v6
  else
    let v7 := (UInt256.ofNat 1)
    let v8 := v7
    let bits_9 := bits_0
    let v10 := v8 <<< bits_9
    let v11 := (UInt256.ofNat 1)
    let v12 ← checkedSub v10 v11
    let v13 := v12
    b3 v13

def SolWord_set_u8 (self : SolWord) (offset : UInt256) (value : UInt256) : FeM (Unit × SolWord) := do
  let mut self := self
  let v3 ← u8_BITS
  let v4 ← field_shift offset v3
  let shift := v4
  let v6 ← u8_BITS
  let v7 ← field_mask v6
  let v8 := v7 <<< shift
  let mask := v8
  let v10 := self.word
  let v11 := ~~~mask
  let v12 := v10 &&& v11
  let v13 ← u8_to_packed value
  let v14 := v13 <<< shift
  let v15 := v14 &&& mask
  let v16 := v12 ||| v15
  self := { self with word := v16 }
  let v17 := ()
  return ((), self)

def round_up_words (len_0 : UInt256) : FeM UInt256 := do
  let len_1 := len_0
  let v2 := (UInt256.ofNat 0)
  let v3 := decide (len_1 = v2)
  if v3 then
    let v4 := (UInt256.ofNat 0)
    return v4
  else
    let v5 := ()
    let v6 := v5
    let len_7 := len_0
    let v8 := (UInt256.ofNat 1)
    let v9 ← checkedSub len_7 v8
    let v10 := (UInt256.ofNat 32)
    let v11 ← checkedDiv v9 v10
    let v12 := (UInt256.ofNat 1)
    let v13 ← checkedAdd v11 v12
    let v14 := (UInt256.ofNat 32)
    let v15 ← checkedMul v13 v14
    return v15

def mem_word_at (base : UInt256) (offset : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize offset
  let v3 := base + v2
  let v4 ← cast_u8_u256 v3
  let v5 ← mload v4
  let v6 := v5
  return v6

def DynString_from_validated_abi_tail (data_0 : UInt256) (len_1 : UInt256) (size_2 : UInt256) : FeM DynString := do
  let data_3 := data_0
  let len_4 := len_1
  let size_5 := size_2
  let v6 := (DynString.mk data_3 len_4 size_5)
  return v6

def DynString_from_abi_tail (data : UInt256) (len_1 : UInt256) (size_2 : UInt256) : FeM DynString := do
  let v3 := (UInt256.ofNat 32)
  let len_4 := len_1
  let v5 := len_4
  let v6 ← round_up_words v5
  let v7 ← checkedAdd v3 v6
  let expected_size := v7
  let size_9 := size_2
  let v10 := decide (size_9 ≠ expected_size)
  let b1 : FeM DynString := do
    let v19 ← trap
    nomatch v19
  if v10 then
    b1
  else
    let v11 := (UInt256.ofNat 0)
    let v12 ← mem_word_at data v11
    let len_13 := len_1
    let v14 := len_13
    let v15 := decide (v12 ≠ v14)
    if v15 then
      b1
    else
      let v16 := ()
      let v17 := v16
      let v18 ← DynString_from_validated_abi_tail data len_1 size_2
      return v18

def Evm_selfbalance : FeM Wei := do
  let v1 ← selfbalance
  let v2 ← wei v1
  return v2

def Address_eq (self : UInt256) (other : UInt256) : FeM Bool := do
  let v2 := self
  let v3 := other
  let v4 := decide (v2 = v3)
  return v4

def ne_Address_Address (self : UInt256) (other : UInt256) : FeM Bool := do
  let v2 ← Address_eq self other
  let v3 := !v2
  return v3

def MemBuffer_with_capacity (capacity_0 : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes capacity_0
  let v2 := (UInt256.ofNat 0)
  let capacity_3 := capacity_0
  let v4 := (MemBuffer.mk v1 v2 capacity_3)
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

def MemBuffer_capacity (self : MemBuffer) : FeM UInt256 := do
  let v1 := self.capacity
  return v1

def finish_raw_call (success_0 : Bool) (ret_1 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut ret_1 := ret_1
  let v3 ← returndatasize
  let returndata_len := v3
  let v6 ← MemBuffer_capacity ret_1
  let v7 := v6
  let v8 := decide (returndata_len.toNat < v7.toNat)
  let b3 : MemBuffer → UInt256 → FeM (RawCallOutcome × MemBuffer) := fun ret_1 v2 => do
    let mut ret_1 := ret_1
    let written := v2
    let (v15, v15_0) ← MemBuffer_set_len ret_1 written
    ret_1 := v15_0
    let success_16 := success_0
    let v17 := (RawCallOutcome.mk success_16 returndata_len)
    return (v17, ret_1)
  if v8 then
    let v9 ← downcast_unchecked_u256_usize returndata_len
    let v10 := v9
    b3 ret_1 v10
  else
    let v12 ← MemBuffer_capacity ret_1
    let v13 := v12
    b3 ret_1 v13

def raw_call_Evm (addr : UInt256) (gas_1 : UInt256) (value : Wei) (args : MemSlice_u8) (ret_5 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut ret_5 := ret_5
  let v6 := addr
  let v7 ← Wei_as_wei value
  let v8 ← MemSlice_ptr_u8 args
  let v9 ← MemSlice_len_u8 args
  let v10 := v9
  let v12 ← MemBuffer_ptr ret_5
  let v14 ← MemBuffer_capacity ret_5
  let v15 := v14
  let v16 ← Fe.EVM.call gas_1 v6 v7 v8 v10 v12 v15
  let v17 := (UInt256.ofNat 0)
  let v18 := decide (v16 ≠ v17)
  let success := v18
  let (v21, v21_0) ← finish_raw_call success ret_5
  ret_5 := v21_0
  return (v21, ret_5)

def send_value_Evm (addr : UInt256) (gas_1 : UInt256) (value : Wei) : FeM RawCallOutcome := do
  let mut addr := addr
  let mut ret : MemBuffer := default
  let v3 ← MemBuffer_empty
  let ret_4 := v3
  ret := ret_4
  let v5 ← MemSlice_empty_u8
  let addr_7 := addr
  let (v9, v9_0) ← raw_call_Evm addr_7 gas_1 value v5 ret
  ret := v9_0
  let _ ← MemBuffer_drop ret
  return v9

def RawCallOutcome_success (self : RawCallOutcome) : FeM Bool := do
  let v1 := self.success
  return v1

def Evm_log2 (data : MemSlice_u8) (topic0 : UInt256) (topic1 : UInt256) : FeM Unit := do
  let v4 ← MemSlice_ptr_u8 data
  let v5 ← MemSlice_len_u8 data
  let v6 := v5
  let v7 ← logData v4 v6 #[topic0, topic1]
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

def u8_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := truncate 8 word
  return v1

def downcast_truncate_u256_u8 (self : UInt256) : FeM UInt256 := do
  let v2 ← u256_to_word self
  let word := v2
  let v4 ← u8_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u8_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u8_MASK
    let v11 := word &&& v10
    let v12 ← u8_from_word v11
    let v13 := v12
    b3 v13

def u8_from_packed (bits : UInt256) : FeM UInt256 := do
  let v1 ← downcast_truncate_u256_u8 bits
  return v1

def SolWord_get_u8 (self : SolWord) (offset : UInt256) : FeM UInt256 := do
  let v2 ← u8_BITS
  let v3 ← field_shift offset v2
  let shift := v3
  let v5 := self.word
  let v6 := v5 >>> shift
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

def DynString_encoded_span (self : DynString) : FeM MemSlice_u8 := do
  let v1 := self.data
  let v2 := self.size
  let v3 ← word_to_usize v2
  let v4 ← MemSlice_from_raw_parts_ref_u8 v1 v3
  return v4

def copy_raw (dest : UInt256) (source : UInt256) (len : UInt256) : FeM Unit := do
  let v3 ← mcopy dest source len
  return ()

def copy (dest : UInt256) (source : MemSlice_u8) : FeM Unit := do
  let v2 ← MemSlice_ptr_u8 source
  let v3 ← MemSlice_len_u8 source
  let v4 ← copy_raw dest v2 v3
  return ()

def AbiFrame_copy_from (self : AbiFrame) («at» : UInt256) (source : MemSlice_u8) : FeM Unit := do
  let v3 ← MemSlice_len_u8 source
  let v4 := v3
  let v5 ← AbiFrame_slice self «at» v4
  let dest := v5
  let v7 := dest.ptr
  let v8 ← copy v7 source
  return ()

def DynString_encode_Sol (self : DynString) (out : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← DynString_encoded_span self
  let v5 ← AbiFrame_copy_from out v2 v3
  return ()

def const_DynString : FeM Bool := do
  let v0 := false
  return v0

def DynString_encoded_size (self : DynString) : FeM UInt256 := do
  let v1 := self.size
  return v1

def DynString_payload_size (self : DynString) : FeM UInt256 := do
  let v1 ← DynString_encoded_size self
  return v1

def abi_payload_size_DynString (value : DynString) : FeM UInt256 := do
  let v1 ← DynString_payload_size value
  return v1

def encode_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_DynString value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let v11 ← DynString_encode_Sol value r2
  let _ ← DynString_drop value
  let buffer_12 := buffer
  return buffer_12

def abi_field_size_DynString (value : DynString) : FeM UInt256 := do
  let v1 ← DynString_payload_size value
  let payload_size := v1
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_DynString (value : DynString) : FeM UInt256 := do
  let v1 ← abi_field_size_DynString value
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

def encode_field_Sol_DynString (value : DynString) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    let v4 ← DynString_payload_size value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← DynString_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← DynString_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← DynString_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def encode_single_root_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let v1 ← abi_single_root_size_DynString value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_DynString
  let v13 ← AbiFrame_tail out v11
  let v16 ← encode_field_Sol_DynString value out v10 v13
  let _ ← DynString_drop value
  let buffer_17 := buffer
  return buffer_17

def encode_return_alloc_Sol_DynString (value_0 : DynString) : FeM MemBuffer := do
  let v1 ← const_DynString
  if v1 then
    let value_2 := value_0
    let v3 ← encode_alloc_Sol_DynString value_2
    return v3
  else
    let v4 := ()
    let v5 := v4
    let value_6 := value_0
    let v7 ← encode_single_root_alloc_Sol_DynString value_6
    return v7

def Evm_return_value_Sol_DynString (value : DynString) : FeM Empty := do
  let mut value := value
  let mut out_r1 : FixedMemBuffer_32 := default
  let mut out_r2 : FixedMemBuffer_64 := default
  let mut out_r3 : MemBuffer := default
  let mut out_r4 : MemBuffer := default
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
      let out_7 := v6
      out_r1 := out_7
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← DynString_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let v14 ← DynString_encode_Sol value r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
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
        let out_27 := v26
        out_r2 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← DynString_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let v34 ← DynString_encode_Sol value r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← DynString_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← DynString_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let out_46 := v45
        out_r3 := out_46
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← DynString_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let v53 ← DynString_encode_Sol value r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let value_60 := value_59
    let v61 ← encode_return_alloc_Sol_DynString value_60
    let out_62 := v61
    out_r4 := out_62
    let v64 ← MemBuffer_span_ref out_r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

def u8_IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def u8_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def u8_encode_Sol (self_0 : UInt256) (out : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let self_3 := self_0
  let v4 := self_3
  let v5 ← AbiFrame_store_word out v2 v4
  return ()

def const_u8 : FeM Bool := do
  let v0 := false
  return v0

def payload_size_u8 (self : UInt256) : FeM UInt256 := do
  let v1 ← u8_HEAD_SIZE
  return v1

def abi_payload_size_u8 (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u8 value
  return v1

def encode_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_u8 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  let v10 ← u8_encode_Sol value r1
  let buffer_11 := buffer
  return buffer_11

def abi_field_size_u8 (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u8 value
  let payload_size := v1
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_u8 (value : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_u8 value
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

def encode_field_Sol_u8 (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_u8 value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← u8_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← u8_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← u8_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def encode_single_root_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let v1 ← abi_single_root_size_u8 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_u8
  let v13 ← AbiFrame_tail out v11
  let v15 ← encode_field_Sol_u8 value out v10 v13
  let buffer_16 := buffer
  return buffer_16

def encode_return_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let v1 ← const_u8
  if v1 then
    let v2 ← encode_alloc_Sol_u8 value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_u8 value
    return v5

def Evm_return_value_Sol_u8 (value : UInt256) : FeM Empty := do
  let mut out_r0 : FixedMemBuffer_32 := default
  let mut out_r1 : FixedMemBuffer_64 := default
  let mut out_r2 : MemBuffer := default
  let mut out_r3 : MemBuffer := default
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
      let out_8 := v7
      out_r0 := out_8
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← u8_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let v14 ← u8_encode_Sol value r4
      let v16 ← FixedMemBuffer_ptr_32 out_r0
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
        let out_27 := v26
        out_r1 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← u8_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let v33 ← u8_encode_Sol value r5
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← u8_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← u8_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← MemBuffer_new v43
        let out_45 := v44
        out_r2 := out_45
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← u8_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let v51 ← u8_encode_Sol value r6
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_u8 value
    let out_58 := v57
    out_r3 := out_58
    let v60 ← MemBuffer_span_ref out_r3
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

def payload_size_tuple (self : Unit) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE
  return v1

def abi_payload_size_tuple (value : Unit) : FeM UInt256 := do
  let v1 ← payload_size_tuple value
  return v1

def tuple_encode_Sol (self : Unit) (out : AbiFrame) : FeM Unit := do
  let v2 := ()
  return ()

def encode_alloc_Sol_tuple (value : Unit) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let value_0 := value
  let v1 ← abi_payload_size_tuple value_0
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let value_9 := value
  let v11 ← tuple_encode_Sol value_9 r2
  let buffer_12 := buffer
  return buffer_12

def encode_abi_payload_tuple (value : Unit) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_tuple value
  return v1

def create2_raw_Evm (value : Wei) (init_code : MemSlice_u8) (salt : UInt256) : FeM UInt256 := do
  let v4 ← Wei_as_wei value
  let v5 ← MemSlice_ptr_u8 init_code
  let v6 ← MemSlice_len_u8 init_code
  let v7 := v6
  let v8 ← create2 v4 v5 v7 salt
  let v9 := v8
  return v9

def copy_returndata : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let v0 ← returndatasize
  let ret_len := v0
  let v2 ← downcast_unchecked_u256_usize ret_len
  let v3 ← MemBuffer_new_uninit v2
  let buffer_4 := v3
  buffer := buffer_4
  let v6 ← MemBuffer_ptr buffer
  let v7 := (UInt256.ofNat 0)
  let v8 ← returndatacopy v6 v7 ret_len
  let buffer_9 := buffer
  return buffer_9

def bubble_last_revert : FeM Empty := do
  let mut ret : MemBuffer := default
  let v0 ← copy_returndata
  let ret_1 := v0
  ret := ret_1
  let v3 ← MemBuffer_ptr ret
  let v5 ← MemBuffer_len ret
  let v6 := v5
  let v7 ← revertData v3 v6
  nomatch v7

def copy_call_outcome («raw» : RawCallOutcome) : FeM CallOutcome := do
  let v1 ← RawCallOutcome_success «raw»
  let v2 ← copy_returndata
  let v3 := (CallOutcome.mk v1 v2)
  return v3

def u256_decode_from_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_u8_word_at input pos
  return v2

def decode_from_prechecked_head_u256_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_decode_from_Sol_MemSlice_u8 input pos
  return v3

def decode_frame_from_Sol_u256_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← decode_from_prechecked_head_u256_Sol_MemSlice_u8 input pos input_len
  return v5

def decode_from_bounded_u256_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← u256_decode_from_Sol_MemSlice_u8 input pos
  return v5

def decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 (input : MemSlice_u8) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← MemSlice_u8_word_at input head_pos
    let tail_rel := v5
    let v7 ← checked_tail_Sol base tail_rel
    let tail_at := v7
    let v9 ← decode_from_bounded_u256_Sol_MemSlice_u8 input tail_at input_len
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← u256_decode_from_Sol_MemSlice_u8 input head_pos
    return v12

def decode_output_at_u256_MemSlice_u8 (input : MemSlice_u8) (base : UInt256) : FeM UInt256 := do
  let v2 ← len_MemSlice_u8 input
  let input_len := v2
  let v4 ← const_u256
  if v4 then
    let v5 ← decode_frame_from_Sol_u256_MemSlice_u8 input base input_len
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← abi_single_root_head_size_u256
    let v9 ← checked_frame_end_Sol base v8 input_len
    let v10 ← decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 input base base input_len
    return v10

def decode_output_u256_MemSlice_u8 (input : MemSlice_u8) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  let v2 ← decode_output_at_u256_MemSlice_u8 input v1
  return v2

def decode_returndata_u256 (ret : MemSlice_u8) : FeM UInt256 := do
  let v1 ← decode_output_u256_MemSlice_u8 ret
  return v1

def CallOutcome_decode_u256 (self : CallOutcome) : FeM UInt256 := do
  let mut returndata : MemBuffer := default
  let v1 := self.success
  let success := v1
  let v3 := self.returndata
  let returndata_4 := v3
  returndata := returndata_4
  let v5 := !success
  if v5 then
    let v7 ← MemBuffer_ptr returndata
    let v9 ← MemBuffer_len returndata
    let v10 := v9
    let v11 ← revertData v7 v10
    nomatch v11
  else
    let v12 := ()
    let v13 := v12
    let v15 ← MemBuffer_span_ref returndata
    let v17 ← decode_returndata_u256 v15
    let _ ← MemBuffer_drop returndata
    return v17

def keccak_words_2 (words : (Vector UInt256 2)) : FeM UInt256 := do
  let v1 := keccakWords words
  return v1

def read_transient_word_Evm (slot_0 : UInt256) : FeM UInt256 := do
  let slot_1 := slot_0
  let v2 := slot_1
  let v3 ← load (Place.mk .transient v2)
  return v3

def storage_at_mut_Evm_u256 (slot : UInt256) : FeM Place := do
  let v1 := slot
  let v2 := (Place.mk .storage v1)
  return v2

def Address_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := self
  let v4 ← AbiFrame_store_word out v2 v3
  return ()

def const_Address : FeM Bool := do
  let v0 := false
  return v0

def payload_size_Address (self : UInt256) : FeM UInt256 := do
  let v1 ← Address_HEAD_SIZE
  return v1

def abi_payload_size_Address (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_Address value
  return v1

def encode_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let value_0 := value
  let v1 ← abi_payload_size_Address value_0
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let value_9 := value
  let v11 ← Address_encode_Sol value_9 r2
  let buffer_12 := buffer
  return buffer_12

def abi_field_size_Address (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_Address value
  let payload_size := v1
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_Address (value : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_Address value
  return v1

def abi_single_root_head_size_Address : FeM UInt256 := do
  let v0 ← abi_field_head_size_Address
  return v0

def encode_field_Sol_Address (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_Address value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← Address_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← Address_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← Address_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def encode_single_root_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let value_0 := value
  let v1 ← abi_single_root_size_Address value_0
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_Address
  let v13 ← AbiFrame_tail out v11
  let value_14 := value
  let v16 ← encode_field_Sol_Address value_14 out v10 v13
  let buffer_17 := buffer
  return buffer_17

def encode_return_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let v1 ← const_Address
  if v1 then
    let v2 ← encode_alloc_Sol_Address value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_Address value
    return v5

def Evm_return_value_Sol_Address (value : UInt256) : FeM Empty := do
  let mut value := value
  let mut out_r1 : FixedMemBuffer_32 := default
  let mut out_r2 : FixedMemBuffer_64 := default
  let mut out_r3 : MemBuffer := default
  let mut out_r4 : MemBuffer := default
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
      let out_7 := v6
      out_r1 := out_7
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← Address_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let value_12 := value
      let v14 ← Address_encode_Sol value_12 r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
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
        let out_27 := v26
        out_r2 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← Address_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let value_32 := value
        let v34 ← Address_encode_Sol value_32 r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← Address_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← Address_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let out_46 := v45
        out_r3 := out_46
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← Address_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let value_51 := value
        let v53 ← Address_encode_Sol value_51 r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let v60 ← encode_return_alloc_Sol_Address value_59
    let out_61 := v60
    out_r4 := out_61
    let v63 ← MemBuffer_span_ref out_r4
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

def «assert» (condition_0 : Bool) : FeM Unit := do
  let condition_1 := condition_0
  let v2 := !condition_1
  if v2 then
    let v3 ← trap
    nomatch v3
  else
    let v4 := ()
    let v5 := v4
    return ()

def u128_to_word (self_0 : UInt256) : FeM UInt256 := do
  let self_1 := self_0
  let v2 := self_1
  return v2

def downcast_truncate_u128_u64 (self : UInt256) : FeM UInt256 := do
  let v2 ← u128_to_word self
  let word := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def u64_wrapping_mul (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := IntTy.mulW (.u 64) self other
  return v2

def u128_wrapping_sub (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := IntTy.subW (.u 128) self other
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

def i64_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 64) word
  return v1

def downcast_truncate_u64_i64 (self : UInt256) : FeM UInt256 := do
  let v2 ← u64_to_word self
  let word := v2
  let v4 ← i64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← i64_MASK
    let v11 := word &&& v10
    let v12 ← i64_from_word v11
    let v13 := v12
    b3 v13

def i64_to_word (self_0 : UInt256) : FeM UInt256 := do
  let self_1 := self_0
  let v2 := self_1
  let v3 := v2
  return v3

def downcast_truncate_i64_u64 (self : UInt256) : FeM UInt256 := do
  let v2 ← i64_to_word self
  let word := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def i128_to_word (self_0 : UInt256) : FeM UInt256 := do
  let self_1 := self_0
  let v2 := self_1
  let v3 := v2
  return v3

def downcast_truncate_i128_u64 (self : UInt256) : FeM UInt256 := do
  let v2 ← i128_to_word self
  let word := v2
  let v4 ← u64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    let v13 := v12
    b3 v13

def downcast_truncate_i128_i64 (self : UInt256) : FeM UInt256 := do
  let v2 ← i128_to_word self
  let word := v2
  let v4 ← i64_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i64_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← i64_MASK
    let v11 := word &&& v10
    let v12 ← i64_from_word v11
    let v13 := v12
    b3 v13

def array_start_usize_3 (self : (Vector UInt256 3)) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def array_has_usize_3 (self : (Vector UInt256 3)) (c_1 : UInt256) : FeM Bool := do
  let c_2 := c_1
  let v3 := (UInt256.ofNat 3)
  let v4 := decide (c_2.toNat < v3.toNat)
  return v4

def array_at_ref_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM UInt256 := do
  checkIndex c 3
  return self[c.toNat]!

def Range_start_1_4 (self : Range_Known_1_Known_4) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  return v1

def range_has (c_0 : UInt256) (lo_1 : UInt256) (hi_2 : UInt256) : FeM Bool := do
  let lo_4 := lo_1
  let c_5 := c_0
  let v6 := decide (lo_4.toNat ≤ c_5.toNat)
  let b3 : Bool → FeM Bool := fun v3 => do
    return v3
  let b2 : FeM Bool := do
    let v12 := false
    let v13 := v12
    b3 v13
  if v6 then
    let c_7 := c_0
    let hi_8 := hi_2
    let v9 := decide (c_7.toNat < hi_8.toNat)
    if v9 then
      let v10 := true
      let v11 := v10
      b3 v11
    else
      b2
  else
    b2

def Range_has_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 1)
  let v3 := (UInt256.ofNat 4)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_1_4 (self : Range_Known_1_Known_4) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def array_next_usize_3 (self : (Vector UInt256 3)) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def Range_start_1_16 (self : Range_Known_1_Known_16) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  return v1

def Range_has_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 1)
  let v3 := (UInt256.ofNat 16)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_1_16 (self : Range_Known_1_Known_16) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def downcast_truncate_u256_usize (self : UInt256) : FeM UInt256 := do
  let v2 ← u256_to_word self
  let word := v2
  let v4 ← usize_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← usize_MASK
    let v11 := word &&& v10
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

def tuple_encode_Sol_bool_u256_u256 (self : (Bool × UInt256 × UInt256)) (out : AbiFrame) : FeM Unit := do
  let mut tails : AbiFrame := default
  let v5 ← tuple_HEAD_SIZE_bool_u256_u256
  let v6 ← AbiFrame_tail out v5
  let tails_7 := v6
  tails := tails_7
  let v8 ← abi_field_head_size_bool
  let head1 := v8
  let v10 ← abi_field_head_size_u256
  let v11 := head1 + v10
  let head2 := v11
  let v13 := (UInt256.ofNat 0)
  let tails_14 := tails
  let tails_15 := tails_14
  let v0_16 := self.fst
  let v17 ← encode_field_Sol_bool v0_16 out v13 tails_15
  let tails_18 := v17
  tails := tails_18
  let v19 := ()
  let tails_20 := tails
  let tails_21 := tails_20
  let v1_22 := self.snd.fst
  let v23 ← encode_field_Sol_u256 v1_22 out head1 tails_21
  let tails_24 := v23
  tails := tails_24
  let v25 := ()
  let tails_26 := tails
  let tails_27 := tails_26
  let v2_28 := self.snd.snd
  let v29 ← encode_field_Sol_u256 v2_28 out head2 tails_27
  return ()

def tuple_IS_PARAM_LIST_bool_u256_u256 : FeM Bool := do
  let v0 := true
  return v0

def dynamic_payload_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← bool_IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_bool value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 := (UInt256.ofNat 0)
    return v5

def tuple_payload_size_bool_u256_u256 (self : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE_bool_u256_u256
  let v2 := self.fst
  let v3 ← dynamic_payload_size_bool v2
  let v4 := v1 + v3
  let v5 := self.snd.fst
  let v6 ← dynamic_payload_size_u256 v5
  let v7 := v4 + v6
  let v8 := self.snd.snd
  let v9 ← dynamic_payload_size_u256 v8
  let v10 := v7 + v9
  return v10

def abi_payload_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_bool_u256_u256 value
  return v1

def encode_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_tuple_bool_u256_u256 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let v11 ← tuple_encode_Sol_bool_u256_u256 value r2
  let buffer_12 := buffer
  return buffer_12

def abi_field_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_bool_u256_u256 value
  let payload_size := v1
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← abi_field_size_tuple_bool_u256_u256 value
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

def encode_field_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    let v4 ← tuple_payload_size_bool_u256_u256 value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← tuple_encode_Sol_bool_u256_u256 value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← tuple_HEAD_SIZE_bool_u256_u256
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← tuple_encode_Sol_bool_u256_u256 value r2
    let tails_24 := tails
    return tails_24

def encode_single_root_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let v1 ← abi_single_root_size_tuple_bool_u256_u256 value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_tuple_bool_u256_u256
  let v13 ← AbiFrame_tail out v11
  let v16 ← encode_field_Sol_tuple_bool_u256_u256 value out v10 v13
  let buffer_17 := buffer
  return buffer_17

def encode_return_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let v1 ← tuple_IS_PARAM_LIST_bool_u256_u256
  if v1 then
    let v2 ← encode_alloc_Sol_tuple_bool_u256_u256 value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_tuple_bool_u256_u256 value
    return v5

def Evm_return_value_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM Empty := do
  let mut value := value
  let mut out_r1 : FixedMemBuffer_32 := default
  let mut out_r2 : FixedMemBuffer_64 := default
  let mut out_r3 : MemBuffer := default
  let mut out_r4 : MemBuffer := default
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
      let out_7 := v6
      out_r1 := out_7
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← tuple_HEAD_SIZE_bool_u256_u256
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let v14 ← tuple_encode_Sol_bool_u256_u256 value r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
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
        let out_27 := v26
        out_r2 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← tuple_HEAD_SIZE_bool_u256_u256
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let v34 ← tuple_encode_Sol_bool_u256_u256 value r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← tuple_HEAD_SIZE_bool_u256_u256
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← tuple_HEAD_SIZE_bool_u256_u256
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let out_46 := v45
        out_r3 := out_46
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← tuple_HEAD_SIZE_bool_u256_u256
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let v53 ← tuple_encode_Sol_bool_u256_u256 value r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let v60 ← encode_return_alloc_Sol_tuple_bool_u256_u256 value_59
    let out_61 := v60
    out_r4 := out_61
    let v63 ← MemBuffer_span_ref out_r4
    let v66 ← Evm_return_bytes v63
    nomatch v66

inductive Option_array_u256_2 where
  | None
  | Some (f0 : (Vector UInt256 2))
deriving Inhabited

def Option_array_u256_2.tag : Option_array_u256_2 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_array_u256_2.Some_0 : Option_array_u256_2 → (Vector UInt256 2)
  | .Some f0 => f0
  | _ => default

def Option_array_u256_2.set_Some_0 (x : Option_array_u256_2) (v : (Vector UInt256 2)) : Option_array_u256_2 :=
  match x with
  | .Some f0 => .Some v
  | x => x

structure Range_Unknown_Unknown where
  start : UInt256
  «end» : UInt256
deriving Inhabited

def Evm_calldatasize : FeM UInt256 := do
  let v1 ← calldatasize
  return v1

def prepare_copy_destination (dest_0 : MemBuffer) (dest_offset_1 : UInt256) (len_2 : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut dest_0 := dest_0
  let dest_offset_3 := dest_offset_1
  let v5 ← MemBuffer_capacity dest_0
  let v6 := v5
  let v7 := decide (v6.toNat < dest_offset_3.toNat)
  let b1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
    let mut dest_0 := dest_0
    let v44 ← trap
    nomatch v44
  if v7 then
    b1 dest_0
  else
    let len_8 := len_2
    let v10 ← MemBuffer_capacity dest_0
    let v11 := v10
    let dest_offset_12 := dest_offset_1
    let v13 ← checkedSub v11 dest_offset_12
    let v14 := decide (v13.toNat < len_8.toNat)
    if v14 then
      b1 dest_0
    else
      let v15 := ()
      let v16 := v15
      let len_17 := len_2
      let v18 := (UInt256.ofNat 0)
      let v19 := decide (len_17 ≠ v18)
      let b7 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
        let mut dest_0 := dest_0
        let v41 ← MemBuffer_ptr dest_0
        let v42 ← downcast_unchecked_u256_usize dest_offset_1
        let v43 := v41 + v42
        return (v43, dest_0)
      let b6 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
        let mut dest_0 := dest_0
        let v38 := ()
        let v39 := v38
        b7 dest_0
      if v19 then
        let dest_offset_20 := dest_offset_1
        let v22 ← MemBuffer_len dest_0
        let v23 := v22
        let v24 := decide (v23.toNat < dest_offset_20.toNat)
        if v24 then
          let v26 ← MemBuffer_ptr dest_0
          let v28 ← MemBuffer_len dest_0
          let v29 := v26 + v28
          let dest_offset_30 := dest_offset_1
          let v32 ← MemBuffer_len dest_0
          let v33 := v32
          let v34 ← checkedSub dest_offset_30 v33
          let v35 ← downcast_unchecked_u256_usize v34
          let v36 ← zero_bytes v29 v35
          let v37 := v36
          b7 dest_0
        else
          b6 dest_0
      else
        b6 dest_0

def Evm_calldatacopy (dest_1 : MemBuffer) (dest_offset_2 : UInt256) (src_offset : UInt256) (len_4 : UInt256) : FeM (Unit × MemBuffer) := do
  let mut dest_1 := dest_1
  let (v6, v6_0) ← prepare_copy_destination dest_1 dest_offset_2 len_4
  dest_1 := v6_0
  let dest_ptr := v6
  let len_8 := len_4
  let v9 := (UInt256.ofNat 0)
  let v10 := decide (len_8 = v9)
  if v10 then
    return ((), dest_1)
  else
    let v11 := ()
    let v12 := v11
    let v13 ← calldatacopy dest_ptr src_offset len_4
    let dest_offset_14 := dest_offset_2
    let len_15 := len_4
    let v16 ← checkedAdd dest_offset_14 len_15
    let v17 ← downcast_unchecked_u256_usize v16
    let «end» := v17
    let v20 ← MemBuffer_len dest_1
    let v21 := decide (v20.toNat < «end».toNat)
    let b6 : MemBuffer → FeM (Unit × MemBuffer) := fun dest_1 => do
      let mut dest_1 := dest_1
      return ((), dest_1)
    if v21 then
      let (v22, v22_0) ← MemBuffer_set_len dest_1 «end»
      dest_1 := v22_0
      let v23 := v22
      b6 dest_1
    else
      let v24 := ()
      let v25 := v24
      b6 dest_1

def MemSlice_u8_value_ptr_u64 (self : MemSlice_u8) (offset_1 : UInt256) : FeM UInt256 := do
  let offset_2 := offset_1
  let v3 := self.len
  let v4 := decide (v3.toNat < offset_2.toNat)
  let b1 : FeM UInt256 := do
    let v14 ← index_out_of_bounds
    nomatch v14
  if v4 then
    b1
  else
    let v5 := (UInt256.ofNat 8)
    let v6 := self.len
    let offset_7 := offset_1
    let v8 ← checkedSub v6 offset_7
    let v9 := decide (v8.toNat < v5.toNat)
    if v9 then
      b1
    else
      let v10 := ()
      let v11 := v10
      let v12 := self.ptr
      let v13 := v12 + offset_1
      return v13

def cast_u8_u64 (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let v2 := ptr_1
  return v2

def MemSlice_u8_read_le_u64 (self : MemSlice_u8) (offset : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_u8_value_ptr_u64 self offset
  let v3 ← cast_u8_u64 v2
  let v4 ← mloadInt (.u 64) v3
  let v5 := IntTy.swapBytes (.u 64) v4
  return v5

def MemBuffer_read_le_u64 (self : MemBuffer) (offset : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_span_ref self
  let v4 ← MemSlice_u8_read_le_u64 v2 offset
  return v4

def Range_Unknown_Unknown_start (self : Range_Unknown_Unknown) : FeM UInt256 := do
  let v1 := self.start
  return v1

def Range_Unknown_Unknown_has (self : Range_Unknown_Unknown) (c : UInt256) : FeM Bool := do
  let v2 := self.start
  let v3 := self.«end»
  let v4 ← range_has c v2 v3
  return v4

def Range_Unknown_Unknown_at (self : Range_Unknown_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_Unknown_Unknown_next (self : Range_Unknown_Unknown) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def MemBuffer_index_ref (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  let v2 := self.len
  let v3 := decide (v2.toNat ≤ index.toNat)
  if v3 then
    let v4 ← index_out_of_bounds
    nomatch v4
  else
    let v5 := ()
    let v6 := v5
    let v7 := self.ptr
    let v8 := v7 + index
    let v9 := v8
    return v9

def MemBuffer_extend_over_u256 (self : MemBuffer) (offset_1 : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut self := self
  let v2 := (UInt256.ofNat 32)
  let size := v2
  let offset_4 := offset_1
  let v5 := self.capacity
  let v6 := decide (v5.toNat < offset_4.toNat)
  let b1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
    let mut self := self
    let v38 ← index_out_of_bounds
    nomatch v38
  if v6 then
    b1 self
  else
    let v7 := self.capacity
    let offset_8 := offset_1
    let v9 := v7 - offset_8
    let v10 := decide (v9.toNat < size.toNat)
    if v10 then
      b1 self
    else
      let v11 := ()
      let v12 := v11
      let offset_13 := offset_1
      let v14 := self.len
      let v15 := decide (v14.toNat < offset_13.toNat)
      let b7 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
        let mut self := self
        let offset_26 := offset_1
        let v27 := offset_26 + size
        let v28 := self.len
        let v29 := decide (v28.toNat < v27.toNat)
        let b10 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
          let mut self := self
          let v36 := self.ptr
          let v37 := v36 + offset_1
          return (v37, self)
        if v29 then
          let offset_30 := offset_1
          let v31 := offset_30 + size
          self := { self with len := v31 }
          let v32 := ()
          let v33 := v32
          b10 self
        else
          let v34 := ()
          let v35 := v34
          b10 self
      if v15 then
        let v16 := self.ptr
        let v17 := self.len
        let v18 := v16 + v17
        let offset_19 := offset_1
        let v20 := self.len
        let v21 := offset_19 - v20
        let v22 ← zero_bytes v18 v21
        let v23 := v22
        b7 self
      else
        let v24 := ()
        let v25 := v24
        b7 self

def MemBuffer_write_be_u256 (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let v3 := value
  let (v4, v4_0) ← MemBuffer_extend_over_u256 self offset
  self := v4_0
  let v5 ← cast_u8_u256 v4
  mstore v5 v3
  let v6 := ()
  return ((), self)

def MemBuffer_write_word (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let (v3, v3_0) ← MemBuffer_write_be_u256 self offset value
  self := v3_0
  return ((), self)

def Evm_revert (data : MemSlice_u8) : FeM Empty := do
  let v2 ← MemSlice_ptr_u8 data
  let v3 ← MemSlice_len_u8 data
  let v4 := v3
  let v5 ← revertData v2 v4
  nomatch v5

structure Range_Known_0_Known_8 where
  start : Unit
  «end» : Unit
deriving Inhabited

structure Range_Known_0_Known_16 where
  start : Unit
  «end» : Unit
deriving Inhabited

def Range_start_0_8 (self : Range_Known_0_Known_8) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def Range_has_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 8)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_0_8 (self : Range_Known_0_Known_8) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def Range_start_0_16 (self : Range_Known_0_Known_16) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def Range_has_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 16)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_0_16 (self : Range_Known_0_Known_16) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def u64_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := IntTy.addW (.u 64) self other
  return v2

def downcast_truncate_u64_usize (self : UInt256) : FeM UInt256 := do
  let v2 ← u64_to_word self
  let word := v2
  let v4 ← usize_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← usize_MASK
    let v11 := word &&& v10
    let v12 ← usize_from_word v11
    let v13 := v12
    b3 v13

def MemBuffer_extend_over_u64 (self : MemBuffer) (offset_1 : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut self := self
  let v2 := (UInt256.ofNat 8)
  let size := v2
  let offset_4 := offset_1
  let v5 := self.capacity
  let v6 := decide (v5.toNat < offset_4.toNat)
  let b1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
    let mut self := self
    let v38 ← index_out_of_bounds
    nomatch v38
  if v6 then
    b1 self
  else
    let v7 := self.capacity
    let offset_8 := offset_1
    let v9 := v7 - offset_8
    let v10 := decide (v9.toNat < size.toNat)
    if v10 then
      b1 self
    else
      let v11 := ()
      let v12 := v11
      let offset_13 := offset_1
      let v14 := self.len
      let v15 := decide (v14.toNat < offset_13.toNat)
      let b7 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
        let mut self := self
        let offset_26 := offset_1
        let v27 := offset_26 + size
        let v28 := self.len
        let v29 := decide (v28.toNat < v27.toNat)
        let b10 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
          let mut self := self
          let v36 := self.ptr
          let v37 := v36 + offset_1
          return (v37, self)
        if v29 then
          let offset_30 := offset_1
          let v31 := offset_30 + size
          self := { self with len := v31 }
          let v32 := ()
          let v33 := v32
          b10 self
        else
          let v34 := ()
          let v35 := v34
          b10 self
      if v15 then
        let v16 := self.ptr
        let v17 := self.len
        let v18 := v16 + v17
        let offset_19 := offset_1
        let v20 := self.len
        let v21 := offset_19 - v20
        let v22 ← zero_bytes v18 v21
        let v23 := v22
        b7 self
      else
        let v24 := ()
        let v25 := v24
        b7 self

def MemBuffer_write_le_u64 (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let v3 := IntTy.swapBytes (.u 64) value
  let (v4, v4_0) ← MemBuffer_extend_over_u64 self offset
  self := v4_0
  let v5 ← cast_u8_u64 v4
  mstoreInt (.u 64) v5 v3
  let v6 := ()
  return ((), self)

structure Range_Known_0_Unknown where
  start : Unit
  «end» : UInt256
deriving Inhabited

structure Range_Known_0_Known_64 where
  start : Unit
  «end» : Unit
deriving Inhabited

def Range_start_0 (self : Range_Known_0_Unknown) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def Range_has_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 0)
  let v3 := self.«end»
  let v4 ← range_has c v2 v3
  return v4

def Range_at_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_start_0_64 (self : Range_Known_0_Known_64) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def Range_has_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 64)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_0_64 (self : Range_Known_0_Known_64) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def downcast_unchecked_usize_u64 (self : UInt256) : FeM UInt256 := do
  let v1 ← usize_to_word self
  let word := v1
  let v3 ← u64_from_word word
  return v3

def downcast_truncate_u64_u8 (self : UInt256) : FeM UInt256 := do
  let v2 ← u64_to_word self
  let word := v2
  let v4 ← u8_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u8_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← u8_MASK
    let v11 := word &&& v10
    let v12 ← u8_from_word v11
    let v13 := v12
    b3 v13

def Range_next_0 (self : Range_Known_0_Unknown) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def array_start_u32_5 (self : (Vector UInt256 5)) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def array_has_u32_5 (self : (Vector UInt256 5)) (c_1 : UInt256) : FeM Bool := do
  let c_2 := c_1
  let v3 := (UInt256.ofNat 5)
  let v4 := decide (c_2.toNat < v3.toNat)
  return v4

def array_at_ref_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM UInt256 := do
  checkIndex c 5
  return self[c.toNat]!

def array_next_u32_5 (self : (Vector UInt256 5)) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def u256_wrapping_sub (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self - other
  return v2

inductive Option_MemBuffer where
  | None
  | Some (f0 : MemBuffer)
deriving Inhabited

def Option_MemBuffer.tag : Option_MemBuffer → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_MemBuffer.Some_0 : Option_MemBuffer → MemBuffer
  | .Some f0 => f0
  | _ => default

def Option_MemBuffer.set_Some_0 (x : Option_MemBuffer) (v : MemBuffer) : Option_MemBuffer :=
  match x with
  | .Some f0 => .Some v
  | x => x

inductive Option_u8 where
  | None
  | Some (f0 : UInt256)
deriving Inhabited

def Option_u8.tag : Option_u8 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_u8.Some_0 : Option_u8 → UInt256
  | .Some f0 => f0
  | _ => default

def Option_u8.set_Some_0 (x : Option_u8) (v : UInt256) : Option_u8 :=
  match x with
  | .Some f0 => .Some v
  | x => x

def try_alloc_bytes (len_0 : UInt256) : FeM Option_u8 := do
  let len_2 := len_0
  let v3 ← MAX_ALLOCATION
  let v4 := decide (v3.toNat < len_2.toNat)
  if v4 then
    let v5 := Option_u8.None
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← alloc len_0
    let ptr := v8
    let v10 := ptr
    let v11 := (UInt256.ofNat 0)
    let v12 := decide (v10 = v11)
    let b6 : Option_u8 → FeM Option_u8 := fun v1 => do
      return v1
    if v12 then
      let v13 := Option_u8.None
      let v14 := v13
      b6 v14
    else
      let v15 := Option_u8.Some ptr
      let v16 := v15
      b6 v16

def MemBuffer_try_new (len_0 : UInt256) : FeM Option_MemBuffer := do
  let v1 ← try_alloc_bytes len_0
  let v2 := match v1 with | .Some .. => true | _ => false
  if v2 then
    let v3 := v1.Some_0
    let ptr := v3
    let v5 ← zero_bytes ptr len_0
    let len_6 := len_0
    let len_7 := len_0
    let v8 := (MemBuffer.mk ptr len_6 len_7)
    let v9 := Option_MemBuffer.Some v8
    return v9
  else
    let v10 := Option_MemBuffer.None
    return v10

def MemBuffer_index_mut (self : MemBuffer) (index : UInt256) : FeM (UInt256 × (UInt256 → FeM MemBuffer)) := do
  let mut self := self
  let v2 := self.len
  let v3 := decide (v2.toNat ≤ index.toNat)
  if v3 then
    let v4 ← index_out_of_bounds
    nomatch v4
  else
    let v5 := ()
    let v6 := v5
    let v7 := self.ptr
    let v8 := v7 + index
    let v9 := v8
    let slide : UInt256 → FeM MemBuffer := fun g => do
      let mut self := self
      return self
    return (v9, slide)

structure MemArray_u64 where
  ptr : UInt256
  len : UInt256
deriving Inhabited

inductive Option_MemArray_u64 where
  | None
  | Some (f0 : MemArray_u64)
deriving Inhabited

def Option_MemArray_u64.tag : Option_MemArray_u64 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_MemArray_u64.Some_0 : Option_MemArray_u64 → MemArray_u64
  | .Some f0 => f0
  | _ => default

def Option_MemArray_u64.set_Some_0 (x : Option_MemArray_u64) (v : MemArray_u64) : Option_MemArray_u64 :=
  match x with
  | .Some f0 => .Some v
  | x => x

inductive Option_u64 where
  | None
  | Some (f0 : UInt256)
deriving Inhabited

def Option_u64.tag : Option_u64 → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_u64.Some_0 : Option_u64 → UInt256
  | .Some f0 => f0
  | _ => default

def Option_u64.set_Some_0 (x : Option_u64) (v : UInt256) : Option_u64 :=
  match x with
  | .Some f0 => .Some v
  | x => x

structure Range_Known_0_Known_5 where
  start : Unit
  «end» : Unit
deriving Inhabited

structure Range_Known_1_Unknown where
  start : Unit
  «end» : UInt256
deriving Inhabited

structure Range_Known_0_Known_6 where
  start : Unit
  «end» : Unit
deriving Inhabited

structure MemSlice_u64 where
  ptr : UInt256
  len : UInt256
deriving Inhabited

def try_alloc_elems_u64 (len_0 : UInt256) : FeM Option_u64 := do
  let v1 := (UInt256.ofNat 8)
  let size := v1
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (size ≠ v3)
  let b2 : FeM Option_u64 := do
    let v10 := ()
    let v11 := v10
    let len_12 := len_0
    let v13 ← checkedMul len_12 size
    let v14 ← try_alloc_bytes v13
    let v15 := match v14 with | .Some .. => true | _ => false
    if v15 then
      let v16 := v14.Some_0
      let v17 ← cast_u8_u64 v16
      let v18 := Option_u64.Some v17
      return v18
    else
      let v19 := Option_u64.None
      return v19
  if v4 then
    let len_5 := len_0
    let v6 ← MAX_ALLOCATION
    let v7 ← checkedDiv v6 size
    let v8 := decide (v7.toNat < len_5.toNat)
    if v8 then
      let v9 := Option_u64.None
      return v9
    else
      b2
  else
    b2

def MemArray_try_new_uninit_u64 (len_0 : UInt256) : FeM Option_MemArray_u64 := do
  let v1 ← try_alloc_elems_u64 len_0
  let v2 := match v1 with | .Some .. => true | _ => false
  if v2 then
    let v3 := v1.Some_0
    let len_4 := len_0
    let v5 := (MemArray_u64.mk v3 len_4)
    let v6 := Option_MemArray_u64.Some v5
    return v6
  else
    let v7 := Option_MemArray_u64.None
    return v7

def cast_u64_u8 (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let v2 := ptr_1
  return v2

def MemArray_byte_len_u64 (self : MemArray_u64) : FeM UInt256 := do
  let v1 := self.len
  let v2 := (UInt256.ofNat 8)
  let v3 ← checkedMul v1 v2
  return v3

def usize_wrapping_mul (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self * other
  return v2

def elem_u64 (ptr : UInt256) (index_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 8)
  let index_3 := index_1
  let v4 ← usize_wrapping_mul index_3 v2
  let v5 := ptr + v4
  return v5

def drop_u64 (value : UInt256) : FeM Unit := do
  let v1 := ()
  return ()

def drop_in_place_u64 (ptr_0 : UInt256) : FeM Unit := do
  let ptr_1 := ptr_0
  let v2 ← mloadInt (.u 64) ptr_1
  let v3 ← drop_u64 v2
  return ()

def drop_range_u64 (ptr : UInt256) (start_1 : UInt256) (end_2 : UInt256) : FeM Unit := do
  let mut r0 : Range_Unknown_Unknown := default
  let v4 := false
  let b3 : Range_Unknown_Unknown → FeM Unit := fun r0 => do
    let mut r0 := r0
    return ()
  if v4 then
    let start_5 := start_1
    let end_6 := end_2
    let v7 := ()
    let v8 := (Range_Unknown_Unknown.mk start_5 end_6)
    r0 := v8
    let v10 ← Range_Unknown_Unknown_start r0
    let v11 := v10
    let rec b4 (r0 : Range_Unknown_Unknown) (v3 : UInt256) : FeM Unit := do
      let mut r0 := r0
      let v13 ← Range_Unknown_Unknown_has r0 v3
      if v13 then
        let v15 ← Range_Unknown_Unknown_at r0 v3
        let i := v15
        let v17 ← elem_u64 ptr i
        let v18 ← drop_in_place_u64 v17
        let v20 ← Range_Unknown_Unknown_next r0 v3
        let v21 := v20
        b4 r0 v21
      else
        let v22 := ()
        let v23 := v22
        b3 r0
    partial_fixpoint
    b4 r0 v11
  else
    let v24 := ()
    let v25 := v24
    b3 r0

def drop_in_place_n_u64 (ptr : UInt256) (len : UInt256) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← drop_range_u64 ptr v2 len
  return ()

def dealloc_u64 (ptr : UInt256) : FeM Unit := do
  let v1 ← cast_u64_u8 ptr
  let v2 := ()
  return ()

def MemArray_drop_u64 (self : MemArray_u64) : FeM Unit := do
  let v1 := self.ptr
  let v2 := self.len
  let v3 ← drop_in_place_n_u64 v1 v2
  let v4 := self.ptr
  let v5 ← dealloc_u64 v4
  return ()

def MemArray_try_new_zeroed_u64 (len : UInt256) : FeM Option_MemArray_u64 := do
  let mut array : MemArray_u64 := default
  let v1 ← MemArray_try_new_uninit_u64 len
  let v2 := match v1 with | .Some .. => true | _ => false
  if v2 then
    let v3 := v1.Some_0
    let array_4 := v3
    array := array_4
    let v5 := array.ptr
    let v6 ← cast_u64_u8 v5
    let v8 ← MemArray_byte_len_u64 array
    let v9 ← zero_bytes v6 v8
    let array_10 := array
    let array_11 := array_10
    let v12 := Option_MemArray_u64.Some array_11
    return v12
  else
    let v13 := Option_MemArray_u64.None
    match v1 with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v1.Some_0
      pure ()
    | _ => pure ()
    return v13

def mem_array_elem_u64 (array : MemArray_u64) (index_1 : UInt256) : FeM UInt256 := do
  let index_2 := index_1
  let v3 := array.len
  let v4 := decide (v3.toNat ≤ index_2.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := array.ptr
    let v9 ← elem_u64 v8 index_1
    return v9

def MemArray_index_mut_u64 (self : MemArray_u64) (index : UInt256) : FeM (UInt256 × (UInt256 → FeM MemArray_u64)) := do
  let mut self := self
  let v4 ← mem_array_elem_u64 self index
  let v5 := v4
  let slide : UInt256 → FeM MemArray_u64 := fun g => do
    let mut self := self
    return self
  return (v5, slide)

def MemArray_len_u64 (self : MemArray_u64) : FeM UInt256 := do
  let v1 := self.len
  return v1

def MemArray_index_ref_u64 (self : MemArray_u64) (index : UInt256) : FeM UInt256 := do
  let v2 ← mem_array_elem_u64 self index
  let v3 := v2
  return v3

def Range_start_0_5 (self : Range_Known_0_Known_5) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def Range_has_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 5)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_0_5 (self : Range_Known_0_Known_5) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def downcast_truncate_usize_usize (self : UInt256) : FeM UInt256 := do
  let v2 ← usize_to_word self
  let word := v2
  let v4 ← usize_IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← usize_MASK
    let v11 := word &&& v10
    let v12 ← usize_from_word v11
    let v13 := v12
    b3 v13

def Range_start_0_6 (self : Range_Known_0_Known_6) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

def Range_has_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 6)
  let v4 ← range_has c v2 v3
  return v4

def Range_at_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_0_6 (self : Range_Known_0_Known_6) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def MemArray_ptr_u64 (self : MemArray_u64) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

def Range_start_1 (self : Range_Known_1_Unknown) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 1)
  return v1

def Range_has_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM Bool := do
  let v2 := (UInt256.ofNat 1)
  let v3 := self.«end»
  let v4 ← range_has c v2 v3
  return v4

def Range_at_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

def Range_next_1 (self : Range_Known_1_Unknown) (c_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let c_3 := c_1
  let v4 ← usize_wrapping_add c_3 v2
  return v4

def MemSlice_from_raw_parts_ref_u64 (ptr : UInt256) (len : UInt256) : FeM MemSlice_u64 := do
  let mut slice : MemSlice_u64 := default
  let v2 := (MemSlice_u64.mk ptr len)
  let slice_3 := v2
  slice := slice_3
  return slice

def MemArray_as_slice_ref_u64 (self : MemArray_u64) : FeM MemSlice_u64 := do
  let v1 := self.ptr
  let v2 := self.len
  let v3 ← MemSlice_from_raw_parts_ref_u64 v1 v2
  return v3

def MemSlice_slice_ref_u64 (self : MemSlice_u64) (offset : UInt256) (len : UInt256) : FeM MemSlice_u64 := do
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let b1 : FeM MemSlice_u64 := do
    let v13 ← index_out_of_bounds
    nomatch v13
  if v4 then
    b1
  else
    let v5 := self.len
    let v6 ← checkedSub v5 offset
    let v7 := decide (v6.toNat < len.toNat)
    if v7 then
      b1
    else
      let v8 := ()
      let v9 := v8
      let v10 := self.ptr
      let v11 ← elem_u64 v10 offset
      let v12 ← MemSlice_from_raw_parts_ref_u64 v11 len
      return v12

def MemSlice_head_ref_u64 (self : MemSlice_u64) (n : UInt256) : FeM MemSlice_u64 := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← MemSlice_slice_ref_u64 self v2 n
  return v3

def MemSlice_from_raw_parts_mut_u64 (ptr : UInt256) (len : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM Unit)) := do
  let mut slice : MemSlice_u64 := default
  let v2 := (MemSlice_u64.mk ptr len)
  let slice_3 := v2
  slice := slice_3
  let slide : MemSlice_u64 → FeM Unit := fun g => do
    let mut slice := slice
    slice := g
    return ()
  return (slice, slide)

def MemArray_as_slice_mut_u64 (self : MemArray_u64) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemArray_u64)) := do
  let mut self := self
  let mut v3 : MemSlice_u64 := default
  let v1 := self.ptr
  let v2 := self.len
  let (v3_g, v3_k) ← MemSlice_from_raw_parts_mut_u64 v1 v2
  v3 := v3_g
  let slide : MemSlice_u64 → FeM MemArray_u64 := fun g => do
    let mut self := self
    let mut v3 := v3
    v3 := g
    let _ ← v3_k v3
    return self
  return (v3, slide)

def MemSlice_slice_mut_u64 (self : MemSlice_u64) (offset : UInt256) (len : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := do
  let mut self := self
  let mut v12 : MemSlice_u64 := default
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let b1 : MemSlice_u64 → MemSlice_u64 → FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := fun self v12 => do
    let mut self := self
    let mut v12 := v12
    let v13 ← index_out_of_bounds
    nomatch v13
  if v4 then
    b1 self v12
  else
    let v5 := self.len
    let v6 ← checkedSub v5 offset
    let v7 := decide (v6.toNat < len.toNat)
    if v7 then
      b1 self v12
    else
      let v8 := ()
      let v9 := v8
      let v10 := self.ptr
      let v11 ← elem_u64 v10 offset
      let (v12_g, v12_k) ← MemSlice_from_raw_parts_mut_u64 v11 len
      v12 := v12_g
      let slide : MemSlice_u64 → FeM MemSlice_u64 := fun g => do
        let mut self := self
        let mut v12 := v12
        v12 := g
        let _ ← v12_k v12
        return self
      return (v12, slide)

def MemSlice_head_mut_u64 (self : MemSlice_u64) (n : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := do
  let mut self := self
  let mut v3 : MemSlice_u64 := default
  let v2 := (UInt256.ofNat 0)
  let (v3_g, v3_k) ← MemSlice_slice_mut_u64 self v2 n
  v3 := v3_g
  let slide : MemSlice_u64 → FeM MemSlice_u64 := fun g => do
    let mut self := self
    let mut v3 := v3
    v3 := g
    let v3_0 ← v3_k v3
    self := v3_0
    return self
  return (v3, slide)

def byte_ptr_u64 (ptr : UInt256) : FeM UInt256 := do
  let v1 ← cast_u64_u8 ptr
  return v1

def MemSlice_byte_len_u64 (self : MemSlice_u64) : FeM UInt256 := do
  let v1 := self.len
  let v2 := (UInt256.ofNat 8)
  let v3 ← checkedMul v1 v2
  return v3

def MemSlice_as_bytes_u64 (self : MemSlice_u64) : FeM MemSlice_u8 := do
  let v1 := self.ptr
  let v2 ← cast_u64_u8 v1
  let v3 ← MemSlice_byte_len_u64 self
  let v4 ← MemSlice_from_raw_parts_ref_u8 v2 v3
  return v4

def MemSlice_copy_from_u64 (self : MemSlice_u64) (source : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut self := self
  let v2 := source.len
  let v3 := self.len
  let v4 := decide (v2 ≠ v3)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := self.ptr
    let v9 ← byte_ptr_u64 v8
    let v10 ← MemSlice_as_bytes_u64 source
    let v12 ← copy v9 v10
    return ((), self)

def MemSlice_len_u64 (self : MemSlice_u64) : FeM UInt256 := do
  let v1 := self.len
  return v1

def MemSlice_ptr_u64 (self : MemSlice_u64) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

def MemSlice_zero_u64 (self_0 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut self_0 := self_0
  let v1 := self_0.ptr
  let v2 := self_0.len
  let v3 ← drop_in_place_n_u64 v1 v2
  let v4 := self_0.ptr
  let v5 ← byte_ptr_u64 v4
  let v7 ← MemSlice_byte_len_u64 self_0
  let v8 ← zero_bytes v5 v7
  return ((), self_0)

def MemArray_zero_u64 (self : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut self := self
  let mut v1 : MemSlice_u64 := default
  let (v1_g, v1_k) ← MemArray_as_slice_mut_u64 self
  v1 := v1_g
  let (v2, v2_0) ← MemSlice_zero_u64 v1
  v1 := v2_0
  let v1_0 ← v1_k v1
  self := v1_0
  return ((), self)

structure FixedBytes_32 where
  val : UInt256
deriving Inhabited

structure Bytes where
  data : UInt256
  len : UInt256
  size : UInt256
deriving Inhabited

structure FixedBytes_4 where
  val : UInt256
deriving Inhabited

def cast_u8_ChunkPair (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let v2 := ptr_1
  return v2

def alloc_ChunkPair : FeM UInt256 := do
  let v0 := (UInt256.ofNat 64)
  let v1 ← alloc_bytes v0
  let v2 ← cast_u8_ChunkPair v1
  return v2

def write_pair (scratch : UInt256) (left_1 : UInt256) (right_2 : UInt256) : FeM Unit := do
  let left_3 := left_1
  mstore scratch left_3
  let v4 := ()
  let right_5 := right_2
  mstore (scratch + UInt256.ofNat 32) right_5
  let v6 := ()
  return ()

def alloc_pair (left : UInt256) (right : UInt256) : FeM UInt256 := do
  let v2 ← alloc_ChunkPair
  let scratch := v2
  let v4 ← write_pair scratch left right
  return scratch

def cast_ChunkPair_u8 (ptr_0 : UInt256) : FeM UInt256 := do
  let ptr_1 := ptr_0
  let v2 := ptr_1
  return v2

def byte_ptr_ChunkPair (ptr : UInt256) : FeM UInt256 := do
  let v1 ← cast_ChunkPair_u8 ptr
  return v1

def pair_offset (scratch : UInt256) : FeM UInt256 := do
  let v1 ← byte_ptr_ChunkPair scratch
  return v1

def sha256_64_to (offset : UInt256) (out : UInt256) : FeM UInt256 := do
  let v2 ← gas
  let v3 := (UInt256.ofNat 2)
  let v4 := (UInt256.ofNat 64)
  let v5 := (UInt256.ofNat 32)
  let v6 ← staticcallPrecompile v2 v3 offset v4 out v5
  let ok := v6
  let v8 := (UInt256.ofNat 0)
  let v9 := decide (ok = v8)
  if v9 then
    let v10 ← revertEmpty
    nomatch v10
  else
    let v11 := ()
    let v12 := v11
    let v13 ← returndatasize
    let v14 := (UInt256.ofNat 32)
    let v15 := decide (v13.toNat < v14.toNat)
    if v15 then
      let v16 ← revertEmpty
      nomatch v16
    else
      let v17 := ()
      let v18 := v17
      let v19 ← mload out
      return v19

def hash_pair_buffer (scratch : UInt256) : FeM UInt256 := do
  let v1 ← pair_offset scratch
  let offset := v1
  let v3 ← sha256_64_to offset offset
  return v3

def hash_pair (left : UInt256) (right : UInt256) : FeM UInt256 := do
  let v2 ← alloc_pair left right
  let scratch := v2
  let v4 ← hash_pair_buffer scratch
  return v4

def Bytes_len (self : Bytes) : FeM UInt256 := do
  let v1 := self.len
  return v1

def ether (n_0 : UInt256) : FeM Wei := do
  let n_1 := n_0
  let v2 := (UInt256.ofNat 1000000000000000000)
  let v3 ← checkedMul n_1 v2
  let v4 := (Wei.mk v3)
  return v4

def Wei_ge (self : Wei) (other : Wei) : FeM Bool := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 := decide (v3.toNat ≤ v2.toNat)
  return v4

def gwei (n_0 : UInt256) : FeM Wei := do
  let n_1 := n_0
  let v2 := (UInt256.ofNat 1000000000)
  let v3 ← checkedMul n_1 v2
  let v4 := (Wei.mk v3)
  return v4

def Wei_rem (self : Wei) (other : Wei) : FeM Wei := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 ← checkedRem v2 v3
  let v5 := (Wei.mk v4)
  return v5

def Wei_eq (self : Wei) (other : Wei) : FeM Bool := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 := decide (v2 = v3)
  return v4

def Wei_div (self : Wei) (other : Wei) : FeM UInt256 := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 ← checkedDiv v2 v3
  return v4

def Bytes_payload_ptr (self : Bytes) : FeM UInt256 := do
  let v1 := self.data
  let v2 := (UInt256.ofNat 32)
  let v3 := v1 + v2
  return v3

def Bytes_span_ref (self : Bytes) : FeM MemSlice_u8 := do
  let v1 ← Bytes_payload_ptr self
  let v2 := self.len
  let v3 ← MemSlice_from_raw_parts_ref_u8 v1 v2
  return v3

def elem_u8 (ptr : UInt256) (index_1 : UInt256) : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1)
  let index_3 := index_1
  let v4 ← usize_wrapping_mul index_3 v2
  let v5 := ptr + v4
  return v5

def MemSlice_slice_ref_u8 (self : MemSlice_u8) (offset : UInt256) (len : UInt256) : FeM MemSlice_u8 := do
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let b1 : FeM MemSlice_u8 := do
    let v13 ← index_out_of_bounds
    nomatch v13
  if v4 then
    b1
  else
    let v5 := self.len
    let v6 ← checkedSub v5 offset
    let v7 := decide (v6.toNat < len.toNat)
    if v7 then
      b1
    else
      let v8 := ()
      let v9 := v8
      let v10 := self.ptr
      let v11 ← elem_u8 v10 offset
      let v12 ← MemSlice_from_raw_parts_ref_u8 v11 len
      return v12

def ByteVector_48_hash_tree_root (input : MemSlice_u8) (start : UInt256) (len_2 : UInt256) : FeM UInt256 := do
  let len_3 := len_2
  let v4 := (UInt256.ofNat 48)
  let v5 := decide (len_3 = v4)
  if v5 then
    let v6 := ()
    let v7 ← downcast_unchecked_u256_usize start
    let v8 ← downcast_unchecked_u256_usize len_2
    let v9 ← MemSlice_slice_ref_u8 input v7 v8
    let v11 ← MemSlice_ptr_u8 v9
    let base := v11
    let v13 := (UInt256.ofNat 32)
    let v14 := base + v13
    let v15 ← mload v14
    let tail := v15
    let v17 ← mload base
    let v18 := (UInt256.ofNat 128)
    let v19 := tail >>> v18
    let v20 := (UInt256.ofNat 128)
    let v21 := v19 <<< v20
    let v22 ← alloc_pair v17 v21
    let scratch := v22
    let v24 ← hash_pair_buffer scratch
    return v24
  else
    M.panic 0x01

def hash_tree_root_ByteVector_48 (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  let v3 ← ByteVector_48_hash_tree_root input start len
  return v3

def Bytes_word_at (self : Bytes) (offset_1 : UInt256) : FeM UInt256 := do
  let offset_2 := offset_1
  let v3 := self.len
  let v4 := decide (v3.toNat ≤ offset_2.toNat)
  if v4 then
    let v5 ← trap
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← Bytes_payload_ptr self
    let v9 := v8 + offset_1
    let v10 := self.len
    let offset_11 := offset_1
    let v12 ← checkedSub v10 offset_11
    let v13 ← readPaddedWord v9 v12
    return v13

def u64_chunk_k1 (v22 : UInt256) (v : UInt256) : FeM UInt256 := do
  let mut v := v
  let v23 := (UInt256.ofNat 16)
  let v24 := v22 <<< v23
  let v_25 := v
  let v_26 := v_25
  let v27 := (UInt256.ofNat 16)
  let v28 := v_26 >>> v27
  let v29 := (UInt256.ofNat 281470681808895)
  let v30 := v28 &&& v29
  let v31 := v24 ||| v30
  let v_32 := v31
  v := v_32
  let v33 := ()
  let v_34 := v
  let v_35 := v_34
  let v36 := (UInt256.ofNat 32)
  let v37 := v_35 <<< v36
  let v_38 := v
  let v_39 := v_38
  let v40 := (UInt256.ofNat 32)
  let v41 := v_39 >>> v40
  let v42 := v37 ||| v41
  let v43 := (UInt256.ofNat 192)
  let v44 := v42 <<< v43
  return v44

def u64_chunk (value_0 : UInt256) : FeM UInt256 := do
  let mut v : UInt256 := default
  let value_1 := value_0
  let v2 := value_1
  let v_3 := v2
  v := v_3
  let v_4 := v
  let v_5 := v_4
  let v6 := (UInt256.ofNat 71777214294589695)
  let v7 := v_5 &&& v6
  let v8 := (UInt256.ofNat 8)
  let v9 := v7 <<< v8
  let v_10 := v
  let v_11 := v_10
  let v12 := (UInt256.ofNat 8)
  let v13 := v_11 >>> v12
  let v14 := (UInt256.ofNat 71777214294589695)
  let v15 := v13 &&& v14
  let v16 := v9 ||| v15
  let v_17 := v16
  v := v_17
  let v18 := ()
  let v_19 := v
  let v_20 := v_19
  let v21 := (UInt256.ofNat 281470681808895)
  let v22 := v_20 &&& v21
  u64_chunk_k1 v22 v

def ByteVector_96_hash_tree_root (input : MemSlice_u8) (start : UInt256) (len_2 : UInt256) : FeM UInt256 := do
  let len_3 := len_2
  let v4 := (UInt256.ofNat 96)
  let v5 := decide (len_3 = v4)
  if v5 then
    let v6 := ()
    let v7 ← downcast_unchecked_u256_usize start
    let v8 ← downcast_unchecked_u256_usize len_2
    let v9 ← MemSlice_slice_ref_u8 input v7 v8
    let v11 ← MemSlice_ptr_u8 v9
    let base := v11
    let v13 := (UInt256.ofNat 0)
    let v14 := (UInt256.ofNat 0)
    let v15 ← alloc_pair v13 v14
    let scratch := v15
    let v17 ← pair_offset scratch
    let offset := v17
    let v19 ← sha256_64_to base offset
    let left := v19
    let v21 := (UInt256.ofNat 64)
    let v22 := base + v21
    let v23 ← mload v22
    let v24 := (UInt256.ofNat 0)
    let v25 ← write_pair scratch v23 v24
    let v26 ← hash_pair_buffer scratch
    let right := v26
    let v28 ← write_pair scratch left right
    let v29 ← hash_pair_buffer scratch
    return v29
  else
    M.panic 0x01

def hash_tree_root_ByteVector_96 (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  let v3 ← ByteVector_96_hash_tree_root input start len
  return v3

def merkleize4 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) : FeM UInt256 := do
  let v4 ← alloc_pair a b
  let scratch := v4
  let v6 ← hash_pair_buffer scratch
  let left := v6
  let v8 ← write_pair scratch c d
  let v9 ← hash_pair_buffer scratch
  let right := v9
  let v11 ← write_pair scratch left right
  let v12 ← hash_pair_buffer scratch
  return v12

def tuple_u256_u256_u256_u256_merkleize (self : (UInt256 × UInt256 × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 := self.fst
  let v2 := self.snd.fst
  let v3 := self.snd.snd.fst
  let v4 := self.snd.snd.snd
  let v5 ← merkleize4 v1 v2 v3 v4
  return v5

def Bytes_from_validated_abi_tail (data_0 : UInt256) (len_1 : UInt256) (size_2 : UInt256) : FeM Bytes := do
  let data_3 := data_0
  let len_4 := len_1
  let size_5 := size_2
  let v6 := (Bytes.mk data_3 len_4 size_5)
  return v6

def Bytes_empty : FeM Bytes := do
  let v0 := (UInt256.ofNat 32)
  let v1 ← alloc_bytes v0
  let data := v1
  let v3 := (UInt256.ofNat 0)
  let v4 ← store_word data v3
  let v5 := (UInt256.ofNat 0)
  let v6 := (UInt256.ofNat 32)
  let v7 ← Bytes_from_validated_abi_tail data v5 v6
  return v7

def Bytes_from_word_prefix (word_0 : UInt256) (len_1 : UInt256) : FeM Bytes := do
  let len_2 := len_1
  let v3 := (UInt256.ofNat 32)
  let v4 := decide (v3.toNat < len_2.toNat)
  if v4 then
    let v5 ← trap
    nomatch v5
  else
    let v6 := ()
    let v7 := v6
    let len_8 := len_1
    let v9 := (UInt256.ofNat 0)
    let v10 := decide (len_8 = v9)
    if v10 then
      let v11 ← Bytes_empty
      return v11
    else
      let v12 := ()
      let v13 := v12
      let v14 := (UInt256.ofNat 64)
      let v15 ← alloc_bytes v14
      let data := v15
      let v17 := (UInt256.ofNat 32)
      let len_18 := len_1
      let v19 := len_18
      let v20 ← checkedSub v17 v19
      let v21 := (UInt256.ofNat 8)
      let v22 ← checkedMul v20 v21
      let shift := v22
      let len_24 := len_1
      let v25 := len_24
      let v26 ← store_word data v25
      let v27 := (UInt256.ofNat 32)
      let v28 := data + v27
      let word_29 := word_0
      let v30 := word_29 >>> shift
      let v31 := v30 <<< shift
      let v32 ← store_word v28 v31
      let v33 := (UInt256.ofNat 64)
      let v34 ← Bytes_from_validated_abi_tail data len_1 v33
      return v34

def serialize_u64 (value : UInt256) : FeM Bytes := do
  let v1 ← u64_chunk value
  let v2 := (UInt256.ofNat 8)
  let v3 ← Bytes_from_word_prefix v1 v2
  return v3

def Bytes_IS_DYNAMIC : FeM Bool := do
  let v0 := true
  return v0

def Bytes_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def abi_field_head_size_Bytes : FeM UInt256 := do
  let v0 ← Bytes_IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← Bytes_HEAD_SIZE
    return v4

def tuple_HEAD_SIZE_Bytes_Bytes_Bytes_Bytes_Bytes : FeM UInt256 := do
  let v0 ← abi_field_head_size_Bytes
  let v1 ← abi_field_head_size_Bytes
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Bytes
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_Bytes
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_Bytes
  let v8 ← checkedAdd v6 v7
  return v8

def Bytes_encoded_size (self : Bytes) : FeM UInt256 := do
  let v1 := self.size
  return v1

def Bytes_payload_size (self : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_encoded_size self
  return v1

def dynamic_payload_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_IS_DYNAMIC
  if v1 then
    let v2 ← Bytes_payload_size value
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 := (UInt256.ofNat 0)
    return v5

def tuple_payload_size_Bytes_Bytes_Bytes_Bytes_Bytes (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE_Bytes_Bytes_Bytes_Bytes_Bytes
  let v4 ← dynamic_payload_size_Bytes self.fst
  let v5 := v1 + v4
  let v8 ← dynamic_payload_size_Bytes self.snd.fst
  let v9 := v5 + v8
  let v12 ← dynamic_payload_size_Bytes self.snd.snd.fst
  let v13 := v9 + v12
  let v16 ← dynamic_payload_size_Bytes self.snd.snd.snd.fst
  let v17 := v13 + v16
  let v20 ← dynamic_payload_size_Bytes self.snd.snd.snd.snd
  let v21 := v17 + v20
  return v21

def abi_payload_size_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_Bytes_Bytes_Bytes_Bytes_Bytes value
  return v1

def Bytes_encoded_span (self : Bytes) : FeM MemSlice_u8 := do
  let v1 := self.data
  let v2 := self.size
  let v3 ← word_to_usize v2
  let v4 ← MemSlice_from_raw_parts_ref_u8 v1 v3
  return v4

def Bytes_encode_Sol (self : Bytes) (out : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← Bytes_encoded_span self
  let v5 ← AbiFrame_copy_from out v2 v3
  return ()

def encode_field_Sol_Bytes (value : Bytes) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    let v4 ← Bytes_payload_size value
    let size := v4
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let v10 ← AbiFrame_store_word out head_pos v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← AbiFrame_slice tails v11 size
    r1 := v13
    let v15 ← Bytes_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← Bytes_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let v23 ← Bytes_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes_k1 (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) (out : AbiFrame) (head1 : UInt256) (head2 : UInt256) (head3 : UInt256) (head4 : UInt256) (tails_29 : AbiFrame) (tails : AbiFrame) : FeM Unit := do
  let mut tails := tails
  let v31 ← encode_field_Sol_Bytes self.snd.fst out head1 tails_29
  let tails_32 := v31
  tails := tails_32
  let v33 := ()
  let tails_34 := tails
  let tails_35 := tails_34
  let v37 ← encode_field_Sol_Bytes self.snd.snd.fst out head2 tails_35
  let tails_38 := v37
  tails := tails_38
  let v39 := ()
  let tails_40 := tails
  let tails_41 := tails_40
  let v43 ← encode_field_Sol_Bytes self.snd.snd.snd.fst out head3 tails_41
  let tails_44 := v43
  tails := tails_44
  let v45 := ()
  let tails_46 := tails
  let tails_47 := tails_46
  let v49 ← encode_field_Sol_Bytes self.snd.snd.snd.snd out head4 tails_47
  return ()

def tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) (out : AbiFrame) : FeM Unit := do
  let mut tails : AbiFrame := default
  let v7 ← tuple_HEAD_SIZE_Bytes_Bytes_Bytes_Bytes_Bytes
  let v8 ← AbiFrame_tail out v7
  let tails_9 := v8
  tails := tails_9
  let v10 ← abi_field_head_size_Bytes
  let head1 := v10
  let v12 ← abi_field_head_size_Bytes
  let v13 := head1 + v12
  let head2 := v13
  let v15 ← abi_field_head_size_Bytes
  let v16 := head2 + v15
  let head3 := v16
  let v18 ← abi_field_head_size_Bytes
  let v19 := head3 + v18
  let head4 := v19
  let v21 := (UInt256.ofNat 0)
  let tails_22 := tails
  let tails_23 := tails_22
  let v25 ← encode_field_Sol_Bytes self.fst out v21 tails_23
  let tails_26 := v25
  tails := tails_26
  let v27 := ()
  let tails_28 := tails
  let tails_29 := tails_28
  tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes_k1 self out head1 head2 head3 head4 tails_29 tails

def Bytes_drop (self : Bytes) : FeM Unit := do
  let v1 := self.data
  let v2 ← dealloc_u8 v1
  return ()

def encode_alloc_Sol_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_tuple_Bytes_Bytes_Bytes_Bytes_Bytes value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let v11 ← tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes value r2
  let _ ← Bytes_drop value.fst
  let _ ← Bytes_drop value.snd.fst
  let _ ← Bytes_drop value.snd.snd.fst
  let _ ← Bytes_drop value.snd.snd.snd.fst
  let _ ← Bytes_drop value.snd.snd.snd.snd
  let buffer_12 := buffer
  return buffer_12

def encode_abi_payload_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value_0 : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM MemBuffer := do
  let value_1 := value_0
  let v2 ← encode_alloc_Sol_tuple_Bytes_Bytes_Bytes_Bytes_Bytes value_1
  return v2

def Evm_log1 (data : MemSlice_u8) (topic0 : UInt256) : FeM Unit := do
  let v3 ← MemSlice_ptr_u8 data
  let v4 ← MemSlice_len_u8 data
  let v5 := v4
  let v6 ← logData v3 v5 #[topic0]
  return ()

def mix_in_length (root : UInt256) (len : UInt256) : FeM UInt256 := do
  let v2 ← u64_chunk len
  let v3 ← hash_pair root v2
  return v3

def FixedBytes_IS_DYNAMIC_32 : FeM Bool := do
  let v0 := false
  return v0

def FixedBytes_HEAD_SIZE_32 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def abi_field_head_size_FixedBytes_32 : FeM UInt256 := do
  let v0 ← FixedBytes_IS_DYNAMIC_32
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← FixedBytes_HEAD_SIZE_32
    return v4

def MAX_PAYLOAD_LEN : FeM UInt256 := do
  let v0 ← MAX_ALLOCATION
  let v1 := v0
  let v2 := (UInt256.ofNat 31)
  let v3 ← checkedSub v1 v2
  return v3

def checked_dynamic_payload_after_head_Sol_CallData (input : CallData) (pos_1 : UInt256) (input_len_2 : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← CallData_word_at input pos_1
  let data_len := v3
  let input_len_5 := input_len_2
  let pos_6 := pos_1
  let v7 := (UInt256.ofNat 32)
  let v8 ← checkedAdd pos_6 v7
  let v9 ← checkedSub input_len_5 v8
  let v10 := decide (v9.toNat < data_len.toNat)
  let b1 : FeM (UInt256 × UInt256) := do
    let v23 ← Sol_decode_error
    nomatch v23
  if v10 then
    b1
  else
    let v11 ← MAX_PAYLOAD_LEN
    let v12 := (UInt256.ofNat 32)
    let v13 ← checkedSub v11 v12
    let v14 := decide (v13.toNat < data_len.toNat)
    if v14 then
      b1
    else
      let v15 := ()
      let v16 := v15
      let v17 := (UInt256.ofNat 63)
      let v18 ← checkedAdd data_len v17
      let v19 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639904)
      let v20 := v18 &&& v19
      let payload_size := v20
      let v22 := (data_len, payload_size)
      return v22

def checked_dynamic_payload_with_input_len_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 := (UInt256.ofNat 32)
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← checked_dynamic_payload_after_head_Sol_CallData input pos input_len
  return v5

def alloc_bytes_unchecked (len : UInt256) : FeM UInt256 := do
  let v1 ← alloc len
  return v1

def CallData_copy_to_memory_unchecked (self : CallData) (dest : UInt256) (src_offset_2 : UInt256) (len : UInt256) : FeM Unit := do
  let v4 := self.base
  let src_offset_5 := src_offset_2
  let v6 := v4 + src_offset_5
  let v7 ← calldatacopy dest v6 len
  return ()

def copy_padded_payload_CallData (input : CallData) (pos : UInt256) (dest : UInt256) (actual_size : UInt256) (size_4 : UInt256) : FeM Unit := do
  let size_5 := size_4
  let v6 := (UInt256.ofNat 32)
  let v7 ← checkedSub size_5 v6
  let v8 ← ptr_offset dest v7
  let v9 := (UInt256.ofNat 0)
  let v10 ← store_word v8 v9
  let v11 ← CallData_copy_to_memory_unchecked input dest pos actual_size
  return ()

def copy_dynamic_bytes_CallData (input : CallData) (pos : UInt256) (dest : UInt256) (data_len_3 : UInt256) (size : UInt256) : FeM Unit := do
  let v5 := (UInt256.ofNat 32)
  let data_len_6 := data_len_3
  let v7 ← checkedAdd v5 data_len_6
  let v8 ← copy_padded_payload_CallData input pos dest v7 size
  return ()

def copy_dynamic_payload_from_with_input_len_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256 × UInt256) := do
  let v3 ← checked_dynamic_payload_with_input_len_Sol_CallData input pos input_len
  let payload := v3
  let v5 := payload.fst
  let data_len := v5
  let v7 := payload.snd
  let payload_size := v7
  let v9 ← word_to_usize payload_size
  let v10 ← alloc_bytes_unchecked v9
  let data_ptr := v10
  let v12 ← copy_dynamic_bytes_CallData input pos data_ptr data_len payload_size
  let v13 := (data_ptr, data_len, payload_size)
  return v13

def Bytes_decode_from_bounded_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  let v3 ← copy_dynamic_payload_from_with_input_len_Sol_CallData input pos input_len
  let payload := v3
  let v5 := payload.fst
  let v6 := payload.snd.fst
  let v7 ← word_to_usize v6
  let v8 := payload.snd.snd
  let v9 ← Bytes_from_validated_abi_tail v5 v7 v8
  return v9

def copy_dynamic_payload_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM (UInt256 × UInt256 × UInt256) := do
  let v2 ← len_CallData input
  let input_len := v2
  let v4 ← copy_dynamic_payload_from_with_input_len_Sol_CallData input pos input_len
  return v4

def Bytes_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM Bytes := do
  let v2 ← copy_dynamic_payload_from_Sol_CallData input pos
  let payload := v2
  let v4 := payload.fst
  let v5 := payload.snd.fst
  let v6 ← word_to_usize v5
  let v7 := payload.snd.snd
  let v8 ← Bytes_from_validated_abi_tail v4 v6 v7
  return v8

def decode_field_from_prechecked_head_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  let v4 ← Bytes_IS_DYNAMIC
  if v4 then
    let v5 ← CallData_word_at input head_pos
    let tail_rel := v5
    let v7 ← checked_tail_Sol base tail_rel
    let tail_at := v7
    let v9 ← Bytes_decode_from_bounded_Sol_CallData input tail_at input_len
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← Bytes_decode_from_Sol_CallData input head_pos
    return v12

def decode_field_from_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM Bytes := do
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    let v4 ← CallData_word_at input head_pos
    let tail_rel := v4
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← Bytes_decode_from_Sol_CallData input v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← Bytes_decode_from_Sol_CallData input head_pos
    return v10

def decode_msg_field_from_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  let v4 ← Bytes_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_Bytes_CallData input base head_pos input_len
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_Bytes_CallData input base head_pos
    return v8

def decode_fixed_bytes_word (word_0 : UInt256) (len_1 : UInt256) : FeM UInt256 := do
  let len_2 := len_1
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (len_2 = v3)
  let b1 : FeM UInt256 := do
    let v33 ← revertEmpty
    nomatch v33
  if v4 then
    b1
  else
    let len_5 := len_1
    let v6 := (UInt256.ofNat 32)
    let v7 := decide (v6.toNat < len_5.toNat)
    if v7 then
      b1
    else
      let v8 := ()
      let v9 := v8
      let len_10 := len_1
      let v11 := (UInt256.ofNat 32)
      let v12 := decide (len_10 = v11)
      if v12 then
        let word_13 := word_0
        return word_13
      else
        let v14 := ()
        let v15 := v14
        let word_16 := word_0
        let len_17 := len_1
        let v18 := (UInt256.ofNat 8)
        let v19 := len_17 * v18
        let v20 := word_16 <<< v19
        let v21 := (UInt256.ofNat 0)
        let v22 := decide (v20 = v21)
        if v22 then
          let word_23 := word_0
          let v24 := (UInt256.ofNat 32)
          let len_25 := len_1
          let v26 := v24 - len_25
          let v27 := (UInt256.ofNat 8)
          let v28 := v26 * v27
          let v29 := word_23 >>> v28
          return v29
        else
          let v30 := ()
          let v31 := v30
          let v32 ← revertEmpty
          let v32 : UInt256 := nomatch v32
          return v32

def FixedBytes_decode_from_Sol_32_CallData (input : CallData) (pos : UInt256) : FeM FixedBytes_32 := do
  let v2 ← CallData_word_at input pos
  let v3 := (UInt256.ofNat 32)
  let v4 := v3
  let v5 ← decode_fixed_bytes_word v2 v4
  let v6 := (FixedBytes_32.mk v5)
  return v6

def decode_from_bounded_FixedBytes_32_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  let v3 ← FixedBytes_HEAD_SIZE_32
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← FixedBytes_decode_from_Sol_32_CallData input pos
  return v5

def decode_field_from_prechecked_head_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  let v4 ← FixedBytes_IS_DYNAMIC_32
  if v4 then
    let v5 ← CallData_word_at input head_pos
    let tail_rel := v5
    let v7 ← checked_tail_Sol base tail_rel
    let tail_at := v7
    let v9 ← decode_from_bounded_FixedBytes_32_Sol_CallData input tail_at input_len
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← FixedBytes_decode_from_Sol_32_CallData input head_pos
    return v12

def decode_field_from_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM FixedBytes_32 := do
  let v3 ← FixedBytes_IS_DYNAMIC_32
  if v3 then
    let v4 ← CallData_word_at input head_pos
    let tail_rel := v4
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← FixedBytes_decode_from_Sol_32_CallData input v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← FixedBytes_decode_from_Sol_32_CallData input head_pos
    return v10

def decode_msg_field_from_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  let v4 ← FixedBytes_IS_DYNAMIC_32
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_FixedBytes_32_CallData input base head_pos input_len
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_FixedBytes_32_CallData input base head_pos
    return v8

def const_Bytes : FeM Bool := do
  let v0 := false
  return v0

def abi_payload_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_payload_size value
  return v1

def encode_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_Bytes value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let v11 ← Bytes_encode_Sol value r2
  let _ ← Bytes_drop value
  let buffer_12 := buffer
  return buffer_12

def abi_field_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_payload_size value
  let payload_size := v1
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    let v4 := (UInt256.ofNat 32)
    let v5 ← checkedAdd v4 payload_size
    return v5
  else
    let v6 := ()
    let v7 := v6
    return payload_size

def abi_single_root_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← abi_field_size_Bytes value
  return v1

def abi_single_root_head_size_Bytes : FeM UInt256 := do
  let v0 ← abi_field_head_size_Bytes
  return v0

def encode_single_root_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let v1 ← abi_single_root_size_Bytes value
  let total := v1
  let v3 ← allocation_len total
  let v4 ← MemBuffer_new v3
  let buffer_5 := v4
  buffer := buffer_5
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  let out_9 := v8
  out := out_9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_Bytes
  let v13 ← AbiFrame_tail out v11
  let v16 ← encode_field_Sol_Bytes value out v10 v13
  let _ ← Bytes_drop value
  let buffer_17 := buffer
  return buffer_17

def encode_return_alloc_Sol_Bytes (value_0 : Bytes) : FeM MemBuffer := do
  let v1 ← const_Bytes
  if v1 then
    let value_2 := value_0
    let v3 ← encode_alloc_Sol_Bytes value_2
    return v3
  else
    let v4 := ()
    let v5 := v4
    let value_6 := value_0
    let v7 ← encode_single_root_alloc_Sol_Bytes value_6
    return v7

def Evm_return_value_Sol_Bytes (value : Bytes) : FeM Empty := do
  let mut value := value
  let mut out_r1 : FixedMemBuffer_32 := default
  let mut out_r2 : FixedMemBuffer_64 := default
  let mut out_r3 : MemBuffer := default
  let mut out_r4 : MemBuffer := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let mut r7 : AbiFrame := default
  let v1 ← Bytes_IS_DYNAMIC
  let v2 := !v1
  if v2 then
    let v3 ← Bytes_HEAD_SIZE
    let v4 := (UInt256.ofNat 32)
    let v5 := decide (v3.toNat ≤ v4.toNat)
    if v5 then
      let v6 ← FixedMemBuffer_alloc_32
      let out_7 := v6
      out_r1 := out_7
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← Bytes_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let v14 ← Bytes_encode_Sol value r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← Bytes_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← Bytes_HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← FixedMemBuffer_alloc_64
        let out_27 := v26
        out_r2 := out_27
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← Bytes_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let v34 ← Bytes_encode_Sol value r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← Bytes_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← Bytes_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let v45 ← MemBuffer_new v44
        let out_46 := v45
        out_r3 := out_46
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← Bytes_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let v53 ← Bytes_encode_Sol value r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let value_60 := value_59
    let v61 ← encode_return_alloc_Sol_Bytes value_60
    let out_62 := v61
    out_r4 := out_62
    let v64 ← MemBuffer_span_ref out_r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

def FixedBytes_IS_DYNAMIC_4 : FeM Bool := do
  let v0 := false
  return v0

def FixedBytes_HEAD_SIZE_4 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def abi_field_head_size_FixedBytes_4 : FeM UInt256 := do
  let v0 ← FixedBytes_IS_DYNAMIC_4
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← FixedBytes_HEAD_SIZE_4
    return v4

def FixedBytes_decode_from_Sol_4_CallData (input : CallData) (pos : UInt256) : FeM FixedBytes_4 := do
  let v2 ← CallData_word_at input pos
  let v3 := (UInt256.ofNat 4)
  let v4 := v3
  let v5 ← decode_fixed_bytes_word v2 v4
  let v6 := (FixedBytes_4.mk v5)
  return v6

def decode_from_bounded_FixedBytes_4_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  let v3 ← FixedBytes_HEAD_SIZE_4
  let v4 ← checked_frame_end_Sol pos v3 input_len
  let v5 ← FixedBytes_decode_from_Sol_4_CallData input pos
  return v5

def decode_field_from_prechecked_head_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  let v4 ← FixedBytes_IS_DYNAMIC_4
  if v4 then
    let v5 ← CallData_word_at input head_pos
    let tail_rel := v5
    let v7 ← checked_tail_Sol base tail_rel
    let tail_at := v7
    let v9 ← decode_from_bounded_FixedBytes_4_Sol_CallData input tail_at input_len
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← FixedBytes_decode_from_Sol_4_CallData input head_pos
    return v12

def decode_field_from_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM FixedBytes_4 := do
  let v3 ← FixedBytes_IS_DYNAMIC_4
  if v3 then
    let v4 ← CallData_word_at input head_pos
    let tail_rel := v4
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← FixedBytes_decode_from_Sol_4_CallData input v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← FixedBytes_decode_from_Sol_4_CallData input head_pos
    return v10

def decode_msg_field_from_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  let v4 ← FixedBytes_IS_DYNAMIC_4
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_FixedBytes_4_CallData input base head_pos input_len
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_FixedBytes_4_CallData input base head_pos
    return v8


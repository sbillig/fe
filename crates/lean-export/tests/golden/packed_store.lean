def Field.pack : Place :=
  .storage (UInt256.ofNat 0)

def Layout.StoragePackedArray_8 : Layout :=
  { storage := [0] }

structure CallData where
  base : UInt256
deriving Inhabited

structure Get0 where
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

structure Get1 where
deriving Inhabited

structure Set0 where
  value : UInt256
deriving Inhabited

structure Set1 where
  value : UInt256
deriving Inhabited

structure SetBoth where
  first : UInt256
  second : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def LANES_PER_SLOT_8 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 256)
  let v1 := (UInt256.ofNat 8)
  let v2 ← checkedDiv v0 v1
  return v2

def LANE_MASK_8 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := (UInt256.ofNat 8)
  let v2 := v0 <<< v1
  let v3 := (UInt256.ofNat 1)
  let v4 ← checkedSub v2 v3
  return v4

def get_8 (v0 : Place) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 ← LANES_PER_SLOT_8
  let v4 ← checkedRem v2 v3
  let v5 := (UInt256.ofNat 8)
  let v6 := v4 * v5
  let v7 := v6
  let v8 := v1
  let v9 ← LANES_PER_SLOT_8
  let v10 ← checkedDiv v8 v9
  let v11 ← load ((v0.field Layout.StoragePackedArray_8 0).entry v10)
  let v12 := v11 >>> v7
  let v13 ← LANE_MASK_8
  let v14 := v12 &&& v13
  return v14

def recv0_arm0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← get_8 v2 v0
  return v3

def recv0_arm1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  let v1 := Field.pack
  let v2 := v1
  let v3 ← get_8 v2 v0
  return v3

def U256_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)
  return v0

def set_8 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := v1
  let v4 ← LANES_PER_SLOT_8
  let v5 ← checkedRem v3 v4
  let v6 := (UInt256.ofNat 8)
  let v7 := v5 * v6
  let v8 := v7
  let v9 ← LANE_MASK_8
  let v10 := v9 <<< v8
  let v11 := v10
  let v12 := v1
  let v13 ← LANES_PER_SLOT_8
  let v14 ← checkedDiv v12 v13
  let v15 := ((v0.field Layout.StoragePackedArray_8 0).entry v14)
  let v16 := v15
  let v17 ← load v16
  let v18 ← U256_MASK
  let v19 := v18 ^^^ v11
  let v20 := v17 &&& v19
  let v21 := v2
  let v22 ← LANE_MASK_8
  let v23 := v21 &&& v22
  let v24 := v23 <<< v8
  let v25 := v20 ||| v24
  store v16 v25
  let v26 := ()
  return ()

def recv0_arm2 (v0 : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 0)
  let v2 := Field.pack
  let v3 ← set_8 v2 v1 v0
  return ()

def recv0_arm3 (v0 : UInt256) : FeM Unit := do
  let v1 := (UInt256.ofNat 1)
  let v2 := Field.pack
  let v3 ← set_8 v2 v1 v0
  return ()

def recv0_arm4 (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := Field.pack
  let v4 ← set_8 v3 v2 v0
  let v5 := (UInt256.ofNat 1)
  let v6 := Field.pack
  let v7 ← set_8 v6 v5 v1
  return ()

def new : FeM CallData := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (CallData.mk v0)
  return v1

def input : FeM CallData := do
  let v1 ← new
  return v1

def SELECTOR_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4)
  return v0

def to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def U64_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

def SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
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

def from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := truncate 64 v0
  return v1

def downcast_truncate_u256_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← to_word v0
  let v3 := v2
  let v4 ← IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← MASK
    let v6 ← SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← from_word v7
    let v9 := v8
    b3 v9
  else
    let v10 ← MASK
    let v11 := v3 &&& v10
    let v12 ← from_word v11
    let v13 := v12
    b3 v13

def size (v0 : CallData) : FeM UInt256 := do
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
  let v1 ← size v0
  let v2 := v1
  return v2

def HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def from_word_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def validate_runtime_head (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := v2
  let v4 := (UInt256.ofNat 0)
  let v5 := decide (v3 ≠ v4)
  let b3 : FeM Unit := do
    return ()
  if v5 then
    let v6 := v0
    let v7 := v1
    let v8 := v6 - v7
    let v9 ← from_word_1 v8
    let v10 := v9
    let v11 ← from_word_1 v2
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

def decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Get0 := do
  let v2 := (Get0.mk)
  return v2

def decode_from_prechecked_head_Get0_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Get0 := do
  let v3 ← decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Get0 : FeM Get0 := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Get0_Sol_CallData v11 v5 v8
  return v12

def IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def HEAD_SIZE_1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def to_word_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

def MASK_1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  let v2 := v1
  return v2

def I256_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 57896044618658097711785492504343953926634992332820282019728792003956564819968)
  return v0

def downcast_saturate_u64_usize (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let v8 ← to_word_1 v0
  let v9 := v8
  let v10 ← IS_SIGNED
  let v11 := !v10
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun r0 v1 => do
    let mut r0 := r0
    return v1
  if v11 then
    let v12 := v9
    r0 := v12
    let v13 ← IS_SIGNED
    let b6 : UInt256 → FeM UInt256 := fun r0 => do
      let mut r0 := r0
      let v24 := r0
      let v25 := v24
      let v26 ← MASK_1
      let v27 := decide (v26.toNat < v25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 v2 => do
        let mut r0 := r0
        let v35 := v2
        b3 r0 v35
      if v27 then
        let v28 ← MASK_1
        let v29 ← from_word_1 v28
        let v30 := v29
        b10 r0 v30
      else
        let v31 := r0
        let v32 := v31
        let v33 ← from_word_1 v32
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
    let v36 ← SIGN_BIT
    let v37 ← MASK_1
    let v38 ← SIGN_BIT
    let v39 ← sign_extend v36 v37 v38
    let v40 := v39
    let v41 ← SIGN_BIT
    let v42 := (UInt256.ofNat 1)
    let v43 ← checkedSub v41 v42
    let v44 := v43
    let v45 := v40
    let v46 := v45
    let v47 := v44
    let v48 := v47
    let v49 ← IS_SIGNED
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
        let v53 ← from_word_1 v40
        let v54 := v53
        b16 r0 v54
      else
        let v55 := UInt256.sgtBool v51 v48
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun r0 v5 => do
          let mut r0 := r0
          let v60 := v5
          b16 r0 v60
        if v55 then
          let v56 ← from_word_1 v44
          let v57 := v56
          b19 r0 v57
        else
          let v58 ← from_word_1 v9
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
        let v66 ← from_word_1 v44
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
          let v71 ← from_word_1 v44
          let v72 := v71
          b25 r0 v72
        else
          let v73 ← from_word_1 v9
          let v74 := v73
          b25 r0 v74

def MAX_ALLOCATION : FeM UInt256 := do
  let v0 ← max
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

def alloc_uninit_32 : FeM FixedMemBuffer_32 := do
  let v0 := (UInt256.ofNat 32)
  let v1 ← alloc_bytes v0
  let v2 := (FixedMemBuffer_32.mk v1)
  return v2

def zero_bytes (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 ← zeroMem v0 v1
  return ()

def alloc_32 : FeM FixedMemBuffer_32 := do
  let v0 ← alloc_uninit_32
  let v1 := v0
  let v2 := v1.ptr
  let v3 := (UInt256.ofNat 32)
  let v4 ← zero_bytes v2 v3
  return v1

def ptr_32 (v0 : FixedMemBuffer_32) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def from_raw_parts (v0 : UInt256) (v1 : UInt256) : FeM AbiFrame := do
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

def slice (v0 : AbiFrame) (v1 : UInt256) (v2 : UInt256) : FeM AbiFrame := do
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

def store_word_1 (v0 : AbiFrame) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let v3 := (UInt256.ofNat 32)
  let v4 ← slice v0 v1 v3
  let v5 := v4
  let v6 := v5.ptr
  let v7 ← store_word v6 v2
  return ()

def encode_Sol (v0 : UInt256) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 ← store_word_1 v1 v2 v0
  return ()

def downcast_unchecked_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let v1 ← to_word v0
  let v2 := v1
  let v3 ← from_word_1 v2
  return v3

def from_raw_parts_ref_u8 (v0 : UInt256) (v1 : UInt256) : FeM MemSlice_u8 := do
  let mut r0 : MemSlice_u8 := default
  let v2 := (MemSlice_u8.mk v0 v1)
  let v3 := v2
  r0 := v3
  return r0

def ptr_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def len_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := v0.len
  return v1

def return_data (v1 : MemSlice_u8) : FeM Empty := do
  let v2 ← ptr_u8 v1
  let v3 ← len_u8 v1
  let v4 := v3
  let v5 ← returnData v2 v4
  nomatch v5

def alloc_uninit_64 : FeM FixedMemBuffer_64 := do
  let v0 := (UInt256.ofNat 64)
  let v1 ← alloc_bytes v0
  let v2 := (FixedMemBuffer_64.mk v1)
  return v2

def alloc_64 : FeM FixedMemBuffer_64 := do
  let v0 ← alloc_uninit_64
  let v1 := v0
  let v2 := v1.ptr
  let v3 := (UInt256.ofNat 64)
  let v4 ← zero_bytes v2 v3
  return v1

def ptr_64 (v0 : FixedMemBuffer_64) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def new_uninit (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes v0
  let v2 := v0
  let v3 := v0
  let v4 := (MemBuffer.mk v1 v2 v3)
  return v4

def new_1 (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← new_uninit v0
  let v2 := v1
  let v3 := v2.ptr
  let v4 ← zero_bytes v3 v0
  return v2

def ptr (v0 : MemBuffer) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def span_ref (v0 : MemBuffer) : FeM MemSlice_u8 := do
  let v1 := v0.ptr
  let v2 := v0.len
  let v3 ← from_raw_parts_ref_u8 v1 v2
  return v3

def const_u256 : FeM Bool := do
  let v0 := false
  return v0

def payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← HEAD_SIZE_1
  return v1

def abi_payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 v0
  return v1

def max_1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  return v1

def allocation_len (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 ← max_1
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
  let v4 ← new_1 v3
  let v5 := v4
  r0 := v5
  let v7 ← ptr r0
  let v8 ← from_raw_parts v7 v2
  r1 := v8
  let v10 ← encode_Sol v0 r1
  let v11 := r0
  return v11

def abi_field_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 v0
  let v2 := v1
  let v3 ← IS_DYNAMIC
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
  let v0 ← IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← HEAD_SIZE_1
    return v4

def abi_single_root_head_size_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

def tail (v0 : AbiFrame) (v1 : UInt256) : FeM AbiFrame := do
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

def len (v0 : AbiFrame) : FeM UInt256 := do
  let v1 := v0.len
  return v1

def encode_field_Sol_u256 (v0 : UInt256) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_u256 v0
    let v5 := v4
    let v6 ← len v1
    let v8 ← len r0
    let v9 := v6 - v8
    let v10 ← store_word_1 v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← slice r0 v11 v5
    r1 := v13
    let v15 ← encode_Sol v0 r1
    let v17 ← tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← HEAD_SIZE_1
    let v21 ← slice v1 v2 v20
    r2 := v21
    let v23 ← encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_u256 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_single_root_size_u256 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← new_1 v3
  let v5 := v4
  r0 := v5
  let v7 ← ptr r0
  let v8 ← from_raw_parts v7 v2
  let v9 := v8
  r1 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_u256
  let v13 ← tail r1 v11
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

def return_bytes (v1 : MemSlice_u8) : FeM Empty := do
  let v2 ← return_data v1
  nomatch v2

def return_value_Sol_u256 (v1 : UInt256) : FeM Empty := do
  let mut r0 : FixedMemBuffer_32 := default
  let mut r1 : FixedMemBuffer_64 := default
  let mut r2 : MemBuffer := default
  let mut r3 : MemBuffer := default
  let mut r4 : AbiFrame := default
  let mut r5 : AbiFrame := default
  let mut r6 : AbiFrame := default
  let v2 ← IS_DYNAMIC
  let v3 := !v2
  if v3 then
    let v4 ← HEAD_SIZE_1
    let v5 := (UInt256.ofNat 32)
    let v6 := decide (v4.toNat ≤ v5.toNat)
    if v6 then
      let v7 ← alloc_32
      let v8 := v7
      r0 := v8
      let v10 ← ptr_32 r0
      let v11 ← HEAD_SIZE_1
      let v12 ← from_raw_parts v10 v11
      r4 := v12
      let v14 ← encode_Sol v1 r4
      let v16 ← ptr_32 r0
      let v17 ← HEAD_SIZE_1
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← from_raw_parts_ref_u8 v16 v18
      let v22 ← return_data v19
      nomatch v22
    else
      let v23 ← HEAD_SIZE_1
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← alloc_64
        let v27 := v26
        r1 := v27
        let v29 ← ptr_64 r1
        let v30 ← HEAD_SIZE_1
        let v31 ← from_raw_parts v29 v30
        r5 := v31
        let v33 ← encode_Sol v1 r5
        let v35 ← ptr_64 r1
        let v36 ← HEAD_SIZE_1
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← from_raw_parts_ref_u8 v35 v37
        let v41 ← return_data v38
        nomatch v41
      else
        let v42 ← HEAD_SIZE_1
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← new_1 v43
        let v45 := v44
        r2 := v45
        let v47 ← ptr r2
        let v48 ← HEAD_SIZE_1
        let v49 ← from_raw_parts v47 v48
        r6 := v49
        let v51 ← encode_Sol v1 r6
        let v53 ← span_ref r2
        let v56 ← return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_u256 v1
    let v58 := v57
    r3 := v58
    let v60 ← span_ref r3
    let v63 ← return_bytes v60
    nomatch v63

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get0
  let result ← recv0_arm0
  return_value_Sol_u256 result

def decode_from_CallData_1 (v0 : CallData) (v1 : UInt256) : FeM Get1 := do
  let v2 := (Get1.mk)
  return v2

def decode_from_prechecked_head_Get1_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Get1 := do
  let v3 ← decode_from_CallData_1 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Get1 : FeM Get1 := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Get1_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Get1
  let result ← recv0_arm1
  return_value_Sol_u256 result

def HEAD_SIZE_2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

def word_at (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0.base
  let v3 := v1
  let v4 := v2 + v3
  let v5 ← calldataload v4
  return v5

def decode_error : FeM Empty := do
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
    let v9 ← decode_error
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
    let v16 ← decode_error
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

def decode_from_Sol_CallData (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← word_at v0 v1
  return v2

def decode_from_bounded_u256_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← HEAD_SIZE_1
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← decode_from_Sol_CallData v0 v1
  return v5

def decode_field_from_prechecked_head_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← IS_DYNAMIC
  if v4 then
    let v5 ← word_at v0 v2
    let v6 := v5
    let v7 ← checked_tail_Sol v1 v6
    let v8 := v7
    let v9 ← decode_from_bounded_u256_Sol_CallData v0 v8 v3
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← decode_from_Sol_CallData v0 v2
    return v12

def decode_field_from_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← IS_DYNAMIC
  if v3 then
    let v4 ← word_at v0 v2
    let v5 := v4
    let v6 ← checked_tail_Sol v1 v5
    let v7 ← decode_from_Sol_CallData v0 v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← decode_from_Sol_CallData v0 v2
    return v10

def decode_msg_field_from_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_u256_CallData v0 v1 v2 v3
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_u256_CallData v0 v1 v2
    return v8

def decode_from_CallData_2 (v0 : CallData) (v1 : UInt256) : FeM Set0 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Set0.mk v5)
  return v6

def decode_from_prechecked_head_Set0_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Set0 := do
  let v3 ← decode_from_CallData_2 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Set0 : FeM Set0 := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_2
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Set0_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set0
  let _ ← recv0_arm2 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def decode_from_CallData_3 (v0 : CallData) (v1 : UInt256) : FeM Set1 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Set1.mk v5)
  return v6

def decode_from_prechecked_head_Set1_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Set1 := do
  let v3 ← decode_from_CallData_3 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Set1 : FeM Set1 := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_2
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Set1_Sol_CallData v11 v5 v8
  return v12

def recv0_arm3_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Set1
  let _ ← recv0_arm3 message.value
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def HEAD_SIZE_3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

def decode_from_CallData_4 (v0 : CallData) (v1 : UInt256) : FeM SetBoth := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (SetBoth.mk v5 v10)
  return v11

def decode_from_prechecked_head_SetBoth_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM SetBoth := do
  let v3 ← decode_from_CallData_4 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_SetBoth : FeM SetBoth := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_3
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_SetBoth_Sol_CallData v11 v5 v8
  return v12

def recv0_arm4_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_SetBoth
  let _ ← recv0_arm4 message.first message.second
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x054c1a75 then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0x90bb7b44 then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0x92cf1ef5 then
      recv0_arm4_entry
    else if selector = UInt256.ofNat 0x9bc39d3c then
      recv0_arm3_entry
    else if selector = UInt256.ofNat 0xf26feec9 then
      recv0_arm2_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM Unit := do
  nonPayable
  init


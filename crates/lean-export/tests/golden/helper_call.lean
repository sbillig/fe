def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Store : Layout :=
  { storage := [0, 1] }

structure Location where
  slot : UInt256
  offset : UInt256
deriving Inhabited

structure MemBuffer where
  ptr : UInt256
  len : UInt256
  capacity : UInt256
deriving Inhabited

structure MemSlice_u8 where
  ptr : UInt256
  len : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def encoded_len (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 32)
  return v1

def to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def IS_SIGNED : FeM Bool := do
  let v0 := false
  return v0

def MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  let v2 := v1
  return v2

def from_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def I256_SIGN_BIT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 57896044618658097711785492504343953926634992332820282019728792003956564819968)
  return v0

def SIGN_BIT : FeM UInt256 := do
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

def downcast_saturate_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let v8 ← to_word v0
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
      let v26 ← MASK
      let v27 := decide (v26.toNat < v25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 v2 => do
        let mut r0 := r0
        let v35 := v2
        b3 r0 v35
      if v27 then
        let v28 ← MASK
        let v29 ← from_word v28
        let v30 := v29
        b10 r0 v30
      else
        let v31 := r0
        let v32 := v31
        let v33 ← from_word v32
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
    let v37 ← MASK
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
        let v53 ← from_word v40
        let v54 := v53
        b16 r0 v54
      else
        let v55 := UInt256.sgtBool v51 v48
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun r0 v5 => do
          let mut r0 := r0
          let v60 := v5
          b16 r0 v60
        if v55 then
          let v56 ← from_word v44
          let v57 := v56
          b19 r0 v57
        else
          let v58 ← from_word v9
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
        let v66 ← from_word v44
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
          let v71 ← from_word v44
          let v72 := v71
          b25 r0 v72
        else
          let v73 ← from_word v9
          let v74 := v73
          b25 r0 v74

def max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def to_word_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

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
      let v26 ← MASK
      let v27 := decide (v26.toNat < v25.toNat)
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 v2 => do
        let mut r0 := r0
        let v35 := v2
        b3 r0 v35
      if v27 then
        let v28 ← MASK
        let v29 ← from_word v28
        let v30 := v29
        b10 r0 v30
      else
        let v31 := r0
        let v32 := v31
        let v33 ← from_word v32
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
    let v37 ← MASK
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
        let v53 ← from_word v40
        let v54 := v53
        b16 r0 v54
      else
        let v55 := UInt256.sgtBool v51 v48
        let b19 : UInt256 → UInt256 → FeM UInt256 := fun r0 v5 => do
          let mut r0 := r0
          let v60 := v5
          b16 r0 v60
        if v55 then
          let v56 ← from_word v44
          let v57 := v56
          b19 r0 v57
        else
          let v58 ← from_word v9
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
        let v66 ← from_word v44
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
          let v71 ← from_word v44
          let v72 := v71
          b25 r0 v72
        else
          let v73 ← from_word v9
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

def new_uninit (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes v0
  let v2 := v0
  let v3 := v0
  let v4 := (MemBuffer.mk v1 v2 v3)
  return v4

def ptr (v0 : MemBuffer) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def write_key (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 ← mstore v0 v1
  return ()

def downcast_unchecked_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let v1 ← to_word v0
  let v2 := v1
  let v3 ← from_word v2
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

def keccak256_1 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← ptr_u8 v0
  let v2 ← len_u8 v0
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

def drop (v0 : MemBuffer) : FeM Unit := do
  let v1 := v0.ptr
  let v2 ← dealloc_u8 v1
  return ()

def mapping_slot_u256 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : MemBuffer := default
  let v2 ← encoded_len v1
  let v3 := v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedAdd v3 v4
  let v6 ← downcast_saturate_u256_usize v5
  let v7 ← new_uninit v6
  let v8 := v7
  r0 := v8
  let v10 ← ptr r0
  let v11 := v10
  let v12 ← write_key v11 v1
  let v13 ← downcast_unchecked_u256_usize v3
  let v14 := v11 + v13
  let v15 ← mstore v14 v0
  let v16 := (UInt256.ofNat 32)
  let v17 ← checkedAdd v3 v16
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← from_raw_parts_ref_u8 v11 v18
  let v21 ← keccak256_1 v19
  let _ ← drop r0
  return v21

def locate_u256_u256 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Location := do
  let v3 ← mapping_slot_u256 v1 v2
  let v4 := (UInt256.ofNat 0)
  let v5 := (Location.mk v3 v4)
  return v5

def add_Store (v0 : UInt256) : FeM UInt256 := do
  let v1 ← load (Field.store.field Layout.Store 0)
  let v2 := v0
  let v3 ← checkedAdd v1 v2
  let v4 := v3
  store (Field.store.field Layout.Store 0) v4
  let v5 := ()
  let v6 := (UInt256.ofNat 1)
  let v7 := v0
  let l0 ← locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot v7
  let v8 ← load (Place.mk .storage l0.slot)
  let v9 ← checkedAdd v8 v6
  let l1 ← locate_u256_u256 (Field.store.field Layout.Store 1) (Field.store.field Layout.Store 1).slot v7
  store (Place.mk .storage l1.slot) v9
  let v10 := ()
  return v4

def recv0_arm0 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← add_Store v0
  return v1


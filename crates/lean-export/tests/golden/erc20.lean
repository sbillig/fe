def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.TokenStore : Layout :=
  { storage := [0, 1, 2] }

structure Transfer where
  «from» : UInt256
  «to» : UInt256
  value : UInt256
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

structure Approval where
  owner : UInt256
  spender : UInt256
  value : UInt256
deriving Inhabited

structure CallData where
  base : UInt256
deriving Inhabited

structure Transfer_1 where
  «to» : UInt256
  value : UInt256
deriving Inhabited

structure FixedMemBuffer_32 where
  ptr : UInt256
deriving Inhabited

structure FixedMemBuffer_64 where
  ptr : UInt256
deriving Inhabited

structure Approve where
  spender : UInt256
  value : UInt256
deriving Inhabited

structure TransferFrom where
  «from» : UInt256
  «to» : UInt256
  value : UInt256
deriving Inhabited

structure BalanceOf where
  owner : UInt256
deriving Inhabited

structure Allowance where
  owner : UInt256
  spender : UInt256
deriving Inhabited

structure TotalSupply where
deriving Inhabited

structure SolDecoder where
  pos : UInt256
  base : UInt256
deriving Inhabited

def caller_1 : FeM UInt256 := do
  let v1 ← caller
  let v2 := v1
  return v2

def zero : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := v0
  return v1

def IS_DYNAMIC : FeM Bool := do
  let v0 := false
  return v0

def HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def abi_field_head_size_u256 : FeM UInt256 := do
  let v0 ← IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← HEAD_SIZE
    return v4

def HEAD_SIZE_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

def payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← HEAD_SIZE
  return v1

def dynamic_payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_u256 v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 := (UInt256.ofNat 0)
    return v5

def payload_size_u256_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← HEAD_SIZE_u256
  let v2 := v0
  let v3 ← dynamic_payload_size_u256 v2
  let v4 := v1 + v3
  return v4

def abi_payload_size_u256 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256_1 v0
  return v1

def max : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := ~~~v0
  return v1

def PANIC_OUT_OF_MEMORY : FeM UInt256 := do
  let v0 := (UInt256.ofNat 65)
  return v0

def word_to_usize (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def allocation_len (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 ← max
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

def max_1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def to_word (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := v1
  return v2

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

def downcast_saturate_u64_usize (v0 : UInt256) : FeM UInt256 := do
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

def MAX_ALLOCATION : FeM UInt256 := do
  let v0 ← max_1
  let v1 ← downcast_saturate_u64_usize v0
  return v1

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

def zero_bytes (v0 : UInt256) (v1 : UInt256) : FeM Unit := do
  let v2 ← zeroMem v0 v1
  return ()

def new (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← new_uninit v0
  let v2 := v1
  let v3 := v2.ptr
  let v4 ← zero_bytes v3 v0
  return v2

def ptr (v0 : MemBuffer) : FeM UInt256 := do
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

def ptr_offset (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize v1
  let v3 := v0 + v2
  return v3

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
    let v20 ← HEAD_SIZE
    let v21 ← slice v1 v2 v20
    r2 := v21
    let v23 ← encode_Sol v0 r2
    let v24 := r0
    return v24

def encode_Sol_u256 (v0 : UInt256) (v1 : AbiFrame) : FeM Unit := do
  let v3 := (UInt256.ofNat 0)
  let v4 ← HEAD_SIZE_u256
  let v5 ← tail v1 v4
  let v6 := v0
  let v7 ← encode_field_Sol_u256 v6 v1 v3 v5
  return ()

def encode_alloc_Sol_u256 (r0 : UInt256) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : MemBuffer := default
  let mut r2 : AbiFrame := default
  let v1 ← abi_payload_size_u256 r0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← new v3
  let v5 := v4
  r1 := v5
  let v7 ← ptr r1
  let v8 ← from_raw_parts v7 v2
  r2 := v8
  let v11 ← encode_Sol_u256 r0 r2
  let v12 := r1
  return v12

def encode_abi_payload_u256 (v0 : UInt256) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_u256 v0
  return v1

def from_raw_parts_ref_u8 (v0 : UInt256) (v1 : UInt256) : FeM MemSlice_u8 := do
  let mut r0 : MemSlice_u8 := default
  let v2 := (MemSlice_u8.mk v0 v1)
  let v3 := v2
  r0 := v3
  return r0

def span_ref (v0 : MemBuffer) : FeM MemSlice_u8 := do
  let v1 := v0.ptr
  let v2 := v0.len
  let v3 ← from_raw_parts_ref_u8 v1 v2
  return v3

def as_topic (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def ptr_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := v0.ptr
  return v1

def len_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 := v0.len
  return v1

def log3 (v1 : MemSlice_u8) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) : FeM Unit := do
  let v5 ← ptr_u8 v1
  let v6 ← len_u8 v1
  let v7 := v6
  let v8 ← logData v5 v7 #[v2, v3, v4]
  return ()

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

def emit_Evm (v0 : Transfer) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := v0.value
  let v3 := v2
  let v4 ← encode_abi_payload_u256 v3
  r0 := v4
  let v6 ← span_ref r0
  let v7 := (UInt256.ofNat 100389287136786176327247604509743168900146139575972864366142685224231313322991) -- a constant, as the compiler evaluates it
  let v8 := v0.«from»
  r1 := v8
  let v9 := r1
  let v10 ← as_topic v9
  let v11 := v0.«to»
  r2 := v11
  let v12 := r2
  let v13 ← as_topic v12
  let v15 ← log3 v6 v7 v10 v13
  let _ ← drop r0
  return ()

def emit_Evm_Transfer (v1 : Transfer) : FeM Unit := do
  let v3 := v1
  let v4 ← emit_Evm v3
  return ()

def init (v0 : UInt256) : FeM Unit := do
  let v2 ← caller_1
  let v3 := v2
  let v4 := v0
  store ((Field.store.field Layout.TokenStore 0).entry v3) v4
  let v5 := ()
  let v6 := v0
  store (Field.store.field Layout.TokenStore 2) v6
  let v7 := ()
  let v8 ← zero
  let v9 := v0
  let v10 := (Transfer.mk v8 v3 v9)
  let v12 ← emit_Evm_Transfer v10
  return ()

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← caller_1
  let v4 := v3
  let v5 ← load ((Field.store.field Layout.TokenStore 0).entry v4)
  let v6 := decide (v1.toNat ≤ v5.toNat)
  if v6 then
    let v7 := ()
    let v8 ← load ((Field.store.field Layout.TokenStore 0).entry v4)
    let v9 ← checkedSub v8 v1
    store ((Field.store.field Layout.TokenStore 0).entry v4) v9
    let v10 := ()
    let v11 ← load ((Field.store.field Layout.TokenStore 0).entry v0)
    let v12 ← checkedAdd v11 v1
    store ((Field.store.field Layout.TokenStore 0).entry v0) v12
    let v13 := ()
    let v14 := (Transfer.mk v4 v0 v1)
    let v16 ← emit_Evm_Transfer v14
    let v17 := true
    return v17
  else
    M.revert (assertData "ERC20: insufficient balance")

def emit_Evm_1 (v0 : Approval) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := v0.value
  let v3 := v2
  let v4 ← encode_abi_payload_u256 v3
  r0 := v4
  let v6 ← span_ref r0
  let v7 := (UInt256.ofNat 63486140976153616755203102783360879283472101686154884697241723088393386309925) -- a constant, as the compiler evaluates it
  let v8 := v0.owner
  r1 := v8
  let v9 := r1
  let v10 ← as_topic v9
  let v11 := v0.spender
  r2 := v11
  let v12 := r2
  let v13 ← as_topic v12
  let v15 ← log3 v6 v7 v10 v13
  let _ ← drop r0
  return ()

def emit_Evm_Approval (v1 : Approval) : FeM Unit := do
  let v3 := v1
  let v4 ← emit_Evm_1 v3
  return ()

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← caller_1
  let v4 := v3
  store (((Field.store.field Layout.TokenStore 1).entry v4).entry v0) v1
  let v5 := ()
  let v6 := (Approval.mk v4 v0 v1)
  let v8 ← emit_Evm_Approval v6
  let v9 := true
  return v9

def recv0_arm2 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Bool := do
  let v4 ← caller_1
  let v5 := v4
  let v6 ← load (((Field.store.field Layout.TokenStore 1).entry v0).entry v5)
  let v7 := v6
  let v8 := decide (v2.toNat ≤ v7.toNat)
  if v8 then
    let v9 := ()
    let v10 ← load ((Field.store.field Layout.TokenStore 0).entry v0)
    let v11 := decide (v2.toNat ≤ v10.toNat)
    if v11 then
      let v12 := ()
      let v13 ← checkedSub v7 v2
      store (((Field.store.field Layout.TokenStore 1).entry v0).entry v5) v13
      let v14 := ()
      let v15 ← load ((Field.store.field Layout.TokenStore 0).entry v0)
      let v16 ← checkedSub v15 v2
      store ((Field.store.field Layout.TokenStore 0).entry v0) v16
      let v17 := ()
      let v18 ← load ((Field.store.field Layout.TokenStore 0).entry v1)
      let v19 ← checkedAdd v18 v2
      store ((Field.store.field Layout.TokenStore 0).entry v1) v19
      let v20 := ()
      let v21 := (Transfer.mk v0 v1 v2)
      let v23 ← emit_Evm_Transfer v21
      let v24 := true
      return v24
    else
      M.revert (assertData "ERC20: insufficient balance")
  else
    M.revert (assertData "ERC20: insufficient allowance")

def recv0_arm3 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← load ((Field.store.field Layout.TokenStore 0).entry v0)
  return v1

def recv0_arm4 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← load (((Field.store.field Layout.TokenStore 1).entry v0).entry v1)
  return v2

def recv0_arm5 : FeM UInt256 := do
  let v0 ← load (Field.store.field Layout.TokenStore 2)
  return v0

def new_1 : FeM CallData := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (CallData.mk v0)
  return v1

def input : FeM CallData := do
  let v1 ← new_1
  return v1

def SELECTOR_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4)
  return v0

def to_word_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  return v1

def U64_MASK : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def MASK_1 : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

def from_word_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := truncate 64 v0
  return v1

def downcast_truncate_u256_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← to_word_1 v0
  let v3 := v2
  let v4 ← IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← MASK_1
    let v6 ← SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← from_word_1 v7
    let v9 := v8
    b3 v9
  else
    let v10 ← MASK_1
    let v11 := v3 &&& v10
    let v12 ← from_word_1 v11
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

def abi_field_head_size_Address : FeM UInt256 := do
  let v0 ← IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← HEAD_SIZE
    return v4

def HEAD_SIZE_1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

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
    let v9 ← from_word v8
    let v10 := v9
    let v11 ← from_word v2
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
  let v3 ← HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← decode_from_Sol_CallData v0 v1
  return v5

def decode_field_from_prechecked_head_Sol_Address_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← IS_DYNAMIC
  if v4 then
    let v5 ← word_at v0 v2
    let v6 := v5
    let v7 ← checked_tail_Sol v1 v6
    let v8 := v7
    let v9 ← decode_from_bounded_Address_Sol_CallData v0 v8 v3
    return v9
  else
    let v10 := ()
    let v11 := v10
    let v12 ← decode_from_Sol_CallData v0 v2
    return v12

def decode_field_from_Sol_Address_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
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

def decode_msg_field_from_Sol_Address_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v4 ← IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_Address_CallData v0 v1 v2 v3
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 ← decode_field_from_Sol_Address_CallData v0 v1 v2
    return v8

def decode_from_Sol_CallData_1 (v0 : CallData) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← word_at v0 v1
  return v2

def decode_from_bounded_u256_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← HEAD_SIZE
  let v4 ← checked_frame_end_Sol v1 v3 v2
  let v5 ← decode_from_Sol_CallData_1 v0 v1
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
    let v12 ← decode_from_Sol_CallData_1 v0 v2
    return v12

def decode_field_from_Sol_u256_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let v3 ← IS_DYNAMIC
  if v3 then
    let v4 ← word_at v0 v2
    let v5 := v4
    let v6 ← checked_tail_Sol v1 v5
    let v7 ← decode_from_Sol_CallData_1 v0 v6
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← decode_from_Sol_CallData_1 v0 v2
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

def decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Transfer_1 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Transfer_1.mk v5 v10)
  return v11

def decode_from_prechecked_head_Transfer_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Transfer_1 := do
  let v3 ← decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_1
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData v11 v5 v8
  return v12

def alloc_uninit_32 : FeM FixedMemBuffer_32 := do
  let v0 := (UInt256.ofNat 32)
  let v1 ← alloc_bytes v0
  let v2 := (FixedMemBuffer_32.mk v1)
  return v2

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

def encode_Sol_1 (v0 : Bool) (v1 : AbiFrame) : FeM Unit := do
  let v2 := (UInt256.ofNat 0)
  let v3 := v0
  let v4 := Bool.toUInt256 v3
  let v5 ← store_word_1 v1 v2 v4
  return ()

def downcast_unchecked_u256_usize (v0 : UInt256) : FeM UInt256 := do
  let v1 ← to_word_1 v0
  let v2 := v1
  let v3 ← from_word v2
  return v3

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

def const_bool : FeM Bool := do
  let v0 := false
  return v0

def payload_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← HEAD_SIZE
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
  let v4 ← new v3
  let v5 := v4
  r0 := v5
  let v7 ← ptr r0
  let v8 ← from_raw_parts v7 v2
  r1 := v8
  let v10 ← encode_Sol_1 v0 r1
  let v11 := r0
  return v11

def abi_field_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← payload_size_bool v0
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

def abi_single_root_size_bool (v0 : Bool) : FeM UInt256 := do
  let v1 ← abi_field_size_bool v0
  return v1

def abi_field_head_size_bool : FeM UInt256 := do
  let v0 ← IS_DYNAMIC
  if v0 then
    let v1 := (UInt256.ofNat 32)
    return v1
  else
    let v2 := ()
    let v3 := v2
    let v4 ← HEAD_SIZE
    return v4

def abi_single_root_head_size_bool : FeM UInt256 := do
  let v0 ← abi_field_head_size_bool
  return v0

def encode_field_Sol_bool (v0 : Bool) (v1 : AbiFrame) (v2 : UInt256) (r0 : AbiFrame) : FeM AbiFrame := do
  let mut r0 := r0
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← IS_DYNAMIC
  if v3 then
    let v4 ← payload_size_bool v0
    let v5 := v4
    let v6 ← len v1
    let v8 ← len r0
    let v9 := v6 - v8
    let v10 ← store_word_1 v1 v2 v9
    let v11 := (UInt256.ofNat 0)
    let v13 ← slice r0 v11 v5
    r1 := v13
    let v15 ← encode_Sol_1 v0 r1
    let v17 ← tail r0 v5
    return v17
  else
    let v18 := ()
    let v19 := v18
    let v20 ← HEAD_SIZE
    let v21 ← slice v1 v2 v20
    r2 := v21
    let v23 ← encode_Sol_1 v0 r2
    let v24 := r0
    return v24

def encode_single_root_alloc_Sol_bool (v0 : Bool) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_single_root_size_bool v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← new v3
  let v5 := v4
  r0 := v5
  let v7 ← ptr r0
  let v8 ← from_raw_parts v7 v2
  let v9 := v8
  r1 := v9
  let v10 := (UInt256.ofNat 0)
  let v11 ← abi_single_root_head_size_bool
  let v13 ← tail r1 v11
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

def return_bytes (v1 : MemSlice_u8) : FeM Empty := do
  let v2 ← return_data v1
  nomatch v2

def return_value_Sol_bool (v1 : Bool) : FeM Empty := do
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
    let v4 ← HEAD_SIZE
    let v5 := (UInt256.ofNat 32)
    let v6 := decide (v4.toNat ≤ v5.toNat)
    if v6 then
      let v7 ← alloc_32
      let v8 := v7
      r0 := v8
      let v10 ← ptr_32 r0
      let v11 ← HEAD_SIZE
      let v12 ← from_raw_parts v10 v11
      r4 := v12
      let v14 ← encode_Sol_1 v1 r4
      let v16 ← ptr_32 r0
      let v17 ← HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← from_raw_parts_ref_u8 v16 v18
      let v22 ← return_data v19
      nomatch v22
    else
      let v23 ← HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← alloc_64
        let v27 := v26
        r1 := v27
        let v29 ← ptr_64 r1
        let v30 ← HEAD_SIZE
        let v31 ← from_raw_parts v29 v30
        r5 := v31
        let v33 ← encode_Sol_1 v1 r5
        let v35 ← ptr_64 r1
        let v36 ← HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← from_raw_parts_ref_u8 v35 v37
        let v41 ← return_data v38
        nomatch v41
      else
        let v42 ← HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← new v43
        let v45 := v44
        r2 := v45
        let v47 ← ptr r2
        let v48 ← HEAD_SIZE
        let v49 ← from_raw_parts v47 v48
        r6 := v49
        let v51 ← encode_Sol_1 v1 r6
        let v53 ← span_ref r2
        let v56 ← return_data v53
        nomatch v56
  else
    let v57 ← encode_return_alloc_Sol_bool v1
    let v58 := v57
    r3 := v58
    let v60 ← span_ref r3
    let v63 ← return_bytes v60
    nomatch v63

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv0_arm0 message.«to» message.value
  return_value_Sol_bool result

def decode_from_CallData_1 (v0 : CallData) (v1 : UInt256) : FeM Approve := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Approve.mk v5 v10)
  return v11

def decode_from_prechecked_head_Approve_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Approve := do
  let v3 ← decode_from_CallData_1 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_1
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv0_arm1 message.spender message.value
  return_value_Sol_bool result

def HEAD_SIZE_2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  return v6

def decode_from_CallData_2 (v0 : CallData) (v1 : UInt256) : FeM TransferFrom := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := v1
  let v12 ← abi_field_head_size_Address
  let v13 ← checkedAdd v11 v12
  let v14 ← abi_field_head_size_Address
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v3
  let v17 := v16
  let v18 := (TransferFrom.mk v5 v10 v17)
  return v18

def decode_from_prechecked_head_TransferFrom_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM TransferFrom := do
  let v3 ← decode_from_CallData_2 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
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
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv0_arm2 message.«from» message.«to» message.value
  return_value_Sol_bool result

def HEAD_SIZE_3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  return v2

def decode_from_CallData_3 (v0 : CallData) (v1 : UInt256) : FeM BalanceOf := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (BalanceOf.mk v5)
  return v6

def decode_from_prechecked_head_BalanceOf_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM BalanceOf := do
  let v3 ← decode_from_CallData_3 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
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
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData v11 v5 v8
  return v12

def const_u256 : FeM Bool := do
  let v0 := false
  return v0

def abi_payload_size_u256_1 (v0 : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 v0
  return v1

def encode_alloc_Sol_u256_1 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_payload_size_u256_1 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← new v3
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

def abi_single_root_head_size_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

def encode_single_root_alloc_Sol_u256 (v0 : UInt256) : FeM MemBuffer := do
  let mut r0 : MemBuffer := default
  let mut r1 : AbiFrame := default
  let v1 ← abi_single_root_size_u256 v0
  let v2 := v1
  let v3 ← allocation_len v2
  let v4 ← new v3
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
    let v2 ← encode_alloc_Sol_u256_1 v0
    return v2
  else
    let v3 := ()
    let v4 := v3
    let v5 ← encode_single_root_alloc_Sol_u256 v0
    return v5

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
    let v4 ← HEAD_SIZE
    let v5 := (UInt256.ofNat 32)
    let v6 := decide (v4.toNat ≤ v5.toNat)
    if v6 then
      let v7 ← alloc_32
      let v8 := v7
      r0 := v8
      let v10 ← ptr_32 r0
      let v11 ← HEAD_SIZE
      let v12 ← from_raw_parts v10 v11
      r4 := v12
      let v14 ← encode_Sol v1 r4
      let v16 ← ptr_32 r0
      let v17 ← HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← from_raw_parts_ref_u8 v16 v18
      let v22 ← return_data v19
      nomatch v22
    else
      let v23 ← HEAD_SIZE
      let v24 := (UInt256.ofNat 64)
      let v25 := decide (v23.toNat ≤ v24.toNat)
      if v25 then
        let v26 ← alloc_64
        let v27 := v26
        r1 := v27
        let v29 ← ptr_64 r1
        let v30 ← HEAD_SIZE
        let v31 ← from_raw_parts v29 v30
        r5 := v31
        let v33 ← encode_Sol v1 r5
        let v35 ← ptr_64 r1
        let v36 ← HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← from_raw_parts_ref_u8 v35 v37
        let v41 ← return_data v38
        nomatch v41
      else
        let v42 ← HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let v44 ← new v43
        let v45 := v44
        r2 := v45
        let v47 ← ptr r2
        let v48 ← HEAD_SIZE
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

def recv0_arm3_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv0_arm3 message.owner
  return_value_Sol_u256 result

def HEAD_SIZE_4 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Address
  let v4 ← checkedAdd v2 v3
  return v4

def decode_from_CallData_4 (v0 : CallData) (v1 : UInt256) : FeM Allowance := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Allowance.mk v5 v10)
  return v11

def decode_from_prechecked_head_Allowance_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Allowance := do
  let v3 ← decode_from_CallData_4 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_4
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData v11 v5 v8
  return v12

def recv0_arm4_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv0_arm4 message.owner message.spender
  return_value_Sol_u256 result

def HEAD_SIZE_5 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def decode_from_CallData_5 (v0 : CallData) (v1 : UInt256) : FeM TotalSupply := do
  let v2 := (TotalSupply.mk)
  return v2

def decode_from_prechecked_head_TotalSupply_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM TotalSupply := do
  let v3 ← decode_from_CallData_5 v0 v1
  return v3

def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut r0 : CallData := default
  let v2 ← input
  let v3 := v2
  r0 := v3
  let v4 ← SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← HEAD_SIZE_5
  let v10 ← validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData v11 v5 v8
  return v12

def recv0_arm5_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv0_arm5
  return_value_Sol_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x095ea7b3 then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0x18160ddd then
      recv0_arm5_entry
    else if selector = UInt256.ofNat 0x23b872dd then
      recv0_arm2_entry
    else if selector = UInt256.ofNat 0x70a08231 then
      recv0_arm3_entry
    else if selector = UInt256.ofNat 0xa9059cbb then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0xdd62ed3e then
      recv0_arm4_entry
    else
      revertEmpty

def with_base (v0 : UInt256) : FeM SolDecoder := do
  let v1 := v0
  let v2 := v0
  let v3 := (SolDecoder.mk v1 v2)
  return v3

def decoder_new : FeM SolDecoder := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← with_base v0
  return v1

def downcast_truncate_usize_u64 (v0 : UInt256) : FeM UInt256 := do
  let v2 ← to_word v0
  let v3 := v2
  let v4 ← IS_SIGNED
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← MASK_1
    let v6 ← SIGN_BIT
    let v7 ← sign_extend v3 v5 v6
    let v8 ← from_word_1 v7
    let v9 := v8
    b3 v9
  else
    let v10 ← MASK_1
    let v11 := v3 &&& v10
    let v12 ← from_word_1 v11
    let v13 := v12
    b3 v13

def size_1 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← len_u8 v0
  let v2 ← downcast_truncate_usize_u64 v1
  return v2

def len_MemSlice_u8 (v0 : MemSlice_u8) : FeM UInt256 := do
  let v1 ← size_1 v0
  let v2 := v1
  return v2

def word_at_1 (v0 : MemSlice_u8) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← len_u8 v0
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
    let v14 ← ptr_u8 v0
    let v15 ← word_to_usize v1
    let v16 := v14 + v15
    let v17 ← readPaddedWord v16 v13
    return v17

def read_word_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
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
      let v13 ← word_at_1 v1 v12
      let v14 := v13
      v0 := { v0 with pos := v5 }
      let v15 := ()
      return (v14, v0)

def base (v0 : SolDecoder) : FeM UInt256 := do
  let v1 := v0.base
  return v1

def pos (v0 : SolDecoder) : FeM UInt256 := do
  let v1 := v0.pos
  return v1

def set_base (v0 : SolDecoder) (v1 : UInt256) : FeM (Unit × SolDecoder) := do
  let mut v0 := v0
  let v2 := v1
  v0 := { v0 with base := v2 }
  let v3 := ()
  return ((), v0)

def set_pos (v0 : SolDecoder) (v1 : UInt256) : FeM (Unit × SolDecoder) := do
  let mut v0 := v0
  let v2 := v1
  v0 := { v0 with pos := v2 }
  let v3 := ()
  return ((), v0)

def decode_payload_Sol_SolDecoder_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let (v2, v2_0) ← read_word_MemSlice_u8 v0 v1
  v0 := v2_0
  return (v2, v0)

def decode_field_Sol_u256_SolDecoder_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let v2 ← IS_DYNAMIC
  if v2 then
    let (v3, v3_0) ← read_word_MemSlice_u8 v0 v1
    v0 := v3_0
    let v4 := v3
    let v6 ← base v0
    let v7 := v6
    let v9 ← pos v0
    let v10 := v9
    let v11 ← checked_tail_Sol v7 v4
    let (v12, v12_0) ← set_base v0 v11
    v0 := v12_0
    let v14 ← base v0
    let (v15, v15_0) ← set_pos v0 v14
    v0 := v15_0
    let (v17, v17_0) ← decode_payload_Sol_SolDecoder_MemSlice_u8 v0 v1
    v0 := v17_0
    let v18 := v17
    let (v19, v19_0) ← set_base v0 v7
    v0 := v19_0
    let (v20, v20_0) ← set_pos v0 v10
    v0 := v20_0
    return (v18, v0)
  else
    let v21 := ()
    let v22 := v21
    let (v24, v24_0) ← decode_payload_Sol_SolDecoder_MemSlice_u8 v0 v1
    v0 := v24_0
    return (v24, v0)

def decode_payload_Sol_u256_SolDecoder_MemSlice_u8 (v0 : SolDecoder) (v1 : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut v0 := v0
  let (v3, v3_0) ← decode_field_Sol_u256_SolDecoder_MemSlice_u8 v0 v1
  v0 := v3_0
  let v4 := v3
  let v5 := v4
  return (v5, v0)

def init_entry (args : ByteArray) : FeM Unit := do
  nonPayable
  let size := UInt256.ofNat args.size
  let tail ← alloc size
  mstoreBytes tail args
  let decoder ← decoder_new
  let (decoded, _) ← decode_payload_Sol_u256_SolDecoder_MemSlice_u8 decoder (MemSlice_u8.mk tail size)
  init decoded


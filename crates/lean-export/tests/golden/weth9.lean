export Lib (Address_as_topic CallData DynString DynString.mk DynString_drop DynString_from_abi_tail DynString_span_ref Evm_caller Evm_input Evm_log2 Evm_log3 Evm_return_value_Sol_DynString Evm_return_value_Sol_bool Evm_return_value_Sol_u256 Evm_return_value_Sol_u8 Evm_selfbalance Evm_value Layout.SolSlot Location Location.mk MemBuffer MemBuffer_drop MemBuffer_span_ref MemSlice_len_u8 MemSlice_u8 MemSlice_u8_word_at PANIC_STORAGE_ENCODING Panic_from RawCallOutcome RawCallOutcome_success SolWord SolWord.mk SolWord_get_u8 SolWord_set_u8 Sol_SELECTOR_SIZE Sol_validate_runtime_head StorageMap_locate_Address_StorageMap_Address_u256 StorageMap_locate_Address_u256 Wei Wei_as_wei abi_field_head_size_Address abi_field_head_size_u256 alloc_bytes array_data_slot decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData downcast_saturate_u256_usize downcast_unchecked_u256_usize encode_abi_payload_tuple_u256 field_shift len_CallData ne_Address_Address revert_error_Panic send_value_Evm store_word u256_max u256_wrapping_add u8_BITS u8_to_packed wei)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Weth9Store : Layout :=
  { storage := [0, 1, 2, 3, 4] }

def Layout.SolString : Layout :=
  { storage := [0] }

def Layout.Payload : Layout :=
  { storage := [0] }

structure Approval where
  src : UInt256
  guy : UInt256
  wad : UInt256
deriving Inhabited

structure Transfer where
  src : UInt256
  dst : UInt256
  wad : UInt256
deriving Inhabited

structure Withdrawal where
  src : UInt256
  wad : UInt256
deriving Inhabited

structure Deposit where
  dst : UInt256
  wad : UInt256
deriving Inhabited

structure Name where
deriving Inhabited

structure Approve where
  guy : UInt256
  wad : UInt256
deriving Inhabited

structure TotalSupply where
deriving Inhabited

structure TransferFrom where
  src : UInt256
  dst : UInt256
  wad : UInt256
deriving Inhabited

structure Withdraw where
  wad : UInt256
deriving Inhabited

structure Decimals where
deriving Inhabited

structure BalanceOf where
  owner : UInt256
deriving Inhabited

structure Symbol where
deriving Inhabited

structure Transfer_1 where
  dst : UInt256
  wad : UInt256
deriving Inhabited

structure Deposit_1 where
deriving Inhabited

structure Allowance where
  owner : UInt256
  guy : UInt256
deriving Inhabited

def stored_len (v0 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := (UInt256.ofNat 1)
  let v4 := v2 &&& v3
  let v5 := (UInt256.ofNat 1)
  let v6 := decide (v4 = v5)
  let v7 := v6
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    let v18 := v1
    let v19 := (UInt256.ofNat 32)
    let v20 := decide (v18.toNat < v19.toNat)
    let v21 := v7 == v20
    if v21 then
      let v22 ← PANIC_STORAGE_ENCODING
      let v23 ← Panic_from v22
      let v24 ← revert_error_Panic v23
      nomatch v24
    else
      let v25 := ()
      let v26 := v25
      return v18
  if v7 then
    let v8 := v0
    let v9 := (UInt256.ofNat 2)
    let v10 ← checkedDiv v8 v9
    let v11 := v10
    b3 v11
  else
    let v12 := v0
    let v13 := (UInt256.ofNat 255)
    let v14 := v12 &&& v13
    let v15 := (UInt256.ofNat 2)
    let v16 ← checkedDiv v14 v15
    let v17 := v16
    b3 v17

def Payload_locate (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Location := do
  let v3 ← array_data_slot v1
  let v4 := v2
  let v5 ← u256_wrapping_add v3 v4
  let v6 := (UInt256.ofNat 0)
  let v7 := (Location.mk v5 v6)
  return v7

def leading_bytes (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v3 := v1
  let v4 := (UInt256.ofNat 32)
  let v5 := decide (v4.toNat ≤ v3.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  if v5 then
    let v6 := v0
    let v7 := v6
    b3 v7
  else
    let v8 := v0
    let v9 ← u256_max
    let v10 := v1
    let v11 := (UInt256.ofNat 8)
    let v12 := v10 * v11
    let v13 := v9 >>> v12
    let v14 := ~~~v13
    let v15 := v8 &&& v14
    let v16 := v15
    b3 v16

def payload_word (v0 : MemSlice_u8) (v1 : UInt256) : FeM UInt256 := do
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
    let v10 ← MemSlice_u8_word_at v0 v1
    let v11 := v1
    let v12 := v4 - v11
    let v13 ← leading_bytes v10 v12
    return v13

def word_count (v0 : UInt256) : FeM UInt256 := do
  let v1 := v0
  let v2 := (UInt256.ofNat 31)
  let v3 := v1 + v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedDiv v3 v4
  return v5

def SolString_store (v0 : Place) (v1 : DynString) : FeM Unit := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v3 ← DynString_span_ref v1
  let v5 ← load ((v0.field Layout.SolString 0).field Layout.Payload 0)
  let v6 := v5
  let v7 ← stored_len v6
  let v8 := v7
  let v9 := (UInt256.ofNat 1)
  let v10 := v6 &&& v9
  let v11 := (UInt256.ofNat 0)
  let v12 := decide (v10 = v11)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → FeM Unit := fun r0 r1 r2 v2 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v17 := v2
    let v19 ← MemSlice_len_u8 v3
    let v20 := v19
    let v21 := v20
    let v22 := (UInt256.ofNat 0)
    let v23 := v22
    r0 := v23
    let v24 := (UInt256.ofNat 32)
    let v25 := decide (v21.toNat < v24.toNat)
    let b6 : UInt256 → UInt256 → UInt256 → FeM Unit := fun r0 r1 r2 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let v63 := r0
      let v64 := v63
      r2 := v64
      let rec b10 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) : FeM Unit := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let v65 := r2
        let v66 := v65
        let v67 := decide (v66.toNat < v17.toNat)
        if v67 then
          let v68 := (UInt256.ofNat 0)
          let v69 := r2
          let l0 ← Payload_locate (v0.field Layout.SolString 0) (v0.field Layout.SolString 0).slot v69
          store (Place.mk .storage l0.slot) v68
          let v70 := ()
          let v71 := (UInt256.ofNat 1)
          let v72 := r2
          let v73 := v72 + v71
          let v74 := v73
          r2 := v74
          let v75 := ()
          b10 r0 r1 r2
        else
          let v76 := ()
          return ()
      partial_fixpoint
      b10 r0 r1 r2
    if v25 then
      let v26 := (UInt256.ofNat 0)
      let v28 ← payload_word v3 v26
      let v29 := (UInt256.ofNat 2)
      let v30 := v21 * v29
      let v31 := v28 ||| v30
      store ((v0.field Layout.SolString 0).field Layout.Payload 0) v31
      let v32 := ()
      let v33 := v32
      b6 r0 r1 r2
    else
      let v34 := (UInt256.ofNat 2)
      let v35 := v21 * v34
      let v36 := (UInt256.ofNat 1)
      let v37 := v35 + v36
      store ((v0.field Layout.SolString 0).field Layout.Payload 0) v37
      let v38 := ()
      let v39 ← word_count v21
      let v40 := v39
      r0 := v40
      let v41 := ()
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      r1 := v43
      let rec b7 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) : FeM Unit := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let v44 := r1
        let v45 := v44
        let v46 := r0
        let v47 := decide (v45.toNat < v46.toNat)
        if v47 then
          let v48 := r1
          let v49 := v48
          let v50 := (UInt256.ofNat 32)
          let v51 := v49 * v50
          let v53 ← payload_word v3 v51
          let v54 := r1
          let l1 ← Payload_locate (v0.field Layout.SolString 0) (v0.field Layout.SolString 0).slot v54
          store (Place.mk .storage l1.slot) v53
          let v55 := ()
          let v56 := (UInt256.ofNat 1)
          let v57 := r1
          let v58 := v57 + v56
          let v59 := v58
          r1 := v59
          let v60 := ()
          b7 r0 r1 r2
        else
          let v61 := ()
          let v62 := v61
          b6 r0 r1 r2
      partial_fixpoint
      b7 r0 r1 r2
  if v12 then
    let v13 := (UInt256.ofNat 0)
    let v14 := v13
    b3 r0 r1 r2 v14
  else
    let v15 ← word_count v8
    let v16 := v15
    b3 r0 r1 r2 v16

def SolSlot_write_u8 (v0 : Place) (v1 : UInt256) (v2 : UInt256) : FeM Unit := do
  let mut r0 : SolWord := default
  let v3 ← u8_BITS
  let v4 := (UInt256.ofNat 256)
  let v5 := decide (v3 = v4)
  let b3 : SolWord → FeM Unit := fun r0 => do
    let mut r0 := r0
    return ()
  if v5 then
    let v6 ← u8_BITS
    let v7 ← field_shift v1 v6
    let v8 ← u8_to_packed v2
    store ((v0.field Layout.SolSlot 0).field Layout.Payload 0) v8
    let v9 := ()
    let v10 := v9
    b3 r0
  else
    let v11 ← load ((v0.field Layout.SolSlot 0).field Layout.Payload 0)
    let v12 := (SolWord.mk v11)
    let v13 := v12
    r0 := v13
    let (v15, v15_0) ← SolWord_set_u8 r0 v1 v2
    r0 := v15_0
    let v16 := r0.word
    store ((v0.field Layout.SolSlot 0).field Layout.Payload 0) v16
    let v17 := ()
    let v18 := v17
    b3 r0

def init : FeM Unit := do
  let mut r1 : DynString := default
  let mut r2 : DynString := default
  let v0_p ← stringLiteral ⟨#[87, 114, 97, 112, 112, 101, 100, 32, 69, 116, 104, 101, 114]⟩
  let v0 := (DynString.mk v0_p (UInt256.ofNat 13) (UInt256.ofNat 64))
  r1 := v0
  let v1 := (Field.store.field Layout.Weth9Store 0)
  let v3 ← SolString_store v1 r1
  let _ ← DynString_drop r1
  let v4_p ← stringLiteral ⟨#[87, 69, 84, 72]⟩
  let v4 := (DynString.mk v4_p (UInt256.ofNat 4) (UInt256.ofNat 64))
  r2 := v4
  let v5 := (Field.store.field Layout.Weth9Store 1)
  let v7 ← SolString_store v5 r2
  let _ ← DynString_drop r2
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 18)
  let v10 := v9
  let v11 := (Field.store.field Layout.Weth9Store 2)
  let v12 ← SolSlot_write_u8 v11 v8 v10
  return ()

def SolString_load (v0 : Place) : FeM DynString := do
  let mut r0 : UInt256 := default
  let v2 ← load ((v0.field Layout.SolString 0).field Layout.Payload 0)
  let v3 := v2
  let v4 ← stored_len v3
  let v5 := v4
  let v6 := (UInt256.ofNat 32)
  let v7 ← word_count v5
  let v8 := (UInt256.ofNat 32)
  let v9 := v7 * v8
  let v10 := v6 + v9
  let v11 := v10
  let v12 ← downcast_saturate_u256_usize v11
  let v13 ← alloc_bytes v12
  let v14 := v13
  let v15 ← store_word v14 v5
  let v16 := (UInt256.ofNat 1)
  let v17 := v3 &&& v16
  let v18 := (UInt256.ofNat 0)
  let v19 := decide (v17 = v18)
  let b3 : UInt256 → FeM DynString := fun r0 => do
    let mut r0 := r0
    let v61 ← downcast_unchecked_u256_usize v5
    let v62 ← DynString_from_abi_tail v14 v61 v11
    return v62
  if v19 then
    let v20 := (UInt256.ofNat 0)
    let v21 := decide (v5 ≠ v20)
    let b6 : UInt256 → Unit → FeM DynString := fun r0 v1 => do
      let mut r0 := r0
      let v29 := v1
      b3 r0
    if v21 then
      let v22 := (UInt256.ofNat 32)
      let v23 := v14 + v22
      let v24 ← leading_bytes v3 v5
      let v25 ← store_word v23 v24
      let v26 := v25
      b6 r0 v26
    else
      let v27 := ()
      let v28 := v27
      b6 r0 v28
  else
    let v30 := (UInt256.ofNat 0)
    let v31 := v30
    r0 := v31
    let rec b7 (r0 : UInt256) : FeM DynString := do
      let mut r0 := r0
      let v32 := r0
      let v33 := v32
      let v34 ← word_count v5
      let v35 := decide (v33.toNat < v34.toNat)
      if v35 then
        let v36 := r0
        let l0 ← Payload_locate (v0.field Layout.SolString 0) (v0.field Layout.SolString 0).slot v36
        let v37 ← load (Place.mk .storage l0.slot)
        let v38 := r0
        let v39 := v38
        let v40 := (UInt256.ofNat 32)
        let v41 := v39 * v40
        let v42 := v5 - v41
        let v43 ← leading_bytes v37 v42
        let v44 := v43
        let v45 := (UInt256.ofNat 32)
        let v46 := r0
        let v47 := v46
        let v48 := (UInt256.ofNat 32)
        let v49 := v47 * v48
        let v50 := v45 + v49
        let v51 ← downcast_unchecked_u256_usize v50
        let v52 := v14 + v51
        let v53 ← store_word v52 v44
        let v54 := (UInt256.ofNat 1)
        let v55 := r0
        let v56 := v55 + v54
        let v57 := v56
        r0 := v57
        let v58 := ()
        b7 r0
      else
        let v59 := ()
        let v60 := v59
        b3 r0
    partial_fixpoint
    b7 r0

def recv0_arm0 : FeM DynString := do
  let v0 := (Field.store.field Layout.Weth9Store 0)
  let v1 := v0
  let v2 ← SolString_load v1
  return v2

def Approval_emit_Evm (v0 : Approval) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := v0.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 63486140976153616755203102783360879283472101686154884697241723088393386309925) -- a constant, as the compiler evaluates it
  let v8 := v0.src
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v11 := v0.guy
  r2 := v11
  let v12 := r2
  let v13 ← Address_as_topic v12
  let v15 ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Approval (v1 : Approval) : FeM Unit := do
  let v3 := v1
  let v4 ← Approval_emit_Evm v3
  return ()

def recv0_arm1 (v0 : UInt256) (v1 : UInt256) : FeM Bool := do
  let v3 ← Evm_caller
  let v4 := v3
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v4
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot v0
  store (Place.mk .storage l1.slot) v1
  let v5 := ()
  let v6 := (Approval.mk v4 v0 v1)
  let v8 ← emit_Evm_Approval v6
  let v9 := true
  return v9

def recv0_arm2 : FeM UInt256 := do
  let mut r1 : Wei := default
  let v1 ← Evm_selfbalance
  r1 := v1
  let v2 := r1
  let v3 ← Wei_as_wei v2
  return v3

def Transfer_emit_Evm (v0 : Transfer) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := v0.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 100389287136786176327247604509743168900146139575972864366142685224231313322991) -- a constant, as the compiler evaluates it
  let v8 := v0.src
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v11 := v0.dst
  r2 := v11
  let v12 := r2
  let v13 ← Address_as_topic v12
  let v15 ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Transfer (v1 : Transfer) : FeM Unit := do
  let v3 := v1
  let v4 ← Transfer_emit_Evm v3
  return ()

def transfer_from_Weth9Store_Evm_Evm (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM Bool := do
  let mut r3 : UInt256 := default
  let v3 := v0
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v3
  let v4 ← load (Place.mk .storage l0.slot)
  let v5 := v2
  let v6 := decide (v5.toNat ≤ v4.toNat)
  if v6 then
    let v7 := ()
    let v9 ← Evm_caller
    let v10 := v9
    r3 := v10
    let v11 := r3
    let v12 ← ne_Address_Address v0 v11
    let b6 : UInt256 → FeM Bool := fun r3 => do
      let mut r3 := r3
      let v34 := v2
      let v35 := v0
      let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v35
      let v36 ← load (Place.mk .storage l1.slot)
      let v37 := v36 - v34
      let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v35
      store (Place.mk .storage l2.slot) v37
      let v38 := ()
      let v39 := v2
      let v40 := v1
      let l3 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v40
      let v41 ← load (Place.mk .storage l3.slot)
      let v42 := v41 + v39
      let l4 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v40
      store (Place.mk .storage l4.slot) v42
      let v43 := ()
      let v44 := v0
      let v45 := v1
      let v46 := v2
      let v47 := (Transfer.mk v44 v45 v46)
      let v49 ← emit_Evm_Transfer v47
      let v50 := true
      return v50
    let b5 : UInt256 → FeM Bool := fun r3 => do
      let mut r3 := r3
      let v32 := ()
      let v33 := v32
      b6 r3
    if v12 then
      let v13 := v0
      let v14 := r3
      let l5 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v13
      let l6 ← StorageMap_locate_Address_u256 (Place.mk .storage l5.slot) (Place.mk .storage l5.slot).slot v14
      let v15 ← load (Place.mk .storage l6.slot)
      let v16 ← u256_max
      let v17 := decide (v15 ≠ v16)
      if v17 then
        let v18 := v0
        let v19 := r3
        let l7 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v18
        let l8 ← StorageMap_locate_Address_u256 (Place.mk .storage l7.slot) (Place.mk .storage l7.slot).slot v19
        let v20 ← load (Place.mk .storage l8.slot)
        let v21 := v2
        let v22 := decide (v21.toNat ≤ v20.toNat)
        if v22 then
          let v23 := ()
          let v24 := v2
          let v25 := v0
          let v26 := r3
          let l9 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v25
          let l10 ← StorageMap_locate_Address_u256 (Place.mk .storage l9.slot) (Place.mk .storage l9.slot).slot v26
          let v27 ← load (Place.mk .storage l10.slot)
          let v28 := v27 - v24
          let v29 := r3
          let l11 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v25
          let l12 ← StorageMap_locate_Address_u256 (Place.mk .storage l11.slot) (Place.mk .storage l11.slot).slot v29
          store (Place.mk .storage l12.slot) v28
          let v30 := ()
          let v31 := v30
          b6 r3
        else
          M.panic 0x01
      else
        b5 r3
    else
      b5 r3
  else
    M.panic 0x01

def recv0_arm3 (r0 : UInt256) (r1 : UInt256) (v0 : UInt256) : FeM Bool := do
  let mut r0 := r0
  let mut r1 := r1
  let v1 := r0
  let v2 := r1
  let v3 ← transfer_from_Weth9Store_Evm_Evm v1 v2 v0
  return v3

def Withdrawal_emit_Evm (v0 : Withdrawal) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v2 := v0.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 57810043145978950376228313794938171962422655018555593468903716172405399886693) -- a constant, as the compiler evaluates it
  let v8 := v0.src
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v12 ← Evm_log2 v6 v7 v10
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Withdrawal (v1 : Withdrawal) : FeM Unit := do
  let v3 := v1
  let v4 ← Withdrawal_emit_Evm v3
  return ()

def recv0_arm4 (v0 : UInt256) : FeM Unit := do
  let mut r2 : Wei := default
  let mut r3 : RawCallOutcome := default
  let v3 ← Evm_caller
  let v4 := v3
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v4
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 := decide (v0.toNat ≤ v5.toNat)
  if v6 then
    let v7 := ()
    let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v4
    let v8 ← load (Place.mk .storage l1.slot)
    let v9 := v8 - v0
    let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v4
    store (Place.mk .storage l2.slot) v9
    let v10 := ()
    let v11 := (UInt256.ofNat 0)
    let v12 := decide (v0 = v11)
    let b6 : Wei → RawCallOutcome → UInt256 → FeM Unit := fun r2 r3 v1 => do
      let mut r2 := r2
      let mut r3 := r3
      let v17 := v1
      let v18 ← wei v0
      r2 := v18
      let v20 := r2
      let v21 ← send_value_Evm v4 v17 v20
      r3 := v21
      let v23 ← RawCallOutcome_success r3
      if v23 then
        let v24 := ()
        let v25 := (Withdrawal.mk v4 v0)
        let v27 ← emit_Evm_Withdrawal v25
        return ()
      else
        M.panic 0x01
    if v12 then
      let v13 := (UInt256.ofNat 2300)
      let v14 := v13
      b6 r2 r3 v14
    else
      let v15 := (UInt256.ofNat 0)
      let v16 := v15
      b6 r2 r3 v16
  else
    M.panic 0x01

def SolSlot_read_u8 (v0 : Place) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : SolWord := default
  let v2 ← load ((v0.field Layout.SolSlot 0).field Layout.Payload 0)
  let v3 := (SolWord.mk v2)
  r0 := v3
  let v4 := r0
  let v5 ← SolWord_get_u8 v4 v1
  return v5

def recv0_arm5 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (Field.store.field Layout.Weth9Store 2)
  let v2 := v1
  let v3 ← SolSlot_read_u8 v2 v0
  return v3

def recv0_arm6 (v0 : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v0
  let v1 ← load (Place.mk .storage l0.slot)
  return v1

def recv0_arm7 : FeM DynString := do
  let v0 := (Field.store.field Layout.Weth9Store 1)
  let v1 := v0
  let v2 ← SolString_load v1
  return v2

def recv0_arm8 (r0 : UInt256) (v0 : UInt256) : FeM Bool := do
  let mut r0 := r0
  let mut r3 : UInt256 := default
  let v2 ← Evm_caller
  r3 := v2
  let v3 := r3
  let v4 := r0
  let v5 ← transfer_from_Weth9Store_Evm_Evm v3 v4 v0
  return v5

def Deposit_emit_Evm (v0 : Deposit) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v2 := v0.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 102222681472383059465863322013072701928378550215632170212813623808969952268444) -- a constant, as the compiler evaluates it
  let v8 := v0.dst
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v12 ← Evm_log2 v6 v7 v10
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Deposit (v1 : Deposit) : FeM Unit := do
  let v3 := v1
  let v4 ← Deposit_emit_Evm v3
  return ()

def deposit_Weth9Store_Evm_Evm : FeM Unit := do
  let mut r3 : Wei := default
  let v1 ← Evm_caller
  let v2 := v1
  let v4 ← Evm_value
  r3 := v4
  let v5 := r3
  let v6 ← Wei_as_wei v5
  let v7 := v6
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v2
  let v8 ← load (Place.mk .storage l0.slot)
  let v9 := v8 + v7
  let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v2
  store (Place.mk .storage l1.slot) v9
  let v10 := ()
  let v11 := (Deposit.mk v2 v7)
  let v13 ← emit_Evm_Deposit v11
  return ()

def recv0_arm9 : FeM Unit := do
  let v0 ← deposit_Weth9Store_Evm_Evm
  return ()

def recv0_arm10 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v0
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot v1
  let v2 ← load (Place.mk .storage l1.slot)
  return v2

def recv1_arm0 : FeM Unit := do
  let v0 ← deposit_Weth9Store_Evm_Evm
  return ()

def Name_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Name_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Name := do
  let v2 := (Name.mk)
  return v2

def decode_from_prechecked_head_Name_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Name := do
  let v3 ← Name_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Name : FeM Name := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Name_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Name_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Name
  let result ← recv0_arm0
  Evm_return_value_Sol_DynString result

def Approve_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_u256
  let v4 := v2 + v3
  return v4

def Approve_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Approve := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 := v6 + v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Approve.mk v5 v10)
  return v11

def decode_from_prechecked_head_Approve_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Approve := do
  let v3 ← Approve_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Approve_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData v11 v5 v8
  return v12

def recv0_arm1_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv0_arm1 message.guy message.wad
  Evm_return_value_Sol_bool result

def TotalSupply_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def TotalSupply_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM TotalSupply := do
  let v2 := (TotalSupply.mk)
  return v2

def decode_from_prechecked_head_TotalSupply_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM TotalSupply := do
  let v3 ← TotalSupply_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← TotalSupply_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData v11 v5 v8
  return v12

def recv0_arm2_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv0_arm2
  Evm_return_value_Sol_u256 result

def TransferFrom_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_Address
  let v4 := v2 + v3
  let v5 ← abi_field_head_size_u256
  let v6 := v4 + v5
  return v6

def TransferFrom_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM TransferFrom := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 := v6 + v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := v1
  let v12 ← abi_field_head_size_Address
  let v13 := v11 + v12
  let v14 ← abi_field_head_size_Address
  let v15 := v13 + v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v3
  let v17 := v16
  let v18 := (TransferFrom.mk v5 v10 v17)
  return v18

def decode_from_prechecked_head_TransferFrom_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM TransferFrom := do
  let v3 ← TransferFrom_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← TransferFrom_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData v11 v5 v8
  return v12

def recv0_arm3_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv0_arm3 message.src message.dst message.wad
  Evm_return_value_Sol_bool result

def Withdraw_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 := v0 + v1
  return v2

def Withdraw_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Withdraw := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (Withdraw.mk v5)
  return v6

def decode_from_prechecked_head_Withdraw_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Withdraw := do
  let v3 ← Withdraw_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Withdraw : FeM Withdraw := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Withdraw_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Withdraw_Sol_CallData v11 v5 v8
  return v12

def recv0_arm4_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Withdraw
  let _ ← recv0_arm4 message.wad
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def Decimals_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Decimals_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Decimals := do
  let v2 := (Decimals.mk)
  return v2

def decode_from_prechecked_head_Decimals_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Decimals := do
  let v3 ← Decimals_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Decimals : FeM Decimals := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Decimals_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Decimals_Sol_CallData v11 v5 v8
  return v12

def recv0_arm5_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Decimals
  let result ← recv0_arm5
  Evm_return_value_Sol_u8 result

def BalanceOf_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  return v2

def BalanceOf_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM BalanceOf := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := (BalanceOf.mk v5)
  return v6

def decode_from_prechecked_head_BalanceOf_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM BalanceOf := do
  let v3 ← BalanceOf_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← BalanceOf_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData v11 v5 v8
  return v12

def recv0_arm6_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv0_arm6 message.owner
  Evm_return_value_Sol_u256 result

def Symbol_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Symbol_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Symbol := do
  let v2 := (Symbol.mk)
  return v2

def decode_from_prechecked_head_Symbol_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Symbol := do
  let v3 ← Symbol_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Symbol : FeM Symbol := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Symbol_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Symbol_Sol_CallData v11 v5 v8
  return v12

def recv0_arm7_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Symbol
  let result ← recv0_arm7
  Evm_return_value_Sol_DynString result

def Transfer_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_u256
  let v4 := v2 + v3
  return v4

def Transfer_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Transfer_1 := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 := v6 + v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Transfer_1.mk v5 v10)
  return v11

def decode_from_prechecked_head_Transfer_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Transfer_1 := do
  let v3 ← Transfer_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Transfer_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData v11 v5 v8
  return v12

def recv0_arm8_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv0_arm8 message.dst message.wad
  Evm_return_value_Sol_bool result

def Deposit_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Deposit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Deposit_1 := do
  let v2 := (Deposit_1.mk)
  return v2

def decode_from_prechecked_head_Deposit_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Deposit_1 := do
  let v3 ← Deposit_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Deposit : FeM Deposit_1 := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Deposit_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Deposit_Sol_CallData v11 v5 v8
  return v12

def recv0_arm9_entry : FeM Empty := do
  let message ← decode_runtime_args_Evm_Sol_Deposit
  let _ ← recv0_arm9
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def Allowance_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_Address
  let v4 := v2 + v3
  return v4

def Allowance_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Allowance := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_Address
  let v8 := v6 + v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := (Allowance.mk v5 v10)
  return v11

def decode_from_prechecked_head_Allowance_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Allowance := do
  let v3 ← Allowance_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Allowance_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData v11 v5 v8
  return v12

def recv0_arm10_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv0_arm10 message.owner message.guy
  Evm_return_value_Sol_u256 result

def recv1_arm0_entry : FeM Empty := do
  let _ ← recv1_arm0
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv1_arm0_entry
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x06fdde03 then
      recv0_arm0_entry
    else if selector = UInt256.ofNat 0x095ea7b3 then
      recv0_arm1_entry
    else if selector = UInt256.ofNat 0x18160ddd then
      recv0_arm2_entry
    else if selector = UInt256.ofNat 0x23b872dd then
      recv0_arm3_entry
    else if selector = UInt256.ofNat 0x2e1a7d4d then
      recv0_arm4_entry
    else if selector = UInt256.ofNat 0x313ce567 then
      recv0_arm5_entry
    else if selector = UInt256.ofNat 0x70a08231 then
      recv0_arm6_entry
    else if selector = UInt256.ofNat 0x95d89b41 then
      recv0_arm7_entry
    else if selector = UInt256.ofNat 0xa9059cbb then
      recv0_arm8_entry
    else if selector = UInt256.ofNat 0xd0e30db0 then
      recv0_arm9_entry
    else if selector = UInt256.ofNat 0xdd62ed3e then
      recv0_arm10_entry
    else
      recv1_arm0_entry

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


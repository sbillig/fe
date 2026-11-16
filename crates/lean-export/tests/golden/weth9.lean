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

def stored_len (head_0 : UInt256) : FeM UInt256 := do
  let head_2 := head_0
  let v3 := (UInt256.ofNat 1)
  let v4 := head_2 &&& v3
  let v5 := (UInt256.ofNat 1)
  let v6 := decide (v4 = v5)
  let long := v6
  let b3 : UInt256 → FeM UInt256 := fun v1 => do
    let len := v1
    let v19 := (UInt256.ofNat 32)
    let v20 := decide (len.toNat < v19.toNat)
    let v21 := long == v20
    if v21 then
      let v22 ← PANIC_STORAGE_ENCODING
      let v23 ← Panic_from v22
      let v24 ← revert_error_Panic v23
      nomatch v24
    else
      let v25 := ()
      let v26 := v25
      return len
  if long then
    let head_8 := head_0
    let v9 := (UInt256.ofNat 2)
    let v10 ← checkedDiv head_8 v9
    let v11 := v10
    b3 v11
  else
    let head_12 := head_0
    let v13 := (UInt256.ofNat 255)
    let v14 := head_12 &&& v13
    let v15 := (UInt256.ofNat 2)
    let v16 ← checkedDiv v14 v15
    let v17 := v16
    b3 v17

def Payload_locate (self : Place) (root : UInt256) (word_2 : UInt256) : FeM Location := do
  let v3 ← array_data_slot root
  let word_4 := word_2
  let v5 ← u256_wrapping_add v3 word_4
  let v6 := (UInt256.ofNat 0)
  let v7 := (Location.mk v5 v6)
  return v7

def leading_bytes (word_0 : UInt256) (count_1 : UInt256) : FeM UInt256 := do
  let count_3 := count_1
  let v4 := (UInt256.ofNat 32)
  let v5 := decide (v4.toNat ≤ count_3.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  if v5 then
    let word_6 := word_0
    let word_7 := word_6
    b3 word_7
  else
    let word_8 := word_0
    let v9 ← u256_max
    let count_10 := count_1
    let v11 := (UInt256.ofNat 8)
    let v12 := count_10 * v11
    let v13 := v9 >>> v12
    let v14 := ~~~v13
    let v15 := word_8 &&& v14
    let v16 := v15
    b3 v16

def payload_word (payload : MemSlice_u8) (offset_1 : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_len_u8 payload
  let v3 := v2
  let len := v3
  let offset_5 := offset_1
  let v6 := decide (len.toNat ≤ offset_5.toNat)
  if v6 then
    let v7 := (UInt256.ofNat 0)
    return v7
  else
    let v8 := ()
    let v9 := v8
    let v10 ← MemSlice_u8_word_at payload offset_1
    let offset_11 := offset_1
    let v12 := len - offset_11
    let v13 ← leading_bytes v10 v12
    return v13

def word_count (len_0 : UInt256) : FeM UInt256 := do
  let len_1 := len_0
  let v2 := (UInt256.ofNat 31)
  let v3 := len_1 + v2
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedDiv v3 v4
  return v5

def SolString_store (self : Place) (value : DynString) : FeM Unit := do
  let mut words : UInt256 := default
  let mut i_r1 : UInt256 := default
  let mut i_r2 : UInt256 := default
  let v3 ← DynString_span_ref value
  let v5 ← load ((self.field Layout.SolString 0).field Layout.Payload 0)
  let old := v5
  let v7 ← stored_len old
  let old_len := v7
  let v9 := (UInt256.ofNat 1)
  let v10 := old &&& v9
  let v11 := (UInt256.ofNat 0)
  let v12 := decide (v10 = v11)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → FeM Unit := fun words i_r1 i_r2 v2 => do
    let mut words := words
    let mut i_r1 := i_r1
    let mut i_r2 := i_r2
    let old_words := v2
    let v19 ← MemSlice_len_u8 v3
    let v20 := v19
    let len := v20
    let v22 := (UInt256.ofNat 0)
    let words_23 := v22
    words := words_23
    let v24 := (UInt256.ofNat 32)
    let v25 := decide (len.toNat < v24.toNat)
    let b6 : UInt256 → UInt256 → UInt256 → FeM Unit := fun words i_r1 i_r2 => do
      let mut words := words
      let mut i_r1 := i_r1
      let mut i_r2 := i_r2
      let words_63 := words
      let words_64 := words_63
      i_r2 := words_64
      let rec b10 (words : UInt256) (i_r1 : UInt256) (i_r2 : UInt256) : FeM Unit := do
        let mut words := words
        let mut i_r1 := i_r1
        let mut i_r2 := i_r2
        let i_65 := i_r2
        let i_66 := i_65
        let v67 := decide (i_66.toNat < old_words.toNat)
        if v67 then
          let v68 := (UInt256.ofNat 0)
          let i_69 := i_r2
          let l0 ← Payload_locate (self.field Layout.SolString 0) (self.field Layout.SolString 0).slot i_69
          store (Place.mk .storage l0.slot) v68
          let v70 := ()
          let v71 := (UInt256.ofNat 1)
          let i_72 := i_r2
          let v73 := i_72 + v71
          let i_74 := v73
          i_r2 := i_74
          let v75 := ()
          b10 words i_r1 i_r2
        else
          let v76 := ()
          return ()
      partial_fixpoint
      b10 words i_r1 i_r2
    if v25 then
      let v26 := (UInt256.ofNat 0)
      let v28 ← payload_word v3 v26
      let v29 := (UInt256.ofNat 2)
      let v30 := len * v29
      let v31 := v28 ||| v30
      store ((self.field Layout.SolString 0).field Layout.Payload 0) v31
      let v32 := ()
      let v33 := v32
      b6 words i_r1 i_r2
    else
      let v34 := (UInt256.ofNat 2)
      let v35 := len * v34
      let v36 := (UInt256.ofNat 1)
      let v37 := v35 + v36
      store ((self.field Layout.SolString 0).field Layout.Payload 0) v37
      let v38 := ()
      let v39 ← word_count len
      let words_40 := v39
      words := words_40
      let v41 := ()
      let v42 := (UInt256.ofNat 0)
      let i_43 := v42
      i_r1 := i_43
      let rec b7 (words : UInt256) (i_r1 : UInt256) (i_r2 : UInt256) : FeM Unit := do
        let mut words := words
        let mut i_r1 := i_r1
        let mut i_r2 := i_r2
        let i_44 := i_r1
        let i_45 := i_44
        let words_46 := words
        let v47 := decide (i_45.toNat < words_46.toNat)
        if v47 then
          let i_48 := i_r1
          let i_49 := i_48
          let v50 := (UInt256.ofNat 32)
          let v51 := i_49 * v50
          let v53 ← payload_word v3 v51
          let i_54 := i_r1
          let l1 ← Payload_locate (self.field Layout.SolString 0) (self.field Layout.SolString 0).slot i_54
          store (Place.mk .storage l1.slot) v53
          let v55 := ()
          let v56 := (UInt256.ofNat 1)
          let i_57 := i_r1
          let v58 := i_57 + v56
          let i_59 := v58
          i_r1 := i_59
          let v60 := ()
          b7 words i_r1 i_r2
        else
          let v61 := ()
          let v62 := v61
          b6 words i_r1 i_r2
      partial_fixpoint
      b7 words i_r1 i_r2
  if v12 then
    let v13 := (UInt256.ofNat 0)
    let v14 := v13
    b3 words i_r1 i_r2 v14
  else
    let v15 ← word_count old_len
    let v16 := v15
    b3 words i_r1 i_r2 v16

def SolSlot_write_u8 (self : Place) (offset : UInt256) (value : UInt256) : FeM Unit := do
  let mut word : SolWord := default
  let v3 ← u8_BITS
  let v4 := (UInt256.ofNat 256)
  let v5 := decide (v3 = v4)
  let b3 : SolWord → FeM Unit := fun word => do
    let mut word := word
    return ()
  if v5 then
    let v6 ← u8_BITS
    let v7 ← field_shift offset v6
    let v8 ← u8_to_packed value
    store ((self.field Layout.SolSlot 0).field Layout.Payload 0) v8
    let v9 := ()
    let v10 := v9
    b3 word
  else
    let v11 ← load ((self.field Layout.SolSlot 0).field Layout.Payload 0)
    let v12 := (SolWord.mk v11)
    let word_13 := v12
    word := word_13
    let (v15, v15_0) ← SolWord_set_u8 word offset value
    word := v15_0
    let v16 := word.word
    store ((self.field Layout.SolSlot 0).field Layout.Payload 0) v16
    let v17 := ()
    let v18 := v17
    b3 word

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

def SolString_load (self : Place) : FeM DynString := do
  let mut i : UInt256 := default
  let v2 ← load ((self.field Layout.SolString 0).field Layout.Payload 0)
  let head := v2
  let v4 ← stored_len head
  let len := v4
  let v6 := (UInt256.ofNat 32)
  let v7 ← word_count len
  let v8 := (UInt256.ofNat 32)
  let v9 := v7 * v8
  let v10 := v6 + v9
  let size := v10
  let v12 ← downcast_saturate_u256_usize size
  let v13 ← alloc_bytes v12
  let data := v13
  let v15 ← store_word data len
  let v16 := (UInt256.ofNat 1)
  let v17 := head &&& v16
  let v18 := (UInt256.ofNat 0)
  let v19 := decide (v17 = v18)
  let b3 : UInt256 → FeM DynString := fun i => do
    let mut i := i
    let v61 ← downcast_unchecked_u256_usize len
    let v62 ← DynString_from_abi_tail data v61 size
    return v62
  if v19 then
    let v20 := (UInt256.ofNat 0)
    let v21 := decide (len ≠ v20)
    let b6 : UInt256 → Unit → FeM DynString := fun i v1 => do
      let mut i := i
      let v29 := v1
      b3 i
    if v21 then
      let v22 := (UInt256.ofNat 32)
      let v23 := data + v22
      let v24 ← leading_bytes head len
      let v25 ← store_word v23 v24
      let v26 := v25
      b6 i v26
    else
      let v27 := ()
      let v28 := v27
      b6 i v28
  else
    let v30 := (UInt256.ofNat 0)
    let i_31 := v30
    i := i_31
    let rec b7 (i : UInt256) : FeM DynString := do
      let mut i := i
      let i_32 := i
      let i_33 := i_32
      let v34 ← word_count len
      let v35 := decide (i_33.toNat < v34.toNat)
      if v35 then
        let i_36 := i
        let l0 ← Payload_locate (self.field Layout.SolString 0) (self.field Layout.SolString 0).slot i_36
        let v37 ← load (Place.mk .storage l0.slot)
        let i_38 := i
        let i_39 := i_38
        let v40 := (UInt256.ofNat 32)
        let v41 := i_39 * v40
        let v42 := len - v41
        let v43 ← leading_bytes v37 v42
        let word := v43
        let v45 := (UInt256.ofNat 32)
        let i_46 := i
        let i_47 := i_46
        let v48 := (UInt256.ofNat 32)
        let v49 := i_47 * v48
        let v50 := v45 + v49
        let v51 ← downcast_unchecked_u256_usize v50
        let v52 := data + v51
        let v53 ← store_word v52 word
        let v54 := (UInt256.ofNat 1)
        let i_55 := i
        let v56 := i_55 + v54
        let i_57 := v56
        i := i_57
        let v58 := ()
        b7 i
      else
        let v59 := ()
        let v60 := v59
        b3 i
    partial_fixpoint
    b7 i

def recv_Name : FeM DynString := do
  let v0 := (Field.store.field Layout.Weth9Store 0)
  let v1 := v0
  let v2 ← SolString_load v1
  return v2

def Approval_emit_Evm (self : Approval) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := self.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 63486140976153616755203102783360879283472101686154884697241723088393386309925) -- a constant, as the compiler evaluates it
  let v8 := self.src
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v11 := self.guy
  r2 := v11
  let v12 := r2
  let v13 ← Address_as_topic v12
  let v15 ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Approval (event_1 : Approval) : FeM Unit := do
  let event_3 := event_1
  let v4 ← Approval_emit_Evm event_3
  return ()

def recv_Approve (guy : UInt256) (wad : UInt256) : FeM Bool := do
  let v3 ← Evm_caller
  let src := v3
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot src
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot guy
  store (Place.mk .storage l1.slot) wad
  let v5 := ()
  let v6 := (Approval.mk src guy wad)
  let v8 ← emit_Evm_Approval v6
  let v9 := true
  return v9

def recv_TotalSupply : FeM UInt256 := do
  let mut r1 : Wei := default
  let v1 ← Evm_selfbalance
  r1 := v1
  let v2 := r1
  let v3 ← Wei_as_wei v2
  return v3

def Transfer_emit_Evm (self : Transfer) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v2 := self.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 100389287136786176327247604509743168900146139575972864366142685224231313322991) -- a constant, as the compiler evaluates it
  let v8 := self.src
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v11 := self.dst
  r2 := v11
  let v12 := r2
  let v13 ← Address_as_topic v12
  let v15 ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Transfer (event_1 : Transfer) : FeM Unit := do
  let event_3 := event_1
  let v4 ← Transfer_emit_Evm event_3
  return ()

def transfer_from_Weth9Store_Evm_Evm (src : UInt256) (dst : UInt256) (wad_2 : UInt256) : FeM Bool := do
  let mut sender : UInt256 := default
  let v3 := src
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v3
  let v4 ← load (Place.mk .storage l0.slot)
  let wad_5 := wad_2
  let v6 := decide (wad_5.toNat ≤ v4.toNat)
  if v6 then
    let v7 := ()
    let v9 ← Evm_caller
    let sender_10 := v9
    sender := sender_10
    let sender_11 := sender
    let v12 ← ne_Address_Address src sender_11
    let b6 : UInt256 → FeM Bool := fun sender => do
      let mut sender := sender
      let wad_34 := wad_2
      let v35 := src
      let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v35
      let v36 ← load (Place.mk .storage l1.slot)
      let v37 := v36 - wad_34
      let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v35
      store (Place.mk .storage l2.slot) v37
      let v38 := ()
      let wad_39 := wad_2
      let v40 := dst
      let l3 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v40
      let v41 ← load (Place.mk .storage l3.slot)
      let v42 := v41 + wad_39
      let l4 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot v40
      store (Place.mk .storage l4.slot) v42
      let v43 := ()
      let v44 := src
      let v45 := dst
      let wad_46 := wad_2
      let v47 := (Transfer.mk v44 v45 wad_46)
      let v49 ← emit_Evm_Transfer v47
      let v50 := true
      return v50
    let b5 : UInt256 → FeM Bool := fun sender => do
      let mut sender := sender
      let v32 := ()
      let v33 := v32
      b6 sender
    if v12 then
      let v13 := src
      let sender_14 := sender
      let l5 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v13
      let l6 ← StorageMap_locate_Address_u256 (Place.mk .storage l5.slot) (Place.mk .storage l5.slot).slot sender_14
      let v15 ← load (Place.mk .storage l6.slot)
      let v16 ← u256_max
      let v17 := decide (v15 ≠ v16)
      if v17 then
        let v18 := src
        let sender_19 := sender
        let l7 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v18
        let l8 ← StorageMap_locate_Address_u256 (Place.mk .storage l7.slot) (Place.mk .storage l7.slot).slot sender_19
        let v20 ← load (Place.mk .storage l8.slot)
        let wad_21 := wad_2
        let v22 := decide (wad_21.toNat ≤ v20.toNat)
        if v22 then
          let v23 := ()
          let wad_24 := wad_2
          let v25 := src
          let sender_26 := sender
          let l9 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v25
          let l10 ← StorageMap_locate_Address_u256 (Place.mk .storage l9.slot) (Place.mk .storage l9.slot).slot sender_26
          let v27 ← load (Place.mk .storage l10.slot)
          let v28 := v27 - wad_24
          let sender_29 := sender
          let l11 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot v25
          let l12 ← StorageMap_locate_Address_u256 (Place.mk .storage l11.slot) (Place.mk .storage l11.slot).slot sender_29
          store (Place.mk .storage l12.slot) v28
          let v30 := ()
          let v31 := v30
          b6 sender
        else
          M.panic 0x01
      else
        b5 sender
    else
      b5 sender
  else
    M.panic 0x01

def recv_TransferFrom (src : UInt256) (dst : UInt256) (wad : UInt256) : FeM Bool := do
  let mut src := src
  let mut dst := dst
  let src_1 := src
  let dst_2 := dst
  let v3 ← transfer_from_Weth9Store_Evm_Evm src_1 dst_2 wad
  return v3

def Withdrawal_emit_Evm (self : Withdrawal) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v2 := self.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 57810043145978950376228313794938171962422655018555593468903716172405399886693) -- a constant, as the compiler evaluates it
  let v8 := self.src
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v12 ← Evm_log2 v6 v7 v10
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Withdrawal (event_1 : Withdrawal) : FeM Unit := do
  let event_3 := event_1
  let v4 ← Withdrawal_emit_Evm event_3
  return ()

def recv_Withdraw (wad : UInt256) : FeM Unit := do
  let mut r2 : Wei := default
  let mut r3 : RawCallOutcome := default
  let v3 ← Evm_caller
  let src := v3
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot src
  let v5 ← load (Place.mk .storage l0.slot)
  let v6 := decide (wad.toNat ≤ v5.toNat)
  if v6 then
    let v7 := ()
    let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot src
    let v8 ← load (Place.mk .storage l1.slot)
    let v9 := v8 - wad
    let l2 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot src
    store (Place.mk .storage l2.slot) v9
    let v10 := ()
    let v11 := (UInt256.ofNat 0)
    let v12 := decide (wad = v11)
    let b6 : Wei → RawCallOutcome → UInt256 → FeM Unit := fun r2 r3 v1 => do
      let mut r2 := r2
      let mut r3 := r3
      let gas_1 := v1
      let v18 ← wei wad
      r2 := v18
      let v20 := r2
      let v21 ← send_value_Evm src gas_1 v20
      r3 := v21
      let v23 ← RawCallOutcome_success r3
      if v23 then
        let v24 := ()
        let v25 := (Withdrawal.mk src wad)
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

def SolSlot_read_u8 (self : Place) (offset : UInt256) : FeM UInt256 := do
  let mut r0 : SolWord := default
  let v2 ← load ((self.field Layout.SolSlot 0).field Layout.Payload 0)
  let v3 := (SolWord.mk v2)
  r0 := v3
  let v4 := r0
  let v5 ← SolWord_get_u8 v4 offset
  return v5

def recv_Decimals : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (Field.store.field Layout.Weth9Store 2)
  let v2 := v1
  let v3 ← SolSlot_read_u8 v2 v0
  return v3

def recv_BalanceOf (owner : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot owner
  let v1 ← load (Place.mk .storage l0.slot)
  return v1

def recv_Symbol : FeM DynString := do
  let v0 := (Field.store.field Layout.Weth9Store 1)
  let v1 := v0
  let v2 ← SolString_load v1
  return v2

def recv_Transfer (dst : UInt256) (wad : UInt256) : FeM Bool := do
  let mut dst := dst
  let mut r3 : UInt256 := default
  let v2 ← Evm_caller
  r3 := v2
  let v3 := r3
  let dst_4 := dst
  let v5 ← transfer_from_Weth9Store_Evm_Evm v3 dst_4 wad
  return v5

def Deposit_emit_Evm (self : Deposit) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v2 := self.wad
  let v3 := v2
  let v4 ← encode_abi_payload_tuple_u256 v3
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 102222681472383059465863322013072701928378550215632170212813623808969952268444) -- a constant, as the compiler evaluates it
  let v8 := self.dst
  r1 := v8
  let v9 := r1
  let v10 ← Address_as_topic v9
  let v12 ← Evm_log2 v6 v7 v10
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_Deposit (event_1 : Deposit) : FeM Unit := do
  let event_3 := event_1
  let v4 ← Deposit_emit_Evm event_3
  return ()

def deposit_Weth9Store_Evm_Evm : FeM Unit := do
  let mut r3 : Wei := default
  let v1 ← Evm_caller
  let dst := v1
  let v4 ← Evm_value
  r3 := v4
  let v5 := r3
  let v6 ← Wei_as_wei v5
  let wad := v6
  let l0 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot dst
  let v8 ← load (Place.mk .storage l0.slot)
  let v9 := v8 + wad
  let l1 ← StorageMap_locate_Address_u256 (Field.store.field Layout.Weth9Store 3) (Field.store.field Layout.Weth9Store 3).slot dst
  store (Place.mk .storage l1.slot) v9
  let v10 := ()
  let v11 := (Deposit.mk dst wad)
  let v13 ← emit_Evm_Deposit v11
  return ()

def recv_Deposit : FeM Unit := do
  let v0 ← deposit_Weth9Store_Evm_Evm
  return ()

def recv_Allowance (owner : UInt256) (guy : UInt256) : FeM UInt256 := do
  let l0 ← StorageMap_locate_Address_StorageMap_Address_u256 (Field.store.field Layout.Weth9Store 4) (Field.store.field Layout.Weth9Store 4).slot owner
  let l1 ← StorageMap_locate_Address_u256 (Place.mk .storage l0.slot) (Place.mk .storage l0.slot).slot guy
  let v2 ← load (Place.mk .storage l1.slot)
  return v2

def recv1_arm0 : FeM Unit := do
  let v0 ← deposit_Weth9Store_Evm_Evm
  return ()

def Name_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Name_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Name := do
  let v2 := (Name.mk)
  return v2

def decode_from_prechecked_head_Name_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Name := do
  let v3 ← Name_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Name : FeM Name := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Name_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Name_Sol_CallData input_11 base len
  return v12

def recv_Name_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Name
  let result ← recv_Name
  Evm_return_value_Sol_DynString result

def Approve_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_u256
  let v4 := v2 + v3
  return v4

def Approve_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Approve := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 := __fe_msg_decode_base_6 + v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Approve.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Approve_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Approve := do
  let v3 ← Approve_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Approve_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData input_11 base len
  return v12

def recv_Approve_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv_Approve message.guy message.wad
  Evm_return_value_Sol_bool result

def TotalSupply_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def TotalSupply_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM TotalSupply := do
  let v2 := (TotalSupply.mk)
  return v2

def decode_from_prechecked_head_TotalSupply_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TotalSupply := do
  let v3 ← TotalSupply_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← TotalSupply_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData input_11 base len
  return v12

def recv_TotalSupply_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv_TotalSupply
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

def TransferFrom_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM TransferFrom := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 := __fe_msg_decode_base_6 + v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let __fe_msg_decode_base_11 := __fe_msg_decode_base_1
  let v12 ← abi_field_head_size_Address
  let v13 := __fe_msg_decode_base_11 + v12
  let v14 ← abi_field_head_size_Address
  let v15 := v13 + v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v15 __fe_msg_decode_input_len
  let __fe_field_2 := v16
  let v18 := (TransferFrom.mk __fe_field_0 __fe_field_1 __fe_field_2)
  return v18

def decode_from_prechecked_head_TransferFrom_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TransferFrom := do
  let v3 ← TransferFrom_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← TransferFrom_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData input_11 base len
  return v12

def recv_TransferFrom_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv_TransferFrom message.src message.dst message.wad
  Evm_return_value_Sol_bool result

def Withdraw_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 := v0 + v1
  return v2

def Withdraw_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Withdraw := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (Withdraw.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_Withdraw_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Withdraw := do
  let v3 ← Withdraw_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Withdraw : FeM Withdraw := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Withdraw_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Withdraw_Sol_CallData input_11 base len
  return v12

def recv_Withdraw_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Withdraw
  let _ ← recv_Withdraw message.wad
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def Decimals_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Decimals_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Decimals := do
  let v2 := (Decimals.mk)
  return v2

def decode_from_prechecked_head_Decimals_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Decimals := do
  let v3 ← Decimals_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Decimals : FeM Decimals := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Decimals_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Decimals_Sol_CallData input_11 base len
  return v12

def recv_Decimals_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Decimals
  let result ← recv_Decimals
  Evm_return_value_Sol_u8 result

def BalanceOf_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  return v2

def BalanceOf_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM BalanceOf := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (BalanceOf.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_BalanceOf_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM BalanceOf := do
  let v3 ← BalanceOf_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← BalanceOf_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData input_11 base len
  return v12

def recv_BalanceOf_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv_BalanceOf message.owner
  Evm_return_value_Sol_u256 result

def Symbol_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Symbol_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Symbol := do
  let v2 := (Symbol.mk)
  return v2

def decode_from_prechecked_head_Symbol_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Symbol := do
  let v3 ← Symbol_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Symbol : FeM Symbol := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Symbol_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Symbol_Sol_CallData input_11 base len
  return v12

def recv_Symbol_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Symbol
  let result ← recv_Symbol
  Evm_return_value_Sol_DynString result

def Transfer_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_u256
  let v4 := v2 + v3
  return v4

def Transfer_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Transfer_1 := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 := __fe_msg_decode_base_6 + v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Transfer_1.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Transfer_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Transfer_1 := do
  let v3 ← Transfer_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Transfer_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData input_11 base len
  return v12

def recv_Transfer_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv_Transfer message.dst message.wad
  Evm_return_value_Sol_bool result

def Deposit_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def Deposit_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM Deposit_1 := do
  let v2 := (Deposit_1.mk)
  return v2

def decode_from_prechecked_head_Deposit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Deposit_1 := do
  let v3 ← Deposit_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Deposit : FeM Deposit_1 := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Deposit_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Deposit_Sol_CallData input_11 base len
  return v12

def recv_Deposit_entry : FeM Empty := do
  let message ← decode_runtime_args_Evm_Sol_Deposit
  let _ ← recv_Deposit
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def Allowance_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Address
  let v2 := v0 + v1
  let v3 ← abi_field_head_size_Address
  let v4 := v2 + v3
  return v4

def Allowance_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Allowance := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Address
  let v8 := __fe_msg_decode_base_6 + v7
  let v9 ← decode_msg_field_from_Sol_Address_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1 := v9
  let v11 := (Allowance.mk __fe_field_0 __fe_field_1)
  return v11

def decode_from_prechecked_head_Allowance_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Allowance := do
  let v3 ← Allowance_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Allowance_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData input_11 base len
  return v12

def recv_Allowance_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv_Allowance message.owner message.guy
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
      recv_Name_entry
    else if selector = UInt256.ofNat 0x095ea7b3 then
      recv_Approve_entry
    else if selector = UInt256.ofNat 0x18160ddd then
      recv_TotalSupply_entry
    else if selector = UInt256.ofNat 0x23b872dd then
      recv_TransferFrom_entry
    else if selector = UInt256.ofNat 0x2e1a7d4d then
      recv_Withdraw_entry
    else if selector = UInt256.ofNat 0x313ce567 then
      recv_Decimals_entry
    else if selector = UInt256.ofNat 0x70a08231 then
      recv_BalanceOf_entry
    else if selector = UInt256.ofNat 0x95d89b41 then
      recv_Symbol_entry
    else if selector = UInt256.ofNat 0xa9059cbb then
      recv_Transfer_entry
    else if selector = UInt256.ofNat 0xd0e30db0 then
      recv_Deposit_entry
    else if selector = UInt256.ofNat 0xdd62ed3e then
      recv_Allowance_entry
    else
      recv1_arm0_entry

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


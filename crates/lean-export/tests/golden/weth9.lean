export Lib (Address_as_topic CallData DynString DynString.mk DynString_drop DynString_from_abi_tail DynString_span_ref Evm_caller Evm_input Evm_log2 Evm_log3 Evm_return_value_Sol_DynString Evm_return_value_Sol_bool Evm_return_value_Sol_u256 Evm_return_value_Sol_u8 Evm_selfbalance Evm_value Layout.SolSlot Location Location.mk MemBuffer MemBuffer_drop MemBuffer_span_ref MemSlice_len_u8 MemSlice_u8 MemSlice_u8_word_at PANIC_STORAGE_ENCODING Panic_from RawCallOutcome RawCallOutcome_success SolWord SolWord.mk SolWord_get_u8 SolWord_set_u8 Sol_SELECTOR_SIZE Sol_validate_runtime_head StorageMap_locate_Address_StorageMap_Address_u256 StorageMap_locate_Address_u256 Wei Wei_as_wei abi_field_head_size_Address abi_field_head_size_u256 alloc_bytes array_data_slot decode_msg_field_from_Sol_Address_CallData decode_msg_field_from_Sol_u256_CallData downcast_saturate_u256_usize downcast_unchecked_u256_usize encode_abi_payload_tuple_u256 field_shift len_CallData ne_Address_Address revert_error_Panic send_value_Evm store_word u256_max u256_wrapping_add u8_BITS u8_to_packed wei)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.Weth9Store : Layout :=
  { storage := [0, 1, 2, 3, 4] }

/-- `store.name` -/
abbrev Field.store.name : Place :=
  Field.store.field Layout.Weth9Store 0

/-- `store.symbol` -/
abbrev Field.store.symbol : Place :=
  Field.store.field Layout.Weth9Store 1

/-- `store.decimals` -/
abbrev Field.store.decimals : Place :=
  Field.store.field Layout.Weth9Store 2

def Layout.SolString : Layout :=
  { storage := [0] }

def Layout.Payload : Layout :=
  { storage := [0] }

/-- `store.allowance` -/
abbrev Field.store.allowance : Place :=
  Field.store.field Layout.Weth9Store 4

structure Approval where
  src : UInt256
  guy : UInt256
  wad : UInt256
deriving Inhabited

/-- `store.balance_of` -/
abbrev Field.store.balance_of : Place :=
  Field.store.field Layout.Weth9Store 3

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

/-- `lib::sol_string::stored_len` -/
def stored_len (head : UInt256) : FeM UInt256 := do
  let long := decide ((head &&& (UInt256.ofNat 1)) = (UInt256.ofNat 1))
  let join : UInt256 → FeM UInt256 := fun len => do
    let v21 := long == (decide (len.toNat < (UInt256.ofNat 32).toNat))
    if v21 then
      let v22 ← PANIC_STORAGE_ENCODING
      let v23 ← Panic_from v22
      let v24 ← revert_error_Panic v23
      nomatch v24
    else
      return len
  if long then
    let v10 ← checkedDiv head (UInt256.ofNat 2)
    join v10
  else
    let v16 ← checkedDiv (head &&& (UInt256.ofNat 255)) (UInt256.ofNat 2)
    join v16

/-- `<Payload as StateIndex>::locate` -/
def Payload_locate (self : Place) (root : UInt256) (word : UInt256) : FeM Location := do
  let v3 ← array_data_slot root
  let v5 ← u256_wrapping_add v3 word
  return Location.mk v5 (UInt256.ofNat 0)

/-- `lib::sol_string::leading_bytes` -/
def leading_bytes (word : UInt256) (count : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  if decide ((UInt256.ofNat 32).toNat ≤ count.toNat) then
    join word
  else
    let v9 ← u256_max
    join (word &&& (~~~(v9 >>> (count * (UInt256.ofNat 8)))))

/-- `lib::sol_string::payload_word` -/
def payload_word (payload : MemSlice_u8) (offset : UInt256) : FeM UInt256 := do
  let len ← MemSlice_len_u8 payload
  if decide (len.toNat ≤ offset.toNat) then
    return UInt256.ofNat 0
  else
    let v10 ← MemSlice_u8_word_at payload offset
    let v13 ← leading_bytes v10 (len - offset)
    return v13

/-- `lib::sol_string::word_count` -/
def word_count (len : UInt256) : FeM UInt256 := do
  let v5 ← checkedDiv (len + (UInt256.ofNat 31)) (UInt256.ofNat 32)
  return v5

/-- `SolString::store` -/
def SolString_store (self : Place) (value : DynString) : FeM Unit := do
  let mut words : UInt256 := default
  let mut i_r1 : UInt256 := default
  let mut i_r2 : UInt256 := default
  let v3 ← DynString_span_ref value
  let old ← load ((self.field Layout.SolString 0).field Layout.Payload 0)
  let old_len ← stored_len old
  let v12 := decide ((old &&& (UInt256.ofNat 1)) = (UInt256.ofNat 0))
  let join : UInt256 → UInt256 → UInt256 → UInt256 → FeM Unit := fun words i_r1 i_r2 old_words => do
    let mut words := words
    let mut i_r1 := i_r1
    let mut i_r2 := i_r2
    let len ← MemSlice_len_u8 v3
    words := (UInt256.ofNat 0)
    let join_1 : UInt256 → UInt256 → UInt256 → FeM Unit := fun words i_r1 i_r2 => do
      let mut words := words
      let mut i_r1 := i_r1
      let mut i_r2 := i_r2
      let words_63 := words
      i_r2 := words_63
      let rec loop (words : UInt256) (i_r1 : UInt256) (i_r2 : UInt256) : FeM Unit := do
        let mut words := words
        let mut i_r1 := i_r1
        let mut i_r2 := i_r2
        let i_65 := i_r2
        if decide (i_65.toNat < old_words.toNat) then
          let i_69 := i_r2
          let l0 ← Payload_locate (self.field Layout.SolString 0) (self.field Layout.SolString 0).slot i_69
          store (Place.mk .storage l0.slot) (UInt256.ofNat 0)
          let i_72 := i_r2
          i_r2 := (i_72 + (UInt256.ofNat 1))
          loop words i_r1 i_r2
        else
          return ()
      partial_fixpoint
      loop words i_r1 i_r2
    if decide (len.toNat < (UInt256.ofNat 32).toNat) then
      let v28 ← payload_word v3 (UInt256.ofNat 0)
      store ((self.field Layout.SolString 0).field Layout.Payload 0) (v28 ||| (len * (UInt256.ofNat 2)))
      join_1 words i_r1 i_r2
    else
      store ((self.field Layout.SolString 0).field Layout.Payload 0) ((len * (UInt256.ofNat 2)) + (UInt256.ofNat 1))
      let words_39 ← word_count len
      words := words_39
      i_r1 := (UInt256.ofNat 0)
      let rec loop_1 (words : UInt256) (i_r1 : UInt256) (i_r2 : UInt256) : FeM Unit := do
        let mut words := words
        let mut i_r1 := i_r1
        let mut i_r2 := i_r2
        let i_44 := i_r1
        let words_46 := words
        if decide (i_44.toNat < words_46.toNat) then
          let i_48 := i_r1
          let v53 ← payload_word v3 (i_48 * (UInt256.ofNat 32))
          let i_54 := i_r1
          let l1 ← Payload_locate (self.field Layout.SolString 0) (self.field Layout.SolString 0).slot i_54
          store (Place.mk .storage l1.slot) v53
          let i_57 := i_r1
          i_r1 := (i_57 + (UInt256.ofNat 1))
          loop_1 words i_r1 i_r2
        else
          join_1 words i_r1 i_r2
      partial_fixpoint
      loop_1 words i_r1 i_r2
  if v12 then
    join words i_r1 i_r2 (UInt256.ofNat 0)
  else
    let v15 ← word_count old_len
    join words i_r1 i_r2 v15

/-- `SolSlot::write<u8>` -/
def SolSlot_write_u8 (self : Place) (offset : UInt256) (value : UInt256) : FeM Unit := do
  let mut word : SolWord := default
  let v3 ← u8_BITS
  let join : SolWord → FeM Unit := fun word => do
    let mut word := word
    return ()
  if decide (v3 = (UInt256.ofNat 256)) then
    let v6 ← u8_BITS
    let _ ← field_shift offset v6
    let v8 ← u8_to_packed value
    store ((self.field Layout.SolSlot 0).field Layout.Payload 0) v8
    join word
  else
    let v11 ← load ((self.field Layout.SolSlot 0).field Layout.Payload 0)
    word := (SolWord.mk v11)
    let (v15, v15_0) ← SolWord_set_u8 word offset value
    word := v15_0
    let v16 := word.word
    store ((self.field Layout.SolSlot 0).field Layout.Payload 0) v16
    join word

/-- `init()` of contract `WETH9` -/
def init : FeM Unit := do
  let mut r1 : DynString := default
  let mut r2 : DynString := default
  let v0_p ← stringLiteral ⟨#[87, 114, 97, 112, 112, 101, 100, 32, 69, 116, 104, 101, 114]⟩
  let v0 := (DynString.mk v0_p (UInt256.ofNat 13) (UInt256.ofNat 64))
  r1 := v0
  let v1 := Field.store.name
  let _ ← SolString_store v1 r1
  let _ ← DynString_drop r1
  let v4_p ← stringLiteral ⟨#[87, 69, 84, 72]⟩
  let v4 := (DynString.mk v4_p (UInt256.ofNat 4) (UInt256.ofNat 64))
  r2 := v4
  let v5 := Field.store.symbol
  let _ ← SolString_store v5 r2
  let _ ← DynString_drop r2
  let v11 := Field.store.decimals
  let _ ← SolSlot_write_u8 v11 (UInt256.ofNat 0) (UInt256.ofNat 18)
  return ()

/-- `SolString::load` -/
def SolString_load (self : Place) : FeM DynString := do
  let mut i : UInt256 := default
  let head ← load ((self.field Layout.SolString 0).field Layout.Payload 0)
  let len ← stored_len head
  let v7 ← word_count len
  let size := (UInt256.ofNat 32) + (v7 * (UInt256.ofNat 32))
  let v12 ← downcast_saturate_u256_usize size
  let data ← alloc_bytes v12
  let _ ← store_word data len
  let v19 := decide ((head &&& (UInt256.ofNat 1)) = (UInt256.ofNat 0))
  let join : UInt256 → FeM DynString := fun i => do
    let mut i := i
    let v61 ← downcast_unchecked_u256_usize len
    let v62 ← DynString_from_abi_tail data v61 size
    return v62
  if v19 then
    let join_1 : UInt256 → Unit → FeM DynString := fun i v1 => do
      let mut i := i
      join i
    if decide (len ≠ (UInt256.ofNat 0)) then
      let v24 ← leading_bytes head len
      let v25 ← store_word (data + (UInt256.ofNat 32)) v24
      join_1 i v25
    else
      join_1 i ()
  else
    i := (UInt256.ofNat 0)
    let rec loop (i : UInt256) : FeM DynString := do
      let mut i := i
      let i_32 := i
      let v34 ← word_count len
      if decide (i_32.toNat < v34.toNat) then
        let i_36 := i
        let l0 ← Payload_locate (self.field Layout.SolString 0) (self.field Layout.SolString 0).slot i_36
        let v37 ← load (Place.mk .storage l0.slot)
        let i_38 := i
        let word ← leading_bytes v37 (len - (i_38 * (UInt256.ofNat 32)))
        let i_46 := i
        let v51 ← downcast_unchecked_u256_usize ((UInt256.ofNat 32) + (i_46 * (UInt256.ofNat 32)))
        let _ ← store_word (data + v51) word
        let i_55 := i
        i := (i_55 + (UInt256.ofNat 1))
        loop i
      else
        join i
    partial_fixpoint
    loop i

/-- `Name -> DynString`, selector `0x06fdde03` (`name()`) -/
def recv_Name : FeM DynString := do
  let v0 := Field.store.name
  let v2 ← SolString_load v0
  return v2

/-- `<Approval as Event>::emit<Evm>` -/
def Approval_emit_Evm (self : Approval) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v4 ← encode_abi_payload_tuple_u256 self.wad
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 63486140976153616755203102783360879283472101686154884697241723088393386309925) -- a constant, as the compiler evaluates it
  r1 := self.src
  let v9 := r1
  let v10 ← Address_as_topic v9
  r2 := self.guy
  let v12 := r2
  let v13 ← Address_as_topic v12
  let _ ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, Approval>` -/
def emit_Evm_Approval (event : Approval) : FeM Unit := do
  let _ ← Approval_emit_Evm event
  return ()

/-- `Approve { guy, wad } -> bool`, selector `0x095ea7b3` (`approve(address,uint256)`) -/
def recv_Approve (guy : UInt256) (wad : UInt256) : FeM Bool := do
  let src ← Evm_caller
  let allowance_src ← StorageMap_locate_Address_StorageMap_Address_u256 Field.store.allowance Field.store.allowance.slot src
  let allowance_src_guy ← StorageMap_locate_Address_u256 (Place.mk .storage allowance_src.slot) (Place.mk .storage allowance_src.slot).slot guy
  store (Place.mk .storage allowance_src_guy.slot) wad
  let _ ← emit_Evm_Approval (Approval.mk src guy wad)
  return true

/-- `TotalSupply -> u256`, selector `0x18160ddd` (`totalSupply()`) -/
def recv_TotalSupply : FeM UInt256 := do
  let mut r1 : Wei := default
  let v1 ← Evm_selfbalance
  r1 := v1
  let v2 := r1
  let v3 ← Wei_as_wei v2
  return v3

/-- `<Transfer as Event>::emit<Evm>` -/
def Transfer_emit_Evm (self : Transfer) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v4 ← encode_abi_payload_tuple_u256 self.wad
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 100389287136786176327247604509743168900146139575972864366142685224231313322991) -- a constant, as the compiler evaluates it
  r1 := self.src
  let v9 := r1
  let v10 ← Address_as_topic v9
  r2 := self.dst
  let v12 := r2
  let v13 ← Address_as_topic v12
  let _ ← Evm_log3 v6 v7 v10 v13
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, Transfer>` -/
def emit_Evm_Transfer (event : Transfer) : FeM Unit := do
  let _ ← Transfer_emit_Evm event
  return ()

/-- `lib::transfer_from<Weth9Store, Evm, Evm>` -/
def transfer_from_Weth9Store_Evm_Evm (src : UInt256) (dst : UInt256) (wad : UInt256) : FeM Bool := do
  let mut sender : UInt256 := default
  let balance_of_entry ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot src
  let v4 ← load (Place.mk .storage balance_of_entry.slot)
  if decide (wad.toNat ≤ v4.toNat) then
    let sender_9 ← Evm_caller
    sender := sender_9
    let sender_11 := sender
    let v12 ← ne_Address_Address src sender_11
    let join : UInt256 → FeM Bool := fun sender => do
      let mut sender := sender
      let v35 := src
      let balance_of_entry_1 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot v35
      let v36 ← load (Place.mk .storage balance_of_entry_1.slot)
      let balance_of_entry_2 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot v35
      store (Place.mk .storage balance_of_entry_2.slot) (v36 - wad)
      let v40 := dst
      let balance_of_entry_3 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot v40
      let v41 ← load (Place.mk .storage balance_of_entry_3.slot)
      let balance_of_entry_4 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot v40
      store (Place.mk .storage balance_of_entry_4.slot) (v41 + wad)
      let _ ← emit_Evm_Transfer (Transfer.mk src dst wad)
      return true
    let join_1 : UInt256 → FeM Bool := fun sender => do
      let mut sender := sender
      join sender
    if v12 then
      let sender_14 := sender
      let allowance_entry ← StorageMap_locate_Address_StorageMap_Address_u256 Field.store.allowance Field.store.allowance.slot src
      let allowance_entry_sender_14 ← StorageMap_locate_Address_u256 (Place.mk .storage allowance_entry.slot) (Place.mk .storage allowance_entry.slot).slot sender_14
      let v15 ← load (Place.mk .storage allowance_entry_sender_14.slot)
      let v16 ← u256_max
      if decide (v15 ≠ v16) then
        let sender_19 := sender
        let allowance_entry_1 ← StorageMap_locate_Address_StorageMap_Address_u256 Field.store.allowance Field.store.allowance.slot src
        let allowance_entry_sender_19 ← StorageMap_locate_Address_u256 (Place.mk .storage allowance_entry_1.slot) (Place.mk .storage allowance_entry_1.slot).slot sender_19
        let v20 ← load (Place.mk .storage allowance_entry_sender_19.slot)
        if decide (wad.toNat ≤ v20.toNat) then
          let v25 := src
          let sender_26 := sender
          let allowance_entry_2 ← StorageMap_locate_Address_StorageMap_Address_u256 Field.store.allowance Field.store.allowance.slot v25
          let allowance_entry_sender_26 ← StorageMap_locate_Address_u256 (Place.mk .storage allowance_entry_2.slot) (Place.mk .storage allowance_entry_2.slot).slot sender_26
          let v27 ← load (Place.mk .storage allowance_entry_sender_26.slot)
          let sender_29 := sender
          let allowance_entry_3 ← StorageMap_locate_Address_StorageMap_Address_u256 Field.store.allowance Field.store.allowance.slot v25
          let allowance_entry_sender_29 ← StorageMap_locate_Address_u256 (Place.mk .storage allowance_entry_3.slot) (Place.mk .storage allowance_entry_3.slot).slot sender_29
          store (Place.mk .storage allowance_entry_sender_29.slot) (v27 - wad)
          join sender
        else
          M.panic 0x01
      else
        join_1 sender
    else
      join_1 sender
  else
    M.panic 0x01

/-- `TransferFrom { src, dst, wad } -> bool`, selector `0x23b872dd` (`transferFrom(address,address,uint256)`) -/
def recv_TransferFrom (src : UInt256) (dst : UInt256) (wad : UInt256) : FeM Bool := do
  let mut src := src
  let mut dst := dst
  let src_1 := src
  let dst_2 := dst
  let v3 ← transfer_from_Weth9Store_Evm_Evm src_1 dst_2 wad
  return v3

/-- `<Withdrawal as Event>::emit<Evm>` -/
def Withdrawal_emit_Evm (self : Withdrawal) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v4 ← encode_abi_payload_tuple_u256 self.wad
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 57810043145978950376228313794938171962422655018555593468903716172405399886693) -- a constant, as the compiler evaluates it
  r1 := self.src
  let v9 := r1
  let v10 ← Address_as_topic v9
  let _ ← Evm_log2 v6 v7 v10
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, Withdrawal>` -/
def emit_Evm_Withdrawal (event : Withdrawal) : FeM Unit := do
  let _ ← Withdrawal_emit_Evm event
  return ()

/-- `Withdraw { wad }`, selector `0x2e1a7d4d` (`withdraw(uint256)`) -/
def recv_Withdraw (wad : UInt256) : FeM Unit := do
  let mut r2 : Wei := default
  let mut r3 : RawCallOutcome := default
  let src ← Evm_caller
  let balance_of_src ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot src
  let v5 ← load (Place.mk .storage balance_of_src.slot)
  if decide (wad.toNat ≤ v5.toNat) then
    let balance_of_src_1 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot src
    let v8 ← load (Place.mk .storage balance_of_src_1.slot)
    let balance_of_src_2 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot src
    store (Place.mk .storage balance_of_src_2.slot) (v8 - wad)
    let join : Wei → RawCallOutcome → UInt256 → FeM Unit := fun r2 r3 gas_1 => do
      let mut r2 := r2
      let mut r3 := r3
      let v18 ← wei wad
      r2 := v18
      let v20 := r2
      let v21 ← send_value_Evm src gas_1 v20
      r3 := v21
      let v23 ← RawCallOutcome_success r3
      if v23 then
        let _ ← emit_Evm_Withdrawal (Withdrawal.mk src wad)
        return ()
      else
        M.panic 0x01
    if decide (wad = (UInt256.ofNat 0)) then
      join r2 r3 (UInt256.ofNat 2300)
    else
      join r2 r3 (UInt256.ofNat 0)
  else
    M.panic 0x01

/-- `SolSlot::read<u8>` -/
def SolSlot_read_u8 (self : Place) (offset : UInt256) : FeM UInt256 := do
  let mut r0 : SolWord := default
  let v2 ← load ((self.field Layout.SolSlot 0).field Layout.Payload 0)
  r0 := (SolWord.mk v2)
  let v4 := r0
  let v5 ← SolWord_get_u8 v4 offset
  return v5

/-- `Decimals -> u8`, selector `0x313ce567` (`decimals()`) -/
def recv_Decimals : FeM UInt256 := do
  let v1 := Field.store.decimals
  let v3 ← SolSlot_read_u8 v1 (UInt256.ofNat 0)
  return v3

/-- `BalanceOf { owner } -> u256`, selector `0x70a08231` (`balanceOf(address)`) -/
def recv_BalanceOf (owner : UInt256) : FeM UInt256 := do
  let balance_of_owner ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot owner
  let v1 ← load (Place.mk .storage balance_of_owner.slot)
  return v1

/-- `Symbol -> DynString`, selector `0x95d89b41` (`symbol()`) -/
def recv_Symbol : FeM DynString := do
  let v0 := Field.store.symbol
  let v2 ← SolString_load v0
  return v2

/-- `Transfer { dst, wad } -> bool`, selector `0xa9059cbb` (`transfer(address,uint256)`) -/
def recv_Transfer (dst : UInt256) (wad : UInt256) : FeM Bool := do
  let mut dst := dst
  let mut r3 : UInt256 := default
  let v2 ← Evm_caller
  r3 := v2
  let v3 := r3
  let dst_4 := dst
  let v5 ← transfer_from_Weth9Store_Evm_Evm v3 dst_4 wad
  return v5

/-- `<Deposit as Event>::emit<Evm>` -/
def Deposit_emit_Evm (self : Deposit) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v4 ← encode_abi_payload_tuple_u256 self.wad
  r0 := v4
  let v6 ← MemBuffer_span_ref r0
  let v7 := (UInt256.ofNat 102222681472383059465863322013072701928378550215632170212813623808969952268444) -- a constant, as the compiler evaluates it
  r1 := self.dst
  let v9 := r1
  let v10 ← Address_as_topic v9
  let _ ← Evm_log2 v6 v7 v10
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, Deposit>` -/
def emit_Evm_Deposit (event : Deposit) : FeM Unit := do
  let _ ← Deposit_emit_Evm event
  return ()

/-- `lib::deposit<Weth9Store, Evm, Evm>` -/
def deposit_Weth9Store_Evm_Evm : FeM Unit := do
  let mut r3 : Wei := default
  let dst ← Evm_caller
  let v4 ← Evm_value
  r3 := v4
  let v5 := r3
  let wad ← Wei_as_wei v5
  let balance_of_dst ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot dst
  let v8 ← load (Place.mk .storage balance_of_dst.slot)
  let balance_of_dst_1 ← StorageMap_locate_Address_u256 Field.store.balance_of Field.store.balance_of.slot dst
  store (Place.mk .storage balance_of_dst_1.slot) (v8 + wad)
  let _ ← emit_Evm_Deposit (Deposit.mk dst wad)
  return ()

/-- `Deposit`, selector `0xd0e30db0` (`deposit()`) -/
def recv_Deposit : FeM Unit := do
  let _ ← deposit_Weth9Store_Evm_Evm
  return ()

/-- `Allowance { owner, guy } -> u256`, selector `0xdd62ed3e` (`allowance(address,address)`) -/
def recv_Allowance (owner : UInt256) (guy : UInt256) : FeM UInt256 := do
  let allowance_owner ← StorageMap_locate_Address_StorageMap_Address_u256 Field.store.allowance Field.store.allowance.slot owner
  let allowance_owner_guy ← StorageMap_locate_Address_u256 (Place.mk .storage allowance_owner.slot) (Place.mk .storage allowance_owner.slot).slot guy
  let v2 ← load (Place.mk .storage allowance_owner_guy.slot)
  return v2

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Unit := do
  let _ ← deposit_Weth9Store_Evm_Evm
  return ()

/-- `<Name as AbiSize>::HEAD_SIZE` -/
def Name_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Name as Decode>::decode_from<CallData>` -/
def Name_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Name := do
  return Name.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Name, Sol, CallData>` -/
def decode_from_prechecked_head_Name_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Name := do
  let v3 ← Name_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Name>` -/
def decode_runtime_args_Evm_Sol_Name : FeM Name := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Name_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Name_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Name`, for `lib::Weth9Msg::Name` -/
def recv_Name_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Name
  let result ← recv_Name
  Evm_return_value_Sol_DynString result

/-- `<Approve as AbiSize>::HEAD_SIZE` -/
def Approve_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v3 ← abi_field_head_size_u256
  return ((UInt256.ofNat 0) + v1) + v3

/-- `<Approve as Decode>::decode_from<CallData>` -/
def Approve_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Approve := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_Address
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 (v1 + v7) v2
  return Approve.mk v4 v9

/-- `core::abi::Decode::decode_from_prechecked_head<Approve, Sol, CallData>` -/
def decode_from_prechecked_head_Approve_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Approve := do
  let v3 ← Approve_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Approve>` -/
def decode_runtime_args_Evm_Sol_Approve : FeM Approve := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Approve_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Approve_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Approve`, for `lib::Weth9Msg::Approve` -/
def recv_Approve_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Approve
  let result ← recv_Approve message.guy message.wad
  Evm_return_value_Sol_bool result

/-- `<TotalSupply as AbiSize>::HEAD_SIZE` -/
def TotalSupply_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<TotalSupply as Decode>::decode_from<CallData>` -/
def TotalSupply_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM TotalSupply := do
  return TotalSupply.mk

/-- `core::abi::Decode::decode_from_prechecked_head<TotalSupply, Sol, CallData>` -/
def decode_from_prechecked_head_TotalSupply_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TotalSupply := do
  let v3 ← TotalSupply_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, TotalSupply>` -/
def decode_runtime_args_Evm_Sol_TotalSupply : FeM TotalSupply := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← TotalSupply_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TotalSupply_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_TotalSupply`, for `lib::Weth9Msg::TotalSupply` -/
def recv_TotalSupply_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TotalSupply
  let result ← recv_TotalSupply
  Evm_return_value_Sol_u256 result

/-- `<TransferFrom as AbiSize>::HEAD_SIZE` -/
def TransferFrom_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v3 ← abi_field_head_size_Address
  let v5 ← abi_field_head_size_u256
  return (((UInt256.ofNat 0) + v1) + v3) + v5

/-- `<TransferFrom as Decode>::decode_from<CallData>` -/
def TransferFrom_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM TransferFrom := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_Address
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 (v1 + v7) v2
  let v12 ← abi_field_head_size_Address
  let v14 ← abi_field_head_size_Address
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 ((v1 + v12) + v14) v2
  return TransferFrom.mk v4 v9 v16

/-- `core::abi::Decode::decode_from_prechecked_head<TransferFrom, Sol, CallData>` -/
def decode_from_prechecked_head_TransferFrom_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TransferFrom := do
  let v3 ← TransferFrom_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, TransferFrom>` -/
def decode_runtime_args_Evm_Sol_TransferFrom : FeM TransferFrom := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← TransferFrom_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_TransferFrom_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_TransferFrom`, for `lib::Weth9Msg::TransferFrom` -/
def recv_TransferFrom_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TransferFrom
  let result ← recv_TransferFrom message.src message.dst message.wad
  Evm_return_value_Sol_bool result

/-- `<Withdraw as AbiSize>::HEAD_SIZE` -/
def Withdraw_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  return (UInt256.ofNat 0) + v1

/-- `<Withdraw as Decode>::decode_from<CallData>` -/
def Withdraw_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Withdraw := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v2
  return Withdraw.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Withdraw, Sol, CallData>` -/
def decode_from_prechecked_head_Withdraw_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Withdraw := do
  let v3 ← Withdraw_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Withdraw>` -/
def decode_runtime_args_Evm_Sol_Withdraw : FeM Withdraw := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Withdraw_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Withdraw_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Withdraw`, for `lib::Weth9Msg::Withdraw` -/
def recv_Withdraw_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Withdraw
  let _ ← recv_Withdraw message.wad
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Decimals as AbiSize>::HEAD_SIZE` -/
def Decimals_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Decimals as Decode>::decode_from<CallData>` -/
def Decimals_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Decimals := do
  return Decimals.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Decimals, Sol, CallData>` -/
def decode_from_prechecked_head_Decimals_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Decimals := do
  let v3 ← Decimals_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Decimals>` -/
def decode_runtime_args_Evm_Sol_Decimals : FeM Decimals := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Decimals_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Decimals_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Decimals`, for `lib::Weth9Msg::Decimals` -/
def recv_Decimals_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Decimals
  let result ← recv_Decimals
  Evm_return_value_Sol_u8 result

/-- `<BalanceOf as AbiSize>::HEAD_SIZE` -/
def BalanceOf_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  return (UInt256.ofNat 0) + v1

/-- `<BalanceOf as Decode>::decode_from<CallData>` -/
def BalanceOf_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM BalanceOf := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  return BalanceOf.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<BalanceOf, Sol, CallData>` -/
def decode_from_prechecked_head_BalanceOf_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM BalanceOf := do
  let v3 ← BalanceOf_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, BalanceOf>` -/
def decode_runtime_args_Evm_Sol_BalanceOf : FeM BalanceOf := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← BalanceOf_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_BalanceOf_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_BalanceOf`, for `lib::Weth9Msg::BalanceOf` -/
def recv_BalanceOf_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_BalanceOf
  let result ← recv_BalanceOf message.owner
  Evm_return_value_Sol_u256 result

/-- `<Symbol as AbiSize>::HEAD_SIZE` -/
def Symbol_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Symbol as Decode>::decode_from<CallData>` -/
def Symbol_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Symbol := do
  return Symbol.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Symbol, Sol, CallData>` -/
def decode_from_prechecked_head_Symbol_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Symbol := do
  let v3 ← Symbol_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Symbol>` -/
def decode_runtime_args_Evm_Sol_Symbol : FeM Symbol := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Symbol_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Symbol_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Symbol`, for `lib::Weth9Msg::Symbol` -/
def recv_Symbol_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Symbol
  let result ← recv_Symbol
  Evm_return_value_Sol_DynString result

/-- `<Transfer as AbiSize>::HEAD_SIZE` -/
def Transfer_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v3 ← abi_field_head_size_u256
  return ((UInt256.ofNat 0) + v1) + v3

/-- `<Transfer as Decode>::decode_from<CallData>` -/
def Transfer_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Transfer_1 := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_Address
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 (v1 + v7) v2
  return Transfer_1.mk v4 v9

/-- `core::abi::Decode::decode_from_prechecked_head<Transfer, Sol, CallData>` -/
def decode_from_prechecked_head_Transfer_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Transfer_1 := do
  let v3 ← Transfer_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Transfer>` -/
def decode_runtime_args_Evm_Sol_Transfer : FeM Transfer_1 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Transfer_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Transfer_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Transfer`, for `lib::Weth9Msg::Transfer` -/
def recv_Transfer_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Transfer
  let result ← recv_Transfer message.dst message.wad
  Evm_return_value_Sol_bool result

/-- `<Deposit as AbiSize>::HEAD_SIZE` -/
def Deposit_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Deposit as Decode>::decode_from<CallData>` -/
def Deposit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Deposit_1 := do
  return Deposit_1.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Deposit, Sol, CallData>` -/
def decode_from_prechecked_head_Deposit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Deposit_1 := do
  let v3 ← Deposit_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Deposit>` -/
def decode_runtime_args_Evm_Sol_Deposit : FeM Deposit_1 := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Deposit_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Deposit_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Deposit`, for `lib::Weth9Msg::Deposit` -/
def recv_Deposit_entry : FeM Empty := do
  let message ← decode_runtime_args_Evm_Sol_Deposit
  let _ ← recv_Deposit
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Allowance as AbiSize>::HEAD_SIZE` -/
def Allowance_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Address
  let v3 ← abi_field_head_size_Address
  return ((UInt256.ofNat 0) + v1) + v3

/-- `<Allowance as Decode>::decode_from<CallData>` -/
def Allowance_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Allowance := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Address_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_Address
  let v9 ← decode_msg_field_from_Sol_Address_CallData v0 v1 (v1 + v7) v2
  return Allowance.mk v4 v9

/-- `core::abi::Decode::decode_from_prechecked_head<Allowance, Sol, CallData>` -/
def decode_from_prechecked_head_Allowance_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Allowance := do
  let v3 ← Allowance_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Allowance>` -/
def decode_runtime_args_Evm_Sol_Allowance : FeM Allowance := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Allowance_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Allowance_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Allowance`, for `lib::Weth9Msg::Allowance` -/
def recv_Allowance_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Allowance
  let result ← recv_Allowance message.owner message.guy
  Evm_return_value_Sol_u256 result

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `WETH9`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
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
      recv_fallback_entry

/-- The deployment entry of contract `WETH9`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


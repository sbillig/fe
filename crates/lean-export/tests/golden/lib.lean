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

/-- `CallData::new` -/
def CallData_new : FeM CallData := do
  let v1 := (CallData.mk (UInt256.ofNat 0))
  return v1

/-- `<Evm as ContractHost>::input` -/
def Evm_input : FeM CallData := do
  let v1 ← CallData_new
  return v1

/-- `<Sol as Abi>::SELECTOR_SIZE` -/
def Sol_SELECTOR_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 4)

/-- `<u256 as IntWord>::to_word` -/
def u256_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<u64 as IntWord>::IS_SIGNED` -/
def u64_IS_SIGNED : FeM Bool := do
  return false

/-- `core::num::U64_MASK` -/
def U64_MASK : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551615)

/-- `<u64 as IntWord>::MASK` -/
def u64_MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

/-- `<u64 as IntWord>::SIGN_BIT` -/
def u64_SIGN_BIT : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `core::num::U256_MASK` -/
def U256_MASK : FeM UInt256 := do
  return (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)

/-- `core::num::sign_extend` -/
def sign_extend (word : UInt256) (mask : UInt256) (sign_bit : UInt256) : FeM UInt256 := do
  let value := word &&& mask
  let v9 := value &&& sign_bit
  let v11 := decide (v9 ≠ (UInt256.ofNat 0))
  let join : UInt256 → FeM UInt256 := fun v3 => do
    return v3
  if v11 then
    let v12 ← U256_MASK
    let v14 := v12 ^^^ mask
    let v15 := value ||| v14
    join v15
  else
    join value

/-- `<u64 as IntWord>::from_word` -/
def u64_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := truncate 64 word
  return v1

/-- `<u256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u256_u64 (self : UInt256) : FeM UInt256 := do
  let word ← u256_to_word self
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    join v12

/-- `<CallData as ByteSource>::size` -/
def CallData_size (self : CallData) : FeM UInt256 := do
  let total ← calldatasize
  let v4 := self.base
  let v5 := decide (total.toNat < v4.toNat)
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v5 then
    join (UInt256.ofNat 0)
  else
    let v8 := self.base
    let v9 ← checkedSub total v8
    let v10 ← downcast_truncate_u256_u64 v9
    join v10

/-- `<CallData as ByteInput>::len` -/
def len_CallData (self : CallData) : FeM UInt256 := do
  let v1 ← CallData_size self
  let v2 := v1
  return v2

/-- `<i256 as IntWord>::from_word` -/
def i256_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := word
  return v1

/-- `<Sol as Abi>::validate_runtime_head` -/
def Sol_validate_runtime_head (input_len : UInt256) (base : UInt256) (head_size : UInt256) : FeM Unit := do
  let v5 := decide (head_size ≠ (UInt256.ofNat 0))
  let join : FeM Unit := do
    return ()
  if v5 then
    let v8 := input_len - base
    let args_len ← i256_from_word v8
    let v11 ← i256_from_word head_size
    let v12 := UInt256.sltBool args_len v11
    if v12 then
      let v13 ← revertEmpty
      nomatch v13
    else
      join
  else
    join

/-- `<u256 as AbiSize>::IS_DYNAMIC` -/
def u256_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<u256 as AbiSize>::HEAD_SIZE` -/
def u256_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `<u64 as Bounded>::max` -/
def u64_max : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551615)

/-- `<u64 as IntWord>::to_word` -/
def u64_to_word (self : UInt256) : FeM UInt256 := do
  let v2 := self
  return v2

/-- `<usize as IntWord>::IS_SIGNED` -/
def usize_IS_SIGNED : FeM Bool := do
  return false

/-- `<usize as IntWord>::MASK` -/
def usize_MASK : FeM UInt256 := do
  let v1 := ~~~(UInt256.ofNat 0)
  let v2 := v1
  return v2

/-- `<usize as IntWord>::from_word` -/
def usize_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := word
  return v1

/-- `core::num::I256_SIGN_BIT` -/
def I256_SIGN_BIT : FeM UInt256 := do
  return (UInt256.ofNat 57896044618658097711785492504343953926634992332820282019728792003956564819968)

/-- `<usize as IntWord>::SIGN_BIT` -/
def usize_SIGN_BIT : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<u64 as IntDowncast>::downcast_saturate` -/
def downcast_saturate_u64_usize (self : UInt256) : FeM UInt256 := do
  let mut unsigned_word : UInt256 := default
  let word ← u64_to_word self
  let v10 ← usize_IS_SIGNED
  let v11 := !v10
  let join : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v1 => do
    let mut unsigned_word := unsigned_word
    return v1
  if v11 then
    unsigned_word := word
    let v13 ← u64_IS_SIGNED
    let join_1 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      let unsigned_word_24 := unsigned_word
      let v26 ← usize_MASK
      let v27 := decide (v26.toNat < unsigned_word_24.toNat)
      let join_2 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v2 => do
        let mut unsigned_word := unsigned_word
        join unsigned_word v2
      if v27 then
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        join_2 unsigned_word v29
      else
        let unsigned_word_31 := unsigned_word
        let v33 ← usize_from_word unsigned_word_31
        join_2 unsigned_word v33
    let join_3 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      join_1 unsigned_word
    if v13 then
      let v14 ← I256_SIGN_BIT
      let v15 := word &&& v14
      let v17 := decide (v15 ≠ (UInt256.ofNat 0))
      if v17 then
        unsigned_word := (UInt256.ofNat 0)
        join_1 unsigned_word
      else
        join_3 unsigned_word
    else
      join_3 unsigned_word
  else
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let t_min_word ← sign_extend v36 v37 v38
    let v41 ← usize_SIGN_BIT
    let t_max_word ← checkedSub v41 (UInt256.ofNat 1)
    let t_min := t_min_word
    let t_max := t_max_word
    let v49 ← u64_IS_SIGNED
    let join_4 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v3 => do
      let mut unsigned_word := unsigned_word
      join unsigned_word v3
    if v49 then
      let x_50 := word
      let v52 := UInt256.sltBool x_50 t_min
      let join_5 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v4 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v4
      if v52 then
        let v53 ← usize_from_word t_min_word
        join_5 unsigned_word v53
      else
        let v55 := UInt256.sgtBool x_50 t_max
        let join_6 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v5 => do
          let mut unsigned_word := unsigned_word
          join_5 unsigned_word v5
        if v55 then
          let v56 ← usize_from_word t_max_word
          join_6 unsigned_word v56
        else
          let v58 ← usize_from_word word
          join_6 unsigned_word v58
    else
      let v62 ← I256_SIGN_BIT
      let v63 := word &&& v62
      let v65 := decide (v63 ≠ (UInt256.ofNat 0))
      let join_7 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v6 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v6
      if v65 then
        let v66 ← usize_from_word t_max_word
        join_7 unsigned_word v66
      else
        let x_68 := word
        let v70 := UInt256.sgtBool x_68 t_max
        let join_8 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v7 => do
          let mut unsigned_word := unsigned_word
          join_7 unsigned_word v7
        if v70 then
          let v71 ← usize_from_word t_max_word
          join_8 unsigned_word v71
        else
          let v73 ← usize_from_word word
          join_8 unsigned_word v73

/-- `core::ptr::MAX_ALLOCATION` -/
def MAX_ALLOCATION : FeM UInt256 := do
  let v0 ← u64_max
  let v1 ← downcast_saturate_u64_usize v0
  return v1

/-- `core::panics::PANIC_OUT_OF_MEMORY` -/
def PANIC_OUT_OF_MEMORY : FeM UInt256 := do
  return (UInt256.ofNat 65)

/-- `core::ptr::alloc_bytes` -/
def alloc_bytes (len : UInt256) : FeM UInt256 := do
  let v2 ← MAX_ALLOCATION
  let v3 := decide (v2.toNat < len.toNat)
  if v3 then
    let v4 ← PANIC_OUT_OF_MEMORY
    let v5 ← panicCode v4
    nomatch v5
  else
    let v8 ← alloc len
    return v8

/-- `FixedMemBuffer<32>::alloc_uninit` -/
def FixedMemBuffer_alloc_uninit_32 : FeM FixedMemBuffer_32 := do
  let v1 ← alloc_bytes (UInt256.ofNat 32)
  let v2 := (FixedMemBuffer_32.mk v1)
  return v2

/-- `core::ptr::zero_bytes` -/
def zero_bytes (ptr : UInt256) (len : UInt256) : FeM Unit := do
  let _ ← zeroMem ptr len
  return ()

/-- `FixedMemBuffer<32>::alloc` -/
def FixedMemBuffer_alloc_32 : FeM FixedMemBuffer_32 := do
  let buffer ← FixedMemBuffer_alloc_uninit_32
  let v2 := buffer.ptr
  let _ ← zero_bytes v2 (UInt256.ofNat 32)
  return buffer

/-- `FixedMemBuffer<32>::ptr` -/
def FixedMemBuffer_ptr_32 (self : FixedMemBuffer_32) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

/-- `AbiFrame::from_raw_parts` -/
def AbiFrame_from_raw_parts (ptr : UInt256) (len : UInt256) : FeM AbiFrame := do
  let v4 := (AbiFrame.mk ptr len)
  return v4

/-- `core::panics::PANIC_INDEX_OUT_OF_BOUNDS` -/
def PANIC_INDEX_OUT_OF_BOUNDS : FeM UInt256 := do
  return (UInt256.ofNat 50)

/-- `core::ptr::index_out_of_bounds` -/
def index_out_of_bounds : FeM Empty := do
  let v0 ← PANIC_INDEX_OUT_OF_BOUNDS
  let v1 ← panicCode v0
  nomatch v1

/-- `core::num::word_to_usize` -/
def word_to_usize (word : UInt256) : FeM UInt256 := do
  let v1 := word
  return v1

/-- `core::abi::ptr_offset` -/
def ptr_offset (ptr : UInt256) (offset : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize offset
  let v3 := ptr + v2
  return v3

/-- `AbiFrame::slice` -/
def AbiFrame_slice (self : AbiFrame) (offset : UInt256) (len : UInt256) : FeM AbiFrame := do
  let v4 := self.len
  let v5 := decide (v4.toNat < offset.toNat)
  let join : FeM AbiFrame := do
    let v17 ← index_out_of_bounds
    nomatch v17
  if v5 then
    join
  else
    let v7 := self.len
    let v9 := v7 - offset
    let v10 := decide (v9.toNat < len.toNat)
    if v10 then
      join
    else
      let v13 := self.ptr
      let v14 ← ptr_offset v13 offset
      let v16 := (AbiFrame.mk v14 len)
      return v16

/-- `core::ptr::cast<u8, u256>` -/
def cast_u8_u256 (ptr : UInt256) : FeM UInt256 := do
  let v2 := ptr
  return v2

/-- `core::abi::store_word` -/
def store_word (ptr : UInt256) (value : UInt256) : FeM Unit := do
  let v2 := value
  let v3 ← cast_u8_u256 ptr
  mstore v3 v2
  return ()

/-- `AbiFrame::store_word` -/
def AbiFrame_store_word (self : AbiFrame) («at» : UInt256) (value : UInt256) : FeM Unit := do
  let word ← AbiFrame_slice self «at» (UInt256.ofNat 32)
  let v6 := word.ptr
  let _ ← store_word v6 value
  return ()

/-- `<u256 as Encode>::encode` -/
def u256_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) self
  return ()

/-- `<u256 as IntDowncast>::downcast_unchecked` -/
def downcast_unchecked_u256_usize (self : UInt256) : FeM UInt256 := do
  let word ← u256_to_word self
  let v3 ← usize_from_word word
  return v3

/-- `MemSlice<u8>::from_raw_parts`, its `ref` body -/
def MemSlice_from_raw_parts_ref_u8 (ptr : UInt256) (len : UInt256) : FeM MemSlice_u8 := do
  let mut slice : MemSlice_u8 := default
  let slice_2 := (MemSlice_u8.mk ptr len)
  slice := slice_2
  return slice

/-- `MemSlice<u8>::ptr` -/
def MemSlice_ptr_u8 (self : MemSlice_u8) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

/-- `MemSlice<u8>::len` -/
def MemSlice_len_u8 (self : MemSlice_u8) : FeM UInt256 := do
  let v1 := self.len
  return v1

/-- `<Evm as RawOps>::return_data` -/
def Evm_return_data (data : MemSlice_u8) : FeM Empty := do
  let v2 ← MemSlice_ptr_u8 data
  let v3 ← MemSlice_len_u8 data
  let v4 := v3
  let v5 ← returnData v2 v4
  nomatch v5

/-- `FixedMemBuffer<64>::alloc_uninit` -/
def FixedMemBuffer_alloc_uninit_64 : FeM FixedMemBuffer_64 := do
  let v1 ← alloc_bytes (UInt256.ofNat 64)
  let v2 := (FixedMemBuffer_64.mk v1)
  return v2

/-- `FixedMemBuffer<64>::alloc` -/
def FixedMemBuffer_alloc_64 : FeM FixedMemBuffer_64 := do
  let buffer ← FixedMemBuffer_alloc_uninit_64
  let v2 := buffer.ptr
  let _ ← zero_bytes v2 (UInt256.ofNat 64)
  return buffer

/-- `FixedMemBuffer<64>::ptr` -/
def FixedMemBuffer_ptr_64 (self : FixedMemBuffer_64) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

/-- `MemBuffer::new_uninit` -/
def MemBuffer_new_uninit (len : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes len
  let v4 := (MemBuffer.mk v1 len len)
  return v4

/-- `MemBuffer::new` -/
def MemBuffer_new (len : UInt256) : FeM MemBuffer := do
  let buffer ← MemBuffer_new_uninit len
  let v3 := buffer.ptr
  let _ ← zero_bytes v3 len
  return buffer

/-- `MemBuffer::ptr` -/
def MemBuffer_ptr (self : MemBuffer) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

/-- `MemBuffer::span`, its `ref` body -/
def MemBuffer_span_ref (self : MemBuffer) : FeM MemSlice_u8 := do
  let v1 := self.ptr
  let v2 := self.len
  let v3 ← MemSlice_from_raw_parts_ref_u8 v1 v2
  return v3

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<u256>` -/
def const_u256 : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<u256>` -/
def payload_size_u256 (self : UInt256) : FeM UInt256 := do
  let v1 ← u256_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<u256>` -/
def abi_payload_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u256 value
  return v1

/-- `<usize as Bounded>::max` -/
def usize_max : FeM UInt256 := do
  let v1 := ~~~(UInt256.ofNat 0)
  return v1

/-- `core::ptr::allocation_len` -/
def allocation_len (len : UInt256) : FeM UInt256 := do
  let v2 ← usize_max
  let v3 := v2
  let v4 := decide (v3.toNat < len.toNat)
  if v4 then
    let v5 ← PANIC_OUT_OF_MEMORY
    let v6 ← panicCode v5
    nomatch v6
  else
    let v9 ← word_to_usize len
    return v9

/-- `core::abi::encode_alloc<Sol, u256>` -/
def encode_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  let total ← abi_payload_size_u256 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  let _ ← u256_encode_Sol value r1
  let buffer_11 := buffer
  return buffer_11

/-- `core::abi::abi_field_size<u256>` -/
def abi_field_size_u256 (value : UInt256) : FeM UInt256 := do
  let payload_size ← payload_size_u256 value
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<u256>` -/
def abi_single_root_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_u256 value
  return v1

/-- `core::abi::abi_field_head_size<u256>` -/
def abi_field_head_size_u256 : FeM UInt256 := do
  let v0 ← u256_IS_DYNAMIC
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← u256_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<u256>` -/
def abi_single_root_head_size_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

/-- `AbiFrame::tail` -/
def AbiFrame_tail (self : AbiFrame) (n : UInt256) : FeM AbiFrame := do
  let v3 := self.len
  let v4 := decide (v3.toNat < n.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v8 := self.ptr
    let v9 ← ptr_offset v8 n
    let v10 := self.len
    let v12 := v10 - n
    let v13 := (AbiFrame.mk v9 v12)
    return v13

/-- `AbiFrame::len` -/
def AbiFrame_len (self : AbiFrame) : FeM UInt256 := do
  let v1 := self.len
  return v1

/-- `core::abi::encode_field<Sol, u256>` -/
def encode_field_Sol_u256 (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let size ← payload_size_u256 value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← u256_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← u256_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← u256_encode_Sol value r2
    let tails_24 := tails
    return tails_24

/-- `core::abi::encode_single_root_alloc<Sol, u256>` -/
def encode_single_root_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let total ← abi_single_root_size_u256 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_u256
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_u256 value out (UInt256.ofNat 0) v13
  let buffer_16 := buffer
  return buffer_16

/-- `core::abi::encode_return_alloc<Sol, u256>` -/
def encode_return_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let v1 ← const_u256
  if v1 then
    let v2 ← encode_alloc_Sol_u256 value
    return v2
  else
    let v5 ← encode_single_root_alloc_Sol_u256 value
    return v5

/-- `<Evm as ContractHost>::return_bytes` -/
def Evm_return_bytes (data : MemSlice_u8) : FeM Empty := do
  let v2 ← Evm_return_data data
  nomatch v2

/-- `<Evm as ContractHost>::return_value<Sol, u256>` -/
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
    let v6 := decide (v4.toNat ≤ (UInt256.ofNat 32).toNat)
    if v6 then
      let out_7 ← FixedMemBuffer_alloc_32
      out_r0 := out_7
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← u256_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let _ ← u256_encode_Sol value r4
      let v16 ← FixedMemBuffer_ptr_32 out_r0
      let v17 ← u256_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← u256_HEAD_SIZE
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r1 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← u256_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let _ ← u256_encode_Sol value r5
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← u256_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← u256_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let out_44 ← MemBuffer_new v43
        out_r2 := out_44
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← u256_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let _ ← u256_encode_Sol value r6
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let out_57 ← encode_return_alloc_Sol_u256 value
    out_r3 := out_57
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

/-- `<CallData as ByteSource>::word_at` -/
def CallData_word_at (self : CallData) (byte_offset : UInt256) : FeM UInt256 := do
  let v2 := self.base
  let v4 := v2 + byte_offset
  let v5 ← calldataload v4
  return v5

/-- `<Sol as Abi>::decode_error` -/
def Sol_decode_error : FeM Empty := do
  let v0 ← revertEmpty
  nomatch v0

/-- `core::abi::checked_tail<Sol>` -/
def checked_tail_Sol (base : UInt256) (tail_rel : UInt256) : FeM UInt256 := do
  let v5 := ~~~(UInt256.ofNat 0)
  let v7 := v5 - base
  let v8 := decide (v7.toNat < tail_rel.toNat)
  if v8 then
    let v9 ← Sol_decode_error
    nomatch v9
  else
    let v14 := base + tail_rel
    return v14

/-- `core::abi::checked_frame_end<Sol>` -/
def checked_frame_end_Sol (pos : UInt256) (head_size : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v5 := decide (input_len.toNat < pos.toNat)
  let join : FeM UInt256 := do
    let v16 ← Sol_decode_error
    nomatch v16
  if v5 then
    join
  else
    let v9 := input_len - pos
    let v10 := decide (v9.toNat < head_size.toNat)
    if v10 then
      join
    else
      let v15 := pos + head_size
      return v15

/-- `<u256 as Decode>::decode_from<CallData>` -/
def u256_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM UInt256 := do
  let v2 ← CallData_word_at input pos
  return v2

/-- `core::abi::Decode::decode_from_bounded<u256, Sol, CallData>` -/
def decode_from_bounded_u256_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← u256_decode_from_Sol_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, u256, CallData>` -/
def decode_field_from_prechecked_head_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let tail_rel ← CallData_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← decode_from_bounded_u256_Sol_CallData input tail_at input_len
    return v9
  else
    let v12 ← u256_decode_from_Sol_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, u256, CallData>` -/
def decode_field_from_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM UInt256 := do
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    let tail_rel ← CallData_word_at input head_pos
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← u256_decode_from_Sol_CallData input v6
    return v7
  else
    let v10 ← u256_decode_from_Sol_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, u256, CallData>` -/
def decode_msg_field_from_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_u256_CallData input base head_pos input_len
    return v5
  else
    let v8 ← decode_field_from_Sol_u256_CallData input base head_pos
    return v8

/-- `SolDecoder::with_base` -/
def SolDecoder_with_base (base : UInt256) : FeM SolDecoder := do
  let v3 := (SolDecoder.mk base base)
  return v3

/-- `<Sol as Abi>::decoder_new` -/
def Sol_decoder_new : FeM SolDecoder := do
  let v1 ← SolDecoder_with_base (UInt256.ofNat 0)
  return v1

/-- `<usize as IntWord>::to_word` -/
def usize_to_word (self : UInt256) : FeM UInt256 := do
  let v2 := self
  return v2

/-- `<usize as IntDowncast>::downcast_truncate` -/
def downcast_truncate_usize_u64 (self : UInt256) : FeM UInt256 := do
  let word ← usize_to_word self
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    join v12

/-- `<MemSlice<u8> as ByteSource>::size` -/
def MemSlice_u8_size (self : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_len_u8 self
  let v2 ← downcast_truncate_usize_u64 v1
  return v2

/-- `<MemSlice<u8> as ByteInput>::len` -/
def len_MemSlice_u8 (self : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_u8_size self
  let v2 := v1
  return v2

/-- `<MemSlice<u8> as ByteSource>::word_at` -/
def MemSlice_u8_word_at (self : MemSlice_u8) (byte_offset : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_len_u8 self
  let len := v2
  let v6 := decide (len.toNat ≤ byte_offset.toNat)
  if v6 then
    return (UInt256.ofNat 0)
  else
    let v11 := len - byte_offset
    let available ← word_to_usize v11
    let v14 ← MemSlice_ptr_u8 self
    let v15 ← word_to_usize byte_offset
    let v16 := v14 + v15
    let v17 ← readPaddedWord v16 available
    return v17

/-- `<SolDecoder as AbiDecoder>::read_word<MemSlice<u8>>` -/
def SolDecoder_read_word_MemSlice_u8 (self : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut self := self
  let v2 := self.pos
  let new_pos ← checkedAdd v2 (UInt256.ofNat 32)
  let v6 := self.pos
  let v7 := decide (new_pos.toNat < v6.toNat)
  let join : SolDecoder → FeM (UInt256 × SolDecoder) := fun self => do
    let mut self := self
    let v16 ← revertEmpty
    nomatch v16
  if v7 then
    join self
  else
    let v8 ← len_MemSlice_u8 input
    let v9 := decide (v8.toNat < new_pos.toNat)
    if v9 then
      join self
    else
      let v12 := self.pos
      let w ← MemSlice_u8_word_at input v12
      self := { self with pos := new_pos }
      return (w, self)

/-- `<SolDecoder as AbiDecoder>::base` -/
def SolDecoder_base (self : SolDecoder) : FeM UInt256 := do
  let v1 := self.base
  return v1

/-- `<SolDecoder as AbiDecoder>::pos` -/
def SolDecoder_pos (self : SolDecoder) : FeM UInt256 := do
  let v1 := self.pos
  return v1

/-- `<SolDecoder as AbiDecoder>::set_base` -/
def SolDecoder_set_base (self : SolDecoder) (new_base : UInt256) : FeM (Unit × SolDecoder) := do
  let mut self := self
  self := { self with base := new_base }
  return ((), self)

/-- `<SolDecoder as AbiDecoder>::set_pos` -/
def SolDecoder_set_pos (self : SolDecoder) (new_pos : UInt256) : FeM (Unit × SolDecoder) := do
  let mut self := self
  self := { self with pos := new_pos }
  return ((), self)

/-- `<u256 as Decode>::decode_payload<SolDecoder, MemSlice<u8>>` -/
def u256_decode_payload_Sol_SolDecoder_MemSlice_u8 (d : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d := d
  let (v2, v2_0) ← SolDecoder_read_word_MemSlice_u8 d input
  d := v2_0
  return (v2, d)

/-- `core::abi::decode_field<Sol, u256, SolDecoder, MemSlice<u8>>` -/
def decode_field_Sol_u256_SolDecoder_MemSlice_u8 (d_0 : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d_0 := d_0
  let v2 ← u256_IS_DYNAMIC
  if v2 then
    let (tail_rel, tail_rel_0) ← SolDecoder_read_word_MemSlice_u8 d_0 input
    d_0 := tail_rel_0
    let parent_base ← SolDecoder_base d_0
    let parent_pos ← SolDecoder_pos d_0
    let v11 ← checked_tail_Sol parent_base tail_rel
    let (v12, v12_0) ← SolDecoder_set_base d_0 v11
    d_0 := v12_0
    let v14 ← SolDecoder_base d_0
    let (v15, v15_0) ← SolDecoder_set_pos d_0 v14
    d_0 := v15_0
    let (value, value_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 d_0 input
    d_0 := value_0
    let (v19, v19_0) ← SolDecoder_set_base d_0 parent_base
    d_0 := v19_0
    let (v20, v20_0) ← SolDecoder_set_pos d_0 parent_pos
    d_0 := v20_0
    return (value, d_0)
  else
    let (v24, v24_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 d_0 input
    d_0 := v24_0
    return (v24, d_0)

/-- `<(u256) as Decode>::decode_payload<SolDecoder, MemSlice<u8>>` -/
def tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 (d : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d := d
  let (v0, v0_0) ← decode_field_Sol_u256_SolDecoder_MemSlice_u8 d input
  d := v0_0
  let v5 := v0
  return (v5, d)

def Layout.StoragePackedArray_8 : Layout :=
  { storage := [0] }

structure Location where
  slot : UInt256
  offset : UInt256
deriving Inhabited

/-- `StoragePackedArray<8>::LANES_PER_SLOT` -/
def StoragePackedArray_LANES_PER_SLOT_8 : FeM UInt256 := do
  let v2 ← checkedDiv (UInt256.ofNat 256) (UInt256.ofNat 8)
  return v2

/-- `<u256 as StorageKey>::encoded_len` -/
def u256_encoded_len (self : UInt256) : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `<u256 as IntWord>::IS_SIGNED` -/
def u256_IS_SIGNED : FeM Bool := do
  return false

/-- `<u256 as IntDowncast>::downcast_saturate` -/
def downcast_saturate_u256_usize (self : UInt256) : FeM UInt256 := do
  let mut unsigned_word : UInt256 := default
  let word ← u256_to_word self
  let v10 ← usize_IS_SIGNED
  let v11 := !v10
  let join : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v1 => do
    let mut unsigned_word := unsigned_word
    return v1
  if v11 then
    unsigned_word := word
    let v13 ← u256_IS_SIGNED
    let join_1 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      let unsigned_word_24 := unsigned_word
      let v26 ← usize_MASK
      let v27 := decide (v26.toNat < unsigned_word_24.toNat)
      let join_2 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v2 => do
        let mut unsigned_word := unsigned_word
        join unsigned_word v2
      if v27 then
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        join_2 unsigned_word v29
      else
        let unsigned_word_31 := unsigned_word
        let v33 ← usize_from_word unsigned_word_31
        join_2 unsigned_word v33
    let join_3 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      join_1 unsigned_word
    if v13 then
      let v14 ← I256_SIGN_BIT
      let v15 := word &&& v14
      let v17 := decide (v15 ≠ (UInt256.ofNat 0))
      if v17 then
        unsigned_word := (UInt256.ofNat 0)
        join_1 unsigned_word
      else
        join_3 unsigned_word
    else
      join_3 unsigned_word
  else
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let t_min_word ← sign_extend v36 v37 v38
    let v41 ← usize_SIGN_BIT
    let t_max_word ← checkedSub v41 (UInt256.ofNat 1)
    let t_min := t_min_word
    let t_max := t_max_word
    let v49 ← u256_IS_SIGNED
    let join_4 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v3 => do
      let mut unsigned_word := unsigned_word
      join unsigned_word v3
    if v49 then
      let x_50 := word
      let v52 := UInt256.sltBool x_50 t_min
      let join_5 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v4 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v4
      if v52 then
        let v53 ← usize_from_word t_min_word
        join_5 unsigned_word v53
      else
        let v55 := UInt256.sgtBool x_50 t_max
        let join_6 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v5 => do
          let mut unsigned_word := unsigned_word
          join_5 unsigned_word v5
        if v55 then
          let v56 ← usize_from_word t_max_word
          join_6 unsigned_word v56
        else
          let v58 ← usize_from_word word
          join_6 unsigned_word v58
    else
      let v62 ← I256_SIGN_BIT
      let v63 := word &&& v62
      let v65 := decide (v63 ≠ (UInt256.ofNat 0))
      let join_7 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v6 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v6
      if v65 then
        let v66 ← usize_from_word t_max_word
        join_7 unsigned_word v66
      else
        let x_68 := word
        let v70 := UInt256.sgtBool x_68 t_max
        let join_8 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v7 => do
          let mut unsigned_word := unsigned_word
          join_7 unsigned_word v7
        if v70 then
          let v71 ← usize_from_word t_max_word
          join_8 unsigned_word v71
        else
          let v73 ← usize_from_word word
          join_8 unsigned_word v73

/-- `<u256 as StorageKey>::write_key` -/
def u256_write_key (ptr : UInt256) (self : UInt256) : FeM Unit := do
  let _ ← mstore ptr self
  return ()

/-- `std::evm::crypto::keccak256` -/
def keccak256_1 (data : MemSlice_u8) : FeM UInt256 := do
  let v1 ← MemSlice_ptr_u8 data
  let v2 ← MemSlice_len_u8 data
  let v3 := v2
  let v4 ← keccak256 v1 v3
  return v4

/-- `core::ptr::cast<u8, u8>` -/
def cast_u8_u8 (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `core::ptr::dealloc<u8>` -/
def dealloc_u8 (ptr : UInt256) : FeM Unit := do
  let _ ← cast_u8_u8 ptr
  let v2 := ()
  return ()

/-- `<MemBuffer as Drop>::drop` -/
def MemBuffer_drop (self : MemBuffer) : FeM Unit := do
  let v1 := self.ptr
  let _ ← dealloc_u8 v1
  return ()

/-- `std::evm::layout::mapping_slot<u256>` -/
def mapping_slot_u256 (root : UInt256) (key : UInt256) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let key_len ← u256_encoded_len key
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  buffer := buffer_7
  let scratch ← MemBuffer_ptr buffer
  let _ ← u256_write_key scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let _ ← mstore v14 root
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<u256, u256> as StateIndex>::locate` -/
def StorageMap_locate_u256_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let v3 ← mapping_slot_u256 root key
  let v5 := (Location.mk v3 (UInt256.ofNat 0))
  return v5

/-- `StoragePackedArray<8>::LANE_MASK` -/
def StoragePackedArray_LANE_MASK_8 : FeM UInt256 := do
  let v2 := (UInt256.ofNat 1) <<< (UInt256.ofNat 8)
  let v4 ← checkedSub v2 (UInt256.ofNat 1)
  return v4

/-- `StoragePackedArray<8>::get` -/
def StoragePackedArray_get_8 (self : Place) (i : UInt256) : FeM UInt256 := do
  let v3 ← StoragePackedArray_LANES_PER_SLOT_8
  let v4 ← checkedRem i v3
  let bit_pos := v4 * (UInt256.ofNat 8)
  let v9 ← StoragePackedArray_LANES_PER_SLOT_8
  let v10 ← checkedDiv i v9
  let l0 ← StorageMap_locate_u256_u256 (self.field Layout.StoragePackedArray_8 0) (self.field Layout.StoragePackedArray_8 0).slot v10
  let v11 ← load (Place.mk .storage l0.slot)
  let v12 := v11 >>> bit_pos
  let v13 ← StoragePackedArray_LANE_MASK_8
  let v14 := v12 &&& v13
  return v14

/-- `StoragePackedArray<8>::set` -/
def StoragePackedArray_set_8 (self : Place) (i : UInt256) (value : UInt256) : FeM Unit := do
  let v4 ← StoragePackedArray_LANES_PER_SLOT_8
  let v5 ← checkedRem i v4
  let bit_pos := v5 * (UInt256.ofNat 8)
  let v9 ← StoragePackedArray_LANE_MASK_8
  let lane_mask_shifted := v9 <<< bit_pos
  let v13 ← StoragePackedArray_LANES_PER_SLOT_8
  let v14 ← checkedDiv i v13
  let l0 ← StorageMap_locate_u256_u256 (self.field Layout.StoragePackedArray_8 0) (self.field Layout.StoragePackedArray_8 0).slot v14
  let v15 := (Place.mk .storage l0.slot)
  let word := v15
  let v17 ← load word
  let v18 ← U256_MASK
  let v19 := v18 ^^^ lane_mask_shifted
  let v20 := v17 &&& v19
  let v22 ← StoragePackedArray_LANE_MASK_8
  let v23 := value &&& v22
  let v24 := v23 <<< bit_pos
  let v25 := v20 ||| v24
  store word v25
  return ()

/-- `<Evm as Ctx>::caller` -/
def Evm_caller : FeM UInt256 := do
  let v1 ← caller
  let v2 := v1
  return v2

/-- `<Address as StorageKey>::encoded_len` -/
def Address_encoded_len (self : UInt256) : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `<Address as StorageKey>::write_key` -/
def Address_write_key (ptr : UInt256) (self : UInt256) : FeM Unit := do
  let v2 := self
  let _ ← u256_write_key ptr v2
  return ()

/-- `std::evm::layout::mapping_slot<Address>` -/
def mapping_slot_Address (root : UInt256) (key : UInt256) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let key_len ← Address_encoded_len key
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  buffer := buffer_7
  let scratch ← MemBuffer_ptr buffer
  let _ ← Address_write_key scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let _ ← mstore v14 root
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<Address, u256> as StateIndex>::locate` -/
def StorageMap_locate_Address_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let v3 ← mapping_slot_Address root key
  let v5 := (Location.mk v3 (UInt256.ofNat 0))
  return v5

/-- `Address::zero` -/
def Address_zero : FeM UInt256 := do
  let v1 := (UInt256.ofNat 0)
  return v1

/-- `<(u256) as AbiSize>::HEAD_SIZE` -/
def tuple_HEAD_SIZE_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

/-- `core::abi::dynamic_payload_size<u256>` -/
def dynamic_payload_size_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← u256_IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_u256 value
    return v2
  else
    return (UInt256.ofNat 0)

/-- `<(u256) as AbiSize>::payload_size` -/
def tuple_payload_size_u256 (self : UInt256) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE_u256
  let v2 := self
  let v3 ← dynamic_payload_size_u256 v2
  let v4 := v1 + v3
  return v4

/-- `core::abi::abi_payload_size<(u256)>` -/
def abi_payload_size_tuple_u256 (value : UInt256) : FeM UInt256 := do
  let v1 ← tuple_payload_size_u256 value
  return v1

/-- `<(u256) as Encode>::encode` -/
def tuple_encode_Sol_u256 (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let v4 ← tuple_HEAD_SIZE_u256
  let v5 ← AbiFrame_tail out v4
  let v0_6 := self
  let _ ← encode_field_Sol_u256 v0_6 out (UInt256.ofNat 0) v5
  return ()

/-- `core::abi::encode_alloc<Sol, (u256)>` -/
def encode_alloc_Sol_tuple_u256 (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let total ← abi_payload_size_tuple_u256 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let _ ← tuple_encode_Sol_u256 value r2
  let buffer_12 := buffer
  return buffer_12

/-- `std::evm::effects::encode_abi_payload<(u256)>` -/
def encode_abi_payload_tuple_u256 (value : UInt256) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_tuple_u256 value
  return v1

/-- `<Address as TopicValue>::as_topic` -/
def Address_as_topic (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

/-- `<Evm as Log>::log3` -/
def Evm_log3 (data : MemSlice_u8) (topic0 : UInt256) (topic1 : UInt256) (topic2 : UInt256) : FeM Unit := do
  let v5 ← MemSlice_ptr_u8 data
  let v6 ← MemSlice_len_u8 data
  let v7 := v6
  let _ ← logData v5 v7 #[topic0, topic1, topic2]
  return ()

/-- `<StorageMap<Address, StorageMap<Address, u256>> as StateIndex>::locate` -/
def StorageMap_locate_Address_StorageMap_Address_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  let v3 ← mapping_slot_Address root key
  let v5 := (Location.mk v3 (UInt256.ofNat 0))
  return v5

/-- `<Address as AbiSize>::IS_DYNAMIC` -/
def Address_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<Address as AbiSize>::HEAD_SIZE` -/
def Address_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `core::abi::abi_field_head_size<Address>` -/
def abi_field_head_size_Address : FeM UInt256 := do
  let v0 ← Address_IS_DYNAMIC
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← Address_HEAD_SIZE
    return v4

/-- `<Address as Decode>::decode_from<CallData>` -/
def Address_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM UInt256 := do
  let word ← CallData_word_at input pos
  let v5 := word >>> (UInt256.ofNat 160)
  let v7 := decide (v5 ≠ (UInt256.ofNat 0))
  if v7 then
    let v8 ← revertEmpty
    nomatch v8
  else
    let v11 := word
    return v11

/-- `core::abi::Decode::decode_from_bounded<Address, Sol, CallData>` -/
def decode_from_bounded_Address_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← Address_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← Address_decode_from_Sol_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, Address, CallData>` -/
def decode_field_from_prechecked_head_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    let tail_rel ← CallData_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← decode_from_bounded_Address_Sol_CallData input tail_at input_len
    return v9
  else
    let v12 ← Address_decode_from_Sol_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, Address, CallData>` -/
def decode_field_from_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM UInt256 := do
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let tail_rel ← CallData_word_at input head_pos
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← Address_decode_from_Sol_CallData input v6
    return v7
  else
    let v10 ← Address_decode_from_Sol_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, Address, CallData>` -/
def decode_msg_field_from_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_Address_CallData input base head_pos input_len
    return v5
  else
    let v8 ← decode_field_from_Sol_Address_CallData input base head_pos
    return v8

/-- `<bool as AbiSize>::IS_DYNAMIC` -/
def bool_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<bool as AbiSize>::HEAD_SIZE` -/
def bool_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `<bool as Encode>::encode` -/
def bool_encode_Sol (self : Bool) (out : AbiFrame) : FeM Unit := do
  let v4 := Bool.toUInt256 self
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) v4
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<bool>` -/
def const_bool : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<bool>` -/
def payload_size_bool (self : Bool) : FeM UInt256 := do
  let v1 ← bool_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<bool>` -/
def abi_payload_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← payload_size_bool value
  return v1

/-- `core::abi::encode_alloc<Sol, bool>` -/
def encode_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  let total ← abi_payload_size_bool value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  let _ ← bool_encode_Sol value r1
  let buffer_11 := buffer
  return buffer_11

/-- `core::abi::abi_field_size<bool>` -/
def abi_field_size_bool (value : Bool) : FeM UInt256 := do
  let payload_size ← payload_size_bool value
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<bool>` -/
def abi_single_root_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← abi_field_size_bool value
  return v1

/-- `core::abi::abi_field_head_size<bool>` -/
def abi_field_head_size_bool : FeM UInt256 := do
  let v0 ← bool_IS_DYNAMIC
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← bool_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<bool>` -/
def abi_single_root_head_size_bool : FeM UInt256 := do
  let v0 ← abi_field_head_size_bool
  return v0

/-- `core::abi::encode_field<Sol, bool>` -/
def encode_field_Sol_bool (value : Bool) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    let size ← payload_size_bool value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← bool_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← bool_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← bool_encode_Sol value r2
    let tails_24 := tails
    return tails_24

/-- `core::abi::encode_single_root_alloc<Sol, bool>` -/
def encode_single_root_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let total ← abi_single_root_size_bool value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_bool
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_bool value out (UInt256.ofNat 0) v13
  let buffer_16 := buffer
  return buffer_16

/-- `core::abi::encode_return_alloc<Sol, bool>` -/
def encode_return_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let v1 ← const_bool
  if v1 then
    let v2 ← encode_alloc_Sol_bool value
    return v2
  else
    let v5 ← encode_single_root_alloc_Sol_bool value
    return v5

/-- `<Evm as ContractHost>::return_value<Sol, bool>` -/
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
    let v6 := decide (v4.toNat ≤ (UInt256.ofNat 32).toNat)
    if v6 then
      let out_7 ← FixedMemBuffer_alloc_32
      out_r0 := out_7
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← bool_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let _ ← bool_encode_Sol value r4
      let v16 ← FixedMemBuffer_ptr_32 out_r0
      let v17 ← bool_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← bool_HEAD_SIZE
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r1 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← bool_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let _ ← bool_encode_Sol value r5
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← bool_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← bool_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let out_44 ← MemBuffer_new v43
        out_r2 := out_44
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← bool_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let _ ← bool_encode_Sol value r6
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let out_57 ← encode_return_alloc_Sol_bool value
    out_r3 := out_57
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

/-- `<(u256, u256) as AbiSize>::HEAD_SIZE` -/
def tuple_HEAD_SIZE_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  return v2

/-- `<(u256, u256) as Decode>::decode_from_prechecked_head<CallData>` -/
def tuple_decode_from_prechecked_head_Sol_u256_u256_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let mut head_pos : UInt256 := default
  head_pos := pos
  let head_pos_5 := head_pos
  let v0 ← decode_field_from_prechecked_head_Sol_u256_CallData input pos head_pos_5 input_len
  let v9 ← abi_field_head_size_u256
  let head_pos_10 := head_pos
  let head_pos_11 ← checkedAdd head_pos_10 v9
  head_pos := head_pos_11
  let head_pos_14 := head_pos
  let v1 ← decode_field_from_prechecked_head_Sol_u256_CallData input pos head_pos_14 input_len
  let v18 := (v0, v1)
  return v18

/-- `core::abi::decode_frame_from<Sol, (u256, u256), CallData>` -/
def decode_frame_from_Sol_tuple_u256_u256_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← tuple_HEAD_SIZE_u256_u256
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← tuple_decode_from_prechecked_head_Sol_u256_u256_CallData input pos input_len
  return v5

/-- `std::abi::sol::decode_input_at<(u256, u256), CallData>` -/
def decode_input_at_tuple_u256_u256_CallData (input : CallData) (base : UInt256) : FeM (UInt256 × UInt256) := do
  let input_len ← len_CallData input
  let v4 ← decode_frame_from_Sol_tuple_u256_u256_CallData input base input_len
  return v4

/-- `Evm::decode_args<(u256, u256)>` -/
def Evm_decode_args_tuple_u256_u256 : FeM (UInt256 × UInt256) := do
  let mut r0 : CallData := default
  let v1 ← Evm_input
  r0 := v1
  let v3 := (UInt256.ofNat 4)
  let v4 := r0
  let v5 ← decode_input_at_tuple_u256_u256_CallData v4 v3
  return v5

/-- `MemBuffer::len` -/
def MemBuffer_len (self : MemBuffer) : FeM UInt256 := do
  let v1 := self.len
  return v1

/-- `std::evm::effects::revert<u256>` -/
def revert_u256 (value : UInt256) : FeM Empty := do
  let mut result : MemBuffer := default
  let result_1 ← encode_single_root_alloc_Sol_u256 value
  result := result_1
  let v4 ← MemBuffer_ptr result
  let v6 ← MemBuffer_len result
  let v7 := v6
  let v8 ← revertData v4 v7
  nomatch v8

structure Wei where
  amount : UInt256
deriving Inhabited

/-- `<i256 as IntWord>::to_word` -/
def i256_to_word (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

/-- `<i8 as IntWord>::IS_SIGNED` -/
def i8_IS_SIGNED : FeM Bool := do
  return true

/-- `core::num::U8_MASK` -/
def U8_MASK : FeM UInt256 := do
  return (UInt256.ofNat 255)

/-- `<i8 as IntWord>::MASK` -/
def i8_MASK : FeM UInt256 := do
  let v0 ← U8_MASK
  return v0

/-- `core::num::I8_SIGN_BIT` -/
def I8_SIGN_BIT : FeM UInt256 := do
  return (UInt256.ofNat 128)

/-- `<i8 as IntWord>::SIGN_BIT` -/
def i8_SIGN_BIT : FeM UInt256 := do
  let v0 ← I8_SIGN_BIT
  return v0

/-- `<i8 as IntWord>::from_word` -/
def i8_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 8) word
  return v1

/-- `<i256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i256_i8 (self : UInt256) : FeM UInt256 := do
  let word ← i256_to_word self
  let v4 ← i8_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i8_MASK
    let v6 ← i8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i8_from_word v7
    join v8
  else
    let v10 ← i8_MASK
    let v11 := word &&& v10
    let v12 ← i8_from_word v11
    join v12

/-- `<u256 as IntDowncast>::downcast_unchecked` -/
def downcast_unchecked_u256_i256 (self : UInt256) : FeM UInt256 := do
  let word ← u256_to_word self
  let v3 ← i256_from_word word
  return v3

/-- `<Evm as Ctx>::address` -/
def Evm_address : FeM UInt256 := do
  let v1 ← selfAddress
  let v2 := v1
  return v2

/-- `Address::word` -/
def Address_word (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

/-- `std::evm::units::wei` -/
def wei (n : UInt256) : FeM Wei := do
  let v2 := (Wei.mk n)
  return v2

/-- `<Evm as Ctx>::value` -/
def Evm_value : FeM Wei := do
  let v1 ← callvalue
  let v2 ← wei v1
  return v2

/-- `Wei::as_wei` -/
def Wei_as_wei (self : Wei) : FeM UInt256 := do
  let v1 := self.amount
  return v1

/-- `<Evm as Ctx>::origin` -/
def Evm_origin : FeM UInt256 := do
  let v1 ← origin
  let v2 := v1
  return v2

/-- `<Evm as Ctx>::block_number` -/
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

/-- `<(u256, u256) as StorageKey>::encoded_len` -/
def tuple_encoded_len_u256_u256 (self : (UInt256 × UInt256)) : FeM UInt256 := do
  let v1 := self.fst
  let v2 ← u256_encoded_len v1
  let v3 := self.snd
  let v4 ← u256_encoded_len v3
  let v5 ← checkedAdd v2 v4
  return v5

/-- `<(u256, u256) as StorageKey>::write_key` -/
def tuple_write_key_u256_u256 (ptr : UInt256) (self : (UInt256 × UInt256)) : FeM Unit := do
  let v2 := self.fst
  let a_len ← u256_encoded_len v2
  let v5 := self.fst
  let _ ← u256_write_key ptr v5
  let v7 ← downcast_unchecked_u256_usize a_len
  let v8 := ptr + v7
  let v9 := self.snd
  let _ ← u256_write_key v8 v9
  return ()

/-- `std::evm::layout::mapping_slot<(u256, u256)>` -/
def mapping_slot_tuple_u256_u256 (root : UInt256) (key : (UInt256 × UInt256)) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let key_len ← tuple_encoded_len_u256_u256 key
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  buffer := buffer_7
  let scratch ← MemBuffer_ptr buffer
  let _ ← tuple_write_key_u256_u256 scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let _ ← mstore v14 root
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<(u256, u256), bool> as StateIndex>::locate` -/
def StorageMap_locate_tuple_u256_u256_bool (self : Place) (root : UInt256) (key : (UInt256 × UInt256)) : FeM Location := do
  let v3 ← mapping_slot_tuple_u256_u256 root key
  let v5 := (Location.mk v3 (UInt256.ofNat 0))
  return v5

/-- `<bool as StorageKey>::encoded_len` -/
def bool_encoded_len (self : Bool) : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `<bool as StorageKey>::write_key` -/
def bool_write_key (ptr : UInt256) (self : Bool) : FeM Unit := do
  let join : UInt256 → FeM Unit := fun v2 => do
    let _ ← u256_write_key ptr v2
    return ()
  if self then
    join (UInt256.ofNat 1)
  else
    join (UInt256.ofNat 0)

/-- `std::evm::layout::mapping_slot<bool>` -/
def mapping_slot_bool (root : UInt256) (key : Bool) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  let key_len ← bool_encoded_len key
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  buffer := buffer_7
  let scratch ← MemBuffer_ptr buffer
  let _ ← bool_write_key scratch key
  let v13 ← downcast_unchecked_u256_usize key_len
  let v14 := scratch + v13
  let _ ← mstore v14 root
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<bool, u256> as StateIndex>::locate` -/
def StorageMap_locate_bool_u256 (self : Place) (root : UInt256) (key : Bool) : FeM Location := do
  let v3 ← mapping_slot_bool root key
  let v5 := (Location.mk v3 (UInt256.ofNat 0))
  return v5

/-- `std::evm::effects::RawOps::calldata_ptr<Evm, [u256; 2]>` -/
def calldata_ptr_Evm_array_u256_2 (offset : UInt256) : FeM UInt256 := do
  let v3 := offset
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

/-- `<[u256; 2] as Collection>::start` -/
def array_start_u256_2 (self : (Vector UInt256 2)) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<[u256; 2] as Collection>::has` -/
def array_has_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM Bool := do
  let v4 := decide (c.toNat < (UInt256.ofNat 2).toNat)
  return v4

/-- `<[u256; 2] as Collection>::at`, its `ref` body -/
def array_at_ref_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM UInt256 := do
  checkIndex c 2
  return self[c.toNat]!

/-- `<usize as WrappingAdd>::wrapping_add` -/
def usize_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self + other
  return v2

/-- `<[u256; 2] as Collection>::next` -/
def array_next_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
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

/-- `DynString::span`, its `ref` body -/
def DynString_span_ref (self : DynString) : FeM MemSlice_u8 := do
  let v1 := self.data
  let v3 := v1 + (UInt256.ofNat 32)
  let v4 := self.len
  let v5 ← MemSlice_from_raw_parts_ref_u8 v3 v4
  return v5

/-- `core::panics::PANIC_STORAGE_ENCODING` -/
def PANIC_STORAGE_ENCODING : FeM UInt256 := do
  return (UInt256.ofNat 34)

/-- `Panic::from` -/
def Panic_from (code : UInt256) : FeM Panic := do
  let v2 := (Panic.mk code)
  return v2

/-- `<Panic as AbiSize>::HEAD_SIZE` -/
def Panic_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Panic as AbiSize>::payload_size` -/
def Panic_payload_size (self : Panic) : FeM UInt256 := do
  let v1 ← Panic_HEAD_SIZE
  let v2 := self.code
  let v3 ← dynamic_payload_size_u256 v2
  let v4 ← checkedAdd v1 v3
  return v4

/-- `core::abi::abi_payload_size<Panic>` -/
def abi_payload_size_Panic (value : Panic) : FeM UInt256 := do
  let v1 ← Panic_payload_size value
  return v1

/-- `MemBuffer::set_len` -/
def MemBuffer_set_len (self : MemBuffer) (len_1 : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let v3 := self.capacity
  let v4 := decide (v3.toNat < len_1.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    self := { self with len := len_1 }
    return ((), self)

/-- `<Panic as Encode>::encode` -/
def Panic_encode (self : Panic) (v1 : AbiFrame) : FeM Unit := do
  let mut r0 : AbiFrame := default
  let mut r1 : UInt256 := default
  let v2 ← Panic_HEAD_SIZE
  let v3 ← AbiFrame_tail v1 v2
  r0 := v3
  r1 := (UInt256.ofNat 0)
  let v7 := self.code
  let v8 := r1
  let v10 := r0
  let _ ← encode_field_Sol_u256 v7 v1 v8 v10
  return ()

/-- `std::evm::effects::encode_calldata<Panic>` -/
def encode_calldata_Panic (selector : UInt256) (message : Panic) : FeM MemBuffer := do
  let mut message := message
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let selector_size := (UInt256.ofNat 4)
  let payload_size ← abi_payload_size_Panic message
  let v7 ← downcast_unchecked_u256_usize payload_size
  let len := selector_size + v7
  let v10 ← Panic_HEAD_SIZE
  let v12 := decide (v10.toNat < (UInt256.ofNat 28).toNat)
  let join : Panic → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun message buffer r2 capacity => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    let buffer_19 ← MemBuffer_new_uninit capacity
    buffer := buffer_19
    let v22 ← MemBuffer_ptr buffer
    let payload := v22 + selector_size
    let (v26, v26_0) ← MemBuffer_set_len buffer len
    buffer := v26_0
    let v28 ← MemBuffer_ptr buffer
    let v30 := selector
    let v32 := v30 <<< (UInt256.ofNat 224)
    let _ ← mstore v28 v32
    let v34 ← downcast_unchecked_u256_usize payload_size
    let _ ← zero_bytes payload v34
    let v36 ← AbiFrame_from_raw_parts payload payload_size
    r2 := v36
    let _ ← Panic_encode message r2
    let buffer_40 := buffer
    return buffer_40
  let join_1 : Panic → MemBuffer → AbiFrame → FeM MemBuffer := fun message buffer r2 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    join message buffer r2 len
  if v12 then
    let v14 := decide (len.toNat < (UInt256.ofNat 32).toNat)
    if v14 then
      join message buffer r2 (UInt256.ofNat 32)
    else
      join_1 message buffer r2
  else
    join_1 message buffer r2

/-- `std::evm::effects::encode_error<Panic>` -/
def encode_error_Panic (error : Panic) : FeM MemBuffer := do
  let v1 := (UInt256.ofNat 1313373041) -- a constant, as the compiler evaluates it
  let v3 ← encode_calldata_Panic v1 error
  return v3

/-- `std::evm::effects::revert_error<Panic>` -/
def revert_error_Panic (error : Panic) : FeM Empty := do
  let mut buffer : MemBuffer := default
  let buffer_2 ← encode_error_Panic error
  buffer := buffer_2
  let v5 ← MemBuffer_ptr buffer
  let v7 ← MemBuffer_len buffer
  let v8 := v7
  let v9 ← revertData v5 v8
  nomatch v9

/-- `std::evm::crypto::keccak_words<1>` -/
def keccak_words_1 (words : (Vector UInt256 1)) : FeM UInt256 := do
  let v1 := keccakWords words
  return v1

/-- `std::evm::layout::array_data_slot` -/
def array_data_slot (root : UInt256) : FeM UInt256 := do
  let mut r0 : (Vector UInt256 1) := default
  let v2 := #v[root]
  r0 := v2
  let v4 ← keccak_words_1 r0
  return v4

/-- `<u256 as WrappingAdd>::wrapping_add` -/
def u256_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self + other
  return v2

/-- `<u256 as Bounded>::max` -/
def u256_max : FeM UInt256 := do
  return (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935)

/-- `<DynString as Drop>::drop` -/
def DynString_drop (self : DynString) : FeM Unit := do
  let v1 := self.data
  let _ ← dealloc_u8 v1
  return ()

/-- `<u8 as SolPacked>::BITS` -/
def u8_BITS : FeM UInt256 := do
  return (UInt256.ofNat 8)

/-- `std::evm::sol_slot::field_shift` -/
def field_shift (offset : UInt256) (bits : UInt256) : FeM UInt256 := do
  let v4 := decide ((UInt256.ofNat 32).toNat ≤ offset.toNat)
  let join : FeM UInt256 := do
    let v17 ← revertEmpty
    nomatch v17
  if v4 then
    join
  else
    let v9 ← checkedMul offset (UInt256.ofNat 8)
    let v10 ← checkedSub (UInt256.ofNat 256) v9
    let v11 := decide (v10.toNat < bits.toNat)
    if v11 then
      join
    else
      let v16 ← checkedMul offset (UInt256.ofNat 8)
      return v16

/-- `<u8 as SolPacked>::to_packed` -/
def u8_to_packed (self : UInt256) : FeM UInt256 := do
  let v1 := self
  return v1

/-- `std::evm::sol_slot::field_mask` -/
def field_mask (bits : UInt256) : FeM UInt256 := do
  let v4 := decide ((UInt256.ofNat 256).toNat ≤ bits.toNat)
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u256_max
    join v5
  else
    let v10 := (UInt256.ofNat 1) <<< bits
    let v12 ← checkedSub v10 (UInt256.ofNat 1)
    join v12

/-- `SolWord::set<u8>` -/
def SolWord_set_u8 (self : SolWord) (offset : UInt256) (value : UInt256) : FeM (Unit × SolWord) := do
  let mut self := self
  let v3 ← u8_BITS
  let shift ← field_shift offset v3
  let v6 ← u8_BITS
  let v7 ← field_mask v6
  let mask := v7 <<< shift
  let v10 := self.word
  let v11 := ~~~mask
  let v12 := v10 &&& v11
  let v13 ← u8_to_packed value
  let v14 := v13 <<< shift
  let v15 := v14 &&& mask
  let v16 := v12 ||| v15
  self := { self with word := v16 }
  return ((), self)

/-- `core::abi::round_up_words` -/
def round_up_words (len : UInt256) : FeM UInt256 := do
  let v3 := decide (len = (UInt256.ofNat 0))
  if v3 then
    return (UInt256.ofNat 0)
  else
    let v9 ← checkedSub len (UInt256.ofNat 1)
    let v11 ← checkedDiv v9 (UInt256.ofNat 32)
    let v13 ← checkedAdd v11 (UInt256.ofNat 1)
    let v15 ← checkedMul v13 (UInt256.ofNat 32)
    return v15

/-- `core::abi::mem_word_at` -/
def mem_word_at (base : UInt256) (offset : UInt256) : FeM UInt256 := do
  let v2 ← word_to_usize offset
  let v3 := base + v2
  let v4 ← cast_u8_u256 v3
  let v5 ← mload v4
  let v6 := v5
  return v6

/-- `DynString::from_validated_abi_tail` -/
def DynString_from_validated_abi_tail (data : UInt256) (len : UInt256) (size : UInt256) : FeM DynString := do
  let v6 := (DynString.mk data len size)
  return v6

/-- `DynString::from_abi_tail` -/
def DynString_from_abi_tail (data : UInt256) (len : UInt256) (size : UInt256) : FeM DynString := do
  let v5 := len
  let v6 ← round_up_words v5
  let expected_size ← checkedAdd (UInt256.ofNat 32) v6
  let v10 := decide (size ≠ expected_size)
  let join : FeM DynString := do
    let v19 ← trap
    nomatch v19
  if v10 then
    join
  else
    let v12 ← mem_word_at data (UInt256.ofNat 0)
    let v14 := len
    let v15 := decide (v12 ≠ v14)
    if v15 then
      join
    else
      let v18 ← DynString_from_validated_abi_tail data len size
      return v18

/-- `<Evm as Ctx>::selfbalance` -/
def Evm_selfbalance : FeM Wei := do
  let v1 ← selfbalance
  let v2 ← wei v1
  return v2

/-- `<Address as Eq>::eq` -/
def Address_eq (self : UInt256) (other : UInt256) : FeM Bool := do
  let v2 := self
  let v3 := other
  let v4 := decide (v2 = v3)
  return v4

/-- `core::ops::Eq::ne<Address, Address>` -/
def ne_Address_Address (self : UInt256) (other : UInt256) : FeM Bool := do
  let v2 ← Address_eq self other
  let v3 := !v2
  return v3

/-- `MemBuffer::with_capacity` -/
def MemBuffer_with_capacity (capacity : UInt256) : FeM MemBuffer := do
  let v1 ← alloc_bytes capacity
  let v4 := (MemBuffer.mk v1 (UInt256.ofNat 0) capacity)
  return v4

/-- `MemBuffer::empty` -/
def MemBuffer_empty : FeM MemBuffer := do
  let v1 ← MemBuffer_with_capacity (UInt256.ofNat 0)
  return v1

/-- `MemSlice<u8>::empty` -/
def MemSlice_empty_u8 : FeM MemSlice_u8 := do
  let v1 ← alloc_bytes (UInt256.ofNat 0)
  let v2 ← cast_u8_u8 v1
  let v4 ← MemSlice_from_raw_parts_ref_u8 v2 (UInt256.ofNat 0)
  return v4

/-- `MemBuffer::capacity` -/
def MemBuffer_capacity (self : MemBuffer) : FeM UInt256 := do
  let v1 := self.capacity
  return v1

/-- `std::evm::effects::finish_raw_call` -/
def finish_raw_call (success : Bool) (ret_1 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut ret_1 := ret_1
  let returndata_len ← returndatasize
  let v6 ← MemBuffer_capacity ret_1
  let v7 := v6
  let v8 := decide (returndata_len.toNat < v7.toNat)
  let join : MemBuffer → UInt256 → FeM (RawCallOutcome × MemBuffer) := fun ret_1 written => do
    let mut ret_1 := ret_1
    let (v15, v15_0) ← MemBuffer_set_len ret_1 written
    ret_1 := v15_0
    let v17 := (RawCallOutcome.mk success returndata_len)
    return (v17, ret_1)
  if v8 then
    let v9 ← downcast_unchecked_u256_usize returndata_len
    join ret_1 v9
  else
    let v12 ← MemBuffer_capacity ret_1
    join ret_1 v12

/-- `std::evm::effects::Call::raw_call<Evm>` -/
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
  let success := decide (v16 ≠ (UInt256.ofNat 0))
  let (v21, v21_0) ← finish_raw_call success ret_5
  ret_5 := v21_0
  return (v21, ret_5)

/-- `std::evm::effects::Call::send_value<Evm>` -/
def send_value_Evm (addr : UInt256) (gas_1 : UInt256) (value : Wei) : FeM RawCallOutcome := do
  let mut addr := addr
  let mut ret : MemBuffer := default
  let ret_3 ← MemBuffer_empty
  ret := ret_3
  let v5 ← MemSlice_empty_u8
  let addr_7 := addr
  let (v9, v9_0) ← raw_call_Evm addr_7 gas_1 value v5 ret
  ret := v9_0
  let _ ← MemBuffer_drop ret
  return v9

/-- `RawCallOutcome::success` -/
def RawCallOutcome_success (self : RawCallOutcome) : FeM Bool := do
  let v1 := self.success
  return v1

/-- `<Evm as Log>::log2` -/
def Evm_log2 (data : MemSlice_u8) (topic0 : UInt256) (topic1 : UInt256) : FeM Unit := do
  let v4 ← MemSlice_ptr_u8 data
  let v5 ← MemSlice_len_u8 data
  let v6 := v5
  let _ ← logData v4 v6 #[topic0, topic1]
  return ()

/-- `<u8 as IntWord>::IS_SIGNED` -/
def u8_IS_SIGNED : FeM Bool := do
  return false

/-- `<u8 as IntWord>::MASK` -/
def u8_MASK : FeM UInt256 := do
  let v0 ← U8_MASK
  return v0

/-- `<u8 as IntWord>::SIGN_BIT` -/
def u8_SIGN_BIT : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<u8 as IntWord>::from_word` -/
def u8_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := truncate 8 word
  return v1

/-- `<u256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u256_u8 (self : UInt256) : FeM UInt256 := do
  let word ← u256_to_word self
  let v4 ← u8_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u8_from_word v7
    join v8
  else
    let v10 ← u8_MASK
    let v11 := word &&& v10
    let v12 ← u8_from_word v11
    join v12

/-- `<u8 as SolPacked>::from_packed` -/
def u8_from_packed (bits : UInt256) : FeM UInt256 := do
  let v1 ← downcast_truncate_u256_u8 bits
  return v1

/-- `SolWord::get<u8>` -/
def SolWord_get_u8 (self : SolWord) (offset : UInt256) : FeM UInt256 := do
  let v2 ← u8_BITS
  let shift ← field_shift offset v2
  let v5 := self.word
  let v6 := v5 >>> shift
  let v7 ← u8_BITS
  let v8 ← field_mask v7
  let v9 := v6 &&& v8
  let v10 ← u8_from_packed v9
  return v10

/-- `<DynString as AbiSize>::IS_DYNAMIC` -/
def DynString_IS_DYNAMIC : FeM Bool := do
  return true

/-- `<DynString as AbiSize>::HEAD_SIZE` -/
def DynString_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `DynString::encoded_span` -/
def DynString_encoded_span (self : DynString) : FeM MemSlice_u8 := do
  let v1 := self.data
  let v2 := self.size
  let v3 ← word_to_usize v2
  let v4 ← MemSlice_from_raw_parts_ref_u8 v1 v3
  return v4

/-- `core::ptr::copy_raw` -/
def copy_raw (dest : UInt256) (source : UInt256) (len : UInt256) : FeM Unit := do
  let _ ← mcopy dest source len
  return ()

/-- `core::ptr::copy` -/
def copy (dest : UInt256) (source : MemSlice_u8) : FeM Unit := do
  let v2 ← MemSlice_ptr_u8 source
  let v3 ← MemSlice_len_u8 source
  let _ ← copy_raw dest v2 v3
  return ()

/-- `AbiFrame::copy_from` -/
def AbiFrame_copy_from (self : AbiFrame) («at» : UInt256) (source : MemSlice_u8) : FeM Unit := do
  let v3 ← MemSlice_len_u8 source
  let v4 := v3
  let dest ← AbiFrame_slice self «at» v4
  let v7 := dest.ptr
  let _ ← copy v7 source
  return ()

/-- `<DynString as Encode>::encode` -/
def DynString_encode_Sol (self : DynString) (out : AbiFrame) : FeM Unit := do
  let v3 ← DynString_encoded_span self
  let _ ← AbiFrame_copy_from out (UInt256.ofNat 0) v3
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<DynString>` -/
def const_DynString : FeM Bool := do
  return false

/-- `DynString::encoded_size` -/
def DynString_encoded_size (self : DynString) : FeM UInt256 := do
  let v1 := self.size
  return v1

/-- `<DynString as AbiSize>::payload_size` -/
def DynString_payload_size (self : DynString) : FeM UInt256 := do
  let v1 ← DynString_encoded_size self
  return v1

/-- `core::abi::abi_payload_size<DynString>` -/
def abi_payload_size_DynString (value : DynString) : FeM UInt256 := do
  let v1 ← DynString_payload_size value
  return v1

/-- `core::abi::encode_alloc<Sol, DynString>` -/
def encode_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let total ← abi_payload_size_DynString value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let _ ← DynString_encode_Sol value r2
  let _ ← DynString_drop value
  let buffer_12 := buffer
  return buffer_12

/-- `core::abi::abi_field_size<DynString>` -/
def abi_field_size_DynString (value : DynString) : FeM UInt256 := do
  let payload_size ← DynString_payload_size value
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<DynString>` -/
def abi_single_root_size_DynString (value : DynString) : FeM UInt256 := do
  let v1 ← abi_field_size_DynString value
  return v1

/-- `core::abi::abi_field_head_size<DynString>` -/
def abi_field_head_size_DynString : FeM UInt256 := do
  let v0 ← DynString_IS_DYNAMIC
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← DynString_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<DynString>` -/
def abi_single_root_head_size_DynString : FeM UInt256 := do
  let v0 ← abi_field_head_size_DynString
  return v0

/-- `core::abi::encode_field<Sol, DynString>` -/
def encode_field_Sol_DynString (value : DynString) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    let size ← DynString_payload_size value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← DynString_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← DynString_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← DynString_encode_Sol value r2
    let tails_24 := tails
    return tails_24

/-- `core::abi::encode_single_root_alloc<Sol, DynString>` -/
def encode_single_root_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let total ← abi_single_root_size_DynString value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_DynString
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_DynString value out (UInt256.ofNat 0) v13
  let _ ← DynString_drop value
  let buffer_17 := buffer
  return buffer_17

/-- `core::abi::encode_return_alloc<Sol, DynString>` -/
def encode_return_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let v1 ← const_DynString
  if v1 then
    let v3 ← encode_alloc_Sol_DynString value
    return v3
  else
    let v7 ← encode_single_root_alloc_Sol_DynString value
    return v7

/-- `<Evm as ContractHost>::return_value<Sol, DynString>` -/
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
    let v5 := decide (v3.toNat ≤ (UInt256.ofNat 32).toNat)
    if v5 then
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← DynString_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let _ ← DynString_encode_Sol value r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← DynString_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← DynString_HEAD_SIZE
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← DynString_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let _ ← DynString_encode_Sol value r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← DynString_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← DynString_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← DynString_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let _ ← DynString_encode_Sol value r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let out_61 ← encode_return_alloc_Sol_DynString value_59
    out_r4 := out_61
    let v64 ← MemBuffer_span_ref out_r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

/-- `<u8 as AbiSize>::IS_DYNAMIC` -/
def u8_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<u8 as AbiSize>::HEAD_SIZE` -/
def u8_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `<u8 as Encode>::encode` -/
def u8_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let v4 := self
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) v4
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<u8>` -/
def const_u8 : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<u8>` -/
def payload_size_u8 (self : UInt256) : FeM UInt256 := do
  let v1 ← u8_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<u8>` -/
def abi_payload_size_u8 (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_u8 value
  return v1

/-- `core::abi::encode_alloc<Sol, u8>` -/
def encode_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  let total ← abi_payload_size_u8 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  let _ ← u8_encode_Sol value r1
  let buffer_11 := buffer
  return buffer_11

/-- `core::abi::abi_field_size<u8>` -/
def abi_field_size_u8 (value : UInt256) : FeM UInt256 := do
  let payload_size ← payload_size_u8 value
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<u8>` -/
def abi_single_root_size_u8 (value : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_u8 value
  return v1

/-- `core::abi::abi_field_head_size<u8>` -/
def abi_field_head_size_u8 : FeM UInt256 := do
  let v0 ← u8_IS_DYNAMIC
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← u8_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<u8>` -/
def abi_single_root_head_size_u8 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u8
  return v0

/-- `core::abi::encode_field<Sol, u8>` -/
def encode_field_Sol_u8 (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    let size ← payload_size_u8 value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← u8_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← u8_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← u8_encode_Sol value r2
    let tails_24 := tails
    return tails_24

/-- `core::abi::encode_single_root_alloc<Sol, u8>` -/
def encode_single_root_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let total ← abi_single_root_size_u8 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_u8
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_u8 value out (UInt256.ofNat 0) v13
  let buffer_16 := buffer
  return buffer_16

/-- `core::abi::encode_return_alloc<Sol, u8>` -/
def encode_return_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let v1 ← const_u8
  if v1 then
    let v2 ← encode_alloc_Sol_u8 value
    return v2
  else
    let v5 ← encode_single_root_alloc_Sol_u8 value
    return v5

/-- `<Evm as ContractHost>::return_value<Sol, u8>` -/
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
    let v6 := decide (v4.toNat ≤ (UInt256.ofNat 32).toNat)
    if v6 then
      let out_7 ← FixedMemBuffer_alloc_32
      out_r0 := out_7
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← u8_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let _ ← u8_encode_Sol value r4
      let v16 ← FixedMemBuffer_ptr_32 out_r0
      let v17 ← u8_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← u8_HEAD_SIZE
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r1 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← u8_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let _ ← u8_encode_Sol value r5
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← u8_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        let v42 ← u8_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let out_44 ← MemBuffer_new v43
        out_r2 := out_44
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← u8_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        let _ ← u8_encode_Sol value r6
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    let out_57 ← encode_return_alloc_Sol_u8 value
    out_r3 := out_57
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

structure CallOutcome where
  success : Bool
  returndata : MemBuffer
deriving Inhabited

/-- `Wei::zero` -/
def Wei_zero : FeM Wei := do
  let v1 := (Wei.mk (UInt256.ofNat 0))
  return v1

/-- `<() as AbiSize>::IS_DYNAMIC` -/
def tuple_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<() as AbiSize>::HEAD_SIZE` -/
def tuple_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `core::abi::AbiSize::payload_size<()>` -/
def payload_size_tuple (self : Unit) : FeM UInt256 := do
  let v1 ← tuple_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<()>` -/
def abi_payload_size_tuple (value : Unit) : FeM UInt256 := do
  let v1 ← payload_size_tuple value
  return v1

/-- `<() as Encode>::encode` -/
def tuple_encode_Sol (self : Unit) (out : AbiFrame) : FeM Unit := do
  return ()

/-- `core::abi::encode_alloc<Sol, ()>` -/
def encode_alloc_Sol_tuple (value : Unit) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let value_0 := value
  let total ← abi_payload_size_tuple value_0
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let value_9 := value
  let _ ← tuple_encode_Sol value_9 r2
  let buffer_12 := buffer
  return buffer_12

/-- `std::evm::effects::encode_abi_payload<()>` -/
def encode_abi_payload_tuple (value : Unit) : FeM MemBuffer := do
  let v1 ← encode_alloc_Sol_tuple value
  return v1

/-- `std::evm::effects::Create::create2_raw<Evm>` -/
def create2_raw_Evm (value : Wei) (init_code : MemSlice_u8) (salt : UInt256) : FeM UInt256 := do
  let v4 ← Wei_as_wei value
  let v5 ← MemSlice_ptr_u8 init_code
  let v6 ← MemSlice_len_u8 init_code
  let v7 := v6
  let v8 ← create2 v4 v5 v7 salt
  let v9 := v8
  return v9

/-- `std::evm::effects::copy_returndata` -/
def copy_returndata : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let ret_len ← returndatasize
  let v2 ← downcast_unchecked_u256_usize ret_len
  let buffer_3 ← MemBuffer_new_uninit v2
  buffer := buffer_3
  let v6 ← MemBuffer_ptr buffer
  let _ ← returndatacopy v6 (UInt256.ofNat 0) ret_len
  let buffer_9 := buffer
  return buffer_9

/-- `std::evm::effects::bubble_last_revert` -/
def bubble_last_revert : FeM Empty := do
  let mut ret : MemBuffer := default
  let ret_0 ← copy_returndata
  ret := ret_0
  let v3 ← MemBuffer_ptr ret
  let v5 ← MemBuffer_len ret
  let v6 := v5
  let v7 ← revertData v3 v6
  nomatch v7

/-- `std::evm::effects::copy_call_outcome` -/
def copy_call_outcome («raw» : RawCallOutcome) : FeM CallOutcome := do
  let v1 ← RawCallOutcome_success «raw»
  let v2 ← copy_returndata
  let v3 := (CallOutcome.mk v1 v2)
  return v3

/-- `<u256 as Decode>::decode_from<MemSlice<u8>>` -/
def u256_decode_from_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_u8_word_at input pos
  return v2

/-- `core::abi::Decode::decode_from_prechecked_head<u256, Sol, MemSlice<u8>>` -/
def decode_from_prechecked_head_u256_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_decode_from_Sol_MemSlice_u8 input pos
  return v3

/-- `core::abi::decode_frame_from<Sol, u256, MemSlice<u8>>` -/
def decode_frame_from_Sol_u256_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← decode_from_prechecked_head_u256_Sol_MemSlice_u8 input pos input_len
  return v5

/-- `core::abi::Decode::decode_from_bounded<u256, Sol, MemSlice<u8>>` -/
def decode_from_bounded_u256_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v3 ← u256_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← u256_decode_from_Sol_MemSlice_u8 input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, u256, MemSlice<u8>>` -/
def decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 (input : MemSlice_u8) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    let tail_rel ← MemSlice_u8_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← decode_from_bounded_u256_Sol_MemSlice_u8 input tail_at input_len
    return v9
  else
    let v12 ← u256_decode_from_Sol_MemSlice_u8 input head_pos
    return v12

/-- `std::abi::sol::decode_output_at<u256, MemSlice<u8>>` -/
def decode_output_at_u256_MemSlice_u8 (input : MemSlice_u8) (base : UInt256) : FeM UInt256 := do
  let input_len ← len_MemSlice_u8 input
  let v4 ← const_u256
  if v4 then
    let v5 ← decode_frame_from_Sol_u256_MemSlice_u8 input base input_len
    return v5
  else
    let v8 ← abi_single_root_head_size_u256
    let _ ← checked_frame_end_Sol base v8 input_len
    let v10 ← decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 input base base input_len
    return v10

/-- `std::abi::sol::decode_output<u256, MemSlice<u8>>` -/
def decode_output_u256_MemSlice_u8 (input : MemSlice_u8) : FeM UInt256 := do
  let v2 ← decode_output_at_u256_MemSlice_u8 input (UInt256.ofNat 0)
  return v2

/-- `std::evm::effects::decode_returndata<u256>` -/
def decode_returndata_u256 (ret : MemSlice_u8) : FeM UInt256 := do
  let v1 ← decode_output_u256_MemSlice_u8 ret
  return v1

/-- `CallOutcome::decode<u256>` -/
def CallOutcome_decode_u256 (self : CallOutcome) : FeM UInt256 := do
  let mut returndata : MemBuffer := default
  let success := self.success
  let returndata_3 := self.returndata
  returndata := returndata_3
  let v5 := !success
  if v5 then
    let v7 ← MemBuffer_ptr returndata
    let v9 ← MemBuffer_len returndata
    let v10 := v9
    let v11 ← revertData v7 v10
    nomatch v11
  else
    let v15 ← MemBuffer_span_ref returndata
    let v17 ← decode_returndata_u256 v15
    let _ ← MemBuffer_drop returndata
    return v17

/-- `std::evm::crypto::keccak_words<2>` -/
def keccak_words_2 (words : (Vector UInt256 2)) : FeM UInt256 := do
  let v1 := keccakWords words
  return v1

/-- `std::evm::effects::read_transient_word<Evm>` -/
def read_transient_word_Evm (slot : UInt256) : FeM UInt256 := do
  let v2 := slot
  let v3 ← load (Place.mk .transient v2)
  return v3

/-- `std::evm::effects::storage_at<Evm, u256>`, its `mut` body -/
def storage_at_mut_Evm_u256 (slot : UInt256) : FeM Place := do
  let v1 := slot
  let v2 := (Place.mk .storage v1)
  return v2

/-- `<Address as Encode>::encode` -/
def Address_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  let v3 := self
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) v3
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<Address>` -/
def const_Address : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<Address>` -/
def payload_size_Address (self : UInt256) : FeM UInt256 := do
  let v1 ← Address_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<Address>` -/
def abi_payload_size_Address (value : UInt256) : FeM UInt256 := do
  let v1 ← payload_size_Address value
  return v1

/-- `core::abi::encode_alloc<Sol, Address>` -/
def encode_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let value_0 := value
  let total ← abi_payload_size_Address value_0
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let value_9 := value
  let _ ← Address_encode_Sol value_9 r2
  let buffer_12 := buffer
  return buffer_12

/-- `core::abi::abi_field_size<Address>` -/
def abi_field_size_Address (value : UInt256) : FeM UInt256 := do
  let payload_size ← payload_size_Address value
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<Address>` -/
def abi_single_root_size_Address (value : UInt256) : FeM UInt256 := do
  let v1 ← abi_field_size_Address value
  return v1

/-- `core::abi::abi_single_root_head_size<Address>` -/
def abi_single_root_head_size_Address : FeM UInt256 := do
  let v0 ← abi_field_head_size_Address
  return v0

/-- `core::abi::encode_field<Sol, Address>` -/
def encode_field_Sol_Address (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    let size ← payload_size_Address value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← Address_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← Address_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← Address_encode_Sol value r2
    let tails_24 := tails
    return tails_24

/-- `core::abi::encode_single_root_alloc<Sol, Address>` -/
def encode_single_root_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let value_0 := value
  let total ← abi_single_root_size_Address value_0
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_Address
  let v13 ← AbiFrame_tail out v11
  let value_14 := value
  let _ ← encode_field_Sol_Address value_14 out (UInt256.ofNat 0) v13
  let buffer_17 := buffer
  return buffer_17

/-- `core::abi::encode_return_alloc<Sol, Address>` -/
def encode_return_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let v1 ← const_Address
  if v1 then
    let v2 ← encode_alloc_Sol_Address value
    return v2
  else
    let v5 ← encode_single_root_alloc_Sol_Address value
    return v5

/-- `<Evm as ContractHost>::return_value<Sol, Address>` -/
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
    let v5 := decide (v3.toNat ≤ (UInt256.ofNat 32).toNat)
    if v5 then
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← Address_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let value_12 := value
      let _ ← Address_encode_Sol value_12 r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← Address_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← Address_HEAD_SIZE
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← Address_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let value_32 := value
        let _ ← Address_encode_Sol value_32 r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← Address_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← Address_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← Address_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let value_51 := value
        let _ ← Address_encode_Sol value_51 r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let out_60 ← encode_return_alloc_Sol_Address value_59
    out_r4 := out_60
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

/-- `core::assert` -/
def «assert» (condition : Bool) : FeM Unit := do
  let v2 := !condition
  if v2 then
    let v3 ← trap
    nomatch v3
  else
    return ()

/-- `<u128 as IntWord>::to_word` -/
def u128_to_word (self : UInt256) : FeM UInt256 := do
  let v2 := self
  return v2

/-- `<u128 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u128_u64 (self : UInt256) : FeM UInt256 := do
  let word ← u128_to_word self
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    join v12

/-- `<u64 as WrappingMul>::wrapping_mul` -/
def u64_wrapping_mul (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := IntTy.mulW (.u 64) self other
  return v2

/-- `<u128 as WrappingSub>::wrapping_sub` -/
def u128_wrapping_sub (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := IntTy.subW (.u 128) self other
  return v2

/-- `<i64 as IntWord>::IS_SIGNED` -/
def i64_IS_SIGNED : FeM Bool := do
  return true

/-- `<i64 as IntWord>::MASK` -/
def i64_MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

/-- `core::num::I64_SIGN_BIT` -/
def I64_SIGN_BIT : FeM UInt256 := do
  return (UInt256.ofNat 9223372036854775808)

/-- `<i64 as IntWord>::SIGN_BIT` -/
def i64_SIGN_BIT : FeM UInt256 := do
  let v0 ← I64_SIGN_BIT
  return v0

/-- `<i64 as IntWord>::from_word` -/
def i64_from_word (word : UInt256) : FeM UInt256 := do
  let v1 := IntTy.cast (.u 256) (.i 64) word
  return v1

/-- `<u64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u64_i64 (self : UInt256) : FeM UInt256 := do
  let word ← u64_to_word self
  let v4 ← i64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i64_from_word v7
    join v8
  else
    let v10 ← i64_MASK
    let v11 := word &&& v10
    let v12 ← i64_from_word v11
    join v12

/-- `<i64 as IntWord>::to_word` -/
def i64_to_word (self : UInt256) : FeM UInt256 := do
  let v2 := self
  let v3 := v2
  return v3

/-- `<i64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i64_u64 (self : UInt256) : FeM UInt256 := do
  let word ← i64_to_word self
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    join v12

/-- `<i128 as IntWord>::to_word` -/
def i128_to_word (self : UInt256) : FeM UInt256 := do
  let v2 := self
  let v3 := v2
  return v3

/-- `<i128 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i128_u64 (self : UInt256) : FeM UInt256 := do
  let word ← i128_to_word self
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    let v10 ← u64_MASK
    let v11 := word &&& v10
    let v12 ← u64_from_word v11
    join v12

/-- `<i128 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i128_i64 (self : UInt256) : FeM UInt256 := do
  let word ← i128_to_word self
  let v4 ← i64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i64_from_word v7
    join v8
  else
    let v10 ← i64_MASK
    let v11 := word &&& v10
    let v12 ← i64_from_word v11
    join v12

/-- `<[usize; 3] as Collection>::start` -/
def array_start_usize_3 (self : (Vector UInt256 3)) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<[usize; 3] as Collection>::has` -/
def array_has_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM Bool := do
  let v4 := decide (c.toNat < (UInt256.ofNat 3).toNat)
  return v4

/-- `<[usize; 3] as Collection>::at`, its `ref` body -/
def array_at_ref_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM UInt256 := do
  checkIndex c 3
  return self[c.toNat]!

/-- `<Range<Known<1>, Known<4>> as Collection>::start` -/
def Range_start_1_4 (self : Range_Known_1_Known_4) : FeM UInt256 := do
  return (UInt256.ofNat 1)

/-- `core::range::range_has` -/
def range_has (c : UInt256) (lo : UInt256) (hi : UInt256) : FeM Bool := do
  let v6 := decide (lo.toNat ≤ c.toNat)
  let join : Bool → FeM Bool := fun v3 => do
    return v3
  let join_1 : FeM Bool := do
    join false
  if v6 then
    let v9 := decide (c.toNat < hi.toNat)
    if v9 then
      join true
    else
      join_1
  else
    join_1

/-- `<Range<Known<1>, Known<4>> as Collection>::has` -/
def Range_has_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 1) (UInt256.ofNat 4)
  return v4

/-- `<Range<Known<1>, Known<4>> as Collection>::at` -/
def Range_at_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<1>, Known<4>> as Collection>::next` -/
def Range_next_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<[usize; 3] as Collection>::next` -/
def array_next_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<Range<Known<1>, Known<16>> as Collection>::start` -/
def Range_start_1_16 (self : Range_Known_1_Known_16) : FeM UInt256 := do
  return (UInt256.ofNat 1)

/-- `<Range<Known<1>, Known<16>> as Collection>::has` -/
def Range_has_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 1) (UInt256.ofNat 16)
  return v4

/-- `<Range<Known<1>, Known<16>> as Collection>::at` -/
def Range_at_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<1>, Known<16>> as Collection>::next` -/
def Range_next_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<u256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u256_usize (self : UInt256) : FeM UInt256 := do
  let word ← u256_to_word self
  let v4 ← usize_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    join v8
  else
    let v10 ← usize_MASK
    let v11 := word &&& v10
    let v12 ← usize_from_word v11
    join v12

/-- `<(bool, u256, u256) as AbiSize>::IS_DYNAMIC` -/
def tuple_IS_DYNAMIC_bool_u256_u256 : FeM Bool := do
  let v1 ← bool_IS_DYNAMIC
  let join : Bool → FeM Bool := fun v0 => do
    return v0
  let join_1 : FeM Bool := do
    join true
  if v1 then
    join_1
  else
    let v2 ← u256_IS_DYNAMIC
    if v2 then
      join_1
    else
      let v3 ← u256_IS_DYNAMIC
      if v3 then
        join_1
      else
        join false

/-- `<(bool, u256, u256) as AbiSize>::HEAD_SIZE` -/
def tuple_HEAD_SIZE_bool_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_bool
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  return v4

/-- `<(bool, u256, u256) as Encode>::encode` -/
def tuple_encode_Sol_bool_u256_u256 (self : (Bool × UInt256 × UInt256)) (out : AbiFrame) : FeM Unit := do
  let mut tails : AbiFrame := default
  let v5 ← tuple_HEAD_SIZE_bool_u256_u256
  let tails_6 ← AbiFrame_tail out v5
  tails := tails_6
  let head1 ← abi_field_head_size_bool
  let v10 ← abi_field_head_size_u256
  let head2 := head1 + v10
  let tails_14 := tails
  let v0_16 := self.fst
  let tails_17 ← encode_field_Sol_bool v0_16 out (UInt256.ofNat 0) tails_14
  tails := tails_17
  let tails_20 := tails
  let v1_22 := self.snd.fst
  let tails_23 ← encode_field_Sol_u256 v1_22 out head1 tails_20
  tails := tails_23
  let tails_26 := tails
  let v2_28 := self.snd.snd
  let _ ← encode_field_Sol_u256 v2_28 out head2 tails_26
  return ()

/-- `<(bool, u256, u256) as AbiSize>::IS_PARAM_LIST` -/
def tuple_IS_PARAM_LIST_bool_u256_u256 : FeM Bool := do
  return true

/-- `core::abi::dynamic_payload_size<bool>` -/
def dynamic_payload_size_bool (value : Bool) : FeM UInt256 := do
  let v1 ← bool_IS_DYNAMIC
  if v1 then
    let v2 ← payload_size_bool value
    return v2
  else
    return (UInt256.ofNat 0)

/-- `<(bool, u256, u256) as AbiSize>::payload_size` -/
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

/-- `core::abi::abi_payload_size<(bool, u256, u256)>` -/
def abi_payload_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_bool_u256_u256 value
  return v1

/-- `core::abi::encode_alloc<Sol, (bool, u256, u256)>` -/
def encode_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let total ← abi_payload_size_tuple_bool_u256_u256 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let _ ← tuple_encode_Sol_bool_u256_u256 value r2
  let buffer_12 := buffer
  return buffer_12

/-- `core::abi::abi_field_size<(bool, u256, u256)>` -/
def abi_field_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let payload_size ← tuple_payload_size_bool_u256_u256 value
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<(bool, u256, u256)>` -/
def abi_single_root_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 ← abi_field_size_tuple_bool_u256_u256 value
  return v1

/-- `core::abi::abi_field_head_size<(bool, u256, u256)>` -/
def abi_field_head_size_tuple_bool_u256_u256 : FeM UInt256 := do
  let v0 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← tuple_HEAD_SIZE_bool_u256_u256
    return v4

/-- `core::abi::abi_single_root_head_size<(bool, u256, u256)>` -/
def abi_single_root_head_size_tuple_bool_u256_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_tuple_bool_u256_u256
  return v0

/-- `core::abi::encode_field<Sol, (bool, u256, u256)>` -/
def encode_field_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    let size ← tuple_payload_size_bool_u256_u256 value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← tuple_encode_Sol_bool_u256_u256 value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← tuple_HEAD_SIZE_bool_u256_u256
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← tuple_encode_Sol_bool_u256_u256 value r2
    let tails_24 := tails
    return tails_24

/-- `core::abi::encode_single_root_alloc<Sol, (bool, u256, u256)>` -/
def encode_single_root_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let total ← abi_single_root_size_tuple_bool_u256_u256 value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_tuple_bool_u256_u256
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_tuple_bool_u256_u256 value out (UInt256.ofNat 0) v13
  let buffer_17 := buffer
  return buffer_17

/-- `core::abi::encode_return_alloc<Sol, (bool, u256, u256)>` -/
def encode_return_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let v1 ← tuple_IS_PARAM_LIST_bool_u256_u256
  if v1 then
    let v2 ← encode_alloc_Sol_tuple_bool_u256_u256 value
    return v2
  else
    let v5 ← encode_single_root_alloc_Sol_tuple_bool_u256_u256 value
    return v5

/-- `<Evm as ContractHost>::return_value<Sol, (bool, u256, u256)>` -/
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
    let v5 := decide (v3.toNat ≤ (UInt256.ofNat 32).toNat)
    if v5 then
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← tuple_HEAD_SIZE_bool_u256_u256
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let _ ← tuple_encode_Sol_bool_u256_u256 value r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← tuple_HEAD_SIZE_bool_u256_u256
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← tuple_HEAD_SIZE_bool_u256_u256
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← tuple_HEAD_SIZE_bool_u256_u256
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let _ ← tuple_encode_Sol_bool_u256_u256 value r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← tuple_HEAD_SIZE_bool_u256_u256
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← tuple_HEAD_SIZE_bool_u256_u256
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← tuple_HEAD_SIZE_bool_u256_u256
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let _ ← tuple_encode_Sol_bool_u256_u256 value r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let out_60 ← encode_return_alloc_Sol_tuple_bool_u256_u256 value_59
    out_r4 := out_60
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

/-- `<Evm as RawOps>::calldatasize` -/
def Evm_calldatasize : FeM UInt256 := do
  let v1 ← calldatasize
  return v1

/-- `std::evm::effects::prepare_copy_destination` -/
def prepare_copy_destination (dest_0 : MemBuffer) (dest_offset : UInt256) (len : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut dest_0 := dest_0
  let v5 ← MemBuffer_capacity dest_0
  let v6 := v5
  let v7 := decide (v6.toNat < dest_offset.toNat)
  let join : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
    let mut dest_0 := dest_0
    let v44 ← trap
    nomatch v44
  if v7 then
    join dest_0
  else
    let v10 ← MemBuffer_capacity dest_0
    let v11 := v10
    let v13 ← checkedSub v11 dest_offset
    let v14 := decide (v13.toNat < len.toNat)
    if v14 then
      join dest_0
    else
      let v19 := decide (len ≠ (UInt256.ofNat 0))
      let join_1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
        let mut dest_0 := dest_0
        let v41 ← MemBuffer_ptr dest_0
        let v42 ← downcast_unchecked_u256_usize dest_offset
        let v43 := v41 + v42
        return (v43, dest_0)
      let join_2 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
        let mut dest_0 := dest_0
        join_1 dest_0
      if v19 then
        let v22 ← MemBuffer_len dest_0
        let v23 := v22
        let v24 := decide (v23.toNat < dest_offset.toNat)
        if v24 then
          let v26 ← MemBuffer_ptr dest_0
          let v28 ← MemBuffer_len dest_0
          let v29 := v26 + v28
          let v32 ← MemBuffer_len dest_0
          let v33 := v32
          let v34 ← checkedSub dest_offset v33
          let v35 ← downcast_unchecked_u256_usize v34
          let _ ← zero_bytes v29 v35
          join_1 dest_0
        else
          join_2 dest_0
      else
        join_2 dest_0

/-- `<Evm as RawOps>::calldatacopy` -/
def Evm_calldatacopy (dest_1 : MemBuffer) (dest_offset : UInt256) (src_offset : UInt256) (len : UInt256) : FeM (Unit × MemBuffer) := do
  let mut dest_1 := dest_1
  let (dest_ptr, dest_ptr_0) ← prepare_copy_destination dest_1 dest_offset len
  dest_1 := dest_ptr_0
  let v10 := decide (len = (UInt256.ofNat 0))
  if v10 then
    return ((), dest_1)
  else
    let _ ← calldatacopy dest_ptr src_offset len
    let v16 ← checkedAdd dest_offset len
    let «end» ← downcast_unchecked_u256_usize v16
    let v20 ← MemBuffer_len dest_1
    let v21 := decide (v20.toNat < «end».toNat)
    let join : MemBuffer → FeM (Unit × MemBuffer) := fun dest_1 => do
      let mut dest_1 := dest_1
      return ((), dest_1)
    if v21 then
      let (v22, v22_0) ← MemBuffer_set_len dest_1 «end»
      dest_1 := v22_0
      join dest_1
    else
      join dest_1

/-- `MemSlice<u8>::value_ptr<u64>` -/
def MemSlice_u8_value_ptr_u64 (self : MemSlice_u8) (offset : UInt256) : FeM UInt256 := do
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let join : FeM UInt256 := do
    let v14 ← index_out_of_bounds
    nomatch v14
  if v4 then
    join
  else
    let v6 := self.len
    let v8 ← checkedSub v6 offset
    let v9 := decide (v8.toNat < (UInt256.ofNat 8).toNat)
    if v9 then
      join
    else
      let v12 := self.ptr
      let v13 := v12 + offset
      return v13

/-- `core::ptr::cast<u8, u64>` -/
def cast_u8_u64 (ptr : UInt256) : FeM UInt256 := do
  let v2 := ptr
  return v2

/-- `MemSlice<u8>::read_le<u64>` -/
def MemSlice_u8_read_le_u64 (self : MemSlice_u8) (offset : UInt256) : FeM UInt256 := do
  let v2 ← MemSlice_u8_value_ptr_u64 self offset
  let v3 ← cast_u8_u64 v2
  let v4 ← mloadInt (.u 64) v3
  let v5 := IntTy.swapBytes (.u 64) v4
  return v5

/-- `MemBuffer::read_le<u64>` -/
def MemBuffer_read_le_u64 (self : MemBuffer) (offset : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_span_ref self
  let v4 ← MemSlice_u8_read_le_u64 v2 offset
  return v4

/-- `<Range<Unknown, Unknown> as Collection>::start` -/
def Range_Unknown_Unknown_start (self : Range_Unknown_Unknown) : FeM UInt256 := do
  let v1 := self.start
  return v1

/-- `<Range<Unknown, Unknown> as Collection>::has` -/
def Range_Unknown_Unknown_has (self : Range_Unknown_Unknown) (c : UInt256) : FeM Bool := do
  let v2 := self.start
  let v3 := self.«end»
  let v4 ← range_has c v2 v3
  return v4

/-- `<Range<Unknown, Unknown> as Collection>::at` -/
def Range_Unknown_Unknown_at (self : Range_Unknown_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Unknown, Unknown> as Collection>::next` -/
def Range_Unknown_Unknown_next (self : Range_Unknown_Unknown) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<MemBuffer as Index>::index`, its `ref` body -/
def MemBuffer_index_ref (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  let v2 := self.len
  let v3 := decide (v2.toNat ≤ index.toNat)
  if v3 then
    let v4 ← index_out_of_bounds
    nomatch v4
  else
    let v7 := self.ptr
    let v8 := v7 + index
    let v9 := v8
    return v9

/-- `MemBuffer::extend_over<u256>` -/
def MemBuffer_extend_over_u256 (self : MemBuffer) (offset : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut self := self
  let size := (UInt256.ofNat 32)
  let v5 := self.capacity
  let v6 := decide (v5.toNat < offset.toNat)
  let join : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
    let mut self := self
    let v38 ← index_out_of_bounds
    nomatch v38
  if v6 then
    join self
  else
    let v7 := self.capacity
    let v9 := v7 - offset
    let v10 := decide (v9.toNat < size.toNat)
    if v10 then
      join self
    else
      let v14 := self.len
      let v15 := decide (v14.toNat < offset.toNat)
      let join_1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
        let mut self := self
        let v27 := offset + size
        let v28 := self.len
        let v29 := decide (v28.toNat < v27.toNat)
        let join_2 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
          let mut self := self
          let v36 := self.ptr
          let v37 := v36 + offset
          return (v37, self)
        if v29 then
          let v31 := offset + size
          self := { self with len := v31 }
          join_2 self
        else
          join_2 self
      if v15 then
        let v16 := self.ptr
        let v17 := self.len
        let v18 := v16 + v17
        let v20 := self.len
        let v21 := offset - v20
        let _ ← zero_bytes v18 v21
        join_1 self
      else
        join_1 self

/-- `MemBuffer::write_be<u256>` -/
def MemBuffer_write_be_u256 (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let v3 := value
  let (v4, v4_0) ← MemBuffer_extend_over_u256 self offset
  self := v4_0
  let v5 ← cast_u8_u256 v4
  mstore v5 v3
  return ((), self)

/-- `MemBuffer::write_word` -/
def MemBuffer_write_word (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let (v3, v3_0) ← MemBuffer_write_be_u256 self offset value
  self := v3_0
  return ((), self)

/-- `<Evm as RawOps>::revert` -/
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

/-- `<Range<Known<0>, Known<8>> as Collection>::start` -/
def Range_start_0_8 (self : Range_Known_0_Known_8) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<Range<Known<0>, Known<8>> as Collection>::has` -/
def Range_has_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 8)
  return v4

/-- `<Range<Known<0>, Known<8>> as Collection>::at` -/
def Range_at_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<8>> as Collection>::next` -/
def Range_next_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<Range<Known<0>, Known<16>> as Collection>::start` -/
def Range_start_0_16 (self : Range_Known_0_Known_16) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<Range<Known<0>, Known<16>> as Collection>::has` -/
def Range_has_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 16)
  return v4

/-- `<Range<Known<0>, Known<16>> as Collection>::at` -/
def Range_at_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<16>> as Collection>::next` -/
def Range_next_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<u64 as WrappingAdd>::wrapping_add` -/
def u64_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := IntTy.addW (.u 64) self other
  return v2

/-- `<u64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u64_usize (self : UInt256) : FeM UInt256 := do
  let word ← u64_to_word self
  let v4 ← usize_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    join v8
  else
    let v10 ← usize_MASK
    let v11 := word &&& v10
    let v12 ← usize_from_word v11
    join v12

/-- `MemBuffer::extend_over<u64>` -/
def MemBuffer_extend_over_u64 (self : MemBuffer) (offset : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut self := self
  let size := (UInt256.ofNat 8)
  let v5 := self.capacity
  let v6 := decide (v5.toNat < offset.toNat)
  let join : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
    let mut self := self
    let v38 ← index_out_of_bounds
    nomatch v38
  if v6 then
    join self
  else
    let v7 := self.capacity
    let v9 := v7 - offset
    let v10 := decide (v9.toNat < size.toNat)
    if v10 then
      join self
    else
      let v14 := self.len
      let v15 := decide (v14.toNat < offset.toNat)
      let join_1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
        let mut self := self
        let v27 := offset + size
        let v28 := self.len
        let v29 := decide (v28.toNat < v27.toNat)
        let join_2 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
          let mut self := self
          let v36 := self.ptr
          let v37 := v36 + offset
          return (v37, self)
        if v29 then
          let v31 := offset + size
          self := { self with len := v31 }
          join_2 self
        else
          join_2 self
      if v15 then
        let v16 := self.ptr
        let v17 := self.len
        let v18 := v16 + v17
        let v20 := self.len
        let v21 := offset - v20
        let _ ← zero_bytes v18 v21
        join_1 self
      else
        join_1 self

/-- `MemBuffer::write_le<u64>` -/
def MemBuffer_write_le_u64 (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  let v3 := IntTy.swapBytes (.u 64) value
  let (v4, v4_0) ← MemBuffer_extend_over_u64 self offset
  self := v4_0
  let v5 ← cast_u8_u64 v4
  mstoreInt (.u 64) v5 v3
  return ((), self)

structure Range_Known_0_Unknown where
  start : Unit
  «end» : UInt256
deriving Inhabited

structure Range_Known_0_Known_64 where
  start : Unit
  «end» : Unit
deriving Inhabited

/-- `<Range<Known<0>, Unknown> as Collection>::start` -/
def Range_start_0 (self : Range_Known_0_Unknown) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<Range<Known<0>, Unknown> as Collection>::has` -/
def Range_has_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM Bool := do
  let v3 := self.«end»
  let v4 ← range_has c (UInt256.ofNat 0) v3
  return v4

/-- `<Range<Known<0>, Unknown> as Collection>::at` -/
def Range_at_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<64>> as Collection>::start` -/
def Range_start_0_64 (self : Range_Known_0_Known_64) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<Range<Known<0>, Known<64>> as Collection>::has` -/
def Range_has_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 64)
  return v4

/-- `<Range<Known<0>, Known<64>> as Collection>::at` -/
def Range_at_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<64>> as Collection>::next` -/
def Range_next_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<usize as IntDowncast>::downcast_unchecked` -/
def downcast_unchecked_usize_u64 (self : UInt256) : FeM UInt256 := do
  let word ← usize_to_word self
  let v3 ← u64_from_word word
  return v3

/-- `<u64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u64_u8 (self : UInt256) : FeM UInt256 := do
  let word ← u64_to_word self
  let v4 ← u8_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u8_from_word v7
    join v8
  else
    let v10 ← u8_MASK
    let v11 := word &&& v10
    let v12 ← u8_from_word v11
    join v12

/-- `<Range<Known<0>, Unknown> as Collection>::next` -/
def Range_next_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<[u32; 5] as Collection>::start` -/
def array_start_u32_5 (self : (Vector UInt256 5)) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<[u32; 5] as Collection>::has` -/
def array_has_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM Bool := do
  let v4 := decide (c.toNat < (UInt256.ofNat 5).toNat)
  return v4

/-- `<[u32; 5] as Collection>::at`, its `ref` body -/
def array_at_ref_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM UInt256 := do
  checkIndex c 5
  return self[c.toNat]!

/-- `<[u32; 5] as Collection>::next` -/
def array_next_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<u256 as WrappingSub>::wrapping_sub` -/
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

/-- `core::ptr::try_alloc_bytes` -/
def try_alloc_bytes (len : UInt256) : FeM Option_u8 := do
  let v3 ← MAX_ALLOCATION
  let v4 := decide (v3.toNat < len.toNat)
  if v4 then
    let v5 := Option_u8.None
    return v5
  else
    let ptr ← alloc len
    let v10 := ptr
    let v12 := decide (v10 = (UInt256.ofNat 0))
    let join : Option_u8 → FeM Option_u8 := fun v1 => do
      return v1
    if v12 then
      let v13 := Option_u8.None
      join v13
    else
      let v15 := Option_u8.Some ptr
      join v15

/-- `MemBuffer::try_new` -/
def MemBuffer_try_new (len : UInt256) : FeM Option_MemBuffer := do
  let v1 ← try_alloc_bytes len
  let v2 := match v1 with | .Some .. => true | _ => false
  if v2 then
    let ptr := v1.Some_0
    let _ ← zero_bytes ptr len
    let v8 := (MemBuffer.mk ptr len len)
    let v9 := Option_MemBuffer.Some v8
    return v9
  else
    let v10 := Option_MemBuffer.None
    return v10

/-- `<MemBuffer as Index>::index`, its `mut` body -/
def MemBuffer_index_mut (self : MemBuffer) (index : UInt256) : FeM (UInt256 × (UInt256 → FeM MemBuffer)) := do
  let mut self := self
  let v2 := self.len
  let v3 := decide (v2.toNat ≤ index.toNat)
  if v3 then
    let v4 ← index_out_of_bounds
    nomatch v4
  else
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

/-- `core::ptr::try_alloc_elems<u64>` -/
def try_alloc_elems_u64 (len : UInt256) : FeM Option_u64 := do
  let size := (UInt256.ofNat 8)
  let v4 := decide (size ≠ (UInt256.ofNat 0))
  let join : FeM Option_u64 := do
    let v13 ← checkedMul len size
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
    let v6 ← MAX_ALLOCATION
    let v7 ← checkedDiv v6 size
    let v8 := decide (v7.toNat < len.toNat)
    if v8 then
      let v9 := Option_u64.None
      return v9
    else
      join
  else
    join

/-- `MemArray<u64>::try_new_uninit` -/
def MemArray_try_new_uninit_u64 (len : UInt256) : FeM Option_MemArray_u64 := do
  let v1 ← try_alloc_elems_u64 len
  let v2 := match v1 with | .Some .. => true | _ => false
  if v2 then
    let v3 := v1.Some_0
    let v5 := (MemArray_u64.mk v3 len)
    let v6 := Option_MemArray_u64.Some v5
    return v6
  else
    let v7 := Option_MemArray_u64.None
    return v7

/-- `core::ptr::cast<u64, u8>` -/
def cast_u64_u8 (ptr : UInt256) : FeM UInt256 := do
  let v2 := ptr
  return v2

/-- `MemArray<u64>::byte_len` -/
def MemArray_byte_len_u64 (self : MemArray_u64) : FeM UInt256 := do
  let v1 := self.len
  let v3 ← checkedMul v1 (UInt256.ofNat 8)
  return v3

/-- `<usize as WrappingMul>::wrapping_mul` -/
def usize_wrapping_mul (self : UInt256) (other : UInt256) : FeM UInt256 := do
  let v2 := self * other
  return v2

/-- `core::ptr::elem<u64>` -/
def elem_u64 (ptr : UInt256) (index : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_mul index (UInt256.ofNat 8)
  let v5 := ptr + v4
  return v5

/-- `core::mem::drop<u64>` -/
def drop_u64 (value : UInt256) : FeM Unit := do
  return ()

/-- `core::ptr::drop_in_place<u64>` -/
def drop_in_place_u64 (ptr : UInt256) : FeM Unit := do
  let v2 ← mloadInt (.u 64) ptr
  let _ ← drop_u64 v2
  return ()

/-- `core::ptr::drop_range<u64>` -/
def drop_range_u64 (ptr : UInt256) (start : UInt256) («end» : UInt256) : FeM Unit := do
  let mut r0 : Range_Unknown_Unknown := default
  let v4 := false
  let join : Range_Unknown_Unknown → FeM Unit := fun r0 => do
    let mut r0 := r0
    return ()
  if v4 then
    let v8 := (Range_Unknown_Unknown.mk start «end»)
    r0 := v8
    let v10 ← Range_Unknown_Unknown_start r0
    let rec loop (r0 : Range_Unknown_Unknown) (v3 : UInt256) : FeM Unit := do
      let mut r0 := r0
      let v13 ← Range_Unknown_Unknown_has r0 v3
      if v13 then
        let v15 ← Range_Unknown_Unknown_at r0 v3
        let i := v15
        let v17 ← elem_u64 ptr i
        let _ ← drop_in_place_u64 v17
        let v20 ← Range_Unknown_Unknown_next r0 v3
        loop r0 v20
      else
        join r0
    partial_fixpoint
    loop r0 v10
  else
    join r0

/-- `core::ptr::drop_in_place_n<u64>` -/
def drop_in_place_n_u64 (ptr : UInt256) (len : UInt256) : FeM Unit := do
  let _ ← drop_range_u64 ptr (UInt256.ofNat 0) len
  return ()

/-- `core::ptr::dealloc<u64>` -/
def dealloc_u64 (ptr : UInt256) : FeM Unit := do
  let _ ← cast_u64_u8 ptr
  let v2 := ()
  return ()

/-- `<MemArray<u64> as Drop>::drop` -/
def MemArray_drop_u64 (self : MemArray_u64) : FeM Unit := do
  let v1 := self.ptr
  let v2 := self.len
  let _ ← drop_in_place_n_u64 v1 v2
  let v4 := self.ptr
  let _ ← dealloc_u64 v4
  return ()

/-- `MemArray<u64>::try_new_zeroed` -/
def MemArray_try_new_zeroed_u64 (len : UInt256) : FeM Option_MemArray_u64 := do
  let mut array : MemArray_u64 := default
  let v1 ← MemArray_try_new_uninit_u64 len
  let v2 := match v1 with | .Some .. => true | _ => false
  if v2 then
    let array_3 := v1.Some_0
    array := array_3
    let v5 := array.ptr
    let v6 ← cast_u64_u8 v5
    let v8 ← MemArray_byte_len_u64 array
    let _ ← zero_bytes v6 v8
    let array_10 := array
    let v12 := Option_MemArray_u64.Some array_10
    return v12
  else
    let v13 := Option_MemArray_u64.None
    match v1 with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v1.Some_0
      pure ()
    | _ => pure ()
    return v13

/-- `core::ptr::mem_array_elem<u64>` -/
def mem_array_elem_u64 (array : MemArray_u64) (index : UInt256) : FeM UInt256 := do
  let v3 := array.len
  let v4 := decide (v3.toNat ≤ index.toNat)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v8 := array.ptr
    let v9 ← elem_u64 v8 index
    return v9

/-- `<MemArray<u64> as Index>::index`, its `mut` body -/
def MemArray_index_mut_u64 (self : MemArray_u64) (index : UInt256) : FeM (UInt256 × (UInt256 → FeM MemArray_u64)) := do
  let mut self := self
  let v4 ← mem_array_elem_u64 self index
  let v5 := v4
  let slide : UInt256 → FeM MemArray_u64 := fun g => do
    let mut self := self
    return self
  return (v5, slide)

/-- `MemArray<u64>::len` -/
def MemArray_len_u64 (self : MemArray_u64) : FeM UInt256 := do
  let v1 := self.len
  return v1

/-- `<MemArray<u64> as Index>::index`, its `ref` body -/
def MemArray_index_ref_u64 (self : MemArray_u64) (index : UInt256) : FeM UInt256 := do
  let v2 ← mem_array_elem_u64 self index
  let v3 := v2
  return v3

/-- `<Range<Known<0>, Known<5>> as Collection>::start` -/
def Range_start_0_5 (self : Range_Known_0_Known_5) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<Range<Known<0>, Known<5>> as Collection>::has` -/
def Range_has_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 5)
  return v4

/-- `<Range<Known<0>, Known<5>> as Collection>::at` -/
def Range_at_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<5>> as Collection>::next` -/
def Range_next_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<usize as IntDowncast>::downcast_truncate` -/
def downcast_truncate_usize_usize (self : UInt256) : FeM UInt256 := do
  let word ← usize_to_word self
  let v4 ← usize_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  if v4 then
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    join v8
  else
    let v10 ← usize_MASK
    let v11 := word &&& v10
    let v12 ← usize_from_word v11
    join v12

/-- `<Range<Known<0>, Known<6>> as Collection>::start` -/
def Range_start_0_6 (self : Range_Known_0_Known_6) : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<Range<Known<0>, Known<6>> as Collection>::has` -/
def Range_has_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM Bool := do
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 6)
  return v4

/-- `<Range<Known<0>, Known<6>> as Collection>::at` -/
def Range_at_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<6>> as Collection>::next` -/
def Range_next_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `MemArray<u64>::ptr` -/
def MemArray_ptr_u64 (self : MemArray_u64) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

/-- `<Range<Known<1>, Unknown> as Collection>::start` -/
def Range_start_1 (self : Range_Known_1_Unknown) : FeM UInt256 := do
  return (UInt256.ofNat 1)

/-- `<Range<Known<1>, Unknown> as Collection>::has` -/
def Range_has_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM Bool := do
  let v3 := self.«end»
  let v4 ← range_has c (UInt256.ofNat 1) v3
  return v4

/-- `<Range<Known<1>, Unknown> as Collection>::at` -/
def Range_at_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<1>, Unknown> as Collection>::next` -/
def Range_next_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `MemSlice<u64>::from_raw_parts`, its `ref` body -/
def MemSlice_from_raw_parts_ref_u64 (ptr : UInt256) (len : UInt256) : FeM MemSlice_u64 := do
  let mut slice : MemSlice_u64 := default
  let slice_2 := (MemSlice_u64.mk ptr len)
  slice := slice_2
  return slice

/-- `MemArray<u64>::as_slice`, its `ref` body -/
def MemArray_as_slice_ref_u64 (self : MemArray_u64) : FeM MemSlice_u64 := do
  let v1 := self.ptr
  let v2 := self.len
  let v3 ← MemSlice_from_raw_parts_ref_u64 v1 v2
  return v3

/-- `MemSlice<u64>::slice`, its `ref` body -/
def MemSlice_slice_ref_u64 (self : MemSlice_u64) (offset : UInt256) (len : UInt256) : FeM MemSlice_u64 := do
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let join : FeM MemSlice_u64 := do
    let v13 ← index_out_of_bounds
    nomatch v13
  if v4 then
    join
  else
    let v5 := self.len
    let v6 ← checkedSub v5 offset
    let v7 := decide (v6.toNat < len.toNat)
    if v7 then
      join
    else
      let v10 := self.ptr
      let v11 ← elem_u64 v10 offset
      let v12 ← MemSlice_from_raw_parts_ref_u64 v11 len
      return v12

/-- `MemSlice<u64>::head`, its `ref` body -/
def MemSlice_head_ref_u64 (self : MemSlice_u64) (n : UInt256) : FeM MemSlice_u64 := do
  let v3 ← MemSlice_slice_ref_u64 self (UInt256.ofNat 0) n
  return v3

/-- `MemSlice<u64>::from_raw_parts`, its `mut` body -/
def MemSlice_from_raw_parts_mut_u64 (ptr : UInt256) (len : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM Unit)) := do
  let mut slice : MemSlice_u64 := default
  let slice_2 := (MemSlice_u64.mk ptr len)
  slice := slice_2
  let slide : MemSlice_u64 → FeM Unit := fun g => do
    let mut slice := slice
    slice := g
    return ()
  return (slice, slide)

/-- `MemArray<u64>::as_slice`, its `mut` body -/
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

/-- `MemSlice<u64>::slice`, its `mut` body -/
def MemSlice_slice_mut_u64 (self : MemSlice_u64) (offset : UInt256) (len : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := do
  let mut self := self
  let mut v12 : MemSlice_u64 := default
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let join : MemSlice_u64 → MemSlice_u64 → FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := fun self v12 => do
    let mut self := self
    let mut v12 := v12
    let v13 ← index_out_of_bounds
    nomatch v13
  if v4 then
    join self v12
  else
    let v5 := self.len
    let v6 ← checkedSub v5 offset
    let v7 := decide (v6.toNat < len.toNat)
    if v7 then
      join self v12
    else
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

/-- `MemSlice<u64>::head`, its `mut` body -/
def MemSlice_head_mut_u64 (self : MemSlice_u64) (n : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := do
  let mut self := self
  let mut v3 : MemSlice_u64 := default
  let (v3_g, v3_k) ← MemSlice_slice_mut_u64 self (UInt256.ofNat 0) n
  v3 := v3_g
  let slide : MemSlice_u64 → FeM MemSlice_u64 := fun g => do
    let mut self := self
    let mut v3 := v3
    v3 := g
    let v3_0 ← v3_k v3
    self := v3_0
    return self
  return (v3, slide)

/-- `core::ptr::byte_ptr<u64>` -/
def byte_ptr_u64 (ptr : UInt256) : FeM UInt256 := do
  let v1 ← cast_u64_u8 ptr
  return v1

/-- `MemSlice<u64>::byte_len` -/
def MemSlice_byte_len_u64 (self : MemSlice_u64) : FeM UInt256 := do
  let v1 := self.len
  let v3 ← checkedMul v1 (UInt256.ofNat 8)
  return v3

/-- `MemSlice<u64>::as_bytes` -/
def MemSlice_as_bytes_u64 (self : MemSlice_u64) : FeM MemSlice_u8 := do
  let v1 := self.ptr
  let v2 ← cast_u64_u8 v1
  let v3 ← MemSlice_byte_len_u64 self
  let v4 ← MemSlice_from_raw_parts_ref_u8 v2 v3
  return v4

/-- `MemSlice<u64>::copy_from` -/
def MemSlice_copy_from_u64 (self : MemSlice_u64) (source : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut self := self
  let v2 := source.len
  let v3 := self.len
  let v4 := decide (v2 ≠ v3)
  if v4 then
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    let v8 := self.ptr
    let v9 ← byte_ptr_u64 v8
    let v10 ← MemSlice_as_bytes_u64 source
    let _ ← copy v9 v10
    return ((), self)

/-- `MemSlice<u64>::len` -/
def MemSlice_len_u64 (self : MemSlice_u64) : FeM UInt256 := do
  let v1 := self.len
  return v1

/-- `MemSlice<u64>::ptr` -/
def MemSlice_ptr_u64 (self : MemSlice_u64) : FeM UInt256 := do
  let v1 := self.ptr
  return v1

/-- `MemSlice<u64>::zero` -/
def MemSlice_zero_u64 (self_0 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut self_0 := self_0
  let v1 := self_0.ptr
  let v2 := self_0.len
  let _ ← drop_in_place_n_u64 v1 v2
  let v4 := self_0.ptr
  let v5 ← byte_ptr_u64 v4
  let v7 ← MemSlice_byte_len_u64 self_0
  let _ ← zero_bytes v5 v7
  return ((), self_0)

/-- `MemArray<u64>::zero` -/
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

/-- `core::ptr::cast<u8, ChunkPair>` -/
def cast_u8_ChunkPair (ptr : UInt256) : FeM UInt256 := do
  let v2 := ptr
  return v2

/-- `core::ptr::alloc<ChunkPair>` -/
def alloc_ChunkPair : FeM UInt256 := do
  let v1 ← alloc_bytes (UInt256.ofNat 64)
  let v2 ← cast_u8_ChunkPair v1
  return v2

/-- `std::evm::ssz::write_pair` -/
def write_pair (scratch : UInt256) (left : UInt256) (right : UInt256) : FeM Unit := do
  mstore scratch left
  mstore (scratch + UInt256.ofNat 32) right
  return ()

/-- `std::evm::ssz::alloc_pair` -/
def alloc_pair (left : UInt256) (right : UInt256) : FeM UInt256 := do
  let scratch ← alloc_ChunkPair
  let _ ← write_pair scratch left right
  return scratch

/-- `core::ptr::cast<ChunkPair, u8>` -/
def cast_ChunkPair_u8 (ptr : UInt256) : FeM UInt256 := do
  let v2 := ptr
  return v2

/-- `core::ptr::byte_ptr<ChunkPair>` -/
def byte_ptr_ChunkPair (ptr : UInt256) : FeM UInt256 := do
  let v1 ← cast_ChunkPair_u8 ptr
  return v1

/-- `std::evm::ssz::pair_offset` -/
def pair_offset (scratch : UInt256) : FeM UInt256 := do
  let v1 ← byte_ptr_ChunkPair scratch
  return v1

/-- `std::evm::ssz::sha256_64_to` -/
def sha256_64_to (offset : UInt256) (out : UInt256) : FeM UInt256 := do
  let v2 ← gas
  let ok ← staticcallPrecompile v2 (UInt256.ofNat 2) offset (UInt256.ofNat 64) out (UInt256.ofNat 32)
  let v9 := decide (ok = (UInt256.ofNat 0))
  if v9 then
    let v10 ← revertEmpty
    nomatch v10
  else
    let v13 ← returndatasize
    let v15 := decide (v13.toNat < (UInt256.ofNat 32).toNat)
    if v15 then
      let v16 ← revertEmpty
      nomatch v16
    else
      let v19 ← mload out
      return v19

/-- `std::evm::ssz::hash_pair_buffer` -/
def hash_pair_buffer (scratch : UInt256) : FeM UInt256 := do
  let offset ← pair_offset scratch
  let v3 ← sha256_64_to offset offset
  return v3

/-- `std::evm::ssz::hash_pair` -/
def hash_pair (left : UInt256) (right : UInt256) : FeM UInt256 := do
  let scratch ← alloc_pair left right
  let v4 ← hash_pair_buffer scratch
  return v4

/-- `Bytes::len` -/
def Bytes_len (self : Bytes) : FeM UInt256 := do
  let v1 := self.len
  return v1

/-- `std::evm::units::ether` -/
def ether (n : UInt256) : FeM Wei := do
  let v3 ← checkedMul n (UInt256.ofNat 1000000000000000000)
  let v4 := (Wei.mk v3)
  return v4

/-- `<Wei as Ord>::ge` -/
def Wei_ge (self : Wei) (other : Wei) : FeM Bool := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 := decide (v3.toNat ≤ v2.toNat)
  return v4

/-- `std::evm::units::gwei` -/
def gwei (n : UInt256) : FeM Wei := do
  let v3 ← checkedMul n (UInt256.ofNat 1000000000)
  let v4 := (Wei.mk v3)
  return v4

/-- `<Wei as Rem>::rem` -/
def Wei_rem (self : Wei) (other : Wei) : FeM Wei := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 ← checkedRem v2 v3
  let v5 := (Wei.mk v4)
  return v5

/-- `<Wei as Eq>::eq` -/
def Wei_eq (self : Wei) (other : Wei) : FeM Bool := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 := decide (v2 = v3)
  return v4

/-- `<Wei as Div>::div` -/
def Wei_div (self : Wei) (other : Wei) : FeM UInt256 := do
  let v2 := self.amount
  let v3 := other.amount
  let v4 ← checkedDiv v2 v3
  return v4

/-- `Bytes::payload_ptr` -/
def Bytes_payload_ptr (self : Bytes) : FeM UInt256 := do
  let v1 := self.data
  let v3 := v1 + (UInt256.ofNat 32)
  return v3

/-- `Bytes::span`, its `ref` body -/
def Bytes_span_ref (self : Bytes) : FeM MemSlice_u8 := do
  let v1 ← Bytes_payload_ptr self
  let v2 := self.len
  let v3 ← MemSlice_from_raw_parts_ref_u8 v1 v2
  return v3

/-- `core::ptr::elem<u8>` -/
def elem_u8 (ptr : UInt256) (index : UInt256) : FeM UInt256 := do
  let v4 ← usize_wrapping_mul index (UInt256.ofNat 1)
  let v5 := ptr + v4
  return v5

/-- `MemSlice<u8>::slice`, its `ref` body -/
def MemSlice_slice_ref_u8 (self : MemSlice_u8) (offset : UInt256) (len : UInt256) : FeM MemSlice_u8 := do
  let v3 := self.len
  let v4 := decide (v3.toNat < offset.toNat)
  let join : FeM MemSlice_u8 := do
    let v13 ← index_out_of_bounds
    nomatch v13
  if v4 then
    join
  else
    let v5 := self.len
    let v6 ← checkedSub v5 offset
    let v7 := decide (v6.toNat < len.toNat)
    if v7 then
      join
    else
      let v10 := self.ptr
      let v11 ← elem_u8 v10 offset
      let v12 ← MemSlice_from_raw_parts_ref_u8 v11 len
      return v12

/-- `<ByteVector<48> as HashTreeRoot>::hash_tree_root` -/
def ByteVector_48_hash_tree_root (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  let v5 := decide (len = (UInt256.ofNat 48))
  if v5 then
    let v7 ← downcast_unchecked_u256_usize start
    let v8 ← downcast_unchecked_u256_usize len
    let v9 ← MemSlice_slice_ref_u8 input v7 v8
    let base ← MemSlice_ptr_u8 v9
    let v14 := base + (UInt256.ofNat 32)
    let tail ← mload v14
    let v17 ← mload base
    let v19 := tail >>> (UInt256.ofNat 128)
    let v21 := v19 <<< (UInt256.ofNat 128)
    let scratch ← alloc_pair v17 v21
    let v24 ← hash_pair_buffer scratch
    return v24
  else
    M.panic 0x01

/-- `std::evm::ssz::hash_tree_root<ByteVector<48>>` -/
def hash_tree_root_ByteVector_48 (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  let v3 ← ByteVector_48_hash_tree_root input start len
  return v3

/-- `Bytes::word_at` -/
def Bytes_word_at (self : Bytes) (offset : UInt256) : FeM UInt256 := do
  let v3 := self.len
  let v4 := decide (v3.toNat ≤ offset.toNat)
  if v4 then
    let v5 ← trap
    nomatch v5
  else
    let v8 ← Bytes_payload_ptr self
    let v9 := v8 + offset
    let v10 := self.len
    let v12 ← checkedSub v10 offset
    let v13 ← readPaddedWord v9 v12
    return v13

def u64_chunk_k1 (v22 : UInt256) (v : UInt256) : FeM UInt256 := do
  let mut v := v
  let v24 := v22 <<< (UInt256.ofNat 16)
  let v_25 := v
  let v28 := v_25 >>> (UInt256.ofNat 16)
  let v30 := v28 &&& (UInt256.ofNat 281470681808895)
  let v_31 := v24 ||| v30
  v := v_31
  let v_34 := v
  let v37 := v_34 <<< (UInt256.ofNat 32)
  let v_38 := v
  let v41 := v_38 >>> (UInt256.ofNat 32)
  let v42 := v37 ||| v41
  let v44 := v42 <<< (UInt256.ofNat 192)
  return v44

/-- `std::evm::ssz::u64_chunk` -/
def u64_chunk (value : UInt256) : FeM UInt256 := do
  let mut v : UInt256 := default
  let v_2 := value
  v := v_2
  let v_4 := v
  let v7 := v_4 &&& (UInt256.ofNat 71777214294589695)
  let v9 := v7 <<< (UInt256.ofNat 8)
  let v_10 := v
  let v13 := v_10 >>> (UInt256.ofNat 8)
  let v15 := v13 &&& (UInt256.ofNat 71777214294589695)
  let v_16 := v9 ||| v15
  v := v_16
  let v_19 := v
  let v22 := v_19 &&& (UInt256.ofNat 281470681808895)
  u64_chunk_k1 v22 v

/-- `<ByteVector<96> as HashTreeRoot>::hash_tree_root` -/
def ByteVector_96_hash_tree_root (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  let v5 := decide (len = (UInt256.ofNat 96))
  if v5 then
    let v7 ← downcast_unchecked_u256_usize start
    let v8 ← downcast_unchecked_u256_usize len
    let v9 ← MemSlice_slice_ref_u8 input v7 v8
    let base ← MemSlice_ptr_u8 v9
    let scratch ← alloc_pair (UInt256.ofNat 0) (UInt256.ofNat 0)
    let offset ← pair_offset scratch
    let left ← sha256_64_to base offset
    let v22 := base + (UInt256.ofNat 64)
    let v23 ← mload v22
    let _ ← write_pair scratch v23 (UInt256.ofNat 0)
    let right ← hash_pair_buffer scratch
    let _ ← write_pair scratch left right
    let v29 ← hash_pair_buffer scratch
    return v29
  else
    M.panic 0x01

/-- `std::evm::ssz::hash_tree_root<ByteVector<96>>` -/
def hash_tree_root_ByteVector_96 (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  let v3 ← ByteVector_96_hash_tree_root input start len
  return v3

/-- `std::evm::ssz::merkleize4` -/
def merkleize4 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) : FeM UInt256 := do
  let scratch ← alloc_pair a b
  let left ← hash_pair_buffer scratch
  let _ ← write_pair scratch c d
  let right ← hash_pair_buffer scratch
  let _ ← write_pair scratch left right
  let v12 ← hash_pair_buffer scratch
  return v12

/-- `<(u256, u256, u256, u256) as Merkleize>::merkleize` -/
def tuple_u256_u256_u256_u256_merkleize (self : (UInt256 × UInt256 × UInt256 × UInt256)) : FeM UInt256 := do
  let v1 := self.fst
  let v2 := self.snd.fst
  let v3 := self.snd.snd.fst
  let v4 := self.snd.snd.snd
  let v5 ← merkleize4 v1 v2 v3 v4
  return v5

/-- `Bytes::from_validated_abi_tail` -/
def Bytes_from_validated_abi_tail (data : UInt256) (len : UInt256) (size : UInt256) : FeM Bytes := do
  let v6 := (Bytes.mk data len size)
  return v6

/-- `Bytes::empty` -/
def Bytes_empty : FeM Bytes := do
  let data ← alloc_bytes (UInt256.ofNat 32)
  let _ ← store_word data (UInt256.ofNat 0)
  let v7 ← Bytes_from_validated_abi_tail data (UInt256.ofNat 0) (UInt256.ofNat 32)
  return v7

/-- `Bytes::from_word_prefix` -/
def Bytes_from_word_prefix (word : UInt256) (len : UInt256) : FeM Bytes := do
  let v4 := decide ((UInt256.ofNat 32).toNat < len.toNat)
  if v4 then
    let v5 ← trap
    nomatch v5
  else
    let v10 := decide (len = (UInt256.ofNat 0))
    if v10 then
      let v11 ← Bytes_empty
      return v11
    else
      let data ← alloc_bytes (UInt256.ofNat 64)
      let v19 := len
      let v20 ← checkedSub (UInt256.ofNat 32) v19
      let shift ← checkedMul v20 (UInt256.ofNat 8)
      let v25 := len
      let _ ← store_word data v25
      let v28 := data + (UInt256.ofNat 32)
      let v30 := word >>> shift
      let v31 := v30 <<< shift
      let _ ← store_word v28 v31
      let v34 ← Bytes_from_validated_abi_tail data len (UInt256.ofNat 64)
      return v34

/-- `std::evm::ssz::serialize_u64` -/
def serialize_u64 (value : UInt256) : FeM Bytes := do
  let v1 ← u64_chunk value
  let v3 ← Bytes_from_word_prefix v1 (UInt256.ofNat 8)
  return v3

/-- `<Bytes as AbiSize>::IS_DYNAMIC` -/
def Bytes_IS_DYNAMIC : FeM Bool := do
  return true

/-- `<Bytes as AbiSize>::HEAD_SIZE` -/
def Bytes_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `core::abi::abi_field_head_size<Bytes>` -/
def abi_field_head_size_Bytes : FeM UInt256 := do
  let v0 ← Bytes_IS_DYNAMIC
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← Bytes_HEAD_SIZE
    return v4

/-- `<(Bytes, Bytes, Bytes, Bytes, Bytes) as AbiSize>::HEAD_SIZE` -/
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

/-- `Bytes::encoded_size` -/
def Bytes_encoded_size (self : Bytes) : FeM UInt256 := do
  let v1 := self.size
  return v1

/-- `<Bytes as AbiSize>::payload_size` -/
def Bytes_payload_size (self : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_encoded_size self
  return v1

/-- `core::abi::dynamic_payload_size<Bytes>` -/
def dynamic_payload_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_IS_DYNAMIC
  if v1 then
    let v2 ← Bytes_payload_size value
    return v2
  else
    return (UInt256.ofNat 0)

/-- `<(Bytes, Bytes, Bytes, Bytes, Bytes) as AbiSize>::payload_size` -/
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

/-- `core::abi::abi_payload_size<(Bytes, Bytes, Bytes, Bytes, Bytes)>` -/
def abi_payload_size_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM UInt256 := do
  let v1 ← tuple_payload_size_Bytes_Bytes_Bytes_Bytes_Bytes value
  return v1

/-- `Bytes::encoded_span` -/
def Bytes_encoded_span (self : Bytes) : FeM MemSlice_u8 := do
  let v1 := self.data
  let v2 := self.size
  let v3 ← word_to_usize v2
  let v4 ← MemSlice_from_raw_parts_ref_u8 v1 v3
  return v4

/-- `<Bytes as Encode>::encode` -/
def Bytes_encode_Sol (self : Bytes) (out : AbiFrame) : FeM Unit := do
  let v3 ← Bytes_encoded_span self
  let _ ← AbiFrame_copy_from out (UInt256.ofNat 0) v3
  return ()

/-- `core::abi::encode_field<Sol, Bytes>` -/
def encode_field_Sol_Bytes (value : Bytes) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    let size ← Bytes_payload_size value
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let v9 := v6 - v8
    let _ ← AbiFrame_store_word out head_pos v9
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← Bytes_encode_Sol value r1
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    let v20 ← Bytes_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← Bytes_encode_Sol value r2
    let tails_24 := tails
    return tails_24

def tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes_k1 (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) (out : AbiFrame) (head1 : UInt256) (head2 : UInt256) (head3 : UInt256) (head4 : UInt256) (tails_28 : AbiFrame) (tails : AbiFrame) : FeM Unit := do
  let mut tails := tails
  let tails_31 ← encode_field_Sol_Bytes self.snd.fst out head1 tails_28
  tails := tails_31
  let tails_34 := tails
  let tails_37 ← encode_field_Sol_Bytes self.snd.snd.fst out head2 tails_34
  tails := tails_37
  let tails_40 := tails
  let tails_43 ← encode_field_Sol_Bytes self.snd.snd.snd.fst out head3 tails_40
  tails := tails_43
  let tails_46 := tails
  let _ ← encode_field_Sol_Bytes self.snd.snd.snd.snd out head4 tails_46
  return ()

/-- `<(Bytes, Bytes, Bytes, Bytes, Bytes) as Encode>::encode` -/
def tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) (out : AbiFrame) : FeM Unit := do
  let mut tails : AbiFrame := default
  let v7 ← tuple_HEAD_SIZE_Bytes_Bytes_Bytes_Bytes_Bytes
  let tails_8 ← AbiFrame_tail out v7
  tails := tails_8
  let head1 ← abi_field_head_size_Bytes
  let v12 ← abi_field_head_size_Bytes
  let head2 := head1 + v12
  let v15 ← abi_field_head_size_Bytes
  let head3 := head2 + v15
  let v18 ← abi_field_head_size_Bytes
  let head4 := head3 + v18
  let tails_22 := tails
  let tails_25 ← encode_field_Sol_Bytes self.fst out (UInt256.ofNat 0) tails_22
  tails := tails_25
  let tails_28 := tails
  tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes_k1 self out head1 head2 head3 head4 tails_28 tails

/-- `<Bytes as Drop>::drop` -/
def Bytes_drop (self : Bytes) : FeM Unit := do
  let v1 := self.data
  let _ ← dealloc_u8 v1
  return ()

/-- `core::abi::encode_alloc<Sol, (Bytes, Bytes, Bytes, Bytes, Bytes)>` -/
def encode_alloc_Sol_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let total ← abi_payload_size_tuple_Bytes_Bytes_Bytes_Bytes_Bytes value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let _ ← tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes value r2
  let _ ← Bytes_drop value.fst
  let _ ← Bytes_drop value.snd.fst
  let _ ← Bytes_drop value.snd.snd.fst
  let _ ← Bytes_drop value.snd.snd.snd.fst
  let _ ← Bytes_drop value.snd.snd.snd.snd
  let buffer_12 := buffer
  return buffer_12

/-- `std::evm::effects::encode_abi_payload<(Bytes, Bytes, Bytes, Bytes, Bytes)>` -/
def encode_abi_payload_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM MemBuffer := do
  let v2 ← encode_alloc_Sol_tuple_Bytes_Bytes_Bytes_Bytes_Bytes value
  return v2

/-- `<Evm as Log>::log1` -/
def Evm_log1 (data : MemSlice_u8) (topic0 : UInt256) : FeM Unit := do
  let v3 ← MemSlice_ptr_u8 data
  let v4 ← MemSlice_len_u8 data
  let v5 := v4
  let _ ← logData v3 v5 #[topic0]
  return ()

/-- `std::evm::ssz::mix_in_length` -/
def mix_in_length (root : UInt256) (len : UInt256) : FeM UInt256 := do
  let v2 ← u64_chunk len
  let v3 ← hash_pair root v2
  return v3

/-- `<FixedBytes<32> as AbiSize>::IS_DYNAMIC` -/
def FixedBytes_IS_DYNAMIC_32 : FeM Bool := do
  return false

/-- `<FixedBytes<32> as AbiSize>::HEAD_SIZE` -/
def FixedBytes_HEAD_SIZE_32 : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `core::abi::abi_field_head_size<FixedBytes<32>>` -/
def abi_field_head_size_FixedBytes_32 : FeM UInt256 := do
  let v0 ← FixedBytes_IS_DYNAMIC_32
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← FixedBytes_HEAD_SIZE_32
    return v4

/-- `core::abi::MAX_PAYLOAD_LEN` -/
def MAX_PAYLOAD_LEN : FeM UInt256 := do
  let v0 ← MAX_ALLOCATION
  let v1 := v0
  let v3 ← checkedSub v1 (UInt256.ofNat 31)
  return v3

/-- `core::abi::checked_dynamic_payload_after_head<Sol, CallData>` -/
def checked_dynamic_payload_after_head_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let data_len ← CallData_word_at input pos
  let v8 ← checkedAdd pos (UInt256.ofNat 32)
  let v9 ← checkedSub input_len v8
  let v10 := decide (v9.toNat < data_len.toNat)
  let join : FeM (UInt256 × UInt256) := do
    let v23 ← Sol_decode_error
    nomatch v23
  if v10 then
    join
  else
    let v11 ← MAX_PAYLOAD_LEN
    let v13 ← checkedSub v11 (UInt256.ofNat 32)
    let v14 := decide (v13.toNat < data_len.toNat)
    if v14 then
      join
    else
      let v18 ← checkedAdd data_len (UInt256.ofNat 63)
      let payload_size := v18 &&& (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639904)
      let v22 := (data_len, payload_size)
      return v22

/-- `core::abi::checked_dynamic_payload_with_input_len<Sol, CallData>` -/
def checked_dynamic_payload_with_input_len_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let _ ← checked_frame_end_Sol pos (UInt256.ofNat 32) input_len
  let v5 ← checked_dynamic_payload_after_head_Sol_CallData input pos input_len
  return v5

/-- `core::ptr::alloc_bytes_unchecked` -/
def alloc_bytes_unchecked (len : UInt256) : FeM UInt256 := do
  let v1 ← alloc len
  return v1

/-- `<CallData as ByteSource>::copy_to_memory_unchecked` -/
def CallData_copy_to_memory_unchecked (self : CallData) (dest : UInt256) (src_offset : UInt256) (len : UInt256) : FeM Unit := do
  let v4 := self.base
  let v6 := v4 + src_offset
  let _ ← calldatacopy dest v6 len
  return ()

/-- `core::abi::copy_padded_payload<CallData>` -/
def copy_padded_payload_CallData (input : CallData) (pos : UInt256) (dest : UInt256) (actual_size : UInt256) (size : UInt256) : FeM Unit := do
  let v7 ← checkedSub size (UInt256.ofNat 32)
  let v8 ← ptr_offset dest v7
  let _ ← store_word v8 (UInt256.ofNat 0)
  let _ ← CallData_copy_to_memory_unchecked input dest pos actual_size
  return ()

/-- `core::abi::copy_dynamic_bytes<CallData>` -/
def copy_dynamic_bytes_CallData (input : CallData) (pos : UInt256) (dest : UInt256) (data_len : UInt256) (size : UInt256) : FeM Unit := do
  let v7 ← checkedAdd (UInt256.ofNat 32) data_len
  let _ ← copy_padded_payload_CallData input pos dest v7 size
  return ()

/-- `core::abi::copy_dynamic_payload_from_with_input_len<Sol, CallData>` -/
def copy_dynamic_payload_from_with_input_len_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256 × UInt256) := do
  let payload ← checked_dynamic_payload_with_input_len_Sol_CallData input pos input_len
  let data_len := payload.fst
  let payload_size := payload.snd
  let v9 ← word_to_usize payload_size
  let data_ptr ← alloc_bytes_unchecked v9
  let _ ← copy_dynamic_bytes_CallData input pos data_ptr data_len payload_size
  let v13 := (data_ptr, data_len, payload_size)
  return v13

/-- `<Bytes as Decode>::decode_from_bounded<CallData>` -/
def Bytes_decode_from_bounded_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  let payload ← copy_dynamic_payload_from_with_input_len_Sol_CallData input pos input_len
  let v5 := payload.fst
  let v6 := payload.snd.fst
  let v7 ← word_to_usize v6
  let v8 := payload.snd.snd
  let v9 ← Bytes_from_validated_abi_tail v5 v7 v8
  return v9

/-- `core::abi::copy_dynamic_payload_from<Sol, CallData>` -/
def copy_dynamic_payload_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM (UInt256 × UInt256 × UInt256) := do
  let input_len ← len_CallData input
  let v4 ← copy_dynamic_payload_from_with_input_len_Sol_CallData input pos input_len
  return v4

/-- `<Bytes as Decode>::decode_from<CallData>` -/
def Bytes_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM Bytes := do
  let payload ← copy_dynamic_payload_from_Sol_CallData input pos
  let v4 := payload.fst
  let v5 := payload.snd.fst
  let v6 ← word_to_usize v5
  let v7 := payload.snd.snd
  let v8 ← Bytes_from_validated_abi_tail v4 v6 v7
  return v8

/-- `core::abi::decode_field_from_prechecked_head<Sol, Bytes, CallData>` -/
def decode_field_from_prechecked_head_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  let v4 ← Bytes_IS_DYNAMIC
  if v4 then
    let tail_rel ← CallData_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← Bytes_decode_from_bounded_Sol_CallData input tail_at input_len
    return v9
  else
    let v12 ← Bytes_decode_from_Sol_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, Bytes, CallData>` -/
def decode_field_from_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM Bytes := do
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    let tail_rel ← CallData_word_at input head_pos
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← Bytes_decode_from_Sol_CallData input v6
    return v7
  else
    let v10 ← Bytes_decode_from_Sol_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, Bytes, CallData>` -/
def decode_msg_field_from_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  let v4 ← Bytes_IS_DYNAMIC
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_Bytes_CallData input base head_pos input_len
    return v5
  else
    let v8 ← decode_field_from_Sol_Bytes_CallData input base head_pos
    return v8

/-- `std::abi::sol::types::decode_fixed_bytes_word` -/
def decode_fixed_bytes_word (word : UInt256) (len : UInt256) : FeM UInt256 := do
  let v4 := decide (len = (UInt256.ofNat 0))
  let join : FeM UInt256 := do
    let v33 ← revertEmpty
    nomatch v33
  if v4 then
    join
  else
    let v7 := decide ((UInt256.ofNat 32).toNat < len.toNat)
    if v7 then
      join
    else
      let v12 := decide (len = (UInt256.ofNat 32))
      if v12 then
        return word
      else
        let v19 := len * (UInt256.ofNat 8)
        let v20 := word <<< v19
        let v22 := decide (v20 = (UInt256.ofNat 0))
        if v22 then
          let v26 := (UInt256.ofNat 32) - len
          let v28 := v26 * (UInt256.ofNat 8)
          let v29 := word >>> v28
          return v29
        else
          let v32 ← revertEmpty
          let v32 : UInt256 := nomatch v32
          return v32

/-- `<FixedBytes<32> as Decode>::decode_from<CallData>` -/
def FixedBytes_decode_from_Sol_32_CallData (input : CallData) (pos : UInt256) : FeM FixedBytes_32 := do
  let v2 ← CallData_word_at input pos
  let v4 := (UInt256.ofNat 32)
  let v5 ← decode_fixed_bytes_word v2 v4
  let v6 := (FixedBytes_32.mk v5)
  return v6

/-- `core::abi::Decode::decode_from_bounded<FixedBytes<32>, Sol, CallData>` -/
def decode_from_bounded_FixedBytes_32_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  let v3 ← FixedBytes_HEAD_SIZE_32
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← FixedBytes_decode_from_Sol_32_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, FixedBytes<32>, CallData>` -/
def decode_field_from_prechecked_head_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  let v4 ← FixedBytes_IS_DYNAMIC_32
  if v4 then
    let tail_rel ← CallData_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← decode_from_bounded_FixedBytes_32_Sol_CallData input tail_at input_len
    return v9
  else
    let v12 ← FixedBytes_decode_from_Sol_32_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, FixedBytes<32>, CallData>` -/
def decode_field_from_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM FixedBytes_32 := do
  let v3 ← FixedBytes_IS_DYNAMIC_32
  if v3 then
    let tail_rel ← CallData_word_at input head_pos
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← FixedBytes_decode_from_Sol_32_CallData input v6
    return v7
  else
    let v10 ← FixedBytes_decode_from_Sol_32_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, FixedBytes<32>, CallData>` -/
def decode_msg_field_from_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  let v4 ← FixedBytes_IS_DYNAMIC_32
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_FixedBytes_32_CallData input base head_pos input_len
    return v5
  else
    let v8 ← decode_field_from_Sol_FixedBytes_32_CallData input base head_pos
    return v8

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<Bytes>` -/
def const_Bytes : FeM Bool := do
  return false

/-- `core::abi::abi_payload_size<Bytes>` -/
def abi_payload_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← Bytes_payload_size value
  return v1

/-- `core::abi::encode_alloc<Sol, Bytes>` -/
def encode_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  let total ← abi_payload_size_Bytes value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  let _ ← Bytes_encode_Sol value r2
  let _ ← Bytes_drop value
  let buffer_12 := buffer
  return buffer_12

/-- `core::abi::abi_field_size<Bytes>` -/
def abi_field_size_Bytes (value : Bytes) : FeM UInt256 := do
  let payload_size ← Bytes_payload_size value
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<Bytes>` -/
def abi_single_root_size_Bytes (value : Bytes) : FeM UInt256 := do
  let v1 ← abi_field_size_Bytes value
  return v1

/-- `core::abi::abi_single_root_head_size<Bytes>` -/
def abi_single_root_head_size_Bytes : FeM UInt256 := do
  let v0 ← abi_field_head_size_Bytes
  return v0

/-- `core::abi::encode_single_root_alloc<Sol, Bytes>` -/
def encode_single_root_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  let total ← abi_single_root_size_Bytes value
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  out := out_8
  let v11 ← abi_single_root_head_size_Bytes
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_Bytes value out (UInt256.ofNat 0) v13
  let _ ← Bytes_drop value
  let buffer_17 := buffer
  return buffer_17

/-- `core::abi::encode_return_alloc<Sol, Bytes>` -/
def encode_return_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let v1 ← const_Bytes
  if v1 then
    let v3 ← encode_alloc_Sol_Bytes value
    return v3
  else
    let v7 ← encode_single_root_alloc_Sol_Bytes value
    return v7

/-- `<Evm as ContractHost>::return_value<Sol, Bytes>` -/
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
    let v5 := decide (v3.toNat ≤ (UInt256.ofNat 32).toNat)
    if v5 then
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← Bytes_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let _ ← Bytes_encode_Sol value r5
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← Bytes_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      let v23 ← Bytes_HEAD_SIZE
      let v25 := decide (v23.toNat ≤ (UInt256.ofNat 64).toNat)
      if v25 then
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← Bytes_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let _ ← Bytes_encode_Sol value r6
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← Bytes_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        let v43 ← Bytes_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← Bytes_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        let _ ← Bytes_encode_Sol value r7
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    let value_59 := value
    let out_61 ← encode_return_alloc_Sol_Bytes value_59
    out_r4 := out_61
    let v64 ← MemBuffer_span_ref out_r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

/-- `<FixedBytes<4> as AbiSize>::IS_DYNAMIC` -/
def FixedBytes_IS_DYNAMIC_4 : FeM Bool := do
  return false

/-- `<FixedBytes<4> as AbiSize>::HEAD_SIZE` -/
def FixedBytes_HEAD_SIZE_4 : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `core::abi::abi_field_head_size<FixedBytes<4>>` -/
def abi_field_head_size_FixedBytes_4 : FeM UInt256 := do
  let v0 ← FixedBytes_IS_DYNAMIC_4
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← FixedBytes_HEAD_SIZE_4
    return v4

/-- `<FixedBytes<4> as Decode>::decode_from<CallData>` -/
def FixedBytes_decode_from_Sol_4_CallData (input : CallData) (pos : UInt256) : FeM FixedBytes_4 := do
  let v2 ← CallData_word_at input pos
  let v4 := (UInt256.ofNat 4)
  let v5 ← decode_fixed_bytes_word v2 v4
  let v6 := (FixedBytes_4.mk v5)
  return v6

/-- `core::abi::Decode::decode_from_bounded<FixedBytes<4>, Sol, CallData>` -/
def decode_from_bounded_FixedBytes_4_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  let v3 ← FixedBytes_HEAD_SIZE_4
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← FixedBytes_decode_from_Sol_4_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, FixedBytes<4>, CallData>` -/
def decode_field_from_prechecked_head_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  let v4 ← FixedBytes_IS_DYNAMIC_4
  if v4 then
    let tail_rel ← CallData_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← decode_from_bounded_FixedBytes_4_Sol_CallData input tail_at input_len
    return v9
  else
    let v12 ← FixedBytes_decode_from_Sol_4_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, FixedBytes<4>, CallData>` -/
def decode_field_from_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM FixedBytes_4 := do
  let v3 ← FixedBytes_IS_DYNAMIC_4
  if v3 then
    let tail_rel ← CallData_word_at input head_pos
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← FixedBytes_decode_from_Sol_4_CallData input v6
    return v7
  else
    let v10 ← FixedBytes_decode_from_Sol_4_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, FixedBytes<4>, CallData>` -/
def decode_msg_field_from_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  let v4 ← FixedBytes_IS_DYNAMIC_4
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_FixedBytes_4_CallData input base head_pos input_len
    return v5
  else
    let v8 ← decode_field_from_Sol_FixedBytes_4_CallData input base head_pos
    return v8

/-- `<(u256, u256) as AbiSize>::IS_DYNAMIC` -/
def tuple_IS_DYNAMIC_u256_u256 : FeM Bool := do
  let v1 ← u256_IS_DYNAMIC
  let join : Bool → FeM Bool := fun v0 => do
    return v0
  let join_1 : FeM Bool := do
    join true
  if v1 then
    join_1
  else
    let v2 ← u256_IS_DYNAMIC
    if v2 then
      join_1
    else
      join false

/-- `core::abi::abi_field_head_size<(u256, u256)>` -/
def abi_field_head_size_tuple_u256_u256 : FeM UInt256 := do
  let v0 ← tuple_IS_DYNAMIC_u256_u256
  if v0 then
    return (UInt256.ofNat 32)
  else
    let v4 ← tuple_HEAD_SIZE_u256_u256
    return v4

/-- `<(u256, u256) as Decode>::decode_from_bounded<CallData>` -/
def tuple_decode_from_bounded_Sol_u256_u256_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← tuple_HEAD_SIZE_u256_u256
  let _ ← checked_frame_end_Sol pos v3 input_len
  let v5 ← tuple_decode_from_prechecked_head_Sol_u256_u256_CallData input pos input_len
  return v5

/-- `<(u256, u256) as Decode>::decode_from<CallData>` -/
def tuple_decode_from_Sol_u256_u256_CallData (input : CallData) (pos : UInt256) : FeM (UInt256 × UInt256) := do
  let mut head_pos : UInt256 := default
  head_pos := pos
  let head_pos_4 := head_pos
  let v0 ← decode_field_from_Sol_u256_CallData input pos head_pos_4
  let v8 ← abi_field_head_size_u256
  let head_pos_9 := head_pos
  let head_pos_10 ← checkedAdd head_pos_9 v8
  head_pos := head_pos_10
  let head_pos_13 := head_pos
  let v1 ← decode_field_from_Sol_u256_CallData input pos head_pos_13
  let v17 := (v0, v1)
  return v17

/-- `core::abi::decode_field_from_prechecked_head<Sol, (u256, u256), CallData>` -/
def decode_field_from_prechecked_head_Sol_tuple_u256_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let v4 ← tuple_IS_DYNAMIC_u256_u256
  if v4 then
    let tail_rel ← CallData_word_at input head_pos
    let tail_at ← checked_tail_Sol base tail_rel
    let v9 ← tuple_decode_from_bounded_Sol_u256_u256_CallData input tail_at input_len
    return v9
  else
    let v12 ← tuple_decode_from_Sol_u256_u256_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, (u256, u256), CallData>` -/
def decode_field_from_Sol_tuple_u256_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 ← tuple_IS_DYNAMIC_u256_u256
  if v3 then
    let tail_rel ← CallData_word_at input head_pos
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← tuple_decode_from_Sol_u256_u256_CallData input v6
    return v7
  else
    let v10 ← tuple_decode_from_Sol_u256_u256_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, (u256, u256), CallData>` -/
def decode_msg_field_from_Sol_tuple_u256_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  let v4 ← tuple_IS_DYNAMIC_u256_u256
  if v4 then
    let v5 ← decode_field_from_prechecked_head_Sol_tuple_u256_u256_CallData input base head_pos input_len
    return v5
  else
    let v8 ← decode_field_from_Sol_tuple_u256_u256_CallData input base head_pos
    return v8


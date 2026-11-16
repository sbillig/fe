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
  return CallData.mk (UInt256.ofNat 0)

/-- `<Evm as ContractHost>::input` -/
def Evm_input : FeM CallData := do
  -- CallData::new()
  let v1 ← CallData_new
  return v1

/-- `<Sol as Abi>::SELECTOR_SIZE` -/
def Sol_SELECTOR_SIZE : FeM UInt256 := do
  return UInt256.ofNat 4

/-- `<u256 as IntWord>::to_word` -/
def u256_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<u64 as IntWord>::IS_SIGNED` -/
def u64_IS_SIGNED : FeM Bool := do
  return false

/-- `core::num::U64_MASK` -/
def U64_MASK : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551615

/-- `<u64 as IntWord>::MASK` -/
def u64_MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

/-- `<u64 as IntWord>::SIGN_BIT` -/
def u64_SIGN_BIT : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `core::num::U256_MASK` -/
def U256_MASK : FeM UInt256 := do
  return UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935

/-- `core::num::sign_extend` -/
def sign_extend (word : UInt256) (mask : UInt256) (sign_bit : UInt256) : FeM UInt256 := do
  -- let value = word & mask
  let value := word &&& mask
  let join : UInt256 → FeM UInt256 := fun v3 => do
    return v3
  -- if value & sign_bit != 0 {
  if decide ((value &&& sign_bit) ≠ (UInt256.ofNat 0)) then
    -- value | (U256_MASK ^ mask)
    let v12 ← U256_MASK
    join (value ||| (v12 ^^^ mask))
  else
    join value

/-- `<u64 as IntWord>::from_word` -/
def u64_from_word (word : UInt256) : FeM UInt256 := do
  return truncate 64 word

/-- `<u256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u256_u64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u256_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u64_MASK
    let v12 ← u64_from_word (word &&& v10)
    join v12

/-- `<CallData as ByteSource>::size` -/
def CallData_size (self : CallData) : FeM UInt256 := do
  -- let total = calldatasize()
  let total ← calldatasize
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if self.base > total {
  if decide (total.toNat < self.base.toNat) then
    join (UInt256.ofNat 0)
  else
    -- (total - self.base).downcast_truncate()
    let v9 ← checkedSub total self.base
    let v10 ← downcast_truncate_u256_u64 v9
    join v10

/-- `<CallData as ByteInput>::len` -/
def len_CallData (self : CallData) : FeM UInt256 := do
  -- self.size() as u256
  let v1 ← CallData_size self
  return v1

/-- `<i256 as IntWord>::from_word` -/
def i256_from_word (word : UInt256) : FeM UInt256 := do
  return word

/-- `<Sol as Abi>::validate_runtime_head` -/
def Sol_validate_runtime_head (input_len : UInt256) (base : UInt256) (head_size : UInt256) : FeM Unit := do
  let join : FeM Unit := do
    return ()
  -- if head_size != 0 {
  if decide (head_size ≠ (UInt256.ofNat 0)) then
    -- let args_len = i256::from_word(input_len - base)
    let args_len ← i256_from_word (input_len - base)
    -- if args_len < i256::from_word(head_size) {
    let v11 ← i256_from_word head_size
    if UInt256.sltBool args_len v11 then
      -- revert_empty()
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
  return UInt256.ofNat 32

/-- `<u64 as Bounded>::max` -/
def u64_max : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551615

/-- `<u64 as IntWord>::to_word` -/
def u64_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<usize as IntWord>::IS_SIGNED` -/
def usize_IS_SIGNED : FeM Bool := do
  return false

/-- `<usize as IntWord>::MASK` -/
def usize_MASK : FeM UInt256 := do
  return ~~~(UInt256.ofNat 0)

/-- `<usize as IntWord>::from_word` -/
def usize_from_word (word : UInt256) : FeM UInt256 := do
  return word

/-- `core::num::I256_SIGN_BIT` -/
def I256_SIGN_BIT : FeM UInt256 := do
  return UInt256.ofNat 57896044618658097711785492504343953926634992332820282019728792003956564819968

/-- `<usize as IntWord>::SIGN_BIT` -/
def usize_SIGN_BIT : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<u64 as IntDowncast>::downcast_saturate` -/
def downcast_saturate_u64_usize (self : UInt256) : FeM UInt256 := do
  let mut unsigned_word : UInt256 := default
  -- let word = IntWord::to_word(self)
  let word ← u64_to_word self
  -- if !T::IS_SIGNED {
  let v10 ← usize_IS_SIGNED
  let join : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v1 => do
    let mut unsigned_word := unsigned_word
    return v1
  -- if !T::IS_SIGNED {
  if !v10 then
    -- var unsigned_word = word
    unsigned_word := word
    -- if S::IS_SIGNED && (word & I256_SIGN_BIT != 0) {
    let v13 ← u64_IS_SIGNED
    let join_1 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      -- if unsigned_word > T::MASK {
      let v26 ← usize_MASK
      let join_2 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v2 => do
        let mut unsigned_word := unsigned_word
        join unsigned_word v2
      -- if unsigned_word > T::MASK {
      if decide (v26.toNat < unsigned_word.toNat) then
        -- T::from_word(T::MASK)
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        join_2 unsigned_word v29
      else
        -- T::from_word(unsigned_word)
        let v33 ← usize_from_word unsigned_word
        join_2 unsigned_word v33
    let join_3 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      join_1 unsigned_word
    -- if S::IS_SIGNED && (word & I256_SIGN_BIT != 0) {
    if v13 then
      -- if S::IS_SIGNED && (word & I256_SIGN_BIT != 0) {
      let v14 ← I256_SIGN_BIT
      if decide ((word &&& v14) ≠ (UInt256.ofNat 0)) then
        -- unsigned_word = 0
        unsigned_word := (UInt256.ofNat 0)
        join_1 unsigned_word
      else
        join_3 unsigned_word
    else
      join_3 unsigned_word
  else
    -- let t_min_word = sign_extend(word: T::SIGN_BIT, mask: T::MASK, sign_bit: T::SIGN_BIT)
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let t_min_word ← sign_extend v36 v37 v38
    -- let t_max_word = T::SIGN_BIT - 1
    let v41 ← usize_SIGN_BIT
    let t_max_word ← checkedSub v41 (UInt256.ofNat 1)
    -- let t_min: i256 = __bitcast(t_min_word)
    let t_min := t_min_word
    -- let t_max: i256 = __bitcast(t_max_word)
    let t_max := t_max_word
    -- if S::IS_SIGNED {
    let v49 ← u64_IS_SIGNED
    let join_4 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v3 => do
      let mut unsigned_word := unsigned_word
      join unsigned_word v3
    -- if S::IS_SIGNED {
    if v49 then
      -- let x: i256 = __bitcast(word)
      let x_50 := word
      let join_5 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v4 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v4
      -- if x < t_min {
      if UInt256.sltBool x_50 t_min then
        -- T::from_word(t_min_word)
        let v53 ← usize_from_word t_min_word
        join_5 unsigned_word v53
      else
        let join_6 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v5 => do
          let mut unsigned_word := unsigned_word
          join_5 unsigned_word v5
        -- if x < t_min {
        if UInt256.sgtBool x_50 t_max then
          -- T::from_word(t_max_word)
          let v56 ← usize_from_word t_max_word
          join_6 unsigned_word v56
        else
          -- T::from_word(word)
          let v58 ← usize_from_word word
          join_6 unsigned_word v58
    else
      -- if word & I256_SIGN_BIT != 0 {
      let v62 ← I256_SIGN_BIT
      let join_7 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v6 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v6
      -- if word & I256_SIGN_BIT != 0 {
      if decide ((word &&& v62) ≠ (UInt256.ofNat 0)) then
        -- T::from_word(t_max_word)
        let v66 ← usize_from_word t_max_word
        join_7 unsigned_word v66
      else
        let join_8 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v7 => do
          let mut unsigned_word := unsigned_word
          join_7 unsigned_word v7
        -- if x > t_max {
        if UInt256.sgtBool word t_max then
          -- T::from_word(t_max_word)
          let v71 ← usize_from_word t_max_word
          join_8 unsigned_word v71
        else
          -- T::from_word(word)
          let v73 ← usize_from_word word
          join_8 unsigned_word v73

/-- `core::ptr::MAX_ALLOCATION` -/
def MAX_ALLOCATION : FeM UInt256 := do
  let v0 ← u64_max
  let v1 ← downcast_saturate_u64_usize v0
  return v1

/-- `core::panics::PANIC_OUT_OF_MEMORY` -/
def PANIC_OUT_OF_MEMORY : FeM UInt256 := do
  return UInt256.ofNat 65

/-- `core::ptr::alloc_bytes` -/
def alloc_bytes (len : UInt256) : FeM UInt256 := do
  -- if len > MAX_ALLOCATION {
  let v2 ← MAX_ALLOCATION
  if decide (v2.toNat < len.toNat) then
    -- panic_code(PANIC_OUT_OF_MEMORY)
    let v4 ← PANIC_OUT_OF_MEMORY
    let v5 ← panicCode v4
    nomatch v5
  else
    -- alloc_raw(len)
    let v8 ← alloc len
    return v8

/-- `FixedMemBuffer<32>::alloc_uninit` -/
def FixedMemBuffer_alloc_uninit_32 : FeM FixedMemBuffer_32 := do
  -- Self {
  let v1 ← alloc_bytes (UInt256.ofNat 32)
  return FixedMemBuffer_32.mk v1

/-- `core::ptr::zero_bytes` -/
def zero_bytes (ptr : UInt256) (len : UInt256) : FeM Unit := do
  -- zero_mem(ptr, len)
  let _ ← zeroMem ptr len
  return ()

/-- `FixedMemBuffer<32>::alloc` -/
def FixedMemBuffer_alloc_32 : FeM FixedMemBuffer_32 := do
  -- Self::alloc_uninit()
  let buffer ← FixedMemBuffer_alloc_uninit_32
  -- zero_bytes(buffer.ptr, N)
  let _ ← zero_bytes buffer.ptr (UInt256.ofNat 32)
  return buffer

/-- `FixedMemBuffer<32>::ptr` -/
def FixedMemBuffer_ptr_32 (self : FixedMemBuffer_32) : FeM UInt256 := do
  return self.ptr

/-- `AbiFrame::from_raw_parts` -/
def AbiFrame_from_raw_parts (ptr : UInt256) (len : UInt256) : FeM AbiFrame := do
  return AbiFrame.mk ptr len

/-- `core::panics::PANIC_INDEX_OUT_OF_BOUNDS` -/
def PANIC_INDEX_OUT_OF_BOUNDS : FeM UInt256 := do
  return UInt256.ofNat 50

/-- `core::ptr::index_out_of_bounds` -/
def index_out_of_bounds : FeM Empty := do
  -- panic_code(PANIC_INDEX_OUT_OF_BOUNDS)
  let v0 ← PANIC_INDEX_OUT_OF_BOUNDS
  let v1 ← panicCode v0
  nomatch v1

/-- `core::num::word_to_usize` -/
def word_to_usize (word : UInt256) : FeM UInt256 := do
  return word

/-- `core::abi::ptr_offset` -/
def ptr_offset (ptr : UInt256) (offset : UInt256) : FeM UInt256 := do
  -- ptr::offset_bytes(ptr, word_to_usize(offset))
  let v2 ← word_to_usize offset
  return ptr + v2

/-- `AbiFrame::slice` -/
def AbiFrame_slice (self : AbiFrame) (offset : UInt256) (len : UInt256) : FeM AbiFrame := do
  let join : FeM AbiFrame := do
    -- ptr::index_out_of_bounds()
    let v17 ← index_out_of_bounds
    nomatch v17
  -- if offset > self.len || len > self.len - offset {
  if decide (self.len.toNat < offset.toNat) then
    join
  else
    -- if offset > self.len || len > self.len - offset {
    if decide ((self.len - offset).toNat < len.toNat) then
      join
    else
      -- AbiFrame { ptr: ptr_offset(self.ptr, offset), len }
      let v14 ← ptr_offset self.ptr offset
      return AbiFrame.mk v14 len

/-- `core::ptr::cast<u8, u256>` -/
def cast_u8_u256 (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `core::abi::store_word` -/
def store_word (ptr : UInt256) (value : UInt256) : FeM Unit := do
  -- *ptr::cast<u8, u256>(ptr) = to_be(value)
  let v3 ← cast_u8_u256 ptr
  mstore v3 value
  return ()

/-- `AbiFrame::store_word` -/
def AbiFrame_store_word (self : AbiFrame) («at» : UInt256) (value : UInt256) : FeM Unit := do
  -- let word = self.slice(offset: at, len: 32)
  let word ← AbiFrame_slice self «at» (UInt256.ofNat 32)
  -- store_word(ptr: word.ptr, value)
  let _ ← store_word word.ptr value
  return ()

/-- `<u256 as Encode>::encode` -/
def u256_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  -- out.store_word(at: 0, value: self)
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) self
  return ()

/-- `<u256 as IntDowncast>::downcast_unchecked` -/
def downcast_unchecked_u256_usize (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u256_to_word self
  -- T::from_word(word)
  let v3 ← usize_from_word word
  return v3

/-- `MemSlice<u8>::from_raw_parts`, its `ref` body -/
def MemSlice_from_raw_parts_ref_u8 (ptr : UInt256) (len : UInt256) : FeM MemSlice_u8 := do
  let mut slice : MemSlice_u8 := default
  -- var slice = Self { ptr, len }
  slice := (MemSlice_u8.mk ptr len)
  return slice

/-- `MemSlice<u8>::ptr` -/
def MemSlice_ptr_u8 (self : MemSlice_u8) : FeM UInt256 := do
  return self.ptr

/-- `MemSlice<u8>::len` -/
def MemSlice_len_u8 (self : MemSlice_u8) : FeM UInt256 := do
  return self.len

/-- `<Evm as RawOps>::return_data` -/
def Evm_return_data (data : MemSlice_u8) : FeM Empty := do
  -- ops::return_data(offset: data.ptr(), len: data.len() as u256)
  let v2 ← MemSlice_ptr_u8 data
  let v3 ← MemSlice_len_u8 data
  let v5 ← returnData v2 v3
  nomatch v5

/-- `FixedMemBuffer<64>::alloc_uninit` -/
def FixedMemBuffer_alloc_uninit_64 : FeM FixedMemBuffer_64 := do
  -- Self {
  let v1 ← alloc_bytes (UInt256.ofNat 64)
  return FixedMemBuffer_64.mk v1

/-- `FixedMemBuffer<64>::alloc` -/
def FixedMemBuffer_alloc_64 : FeM FixedMemBuffer_64 := do
  -- Self::alloc_uninit()
  let buffer ← FixedMemBuffer_alloc_uninit_64
  -- zero_bytes(buffer.ptr, N)
  let _ ← zero_bytes buffer.ptr (UInt256.ofNat 64)
  return buffer

/-- `FixedMemBuffer<64>::ptr` -/
def FixedMemBuffer_ptr_64 (self : FixedMemBuffer_64) : FeM UInt256 := do
  return self.ptr

/-- `MemBuffer::new_uninit` -/
def MemBuffer_new_uninit (len : UInt256) : FeM MemBuffer := do
  -- Self {
  let v1 ← alloc_bytes len
  return MemBuffer.mk v1 len len

/-- `MemBuffer::new` -/
def MemBuffer_new (len : UInt256) : FeM MemBuffer := do
  -- Self::new_uninit(len)
  let buffer ← MemBuffer_new_uninit len
  -- zero_bytes(buffer.ptr, len)
  let _ ← zero_bytes buffer.ptr len
  return buffer

/-- `MemBuffer::ptr` -/
def MemBuffer_ptr (self : MemBuffer) : FeM UInt256 := do
  return self.ptr

/-- `MemBuffer::span`, its `ref` body -/
def MemBuffer_span_ref (self : MemBuffer) : FeM MemSlice_u8 := do
  -- MemSpan::from_raw_parts(ptr: self.ptr, len: self.len)
  let v3 ← MemSlice_from_raw_parts_ref_u8 self.ptr self.len
  return v3

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<u256>` -/
def const_u256 : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<u256>` -/
def payload_size_u256 (self : UInt256) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← u256_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<u256>` -/
def abi_payload_size_u256 (value : UInt256) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← payload_size_u256 value
  return v1

/-- `<usize as Bounded>::max` -/
def usize_max : FeM UInt256 := do
  return ~~~(UInt256.ofNat 0)

/-- `core::ptr::allocation_len` -/
def allocation_len (len : UInt256) : FeM UInt256 := do
  -- if len > usize::max() as u256 {
  let v2 ← usize_max
  if decide (v2.toNat < len.toNat) then
    -- panic_code(PANIC_OUT_OF_MEMORY)
    let v5 ← PANIC_OUT_OF_MEMORY
    let v6 ← panicCode v5
    nomatch v6
  else
    -- word_to_usize(len)
    let v9 ← word_to_usize len
    return v9

/-- `core::abi::encode_alloc<Sol, u256>` -/
def encode_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_u256 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← u256_encode_Sol value r1
  return buffer

/-- `core::abi::abi_field_size<u256>` -/
def abi_field_size_u256 (value : UInt256) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← payload_size_u256 value
  -- if T::IS_DYNAMIC {
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<u256>` -/
def abi_single_root_size_u256 (value : UInt256) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_u256 value
  return v1

/-- `core::abi::abi_field_head_size<u256>` -/
def abi_field_head_size_u256 : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← u256_IS_DYNAMIC
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← u256_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<u256>` -/
def abi_single_root_head_size_u256 : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_u256
  return v0

/-- `AbiFrame::tail` -/
def AbiFrame_tail (self : AbiFrame) (n : UInt256) : FeM AbiFrame := do
  -- if n > self.len {
  if decide (self.len.toNat < n.toNat) then
    -- ptr::index_out_of_bounds()
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    -- AbiFrame { ptr: ptr_offset(self.ptr, n), len: self.len - n }
    let v9 ← ptr_offset self.ptr n
    return AbiFrame.mk v9 (self.len - n)

/-- `AbiFrame::len` -/
def AbiFrame_len (self : AbiFrame) : FeM UInt256 := do
  return self.len

/-- `core::abi::encode_field<Sol, u256>` -/
def encode_field_Sol_u256 (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    -- let size = value.payload_size()
    let size ← payload_size_u256 value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← u256_encode_Sol value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← u256_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← u256_encode_Sol value r2
    return tails

/-- `core::abi::encode_single_root_alloc<Sol, u256>` -/
def encode_single_root_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_u256 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_u256
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_u256 value out (UInt256.ofNat 0) v13
  return buffer

/-- `core::abi::encode_return_alloc<Sol, u256>` -/
def encode_return_alloc_Sol_u256 (value : UInt256) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← const_u256
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v2 ← encode_alloc_Sol_u256 value
    return v2
  else
    -- encode_single_root_alloc<A, T>(value)
    let v5 ← encode_single_root_alloc_Sol_u256 value
    return v5

/-- `<Evm as ContractHost>::return_bytes` -/
def Evm_return_bytes (data : MemSlice_u8) : FeM Empty := do
  -- self.return_data(data)
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
  -- if !T::IS_DYNAMIC {
  let v2 ← u256_IS_DYNAMIC
  if !v2 then
    -- if T::HEAD_SIZE <= 32 {
    let v4 ← u256_HEAD_SIZE
    if decide (v4.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_7 ← FixedMemBuffer_alloc_32
      out_r0 := out_7
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← u256_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let _ ← u256_encode_Sol value r4
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r0
      let v17 ← u256_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← u256_HEAD_SIZE
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r1 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← u256_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let _ ← u256_encode_Sol value r5
        -- self.return_data(
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← u256_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v42 ← u256_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let out_44 ← MemBuffer_new v43
        out_r2 := out_44
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← u256_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let _ ← u256_encode_Sol value r6
        -- self.return_data(data: out.span())
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    -- let out = encode_return_alloc<A, T>(value)
    let out_57 ← encode_return_alloc_Sol_u256 value
    out_r3 := out_57
    -- self.return_bytes(data: out.span())
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

/-- `<CallData as ByteSource>::word_at` -/
def CallData_word_at (self : CallData) (byte_offset : UInt256) : FeM UInt256 := do
  -- calldataload(self.base + byte_offset)
  let v5 ← calldataload (self.base + byte_offset)
  return v5

/-- `<Sol as Abi>::decode_error` -/
def Sol_decode_error : FeM Empty := do
  -- revert_empty()
  let v0 ← revertEmpty
  nomatch v0

/-- `core::abi::checked_tail<Sol>` -/
def checked_tail_Sol (base : UInt256) (tail_rel : UInt256) : FeM UInt256 := do
  -- if tail_rel > (!(0 as u256)) - base {
  let v8 := decide (((~~~(UInt256.ofNat 0)) - base).toNat < tail_rel.toNat)
  if v8 then
    -- A::decode_error()
    let v9 ← Sol_decode_error
    nomatch v9
  else
    return base + tail_rel

/-- `core::abi::checked_frame_end<Sol>` -/
def checked_frame_end_Sol (pos : UInt256) (head_size : UInt256) (input_len : UInt256) : FeM UInt256 := do
  let join : FeM UInt256 := do
    -- A::decode_error()
    let v16 ← Sol_decode_error
    nomatch v16
  -- if pos > input_len || head_size > input_len - pos {
  if decide (input_len.toNat < pos.toNat) then
    join
  else
    -- if pos > input_len || head_size > input_len - pos {
    if decide ((input_len - pos).toNat < head_size.toNat) then
      join
    else
      return pos + head_size

/-- `<u256 as Decode>::decode_from<CallData>` -/
def u256_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM UInt256 := do
  -- input.word_at(pos)
  let v2 ← CallData_word_at input pos
  return v2

/-- `core::abi::Decode::decode_from_bounded<u256, Sol, CallData>` -/
def decode_from_bounded_u256_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- checked_frame_end<A>(pos, Self::HEAD_SIZE, input_len)
  let v3 ← u256_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- Self::decode_from(input, pos)
  let v5 ← u256_decode_from_Sol_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, u256, CallData>` -/
def decode_field_from_prechecked_head_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← decode_from_bounded_u256_Sol_CallData input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← u256_decode_from_Sol_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, u256, CallData>` -/
def decode_field_from_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v3 ← u256_IS_DYNAMIC
  if v3 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- return T::decode_from(input, checked_tail<A>(base, tail_rel))
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← u256_decode_from_Sol_CallData input v6
    return v7
  else
    -- T::decode_from(input, head_pos)
    let v10 ← u256_decode_from_Sol_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, u256, CallData>` -/
def decode_msg_field_from_Sol_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    -- return decode_field_from_prechecked_head<A, T, I>(input, base: base, head_pos: head_pos, input_len: input_len)
    let v5 ← decode_field_from_prechecked_head_Sol_u256_CallData input base head_pos input_len
    return v5
  else
    -- decode_field_from<A, T, I>(input, base: base, head_pos: head_pos)
    let v8 ← decode_field_from_Sol_u256_CallData input base head_pos
    return v8

/-- `SolDecoder::with_base` -/
def SolDecoder_with_base (base : UInt256) : FeM SolDecoder := do
  return SolDecoder.mk base base

/-- `<Sol as Abi>::decoder_new` -/
def Sol_decoder_new : FeM SolDecoder := do
  -- SolDecoder::with_base(0)
  let v1 ← SolDecoder_with_base (UInt256.ofNat 0)
  return v1

/-- `<usize as IntWord>::to_word` -/
def usize_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<usize as IntDowncast>::downcast_truncate` -/
def downcast_truncate_usize_u64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← usize_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u64_MASK
    let v12 ← u64_from_word (word &&& v10)
    join v12

/-- `<MemSlice<u8> as ByteSource>::size` -/
def MemSlice_u8_size (self : MemSlice_u8) : FeM UInt256 := do
  -- self.len().downcast_truncate()
  let v1 ← MemSlice_len_u8 self
  let v2 ← downcast_truncate_usize_u64 v1
  return v2

/-- `<MemSlice<u8> as ByteInput>::len` -/
def len_MemSlice_u8 (self : MemSlice_u8) : FeM UInt256 := do
  -- self.size() as u256
  let v1 ← MemSlice_u8_size self
  return v1

/-- `<MemSlice<u8> as ByteSource>::word_at` -/
def MemSlice_u8_word_at (self : MemSlice_u8) (byte_offset : UInt256) : FeM UInt256 := do
  -- let len = self.len() as u256
  let len ← MemSlice_len_u8 self
  -- if byte_offset >= len {
  if decide (len.toNat ≤ byte_offset.toNat) then
    return UInt256.ofNat 0
  else
    -- let available = word_to_usize(len - byte_offset)
    let available ← word_to_usize (len - byte_offset)
    -- ptr::read_padded_word(ptr::offset_bytes(self.ptr(), word_to_usize(byte_offset)), available)
    let v14 ← MemSlice_ptr_u8 self
    let v15 ← word_to_usize byte_offset
    let v17 ← readPaddedWord (v14 + v15) available
    return v17

/-- `<SolDecoder as AbiDecoder>::read_word<MemSlice<u8>>` -/
def SolDecoder_read_word_MemSlice_u8 (self : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut self := self
  -- let new_pos = self.pos + 32
  let new_pos ← checkedAdd self.pos (UInt256.ofNat 32)
  let join : SolDecoder → FeM (UInt256 × SolDecoder) := fun self => do
    let mut self := self
    -- revert_empty()
    let v16 ← revertEmpty
    nomatch v16
  -- if new_pos < self.pos || new_pos > input.len() {
  if decide (new_pos.toNat < self.pos.toNat) then
    join self
  else
    -- if new_pos < self.pos || new_pos > input.len() {
    let v8 ← len_MemSlice_u8 input
    if decide (v8.toNat < new_pos.toNat) then
      join self
    else
      -- let w = input.word_at(self.pos)
      let w ← MemSlice_u8_word_at input self.pos
      -- self.pos = new_pos
      self := { self with pos := new_pos }
      return (w, self)

/-- `<SolDecoder as AbiDecoder>::base` -/
def SolDecoder_base (self : SolDecoder) : FeM UInt256 := do
  return self.base

/-- `<SolDecoder as AbiDecoder>::pos` -/
def SolDecoder_pos (self : SolDecoder) : FeM UInt256 := do
  return self.pos

/-- `<SolDecoder as AbiDecoder>::set_base` -/
def SolDecoder_set_base (self : SolDecoder) (new_base : UInt256) : FeM (Unit × SolDecoder) := do
  let mut self := self
  -- self.base = new_base
  self := { self with base := new_base }
  return ((), self)

/-- `<SolDecoder as AbiDecoder>::set_pos` -/
def SolDecoder_set_pos (self : SolDecoder) (new_pos : UInt256) : FeM (Unit × SolDecoder) := do
  let mut self := self
  -- self.pos = new_pos
  self := { self with pos := new_pos }
  return ((), self)

/-- `<u256 as Decode>::decode_payload<SolDecoder, MemSlice<u8>>` -/
def u256_decode_payload_Sol_SolDecoder_MemSlice_u8 (d : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d := d
  -- d.read_word(input)
  let (v2, v2_0) ← SolDecoder_read_word_MemSlice_u8 d input
  d := v2_0
  return (v2, d)

/-- `core::abi::decode_field<Sol, u256, SolDecoder, MemSlice<u8>>` -/
def decode_field_Sol_u256_SolDecoder_MemSlice_u8 (d_0 : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d_0 := d_0
  -- if T::IS_DYNAMIC {
  let v2 ← u256_IS_DYNAMIC
  if v2 then
    -- let tail_rel = d.read_word(input)
    let (tail_rel, tail_rel_0) ← SolDecoder_read_word_MemSlice_u8 d_0 input
    d_0 := tail_rel_0
    -- let parent_base = d.base()
    let parent_base ← SolDecoder_base d_0
    -- let parent_pos = d.pos()
    let parent_pos ← SolDecoder_pos d_0
    -- d.set_base(checked_tail<A>(parent_base, tail_rel))
    let v11 ← checked_tail_Sol parent_base tail_rel
    let (v12, v12_0) ← SolDecoder_set_base d_0 v11
    d_0 := v12_0
    -- d.set_pos(d.base())
    let v14 ← SolDecoder_base d_0
    let (v15, v15_0) ← SolDecoder_set_pos d_0 v14
    d_0 := v15_0
    -- let value = T::decode_payload(mut d, input)
    let (value, value_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 d_0 input
    d_0 := value_0
    -- d.set_base(parent_base)
    let (v19, v19_0) ← SolDecoder_set_base d_0 parent_base
    d_0 := v19_0
    -- d.set_pos(parent_pos)
    let (v20, v20_0) ← SolDecoder_set_pos d_0 parent_pos
    d_0 := v20_0
    return (value, d_0)
  else
    -- T::decode_payload(mut d, input)
    let (v24, v24_0) ← u256_decode_payload_Sol_SolDecoder_MemSlice_u8 d_0 input
    d_0 := v24_0
    return (v24, d_0)

/-- `<(u256) as Decode>::decode_payload<SolDecoder, MemSlice<u8>>` -/
def tuple_decode_payload_Sol_u256_SolDecoder_MemSlice_u8 (d : SolDecoder) (input : MemSlice_u8) : FeM (UInt256 × SolDecoder) := do
  let mut d := d
  -- let v0: T0 = decode_field<A, T0, D, I>(mut d, input)
  let (v0, v0_0) ← decode_field_Sol_u256_SolDecoder_MemSlice_u8 d input
  d := v0_0
  return (v0, d)

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
  return UInt256.ofNat 32

/-- `<u256 as IntWord>::IS_SIGNED` -/
def u256_IS_SIGNED : FeM Bool := do
  return false

/-- `<u256 as IntDowncast>::downcast_saturate` -/
def downcast_saturate_u256_usize (self : UInt256) : FeM UInt256 := do
  let mut unsigned_word : UInt256 := default
  -- let word = IntWord::to_word(self)
  let word ← u256_to_word self
  -- if !T::IS_SIGNED {
  let v10 ← usize_IS_SIGNED
  let join : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v1 => do
    let mut unsigned_word := unsigned_word
    return v1
  -- if !T::IS_SIGNED {
  if !v10 then
    -- var unsigned_word = word
    unsigned_word := word
    -- if S::IS_SIGNED && (word & I256_SIGN_BIT != 0) {
    let v13 ← u256_IS_SIGNED
    let join_1 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      -- if unsigned_word > T::MASK {
      let v26 ← usize_MASK
      let join_2 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v2 => do
        let mut unsigned_word := unsigned_word
        join unsigned_word v2
      -- if unsigned_word > T::MASK {
      if decide (v26.toNat < unsigned_word.toNat) then
        -- T::from_word(T::MASK)
        let v28 ← usize_MASK
        let v29 ← usize_from_word v28
        join_2 unsigned_word v29
      else
        -- T::from_word(unsigned_word)
        let v33 ← usize_from_word unsigned_word
        join_2 unsigned_word v33
    let join_3 : UInt256 → FeM UInt256 := fun unsigned_word => do
      let mut unsigned_word := unsigned_word
      join_1 unsigned_word
    -- if S::IS_SIGNED && (word & I256_SIGN_BIT != 0) {
    if v13 then
      -- if S::IS_SIGNED && (word & I256_SIGN_BIT != 0) {
      let v14 ← I256_SIGN_BIT
      if decide ((word &&& v14) ≠ (UInt256.ofNat 0)) then
        -- unsigned_word = 0
        unsigned_word := (UInt256.ofNat 0)
        join_1 unsigned_word
      else
        join_3 unsigned_word
    else
      join_3 unsigned_word
  else
    -- let t_min_word = sign_extend(word: T::SIGN_BIT, mask: T::MASK, sign_bit: T::SIGN_BIT)
    let v36 ← usize_SIGN_BIT
    let v37 ← usize_MASK
    let v38 ← usize_SIGN_BIT
    let t_min_word ← sign_extend v36 v37 v38
    -- let t_max_word = T::SIGN_BIT - 1
    let v41 ← usize_SIGN_BIT
    let t_max_word ← checkedSub v41 (UInt256.ofNat 1)
    -- let t_min: i256 = __bitcast(t_min_word)
    let t_min := t_min_word
    -- let t_max: i256 = __bitcast(t_max_word)
    let t_max := t_max_word
    -- if S::IS_SIGNED {
    let v49 ← u256_IS_SIGNED
    let join_4 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v3 => do
      let mut unsigned_word := unsigned_word
      join unsigned_word v3
    -- if S::IS_SIGNED {
    if v49 then
      -- let x: i256 = __bitcast(word)
      let x_50 := word
      let join_5 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v4 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v4
      -- if x < t_min {
      if UInt256.sltBool x_50 t_min then
        -- T::from_word(t_min_word)
        let v53 ← usize_from_word t_min_word
        join_5 unsigned_word v53
      else
        let join_6 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v5 => do
          let mut unsigned_word := unsigned_word
          join_5 unsigned_word v5
        -- if x < t_min {
        if UInt256.sgtBool x_50 t_max then
          -- T::from_word(t_max_word)
          let v56 ← usize_from_word t_max_word
          join_6 unsigned_word v56
        else
          -- T::from_word(word)
          let v58 ← usize_from_word word
          join_6 unsigned_word v58
    else
      -- if word & I256_SIGN_BIT != 0 {
      let v62 ← I256_SIGN_BIT
      let join_7 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v6 => do
        let mut unsigned_word := unsigned_word
        join_4 unsigned_word v6
      -- if word & I256_SIGN_BIT != 0 {
      if decide ((word &&& v62) ≠ (UInt256.ofNat 0)) then
        -- T::from_word(t_max_word)
        let v66 ← usize_from_word t_max_word
        join_7 unsigned_word v66
      else
        let join_8 : UInt256 → UInt256 → FeM UInt256 := fun unsigned_word v7 => do
          let mut unsigned_word := unsigned_word
          join_7 unsigned_word v7
        -- if x > t_max {
        if UInt256.sgtBool word t_max then
          -- T::from_word(t_max_word)
          let v71 ← usize_from_word t_max_word
          join_8 unsigned_word v71
        else
          -- T::from_word(word)
          let v73 ← usize_from_word word
          join_8 unsigned_word v73

/-- `<u256 as StorageKey>::write_key` -/
def u256_write_key (ptr : UInt256) (self : UInt256) : FeM Unit := do
  -- mstore(addr: ptr, value: self)
  let _ ← mstore ptr self
  return ()

/-- `std::evm::crypto::keccak256` -/
def keccak256_1 (data : MemSlice_u8) : FeM UInt256 := do
  -- ops::keccak256(offset: data.ptr(), len: data.len() as u256)
  let v1 ← MemSlice_ptr_u8 data
  let v2 ← MemSlice_len_u8 data
  let v4 ← keccak256 v1 v2
  return v4

/-- `core::ptr::cast<u8, u8>` -/
def cast_u8_u8 (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `core::ptr::dealloc<u8>` -/
def dealloc_u8 (ptr : UInt256) : FeM Unit := do
  -- dealloc_raw(cast<T, u8>(ptr))
  let _ ← cast_u8_u8 ptr
  return ()

/-- `<MemBuffer as Drop>::drop` -/
def MemBuffer_drop (self : MemBuffer) : FeM Unit := do
  -- dealloc(self.ptr)
  let _ ← dealloc_u8 self.ptr
  return ()

/-- `std::evm::layout::mapping_slot<u256>` -/
def mapping_slot_u256 (root : UInt256) (key : UInt256) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  -- let key_len = K::encoded_len(key)
  let key_len ← u256_encoded_len key
  -- ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate())
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  -- let buffer = unsafe { ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate()) }
  buffer := buffer_7
  -- let scratch = buffer.ptr()
  let scratch ← MemBuffer_ptr buffer
  -- K::write_key(ptr: scratch, key)
  let _ ← u256_write_key scratch key
  -- mstore(addr: ptr::offset_bytes(scratch, key_len.downcast_unchecked()), value: root)
  let v13 ← downcast_unchecked_u256_usize key_len
  let _ ← mstore (scratch + v13) root
  -- keccak256(ptr::MemSpan::from_raw_parts(ptr: scratch, len: (key_len + 32).downcast_unchecked()))
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<u256, u256> as StateIndex>::locate` -/
def StorageMap_locate_u256_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  -- Location { slot: mapping_slot(root, key), offset: 0 }
  let v3 ← mapping_slot_u256 root key
  return Location.mk v3 (UInt256.ofNat 0)

/-- `StoragePackedArray<8>::LANE_MASK` -/
def StoragePackedArray_LANE_MASK_8 : FeM UInt256 := do
  let v4 ← checkedSub ((UInt256.ofNat 1) <<< (UInt256.ofNat 8)) (UInt256.ofNat 1)
  return v4

/-- `StoragePackedArray<8>::get` -/
def StoragePackedArray_get_8 (self : Place) (i : UInt256) : FeM UInt256 := do
  -- let bit_pos: u256 = (i % Self::LANES_PER_SLOT) * BITS
  let v3 ← StoragePackedArray_LANES_PER_SLOT_8
  let v4 ← checkedRem i v3
  -- (self.words[i / Self::LANES_PER_SLOT] >> bit_pos) & Self::LANE_MASK
  let v9 ← StoragePackedArray_LANES_PER_SLOT_8
  let v10 ← checkedDiv i v9
  let l0 ← StorageMap_locate_u256_u256 (self.field Layout.StoragePackedArray_8 0) (self.field Layout.StoragePackedArray_8 0).slot v10
  let v11 ← load (Place.mk .storage l0.slot)
  let v13 ← StoragePackedArray_LANE_MASK_8
  return (v11 >>> (v4 * (UInt256.ofNat 8))) &&& v13

/-- `StoragePackedArray<8>::set` -/
def StoragePackedArray_set_8 (self : Place) (i : UInt256) (value : UInt256) : FeM Unit := do
  -- let bit_pos: u256 = (i % Self::LANES_PER_SLOT) * BITS
  let v4 ← StoragePackedArray_LANES_PER_SLOT_8
  let v5 ← checkedRem i v4
  let bit_pos := v5 * (UInt256.ofNat 8)
  -- let lane_mask_shifted: u256 = Self::LANE_MASK << bit_pos
  let v9 ← StoragePackedArray_LANE_MASK_8
  -- let word = mut self.words[i / Self::LANES_PER_SLOT]
  let v13 ← StoragePackedArray_LANES_PER_SLOT_8
  let v14 ← checkedDiv i v13
  let l0 ← StorageMap_locate_u256_u256 (self.field Layout.StoragePackedArray_8 0) (self.field Layout.StoragePackedArray_8 0).slot v14
  let v15 := (Place.mk .storage l0.slot)
  let word := v15
  -- word = (word & (U256_MASK ^ lane_mask_shifted)) | ((value & Self::LANE_MASK) << bit_pos)
  let v17 ← load word
  let v18 ← U256_MASK
  let v22 ← StoragePackedArray_LANE_MASK_8
  let v25 := (v17 &&& (v18 ^^^ (v9 <<< bit_pos))) ||| ((value &&& v22) <<< bit_pos)
  store word v25
  return ()

/-- `<Evm as Ctx>::caller` -/
def Evm_caller : FeM UInt256 := do
  -- Address { inner: ops::caller() }
  let v1 ← caller
  return v1

/-- `<Address as StorageKey>::encoded_len` -/
def Address_encoded_len (self : UInt256) : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `<Address as StorageKey>::write_key` -/
def Address_write_key (ptr : UInt256) (self : UInt256) : FeM Unit := do
  -- u256::write_key(ptr: ptr, self.inner)
  let _ ← u256_write_key ptr self
  return ()

/-- `std::evm::layout::mapping_slot<Address>` -/
def mapping_slot_Address (root : UInt256) (key : UInt256) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  -- let key_len = K::encoded_len(key)
  let key_len ← Address_encoded_len key
  -- ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate())
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  -- let buffer = unsafe { ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate()) }
  buffer := buffer_7
  -- let scratch = buffer.ptr()
  let scratch ← MemBuffer_ptr buffer
  -- K::write_key(ptr: scratch, key)
  let _ ← Address_write_key scratch key
  -- mstore(addr: ptr::offset_bytes(scratch, key_len.downcast_unchecked()), value: root)
  let v13 ← downcast_unchecked_u256_usize key_len
  let _ ← mstore (scratch + v13) root
  -- keccak256(ptr::MemSpan::from_raw_parts(ptr: scratch, len: (key_len + 32).downcast_unchecked()))
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<Address, u256> as StateIndex>::locate` -/
def StorageMap_locate_Address_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  -- Location { slot: mapping_slot(root, key), offset: 0 }
  let v3 ← mapping_slot_Address root key
  return Location.mk v3 (UInt256.ofNat 0)

/-- `Address::zero` -/
def Address_zero : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<(u256) as AbiSize>::HEAD_SIZE` -/
def tuple_HEAD_SIZE_u256 : FeM UInt256 := do
  let v0 ← abi_field_head_size_u256
  return v0

/-- `core::abi::dynamic_payload_size<u256>` -/
def dynamic_payload_size_u256 (value : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v1 ← u256_IS_DYNAMIC
  if v1 then
    -- return value.payload_size()
    let v2 ← payload_size_u256 value
    return v2
  else
    return UInt256.ofNat 0

/-- `<(u256) as AbiSize>::payload_size` -/
def tuple_payload_size_u256 (self : UInt256) : FeM UInt256 := do
  -- Self::HEAD_SIZE + dynamic_payload_size(self.0)
  let v1 ← tuple_HEAD_SIZE_u256
  let v3 ← dynamic_payload_size_u256 self
  return v1 + v3

/-- `core::abi::abi_payload_size<(u256)>` -/
def abi_payload_size_tuple_u256 (value : UInt256) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← tuple_payload_size_u256 value
  return v1

/-- `<(u256) as Encode>::encode` -/
def tuple_encode_Sol_u256 (self : UInt256) (out : AbiFrame) : FeM Unit := do
  -- encode_field<A, T0>(v0, out, head_pos: 0, tails: out.tail(Self::HEAD_SIZE))
  let v4 ← tuple_HEAD_SIZE_u256
  let v5 ← AbiFrame_tail out v4
  let _ ← encode_field_Sol_u256 self out (UInt256.ofNat 0) v5
  return ()

/-- `core::abi::encode_alloc<Sol, (u256)>` -/
def encode_alloc_Sol_tuple_u256 (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_tuple_u256 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← tuple_encode_Sol_u256 value r2
  return buffer

/-- `std::evm::effects::encode_abi_payload<(u256)>` -/
def encode_abi_payload_tuple_u256 (value : UInt256) : FeM MemBuffer := do
  -- encode_alloc<Sol, T>(value)
  let v1 ← encode_alloc_Sol_tuple_u256 value
  return v1

/-- `<Address as TopicValue>::as_topic` -/
def Address_as_topic (self : UInt256) : FeM UInt256 := do
  return self

/-- `<Evm as Log>::log3` -/
def Evm_log3 (data : MemSlice_u8) (topic0 : UInt256) (topic1 : UInt256) (topic2 : UInt256) : FeM Unit := do
  -- ops::log3(
  let v5 ← MemSlice_ptr_u8 data
  let v6 ← MemSlice_len_u8 data
  let _ ← logData v5 v6 #[topic0, topic1, topic2]
  return ()

/-- `<StorageMap<Address, StorageMap<Address, u256>> as StateIndex>::locate` -/
def StorageMap_locate_Address_StorageMap_Address_u256 (self : Place) (root : UInt256) (key : UInt256) : FeM Location := do
  -- Location { slot: mapping_slot(root, key), offset: 0 }
  let v3 ← mapping_slot_Address root key
  return Location.mk v3 (UInt256.ofNat 0)

/-- `<Address as AbiSize>::IS_DYNAMIC` -/
def Address_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<Address as AbiSize>::HEAD_SIZE` -/
def Address_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `core::abi::abi_field_head_size<Address>` -/
def abi_field_head_size_Address : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← Address_IS_DYNAMIC
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← Address_HEAD_SIZE
    return v4

/-- `<Address as Decode>::decode_from<CallData>` -/
def Address_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM UInt256 := do
  -- let word = input.word_at(pos)
  let word ← CallData_word_at input pos
  -- if word >> 160 != 0 {
  let v7 := decide ((word >>> (UInt256.ofNat 160)) ≠ (UInt256.ofNat 0))
  if v7 then
    -- ops::revert_empty()
    let v8 ← revertEmpty
    nomatch v8
  else
    return word

/-- `core::abi::Decode::decode_from_bounded<Address, Sol, CallData>` -/
def decode_from_bounded_Address_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- checked_frame_end<A>(pos, Self::HEAD_SIZE, input_len)
  let v3 ← Address_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- Self::decode_from(input, pos)
  let v5 ← Address_decode_from_Sol_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, Address, CallData>` -/
def decode_field_from_prechecked_head_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← decode_from_bounded_Address_Sol_CallData input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← Address_decode_from_Sol_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, Address, CallData>` -/
def decode_field_from_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- return T::decode_from(input, checked_tail<A>(base, tail_rel))
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← Address_decode_from_Sol_CallData input v6
    return v7
  else
    -- T::decode_from(input, head_pos)
    let v10 ← Address_decode_from_Sol_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, Address, CallData>` -/
def decode_msg_field_from_Sol_Address_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v4 ← Address_IS_DYNAMIC
  if v4 then
    -- return decode_field_from_prechecked_head<A, T, I>(input, base: base, head_pos: head_pos, input_len: input_len)
    let v5 ← decode_field_from_prechecked_head_Sol_Address_CallData input base head_pos input_len
    return v5
  else
    -- decode_field_from<A, T, I>(input, base: base, head_pos: head_pos)
    let v8 ← decode_field_from_Sol_Address_CallData input base head_pos
    return v8

/-- `<bool as AbiSize>::IS_DYNAMIC` -/
def bool_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<bool as AbiSize>::HEAD_SIZE` -/
def bool_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `<bool as Encode>::encode` -/
def bool_encode_Sol (self : Bool) (out : AbiFrame) : FeM Unit := do
  -- out.store_word(at: 0, value: self as u256)
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) (Bool.toUInt256 self)
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<bool>` -/
def const_bool : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<bool>` -/
def payload_size_bool (self : Bool) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← bool_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<bool>` -/
def abi_payload_size_bool (value : Bool) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← payload_size_bool value
  return v1

/-- `core::abi::encode_alloc<Sol, bool>` -/
def encode_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_bool value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← bool_encode_Sol value r1
  return buffer

/-- `core::abi::abi_field_size<bool>` -/
def abi_field_size_bool (value : Bool) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← payload_size_bool value
  -- if T::IS_DYNAMIC {
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<bool>` -/
def abi_single_root_size_bool (value : Bool) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_bool value
  return v1

/-- `core::abi::abi_field_head_size<bool>` -/
def abi_field_head_size_bool : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← bool_IS_DYNAMIC
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← bool_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<bool>` -/
def abi_single_root_head_size_bool : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_bool
  return v0

/-- `core::abi::encode_field<Sol, bool>` -/
def encode_field_Sol_bool (value : Bool) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← bool_IS_DYNAMIC
  if v3 then
    -- let size = value.payload_size()
    let size ← payload_size_bool value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← bool_encode_Sol value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← bool_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← bool_encode_Sol value r2
    return tails

/-- `core::abi::encode_single_root_alloc<Sol, bool>` -/
def encode_single_root_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_bool value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_bool
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_bool value out (UInt256.ofNat 0) v13
  return buffer

/-- `core::abi::encode_return_alloc<Sol, bool>` -/
def encode_return_alloc_Sol_bool (value : Bool) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← const_bool
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v2 ← encode_alloc_Sol_bool value
    return v2
  else
    -- encode_single_root_alloc<A, T>(value)
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
  -- if !T::IS_DYNAMIC {
  let v2 ← bool_IS_DYNAMIC
  if !v2 then
    -- if T::HEAD_SIZE <= 32 {
    let v4 ← bool_HEAD_SIZE
    if decide (v4.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_7 ← FixedMemBuffer_alloc_32
      out_r0 := out_7
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← bool_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let _ ← bool_encode_Sol value r4
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r0
      let v17 ← bool_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← bool_HEAD_SIZE
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r1 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← bool_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let _ ← bool_encode_Sol value r5
        -- self.return_data(
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← bool_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v42 ← bool_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let out_44 ← MemBuffer_new v43
        out_r2 := out_44
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← bool_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let _ ← bool_encode_Sol value r6
        -- self.return_data(data: out.span())
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    -- let out = encode_return_alloc<A, T>(value)
    let out_57 ← encode_return_alloc_Sol_bool value
    out_r3 := out_57
    -- self.return_bytes(data: out.span())
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
  -- var head_pos = pos
  head_pos := pos
  -- let v0: T0 = decode_field_from_prechecked_head<A, T0, I>(input, base: pos, head_pos: head_pos, input_len: input_len)
  let v0 ← decode_field_from_prechecked_head_Sol_u256_CallData input pos head_pos input_len
  -- head_pos += abi_field_head_size<T0>()
  let v9 ← abi_field_head_size_u256
  let head_pos_11 ← checkedAdd head_pos v9
  head_pos := head_pos_11
  -- let v1: T1 = decode_field_from_prechecked_head<A, T1, I>(input, base: pos, head_pos: head_pos, input_len: input_len)
  let v1 ← decode_field_from_prechecked_head_Sol_u256_CallData input pos head_pos input_len
  return (v0, v1)

/-- `core::abi::decode_frame_from<Sol, (u256, u256), CallData>` -/
def decode_frame_from_Sol_tuple_u256_u256_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  -- checked_frame_end<A>(pos, T::HEAD_SIZE, input_len)
  let v3 ← tuple_HEAD_SIZE_u256_u256
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- T::decode_from_prechecked_head(input, pos, input_len)
  let v5 ← tuple_decode_from_prechecked_head_Sol_u256_u256_CallData input pos input_len
  return v5

/-- `std::abi::sol::decode_input_at<(u256, u256), CallData>` -/
def decode_input_at_tuple_u256_u256_CallData (input : CallData) (base : UInt256) : FeM (UInt256 × UInt256) := do
  -- let input_len = input.len()
  let input_len ← len_CallData input
  -- decode_frame_from<Sol, T, I>(input, base, input_len)
  let v4 ← decode_frame_from_Sol_tuple_u256_u256_CallData input base input_len
  return v4

/-- `Evm::decode_args<(u256, u256)>` -/
def Evm_decode_args_tuple_u256_u256 : FeM (UInt256 × UInt256) := do
  let mut r0 : CallData := default
  -- decode_input_at(self.input(), size_of<Sol::Selector>() as u256)
  let v1 ← Evm_input
  r0 := v1
  let v5 ← decode_input_at_tuple_u256_u256_CallData r0 (UInt256.ofNat 4)
  return v5

/-- `MemBuffer::len` -/
def MemBuffer_len (self : MemBuffer) : FeM UInt256 := do
  return self.len

/-- `std::evm::effects::revert<u256>` -/
def revert_u256 (value : UInt256) : FeM Empty := do
  let mut result : MemBuffer := default
  -- let result = encode_single_root_alloc<Sol, T>(value)
  let result_1 ← encode_single_root_alloc_Sol_u256 value
  result := result_1
  -- ops::revert(offset: result.ptr(), len: result.len() as u256)
  let v4 ← MemBuffer_ptr result
  let v6 ← MemBuffer_len result
  let v8 ← revertData v4 v6
  nomatch v8

structure Wei where
  amount : UInt256
deriving Inhabited

/-- `<i256 as IntWord>::to_word` -/
def i256_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<i8 as IntWord>::IS_SIGNED` -/
def i8_IS_SIGNED : FeM Bool := do
  return true

/-- `core::num::U8_MASK` -/
def U8_MASK : FeM UInt256 := do
  return UInt256.ofNat 255

/-- `<i8 as IntWord>::MASK` -/
def i8_MASK : FeM UInt256 := do
  let v0 ← U8_MASK
  return v0

/-- `core::num::I8_SIGN_BIT` -/
def I8_SIGN_BIT : FeM UInt256 := do
  return UInt256.ofNat 128

/-- `<i8 as IntWord>::SIGN_BIT` -/
def i8_SIGN_BIT : FeM UInt256 := do
  let v0 ← I8_SIGN_BIT
  return v0

/-- `<i8 as IntWord>::from_word` -/
def i8_from_word (word : UInt256) : FeM UInt256 := do
  return IntTy.cast (.u 256) (.i 8) word

/-- `<i256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i256_i8 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← i256_to_word self
  -- if T::IS_SIGNED {
  let v4 ← i8_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← i8_MASK
    let v6 ← i8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i8_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← i8_MASK
    let v12 ← i8_from_word (word &&& v10)
    join v12

/-- `<u256 as IntDowncast>::downcast_unchecked` -/
def downcast_unchecked_u256_i256 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u256_to_word self
  -- T::from_word(word)
  let v3 ← i256_from_word word
  return v3

/-- `<Evm as Ctx>::address` -/
def Evm_address : FeM UInt256 := do
  -- Address { inner: ops::address() }
  let v1 ← selfAddress
  return v1

/-- `Address::word` -/
def Address_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `std::evm::units::wei` -/
def wei (n : UInt256) : FeM Wei := do
  return Wei.mk n

/-- `<Evm as Ctx>::value` -/
def Evm_value : FeM Wei := do
  -- super::units::wei(ops::callvalue())
  let v1 ← callvalue
  let v2 ← wei v1
  return v2

/-- `Wei::as_wei` -/
def Wei_as_wei (self : Wei) : FeM UInt256 := do
  return self.amount

/-- `<Evm as Ctx>::origin` -/
def Evm_origin : FeM UInt256 := do
  -- Address { inner: ops::origin() }
  let v1 ← origin
  return v1

/-- `<Evm as Ctx>::block_number` -/
def Evm_block_number : FeM UInt256 := do
  -- ops::number()
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
  -- A::encoded_len(self.0) + B::encoded_len(self.1)
  let v2 ← u256_encoded_len self.fst
  let v4 ← u256_encoded_len self.snd
  let v5 ← checkedAdd v2 v4
  return v5

/-- `<(u256, u256) as StorageKey>::write_key` -/
def tuple_write_key_u256_u256 (ptr : UInt256) (self : (UInt256 × UInt256)) : FeM Unit := do
  -- let a_len = A::encoded_len(self.0)
  let a_len ← u256_encoded_len self.fst
  -- A::write_key(ptr: ptr, self.0)
  let _ ← u256_write_key ptr self.fst
  -- B::write_key(ptr: ptr::offset_bytes(ptr, a_len.downcast_unchecked()), self.1)
  let v7 ← downcast_unchecked_u256_usize a_len
  let _ ← u256_write_key (ptr + v7) self.snd
  return ()

/-- `std::evm::layout::mapping_slot<(u256, u256)>` -/
def mapping_slot_tuple_u256_u256 (root : UInt256) (key : (UInt256 × UInt256)) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  -- let key_len = K::encoded_len(key)
  let key_len ← tuple_encoded_len_u256_u256 key
  -- ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate())
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  -- let buffer = unsafe { ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate()) }
  buffer := buffer_7
  -- let scratch = buffer.ptr()
  let scratch ← MemBuffer_ptr buffer
  -- K::write_key(ptr: scratch, key)
  let _ ← tuple_write_key_u256_u256 scratch key
  -- mstore(addr: ptr::offset_bytes(scratch, key_len.downcast_unchecked()), value: root)
  let v13 ← downcast_unchecked_u256_usize key_len
  let _ ← mstore (scratch + v13) root
  -- keccak256(ptr::MemSpan::from_raw_parts(ptr: scratch, len: (key_len + 32).downcast_unchecked()))
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<(u256, u256), bool> as StateIndex>::locate` -/
def StorageMap_locate_tuple_u256_u256_bool (self : Place) (root : UInt256) (key : (UInt256 × UInt256)) : FeM Location := do
  -- Location { slot: mapping_slot(root, key), offset: 0 }
  let v3 ← mapping_slot_tuple_u256_u256 root key
  return Location.mk v3 (UInt256.ofNat 0)

/-- `<bool as StorageKey>::encoded_len` -/
def bool_encoded_len (self : Bool) : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `<bool as StorageKey>::write_key` -/
def bool_write_key (ptr : UInt256) (self : Bool) : FeM Unit := do
  let join : UInt256 → FeM Unit := fun v2 => do
    -- u256::write_key(ptr: ptr, if self { 1 } else { 0 })
    let _ ← u256_write_key ptr v2
    return ()
  if self then
    join (UInt256.ofNat 1)
  else
    join (UInt256.ofNat 0)

/-- `std::evm::layout::mapping_slot<bool>` -/
def mapping_slot_bool (root : UInt256) (key : Bool) : FeM UInt256 := do
  let mut buffer : MemBuffer := default
  -- let key_len = K::encoded_len(key)
  let key_len ← bool_encoded_len key
  -- ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate())
  let v5 ← checkedAdd key_len (UInt256.ofNat 32)
  let v6 ← downcast_saturate_u256_usize v5
  let buffer_7 ← MemBuffer_new_uninit v6
  -- let buffer = unsafe { ptr::MemBuffer::new_uninit((key_len + 32).downcast_saturate()) }
  buffer := buffer_7
  -- let scratch = buffer.ptr()
  let scratch ← MemBuffer_ptr buffer
  -- K::write_key(ptr: scratch, key)
  let _ ← bool_write_key scratch key
  -- mstore(addr: ptr::offset_bytes(scratch, key_len.downcast_unchecked()), value: root)
  let v13 ← downcast_unchecked_u256_usize key_len
  let _ ← mstore (scratch + v13) root
  -- keccak256(ptr::MemSpan::from_raw_parts(ptr: scratch, len: (key_len + 32).downcast_unchecked()))
  let v17 ← checkedAdd key_len (UInt256.ofNat 32)
  let v18 ← downcast_unchecked_u256_usize v17
  let v19 ← MemSlice_from_raw_parts_ref_u8 scratch v18
  let v21 ← keccak256_1 v19
  let _ ← MemBuffer_drop buffer
  return v21

/-- `<StorageMap<bool, u256> as StateIndex>::locate` -/
def StorageMap_locate_bool_u256 (self : Place) (root : UInt256) (key : Bool) : FeM Location := do
  -- Location { slot: mapping_slot(root, key), offset: 0 }
  let v3 ← mapping_slot_bool root key
  return Location.mk v3 (UInt256.ofNat 0)

/-- `std::evm::effects::RawOps::calldata_ptr<Evm, [u256; 2]>` -/
def calldata_ptr_Evm_array_u256_2 (offset : UInt256) : FeM UInt256 := do
  return offset

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
  return UInt256.ofNat 0

/-- `<[u256; 2] as Collection>::has` -/
def array_has_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM Bool := do
  return decide (c.toNat < (UInt256.ofNat 2).toNat)

/-- `<[u256; 2] as Collection>::at`, its `ref` body -/
def array_at_ref_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM UInt256 := do
  -- mut? self[c]
  checkIndex c 2
  return self[c.toNat]!

/-- `<usize as WrappingAdd>::wrapping_add` -/
def usize_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return self + other

/-- `<[u256; 2] as Collection>::next` -/
def array_next_u256_2 (self : (Vector UInt256 2)) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
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
  -- ptr::MemSpan::from_raw_parts(
  let v5 ← MemSlice_from_raw_parts_ref_u8 (self.data + (UInt256.ofNat 32)) self.len
  return v5

/-- `core::panics::PANIC_STORAGE_ENCODING` -/
def PANIC_STORAGE_ENCODING : FeM UInt256 := do
  return UInt256.ofNat 34

/-- `Panic::from` -/
def Panic_from (code : UInt256) : FeM Panic := do
  return Panic.mk code

/-- `<Panic as AbiSize>::HEAD_SIZE` -/
def Panic_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Panic as AbiSize>::payload_size` -/
def Panic_payload_size (self : Panic) : FeM UInt256 := do
  let v1 ← Panic_HEAD_SIZE
  let v3 ← dynamic_payload_size_u256 self.code
  let v4 ← checkedAdd v1 v3
  return v4

/-- `core::abi::abi_payload_size<Panic>` -/
def abi_payload_size_Panic (value : Panic) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← Panic_payload_size value
  return v1

/-- `MemBuffer::set_len` -/
def MemBuffer_set_len (self : MemBuffer) (len_1 : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  -- if len > self.capacity {
  if decide (self.capacity.toNat < len_1.toNat) then
    -- index_out_of_bounds()
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    -- self.len = len
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
  let _ ← encode_field_Sol_u256 self.code v1 r1 r0
  return ()

/-- `std::evm::effects::encode_calldata<Panic>` -/
def encode_calldata_Panic (selector : UInt256) (message : Panic) : FeM MemBuffer := do
  let mut message := message
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let selector_size = size_of<Sol::Selector>()
  let selector_size := (UInt256.ofNat 4)
  -- let payload_size = abi_payload_size(message)
  let payload_size ← abi_payload_size_Panic message
  -- let len = selector_size + payload_size.downcast_unchecked()
  let v7 ← downcast_unchecked_u256_usize payload_size
  let len := selector_size + v7
  -- let capacity = if M::HEAD_SIZE < 28 && len < 32 { 32 } else { len }
  let v10 ← Panic_HEAD_SIZE
  let join : Panic → MemBuffer → AbiFrame → UInt256 → FeM MemBuffer := fun message buffer r2 capacity => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    -- MemBuffer::new_uninit(capacity)
    let buffer_19 ← MemBuffer_new_uninit capacity
    -- var buffer = unsafe { MemBuffer::new_uninit(capacity) }
    buffer := buffer_19
    -- let payload = ptr::offset_bytes(buffer.ptr(), selector_size)
    let v22 ← MemBuffer_ptr buffer
    let payload := v22 + selector_size
    -- buffer.set_len(len)
    let (v26, v26_0) ← MemBuffer_set_len buffer len
    buffer := v26_0
    -- ops::mstore(addr: buffer.ptr(), value: (selector as u256) << 224)
    let v28 ← MemBuffer_ptr buffer
    let _ ← mstore v28 (selector <<< (UInt256.ofNat 224))
    -- ptr::zero_bytes(payload, payload_size.downcast_unchecked())
    let v34 ← downcast_unchecked_u256_usize payload_size
    let _ ← zero_bytes payload v34
    -- message.encode(AbiFrame::from_raw_parts(ptr: payload, len: payload_size))
    let v36 ← AbiFrame_from_raw_parts payload payload_size
    r2 := v36
    let _ ← Panic_encode message r2
    return buffer
  let join_1 : Panic → MemBuffer → AbiFrame → FeM MemBuffer := fun message buffer r2 => do
    let mut message := message
    let mut buffer := buffer
    let mut r2 := r2
    join message buffer r2 len
  -- let capacity = if M::HEAD_SIZE < 28 && len < 32 { 32 } else { len }
  if decide (v10.toNat < (UInt256.ofNat 28).toNat) then
    -- let capacity = if M::HEAD_SIZE < 28 && len < 32 { 32 } else { len }
    if decide (len.toNat < (UInt256.ofNat 32).toNat) then
      join message buffer r2 (UInt256.ofNat 32)
    else
      join_1 message buffer r2
  else
    join_1 message buffer r2

/-- `std::evm::effects::encode_error<Panic>` -/
def encode_error_Panic (error : Panic) : FeM MemBuffer := do
  -- encode_calldata(<T as ErrorVariant<Sol>>::SELECTOR, error)
  let v1 := (UInt256.ofNat 1313373041) -- a constant, as the compiler evaluates it
  let v3 ← encode_calldata_Panic v1 error
  return v3

/-- `std::evm::effects::revert_error<Panic>` -/
def revert_error_Panic (error : Panic) : FeM Empty := do
  let mut buffer : MemBuffer := default
  -- let buffer = encode_error(error)
  let buffer_2 ← encode_error_Panic error
  buffer := buffer_2
  -- ops::revert(offset: buffer.ptr(), len: buffer.len() as u256)
  let v5 ← MemBuffer_ptr buffer
  let v7 ← MemBuffer_len buffer
  let v9 ← revertData v5 v7
  nomatch v9

/-- `std::evm::crypto::keccak_words<1>` -/
def keccak_words_1 (words : (Vector UInt256 1)) : FeM UInt256 := do
  return keccakWords words

/-- `std::evm::layout::array_data_slot` -/
def array_data_slot (root : UInt256) : FeM UInt256 := do
  let mut r0 : (Vector UInt256 1) := default
  -- keccak_words([root])
  r0 := #v[root]
  let v4 ← keccak_words_1 r0
  return v4

/-- `<u256 as WrappingAdd>::wrapping_add` -/
def u256_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return self + other

/-- `<u256 as Bounded>::max` -/
def u256_max : FeM UInt256 := do
  return UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639935

/-- `<DynString as Drop>::drop` -/
def DynString_drop (self : DynString) : FeM Unit := do
  -- ptr::dealloc(self.data)
  let _ ← dealloc_u8 self.data
  return ()

/-- `<u8 as SolPacked>::BITS` -/
def u8_BITS : FeM UInt256 := do
  return UInt256.ofNat 8

/-- `std::evm::sol_slot::field_shift` -/
def field_shift (offset : UInt256) (bits : UInt256) : FeM UInt256 := do
  let join : FeM UInt256 := do
    -- revert_empty()
    let v17 ← revertEmpty
    nomatch v17
  -- if offset >= 32 || bits > 256 - offset * 8 {
  if decide ((UInt256.ofNat 32).toNat ≤ offset.toNat) then
    join
  else
    -- if offset >= 32 || bits > 256 - offset * 8 {
    let v9 ← checkedMul offset (UInt256.ofNat 8)
    let v10 ← checkedSub (UInt256.ofNat 256) v9
    if decide (v10.toNat < bits.toNat) then
      join
    else
      -- offset * 8
      let v16 ← checkedMul offset (UInt256.ofNat 8)
      return v16

/-- `<u8 as SolPacked>::to_packed` -/
def u8_to_packed (self : UInt256) : FeM UInt256 := do
  return self

/-- `std::evm::sol_slot::field_mask` -/
def field_mask (bits : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if bits >= 256 { u256::max() } else { ((1 as u256) << bits) - 1 }
  if decide ((UInt256.ofNat 256).toNat ≤ bits.toNat) then
    -- u256::max()
    let v5 ← u256_max
    join v5
  else
    -- ((1 as u256) << bits) - 1
    let v12 ← checkedSub ((UInt256.ofNat 1) <<< bits) (UInt256.ofNat 1)
    join v12

/-- `SolWord::set<u8>` -/
def SolWord_set_u8 (self : SolWord) (offset : UInt256) (value : UInt256) : FeM (Unit × SolWord) := do
  let mut self := self
  -- let shift = field_shift(offset, bits: T::BITS)
  let v3 ← u8_BITS
  let shift ← field_shift offset v3
  -- let mask = field_mask(T::BITS) << shift
  let v6 ← u8_BITS
  let v7 ← field_mask v6
  let mask := v7 <<< shift
  -- self.word = (self.word & !mask) | ((value.to_packed() << shift) & mask)
  let v12 := self.word &&& ~~~mask
  let v13 ← u8_to_packed value
  self := { self with word := (v12 ||| ((v13 <<< shift) &&& mask)) }
  return ((), self)

/-- `core::abi::round_up_words` -/
def round_up_words (len : UInt256) : FeM UInt256 := do
  -- if len == 0 {
  if decide (len = (UInt256.ofNat 0)) then
    return UInt256.ofNat 0
  else
    -- (((len - 1) / 32) + 1) * 32
    let v9 ← checkedSub len (UInt256.ofNat 1)
    let v11 ← checkedDiv v9 (UInt256.ofNat 32)
    let v13 ← checkedAdd v11 (UInt256.ofNat 1)
    let v15 ← checkedMul v13 (UInt256.ofNat 32)
    return v15

/-- `core::abi::mem_word_at` -/
def mem_word_at (base : UInt256) (offset : UInt256) : FeM UInt256 := do
  -- from_be(*ptr::cast<u8, u256>(ptr::offset_bytes(base, word_to_usize(offset))))
  let v2 ← word_to_usize offset
  let v4 ← cast_u8_u256 (base + v2)
  let v5 ← mload v4
  return v5

/-- `DynString::from_validated_abi_tail` -/
def DynString_from_validated_abi_tail (data : UInt256) (len : UInt256) (size : UInt256) : FeM DynString := do
  return DynString.mk data len size

/-- `DynString::from_abi_tail` -/
def DynString_from_abi_tail (data : UInt256) (len : UInt256) (size : UInt256) : FeM DynString := do
  -- let expected_size = 32 + round_up_words(len as u256)
  let v6 ← round_up_words len
  let expected_size ← checkedAdd (UInt256.ofNat 32) v6
  let join : FeM DynString := do
    -- super::panic()
    let v19 ← trap
    nomatch v19
  -- if size != expected_size || mem_word_at(base: data, offset: 0) != len as u256 {
  if decide (size ≠ expected_size) then
    join
  else
    -- if size != expected_size || mem_word_at(base: data, offset: 0) != len as u256 {
    let v12 ← mem_word_at data (UInt256.ofNat 0)
    if decide (v12 ≠ len) then
      join
    else
      -- Self::from_validated_abi_tail(data, len, size)
      let v18 ← DynString_from_validated_abi_tail data len size
      return v18

/-- `<Evm as Ctx>::selfbalance` -/
def Evm_selfbalance : FeM Wei := do
  -- super::units::wei(ops::selfbalance())
  let v1 ← selfbalance
  let v2 ← wei v1
  return v2

/-- `<Address as Eq>::eq` -/
def Address_eq (self : UInt256) (other : UInt256) : FeM Bool := do
  return decide (self = other)

/-- `core::ops::Eq::ne<Address, Address>` -/
def ne_Address_Address (self : UInt256) (other : UInt256) : FeM Bool := do
  -- !self.eq(other)
  let v2 ← Address_eq self other
  return !v2

/-- `MemBuffer::with_capacity` -/
def MemBuffer_with_capacity (capacity : UInt256) : FeM MemBuffer := do
  -- Self {
  let v1 ← alloc_bytes capacity
  return MemBuffer.mk v1 (UInt256.ofNat 0) capacity

/-- `MemBuffer::empty` -/
def MemBuffer_empty : FeM MemBuffer := do
  -- Self::with_capacity(0)
  let v1 ← MemBuffer_with_capacity (UInt256.ofNat 0)
  return v1

/-- `MemSlice<u8>::empty` -/
def MemSlice_empty_u8 : FeM MemSlice_u8 := do
  -- Self::from_raw_parts(ptr: cast<u8, T>(alloc_bytes(0)), len: 0)
  let v1 ← alloc_bytes (UInt256.ofNat 0)
  let v2 ← cast_u8_u8 v1
  let v4 ← MemSlice_from_raw_parts_ref_u8 v2 (UInt256.ofNat 0)
  return v4

/-- `MemBuffer::capacity` -/
def MemBuffer_capacity (self : MemBuffer) : FeM UInt256 := do
  return self.capacity

/-- `std::evm::effects::finish_raw_call` -/
def finish_raw_call (success : Bool) (ret_1 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut ret_1 := ret_1
  -- let returndata_len = ops::returndatasize()
  let returndata_len ← returndatasize
  -- let written = if returndata_len < ret.capacity() as u256 {
  let v6 ← MemBuffer_capacity ret_1
  let join : MemBuffer → UInt256 → FeM (RawCallOutcome × MemBuffer) := fun ret_1 written => do
    let mut ret_1 := ret_1
    -- ret.set_len(written)
    let (v15, v15_0) ← MemBuffer_set_len ret_1 written
    ret_1 := v15_0
    return ((RawCallOutcome.mk success returndata_len), ret_1)
  -- let written = if returndata_len < ret.capacity() as u256 {
  if decide (returndata_len.toNat < v6.toNat) then
    -- returndata_len.downcast_unchecked()
    let v9 ← downcast_unchecked_u256_usize returndata_len
    join ret_1 v9
  else
    -- ret.capacity()
    let v12 ← MemBuffer_capacity ret_1
    join ret_1 v12

/-- `std::evm::effects::Call::raw_call<Evm>` -/
def raw_call_Evm (addr : UInt256) (gas_1 : UInt256) (value : Wei) (args : MemSlice_u8) (ret_5 : MemBuffer) : FeM (RawCallOutcome × MemBuffer) := do
  let mut ret_5 := ret_5
  -- ops::call(
  let v7 ← Wei_as_wei value
  let v8 ← MemSlice_ptr_u8 args
  let v9 ← MemSlice_len_u8 args
  let v12 ← MemBuffer_ptr ret_5
  let v14 ← MemBuffer_capacity ret_5
  let v16 ← Fe.EVM.call gas_1 addr v7 v8 v9 v12 v14
  -- finish_raw_call(success, mut ret)
  let (v21, v21_0) ← finish_raw_call (decide (v16 ≠ (UInt256.ofNat 0))) ret_5
  ret_5 := v21_0
  return (v21, ret_5)

/-- `std::evm::effects::Call::send_value<Evm>` -/
def send_value_Evm (addr : UInt256) (gas_1 : UInt256) (value : Wei) : FeM RawCallOutcome := do
  let mut addr := addr
  let mut ret : MemBuffer := default
  -- var ret = MemBuffer::empty()
  let ret_3 ← MemBuffer_empty
  ret := ret_3
  -- self.raw_call(addr, gas, value, args: MemSpan::empty(), ret: mut ret)
  let v5 ← MemSlice_empty_u8
  let addr_7 := addr
  let (v9, v9_0) ← raw_call_Evm addr_7 gas_1 value v5 ret
  ret := v9_0
  let _ ← MemBuffer_drop ret
  return v9

/-- `RawCallOutcome::success` -/
def RawCallOutcome_success (self : RawCallOutcome) : FeM Bool := do
  return self.success

/-- `<Evm as Log>::log2` -/
def Evm_log2 (data : MemSlice_u8) (topic0 : UInt256) (topic1 : UInt256) : FeM Unit := do
  -- ops::log2(offset: data.ptr(), len: data.len() as u256, topic0: topic0, topic1: topic1)
  let v4 ← MemSlice_ptr_u8 data
  let v5 ← MemSlice_len_u8 data
  let _ ← logData v4 v5 #[topic0, topic1]
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
  return UInt256.ofNat 0

/-- `<u8 as IntWord>::from_word` -/
def u8_from_word (word : UInt256) : FeM UInt256 := do
  return truncate 8 word

/-- `<u256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u256_u8 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u256_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u8_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u8_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u8_MASK
    let v12 ← u8_from_word (word &&& v10)
    join v12

/-- `<u8 as SolPacked>::from_packed` -/
def u8_from_packed (bits : UInt256) : FeM UInt256 := do
  -- bits.downcast_truncate()
  let v1 ← downcast_truncate_u256_u8 bits
  return v1

/-- `SolWord::get<u8>` -/
def SolWord_get_u8 (self : SolWord) (offset : UInt256) : FeM UInt256 := do
  -- let shift = field_shift(offset, bits: T::BITS)
  let v2 ← u8_BITS
  let shift ← field_shift offset v2
  -- T::from_packed((self.word >> shift) & field_mask(T::BITS))
  let v7 ← u8_BITS
  let v8 ← field_mask v7
  let v10 ← u8_from_packed ((self.word >>> shift) &&& v8)
  return v10

/-- `<DynString as AbiSize>::IS_DYNAMIC` -/
def DynString_IS_DYNAMIC : FeM Bool := do
  return true

/-- `<DynString as AbiSize>::HEAD_SIZE` -/
def DynString_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `DynString::encoded_span` -/
def DynString_encoded_span (self : DynString) : FeM MemSlice_u8 := do
  -- ptr::MemSpan::from_raw_parts(ptr: self.data, len: word_to_usize(self.size))
  let v3 ← word_to_usize self.size
  let v4 ← MemSlice_from_raw_parts_ref_u8 self.data v3
  return v4

/-- `core::ptr::copy_raw` -/
def copy_raw (dest : UInt256) (source : UInt256) (len : UInt256) : FeM Unit := do
  -- copy_mem(dest, source, len)
  let _ ← mcopy dest source len
  return ()

/-- `core::ptr::copy` -/
def copy (dest : UInt256) (source : MemSlice_u8) : FeM Unit := do
  -- copy_raw(dest, source: source.ptr(), len: source.len())
  let v2 ← MemSlice_ptr_u8 source
  let v3 ← MemSlice_len_u8 source
  let _ ← copy_raw dest v2 v3
  return ()

/-- `AbiFrame::copy_from` -/
def AbiFrame_copy_from (self : AbiFrame) («at» : UInt256) (source : MemSlice_u8) : FeM Unit := do
  -- let dest = self.slice(offset: at, len: source.len() as u256)
  let v3 ← MemSlice_len_u8 source
  let dest ← AbiFrame_slice self «at» v3
  -- ptr::copy(dest: dest.ptr, source)
  let _ ← copy dest.ptr source
  return ()

/-- `<DynString as Encode>::encode` -/
def DynString_encode_Sol (self : DynString) (out : AbiFrame) : FeM Unit := do
  -- out.copy_from(at: 0, self.encoded_span())
  let v3 ← DynString_encoded_span self
  let _ ← AbiFrame_copy_from out (UInt256.ofNat 0) v3
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<DynString>` -/
def const_DynString : FeM Bool := do
  return false

/-- `DynString::encoded_size` -/
def DynString_encoded_size (self : DynString) : FeM UInt256 := do
  return self.size

/-- `<DynString as AbiSize>::payload_size` -/
def DynString_payload_size (self : DynString) : FeM UInt256 := do
  -- self.encoded_size()
  let v1 ← DynString_encoded_size self
  return v1

/-- `core::abi::abi_payload_size<DynString>` -/
def abi_payload_size_DynString (value : DynString) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← DynString_payload_size value
  return v1

/-- `core::abi::encode_alloc<Sol, DynString>` -/
def encode_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_DynString value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← DynString_encode_Sol value r2
  let _ ← DynString_drop value
  return buffer

/-- `core::abi::abi_field_size<DynString>` -/
def abi_field_size_DynString (value : DynString) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← DynString_payload_size value
  -- if T::IS_DYNAMIC {
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<DynString>` -/
def abi_single_root_size_DynString (value : DynString) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_DynString value
  return v1

/-- `core::abi::abi_field_head_size<DynString>` -/
def abi_field_head_size_DynString : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← DynString_IS_DYNAMIC
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← DynString_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<DynString>` -/
def abi_single_root_head_size_DynString : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_DynString
  return v0

/-- `core::abi::encode_field<Sol, DynString>` -/
def encode_field_Sol_DynString (value : DynString) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← DynString_IS_DYNAMIC
  if v3 then
    -- let size = value.payload_size()
    let size ← DynString_payload_size value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← DynString_encode_Sol value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← DynString_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← DynString_encode_Sol value r2
    return tails

/-- `core::abi::encode_single_root_alloc<Sol, DynString>` -/
def encode_single_root_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_DynString value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_DynString
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_DynString value out (UInt256.ofNat 0) v13
  let _ ← DynString_drop value
  return buffer

/-- `core::abi::encode_return_alloc<Sol, DynString>` -/
def encode_return_alloc_Sol_DynString (value : DynString) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← const_DynString
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v3 ← encode_alloc_Sol_DynString value
    return v3
  else
    -- encode_single_root_alloc<A, T>(value)
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
  -- if !T::IS_DYNAMIC {
  let v1 ← DynString_IS_DYNAMIC
  if !v1 then
    -- if T::HEAD_SIZE <= 32 {
    let v3 ← DynString_HEAD_SIZE
    if decide (v3.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← DynString_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let _ ← DynString_encode_Sol value r5
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← DynString_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← DynString_HEAD_SIZE
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← DynString_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let _ ← DynString_encode_Sol value r6
        -- self.return_data(
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← DynString_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v43 ← DynString_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← DynString_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let _ ← DynString_encode_Sol value r7
        -- self.return_data(data: out.span())
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    -- let out = encode_return_alloc<A, T>(value)
    let value_59 := value
    let out_61 ← encode_return_alloc_Sol_DynString value_59
    out_r4 := out_61
    -- self.return_bytes(data: out.span())
    let v64 ← MemBuffer_span_ref out_r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

/-- `<u8 as AbiSize>::IS_DYNAMIC` -/
def u8_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<u8 as AbiSize>::HEAD_SIZE` -/
def u8_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `<u8 as Encode>::encode` -/
def u8_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  -- out.store_word(at: 0, value: self as u256)
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) self
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<u8>` -/
def const_u8 : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<u8>` -/
def payload_size_u8 (self : UInt256) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← u8_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<u8>` -/
def abi_payload_size_u8 (value : UInt256) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← payload_size_u8 value
  return v1

/-- `core::abi::encode_alloc<Sol, u8>` -/
def encode_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut r1 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_u8 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r1 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← u8_encode_Sol value r1
  return buffer

/-- `core::abi::abi_field_size<u8>` -/
def abi_field_size_u8 (value : UInt256) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← payload_size_u8 value
  -- if T::IS_DYNAMIC {
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<u8>` -/
def abi_single_root_size_u8 (value : UInt256) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_u8 value
  return v1

/-- `core::abi::abi_field_head_size<u8>` -/
def abi_field_head_size_u8 : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← u8_IS_DYNAMIC
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← u8_HEAD_SIZE
    return v4

/-- `core::abi::abi_single_root_head_size<u8>` -/
def abi_single_root_head_size_u8 : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_u8
  return v0

/-- `core::abi::encode_field<Sol, u8>` -/
def encode_field_Sol_u8 (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← u8_IS_DYNAMIC
  if v3 then
    -- let size = value.payload_size()
    let size ← payload_size_u8 value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← u8_encode_Sol value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← u8_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← u8_encode_Sol value r2
    return tails

/-- `core::abi::encode_single_root_alloc<Sol, u8>` -/
def encode_single_root_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_u8 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_u8
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_u8 value out (UInt256.ofNat 0) v13
  return buffer

/-- `core::abi::encode_return_alloc<Sol, u8>` -/
def encode_return_alloc_Sol_u8 (value : UInt256) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← const_u8
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v2 ← encode_alloc_Sol_u8 value
    return v2
  else
    -- encode_single_root_alloc<A, T>(value)
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
  -- if !T::IS_DYNAMIC {
  let v2 ← u8_IS_DYNAMIC
  if !v2 then
    -- if T::HEAD_SIZE <= 32 {
    let v4 ← u8_HEAD_SIZE
    if decide (v4.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_7 ← FixedMemBuffer_alloc_32
      out_r0 := out_7
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v10 ← FixedMemBuffer_ptr_32 out_r0
      let v11 ← u8_HEAD_SIZE
      let v12 ← AbiFrame_from_raw_parts v10 v11
      r4 := v12
      let _ ← u8_encode_Sol value r4
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r0
      let v17 ← u8_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← u8_HEAD_SIZE
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r1 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r1
        let v30 ← u8_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r5 := v31
        let _ ← u8_encode_Sol value r5
        -- self.return_data(
        let v35 ← FixedMemBuffer_ptr_64 out_r1
        let v36 ← u8_HEAD_SIZE
        let v37 ← downcast_unchecked_u256_usize v36
        let v38 ← MemSlice_from_raw_parts_ref_u8 v35 v37
        let v41 ← Evm_return_data v38
        nomatch v41
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v42 ← u8_HEAD_SIZE
        let v43 ← downcast_unchecked_u256_usize v42
        let out_44 ← MemBuffer_new v43
        out_r2 := out_44
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v47 ← MemBuffer_ptr out_r2
        let v48 ← u8_HEAD_SIZE
        let v49 ← AbiFrame_from_raw_parts v47 v48
        r6 := v49
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let _ ← u8_encode_Sol value r6
        -- self.return_data(data: out.span())
        let v53 ← MemBuffer_span_ref out_r2
        let v56 ← Evm_return_data v53
        nomatch v56
  else
    -- let out = encode_return_alloc<A, T>(value)
    let out_57 ← encode_return_alloc_Sol_u8 value
    out_r3 := out_57
    -- self.return_bytes(data: out.span())
    let v60 ← MemBuffer_span_ref out_r3
    let v63 ← Evm_return_bytes v60
    nomatch v63

structure CallOutcome where
  success : Bool
  returndata : MemBuffer
deriving Inhabited

/-- `Wei::zero` -/
def Wei_zero : FeM Wei := do
  return Wei.mk (UInt256.ofNat 0)

/-- `<() as AbiSize>::IS_DYNAMIC` -/
def tuple_IS_DYNAMIC : FeM Bool := do
  return false

/-- `<() as AbiSize>::HEAD_SIZE` -/
def tuple_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `core::abi::AbiSize::payload_size<()>` -/
def payload_size_tuple (self : Unit) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← tuple_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<()>` -/
def abi_payload_size_tuple (value : Unit) : FeM UInt256 := do
  -- value.payload_size()
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
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_tuple value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let value_9 := value
  let _ ← tuple_encode_Sol value_9 r2
  return buffer

/-- `std::evm::effects::encode_abi_payload<()>` -/
def encode_abi_payload_tuple (value : Unit) : FeM MemBuffer := do
  -- encode_alloc<Sol, T>(value)
  let v1 ← encode_alloc_Sol_tuple value
  return v1

/-- `std::evm::effects::Create::create2_raw<Evm>` -/
def create2_raw_Evm (value : Wei) (init_code : MemSlice_u8) (salt : UInt256) : FeM UInt256 := do
  -- Address {
  let v4 ← Wei_as_wei value
  let v5 ← MemSlice_ptr_u8 init_code
  let v6 ← MemSlice_len_u8 init_code
  let v8 ← create2 v4 v5 v6 salt
  return v8

/-- `std::evm::effects::copy_returndata` -/
def copy_returndata : FeM MemBuffer := do
  let mut buffer : MemBuffer := default
  -- let ret_len = ops::returndatasize()
  let ret_len ← returndatasize
  -- MemBuffer::new_uninit(ret_len.downcast_unchecked())
  let v2 ← downcast_unchecked_u256_usize ret_len
  let buffer_3 ← MemBuffer_new_uninit v2
  -- let buffer = unsafe { MemBuffer::new_uninit(ret_len.downcast_unchecked()) }
  buffer := buffer_3
  -- ops::returndatacopy(dest: buffer.ptr(), offset: 0, len: ret_len)
  let v6 ← MemBuffer_ptr buffer
  let _ ← returndatacopy v6 (UInt256.ofNat 0) ret_len
  return buffer

/-- `std::evm::effects::bubble_last_revert` -/
def bubble_last_revert : FeM Empty := do
  let mut ret : MemBuffer := default
  -- let ret = copy_returndata()
  let ret_0 ← copy_returndata
  ret := ret_0
  -- ops::revert(offset: ret.ptr(), len: ret.len() as u256)
  let v3 ← MemBuffer_ptr ret
  let v5 ← MemBuffer_len ret
  let v7 ← revertData v3 v5
  nomatch v7

/-- `std::evm::effects::copy_call_outcome` -/
def copy_call_outcome («raw» : RawCallOutcome) : FeM CallOutcome := do
  -- CallOutcome {
  let v1 ← RawCallOutcome_success «raw»
  let v2 ← copy_returndata
  return CallOutcome.mk v1 v2

/-- `<u256 as Decode>::decode_from<MemSlice<u8>>` -/
def u256_decode_from_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) : FeM UInt256 := do
  -- input.word_at(pos)
  let v2 ← MemSlice_u8_word_at input pos
  return v2

/-- `core::abi::Decode::decode_from_prechecked_head<u256, Sol, MemSlice<u8>>` -/
def decode_from_prechecked_head_u256_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- Self::decode_from(input, pos)
  let v3 ← u256_decode_from_Sol_MemSlice_u8 input pos
  return v3

/-- `core::abi::decode_frame_from<Sol, u256, MemSlice<u8>>` -/
def decode_frame_from_Sol_u256_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- checked_frame_end<A>(pos, T::HEAD_SIZE, input_len)
  let v3 ← u256_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- T::decode_from_prechecked_head(input, pos, input_len)
  let v5 ← decode_from_prechecked_head_u256_Sol_MemSlice_u8 input pos input_len
  return v5

/-- `core::abi::Decode::decode_from_bounded<u256, Sol, MemSlice<u8>>` -/
def decode_from_bounded_u256_Sol_MemSlice_u8 (input : MemSlice_u8) (pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- checked_frame_end<A>(pos, Self::HEAD_SIZE, input_len)
  let v3 ← u256_HEAD_SIZE
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- Self::decode_from(input, pos)
  let v5 ← u256_decode_from_Sol_MemSlice_u8 input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, u256, MemSlice<u8>>` -/
def decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 (input : MemSlice_u8) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v4 ← u256_IS_DYNAMIC
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← MemSlice_u8_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← decode_from_bounded_u256_Sol_MemSlice_u8 input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← u256_decode_from_Sol_MemSlice_u8 input head_pos
    return v12

/-- `std::abi::sol::decode_output_at<u256, MemSlice<u8>>` -/
def decode_output_at_u256_MemSlice_u8 (input : MemSlice_u8) (base : UInt256) : FeM UInt256 := do
  -- let input_len = input.len()
  let input_len ← len_MemSlice_u8 input
  -- if T::IS_PARAM_LIST {
  let v4 ← const_u256
  if v4 then
    -- return decode_frame_from<Sol, T, I>(input, base, input_len)
    let v5 ← decode_frame_from_Sol_u256_MemSlice_u8 input base input_len
    return v5
  else
    -- checked_frame_end<Sol>(base, abi_single_root_head_size<T>(), input_len)
    let v8 ← abi_single_root_head_size_u256
    let _ ← checked_frame_end_Sol base v8 input_len
    -- decode_field_from_prechecked_head<Sol, T, I>(input, base: base, head_pos: base, input_len: input_len)
    let v10 ← decode_field_from_prechecked_head_Sol_u256_MemSlice_u8 input base base input_len
    return v10

/-- `std::abi::sol::decode_output<u256, MemSlice<u8>>` -/
def decode_output_u256_MemSlice_u8 (input : MemSlice_u8) : FeM UInt256 := do
  -- decode_output_at(input, 0)
  let v2 ← decode_output_at_u256_MemSlice_u8 input (UInt256.ofNat 0)
  return v2

/-- `std::evm::effects::decode_returndata<u256>` -/
def decode_returndata_u256 (ret : MemSlice_u8) : FeM UInt256 := do
  -- decode_output(ret)
  let v1 ← decode_output_u256_MemSlice_u8 ret
  return v1

/-- `CallOutcome::decode<u256>` -/
def CallOutcome_decode_u256 (self : CallOutcome) : FeM UInt256 := do
  let mut returndata : MemBuffer := default
  -- let returndata = self.returndata
  let returndata_3 := self.returndata
  returndata := returndata_3
  -- if !success {
  if !self.success then
    -- ops::revert(offset: returndata.ptr(), len: returndata.len() as u256)
    let v7 ← MemBuffer_ptr returndata
    let v9 ← MemBuffer_len returndata
    let v11 ← revertData v7 v9
    nomatch v11
  else
    -- decode_returndata(returndata.span())
    let v15 ← MemBuffer_span_ref returndata
    let v17 ← decode_returndata_u256 v15
    let _ ← MemBuffer_drop returndata
    return v17

/-- `std::evm::crypto::keccak_words<2>` -/
def keccak_words_2 (words : (Vector UInt256 2)) : FeM UInt256 := do
  return keccakWords words

/-- `std::evm::effects::read_transient_word<Evm>` -/
def read_transient_word_Evm (slot : UInt256) : FeM UInt256 := do
  -- *(slot as TStorPtr<u256>)
  let v3 ← load (Place.mk .transient slot)
  return v3

/-- `std::evm::effects::storage_at<Evm, u256>`, its `mut` body -/
def storage_at_mut_Evm_u256 (slot : UInt256) : FeM Place := do
  -- mut? *(slot as StorPtr<T>)
  let v2 := (Place.mk .storage slot)
  return v2

/-- `<Address as Encode>::encode` -/
def Address_encode_Sol (self : UInt256) (out : AbiFrame) : FeM Unit := do
  -- out.store_word(at: 0, value: self.inner)
  let _ ← AbiFrame_store_word out (UInt256.ofNat 0) self
  return ()

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<Address>` -/
def const_Address : FeM Bool := do
  return false

/-- `core::abi::AbiSize::payload_size<Address>` -/
def payload_size_Address (self : UInt256) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← Address_HEAD_SIZE
  return v1

/-- `core::abi::abi_payload_size<Address>` -/
def abi_payload_size_Address (value : UInt256) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← payload_size_Address value
  return v1

/-- `core::abi::encode_alloc<Sol, Address>` -/
def encode_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_Address value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let value_9 := value
  let _ ← Address_encode_Sol value_9 r2
  return buffer

/-- `core::abi::abi_field_size<Address>` -/
def abi_field_size_Address (value : UInt256) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← payload_size_Address value
  -- if T::IS_DYNAMIC {
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<Address>` -/
def abi_single_root_size_Address (value : UInt256) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_Address value
  return v1

/-- `core::abi::abi_single_root_head_size<Address>` -/
def abi_single_root_head_size_Address : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_Address
  return v0

/-- `core::abi::encode_field<Sol, Address>` -/
def encode_field_Sol_Address (value : UInt256) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← Address_IS_DYNAMIC
  if v3 then
    -- let size = value.payload_size()
    let size ← payload_size_Address value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← Address_encode_Sol value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← Address_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← Address_encode_Sol value r2
    return tails

/-- `core::abi::encode_single_root_alloc<Sol, Address>` -/
def encode_single_root_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_Address value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_Address
  let v13 ← AbiFrame_tail out v11
  let value_14 := value
  let _ ← encode_field_Sol_Address value_14 out (UInt256.ofNat 0) v13
  return buffer

/-- `core::abi::encode_return_alloc<Sol, Address>` -/
def encode_return_alloc_Sol_Address (value : UInt256) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← const_Address
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v2 ← encode_alloc_Sol_Address value
    return v2
  else
    -- encode_single_root_alloc<A, T>(value)
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
  -- if !T::IS_DYNAMIC {
  let v1 ← Address_IS_DYNAMIC
  if !v1 then
    -- if T::HEAD_SIZE <= 32 {
    let v3 ← Address_HEAD_SIZE
    if decide (v3.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← Address_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let value_12 := value
      let _ ← Address_encode_Sol value_12 r5
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← Address_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← Address_HEAD_SIZE
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← Address_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let value_32 := value
        let _ ← Address_encode_Sol value_32 r6
        -- self.return_data(
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← Address_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v43 ← Address_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← Address_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let value_51 := value
        let _ ← Address_encode_Sol value_51 r7
        -- self.return_data(data: out.span())
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    -- let out = encode_return_alloc<A, T>(value)
    let out_60 ← encode_return_alloc_Sol_Address value
    out_r4 := out_60
    -- self.return_bytes(data: out.span())
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
  -- if !condition {
  if !condition then
    -- panic()
    let v3 ← trap
    nomatch v3
  else
    return ()

/-- `<u128 as IntWord>::to_word` -/
def u128_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<u128 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u128_u64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u128_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u64_MASK
    let v12 ← u64_from_word (word &&& v10)
    join v12

/-- `<u64 as WrappingMul>::wrapping_mul` -/
def u64_wrapping_mul (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return IntTy.mulW (.u 64) self other

/-- `<u128 as WrappingSub>::wrapping_sub` -/
def u128_wrapping_sub (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return IntTy.subW (.u 128) self other

/-- `<i64 as IntWord>::IS_SIGNED` -/
def i64_IS_SIGNED : FeM Bool := do
  return true

/-- `<i64 as IntWord>::MASK` -/
def i64_MASK : FeM UInt256 := do
  let v0 ← U64_MASK
  return v0

/-- `core::num::I64_SIGN_BIT` -/
def I64_SIGN_BIT : FeM UInt256 := do
  return UInt256.ofNat 9223372036854775808

/-- `<i64 as IntWord>::SIGN_BIT` -/
def i64_SIGN_BIT : FeM UInt256 := do
  let v0 ← I64_SIGN_BIT
  return v0

/-- `<i64 as IntWord>::from_word` -/
def i64_from_word (word : UInt256) : FeM UInt256 := do
  return IntTy.cast (.u 256) (.i 64) word

/-- `<u64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u64_i64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u64_to_word self
  -- if T::IS_SIGNED {
  let v4 ← i64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← i64_MASK
    let v12 ← i64_from_word (word &&& v10)
    join v12

/-- `<i64 as IntWord>::to_word` -/
def i64_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<i64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i64_u64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← i64_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u64_MASK
    let v12 ← u64_from_word (word &&& v10)
    join v12

/-- `<i128 as IntWord>::to_word` -/
def i128_to_word (self : UInt256) : FeM UInt256 := do
  return self

/-- `<i128 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i128_u64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← i128_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u64_MASK
    let v6 ← u64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u64_MASK
    let v12 ← u64_from_word (word &&& v10)
    join v12

/-- `<i128 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_i128_i64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← i128_to_word self
  -- if T::IS_SIGNED {
  let v4 ← i64_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← i64_MASK
    let v6 ← i64_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← i64_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← i64_MASK
    let v12 ← i64_from_word (word &&& v10)
    join v12

/-- `<[usize; 3] as Collection>::start` -/
def array_start_usize_3 (self : (Vector UInt256 3)) : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<[usize; 3] as Collection>::has` -/
def array_has_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM Bool := do
  return decide (c.toNat < (UInt256.ofNat 3).toNat)

/-- `<[usize; 3] as Collection>::at`, its `ref` body -/
def array_at_ref_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM UInt256 := do
  -- mut? self[c]
  checkIndex c 3
  return self[c.toNat]!

/-- `<Range<Known<1>, Known<4>> as Collection>::start` -/
def Range_start_1_4 (self : Range_Known_1_Known_4) : FeM UInt256 := do
  return UInt256.ofNat 1

/-- `core::range::range_has` -/
def range_has (c : UInt256) (lo : UInt256) (hi : UInt256) : FeM Bool := do
  let join : Bool → FeM Bool := fun v3 => do
    return v3
  let join_1 : FeM Bool := do
    join false
  -- lo <= c && c < hi
  if decide (lo.toNat ≤ c.toNat) then
    -- lo <= c && c < hi
    if decide (c.toNat < hi.toNat) then
      join true
    else
      join_1
  else
    join_1

/-- `<Range<Known<1>, Known<4>> as Collection>::has` -/
def Range_has_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 1) (UInt256.ofNat 4)
  return v4

/-- `<Range<Known<1>, Known<4>> as Collection>::at` -/
def Range_at_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<1>, Known<4>> as Collection>::next` -/
def Range_next_1_4 (self : Range_Known_1_Known_4) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<[usize; 3] as Collection>::next` -/
def array_next_usize_3 (self : (Vector UInt256 3)) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<Range<Known<1>, Known<16>> as Collection>::start` -/
def Range_start_1_16 (self : Range_Known_1_Known_16) : FeM UInt256 := do
  return UInt256.ofNat 1

/-- `<Range<Known<1>, Known<16>> as Collection>::has` -/
def Range_has_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 1) (UInt256.ofNat 16)
  return v4

/-- `<Range<Known<1>, Known<16>> as Collection>::at` -/
def Range_at_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<1>, Known<16>> as Collection>::next` -/
def Range_next_1_16 (self : Range_Known_1_Known_16) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<u256 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u256_usize (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u256_to_word self
  -- if T::IS_SIGNED {
  let v4 ← usize_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← usize_MASK
    let v12 ← usize_from_word (word &&& v10)
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
  -- var tails = out.tail(Self::HEAD_SIZE)
  let v5 ← tuple_HEAD_SIZE_bool_u256_u256
  let tails_6 ← AbiFrame_tail out v5
  tails := tails_6
  -- let head1 = abi_field_head_size<T0>()
  let head1 ← abi_field_head_size_bool
  -- let head2 = head1 + abi_field_head_size<T1>()
  let v10 ← abi_field_head_size_u256
  -- tails = encode_field<A, T0>(v0, out, head_pos: 0, tails)
  let tails_17 ← encode_field_Sol_bool self.fst out (UInt256.ofNat 0) tails
  tails := tails_17
  -- tails = encode_field<A, T1>(v1, out, head_pos: head1, tails)
  let tails_23 ← encode_field_Sol_u256 self.snd.fst out head1 tails
  tails := tails_23
  -- encode_field<A, T2>(v2, out, head_pos: head2, tails)
  let _ ← encode_field_Sol_u256 self.snd.snd out (head1 + v10) tails
  return ()

/-- `<(bool, u256, u256) as AbiSize>::IS_PARAM_LIST` -/
def tuple_IS_PARAM_LIST_bool_u256_u256 : FeM Bool := do
  return true

/-- `core::abi::dynamic_payload_size<bool>` -/
def dynamic_payload_size_bool (value : Bool) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v1 ← bool_IS_DYNAMIC
  if v1 then
    -- return value.payload_size()
    let v2 ← payload_size_bool value
    return v2
  else
    return UInt256.ofNat 0

/-- `<(bool, u256, u256) as AbiSize>::payload_size` -/
def tuple_payload_size_bool_u256_u256 (self : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← tuple_HEAD_SIZE_bool_u256_u256
  let v3 ← dynamic_payload_size_bool self.fst
  let v6 ← dynamic_payload_size_u256 self.snd.fst
  let v9 ← dynamic_payload_size_u256 self.snd.snd
  return ((v1 + v3) + v6) + v9

/-- `core::abi::abi_payload_size<(bool, u256, u256)>` -/
def abi_payload_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← tuple_payload_size_bool_u256_u256 value
  return v1

/-- `core::abi::encode_alloc<Sol, (bool, u256, u256)>` -/
def encode_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_tuple_bool_u256_u256 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← tuple_encode_Sol_bool_u256_u256 value r2
  return buffer

/-- `core::abi::abi_field_size<(bool, u256, u256)>` -/
def abi_field_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← tuple_payload_size_bool_u256_u256 value
  -- if T::IS_DYNAMIC {
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<(bool, u256, u256)>` -/
def abi_single_root_size_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_tuple_bool_u256_u256 value
  return v1

/-- `core::abi::abi_field_head_size<(bool, u256, u256)>` -/
def abi_field_head_size_tuple_bool_u256_u256 : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← tuple_HEAD_SIZE_bool_u256_u256
    return v4

/-- `core::abi::abi_single_root_head_size<(bool, u256, u256)>` -/
def abi_single_root_head_size_tuple_bool_u256_u256 : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_tuple_bool_u256_u256
  return v0

/-- `core::abi::encode_field<Sol, (bool, u256, u256)>` -/
def encode_field_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← tuple_IS_DYNAMIC_bool_u256_u256
  if v3 then
    -- let size = value.payload_size()
    let size ← tuple_payload_size_bool_u256_u256 value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← tuple_encode_Sol_bool_u256_u256 value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← tuple_HEAD_SIZE_bool_u256_u256
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← tuple_encode_Sol_bool_u256_u256 value r2
    return tails

/-- `core::abi::encode_single_root_alloc<Sol, (bool, u256, u256)>` -/
def encode_single_root_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_tuple_bool_u256_u256 value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_tuple_bool_u256_u256
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_tuple_bool_u256_u256 value out (UInt256.ofNat 0) v13
  return buffer

/-- `core::abi::encode_return_alloc<Sol, (bool, u256, u256)>` -/
def encode_return_alloc_Sol_tuple_bool_u256_u256 (value : (Bool × UInt256 × UInt256)) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← tuple_IS_PARAM_LIST_bool_u256_u256
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v2 ← encode_alloc_Sol_tuple_bool_u256_u256 value
    return v2
  else
    -- encode_single_root_alloc<A, T>(value)
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
  -- if !T::IS_DYNAMIC {
  let v1 ← tuple_IS_DYNAMIC_bool_u256_u256
  if !v1 then
    -- if T::HEAD_SIZE <= 32 {
    let v3 ← tuple_HEAD_SIZE_bool_u256_u256
    if decide (v3.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← tuple_HEAD_SIZE_bool_u256_u256
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let _ ← tuple_encode_Sol_bool_u256_u256 value r5
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← tuple_HEAD_SIZE_bool_u256_u256
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← tuple_HEAD_SIZE_bool_u256_u256
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← tuple_HEAD_SIZE_bool_u256_u256
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let _ ← tuple_encode_Sol_bool_u256_u256 value r6
        -- self.return_data(
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← tuple_HEAD_SIZE_bool_u256_u256
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v43 ← tuple_HEAD_SIZE_bool_u256_u256
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← tuple_HEAD_SIZE_bool_u256_u256
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let _ ← tuple_encode_Sol_bool_u256_u256 value r7
        -- self.return_data(data: out.span())
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    -- let out = encode_return_alloc<A, T>(value)
    let out_60 ← encode_return_alloc_Sol_tuple_bool_u256_u256 value
    out_r4 := out_60
    -- self.return_bytes(data: out.span())
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
  -- ops::calldatasize()
  let v1 ← calldatasize
  return v1

/-- `std::evm::effects::prepare_copy_destination` -/
def prepare_copy_destination (dest_0 : MemBuffer) (dest_offset : UInt256) (len : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut dest_0 := dest_0
  -- if dest_offset > dest.capacity() as u256 || len > dest.capacity() as u256 - dest_offset {
  let v5 ← MemBuffer_capacity dest_0
  let join : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
    let mut dest_0 := dest_0
    -- core::panic()
    let v44 ← trap
    nomatch v44
  -- if dest_offset > dest.capacity() as u256 || len > dest.capacity() as u256 - dest_offset {
  if decide (v5.toNat < dest_offset.toNat) then
    join dest_0
  else
    -- if dest_offset > dest.capacity() as u256 || len > dest.capacity() as u256 - dest_offset {
    let v10 ← MemBuffer_capacity dest_0
    let v13 ← checkedSub v10 dest_offset
    if decide (v13.toNat < len.toNat) then
      join dest_0
    else
      let join_1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
        let mut dest_0 := dest_0
        -- ptr::offset_bytes(dest.ptr(), dest_offset.downcast_unchecked())
        let v41 ← MemBuffer_ptr dest_0
        let v42 ← downcast_unchecked_u256_usize dest_offset
        return ((v41 + v42), dest_0)
      let join_2 : MemBuffer → FeM (UInt256 × MemBuffer) := fun dest_0 => do
        let mut dest_0 := dest_0
        join_1 dest_0
      -- if len != 0 && dest_offset > dest.len() as u256 {
      if decide (len ≠ (UInt256.ofNat 0)) then
        -- if len != 0 && dest_offset > dest.len() as u256 {
        let v22 ← MemBuffer_len dest_0
        if decide (v22.toNat < dest_offset.toNat) then
          -- ptr::zero_bytes(
          let v26 ← MemBuffer_ptr dest_0
          let v28 ← MemBuffer_len dest_0
          let v32 ← MemBuffer_len dest_0
          let v34 ← checkedSub dest_offset v32
          let v35 ← downcast_unchecked_u256_usize v34
          let _ ← zero_bytes (v26 + v28) v35
          join_1 dest_0
        else
          join_2 dest_0
      else
        join_2 dest_0

/-- `<Evm as RawOps>::calldatacopy` -/
def Evm_calldatacopy (dest_1 : MemBuffer) (dest_offset : UInt256) (src_offset : UInt256) (len : UInt256) : FeM (Unit × MemBuffer) := do
  let mut dest_1 := dest_1
  -- let dest_ptr = prepare_copy_destination(
  let (dest_ptr, dest_ptr_0) ← prepare_copy_destination dest_1 dest_offset len
  dest_1 := dest_ptr_0
  -- if len == 0 {
  if decide (len = (UInt256.ofNat 0)) then
    return ((), dest_1)
  else
    -- ops::calldatacopy(
    let _ ← calldatacopy dest_ptr src_offset len
    -- let end: usize = (dest_offset + len).downcast_unchecked()
    let v16 ← checkedAdd dest_offset len
    let «end» ← downcast_unchecked_u256_usize v16
    -- if end > dest.len() {
    let v20 ← MemBuffer_len dest_1
    let join : MemBuffer → FeM (Unit × MemBuffer) := fun dest_1 => do
      let mut dest_1 := dest_1
      return ((), dest_1)
    -- if end > dest.len() {
    if decide (v20.toNat < «end».toNat) then
      -- dest.set_len(end)
      let (v22, v22_0) ← MemBuffer_set_len dest_1 «end»
      dest_1 := v22_0
      join dest_1
    else
      join dest_1

/-- `MemSlice<u8>::value_ptr<u64>` -/
def MemSlice_u8_value_ptr_u64 (self : MemSlice_u8) (offset : UInt256) : FeM UInt256 := do
  let join : FeM UInt256 := do
    -- index_out_of_bounds()
    let v14 ← index_out_of_bounds
    nomatch v14
  -- if offset > self.len || size_of<T>() > self.len - offset {
  if decide (self.len.toNat < offset.toNat) then
    join
  else
    -- if offset > self.len || size_of<T>() > self.len - offset {
    let v8 ← checkedSub self.len offset
    if decide (v8.toNat < (UInt256.ofNat 8).toNat) then
      join
    else
      return self.ptr + offset

/-- `core::ptr::cast<u8, u64>` -/
def cast_u8_u64 (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `MemSlice<u8>::read_le<u64>` -/
def MemSlice_u8_read_le_u64 (self : MemSlice_u8) (offset : UInt256) : FeM UInt256 := do
  -- *cast<u8, T>(self.value_ptr<T>(offset))
  let v2 ← MemSlice_u8_value_ptr_u64 self offset
  let v3 ← cast_u8_u64 v2
  let v4 ← mloadInt (.u 64) v3
  return IntTy.swapBytes (.u 64) v4

/-- `MemBuffer::read_le<u64>` -/
def MemBuffer_read_le_u64 (self : MemBuffer) (offset : UInt256) : FeM UInt256 := do
  -- self.span().read_le<T>(offset)
  let v2 ← MemBuffer_span_ref self
  let v4 ← MemSlice_u8_read_le_u64 v2 offset
  return v4

/-- `<Range<Unknown, Unknown> as Collection>::start` -/
def Range_Unknown_Unknown_start (self : Range_Unknown_Unknown) : FeM UInt256 := do
  return self.start

/-- `<Range<Unknown, Unknown> as Collection>::has` -/
def Range_Unknown_Unknown_has (self : Range_Unknown_Unknown) (c : UInt256) : FeM Bool := do
  -- range_has(c, self.start, self.end)
  let v4 ← range_has c self.start self.«end»
  return v4

/-- `<Range<Unknown, Unknown> as Collection>::at` -/
def Range_Unknown_Unknown_at (self : Range_Unknown_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Unknown, Unknown> as Collection>::next` -/
def Range_Unknown_Unknown_next (self : Range_Unknown_Unknown) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<MemBuffer as Index>::index`, its `ref` body -/
def MemBuffer_index_ref (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  -- if index >= self.len {
  if decide (self.len.toNat ≤ index.toNat) then
    -- index_out_of_bounds()
    let v4 ← index_out_of_bounds
    nomatch v4
  else
    -- mut? *offset_bytes(self.ptr, index)
    let v9 := (self.ptr + index)
    return v9

/-- `MemBuffer::extend_over<u256>` -/
def MemBuffer_extend_over_u256 (self : MemBuffer) (offset : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut self := self
  -- let size = size_of<T>()
  let size := (UInt256.ofNat 32)
  let join : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
    let mut self := self
    -- index_out_of_bounds()
    let v38 ← index_out_of_bounds
    nomatch v38
  -- if offset > self.capacity || size > self.capacity - offset {
  if decide (self.capacity.toNat < offset.toNat) then
    join self
  else
    -- if offset > self.capacity || size > self.capacity - offset {
    let v10 := decide ((self.capacity - offset).toNat < size.toNat)
    if v10 then
      join self
    else
      let join_1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
        let mut self := self
        let join_2 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
          let mut self := self
          return ((self.ptr + offset), self)
        -- if offset + size > self.len {
        if decide (self.len.toNat < (offset + size).toNat) then
          -- self.len = offset + size
          self := { self with len := (offset + size) }
          join_2 self
        else
          join_2 self
      -- if offset > self.len {
      if decide (self.len.toNat < offset.toNat) then
        -- zero_bytes(ptr: offset_bytes(self.ptr, self.len), len: (offset - self.len))
        let _ ← zero_bytes (self.ptr + self.len) (offset - self.len)
        join_1 self
      else
        join_1 self

/-- `MemBuffer::write_be<u256>` -/
def MemBuffer_write_be_u256 (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  -- *cast<u8, T>(self.extend_over<T>(offset)) = to_be(value)
  let (v4, v4_0) ← MemBuffer_extend_over_u256 self offset
  self := v4_0
  let v5 ← cast_u8_u256 v4
  mstore v5 value
  return ((), self)

/-- `MemBuffer::write_word` -/
def MemBuffer_write_word (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  -- self.write_be<u256>(offset, value)
  let (v3, v3_0) ← MemBuffer_write_be_u256 self offset value
  self := v3_0
  return ((), self)

/-- `<Evm as RawOps>::revert` -/
def Evm_revert (data : MemSlice_u8) : FeM Empty := do
  -- ops::revert(offset: data.ptr(), len: data.len() as u256)
  let v2 ← MemSlice_ptr_u8 data
  let v3 ← MemSlice_len_u8 data
  let v5 ← revertData v2 v3
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
  return UInt256.ofNat 0

/-- `<Range<Known<0>, Known<8>> as Collection>::has` -/
def Range_has_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 8)
  return v4

/-- `<Range<Known<0>, Known<8>> as Collection>::at` -/
def Range_at_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<8>> as Collection>::next` -/
def Range_next_0_8 (self : Range_Known_0_Known_8) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<Range<Known<0>, Known<16>> as Collection>::start` -/
def Range_start_0_16 (self : Range_Known_0_Known_16) : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Range<Known<0>, Known<16>> as Collection>::has` -/
def Range_has_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 16)
  return v4

/-- `<Range<Known<0>, Known<16>> as Collection>::at` -/
def Range_at_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<16>> as Collection>::next` -/
def Range_next_0_16 (self : Range_Known_0_Known_16) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<u64 as WrappingAdd>::wrapping_add` -/
def u64_wrapping_add (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return IntTy.addW (.u 64) self other

/-- `<u64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u64_usize (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u64_to_word self
  -- if T::IS_SIGNED {
  let v4 ← usize_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← usize_MASK
    let v12 ← usize_from_word (word &&& v10)
    join v12

/-- `MemBuffer::extend_over<u64>` -/
def MemBuffer_extend_over_u64 (self : MemBuffer) (offset : UInt256) : FeM (UInt256 × MemBuffer) := do
  let mut self := self
  -- let size = size_of<T>()
  let size := (UInt256.ofNat 8)
  let join : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
    let mut self := self
    -- index_out_of_bounds()
    let v38 ← index_out_of_bounds
    nomatch v38
  -- if offset > self.capacity || size > self.capacity - offset {
  if decide (self.capacity.toNat < offset.toNat) then
    join self
  else
    -- if offset > self.capacity || size > self.capacity - offset {
    let v10 := decide ((self.capacity - offset).toNat < size.toNat)
    if v10 then
      join self
    else
      let join_1 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
        let mut self := self
        let join_2 : MemBuffer → FeM (UInt256 × MemBuffer) := fun self => do
          let mut self := self
          return ((self.ptr + offset), self)
        -- if offset + size > self.len {
        if decide (self.len.toNat < (offset + size).toNat) then
          -- self.len = offset + size
          self := { self with len := (offset + size) }
          join_2 self
        else
          join_2 self
      -- if offset > self.len {
      if decide (self.len.toNat < offset.toNat) then
        -- zero_bytes(ptr: offset_bytes(self.ptr, self.len), len: (offset - self.len))
        let _ ← zero_bytes (self.ptr + self.len) (offset - self.len)
        join_1 self
      else
        join_1 self

/-- `MemBuffer::write_le<u64>` -/
def MemBuffer_write_le_u64 (self : MemBuffer) (offset : UInt256) (value : UInt256) : FeM (Unit × MemBuffer) := do
  let mut self := self
  -- *cast<u8, T>(self.extend_over<T>(offset)) = to_le(value)
  let (v4, v4_0) ← MemBuffer_extend_over_u64 self offset
  self := v4_0
  let v5 ← cast_u8_u64 v4
  mstoreInt (.u 64) v5 (IntTy.swapBytes (.u 64) value)
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
  return UInt256.ofNat 0

/-- `<Range<Known<0>, Unknown> as Collection>::has` -/
def Range_has_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, self.end)
  let v4 ← range_has c (UInt256.ofNat 0) self.«end»
  return v4

/-- `<Range<Known<0>, Unknown> as Collection>::at` -/
def Range_at_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<64>> as Collection>::start` -/
def Range_start_0_64 (self : Range_Known_0_Known_64) : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Range<Known<0>, Known<64>> as Collection>::has` -/
def Range_has_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 64)
  return v4

/-- `<Range<Known<0>, Known<64>> as Collection>::at` -/
def Range_at_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<64>> as Collection>::next` -/
def Range_next_0_64 (self : Range_Known_0_Known_64) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<usize as IntDowncast>::downcast_unchecked` -/
def downcast_unchecked_usize_u64 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← usize_to_word self
  -- T::from_word(word)
  let v3 ← u64_from_word word
  return v3

/-- `<u64 as IntDowncast>::downcast_truncate` -/
def downcast_truncate_u64_u8 (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← u64_to_word self
  -- if T::IS_SIGNED {
  let v4 ← u8_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← u8_MASK
    let v6 ← u8_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← u8_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← u8_MASK
    let v12 ← u8_from_word (word &&& v10)
    join v12

/-- `<Range<Known<0>, Unknown> as Collection>::next` -/
def Range_next_0 (self : Range_Known_0_Unknown) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<[u32; 5] as Collection>::start` -/
def array_start_u32_5 (self : (Vector UInt256 5)) : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<[u32; 5] as Collection>::has` -/
def array_has_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM Bool := do
  return decide (c.toNat < (UInt256.ofNat 5).toNat)

/-- `<[u32; 5] as Collection>::at`, its `ref` body -/
def array_at_ref_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM UInt256 := do
  -- mut? self[c]
  checkIndex c 5
  return self[c.toNat]!

/-- `<[u32; 5] as Collection>::next` -/
def array_next_u32_5 (self : (Vector UInt256 5)) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<u256 as WrappingSub>::wrapping_sub` -/
def u256_wrapping_sub (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return self - other

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
  -- if len > MAX_ALLOCATION {
  let v3 ← MAX_ALLOCATION
  if decide (v3.toNat < len.toNat) then
    return Option_u8.None
  else
    -- let ptr = try_alloc_raw(len)
    let ptr ← alloc len
    let join : Option_u8 → FeM Option_u8 := fun v1 => do
      return v1
    -- if ptr as usize == 0 { Option::None } else { Option::Some(ptr) }
    if decide (ptr = (UInt256.ofNat 0)) then
      join Option_u8.None
    else
      join (Option_u8.Some ptr)

/-- `MemBuffer::try_new` -/
def MemBuffer_try_new (len : UInt256) : FeM Option_MemBuffer := do
  -- let ptr = try_alloc_bytes(len)?
  let v1 ← try_alloc_bytes len
  if match v1 with | .Some .. => true | _ => false then
    -- let ptr = try_alloc_bytes(len)?
    let ptr := v1.Some_0
    -- zero_bytes(ptr, len)
    let _ ← zero_bytes ptr len
    -- Self { ptr, len, capacity: len }
    let v8 := (MemBuffer.mk ptr len len)
    return Option_MemBuffer.Some v8
  else
    return Option_MemBuffer.None

/-- `<MemBuffer as Index>::index`, its `mut` body -/
def MemBuffer_index_mut (self : MemBuffer) (index : UInt256) : FeM (UInt256 × (UInt256 → FeM MemBuffer)) := do
  let mut self := self
  -- if index >= self.len {
  if decide (self.len.toNat ≤ index.toNat) then
    -- index_out_of_bounds()
    let v4 ← index_out_of_bounds
    nomatch v4
  else
    -- mut? *offset_bytes(self.ptr, index)
    let v9 := (self.ptr + index)
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
  -- let size = size_of<T>()
  let size := (UInt256.ofNat 8)
  let join : FeM Option_u64 := do
    -- Option::Some(cast<u8, T>(try_alloc_bytes(len * size)?))
    let v13 ← checkedMul len size
    let v14 ← try_alloc_bytes v13
    if match v14 with | .Some .. => true | _ => false then
      -- Option::Some(cast<u8, T>(try_alloc_bytes(len * size)?))
      let v17 ← cast_u8_u64 v14.Some_0
      return Option_u64.Some v17
    else
      return Option_u64.None
  -- if size != 0 && len > MAX_ALLOCATION / size {
  if decide (size ≠ (UInt256.ofNat 0)) then
    -- if size != 0 && len > MAX_ALLOCATION / size {
    let v6 ← MAX_ALLOCATION
    let v7 ← checkedDiv v6 size
    if decide (v7.toNat < len.toNat) then
      return Option_u64.None
    else
      join
  else
    join

/-- `MemArray<u64>::try_new_uninit` -/
def MemArray_try_new_uninit_u64 (len : UInt256) : FeM Option_MemArray_u64 := do
  -- Option::Some(Self { ptr: try_alloc_elems(len)?, len })
  let v1 ← try_alloc_elems_u64 len
  if match v1 with | .Some .. => true | _ => false then
    -- Option::Some(Self { ptr: try_alloc_elems(len)?, len })
    let v5 := (MemArray_u64.mk v1.Some_0 len)
    return Option_MemArray_u64.Some v5
  else
    return Option_MemArray_u64.None

/-- `core::ptr::cast<u64, u8>` -/
def cast_u64_u8 (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `MemArray<u64>::byte_len` -/
def MemArray_byte_len_u64 (self : MemArray_u64) : FeM UInt256 := do
  -- self.len * size_of<T>()
  let v3 ← checkedMul self.len (UInt256.ofNat 8)
  return v3

/-- `<usize as WrappingMul>::wrapping_mul` -/
def usize_wrapping_mul (self : UInt256) (other : UInt256) : FeM UInt256 := do
  return self * other

/-- `core::ptr::elem<u64>` -/
def elem_u64 (ptr : UInt256) (index : UInt256) : FeM UInt256 := do
  -- offset_bytes(ptr, index.wrapping_mul(size_of<T>()))
  let v4 ← usize_wrapping_mul index (UInt256.ofNat 8)
  return ptr + v4

/-- `core::mem::drop<u64>` -/
def drop_u64 (value : UInt256) : FeM Unit := do
  return ()

/-- `core::ptr::drop_in_place<u64>` -/
def drop_in_place_u64 (ptr : UInt256) : FeM Unit := do
  -- drop(*ptr)
  let v2 ← mloadInt (.u 64) ptr
  let _ ← drop_u64 v2
  return ()

/-- `core::ptr::drop_range<u64>` -/
def drop_range_u64 (ptr : UInt256) (start : UInt256) («end» : UInt256) : FeM Unit := do
  let mut r0 : Range_Unknown_Unknown := default
  let join : Range_Unknown_Unknown → FeM Unit := fun r0 => do
    let mut r0 := r0
    return ()
  -- if needs_drop<T>() {
  if false then
    -- for i in start..end {
    r0 := (Range_Unknown_Unknown.mk start «end»)
    let v10 ← Range_Unknown_Unknown_start r0
    let rec loop (r0 : Range_Unknown_Unknown) (v3 : UInt256) : FeM Unit := do
      let mut r0 := r0
      -- for i in start..end {
      let v13 ← Range_Unknown_Unknown_has r0 v3
      if v13 then
        -- for i in start..end {
        let v15 ← Range_Unknown_Unknown_at r0 v3
        -- drop_in_place(elem(ptr, i))
        let v17 ← elem_u64 ptr v15
        let _ ← drop_in_place_u64 v17
        -- for i in start..end {
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
  -- drop_range(ptr, start: 0, end: len)
  let _ ← drop_range_u64 ptr (UInt256.ofNat 0) len
  return ()

/-- `core::ptr::dealloc<u64>` -/
def dealloc_u64 (ptr : UInt256) : FeM Unit := do
  -- dealloc_raw(cast<T, u8>(ptr))
  let _ ← cast_u64_u8 ptr
  return ()

/-- `<MemArray<u64> as Drop>::drop` -/
def MemArray_drop_u64 (self : MemArray_u64) : FeM Unit := do
  -- drop_in_place_n(self.ptr, self.len)
  let _ ← drop_in_place_n_u64 self.ptr self.len
  -- dealloc(self.ptr)
  let _ ← dealloc_u64 self.ptr
  return ()

/-- `MemArray<u64>::try_new_zeroed` -/
def MemArray_try_new_zeroed_u64 (len : UInt256) : FeM Option_MemArray_u64 := do
  let mut array : MemArray_u64 := default
  -- Self::try_new_uninit(len)?
  let v1 ← MemArray_try_new_uninit_u64 len
  if match v1 with | .Some .. => true | _ => false then
    -- Self::try_new_uninit(len)?
    let array_3 := v1.Some_0
    -- let array = unsafe { Self::try_new_uninit(len)? }
    array := array_3
    -- zero_bytes(cast<T, u8>(array.ptr), array.byte_len())
    let v6 ← cast_u64_u8 array.ptr
    let v8 ← MemArray_byte_len_u64 array
    let _ ← zero_bytes v6 v8
    -- Option::Some(array)
    let array_10 := array
    return Option_MemArray_u64.Some array_10
  else
    match v1 with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v1.Some_0
      pure ()
    | _ => pure ()
    return Option_MemArray_u64.None

/-- `core::ptr::mem_array_elem<u64>` -/
def mem_array_elem_u64 (array : MemArray_u64) (index : UInt256) : FeM UInt256 := do
  -- if index >= array.len {
  if decide (array.len.toNat ≤ index.toNat) then
    -- index_out_of_bounds()
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    -- elem(array.ptr, index)
    let v9 ← elem_u64 array.ptr index
    return v9

/-- `<MemArray<u64> as Index>::index`, its `mut` body -/
def MemArray_index_mut_u64 (self : MemArray_u64) (index : UInt256) : FeM (UInt256 × (UInt256 → FeM MemArray_u64)) := do
  let mut self := self
  -- mut? *mem_array_elem(self, index)
  let v4 ← mem_array_elem_u64 self index
  let v5 := v4
  let slide : UInt256 → FeM MemArray_u64 := fun g => do
    let mut self := self
    return self
  return (v5, slide)

/-- `MemArray<u64>::len` -/
def MemArray_len_u64 (self : MemArray_u64) : FeM UInt256 := do
  return self.len

/-- `<MemArray<u64> as Index>::index`, its `ref` body -/
def MemArray_index_ref_u64 (self : MemArray_u64) (index : UInt256) : FeM UInt256 := do
  -- mut? *mem_array_elem(self, index)
  let v2 ← mem_array_elem_u64 self index
  let v3 := v2
  return v3

/-- `<Range<Known<0>, Known<5>> as Collection>::start` -/
def Range_start_0_5 (self : Range_Known_0_Known_5) : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Range<Known<0>, Known<5>> as Collection>::has` -/
def Range_has_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 5)
  return v4

/-- `<Range<Known<0>, Known<5>> as Collection>::at` -/
def Range_at_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<5>> as Collection>::next` -/
def Range_next_0_5 (self : Range_Known_0_Known_5) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `<usize as IntDowncast>::downcast_truncate` -/
def downcast_truncate_usize_usize (self : UInt256) : FeM UInt256 := do
  -- let word = IntWord::to_word(self)
  let word ← usize_to_word self
  -- if T::IS_SIGNED {
  let v4 ← usize_IS_SIGNED
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if T::IS_SIGNED {
  if v4 then
    -- T::from_word(sign_extend(word: word, mask: T::MASK, sign_bit: T::SIGN_BIT))
    let v5 ← usize_MASK
    let v6 ← usize_SIGN_BIT
    let v7 ← sign_extend word v5 v6
    let v8 ← usize_from_word v7
    join v8
  else
    -- T::from_word(word & T::MASK)
    let v10 ← usize_MASK
    let v12 ← usize_from_word (word &&& v10)
    join v12

/-- `<Range<Known<0>, Known<6>> as Collection>::start` -/
def Range_start_0_6 (self : Range_Known_0_Known_6) : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Range<Known<0>, Known<6>> as Collection>::has` -/
def Range_has_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, HI)
  let v4 ← range_has c (UInt256.ofNat 0) (UInt256.ofNat 6)
  return v4

/-- `<Range<Known<0>, Known<6>> as Collection>::at` -/
def Range_at_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<0>, Known<6>> as Collection>::next` -/
def Range_next_0_6 (self : Range_Known_0_Known_6) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `MemArray<u64>::ptr` -/
def MemArray_ptr_u64 (self : MemArray_u64) : FeM UInt256 := do
  return self.ptr

/-- `<Range<Known<1>, Unknown> as Collection>::start` -/
def Range_start_1 (self : Range_Known_1_Unknown) : FeM UInt256 := do
  return UInt256.ofNat 1

/-- `<Range<Known<1>, Unknown> as Collection>::has` -/
def Range_has_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM Bool := do
  -- range_has(c, LO, self.end)
  let v4 ← range_has c (UInt256.ofNat 1) self.«end»
  return v4

/-- `<Range<Known<1>, Unknown> as Collection>::at` -/
def Range_at_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM UInt256 := do
  let mut c := c
  return c

/-- `<Range<Known<1>, Unknown> as Collection>::next` -/
def Range_next_1 (self : Range_Known_1_Unknown) (c : UInt256) : FeM UInt256 := do
  -- c.wrapping_add(1)
  let v4 ← usize_wrapping_add c (UInt256.ofNat 1)
  return v4

/-- `MemSlice<u64>::from_raw_parts`, its `ref` body -/
def MemSlice_from_raw_parts_ref_u64 (ptr : UInt256) (len : UInt256) : FeM MemSlice_u64 := do
  let mut slice : MemSlice_u64 := default
  -- var slice = Self { ptr, len }
  slice := (MemSlice_u64.mk ptr len)
  return slice

/-- `MemArray<u64>::as_slice`, its `ref` body -/
def MemArray_as_slice_ref_u64 (self : MemArray_u64) : FeM MemSlice_u64 := do
  -- MemSlice::from_raw_parts(ptr: self.ptr, len: self.len)
  let v3 ← MemSlice_from_raw_parts_ref_u64 self.ptr self.len
  return v3

/-- `MemSlice<u64>::slice`, its `ref` body -/
def MemSlice_slice_ref_u64 (self : MemSlice_u64) (offset : UInt256) (len : UInt256) : FeM MemSlice_u64 := do
  let join : FeM MemSlice_u64 := do
    -- index_out_of_bounds()
    let v13 ← index_out_of_bounds
    nomatch v13
  -- if offset > self.len || len > self.len - offset {
  if decide (self.len.toNat < offset.toNat) then
    join
  else
    -- if offset > self.len || len > self.len - offset {
    let v6 ← checkedSub self.len offset
    if decide (v6.toNat < len.toNat) then
      join
    else
      -- Self::from_raw_parts(ptr: elem(self.ptr, offset), len)
      let v11 ← elem_u64 self.ptr offset
      let v12 ← MemSlice_from_raw_parts_ref_u64 v11 len
      return v12

/-- `MemSlice<u64>::head`, its `ref` body -/
def MemSlice_head_ref_u64 (self : MemSlice_u64) (n : UInt256) : FeM MemSlice_u64 := do
  -- self.slice(offset: 0, len: n)
  let v3 ← MemSlice_slice_ref_u64 self (UInt256.ofNat 0) n
  return v3

/-- `MemSlice<u64>::from_raw_parts`, its `mut` body -/
def MemSlice_from_raw_parts_mut_u64 (ptr : UInt256) (len : UInt256) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM Unit)) := do
  let mut slice : MemSlice_u64 := default
  -- var slice = Self { ptr, len }
  slice := (MemSlice_u64.mk ptr len)
  let slide : MemSlice_u64 → FeM Unit := fun g => do
    let mut slice := slice
    slice := g
    return ()
  return (slice, slide)

/-- `MemArray<u64>::as_slice`, its `mut` body -/
def MemArray_as_slice_mut_u64 (self : MemArray_u64) : FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemArray_u64)) := do
  let mut self := self
  let mut v3 : MemSlice_u64 := default
  -- MemSlice::from_raw_parts(ptr: self.ptr, len: self.len)
  let (v3_g, v3_k) ← MemSlice_from_raw_parts_mut_u64 self.ptr self.len
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
  let join : MemSlice_u64 → MemSlice_u64 → FeM (MemSlice_u64 × (MemSlice_u64 → FeM MemSlice_u64)) := fun self v12 => do
    let mut self := self
    let mut v12 := v12
    -- index_out_of_bounds()
    let v13 ← index_out_of_bounds
    nomatch v13
  -- if offset > self.len || len > self.len - offset {
  if decide (self.len.toNat < offset.toNat) then
    join self v12
  else
    -- if offset > self.len || len > self.len - offset {
    let v6 ← checkedSub self.len offset
    if decide (v6.toNat < len.toNat) then
      join self v12
    else
      -- Self::from_raw_parts(ptr: elem(self.ptr, offset), len)
      let v11 ← elem_u64 self.ptr offset
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
  -- self.slice(offset: 0, len: n)
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
  -- cast<T, u8>(ptr)
  let v1 ← cast_u64_u8 ptr
  return v1

/-- `MemSlice<u64>::byte_len` -/
def MemSlice_byte_len_u64 (self : MemSlice_u64) : FeM UInt256 := do
  -- self.len * size_of<T>()
  let v3 ← checkedMul self.len (UInt256.ofNat 8)
  return v3

/-- `MemSlice<u64>::as_bytes` -/
def MemSlice_as_bytes_u64 (self : MemSlice_u64) : FeM MemSlice_u8 := do
  -- MemSlice::from_raw_parts(ptr: cast<T, u8>(self.ptr), len: self.byte_len())
  let v2 ← cast_u64_u8 self.ptr
  let v3 ← MemSlice_byte_len_u64 self
  let v4 ← MemSlice_from_raw_parts_ref_u8 v2 v3
  return v4

/-- `MemSlice<u64>::copy_from` -/
def MemSlice_copy_from_u64 (self : MemSlice_u64) (source : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut self := self
  -- if source.len != self.len {
  if decide (source.len ≠ self.len) then
    -- index_out_of_bounds()
    let v5 ← index_out_of_bounds
    nomatch v5
  else
    -- copy(dest: byte_ptr(self.ptr), source: source.as_bytes())
    let v9 ← byte_ptr_u64 self.ptr
    let v10 ← MemSlice_as_bytes_u64 source
    let _ ← copy v9 v10
    return ((), self)

/-- `MemSlice<u64>::len` -/
def MemSlice_len_u64 (self : MemSlice_u64) : FeM UInt256 := do
  return self.len

/-- `MemSlice<u64>::ptr` -/
def MemSlice_ptr_u64 (self : MemSlice_u64) : FeM UInt256 := do
  return self.ptr

/-- `MemSlice<u64>::zero` -/
def MemSlice_zero_u64 (self_0 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut self_0 := self_0
  -- drop_in_place_n(self.ptr, self.len)
  let _ ← drop_in_place_n_u64 self_0.ptr self_0.len
  -- zero_bytes(byte_ptr(self.ptr), self.byte_len())
  let v5 ← byte_ptr_u64 self_0.ptr
  let v7 ← MemSlice_byte_len_u64 self_0
  let _ ← zero_bytes v5 v7
  return ((), self_0)

/-- `MemArray<u64>::zero` -/
def MemArray_zero_u64 (self : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut self := self
  let mut v1 : MemSlice_u64 := default
  -- self.as_slice().zero()
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
  return ptr

/-- `core::ptr::alloc<ChunkPair>` -/
def alloc_ChunkPair : FeM UInt256 := do
  -- cast<u8, T>(alloc_bytes(size_of<T>()))
  let v1 ← alloc_bytes (UInt256.ofNat 64)
  let v2 ← cast_u8_ChunkPair v1
  return v2

/-- `std::evm::ssz::write_pair` -/
def write_pair (scratch : UInt256) (left : UInt256) (right : UInt256) : FeM Unit := do
  -- scratch.left = left
  mstore scratch left
  -- scratch.right = right
  mstore (scratch + UInt256.ofNat 32) right
  return ()

/-- `std::evm::ssz::alloc_pair` -/
def alloc_pair (left : UInt256) (right : UInt256) : FeM UInt256 := do
  -- let scratch = ptr::alloc<ChunkPair>()
  let scratch ← alloc_ChunkPair
  -- write_pair(scratch, left, right)
  let _ ← write_pair scratch left right
  return scratch

/-- `core::ptr::cast<ChunkPair, u8>` -/
def cast_ChunkPair_u8 (ptr : UInt256) : FeM UInt256 := do
  return ptr

/-- `core::ptr::byte_ptr<ChunkPair>` -/
def byte_ptr_ChunkPair (ptr : UInt256) : FeM UInt256 := do
  -- cast<T, u8>(ptr)
  let v1 ← cast_ChunkPair_u8 ptr
  return v1

/-- `std::evm::ssz::pair_offset` -/
def pair_offset (scratch : UInt256) : FeM UInt256 := do
  -- ptr::byte_ptr(scratch)
  let v1 ← byte_ptr_ChunkPair scratch
  return v1

/-- `std::evm::ssz::sha256_64_to` -/
def sha256_64_to (offset : UInt256) (out : UInt256) : FeM UInt256 := do
  -- ops::staticcall_precompile(
  let v2 ← gas
  let ok ← staticcallPrecompile v2 (UInt256.ofNat 2) offset (UInt256.ofNat 64) out (UInt256.ofNat 32)
  -- if ok == 0 {
  if decide (ok = (UInt256.ofNat 0)) then
    -- ops::revert_empty()
    let v10 ← revertEmpty
    nomatch v10
  else
    -- if ops::returndatasize() < 32 {
    let v13 ← returndatasize
    if decide (v13.toNat < (UInt256.ofNat 32).toNat) then
      -- ops::revert_empty()
      let v16 ← revertEmpty
      nomatch v16
    else
      -- ops::mload(out)
      let v19 ← mload out
      return v19

/-- `std::evm::ssz::hash_pair_buffer` -/
def hash_pair_buffer (scratch : UInt256) : FeM UInt256 := do
  -- let offset = pair_offset(scratch)
  let offset ← pair_offset scratch
  -- sha256_64_to(offset, offset)
  let v3 ← sha256_64_to offset offset
  return v3

/-- `std::evm::ssz::hash_pair` -/
def hash_pair (left : UInt256) (right : UInt256) : FeM UInt256 := do
  -- let scratch = alloc_pair(left, right)
  let scratch ← alloc_pair left right
  -- hash_pair_buffer(scratch)
  let v4 ← hash_pair_buffer scratch
  return v4

/-- `Bytes::len` -/
def Bytes_len (self : Bytes) : FeM UInt256 := do
  return self.len

/-- `std::evm::units::ether` -/
def ether (n : UInt256) : FeM Wei := do
  -- Wei { amount: n * 1_000_000_000_000_000_000 }
  let v3 ← checkedMul n (UInt256.ofNat 1000000000000000000)
  return Wei.mk v3

/-- `<Wei as Ord>::ge` -/
def Wei_ge (self : Wei) (other : Wei) : FeM Bool := do
  return decide (other.amount.toNat ≤ self.amount.toNat)

/-- `std::evm::units::gwei` -/
def gwei (n : UInt256) : FeM Wei := do
  -- Wei { amount: n * 1_000_000_000 }
  let v3 ← checkedMul n (UInt256.ofNat 1000000000)
  return Wei.mk v3

/-- `<Wei as Rem>::rem` -/
def Wei_rem (self : Wei) (other : Wei) : FeM Wei := do
  -- Wei { amount: self.amount % other.amount }
  let v4 ← checkedRem self.amount other.amount
  return Wei.mk v4

/-- `<Wei as Eq>::eq` -/
def Wei_eq (self : Wei) (other : Wei) : FeM Bool := do
  return decide (self.amount = other.amount)

/-- `<Wei as Div>::div` -/
def Wei_div (self : Wei) (other : Wei) : FeM UInt256 := do
  -- self.amount / other.amount
  let v4 ← checkedDiv self.amount other.amount
  return v4

/-- `Bytes::payload_ptr` -/
def Bytes_payload_ptr (self : Bytes) : FeM UInt256 := do
  return self.data + (UInt256.ofNat 32)

/-- `Bytes::span`, its `ref` body -/
def Bytes_span_ref (self : Bytes) : FeM MemSlice_u8 := do
  -- ptr::MemSpan::from_raw_parts(ptr: self.payload_ptr(), len: self.len)
  let v1 ← Bytes_payload_ptr self
  let v3 ← MemSlice_from_raw_parts_ref_u8 v1 self.len
  return v3

/-- `core::ptr::elem<u8>` -/
def elem_u8 (ptr : UInt256) (index : UInt256) : FeM UInt256 := do
  -- offset_bytes(ptr, index.wrapping_mul(size_of<T>()))
  let v4 ← usize_wrapping_mul index (UInt256.ofNat 1)
  return ptr + v4

/-- `MemSlice<u8>::slice`, its `ref` body -/
def MemSlice_slice_ref_u8 (self : MemSlice_u8) (offset : UInt256) (len : UInt256) : FeM MemSlice_u8 := do
  let join : FeM MemSlice_u8 := do
    -- index_out_of_bounds()
    let v13 ← index_out_of_bounds
    nomatch v13
  -- if offset > self.len || len > self.len - offset {
  if decide (self.len.toNat < offset.toNat) then
    join
  else
    -- if offset > self.len || len > self.len - offset {
    let v6 ← checkedSub self.len offset
    if decide (v6.toNat < len.toNat) then
      join
    else
      -- Self::from_raw_parts(ptr: elem(self.ptr, offset), len)
      let v11 ← elem_u8 self.ptr offset
      let v12 ← MemSlice_from_raw_parts_ref_u8 v11 len
      return v12

/-- `<ByteVector<48> as HashTreeRoot>::hash_tree_root` -/
def ByteVector_48_hash_tree_root (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  -- assert!(len == 48)
  if decide (len = (UInt256.ofNat 48)) then
    -- let base = input.slice(offset: start.downcast_unchecked(), len: len.downcast_unchecked()).ptr()
    let v7 ← downcast_unchecked_u256_usize start
    let v8 ← downcast_unchecked_u256_usize len
    let v9 ← MemSlice_slice_ref_u8 input v7 v8
    let base ← MemSlice_ptr_u8 v9
    -- ops::mload(ptr::offset_bytes(base, 32))
    let tail ← mload (base + (UInt256.ofNat 32))
    -- alloc_pair(ops::mload(base), (tail >> 128) << 128)
    let v17 ← mload base
    let v21 := (tail >>> (UInt256.ofNat 128)) <<< (UInt256.ofNat 128)
    let scratch ← alloc_pair v17 v21
    -- hash_pair_buffer(scratch)
    let v24 ← hash_pair_buffer scratch
    return v24
  else
    M.panic 0x01

/-- `std::evm::ssz::hash_tree_root<ByteVector<48>>` -/
def hash_tree_root_ByteVector_48 (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  -- T::hash_tree_root(input, start, len)
  let v3 ← ByteVector_48_hash_tree_root input start len
  return v3

/-- `Bytes::word_at` -/
def Bytes_word_at (self : Bytes) (offset : UInt256) : FeM UInt256 := do
  -- if offset >= self.len {
  if decide (self.len.toNat ≤ offset.toNat) then
    -- super::panic()
    let v5 ← trap
    nomatch v5
  else
    -- ptr::read_padded_word(ptr::offset_bytes(self.payload_ptr(), offset), self.len - offset)
    let v8 ← Bytes_payload_ptr self
    let v12 ← checkedSub self.len offset
    let v13 ← readPaddedWord (v8 + offset) v12
    return v13

def u64_chunk_k1 (v : UInt256) : FeM UInt256 := do
  let mut v := v
  -- v = ((v & 0x0000ffff0000ffff) << 16) | ((v >> 16) & 0x0000ffff0000ffff)
  let v24 := (v &&& (UInt256.ofNat 281470681808895)) <<< (UInt256.ofNat 16)
  let v30 := (v >>> (UInt256.ofNat 16)) &&& (UInt256.ofNat 281470681808895)
  v := (v24 ||| v30)
  -- ((v << 32) | (v >> 32)) << 192
  let v42 := (v <<< (UInt256.ofNat 32)) ||| (v >>> (UInt256.ofNat 32))
  return v42 <<< (UInt256.ofNat 192)

/-- `std::evm::ssz::u64_chunk` -/
def u64_chunk (value : UInt256) : FeM UInt256 := do
  let mut v : UInt256 := default
  -- var v = value as u256
  v := value
  -- v = ((v & 0x00ff00ff00ff00ff) << 8) | ((v >> 8) & 0x00ff00ff00ff00ff)
  let v9 := (v &&& (UInt256.ofNat 71777214294589695)) <<< (UInt256.ofNat 8)
  let v15 := (v >>> (UInt256.ofNat 8)) &&& (UInt256.ofNat 71777214294589695)
  v := (v9 ||| v15)
  u64_chunk_k1 v

/-- `<ByteVector<96> as HashTreeRoot>::hash_tree_root` -/
def ByteVector_96_hash_tree_root (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  -- assert!(len == 96)
  if decide (len = (UInt256.ofNat 96)) then
    -- let base = input.slice(offset: start.downcast_unchecked(), len: len.downcast_unchecked()).ptr()
    let v7 ← downcast_unchecked_u256_usize start
    let v8 ← downcast_unchecked_u256_usize len
    let v9 ← MemSlice_slice_ref_u8 input v7 v8
    let base ← MemSlice_ptr_u8 v9
    -- let scratch = alloc_pair(0, 0)
    let scratch ← alloc_pair (UInt256.ofNat 0) (UInt256.ofNat 0)
    -- let offset = pair_offset(scratch)
    let offset ← pair_offset scratch
    -- let left = sha256_64_to(base, offset)
    let left ← sha256_64_to base offset
    -- write_pair(scratch, ops::mload(ptr::offset_bytes(base, 64)), 0)
    let v23 ← mload (base + (UInt256.ofNat 64))
    let _ ← write_pair scratch v23 (UInt256.ofNat 0)
    -- let right = hash_pair_buffer(scratch)
    let right ← hash_pair_buffer scratch
    -- write_pair(scratch, left, right)
    let _ ← write_pair scratch left right
    -- hash_pair_buffer(scratch)
    let v29 ← hash_pair_buffer scratch
    return v29
  else
    M.panic 0x01

/-- `std::evm::ssz::hash_tree_root<ByteVector<96>>` -/
def hash_tree_root_ByteVector_96 (input : MemSlice_u8) (start : UInt256) (len : UInt256) : FeM UInt256 := do
  -- T::hash_tree_root(input, start, len)
  let v3 ← ByteVector_96_hash_tree_root input start len
  return v3

/-- `std::evm::ssz::merkleize4` -/
def merkleize4 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) : FeM UInt256 := do
  -- let scratch = alloc_pair(a, b)
  let scratch ← alloc_pair a b
  -- let left = hash_pair_buffer(scratch)
  let left ← hash_pair_buffer scratch
  -- write_pair(scratch, c, d)
  let _ ← write_pair scratch c d
  -- let right = hash_pair_buffer(scratch)
  let right ← hash_pair_buffer scratch
  -- write_pair(scratch, left, right)
  let _ ← write_pair scratch left right
  -- hash_pair_buffer(scratch)
  let v12 ← hash_pair_buffer scratch
  return v12

/-- `<(u256, u256, u256, u256) as Merkleize>::merkleize` -/
def tuple_u256_u256_u256_u256_merkleize (self : (UInt256 × UInt256 × UInt256 × UInt256)) : FeM UInt256 := do
  -- merkleize4(self.0, self.1, self.2, self.3)
  let v5 ← merkleize4 self.fst self.snd.fst self.snd.snd.fst self.snd.snd.snd
  return v5

/-- `Bytes::from_validated_abi_tail` -/
def Bytes_from_validated_abi_tail (data : UInt256) (len : UInt256) (size : UInt256) : FeM Bytes := do
  return Bytes.mk data len size

/-- `Bytes::empty` -/
def Bytes_empty : FeM Bytes := do
  -- let data = ptr::alloc_bytes(32)
  let data ← alloc_bytes (UInt256.ofNat 32)
  -- store_word(ptr: data, value: 0)
  let _ ← store_word data (UInt256.ofNat 0)
  -- Self::from_validated_abi_tail(data: data, len: 0, size: 32)
  let v7 ← Bytes_from_validated_abi_tail data (UInt256.ofNat 0) (UInt256.ofNat 32)
  return v7

/-- `Bytes::from_word_prefix` -/
def Bytes_from_word_prefix (word : UInt256) (len : UInt256) : FeM Bytes := do
  -- if len > 32 {
  if decide ((UInt256.ofNat 32).toNat < len.toNat) then
    -- super::panic()
    let v5 ← trap
    nomatch v5
  else
    -- if len == 0 {
    if decide (len = (UInt256.ofNat 0)) then
      -- return Self::empty()
      let v11 ← Bytes_empty
      return v11
    else
      -- let data = ptr::alloc_bytes(64)
      let data ← alloc_bytes (UInt256.ofNat 64)
      -- let shift = (32 - len as u256) * 8
      let v20 ← checkedSub (UInt256.ofNat 32) len
      let shift ← checkedMul v20 (UInt256.ofNat 8)
      -- store_word(ptr: data, value: len as u256)
      let _ ← store_word data len
      -- store_word(
      let _ ← store_word (data + (UInt256.ofNat 32)) ((word >>> shift) <<< shift)
      -- Self::from_validated_abi_tail(data: data, len: len, size: 64)
      let v34 ← Bytes_from_validated_abi_tail data len (UInt256.ofNat 64)
      return v34

/-- `std::evm::ssz::serialize_u64` -/
def serialize_u64 (value : UInt256) : FeM Bytes := do
  -- Bytes::from_word_prefix(word: u64_chunk(value), len: 8)
  let v1 ← u64_chunk value
  let v3 ← Bytes_from_word_prefix v1 (UInt256.ofNat 8)
  return v3

/-- `<Bytes as AbiSize>::IS_DYNAMIC` -/
def Bytes_IS_DYNAMIC : FeM Bool := do
  return true

/-- `<Bytes as AbiSize>::HEAD_SIZE` -/
def Bytes_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `core::abi::abi_field_head_size<Bytes>` -/
def abi_field_head_size_Bytes : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← Bytes_IS_DYNAMIC
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
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
  return self.size

/-- `<Bytes as AbiSize>::payload_size` -/
def Bytes_payload_size (self : Bytes) : FeM UInt256 := do
  -- self.encoded_size()
  let v1 ← Bytes_encoded_size self
  return v1

/-- `core::abi::dynamic_payload_size<Bytes>` -/
def dynamic_payload_size_Bytes (value : Bytes) : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v1 ← Bytes_IS_DYNAMIC
  if v1 then
    -- return value.payload_size()
    let v2 ← Bytes_payload_size value
    return v2
  else
    return UInt256.ofNat 0

/-- `<(Bytes, Bytes, Bytes, Bytes, Bytes) as AbiSize>::payload_size` -/
def tuple_payload_size_Bytes_Bytes_Bytes_Bytes_Bytes (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM UInt256 := do
  -- Self::HEAD_SIZE
  let v1 ← tuple_HEAD_SIZE_Bytes_Bytes_Bytes_Bytes_Bytes
  let v4 ← dynamic_payload_size_Bytes self.fst
  let v8 ← dynamic_payload_size_Bytes self.snd.fst
  let v12 ← dynamic_payload_size_Bytes self.snd.snd.fst
  let v16 ← dynamic_payload_size_Bytes self.snd.snd.snd.fst
  let v20 ← dynamic_payload_size_Bytes self.snd.snd.snd.snd
  return ((((v1 + v4) + v8) + v12) + v16) + v20

/-- `core::abi::abi_payload_size<(Bytes, Bytes, Bytes, Bytes, Bytes)>` -/
def abi_payload_size_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← tuple_payload_size_Bytes_Bytes_Bytes_Bytes_Bytes value
  return v1

/-- `Bytes::encoded_span` -/
def Bytes_encoded_span (self : Bytes) : FeM MemSlice_u8 := do
  -- ptr::MemSpan::from_raw_parts(ptr: self.data, len: word_to_usize(self.size))
  let v3 ← word_to_usize self.size
  let v4 ← MemSlice_from_raw_parts_ref_u8 self.data v3
  return v4

/-- `<Bytes as Encode>::encode` -/
def Bytes_encode_Sol (self : Bytes) (out : AbiFrame) : FeM Unit := do
  -- out.copy_from(at: 0, self.encoded_span())
  let v3 ← Bytes_encoded_span self
  let _ ← AbiFrame_copy_from out (UInt256.ofNat 0) v3
  return ()

/-- `core::abi::encode_field<Sol, Bytes>` -/
def encode_field_Sol_Bytes (value : Bytes) (out : AbiFrame) (head_pos : UInt256) (tails : AbiFrame) : FeM AbiFrame := do
  let mut tails := tails
  let mut r1 : AbiFrame := default
  let mut r2 : AbiFrame := default
  -- if T::IS_DYNAMIC {
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    -- let size = value.payload_size()
    let size ← Bytes_payload_size value
    -- out.store_word(at: head_pos, value: out.len() - tails.len())
    let v6 ← AbiFrame_len out
    let v8 ← AbiFrame_len tails
    let _ ← AbiFrame_store_word out head_pos (v6 - v8)
    -- value.encode(tails.slice(offset: 0, len: size))
    let v13 ← AbiFrame_slice tails (UInt256.ofNat 0) size
    r1 := v13
    let _ ← Bytes_encode_Sol value r1
    -- return tails.tail(size)
    let v17 ← AbiFrame_tail tails size
    return v17
  else
    -- value.encode(out.slice(offset: head_pos, len: T::HEAD_SIZE))
    let v20 ← Bytes_HEAD_SIZE
    let v21 ← AbiFrame_slice out head_pos v20
    r2 := v21
    let _ ← Bytes_encode_Sol value r2
    return tails

def tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes_k1 (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) (out : AbiFrame) (head1 : UInt256) (head2 : UInt256) (head3 : UInt256) (v18 : UInt256) (tails_28 : AbiFrame) (tails : AbiFrame) : FeM Unit := do
  let mut tails := tails
  let tails_31 ← encode_field_Sol_Bytes self.snd.fst out head1 tails_28
  tails := tails_31
  -- tails = encode_field<A, T2>(v2, out, head_pos: head2, tails)
  let tails_34 := tails
  let tails_37 ← encode_field_Sol_Bytes self.snd.snd.fst out head2 tails_34
  tails := tails_37
  -- tails = encode_field<A, T3>(v3, out, head_pos: head3, tails)
  let tails_40 := tails
  let tails_43 ← encode_field_Sol_Bytes self.snd.snd.snd.fst out head3 tails_40
  tails := tails_43
  -- encode_field<A, T4>(v4, out, head_pos: head4, tails)
  let tails_46 := tails
  let _ ← encode_field_Sol_Bytes self.snd.snd.snd.snd out (head3 + v18) tails_46
  return ()

/-- `<(Bytes, Bytes, Bytes, Bytes, Bytes) as Encode>::encode` -/
def tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes (self : (Bytes × Bytes × Bytes × Bytes × Bytes)) (out : AbiFrame) : FeM Unit := do
  let mut tails : AbiFrame := default
  -- var tails = out.tail(Self::HEAD_SIZE)
  let v7 ← tuple_HEAD_SIZE_Bytes_Bytes_Bytes_Bytes_Bytes
  let tails_8 ← AbiFrame_tail out v7
  tails := tails_8
  -- let head1 = abi_field_head_size<T0>()
  let head1 ← abi_field_head_size_Bytes
  -- let head2 = head1 + abi_field_head_size<T1>()
  let v12 ← abi_field_head_size_Bytes
  let head2 := head1 + v12
  -- let head3 = head2 + abi_field_head_size<T2>()
  let v15 ← abi_field_head_size_Bytes
  let head3 := head2 + v15
  -- let head4 = head3 + abi_field_head_size<T3>()
  let v18 ← abi_field_head_size_Bytes
  -- tails = encode_field<A, T0>(v0, out, head_pos: 0, tails)
  let tails_22 := tails
  let tails_25 ← encode_field_Sol_Bytes self.fst out (UInt256.ofNat 0) tails_22
  tails := tails_25
  -- tails = encode_field<A, T1>(v1, out, head_pos: head1, tails)
  let tails_28 := tails
  tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes_k1 self out head1 head2 head3 v18 tails_28 tails

/-- `<Bytes as Drop>::drop` -/
def Bytes_drop (self : Bytes) : FeM Unit := do
  -- ptr::dealloc(self.data)
  let _ ← dealloc_u8 self.data
  return ()

/-- `core::abi::encode_alloc<Sol, (Bytes, Bytes, Bytes, Bytes, Bytes)>` -/
def encode_alloc_Sol_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_tuple_Bytes_Bytes_Bytes_Bytes_Bytes value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← tuple_encode_Sol_Bytes_Bytes_Bytes_Bytes_Bytes value r2
  let _ ← Bytes_drop value.fst
  let _ ← Bytes_drop value.snd.fst
  let _ ← Bytes_drop value.snd.snd.fst
  let _ ← Bytes_drop value.snd.snd.snd.fst
  let _ ← Bytes_drop value.snd.snd.snd.snd
  return buffer

/-- `std::evm::effects::encode_abi_payload<(Bytes, Bytes, Bytes, Bytes, Bytes)>` -/
def encode_abi_payload_tuple_Bytes_Bytes_Bytes_Bytes_Bytes (value : (Bytes × Bytes × Bytes × Bytes × Bytes)) : FeM MemBuffer := do
  -- encode_alloc<Sol, T>(value)
  let v2 ← encode_alloc_Sol_tuple_Bytes_Bytes_Bytes_Bytes_Bytes value
  return v2

/-- `<Evm as Log>::log1` -/
def Evm_log1 (data : MemSlice_u8) (topic0 : UInt256) : FeM Unit := do
  -- ops::log1(offset: data.ptr(), len: data.len() as u256, topic0: topic0)
  let v3 ← MemSlice_ptr_u8 data
  let v4 ← MemSlice_len_u8 data
  let _ ← logData v3 v4 #[topic0]
  return ()

/-- `std::evm::ssz::mix_in_length` -/
def mix_in_length (root : UInt256) (len : UInt256) : FeM UInt256 := do
  -- hash_pair(left: root, right: u64_chunk(len))
  let v2 ← u64_chunk len
  let v3 ← hash_pair root v2
  return v3

/-- `<FixedBytes<32> as AbiSize>::IS_DYNAMIC` -/
def FixedBytes_IS_DYNAMIC_32 : FeM Bool := do
  return false

/-- `<FixedBytes<32> as AbiSize>::HEAD_SIZE` -/
def FixedBytes_HEAD_SIZE_32 : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `core::abi::abi_field_head_size<FixedBytes<32>>` -/
def abi_field_head_size_FixedBytes_32 : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← FixedBytes_IS_DYNAMIC_32
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← FixedBytes_HEAD_SIZE_32
    return v4

/-- `core::abi::MAX_PAYLOAD_LEN` -/
def MAX_PAYLOAD_LEN : FeM UInt256 := do
  let v0 ← MAX_ALLOCATION
  let v3 ← checkedSub v0 (UInt256.ofNat 31)
  return v3

/-- `core::abi::checked_dynamic_payload_after_head<Sol, CallData>` -/
def checked_dynamic_payload_after_head_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  -- let data_len = input.word_at(pos)
  let data_len ← CallData_word_at input pos
  -- if data_len > input_len - (pos + 32) || data_len > MAX_PAYLOAD_LEN - 32 {
  let v8 ← checkedAdd pos (UInt256.ofNat 32)
  let v9 ← checkedSub input_len v8
  let join : FeM (UInt256 × UInt256) := do
    -- A::decode_error()
    let v23 ← Sol_decode_error
    nomatch v23
  -- if data_len > input_len - (pos + 32) || data_len > MAX_PAYLOAD_LEN - 32 {
  if decide (v9.toNat < data_len.toNat) then
    join
  else
    -- if data_len > input_len - (pos + 32) || data_len > MAX_PAYLOAD_LEN - 32 {
    let v11 ← MAX_PAYLOAD_LEN
    let v13 ← checkedSub v11 (UInt256.ofNat 32)
    if decide (v13.toNat < data_len.toNat) then
      join
    else
      -- let payload_size = (data_len + 63) & 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0
      let v18 ← checkedAdd data_len (UInt256.ofNat 63)
      let payload_size := v18 &&& (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007913129639904)
      return (data_len, payload_size)

/-- `core::abi::checked_dynamic_payload_with_input_len<Sol, CallData>` -/
def checked_dynamic_payload_with_input_len_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  -- checked_frame_end<A>(pos, 32, input_len)
  let _ ← checked_frame_end_Sol pos (UInt256.ofNat 32) input_len
  -- checked_dynamic_payload_after_head<A, I>(input, pos, input_len)
  let v5 ← checked_dynamic_payload_after_head_Sol_CallData input pos input_len
  return v5

/-- `core::ptr::alloc_bytes_unchecked` -/
def alloc_bytes_unchecked (len : UInt256) : FeM UInt256 := do
  -- alloc_raw(len)
  let v1 ← alloc len
  return v1

/-- `<CallData as ByteSource>::copy_to_memory_unchecked` -/
def CallData_copy_to_memory_unchecked (self : CallData) (dest : UInt256) (src_offset : UInt256) (len : UInt256) : FeM Unit := do
  -- calldatacopy(dest: dest, offset: self.base + src_offset, len: len)
  let _ ← calldatacopy dest (self.base + src_offset) len
  return ()

/-- `core::abi::copy_padded_payload<CallData>` -/
def copy_padded_payload_CallData (input : CallData) (pos : UInt256) (dest : UInt256) (actual_size : UInt256) (size : UInt256) : FeM Unit := do
  -- store_word(ptr: ptr_offset(dest, size - 32), value: 0)
  let v7 ← checkedSub size (UInt256.ofNat 32)
  let v8 ← ptr_offset dest v7
  let _ ← store_word v8 (UInt256.ofNat 0)
  -- input.copy_to_memory_unchecked(dest: dest, src_offset: pos, len: actual_size)
  let _ ← CallData_copy_to_memory_unchecked input dest pos actual_size
  return ()

/-- `core::abi::copy_dynamic_bytes<CallData>` -/
def copy_dynamic_bytes_CallData (input : CallData) (pos : UInt256) (dest : UInt256) (data_len : UInt256) (size : UInt256) : FeM Unit := do
  -- copy_padded_payload(input, pos, dest, 32 + data_len, size)
  let v7 ← checkedAdd (UInt256.ofNat 32) data_len
  let _ ← copy_padded_payload_CallData input pos dest v7 size
  return ()

/-- `core::abi::copy_dynamic_payload_from_with_input_len<Sol, CallData>` -/
def copy_dynamic_payload_from_with_input_len_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256 × UInt256) := do
  -- let payload: (u256, u256) = checked_dynamic_payload_with_input_len<A, I>(input, pos, input_len)
  let payload ← checked_dynamic_payload_with_input_len_Sol_CallData input pos input_len
  -- let data_len = payload.0
  let data_len := payload.fst
  -- let payload_size = payload.1
  let payload_size := payload.snd
  -- ptr::alloc_bytes_unchecked(word_to_usize(payload_size))
  let v9 ← word_to_usize payload_size
  let data_ptr ← alloc_bytes_unchecked v9
  -- copy_dynamic_bytes(input, pos, data_ptr, data_len, payload_size)
  let _ ← copy_dynamic_bytes_CallData input pos data_ptr data_len payload_size
  return (data_ptr, data_len, payload_size)

/-- `<Bytes as Decode>::decode_from_bounded<CallData>` -/
def Bytes_decode_from_bounded_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  -- let payload: (*u8, u256, u256) = copy_dynamic_payload_from_with_input_len<A, I>(input, pos, input_len)
  let payload ← copy_dynamic_payload_from_with_input_len_Sol_CallData input pos input_len
  -- Bytes::from_validated_abi_tail(
  let v7 ← word_to_usize payload.snd.fst
  let v9 ← Bytes_from_validated_abi_tail payload.fst v7 payload.snd.snd
  return v9

/-- `core::abi::copy_dynamic_payload_from<Sol, CallData>` -/
def copy_dynamic_payload_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM (UInt256 × UInt256 × UInt256) := do
  -- let input_len = input.len()
  let input_len ← len_CallData input
  -- copy_dynamic_payload_from_with_input_len<A, I>(input, pos, input_len)
  let v4 ← copy_dynamic_payload_from_with_input_len_Sol_CallData input pos input_len
  return v4

/-- `<Bytes as Decode>::decode_from<CallData>` -/
def Bytes_decode_from_Sol_CallData (input : CallData) (pos : UInt256) : FeM Bytes := do
  -- let payload: (*u8, u256, u256) = copy_dynamic_payload_from<A, I>(input, pos)
  let payload ← copy_dynamic_payload_from_Sol_CallData input pos
  -- Bytes::from_validated_abi_tail(
  let v6 ← word_to_usize payload.snd.fst
  let v8 ← Bytes_from_validated_abi_tail payload.fst v6 payload.snd.snd
  return v8

/-- `core::abi::decode_field_from_prechecked_head<Sol, Bytes, CallData>` -/
def decode_field_from_prechecked_head_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  -- if T::IS_DYNAMIC {
  let v4 ← Bytes_IS_DYNAMIC
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← Bytes_decode_from_bounded_Sol_CallData input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← Bytes_decode_from_Sol_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, Bytes, CallData>` -/
def decode_field_from_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM Bytes := do
  -- if T::IS_DYNAMIC {
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- return T::decode_from(input, checked_tail<A>(base, tail_rel))
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← Bytes_decode_from_Sol_CallData input v6
    return v7
  else
    -- T::decode_from(input, head_pos)
    let v10 ← Bytes_decode_from_Sol_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, Bytes, CallData>` -/
def decode_msg_field_from_Sol_Bytes_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM Bytes := do
  -- if T::IS_DYNAMIC {
  let v4 ← Bytes_IS_DYNAMIC
  if v4 then
    -- return decode_field_from_prechecked_head<A, T, I>(input, base: base, head_pos: head_pos, input_len: input_len)
    let v5 ← decode_field_from_prechecked_head_Sol_Bytes_CallData input base head_pos input_len
    return v5
  else
    -- decode_field_from<A, T, I>(input, base: base, head_pos: head_pos)
    let v8 ← decode_field_from_Sol_Bytes_CallData input base head_pos
    return v8

/-- `std::abi::sol::types::decode_fixed_bytes_word` -/
def decode_fixed_bytes_word (word : UInt256) (len : UInt256) : FeM UInt256 := do
  let join : FeM UInt256 := do
    -- revert_empty()
    let v33 ← revertEmpty
    nomatch v33
  -- if len == 0 || len > 32 {
  if decide (len = (UInt256.ofNat 0)) then
    join
  else
    -- if len == 0 || len > 32 {
    if decide ((UInt256.ofNat 32).toNat < len.toNat) then
      join
    else
      -- if len == 32 {
      if decide (len = (UInt256.ofNat 32)) then
        return word
      else
        -- if word << (len * 8) == 0 {
        let v22 := decide ((word <<< (len * (UInt256.ofNat 8))) = (UInt256.ofNat 0))
        if v22 then
          -- return word >> ((32 - len) * 8)
          let v29 := word >>> (((UInt256.ofNat 32) - len) * (UInt256.ofNat 8))
          return v29
        else
          -- revert_empty()
          let v32 ← revertEmpty
          let v32 : UInt256 := nomatch v32
          return v32

/-- `<FixedBytes<32> as Decode>::decode_from<CallData>` -/
def FixedBytes_decode_from_Sol_32_CallData (input : CallData) (pos : UInt256) : FeM FixedBytes_32 := do
  -- FixedBytes { val: decode_fixed_bytes_word(word: input.word_at(pos), len: N as u256) }
  let v2 ← CallData_word_at input pos
  let v5 ← decode_fixed_bytes_word v2 (UInt256.ofNat 32)
  return FixedBytes_32.mk v5

/-- `core::abi::Decode::decode_from_bounded<FixedBytes<32>, Sol, CallData>` -/
def decode_from_bounded_FixedBytes_32_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  -- checked_frame_end<A>(pos, Self::HEAD_SIZE, input_len)
  let v3 ← FixedBytes_HEAD_SIZE_32
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- Self::decode_from(input, pos)
  let v5 ← FixedBytes_decode_from_Sol_32_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, FixedBytes<32>, CallData>` -/
def decode_field_from_prechecked_head_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  -- if T::IS_DYNAMIC {
  let v4 ← FixedBytes_IS_DYNAMIC_32
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← decode_from_bounded_FixedBytes_32_Sol_CallData input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← FixedBytes_decode_from_Sol_32_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, FixedBytes<32>, CallData>` -/
def decode_field_from_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM FixedBytes_32 := do
  -- if T::IS_DYNAMIC {
  let v3 ← FixedBytes_IS_DYNAMIC_32
  if v3 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- return T::decode_from(input, checked_tail<A>(base, tail_rel))
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← FixedBytes_decode_from_Sol_32_CallData input v6
    return v7
  else
    -- T::decode_from(input, head_pos)
    let v10 ← FixedBytes_decode_from_Sol_32_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, FixedBytes<32>, CallData>` -/
def decode_msg_field_from_Sol_FixedBytes_32_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_32 := do
  -- if T::IS_DYNAMIC {
  let v4 ← FixedBytes_IS_DYNAMIC_32
  if v4 then
    -- return decode_field_from_prechecked_head<A, T, I>(input, base: base, head_pos: head_pos, input_len: input_len)
    let v5 ← decode_field_from_prechecked_head_Sol_FixedBytes_32_CallData input base head_pos input_len
    return v5
  else
    -- decode_field_from<A, T, I>(input, base: base, head_pos: head_pos)
    let v8 ← decode_field_from_Sol_FixedBytes_32_CallData input base head_pos
    return v8

/-- A `const` block in `core::abi::AbiSize::{fn_body}`, at `<Bytes>` -/
def const_Bytes : FeM Bool := do
  return false

/-- `core::abi::abi_payload_size<Bytes>` -/
def abi_payload_size_Bytes (value : Bytes) : FeM UInt256 := do
  -- value.payload_size()
  let v1 ← Bytes_payload_size value
  return v1

/-- `core::abi::encode_alloc<Sol, Bytes>` -/
def encode_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut r2 : AbiFrame := default
  -- let total = abi_payload_size(value)
  let total ← abi_payload_size_Bytes value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let v8 ← AbiFrame_from_raw_parts v7 total
  r2 := v8
  -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) })
  let _ ← Bytes_encode_Sol value r2
  let _ ← Bytes_drop value
  return buffer

/-- `core::abi::abi_field_size<Bytes>` -/
def abi_field_size_Bytes (value : Bytes) : FeM UInt256 := do
  -- let payload_size = value.payload_size()
  let payload_size ← Bytes_payload_size value
  -- if T::IS_DYNAMIC {
  let v3 ← Bytes_IS_DYNAMIC
  if v3 then
    -- return 32 + payload_size
    let v5 ← checkedAdd (UInt256.ofNat 32) payload_size
    return v5
  else
    return payload_size

/-- `core::abi::abi_single_root_size<Bytes>` -/
def abi_single_root_size_Bytes (value : Bytes) : FeM UInt256 := do
  -- abi_field_size(value)
  let v1 ← abi_field_size_Bytes value
  return v1

/-- `core::abi::abi_single_root_head_size<Bytes>` -/
def abi_single_root_head_size_Bytes : FeM UInt256 := do
  -- abi_field_head_size<T>()
  let v0 ← abi_field_head_size_Bytes
  return v0

/-- `core::abi::encode_single_root_alloc<Sol, Bytes>` -/
def encode_single_root_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  let mut value := value
  let mut buffer : MemBuffer := default
  let mut out : AbiFrame := default
  -- let total = abi_single_root_size(value)
  let total ← abi_single_root_size_Bytes value
  -- let buffer = ptr::MemBuffer::new(ptr::allocation_len(total))
  let v3 ← allocation_len total
  let buffer_4 ← MemBuffer_new v3
  buffer := buffer_4
  -- AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total)
  let v7 ← MemBuffer_ptr buffer
  let out_8 ← AbiFrame_from_raw_parts v7 total
  -- let out = unsafe { AbiFrame::from_raw_parts(ptr: buffer.ptr(), len: total) }
  out := out_8
  -- encode_field<A, T>(value, out, head_pos: 0, tails: out.tail(abi_single_root_head_size<T>()))
  let v11 ← abi_single_root_head_size_Bytes
  let v13 ← AbiFrame_tail out v11
  let _ ← encode_field_Sol_Bytes value out (UInt256.ofNat 0) v13
  let _ ← Bytes_drop value
  return buffer

/-- `core::abi::encode_return_alloc<Sol, Bytes>` -/
def encode_return_alloc_Sol_Bytes (value : Bytes) : FeM MemBuffer := do
  -- if T::IS_PARAM_LIST {
  let v1 ← const_Bytes
  if v1 then
    -- return encode_alloc<A, T>(value)
    let v3 ← encode_alloc_Sol_Bytes value
    return v3
  else
    -- encode_single_root_alloc<A, T>(value)
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
  -- if !T::IS_DYNAMIC {
  let v1 ← Bytes_IS_DYNAMIC
  if !v1 then
    -- if T::HEAD_SIZE <= 32 {
    let v3 ← Bytes_HEAD_SIZE
    if decide (v3.toNat ≤ (UInt256.ofNat 32).toNat) then
      -- let out = ptr::FixedMemBuffer<32>::alloc()
      let out_6 ← FixedMemBuffer_alloc_32
      out_r1 := out_6
      -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
      let v9 ← FixedMemBuffer_ptr_32 out_r1
      let v10 ← Bytes_HEAD_SIZE
      let v11 ← AbiFrame_from_raw_parts v9 v10
      r5 := v11
      let _ ← Bytes_encode_Sol value r5
      -- self.return_data(
      let v16 ← FixedMemBuffer_ptr_32 out_r1
      let v17 ← Bytes_HEAD_SIZE
      let v18 ← downcast_unchecked_u256_usize v17
      let v19 ← MemSlice_from_raw_parts_ref_u8 v16 v18
      let v22 ← Evm_return_data v19
      nomatch v22
    else
      -- if T::HEAD_SIZE <= 32 {
      let v23 ← Bytes_HEAD_SIZE
      if decide (v23.toNat ≤ (UInt256.ofNat 64).toNat) then
        -- let out = ptr::FixedMemBuffer<64>::alloc()
        let out_26 ← FixedMemBuffer_alloc_64
        out_r2 := out_26
        -- value.encode(AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE))
        let v29 ← FixedMemBuffer_ptr_64 out_r2
        let v30 ← Bytes_HEAD_SIZE
        let v31 ← AbiFrame_from_raw_parts v29 v30
        r6 := v31
        let _ ← Bytes_encode_Sol value r6
        -- self.return_data(
        let v36 ← FixedMemBuffer_ptr_64 out_r2
        let v37 ← Bytes_HEAD_SIZE
        let v38 ← downcast_unchecked_u256_usize v37
        let v39 ← MemSlice_from_raw_parts_ref_u8 v36 v38
        let v42 ← Evm_return_data v39
        nomatch v42
      else
        -- let out = MemBuffer::new(T::HEAD_SIZE.downcast_unchecked())
        let v43 ← Bytes_HEAD_SIZE
        let v44 ← downcast_unchecked_u256_usize v43
        let out_45 ← MemBuffer_new v44
        out_r3 := out_45
        -- AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE)
        let v48 ← MemBuffer_ptr out_r3
        let v49 ← Bytes_HEAD_SIZE
        let v50 ← AbiFrame_from_raw_parts v48 v49
        r7 := v50
        -- value.encode(unsafe { AbiFrame::from_raw_parts(ptr: out.ptr(), len: T::HEAD_SIZE) })
        let _ ← Bytes_encode_Sol value r7
        -- self.return_data(data: out.span())
        let v55 ← MemBuffer_span_ref out_r3
        let v58 ← Evm_return_data v55
        nomatch v58
  else
    -- let out = encode_return_alloc<A, T>(value)
    let value_59 := value
    let out_61 ← encode_return_alloc_Sol_Bytes value_59
    out_r4 := out_61
    -- self.return_bytes(data: out.span())
    let v64 ← MemBuffer_span_ref out_r4
    let v67 ← Evm_return_bytes v64
    nomatch v67

/-- `<FixedBytes<4> as AbiSize>::IS_DYNAMIC` -/
def FixedBytes_IS_DYNAMIC_4 : FeM Bool := do
  return false

/-- `<FixedBytes<4> as AbiSize>::HEAD_SIZE` -/
def FixedBytes_HEAD_SIZE_4 : FeM UInt256 := do
  return UInt256.ofNat 32

/-- `core::abi::abi_field_head_size<FixedBytes<4>>` -/
def abi_field_head_size_FixedBytes_4 : FeM UInt256 := do
  -- if T::IS_DYNAMIC {
  let v0 ← FixedBytes_IS_DYNAMIC_4
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← FixedBytes_HEAD_SIZE_4
    return v4

/-- `<FixedBytes<4> as Decode>::decode_from<CallData>` -/
def FixedBytes_decode_from_Sol_4_CallData (input : CallData) (pos : UInt256) : FeM FixedBytes_4 := do
  -- FixedBytes { val: decode_fixed_bytes_word(word: input.word_at(pos), len: N as u256) }
  let v2 ← CallData_word_at input pos
  let v5 ← decode_fixed_bytes_word v2 (UInt256.ofNat 4)
  return FixedBytes_4.mk v5

/-- `core::abi::Decode::decode_from_bounded<FixedBytes<4>, Sol, CallData>` -/
def decode_from_bounded_FixedBytes_4_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  -- checked_frame_end<A>(pos, Self::HEAD_SIZE, input_len)
  let v3 ← FixedBytes_HEAD_SIZE_4
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- Self::decode_from(input, pos)
  let v5 ← FixedBytes_decode_from_Sol_4_CallData input pos
  return v5

/-- `core::abi::decode_field_from_prechecked_head<Sol, FixedBytes<4>, CallData>` -/
def decode_field_from_prechecked_head_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  -- if T::IS_DYNAMIC {
  let v4 ← FixedBytes_IS_DYNAMIC_4
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← decode_from_bounded_FixedBytes_4_Sol_CallData input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← FixedBytes_decode_from_Sol_4_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, FixedBytes<4>, CallData>` -/
def decode_field_from_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM FixedBytes_4 := do
  -- if T::IS_DYNAMIC {
  let v3 ← FixedBytes_IS_DYNAMIC_4
  if v3 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- return T::decode_from(input, checked_tail<A>(base, tail_rel))
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← FixedBytes_decode_from_Sol_4_CallData input v6
    return v7
  else
    -- T::decode_from(input, head_pos)
    let v10 ← FixedBytes_decode_from_Sol_4_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, FixedBytes<4>, CallData>` -/
def decode_msg_field_from_Sol_FixedBytes_4_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM FixedBytes_4 := do
  -- if T::IS_DYNAMIC {
  let v4 ← FixedBytes_IS_DYNAMIC_4
  if v4 then
    -- return decode_field_from_prechecked_head<A, T, I>(input, base: base, head_pos: head_pos, input_len: input_len)
    let v5 ← decode_field_from_prechecked_head_Sol_FixedBytes_4_CallData input base head_pos input_len
    return v5
  else
    -- decode_field_from<A, T, I>(input, base: base, head_pos: head_pos)
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
  -- if T::IS_DYNAMIC {
  let v0 ← tuple_IS_DYNAMIC_u256_u256
  if v0 then
    return UInt256.ofNat 32
  else
    -- <T as AbiSize>::HEAD_SIZE
    let v4 ← tuple_HEAD_SIZE_u256_u256
    return v4

/-- `<(u256, u256) as Decode>::decode_from_bounded<CallData>` -/
def tuple_decode_from_bounded_Sol_u256_u256_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  -- checked_frame_end<A>(pos, Self::HEAD_SIZE, input_len)
  let v3 ← tuple_HEAD_SIZE_u256_u256
  let _ ← checked_frame_end_Sol pos v3 input_len
  -- Self::decode_from_prechecked_head(input, pos, input_len)
  let v5 ← tuple_decode_from_prechecked_head_Sol_u256_u256_CallData input pos input_len
  return v5

/-- `<(u256, u256) as Decode>::decode_from<CallData>` -/
def tuple_decode_from_Sol_u256_u256_CallData (input : CallData) (pos : UInt256) : FeM (UInt256 × UInt256) := do
  let mut head_pos : UInt256 := default
  -- var head_pos = pos
  head_pos := pos
  -- let v0: T0 = decode_field_from<A, T0, I>(input, base: pos, head_pos: head_pos)
  let v0 ← decode_field_from_Sol_u256_CallData input pos head_pos
  -- head_pos += abi_field_head_size<T0>()
  let v8 ← abi_field_head_size_u256
  let head_pos_10 ← checkedAdd head_pos v8
  head_pos := head_pos_10
  -- let v1: T1 = decode_field_from<A, T1, I>(input, base: pos, head_pos: head_pos)
  let v1 ← decode_field_from_Sol_u256_CallData input pos head_pos
  return (v0, v1)

/-- `core::abi::decode_field_from_prechecked_head<Sol, (u256, u256), CallData>` -/
def decode_field_from_prechecked_head_Sol_tuple_u256_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  -- if T::IS_DYNAMIC {
  let v4 ← tuple_IS_DYNAMIC_u256_u256
  if v4 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- let tail_at = checked_tail<A>(base, tail_rel)
    let tail_at ← checked_tail_Sol base tail_rel
    -- return T::decode_from_bounded(input, tail_at, input_len)
    let v9 ← tuple_decode_from_bounded_Sol_u256_u256_CallData input tail_at input_len
    return v9
  else
    -- T::decode_from(input, head_pos)
    let v12 ← tuple_decode_from_Sol_u256_u256_CallData input head_pos
    return v12

/-- `core::abi::decode_field_from<Sol, (u256, u256), CallData>` -/
def decode_field_from_Sol_tuple_u256_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) : FeM (UInt256 × UInt256) := do
  -- if T::IS_DYNAMIC {
  let v3 ← tuple_IS_DYNAMIC_u256_u256
  if v3 then
    -- let tail_rel = input.word_at(head_pos)
    let tail_rel ← CallData_word_at input head_pos
    -- return T::decode_from(input, checked_tail<A>(base, tail_rel))
    let v6 ← checked_tail_Sol base tail_rel
    let v7 ← tuple_decode_from_Sol_u256_u256_CallData input v6
    return v7
  else
    -- T::decode_from(input, head_pos)
    let v10 ← tuple_decode_from_Sol_u256_u256_CallData input head_pos
    return v10

/-- `core::abi::decode_msg_field_from<Sol, (u256, u256), CallData>` -/
def decode_msg_field_from_Sol_tuple_u256_u256_CallData (input : CallData) (base : UInt256) (head_pos : UInt256) (input_len : UInt256) : FeM (UInt256 × UInt256) := do
  -- if T::IS_DYNAMIC {
  let v4 ← tuple_IS_DYNAMIC_u256_u256
  if v4 then
    -- return decode_field_from_prechecked_head<A, T, I>(input, base: base, head_pos: head_pos, input_len: input_len)
    let v5 ← decode_field_from_prechecked_head_Sol_tuple_u256_u256_CallData input base head_pos input_len
    return v5
  else
    -- decode_field_from<A, T, I>(input, base: base, head_pos: head_pos)
    let v8 ← decode_field_from_Sol_tuple_u256_u256_CallData input base head_pos
    return v8


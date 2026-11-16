export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_index_ref MemBuffer_len MemBuffer_new MemBuffer_span_ref MemBuffer_write_le_u64 MemSlice_empty_u8 Range_Known_0_Known_16 Range_Known_0_Known_16.mk Range_Known_0_Known_8 Range_Known_0_Known_8.mk Range_at_0_16 Range_at_0_8 Range_has_0_16 Range_has_0_8 Range_next_0_16 Range_next_0_8 Range_start_0_16 Range_start_0_8 downcast_truncate_u64_usize downcast_unchecked_u256_usize u64_wrapping_add «assert»)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1)

/-- The `init` of contract `Blake2F`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_1 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  -- let len = raw.calldatasize()
  let len ← Evm_calldatasize
  -- var input = MemBuffer::new(len.downcast_unchecked())
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  -- raw.calldatacopy(dest: mut input, dest_offset: 0, src_offset: 0, len)
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  return input

/-- `lib::blake2f::INPUT_LEN` -/
def INPUT_LEN : FeM UInt256 := do
  return UInt256.ofNat 213

/-- `lib::blake2f::IV` -/
def IV : FeM (Vector UInt256 8) := do
  let v8 := #v[(UInt256.ofNat 7640891576956012808), (UInt256.ofNat 13503953896175478587), (UInt256.ofNat 4354685564936845355), (UInt256.ofNat 11912009170470909681), (UInt256.ofNat 5840696475078001361), (UInt256.ofNat 11170449401992604703), (UInt256.ofNat 2270897969802886507), (UInt256.ofNat 6620516959819538809)]
  return v8

/-- `lib::blake2f::rounds<MemBuffer>` -/
def rounds_MemBuffer (input : MemBuffer) : FeM UInt256 := do
  -- core::assert(input.len() == INPUT_LEN)
  let v1 ← MemBuffer_len_1 input
  let v2 ← INPUT_LEN
  let _ ← «assert» (decide (v1 = v2))
  -- (input.byte_at(0) as u64) << 24 | (input.byte_at(1) as u64) << 16
  let v6 ← MemBuffer_byte_at input (UInt256.ofNat 0)
  let v11 ← MemBuffer_byte_at input (UInt256.ofNat 1)
  let v15 := (IntTy.shl (.u 64) v6 (UInt256.ofNat 24)) ||| (IntTy.shl (.u 64) v11 (UInt256.ofNat 16))
  let v17 ← MemBuffer_byte_at input (UInt256.ofNat 2)
  let v23 ← MemBuffer_byte_at input (UInt256.ofNat 3)
  let v25 := (v15 ||| (IntTy.shl (.u 64) v17 (UInt256.ofNat 8))) ||| v23
  return v25

/-- `lib::blake2f::rotate_right` -/
def rotate_right (value : UInt256) («by» : UInt256) : FeM UInt256 := do
  -- (value >> by) | (value << (64 - by))
  let v8 ← IntTy.sub (.u 64) (UInt256.ofNat 64) «by»
  return (value >>> «by») ||| (IntTy.shl (.u 64) value v8)

def mix_k4 (b : UInt256) (v72 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex b 16
  v := (v.set! b.toNat v72)
  return ((), v)

def mix_k3 (b : UInt256) (c : UInt256) (d : UInt256) (v56 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex d 16
  v := (v.set! d.toNat v56)
  -- v[c] = v[c].wrapping_add(v[d])
  checkIndex d 16
  checkIndex c 16
  let v63 ← u64_wrapping_add v[c.toNat]! v[d.toNat]!
  checkIndex c 16
  v := (v.set! c.toNat v63)
  -- v[b] = rotate_right(v[b] ^ v[c], by: 63)
  checkIndex b 16
  checkIndex c 16
  let v72 ← rotate_right (v[b.toNat]! ^^^ v[c.toNat]!) (UInt256.ofNat 63)
  mix_k4 b v72 v

def mix_k2 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (y : UInt256) (v38 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  v := (v.set! b.toNat v38)
  -- v[a] = v[a].wrapping_add(v[b]).wrapping_add(y)
  checkIndex b 16
  checkIndex a 16
  let v45 ← u64_wrapping_add v[a.toNat]! v[b.toNat]!
  let v47 ← u64_wrapping_add v45 y
  checkIndex a 16
  v := (v.set! a.toNat v47)
  -- v[d] = rotate_right(v[d] ^ v[a], by: 16)
  checkIndex d 16
  checkIndex a 16
  let v56 ← rotate_right (v[d.toNat]! ^^^ v[a.toNat]!) (UInt256.ofNat 16)
  mix_k3 b c d v56 v

def mix_k1 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (y : UInt256) (v22 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  v := (v.set! d.toNat v22)
  -- v[c] = v[c].wrapping_add(v[d])
  checkIndex d 16
  checkIndex c 16
  let v29 ← u64_wrapping_add v[c.toNat]! v[d.toNat]!
  checkIndex c 16
  v := (v.set! c.toNat v29)
  -- v[b] = rotate_right(v[b] ^ v[c], by: 24)
  checkIndex b 16
  checkIndex c 16
  let v38 ← rotate_right (v[b.toNat]! ^^^ v[c.toNat]!) (UInt256.ofNat 24)
  checkIndex b 16
  mix_k2 a b c d y v38 v

/-- `lib::blake2f::mix` -/
def mix (v : (Vector UInt256 16)) (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (x : UInt256) (y : UInt256) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- v[a] = v[a].wrapping_add(v[b]).wrapping_add(x)
  checkIndex b 16
  checkIndex a 16
  let v11 ← u64_wrapping_add v[a.toNat]! v[b.toNat]!
  let v13 ← u64_wrapping_add v11 x
  checkIndex a 16
  v := (v.set! a.toNat v13)
  -- v[d] = rotate_right(v[d] ^ v[a], by: 32)
  checkIndex d 16
  checkIndex a 16
  let v22 ← rotate_right (v[d.toNat]! ^^^ v[a.toNat]!) (UInt256.ofNat 32)
  checkIndex d 16
  mix_k1 a b c d y v22 v

def ten_rounds_k36 (m : (Vector UInt256 16)) (v787 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v789 := (UInt256.ofNat 12)
  checkIndex v789 16
  let (v791, v791_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v787.toNat]! m[v789.toNat]!
  v := v791_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[13], y: m[0])
  let v797 := (UInt256.ofNat 13)
  checkIndex v797 16
  let v799 := (UInt256.ofNat 0)
  checkIndex v799 16
  let (v801, v801_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v797.toNat]! m[v799.toNat]!
  v := v801_0
  return ((), v)

def ten_rounds_k35 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[15], y: m[11])
  let v767 := (UInt256.ofNat 15)
  checkIndex v767 16
  let v769 := (UInt256.ofNat 11)
  checkIndex v769 16
  let (v771, v771_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v767.toNat]! m[v769.toNat]!
  v := v771_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[9], y: m[14])
  let v777 := (UInt256.ofNat 9)
  checkIndex v777 16
  let v779 := (UInt256.ofNat 14)
  checkIndex v779 16
  let (v781, v781_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v777.toNat]! m[v779.toNat]!
  v := v781_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[3], y: m[12])
  let v787 := (UInt256.ofNat 3)
  checkIndex v787 16
  ten_rounds_k36 m v787 v

def ten_rounds_k34 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[7], y: m[6])
  let v747 := (UInt256.ofNat 7)
  checkIndex v747 16
  let v749 := (UInt256.ofNat 6)
  checkIndex v749 16
  let (v751, v751_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v747.toNat]! m[v749.toNat]!
  v := v751_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[1], y: m[5])
  let v757 := (UInt256.ofNat 1)
  checkIndex v757 16
  let v759 := (UInt256.ofNat 5)
  checkIndex v759 16
  let (v761, v761_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v757.toNat]! m[v759.toNat]!
  v := v761_0
  ten_rounds_k35 m v

def ten_rounds_k33 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[10], y: m[2])
  let v727 := (UInt256.ofNat 10)
  checkIndex v727 16
  let v729 := (UInt256.ofNat 2)
  checkIndex v729 16
  let (v731, v731_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v727.toNat]! m[v729.toNat]!
  v := v731_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[8], y: m[4])
  let v737 := (UInt256.ofNat 8)
  checkIndex v737 16
  let v739 := (UInt256.ofNat 4)
  checkIndex v739 16
  let (v741, v741_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v737.toNat]! m[v739.toNat]!
  v := v741_0
  ten_rounds_k34 m v

def ten_rounds_k32 (m : (Vector UInt256 16)) (v697 : UInt256) (v699 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v701, v701_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v697.toNat]! m[v699.toNat]!
  v := v701_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[1], y: m[4])
  let v707 := (UInt256.ofNat 1)
  checkIndex v707 16
  let v709 := (UInt256.ofNat 4)
  checkIndex v709 16
  let (v711, v711_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v707.toNat]! m[v709.toNat]!
  v := v711_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[10], y: m[5])
  let v717 := (UInt256.ofNat 10)
  checkIndex v717 16
  let v719 := (UInt256.ofNat 5)
  checkIndex v719 16
  let (v721, v721_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v717.toNat]! m[v719.toNat]!
  v := v721_0
  ten_rounds_k33 m v

def ten_rounds_k31 (m : (Vector UInt256 16)) (v677 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v679 := (UInt256.ofNat 8)
  checkIndex v679 16
  let (v681, v681_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v677.toNat]! m[v679.toNat]!
  v := v681_0
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[12], y: m[2])
  let v687 := (UInt256.ofNat 12)
  checkIndex v687 16
  let v689 := (UInt256.ofNat 2)
  checkIndex v689 16
  let (v691, v691_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v687.toNat]! m[v689.toNat]!
  v := v691_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[13], y: m[7])
  let v697 := (UInt256.ofNat 13)
  checkIndex v697 16
  let v699 := (UInt256.ofNat 7)
  checkIndex v699 16
  ten_rounds_k32 m v697 v699 v

def ten_rounds_k30 (m : (Vector UInt256 16)) (v657 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v657 16
  let v659 := (UInt256.ofNat 9)
  checkIndex v659 16
  let (v661, v661_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v657.toNat]! m[v659.toNat]!
  v := v661_0
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[11], y: m[3])
  let v667 := (UInt256.ofNat 11)
  checkIndex v667 16
  let v669 := (UInt256.ofNat 3)
  checkIndex v669 16
  let (v671, v671_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v667.toNat]! m[v669.toNat]!
  v := v671_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[0], y: m[8])
  let v677 := (UInt256.ofNat 0)
  checkIndex v677 16
  ten_rounds_k31 m v677 v

def ten_rounds_k29 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[2], y: m[10])
  let v637 := (UInt256.ofNat 2)
  checkIndex v637 16
  let v639 := (UInt256.ofNat 10)
  checkIndex v639 16
  let (v641, v641_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v637.toNat]! m[v639.toNat]!
  v := v641_0
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[6], y: m[15])
  let v647 := (UInt256.ofNat 6)
  checkIndex v647 16
  let v649 := (UInt256.ofNat 15)
  checkIndex v649 16
  let (v651, v651_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v647.toNat]! m[v649.toNat]!
  v := v651_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[14], y: m[9])
  let v657 := (UInt256.ofNat 14)
  ten_rounds_k30 m v657 v

def ten_rounds_k28 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[15], y: m[4])
  let v617 := (UInt256.ofNat 15)
  checkIndex v617 16
  let v619 := (UInt256.ofNat 4)
  checkIndex v619 16
  let (v621, v621_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v617.toNat]! m[v619.toNat]!
  v := v621_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[8], y: m[6])
  let v627 := (UInt256.ofNat 8)
  checkIndex v627 16
  let v629 := (UInt256.ofNat 6)
  checkIndex v629 16
  let (v631, v631_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v627.toNat]! m[v629.toNat]!
  v := v631_0
  ten_rounds_k29 m v

def ten_rounds_k27 (m : (Vector UInt256 16)) (v587 : UInt256) (v589 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v591, v591_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v587.toNat]! m[v589.toNat]!
  v := v591_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[3], y: m[9])
  let v597 := (UInt256.ofNat 3)
  checkIndex v597 16
  let v599 := (UInt256.ofNat 9)
  checkIndex v599 16
  let (v601, v601_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v597.toNat]! m[v599.toNat]!
  v := v601_0
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[5], y: m[0])
  let v607 := (UInt256.ofNat 5)
  checkIndex v607 16
  let v609 := (UInt256.ofNat 0)
  checkIndex v609 16
  let (v611, v611_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v607.toNat]! m[v609.toNat]!
  v := v611_0
  ten_rounds_k28 m v

def ten_rounds_k26 (m : (Vector UInt256 16)) (v567 : UInt256) (v569 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v569 16
  let (v571, v571_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v567.toNat]! m[v569.toNat]!
  v := v571_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[7], y: m[14])
  let v577 := (UInt256.ofNat 7)
  checkIndex v577 16
  let v579 := (UInt256.ofNat 14)
  checkIndex v579 16
  let (v581, v581_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v577.toNat]! m[v579.toNat]!
  v := v581_0
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[12], y: m[1])
  let v587 := (UInt256.ofNat 12)
  checkIndex v587 16
  let v589 := (UInt256.ofNat 1)
  checkIndex v589 16
  ten_rounds_k27 m v587 v589 v

def ten_rounds_k25 (m : (Vector UInt256 16)) (v547 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v549 := (UInt256.ofNat 2)
  checkIndex v549 16
  let (v551, v551_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v547.toNat]! m[v549.toNat]!
  v := v551_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[8], y: m[11])
  let v557 := (UInt256.ofNat 8)
  checkIndex v557 16
  let v559 := (UInt256.ofNat 11)
  checkIndex v559 16
  let (v561, v561_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v557.toNat]! m[v559.toNat]!
  v := v561_0
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[13], y: m[11])
  let v567 := (UInt256.ofNat 13)
  checkIndex v567 16
  let v569 := (UInt256.ofNat 11)
  ten_rounds_k26 m v567 v569 v

def ten_rounds_k24 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[0], y: m[7])
  let v527 := (UInt256.ofNat 0)
  checkIndex v527 16
  let v529 := (UInt256.ofNat 7)
  checkIndex v529 16
  let (v531, v531_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v527.toNat]! m[v529.toNat]!
  v := v531_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[6], y: m[3])
  let v537 := (UInt256.ofNat 6)
  checkIndex v537 16
  let v539 := (UInt256.ofNat 3)
  checkIndex v539 16
  let (v541, v541_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v537.toNat]! m[v539.toNat]!
  v := v541_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[9], y: m[2])
  let v547 := (UInt256.ofNat 9)
  checkIndex v547 16
  ten_rounds_k25 m v547 v

def ten_rounds_k23 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[14], y: m[13])
  let v507 := (UInt256.ofNat 14)
  checkIndex v507 16
  let v509 := (UInt256.ofNat 13)
  checkIndex v509 16
  let (v511, v511_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v507.toNat]! m[v509.toNat]!
  v := v511_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[4], y: m[10])
  let v517 := (UInt256.ofNat 4)
  checkIndex v517 16
  let v519 := (UInt256.ofNat 10)
  checkIndex v519 16
  let (v521, v521_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v517.toNat]! m[v519.toNat]!
  v := v521_0
  ten_rounds_k24 m v

def ten_rounds_k22 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[12], y: m[5])
  let v487 := (UInt256.ofNat 12)
  checkIndex v487 16
  let v489 := (UInt256.ofNat 5)
  checkIndex v489 16
  let (v491, v491_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v487.toNat]! m[v489.toNat]!
  v := v491_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[1], y: m[15])
  let v497 := (UInt256.ofNat 1)
  checkIndex v497 16
  let v499 := (UInt256.ofNat 15)
  checkIndex v499 16
  let (v501, v501_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v497.toNat]! m[v499.toNat]!
  v := v501_0
  ten_rounds_k23 m v

def ten_rounds_k21 (m : (Vector UInt256 16)) (v457 : UInt256) (v459 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v461, v461_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v457.toNat]! m[v459.toNat]!
  v := v461_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[15], y: m[14])
  let v467 := (UInt256.ofNat 15)
  checkIndex v467 16
  let v469 := (UInt256.ofNat 14)
  checkIndex v469 16
  let (v471, v471_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v467.toNat]! m[v469.toNat]!
  v := v471_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[1], y: m[9])
  let v477 := (UInt256.ofNat 1)
  checkIndex v477 16
  let v479 := (UInt256.ofNat 9)
  checkIndex v479 16
  let (v481, v481_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v477.toNat]! m[v479.toNat]!
  v := v481_0
  ten_rounds_k22 m v

def ten_rounds_k20 (m : (Vector UInt256 16)) (v437 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v439 := (UInt256.ofNat 3)
  checkIndex v439 16
  let (v441, v441_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v437.toNat]! m[v439.toNat]!
  v := v441_0
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[4], y: m[13])
  let v447 := (UInt256.ofNat 4)
  checkIndex v447 16
  let v449 := (UInt256.ofNat 13)
  checkIndex v449 16
  let (v451, v451_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v447.toNat]! m[v449.toNat]!
  v := v451_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[7], y: m[5])
  let v457 := (UInt256.ofNat 7)
  checkIndex v457 16
  let v459 := (UInt256.ofNat 5)
  checkIndex v459 16
  ten_rounds_k21 m v457 v459 v

def ten_rounds_k19 (m : (Vector UInt256 16)) (v417 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v417 16
  let v419 := (UInt256.ofNat 10)
  checkIndex v419 16
  let (v421, v421_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v417.toNat]! m[v419.toNat]!
  v := v421_0
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[0], y: m[11])
  let v427 := (UInt256.ofNat 0)
  checkIndex v427 16
  let v429 := (UInt256.ofNat 11)
  checkIndex v429 16
  let (v431, v431_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v427.toNat]! m[v429.toNat]!
  v := v431_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[8], y: m[3])
  let v437 := (UInt256.ofNat 8)
  checkIndex v437 16
  ten_rounds_k20 m v437 v

def ten_rounds_k18 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[3], y: m[13])
  let v397 := (UInt256.ofNat 3)
  checkIndex v397 16
  let v399 := (UInt256.ofNat 13)
  checkIndex v399 16
  let (v401, v401_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v397.toNat]! m[v399.toNat]!
  v := v401_0
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[2], y: m[12])
  let v407 := (UInt256.ofNat 2)
  checkIndex v407 16
  let v409 := (UInt256.ofNat 12)
  checkIndex v409 16
  let (v411, v411_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v407.toNat]! m[v409.toNat]!
  v := v411_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[6], y: m[10])
  let v417 := (UInt256.ofNat 6)
  ten_rounds_k19 m v417 v

def ten_rounds_k17 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[11], y: m[12])
  let v377 := (UInt256.ofNat 11)
  checkIndex v377 16
  let v379 := (UInt256.ofNat 12)
  checkIndex v379 16
  let (v381, v381_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v377.toNat]! m[v379.toNat]!
  v := v381_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[6], y: m[8])
  let v387 := (UInt256.ofNat 6)
  checkIndex v387 16
  let v389 := (UInt256.ofNat 8)
  checkIndex v389 16
  let (v391, v391_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v387.toNat]! m[v389.toNat]!
  v := v391_0
  ten_rounds_k18 m v

def ten_rounds_k16 (m : (Vector UInt256 16)) (v347 : UInt256) (v349 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v351, v351_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v347.toNat]! m[v349.toNat]!
  v := v351_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[10], y: m[15])
  let v357 := (UInt256.ofNat 10)
  checkIndex v357 16
  let v359 := (UInt256.ofNat 15)
  checkIndex v359 16
  let (v361, v361_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v357.toNat]! m[v359.toNat]!
  v := v361_0
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[14], y: m[1])
  let v367 := (UInt256.ofNat 14)
  checkIndex v367 16
  let v369 := (UInt256.ofNat 1)
  checkIndex v369 16
  let (v371, v371_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v367.toNat]! m[v369.toNat]!
  v := v371_0
  ten_rounds_k17 m v

def ten_rounds_k15 (m : (Vector UInt256 16)) (v327 : UInt256) (v329 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v329 16
  let (v331, v331_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v327.toNat]! m[v329.toNat]!
  v := v331_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[5], y: m[7])
  let v337 := (UInt256.ofNat 5)
  checkIndex v337 16
  let v339 := (UInt256.ofNat 7)
  checkIndex v339 16
  let (v341, v341_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v337.toNat]! m[v339.toNat]!
  v := v341_0
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[2], y: m[4])
  let v347 := (UInt256.ofNat 2)
  checkIndex v347 16
  let v349 := (UInt256.ofNat 4)
  checkIndex v349 16
  ten_rounds_k16 m v347 v349 v

def ten_rounds_k14 (m : (Vector UInt256 16)) (v307 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v309 := (UInt256.ofNat 0)
  checkIndex v309 16
  let (v311, v311_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v307.toNat]! m[v309.toNat]!
  v := v311_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[15], y: m[8])
  let v317 := (UInt256.ofNat 15)
  checkIndex v317 16
  let v319 := (UInt256.ofNat 8)
  checkIndex v319 16
  let (v321, v321_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v317.toNat]! m[v319.toNat]!
  v := v321_0
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[9], y: m[0])
  let v327 := (UInt256.ofNat 9)
  checkIndex v327 16
  let v329 := (UInt256.ofNat 0)
  ten_rounds_k15 m v327 v329 v

def ten_rounds_k13 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[2], y: m[6])
  let v287 := (UInt256.ofNat 2)
  checkIndex v287 16
  let v289 := (UInt256.ofNat 6)
  checkIndex v289 16
  let (v291, v291_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v287.toNat]! m[v289.toNat]!
  v := v291_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[5], y: m[10])
  let v297 := (UInt256.ofNat 5)
  checkIndex v297 16
  let v299 := (UInt256.ofNat 10)
  checkIndex v299 16
  let (v301, v301_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v297.toNat]! m[v299.toNat]!
  v := v301_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[4], y: m[0])
  let v307 := (UInt256.ofNat 4)
  checkIndex v307 16
  ten_rounds_k14 m v307 v

def ten_rounds_k12 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[13], y: m[12])
  let v267 := (UInt256.ofNat 13)
  checkIndex v267 16
  let v269 := (UInt256.ofNat 12)
  checkIndex v269 16
  let (v271, v271_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v267.toNat]! m[v269.toNat]!
  v := v271_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[11], y: m[14])
  let v277 := (UInt256.ofNat 11)
  checkIndex v277 16
  let v279 := (UInt256.ofNat 14)
  checkIndex v279 16
  let (v281, v281_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v277.toNat]! m[v279.toNat]!
  v := v281_0
  ten_rounds_k13 m v

def ten_rounds_k11 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[7], y: m[9])
  let v247 := (UInt256.ofNat 7)
  checkIndex v247 16
  let v249 := (UInt256.ofNat 9)
  checkIndex v249 16
  let (v251, v251_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v247.toNat]! m[v249.toNat]!
  v := v251_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[3], y: m[1])
  let v257 := (UInt256.ofNat 3)
  checkIndex v257 16
  let v259 := (UInt256.ofNat 1)
  checkIndex v259 16
  let (v261, v261_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v257.toNat]! m[v259.toNat]!
  v := v261_0
  ten_rounds_k12 m v

def ten_rounds_k10 (m : (Vector UInt256 16)) (v217 : UInt256) (v219 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v221, v221_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v217.toNat]! m[v219.toNat]!
  v := v221_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[7], y: m[1])
  let v227 := (UInt256.ofNat 7)
  checkIndex v227 16
  let v229 := (UInt256.ofNat 1)
  checkIndex v229 16
  let (v231, v231_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v227.toNat]! m[v229.toNat]!
  v := v231_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[9], y: m[4])
  let v237 := (UInt256.ofNat 9)
  checkIndex v237 16
  let v239 := (UInt256.ofNat 4)
  checkIndex v239 16
  let (v241, v241_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v237.toNat]! m[v239.toNat]!
  v := v241_0
  ten_rounds_k11 m v

def ten_rounds_k9 (m : (Vector UInt256 16)) (v197 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v199 := (UInt256.ofNat 13)
  checkIndex v199 16
  let (v201, v201_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v197.toNat]! m[v199.toNat]!
  v := v201_0
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[10], y: m[14])
  let v207 := (UInt256.ofNat 10)
  checkIndex v207 16
  let v209 := (UInt256.ofNat 14)
  checkIndex v209 16
  let (v211, v211_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v207.toNat]! m[v209.toNat]!
  v := v211_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[3], y: m[6])
  let v217 := (UInt256.ofNat 3)
  checkIndex v217 16
  let v219 := (UInt256.ofNat 6)
  checkIndex v219 16
  ten_rounds_k10 m v217 v219 v

def ten_rounds_k8 (m : (Vector UInt256 16)) (v177 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v177 16
  let v179 := (UInt256.ofNat 0)
  checkIndex v179 16
  let (v181, v181_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v177.toNat]! m[v179.toNat]!
  v := v181_0
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[5], y: m[2])
  let v187 := (UInt256.ofNat 5)
  checkIndex v187 16
  let v189 := (UInt256.ofNat 2)
  checkIndex v189 16
  let (v191, v191_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v187.toNat]! m[v189.toNat]!
  v := v191_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[15], y: m[13])
  let v197 := (UInt256.ofNat 15)
  checkIndex v197 16
  ten_rounds_k9 m v197 v

def ten_rounds_k7 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[5], y: m[3])
  let v157 := (UInt256.ofNat 5)
  checkIndex v157 16
  let v159 := (UInt256.ofNat 3)
  checkIndex v159 16
  let (v161, v161_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v157.toNat]! m[v159.toNat]!
  v := v161_0
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[11], y: m[8])
  let v167 := (UInt256.ofNat 11)
  checkIndex v167 16
  let v169 := (UInt256.ofNat 8)
  checkIndex v169 16
  let (v171, v171_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v167.toNat]! m[v169.toNat]!
  v := v171_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[12], y: m[0])
  let v177 := (UInt256.ofNat 12)
  ten_rounds_k8 m v177 v

def ten_rounds_k6 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[0], y: m[2])
  let v137 := (UInt256.ofNat 0)
  checkIndex v137 16
  let v139 := (UInt256.ofNat 2)
  checkIndex v139 16
  let (v141, v141_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v137.toNat]! m[v139.toNat]!
  v := v141_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[11], y: m[7])
  let v147 := (UInt256.ofNat 11)
  checkIndex v147 16
  let v149 := (UInt256.ofNat 7)
  checkIndex v149 16
  let (v151, v151_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v147.toNat]! m[v149.toNat]!
  v := v151_0
  ten_rounds_k7 m v

def ten_rounds_k5 (m : (Vector UInt256 16)) (v107 : UInt256) (v109 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v111, v111_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v107.toNat]! m[v109.toNat]!
  v := v111_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[13], y: m[6])
  let v117 := (UInt256.ofNat 13)
  checkIndex v117 16
  let v119 := (UInt256.ofNat 6)
  checkIndex v119 16
  let (v121, v121_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v117.toNat]! m[v119.toNat]!
  v := v121_0
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[1], y: m[12])
  let v127 := (UInt256.ofNat 1)
  checkIndex v127 16
  let v129 := (UInt256.ofNat 12)
  checkIndex v129 16
  let (v131, v131_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v127.toNat]! m[v129.toNat]!
  v := v131_0
  ten_rounds_k6 m v

def ten_rounds_k4 (m : (Vector UInt256 16)) (v87 : UInt256) (v89 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v89 16
  let (v91, v91_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v87.toNat]! m[v89.toNat]!
  v := v91_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[4], y: m[8])
  let v97 := (UInt256.ofNat 4)
  checkIndex v97 16
  let v99 := (UInt256.ofNat 8)
  checkIndex v99 16
  let (v101, v101_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v97.toNat]! m[v99.toNat]!
  v := v101_0
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[9], y: m[15])
  let v107 := (UInt256.ofNat 9)
  checkIndex v107 16
  let v109 := (UInt256.ofNat 15)
  checkIndex v109 16
  ten_rounds_k5 m v107 v109 v

def ten_rounds_k3 (m : (Vector UInt256 16)) (v67 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v69 := (UInt256.ofNat 13)
  checkIndex v69 16
  let (v71, v71_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v67.toNat]! m[v69.toNat]!
  v := v71_0
  -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[14], y: m[15])
  let v77 := (UInt256.ofNat 14)
  checkIndex v77 16
  let v79 := (UInt256.ofNat 15)
  checkIndex v79 16
  let (v81, v81_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v77.toNat]! m[v79.toNat]!
  v := v81_0
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[14], y: m[10])
  let v87 := (UInt256.ofNat 14)
  checkIndex v87 16
  let v89 := (UInt256.ofNat 10)
  ten_rounds_k4 m v87 v89 v

def ten_rounds_k2 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[8], y: m[9])
  let v47 := (UInt256.ofNat 8)
  checkIndex v47 16
  let v49 := (UInt256.ofNat 9)
  checkIndex v49 16
  let (v51, v51_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v47.toNat]! m[v49.toNat]!
  v := v51_0
  -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[10], y: m[11])
  let v57 := (UInt256.ofNat 10)
  checkIndex v57 16
  let v59 := (UInt256.ofNat 11)
  checkIndex v59 16
  let (v61, v61_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v57.toNat]! m[v59.toNat]!
  v := v61_0
  -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[12], y: m[13])
  let v67 := (UInt256.ofNat 12)
  checkIndex v67 16
  ten_rounds_k3 m v67 v

def ten_rounds_k1 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[4], y: m[5])
  let v27 := (UInt256.ofNat 4)
  checkIndex v27 16
  let v29 := (UInt256.ofNat 5)
  checkIndex v29 16
  let (v31, v31_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v27.toNat]! m[v29.toNat]!
  v := v31_0
  -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[6], y: m[7])
  let v37 := (UInt256.ofNat 6)
  checkIndex v37 16
  let v39 := (UInt256.ofNat 7)
  checkIndex v39 16
  let (v41, v41_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v37.toNat]! m[v39.toNat]!
  v := v41_0
  ten_rounds_k2 m v

/-- `lib::blake2f::ten_rounds` -/
def ten_rounds (v : (Vector UInt256 16)) (m : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[0], y: m[1])
  let v7 := (UInt256.ofNat 0)
  checkIndex v7 16
  let v9 := (UInt256.ofNat 1)
  checkIndex v9 16
  let (v11, v11_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v7.toNat]! m[v9.toNat]!
  v := v11_0
  -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[2], y: m[3])
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  let v19 := (UInt256.ofNat 3)
  checkIndex v19 16
  let (v21, v21_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v17.toNat]! m[v19.toNat]!
  v := v21_0
  ten_rounds_k1 m v

def SIGMA_k6 : FeM (Vector UInt256 160) := do
  let v160 := #v[(UInt256.ofNat 0), (UInt256.ofNat 1), (UInt256.ofNat 2), (UInt256.ofNat 3), (UInt256.ofNat 4), (UInt256.ofNat 5), (UInt256.ofNat 6), (UInt256.ofNat 7), (UInt256.ofNat 8), (UInt256.ofNat 9), (UInt256.ofNat 10), (UInt256.ofNat 11), (UInt256.ofNat 12), (UInt256.ofNat 13), (UInt256.ofNat 14), (UInt256.ofNat 15), (UInt256.ofNat 14), (UInt256.ofNat 10), (UInt256.ofNat 4), (UInt256.ofNat 8), (UInt256.ofNat 9), (UInt256.ofNat 15), (UInt256.ofNat 13), (UInt256.ofNat 6), (UInt256.ofNat 1), (UInt256.ofNat 12), (UInt256.ofNat 0), (UInt256.ofNat 2), (UInt256.ofNat 11), (UInt256.ofNat 7), (UInt256.ofNat 5), (UInt256.ofNat 3), (UInt256.ofNat 11), (UInt256.ofNat 8), (UInt256.ofNat 12), (UInt256.ofNat 0), (UInt256.ofNat 5), (UInt256.ofNat 2), (UInt256.ofNat 15), (UInt256.ofNat 13), (UInt256.ofNat 10), (UInt256.ofNat 14), (UInt256.ofNat 3), (UInt256.ofNat 6), (UInt256.ofNat 7), (UInt256.ofNat 1), (UInt256.ofNat 9), (UInt256.ofNat 4), (UInt256.ofNat 7), (UInt256.ofNat 9), (UInt256.ofNat 3), (UInt256.ofNat 1), (UInt256.ofNat 13), (UInt256.ofNat 12), (UInt256.ofNat 11), (UInt256.ofNat 14), (UInt256.ofNat 2), (UInt256.ofNat 6), (UInt256.ofNat 5), (UInt256.ofNat 10), (UInt256.ofNat 4), (UInt256.ofNat 0), (UInt256.ofNat 15), (UInt256.ofNat 8), (UInt256.ofNat 9), (UInt256.ofNat 0), (UInt256.ofNat 5), (UInt256.ofNat 7), (UInt256.ofNat 2), (UInt256.ofNat 4), (UInt256.ofNat 10), (UInt256.ofNat 15), (UInt256.ofNat 14), (UInt256.ofNat 1), (UInt256.ofNat 11), (UInt256.ofNat 12), (UInt256.ofNat 6), (UInt256.ofNat 8), (UInt256.ofNat 3), (UInt256.ofNat 13), (UInt256.ofNat 2), (UInt256.ofNat 12), (UInt256.ofNat 6), (UInt256.ofNat 10), (UInt256.ofNat 0), (UInt256.ofNat 11), (UInt256.ofNat 8), (UInt256.ofNat 3), (UInt256.ofNat 4), (UInt256.ofNat 13), (UInt256.ofNat 7), (UInt256.ofNat 5), (UInt256.ofNat 15), (UInt256.ofNat 14), (UInt256.ofNat 1), (UInt256.ofNat 9), (UInt256.ofNat 12), (UInt256.ofNat 5), (UInt256.ofNat 1), (UInt256.ofNat 15), (UInt256.ofNat 14), (UInt256.ofNat 13), (UInt256.ofNat 4), (UInt256.ofNat 10), (UInt256.ofNat 0), (UInt256.ofNat 7), (UInt256.ofNat 6), (UInt256.ofNat 3), (UInt256.ofNat 9), (UInt256.ofNat 2), (UInt256.ofNat 8), (UInt256.ofNat 11), (UInt256.ofNat 13), (UInt256.ofNat 11), (UInt256.ofNat 7), (UInt256.ofNat 14), (UInt256.ofNat 12), (UInt256.ofNat 1), (UInt256.ofNat 3), (UInt256.ofNat 9), (UInt256.ofNat 5), (UInt256.ofNat 0), (UInt256.ofNat 15), (UInt256.ofNat 4), (UInt256.ofNat 8), (UInt256.ofNat 6), (UInt256.ofNat 2), (UInt256.ofNat 10), (UInt256.ofNat 6), (UInt256.ofNat 15), (UInt256.ofNat 14), (UInt256.ofNat 9), (UInt256.ofNat 11), (UInt256.ofNat 3), (UInt256.ofNat 0), (UInt256.ofNat 8), (UInt256.ofNat 12), (UInt256.ofNat 2), (UInt256.ofNat 13), (UInt256.ofNat 7), (UInt256.ofNat 1), (UInt256.ofNat 4), (UInt256.ofNat 10), (UInt256.ofNat 5), (UInt256.ofNat 10), (UInt256.ofNat 2), (UInt256.ofNat 8), (UInt256.ofNat 4), (UInt256.ofNat 7), (UInt256.ofNat 6), (UInt256.ofNat 1), (UInt256.ofNat 5), (UInt256.ofNat 15), (UInt256.ofNat 11), (UInt256.ofNat 9), (UInt256.ofNat 14), (UInt256.ofNat 3), (UInt256.ofNat 12), (UInt256.ofNat 13), (UInt256.ofNat 0)]
  return v160

def SIGMA_k5 : FeM (Vector UInt256 160) := do
  SIGMA_k6

def SIGMA_k4 : FeM (Vector UInt256 160) := do
  SIGMA_k5

def SIGMA_k3 : FeM (Vector UInt256 160) := do
  SIGMA_k4

def SIGMA_k2 : FeM (Vector UInt256 160) := do
  SIGMA_k3

def SIGMA_k1 : FeM (Vector UInt256 160) := do
  SIGMA_k2

/-- `lib::blake2f::SIGMA` -/
def SIGMA : FeM (Vector UInt256 160) := do
  SIGMA_k1

/-- `lib::blake2f::compress<MemBuffer>` -/
def compress_MemBuffer (input : MemBuffer) : FeM (Vector UInt256 8) := do
  let mut h : (Vector UInt256 8) := default
  let mut m : (Vector UInt256 16) := default
  let mut v : (Vector UInt256 16) := default
  let mut round : UInt256 := default
  let mut r4 : Range_Known_0_Known_8 := default
  let mut r5 : Range_Known_0_Known_16 := default
  let mut r6 : Range_Known_0_Known_8 := default
  let mut r7 : (Vector UInt256 16) := default
  let mut r8 : Range_Known_0_Known_8 := default
  -- core::assert(input.len() == INPUT_LEN && input.byte_at(212) <= 1)
  let v6 ← MemBuffer_len_1 input
  let v7 ← INPUT_LEN
  let join : (Vector UInt256 8) → (Vector UInt256 16) → (Vector UInt256 16) → UInt256 → Range_Known_0_Known_8 → Range_Known_0_Known_16 → Range_Known_0_Known_8 → (Vector UInt256 16) → Range_Known_0_Known_8 → Bool → FeM (Vector UInt256 8) := fun h m v round r4 r5 r6 r7 r8 v1 => do
    let mut h := h
    let mut m := m
    let mut v := v
    let mut round := round
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    -- core::assert(input.len() == INPUT_LEN && input.byte_at(212) <= 1)
    let _ ← «assert» v1
    -- var h: [u64; 8] = [0; 8]
    h := (Vector.replicate 8 (UInt256.ofNat 0))
    -- var m: [u64; 16] = [0; 16]
    m := (Vector.replicate 16 (UInt256.ofNat 0))
    -- for i in 0..8 { h[i] = input.lane_at(4 + i * 8) }
    r4 := (Range_Known_0_Known_8.mk () ())
    let v29 ← Range_start_0_8 r4
    let rec loop (h : (Vector UInt256 8)) (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) (round : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v2 : UInt256) : FeM (Vector UInt256 8) := do
      let mut h := h
      let mut m := m
      let mut v := v
      let mut round := round
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let mut r7 := r7
      let mut r8 := r8
      -- for i in 0..8 { h[i] = input.lane_at(4 + i * 8) }
      let v32 ← Range_has_0_8 r4 v2
      if v32 then
        let v34 ← Range_at_0_8 r4 v2
        let i_35 := v34
        -- h[i] = input.lane_at(4 + i * 8)
        let v38 ← checkedMul i_35 (UInt256.ofNat 8)
        let v39 ← checkedAdd (UInt256.ofNat 4) v38
        let v40 ← MemBuffer_lane_at input v39
        checkIndex i_35 8
        h := (h.set! i_35.toNat v40)
        -- for i in 0..8 { h[i] = input.lane_at(4 + i * 8) }
        let v43 ← Range_next_0_8 r4 v2
        loop h m v round r4 r5 r6 r7 r8 v43
      else
        -- for i in 0..16 { m[i] = input.lane_at(68 + i * 8) }
        r5 := (Range_Known_0_Known_16.mk () ())
        let v50 ← Range_start_0_16 r5
        let rec loop_1 (h : (Vector UInt256 8)) (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) (round : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v3 : UInt256) : FeM (Vector UInt256 8) := do
          let mut h := h
          let mut m := m
          let mut v := v
          let mut round := round
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut r7 := r7
          let mut r8 := r8
          -- for i in 0..16 { m[i] = input.lane_at(68 + i * 8) }
          let v53 ← Range_has_0_16 r5 v3
          if v53 then
            let v55 ← Range_at_0_16 r5 v3
            let i_56 := v55
            -- m[i] = input.lane_at(68 + i * 8)
            let v59 ← checkedMul i_56 (UInt256.ofNat 8)
            let v60 ← checkedAdd (UInt256.ofNat 68) v59
            let v61 ← MemBuffer_lane_at input v60
            checkIndex i_56 16
            m := (m.set! i_56.toNat v61)
            -- for i in 0..16 { m[i] = input.lane_at(68 + i * 8) }
            let v64 ← Range_next_0_16 r5 v3
            loop_1 h m v round r4 r5 r6 r7 r8 v64
          else
            -- var v: [u64; 16] = [0; 16]
            v := (Vector.replicate 16 (UInt256.ofNat 0))
            -- for i in 0..8 {
            r6 := (Range_Known_0_Known_8.mk () ())
            let v74 ← Range_start_0_8 r6
            let rec loop_2 (h : (Vector UInt256 8)) (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) (round : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v4 : UInt256) : FeM (Vector UInt256 8) := do
              let mut h := h
              let mut m := m
              let mut v := v
              let mut round := round
              let mut r4 := r4
              let mut r5 := r5
              let mut r6 := r6
              let mut r7 := r7
              let mut r8 := r8
              -- for i in 0..8 {
              let v77 ← Range_has_0_8 r6 v4
              if v77 then
                let v79 ← Range_at_0_8 r6 v4
                let i_80 := v79
                -- v[i] = h[i]
                checkIndex i_80 8
                checkIndex i_80 16
                v := (v.set! i_80.toNat h[i_80.toNat]!)
                -- v[i + 8] = IV[i]
                let v83 ← IV
                checkIndex i_80 8
                let v86 ← checkedAdd i_80 (UInt256.ofNat 8)
                checkIndex v86 16
                v := (v.set! v86.toNat v83[i_80.toNat]!)
                -- for i in 0..8 {
                let v89 ← Range_next_0_8 r6 v4
                loop_2 h m v round r4 r5 r6 r7 r8 v89
              else
                -- v[12] ^= input.lane_at(196)
                let v92 ← MemBuffer_lane_at input (UInt256.ofNat 196)
                let v93 := (UInt256.ofNat 12)
                checkIndex v93 16
                checkIndex v93 16
                v := (v.set! v93.toNat (v[v93.toNat]! ^^^ v92))
                -- v[13] ^= input.lane_at(204)
                let v98 ← MemBuffer_lane_at input (UInt256.ofNat 204)
                let v99 := (UInt256.ofNat 13)
                checkIndex v99 16
                checkIndex v99 16
                v := (v.set! v99.toNat (v[v99.toNat]! ^^^ v98))
                -- if input.byte_at(212) == 1 { v[14] = !v[14] }
                let v104 ← MemBuffer_byte_at input (UInt256.ofNat 212)
                let join_1 : (Vector UInt256 8) → (Vector UInt256 16) → (Vector UInt256 16) → UInt256 → Range_Known_0_Known_8 → Range_Known_0_Known_16 → Range_Known_0_Known_8 → (Vector UInt256 16) → Range_Known_0_Known_8 → FeM (Vector UInt256 8) := fun h m v round r4 r5 r6 r7 r8 => do
                  let mut h := h
                  let mut m := m
                  let mut v := v
                  let mut round := round
                  let mut r4 := r4
                  let mut r5 := r5
                  let mut r6 := r6
                  let mut r7 := r7
                  let mut r8 := r8
                  -- let total = rounds(input)
                  let total ← rounds_MemBuffer input
                  -- var round: u64 = 0
                  round := (UInt256.ofNat 0)
                  let rec loop_3 (h : (Vector UInt256 8)) (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) (round : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) : FeM (Vector UInt256 8) := do
                    let mut h := h
                    let mut m := m
                    let mut v := v
                    let mut round := round
                    let mut r4 := r4
                    let mut r5 := r5
                    let mut r6 := r6
                    let mut r7 := r7
                    let mut r8 := r8
                    -- while total - round >= 10 {
                    let v120 ← IntTy.sub (.u 64) total round
                    if decide ((UInt256.ofNat 10).toNat ≤ v120.toNat) then
                      -- ten_rounds(mut v, m)
                      r7 := m
                      let (v127, v127_0) ← ten_rounds v r7
                      v := v127_0
                      -- round += 10
                      let round_130 ← IntTy.add (.u 64) round (UInt256.ofNat 10)
                      round := round_130
                      loop_3 h m v round r4 r5 r6 r7 r8
                    else
                      let rec loop_4 (h : (Vector UInt256 8)) (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) (round : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) : FeM (Vector UInt256 8) := do
                        let mut h := h
                        let mut m := m
                        let mut v := v
                        let mut round := round
                        let mut r4 := r4
                        let mut r5 := r5
                        let mut r6 := r6
                        let mut r7 := r7
                        let mut r8 := r8
                        -- while round < total {
                        if decide (round.toNat < total.toNat) then
                          -- let s = ((round % 10) * 16).downcast_truncate()
                          let v139 ← checkedRem round (UInt256.ofNat 10)
                          let v141 ← IntTy.mul (.u 64) v139 (UInt256.ofNat 16)
                          let s ← downcast_truncate_u64_usize v141
                          -- mix(mut v, a: 0, b: 4, c: 8, d: 12, x: m[SIGMA[s] as usize], y: m[SIGMA[s + 1] as usize])
                          let v149 ← SIGMA
                          checkIndex s 160
                          let v150 := v149[s.toNat]!
                          checkIndex v150 16
                          let v153 ← SIGMA
                          let v155 ← checkedAdd s (UInt256.ofNat 1)
                          checkIndex v155 160
                          let v156 := v153[v155.toNat]!
                          checkIndex v156 16
                          let (v159, v159_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) m[v150.toNat]! m[v156.toNat]!
                          v := v159_0
                          -- mix(mut v, a: 1, b: 5, c: 9, d: 13, x: m[SIGMA[s + 2] as usize], y: m[SIGMA[s + 3] as usize])
                          let v165 ← SIGMA
                          let v167 ← checkedAdd s (UInt256.ofNat 2)
                          checkIndex v167 160
                          let v168 := v165[v167.toNat]!
                          checkIndex v168 16
                          let v171 ← SIGMA
                          let v173 ← checkedAdd s (UInt256.ofNat 3)
                          checkIndex v173 160
                          let v174 := v171[v173.toNat]!
                          checkIndex v174 16
                          let (v177, v177_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) m[v168.toNat]! m[v174.toNat]!
                          v := v177_0
                          -- mix(mut v, a: 2, b: 6, c: 10, d: 14, x: m[SIGMA[s + 4] as usize], y: m[SIGMA[s + 5] as usize])
                          let v183 ← SIGMA
                          let v185 ← checkedAdd s (UInt256.ofNat 4)
                          checkIndex v185 160
                          let v186 := v183[v185.toNat]!
                          checkIndex v186 16
                          let v189 ← SIGMA
                          let v191 ← checkedAdd s (UInt256.ofNat 5)
                          checkIndex v191 160
                          let v192 := v189[v191.toNat]!
                          checkIndex v192 16
                          let (v195, v195_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) m[v186.toNat]! m[v192.toNat]!
                          v := v195_0
                          -- mix(mut v, a: 3, b: 7, c: 11, d: 15, x: m[SIGMA[s + 6] as usize], y: m[SIGMA[s + 7] as usize])
                          let v201 ← SIGMA
                          let v203 ← checkedAdd s (UInt256.ofNat 6)
                          checkIndex v203 160
                          let v204 := v201[v203.toNat]!
                          checkIndex v204 16
                          let v207 ← SIGMA
                          let v209 ← checkedAdd s (UInt256.ofNat 7)
                          checkIndex v209 160
                          let v210 := v207[v209.toNat]!
                          checkIndex v210 16
                          let (v213, v213_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) m[v204.toNat]! m[v210.toNat]!
                          v := v213_0
                          -- mix(mut v, a: 0, b: 5, c: 10, d: 15, x: m[SIGMA[s + 8] as usize], y: m[SIGMA[s + 9] as usize])
                          let v219 ← SIGMA
                          let v221 ← checkedAdd s (UInt256.ofNat 8)
                          checkIndex v221 160
                          let v222 := v219[v221.toNat]!
                          checkIndex v222 16
                          let v225 ← SIGMA
                          let v227 ← checkedAdd s (UInt256.ofNat 9)
                          checkIndex v227 160
                          let v228 := v225[v227.toNat]!
                          checkIndex v228 16
                          let (v231, v231_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) m[v222.toNat]! m[v228.toNat]!
                          v := v231_0
                          -- mix(mut v, a: 1, b: 6, c: 11, d: 12, x: m[SIGMA[s + 10] as usize], y: m[SIGMA[s + 11] as usize])
                          let v237 ← SIGMA
                          let v239 ← checkedAdd s (UInt256.ofNat 10)
                          checkIndex v239 160
                          let v240 := v237[v239.toNat]!
                          checkIndex v240 16
                          let v243 ← SIGMA
                          let v245 ← checkedAdd s (UInt256.ofNat 11)
                          checkIndex v245 160
                          let v246 := v243[v245.toNat]!
                          checkIndex v246 16
                          let (v249, v249_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) m[v240.toNat]! m[v246.toNat]!
                          v := v249_0
                          -- mix(mut v, a: 2, b: 7, c: 8, d: 13, x: m[SIGMA[s + 12] as usize], y: m[SIGMA[s + 13] as usize])
                          let v255 ← SIGMA
                          let v257 ← checkedAdd s (UInt256.ofNat 12)
                          checkIndex v257 160
                          let v258 := v255[v257.toNat]!
                          checkIndex v258 16
                          let v261 ← SIGMA
                          let v263 ← checkedAdd s (UInt256.ofNat 13)
                          checkIndex v263 160
                          let v264 := v261[v263.toNat]!
                          checkIndex v264 16
                          let (v267, v267_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) m[v258.toNat]! m[v264.toNat]!
                          v := v267_0
                          -- mix(mut v, a: 3, b: 4, c: 9, d: 14, x: m[SIGMA[s + 14] as usize], y: m[SIGMA[s + 15] as usize])
                          let v273 ← SIGMA
                          let v275 ← checkedAdd s (UInt256.ofNat 14)
                          checkIndex v275 160
                          let v276 := v273[v275.toNat]!
                          checkIndex v276 16
                          let v279 ← SIGMA
                          let v281 ← checkedAdd s (UInt256.ofNat 15)
                          checkIndex v281 160
                          let v282 := v279[v281.toNat]!
                          checkIndex v282 16
                          let (v285, v285_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) m[v276.toNat]! m[v282.toNat]!
                          v := v285_0
                          -- round += 1
                          let round_288 ← IntTy.add (.u 64) round (UInt256.ofNat 1)
                          round := round_288
                          loop_4 h m v round r4 r5 r6 r7 r8
                        else
                          -- for i in 0..8 { h[i] ^= v[i] ^ v[i + 8] }
                          r8 := (Range_Known_0_Known_8.mk () ())
                          let v296 ← Range_start_0_8 r8
                          let rec loop_5 (h : (Vector UInt256 8)) (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) (round : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v5 : UInt256) : FeM (Vector UInt256 8) := do
                            let mut h := h
                            let mut m := m
                            let mut v := v
                            let mut round := round
                            let mut r4 := r4
                            let mut r5 := r5
                            let mut r6 := r6
                            let mut r7 := r7
                            let mut r8 := r8
                            -- for i in 0..8 { h[i] ^= v[i] ^ v[i + 8] }
                            let v299 ← Range_has_0_8 r8 v5
                            if v299 then
                              let v301 ← Range_at_0_8 r8 v5
                              let i_302 := v301
                              -- h[i] ^= v[i] ^ v[i + 8]
                              checkIndex i_302 16
                              let v305 ← checkedAdd i_302 (UInt256.ofNat 8)
                              checkIndex v305 16
                              checkIndex i_302 8
                              let v309 := h[i_302.toNat]! ^^^ (v[i_302.toNat]! ^^^ v[v305.toNat]!)
                              checkIndex i_302 8
                              h := (h.set! i_302.toNat v309)
                              -- for i in 0..8 { h[i] ^= v[i] ^ v[i + 8] }
                              let v312 ← Range_next_0_8 r8 v5
                              loop_5 h m v round r4 r5 r6 r7 r8 v312
                            else
                              return h
                          partial_fixpoint
                          loop_5 h m v round r4 r5 r6 r7 r8 v296
                      partial_fixpoint
                      loop_4 h m v round r4 r5 r6 r7 r8
                  partial_fixpoint
                  loop_3 h m v round r4 r5 r6 r7 r8
                if decide (v104 = (UInt256.ofNat 1)) then
                  -- v[14] = !v[14]
                  let v107 := (UInt256.ofNat 14)
                  checkIndex v107 16
                  let v110 := (UInt256.ofNat 14)
                  checkIndex v110 16
                  v := (v.set! v110.toNat (IntTy.not (.u 64) v[v107.toNat]!))
                  join_1 h m v round r4 r5 r6 r7 r8
                else
                  join_1 h m v round r4 r5 r6 r7 r8
            partial_fixpoint
            loop_2 h m v round r4 r5 r6 r7 r8 v74
        partial_fixpoint
        loop_1 h m v round r4 r5 r6 r7 r8 v50
    partial_fixpoint
    loop h m v round r4 r5 r6 r7 r8 v29
  let join_2 : (Vector UInt256 8) → (Vector UInt256 16) → (Vector UInt256 16) → UInt256 → Range_Known_0_Known_8 → Range_Known_0_Known_16 → Range_Known_0_Known_8 → (Vector UInt256 16) → Range_Known_0_Known_8 → FeM (Vector UInt256 8) := fun h m v round r4 r5 r6 r7 r8 => do
    let mut h := h
    let mut m := m
    let mut v := v
    let mut round := round
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    join h m v round r4 r5 r6 r7 r8 false
  if decide (v6 = v7) then
    let v10 ← MemBuffer_byte_at input (UInt256.ofNat 212)
    if decide (v10.toNat ≤ (UInt256.ofNat 1).toNat) then
      join h m v round r4 r5 r6 r7 r8 true
    else
      join_2 h m v round r4 r5 r6 r7 r8
  else
    join_2 h m v round r4 r5 r6 r7 r8

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Empty := do
  let mut input : MemBuffer := default
  let mut output : MemBuffer := default
  let mut r3 : Range_Known_0_Known_8 := default
  -- let input = input()
  let input_1 ← input_Evm_1
  input := input_1
  -- if input.len() != blake2f::INPUT_LEN || input[212] > 1 {
  let v4 ← MemBuffer_len input
  let v5 ← INPUT_LEN
  let join : MemBuffer → MemBuffer → Range_Known_0_Known_8 → FeM Empty := fun input output r3 => do
    let mut input := input
    let mut output := output
    let mut r3 := r3
    -- raw.revert(data: MemSpan::empty())
    let v46 ← MemSlice_empty_u8
    let v49 ← Evm_revert v46
    nomatch v49
  if decide (v4 ≠ v5) then
    join input output r3
  else
    let v9 ← MemBuffer_index_ref input (UInt256.ofNat 212)
    let v10 ← mloadInt (.u 8) v9
    if decide ((UInt256.ofNat 1).toNat < v10.toNat) then
      join input output r3
    else
      -- let state = blake2f::compress(input)
      let state ← compress_MemBuffer input
      -- var output = MemBuffer::new(64)
      let output_19 ← MemBuffer_new (UInt256.ofNat 64)
      output := output_19
      -- for i in 0..8 { output.write_le<u64>(offset: i * 8, value: state[i]) }
      r3 := (Range_Known_0_Known_8.mk () ())
      let v26 ← Range_start_0_8 r3
      let rec loop (input : MemBuffer) (output : MemBuffer) (r3 : Range_Known_0_Known_8) (v0 : UInt256) : FeM Empty := do
        let mut input := input
        let mut output := output
        let mut r3 := r3
        -- for i in 0..8 { output.write_le<u64>(offset: i * 8, value: state[i]) }
        let v29 ← Range_has_0_8 r3 v0
        if v29 then
          let v31 ← Range_at_0_8 r3 v0
          let i := v31
          -- output.write_le<u64>(offset: i * 8, value: state[i])
          let v34 ← checkedMul i (UInt256.ofNat 8)
          checkIndex i 8
          let (v37, v37_0) ← MemBuffer_write_le_u64 output v34 state[i.toNat]!
          output := v37_0
          -- for i in 0..8 { output.write_le<u64>(offset: i * 8, value: state[i]) }
          let v39 ← Range_next_0_8 r3 v0
          loop input output r3 v39
        else
          -- raw.return_data(data: output.span())
          let v42 ← MemBuffer_span_ref output
          let v45 ← Evm_return_data v42
          nomatch v45
      partial_fixpoint
      loop input output r3 v26

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  nonPayable
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `Blake2F`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
  else
    let selector ← calldataSelector
    recv_fallback_entry

/-- The deployment entry of contract `Blake2F`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


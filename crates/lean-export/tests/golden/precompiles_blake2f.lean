export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_index_ref MemBuffer_len MemBuffer_new MemBuffer_span_ref MemBuffer_write_le_u64 MemSlice_empty_u8 Range_Known_0_Known_16 Range_Known_0_Known_16.mk Range_Known_0_Known_8 Range_Known_0_Known_8.mk Range_at_0_16 Range_at_0_8 Range_has_0_16 Range_has_0_8 Range_next_0_16 Range_next_0_8 Range_start_0_16 Range_start_0_8 downcast_truncate_u64_usize downcast_unchecked_u256_usize u64_wrapping_add «assert»)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1)

/-- The `init` of contract `Blake2F`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_1 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let len ← Evm_calldatasize
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  let input_11 := input
  return input_11

/-- `lib::blake2f::INPUT_LEN` -/
def INPUT_LEN : FeM UInt256 := do
  return (UInt256.ofNat 213)

/-- `lib::blake2f::IV` -/
def IV : FeM (Vector UInt256 8) := do
  let v8 := #v[(UInt256.ofNat 7640891576956012808), (UInt256.ofNat 13503953896175478587), (UInt256.ofNat 4354685564936845355), (UInt256.ofNat 11912009170470909681), (UInt256.ofNat 5840696475078001361), (UInt256.ofNat 11170449401992604703), (UInt256.ofNat 2270897969802886507), (UInt256.ofNat 6620516959819538809)]
  return v8

/-- `lib::blake2f::rounds<MemBuffer>` -/
def rounds_MemBuffer (input : MemBuffer) : FeM UInt256 := do
  let v1 ← MemBuffer_len_1 input
  let v2 ← INPUT_LEN
  let v3 := decide (v1 = v2)
  let _ ← «assert» v3
  let v6 ← MemBuffer_byte_at input (UInt256.ofNat 0)
  let v7 := v6
  let v9 := IntTy.shl (.u 64) v7 (UInt256.ofNat 24)
  let v11 ← MemBuffer_byte_at input (UInt256.ofNat 1)
  let v12 := v11
  let v14 := IntTy.shl (.u 64) v12 (UInt256.ofNat 16)
  let v15 := v9 ||| v14
  let v17 ← MemBuffer_byte_at input (UInt256.ofNat 2)
  let v18 := v17
  let v20 := IntTy.shl (.u 64) v18 (UInt256.ofNat 8)
  let v21 := v15 ||| v20
  let v23 ← MemBuffer_byte_at input (UInt256.ofNat 3)
  let v24 := v23
  let v25 := v21 ||| v24
  return v25

/-- `lib::blake2f::rotate_right` -/
def rotate_right (value : UInt256) («by» : UInt256) : FeM UInt256 := do
  let v4 := value >>> «by»
  let v8 ← IntTy.sub (.u 64) (UInt256.ofNat 64) «by»
  let v9 := IntTy.shl (.u 64) value v8
  let v10 := v4 ||| v9
  return v10

def mix_k4 (b : UInt256) (v72 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex b 16
  v := (v.set! b.toNat v72)
  return ((), v)

def mix_k3 (b : UInt256) (c : UInt256) (d : UInt256) (v56 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex d 16
  v := (v.set! d.toNat v56)
  checkIndex d 16
  let v60 := v[d.toNat]!
  checkIndex c 16
  let v62 := v[c.toNat]!
  let v63 ← u64_wrapping_add v62 v60
  checkIndex c 16
  v := (v.set! c.toNat v63)
  checkIndex b 16
  let v67 := v[b.toNat]!
  checkIndex c 16
  let v69 := v[c.toNat]!
  let v70 := v67 ^^^ v69
  let v72 ← rotate_right v70 (UInt256.ofNat 63)
  mix_k4 b v72 v

def mix_k2 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (y : UInt256) (v38 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  v := (v.set! b.toNat v38)
  checkIndex b 16
  let v42 := v[b.toNat]!
  checkIndex a 16
  let v44 := v[a.toNat]!
  let v45 ← u64_wrapping_add v44 v42
  let v47 ← u64_wrapping_add v45 y
  checkIndex a 16
  v := (v.set! a.toNat v47)
  checkIndex d 16
  let v51 := v[d.toNat]!
  checkIndex a 16
  let v53 := v[a.toNat]!
  let v54 := v51 ^^^ v53
  let v56 ← rotate_right v54 (UInt256.ofNat 16)
  mix_k3 b c d v56 v

def mix_k1 (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (y : UInt256) (v22 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  v := (v.set! d.toNat v22)
  checkIndex d 16
  let v26 := v[d.toNat]!
  checkIndex c 16
  let v28 := v[c.toNat]!
  let v29 ← u64_wrapping_add v28 v26
  checkIndex c 16
  v := (v.set! c.toNat v29)
  checkIndex b 16
  let v33 := v[b.toNat]!
  checkIndex c 16
  let v35 := v[c.toNat]!
  let v36 := v33 ^^^ v35
  let v38 ← rotate_right v36 (UInt256.ofNat 24)
  checkIndex b 16
  mix_k2 a b c d y v38 v

/-- `lib::blake2f::mix` -/
def mix (v : (Vector UInt256 16)) (a : UInt256) (b : UInt256) (c : UInt256) (d : UInt256) (x : UInt256) (y : UInt256) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex b 16
  let v8 := v[b.toNat]!
  checkIndex a 16
  let v10 := v[a.toNat]!
  let v11 ← u64_wrapping_add v10 v8
  let v13 ← u64_wrapping_add v11 x
  checkIndex a 16
  v := (v.set! a.toNat v13)
  checkIndex d 16
  let v17 := v[d.toNat]!
  checkIndex a 16
  let v19 := v[a.toNat]!
  let v20 := v17 ^^^ v19
  let v22 ← rotate_right v20 (UInt256.ofNat 32)
  checkIndex d 16
  mix_k1 a b c d y v22 v

def ten_rounds_k36 (m : (Vector UInt256 16)) (v787 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v788 := m[v787.toNat]!
  let v789 := (UInt256.ofNat 12)
  checkIndex v789 16
  let v790 := m[v789.toNat]!
  let (v791, v791_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v788 v790
  v := v791_0
  let v797 := (UInt256.ofNat 13)
  checkIndex v797 16
  let v798 := m[v797.toNat]!
  let v799 := (UInt256.ofNat 0)
  checkIndex v799 16
  let v800 := m[v799.toNat]!
  let (v801, v801_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v798 v800
  v := v801_0
  return ((), v)

def ten_rounds_k35 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v767 := (UInt256.ofNat 15)
  checkIndex v767 16
  let v768 := m[v767.toNat]!
  let v769 := (UInt256.ofNat 11)
  checkIndex v769 16
  let v770 := m[v769.toNat]!
  let (v771, v771_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v768 v770
  v := v771_0
  let v777 := (UInt256.ofNat 9)
  checkIndex v777 16
  let v778 := m[v777.toNat]!
  let v779 := (UInt256.ofNat 14)
  checkIndex v779 16
  let v780 := m[v779.toNat]!
  let (v781, v781_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v778 v780
  v := v781_0
  let v787 := (UInt256.ofNat 3)
  checkIndex v787 16
  ten_rounds_k36 m v787 v

def ten_rounds_k34 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v747 := (UInt256.ofNat 7)
  checkIndex v747 16
  let v748 := m[v747.toNat]!
  let v749 := (UInt256.ofNat 6)
  checkIndex v749 16
  let v750 := m[v749.toNat]!
  let (v751, v751_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v748 v750
  v := v751_0
  let v757 := (UInt256.ofNat 1)
  checkIndex v757 16
  let v758 := m[v757.toNat]!
  let v759 := (UInt256.ofNat 5)
  checkIndex v759 16
  let v760 := m[v759.toNat]!
  let (v761, v761_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v758 v760
  v := v761_0
  ten_rounds_k35 m v

def ten_rounds_k33 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v727 := (UInt256.ofNat 10)
  checkIndex v727 16
  let v728 := m[v727.toNat]!
  let v729 := (UInt256.ofNat 2)
  checkIndex v729 16
  let v730 := m[v729.toNat]!
  let (v731, v731_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v728 v730
  v := v731_0
  let v737 := (UInt256.ofNat 8)
  checkIndex v737 16
  let v738 := m[v737.toNat]!
  let v739 := (UInt256.ofNat 4)
  checkIndex v739 16
  let v740 := m[v739.toNat]!
  let (v741, v741_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v738 v740
  v := v741_0
  ten_rounds_k34 m v

def ten_rounds_k32 (m : (Vector UInt256 16)) (v698 : UInt256) (v699 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v700 := m[v699.toNat]!
  let (v701, v701_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v698 v700
  v := v701_0
  let v707 := (UInt256.ofNat 1)
  checkIndex v707 16
  let v708 := m[v707.toNat]!
  let v709 := (UInt256.ofNat 4)
  checkIndex v709 16
  let v710 := m[v709.toNat]!
  let (v711, v711_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v708 v710
  v := v711_0
  let v717 := (UInt256.ofNat 10)
  checkIndex v717 16
  let v718 := m[v717.toNat]!
  let v719 := (UInt256.ofNat 5)
  checkIndex v719 16
  let v720 := m[v719.toNat]!
  let (v721, v721_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v718 v720
  v := v721_0
  ten_rounds_k33 m v

def ten_rounds_k31 (m : (Vector UInt256 16)) (v678 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v679 := (UInt256.ofNat 8)
  checkIndex v679 16
  let v680 := m[v679.toNat]!
  let (v681, v681_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v678 v680
  v := v681_0
  let v687 := (UInt256.ofNat 12)
  checkIndex v687 16
  let v688 := m[v687.toNat]!
  let v689 := (UInt256.ofNat 2)
  checkIndex v689 16
  let v690 := m[v689.toNat]!
  let (v691, v691_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v688 v690
  v := v691_0
  let v697 := (UInt256.ofNat 13)
  checkIndex v697 16
  let v698 := m[v697.toNat]!
  let v699 := (UInt256.ofNat 7)
  checkIndex v699 16
  ten_rounds_k32 m v698 v699 v

def ten_rounds_k30 (m : (Vector UInt256 16)) (v657 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v657 16
  let v658 := m[v657.toNat]!
  let v659 := (UInt256.ofNat 9)
  checkIndex v659 16
  let v660 := m[v659.toNat]!
  let (v661, v661_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v658 v660
  v := v661_0
  let v667 := (UInt256.ofNat 11)
  checkIndex v667 16
  let v668 := m[v667.toNat]!
  let v669 := (UInt256.ofNat 3)
  checkIndex v669 16
  let v670 := m[v669.toNat]!
  let (v671, v671_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v668 v670
  v := v671_0
  let v677 := (UInt256.ofNat 0)
  checkIndex v677 16
  let v678 := m[v677.toNat]!
  ten_rounds_k31 m v678 v

def ten_rounds_k29 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v637 := (UInt256.ofNat 2)
  checkIndex v637 16
  let v638 := m[v637.toNat]!
  let v639 := (UInt256.ofNat 10)
  checkIndex v639 16
  let v640 := m[v639.toNat]!
  let (v641, v641_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v638 v640
  v := v641_0
  let v647 := (UInt256.ofNat 6)
  checkIndex v647 16
  let v648 := m[v647.toNat]!
  let v649 := (UInt256.ofNat 15)
  checkIndex v649 16
  let v650 := m[v649.toNat]!
  let (v651, v651_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v648 v650
  v := v651_0
  let v657 := (UInt256.ofNat 14)
  ten_rounds_k30 m v657 v

def ten_rounds_k28 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v617 := (UInt256.ofNat 15)
  checkIndex v617 16
  let v618 := m[v617.toNat]!
  let v619 := (UInt256.ofNat 4)
  checkIndex v619 16
  let v620 := m[v619.toNat]!
  let (v621, v621_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v618 v620
  v := v621_0
  let v627 := (UInt256.ofNat 8)
  checkIndex v627 16
  let v628 := m[v627.toNat]!
  let v629 := (UInt256.ofNat 6)
  checkIndex v629 16
  let v630 := m[v629.toNat]!
  let (v631, v631_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v628 v630
  v := v631_0
  ten_rounds_k29 m v

def ten_rounds_k27 (m : (Vector UInt256 16)) (v588 : UInt256) (v590 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v591, v591_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v588 v590
  v := v591_0
  let v597 := (UInt256.ofNat 3)
  checkIndex v597 16
  let v598 := m[v597.toNat]!
  let v599 := (UInt256.ofNat 9)
  checkIndex v599 16
  let v600 := m[v599.toNat]!
  let (v601, v601_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v598 v600
  v := v601_0
  let v607 := (UInt256.ofNat 5)
  checkIndex v607 16
  let v608 := m[v607.toNat]!
  let v609 := (UInt256.ofNat 0)
  checkIndex v609 16
  let v610 := m[v609.toNat]!
  let (v611, v611_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v608 v610
  v := v611_0
  ten_rounds_k28 m v

def ten_rounds_k26 (m : (Vector UInt256 16)) (v568 : UInt256) (v569 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v569 16
  let v570 := m[v569.toNat]!
  let (v571, v571_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v568 v570
  v := v571_0
  let v577 := (UInt256.ofNat 7)
  checkIndex v577 16
  let v578 := m[v577.toNat]!
  let v579 := (UInt256.ofNat 14)
  checkIndex v579 16
  let v580 := m[v579.toNat]!
  let (v581, v581_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v578 v580
  v := v581_0
  let v587 := (UInt256.ofNat 12)
  checkIndex v587 16
  let v588 := m[v587.toNat]!
  let v589 := (UInt256.ofNat 1)
  checkIndex v589 16
  let v590 := m[v589.toNat]!
  ten_rounds_k27 m v588 v590 v

def ten_rounds_k25 (m : (Vector UInt256 16)) (v547 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v548 := m[v547.toNat]!
  let v549 := (UInt256.ofNat 2)
  checkIndex v549 16
  let v550 := m[v549.toNat]!
  let (v551, v551_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v548 v550
  v := v551_0
  let v557 := (UInt256.ofNat 8)
  checkIndex v557 16
  let v558 := m[v557.toNat]!
  let v559 := (UInt256.ofNat 11)
  checkIndex v559 16
  let v560 := m[v559.toNat]!
  let (v561, v561_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v558 v560
  v := v561_0
  let v567 := (UInt256.ofNat 13)
  checkIndex v567 16
  let v568 := m[v567.toNat]!
  let v569 := (UInt256.ofNat 11)
  ten_rounds_k26 m v568 v569 v

def ten_rounds_k24 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v527 := (UInt256.ofNat 0)
  checkIndex v527 16
  let v528 := m[v527.toNat]!
  let v529 := (UInt256.ofNat 7)
  checkIndex v529 16
  let v530 := m[v529.toNat]!
  let (v531, v531_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v528 v530
  v := v531_0
  let v537 := (UInt256.ofNat 6)
  checkIndex v537 16
  let v538 := m[v537.toNat]!
  let v539 := (UInt256.ofNat 3)
  checkIndex v539 16
  let v540 := m[v539.toNat]!
  let (v541, v541_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v538 v540
  v := v541_0
  let v547 := (UInt256.ofNat 9)
  checkIndex v547 16
  ten_rounds_k25 m v547 v

def ten_rounds_k23 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v507 := (UInt256.ofNat 14)
  checkIndex v507 16
  let v508 := m[v507.toNat]!
  let v509 := (UInt256.ofNat 13)
  checkIndex v509 16
  let v510 := m[v509.toNat]!
  let (v511, v511_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v508 v510
  v := v511_0
  let v517 := (UInt256.ofNat 4)
  checkIndex v517 16
  let v518 := m[v517.toNat]!
  let v519 := (UInt256.ofNat 10)
  checkIndex v519 16
  let v520 := m[v519.toNat]!
  let (v521, v521_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v518 v520
  v := v521_0
  ten_rounds_k24 m v

def ten_rounds_k22 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v487 := (UInt256.ofNat 12)
  checkIndex v487 16
  let v488 := m[v487.toNat]!
  let v489 := (UInt256.ofNat 5)
  checkIndex v489 16
  let v490 := m[v489.toNat]!
  let (v491, v491_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v488 v490
  v := v491_0
  let v497 := (UInt256.ofNat 1)
  checkIndex v497 16
  let v498 := m[v497.toNat]!
  let v499 := (UInt256.ofNat 15)
  checkIndex v499 16
  let v500 := m[v499.toNat]!
  let (v501, v501_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v498 v500
  v := v501_0
  ten_rounds_k23 m v

def ten_rounds_k21 (m : (Vector UInt256 16)) (v458 : UInt256) (v459 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v460 := m[v459.toNat]!
  let (v461, v461_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v458 v460
  v := v461_0
  let v467 := (UInt256.ofNat 15)
  checkIndex v467 16
  let v468 := m[v467.toNat]!
  let v469 := (UInt256.ofNat 14)
  checkIndex v469 16
  let v470 := m[v469.toNat]!
  let (v471, v471_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v468 v470
  v := v471_0
  let v477 := (UInt256.ofNat 1)
  checkIndex v477 16
  let v478 := m[v477.toNat]!
  let v479 := (UInt256.ofNat 9)
  checkIndex v479 16
  let v480 := m[v479.toNat]!
  let (v481, v481_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v478 v480
  v := v481_0
  ten_rounds_k22 m v

def ten_rounds_k20 (m : (Vector UInt256 16)) (v438 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v439 := (UInt256.ofNat 3)
  checkIndex v439 16
  let v440 := m[v439.toNat]!
  let (v441, v441_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v438 v440
  v := v441_0
  let v447 := (UInt256.ofNat 4)
  checkIndex v447 16
  let v448 := m[v447.toNat]!
  let v449 := (UInt256.ofNat 13)
  checkIndex v449 16
  let v450 := m[v449.toNat]!
  let (v451, v451_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v448 v450
  v := v451_0
  let v457 := (UInt256.ofNat 7)
  checkIndex v457 16
  let v458 := m[v457.toNat]!
  let v459 := (UInt256.ofNat 5)
  checkIndex v459 16
  ten_rounds_k21 m v458 v459 v

def ten_rounds_k19 (m : (Vector UInt256 16)) (v417 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v417 16
  let v418 := m[v417.toNat]!
  let v419 := (UInt256.ofNat 10)
  checkIndex v419 16
  let v420 := m[v419.toNat]!
  let (v421, v421_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v418 v420
  v := v421_0
  let v427 := (UInt256.ofNat 0)
  checkIndex v427 16
  let v428 := m[v427.toNat]!
  let v429 := (UInt256.ofNat 11)
  checkIndex v429 16
  let v430 := m[v429.toNat]!
  let (v431, v431_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v428 v430
  v := v431_0
  let v437 := (UInt256.ofNat 8)
  checkIndex v437 16
  let v438 := m[v437.toNat]!
  ten_rounds_k20 m v438 v

def ten_rounds_k18 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v397 := (UInt256.ofNat 3)
  checkIndex v397 16
  let v398 := m[v397.toNat]!
  let v399 := (UInt256.ofNat 13)
  checkIndex v399 16
  let v400 := m[v399.toNat]!
  let (v401, v401_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v398 v400
  v := v401_0
  let v407 := (UInt256.ofNat 2)
  checkIndex v407 16
  let v408 := m[v407.toNat]!
  let v409 := (UInt256.ofNat 12)
  checkIndex v409 16
  let v410 := m[v409.toNat]!
  let (v411, v411_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v408 v410
  v := v411_0
  let v417 := (UInt256.ofNat 6)
  ten_rounds_k19 m v417 v

def ten_rounds_k17 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v377 := (UInt256.ofNat 11)
  checkIndex v377 16
  let v378 := m[v377.toNat]!
  let v379 := (UInt256.ofNat 12)
  checkIndex v379 16
  let v380 := m[v379.toNat]!
  let (v381, v381_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v378 v380
  v := v381_0
  let v387 := (UInt256.ofNat 6)
  checkIndex v387 16
  let v388 := m[v387.toNat]!
  let v389 := (UInt256.ofNat 8)
  checkIndex v389 16
  let v390 := m[v389.toNat]!
  let (v391, v391_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v388 v390
  v := v391_0
  ten_rounds_k18 m v

def ten_rounds_k16 (m : (Vector UInt256 16)) (v348 : UInt256) (v350 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v351, v351_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v348 v350
  v := v351_0
  let v357 := (UInt256.ofNat 10)
  checkIndex v357 16
  let v358 := m[v357.toNat]!
  let v359 := (UInt256.ofNat 15)
  checkIndex v359 16
  let v360 := m[v359.toNat]!
  let (v361, v361_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v358 v360
  v := v361_0
  let v367 := (UInt256.ofNat 14)
  checkIndex v367 16
  let v368 := m[v367.toNat]!
  let v369 := (UInt256.ofNat 1)
  checkIndex v369 16
  let v370 := m[v369.toNat]!
  let (v371, v371_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v368 v370
  v := v371_0
  ten_rounds_k17 m v

def ten_rounds_k15 (m : (Vector UInt256 16)) (v328 : UInt256) (v329 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v329 16
  let v330 := m[v329.toNat]!
  let (v331, v331_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v328 v330
  v := v331_0
  let v337 := (UInt256.ofNat 5)
  checkIndex v337 16
  let v338 := m[v337.toNat]!
  let v339 := (UInt256.ofNat 7)
  checkIndex v339 16
  let v340 := m[v339.toNat]!
  let (v341, v341_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v338 v340
  v := v341_0
  let v347 := (UInt256.ofNat 2)
  checkIndex v347 16
  let v348 := m[v347.toNat]!
  let v349 := (UInt256.ofNat 4)
  checkIndex v349 16
  let v350 := m[v349.toNat]!
  ten_rounds_k16 m v348 v350 v

def ten_rounds_k14 (m : (Vector UInt256 16)) (v307 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v308 := m[v307.toNat]!
  let v309 := (UInt256.ofNat 0)
  checkIndex v309 16
  let v310 := m[v309.toNat]!
  let (v311, v311_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v308 v310
  v := v311_0
  let v317 := (UInt256.ofNat 15)
  checkIndex v317 16
  let v318 := m[v317.toNat]!
  let v319 := (UInt256.ofNat 8)
  checkIndex v319 16
  let v320 := m[v319.toNat]!
  let (v321, v321_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v318 v320
  v := v321_0
  let v327 := (UInt256.ofNat 9)
  checkIndex v327 16
  let v328 := m[v327.toNat]!
  let v329 := (UInt256.ofNat 0)
  ten_rounds_k15 m v328 v329 v

def ten_rounds_k13 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v287 := (UInt256.ofNat 2)
  checkIndex v287 16
  let v288 := m[v287.toNat]!
  let v289 := (UInt256.ofNat 6)
  checkIndex v289 16
  let v290 := m[v289.toNat]!
  let (v291, v291_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v288 v290
  v := v291_0
  let v297 := (UInt256.ofNat 5)
  checkIndex v297 16
  let v298 := m[v297.toNat]!
  let v299 := (UInt256.ofNat 10)
  checkIndex v299 16
  let v300 := m[v299.toNat]!
  let (v301, v301_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v298 v300
  v := v301_0
  let v307 := (UInt256.ofNat 4)
  checkIndex v307 16
  ten_rounds_k14 m v307 v

def ten_rounds_k12 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v267 := (UInt256.ofNat 13)
  checkIndex v267 16
  let v268 := m[v267.toNat]!
  let v269 := (UInt256.ofNat 12)
  checkIndex v269 16
  let v270 := m[v269.toNat]!
  let (v271, v271_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v268 v270
  v := v271_0
  let v277 := (UInt256.ofNat 11)
  checkIndex v277 16
  let v278 := m[v277.toNat]!
  let v279 := (UInt256.ofNat 14)
  checkIndex v279 16
  let v280 := m[v279.toNat]!
  let (v281, v281_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v278 v280
  v := v281_0
  ten_rounds_k13 m v

def ten_rounds_k11 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v247 := (UInt256.ofNat 7)
  checkIndex v247 16
  let v248 := m[v247.toNat]!
  let v249 := (UInt256.ofNat 9)
  checkIndex v249 16
  let v250 := m[v249.toNat]!
  let (v251, v251_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v248 v250
  v := v251_0
  let v257 := (UInt256.ofNat 3)
  checkIndex v257 16
  let v258 := m[v257.toNat]!
  let v259 := (UInt256.ofNat 1)
  checkIndex v259 16
  let v260 := m[v259.toNat]!
  let (v261, v261_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v258 v260
  v := v261_0
  ten_rounds_k12 m v

def ten_rounds_k10 (m : (Vector UInt256 16)) (v218 : UInt256) (v219 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v220 := m[v219.toNat]!
  let (v221, v221_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v218 v220
  v := v221_0
  let v227 := (UInt256.ofNat 7)
  checkIndex v227 16
  let v228 := m[v227.toNat]!
  let v229 := (UInt256.ofNat 1)
  checkIndex v229 16
  let v230 := m[v229.toNat]!
  let (v231, v231_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v228 v230
  v := v231_0
  let v237 := (UInt256.ofNat 9)
  checkIndex v237 16
  let v238 := m[v237.toNat]!
  let v239 := (UInt256.ofNat 4)
  checkIndex v239 16
  let v240 := m[v239.toNat]!
  let (v241, v241_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v238 v240
  v := v241_0
  ten_rounds_k11 m v

def ten_rounds_k9 (m : (Vector UInt256 16)) (v198 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v199 := (UInt256.ofNat 13)
  checkIndex v199 16
  let v200 := m[v199.toNat]!
  let (v201, v201_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v198 v200
  v := v201_0
  let v207 := (UInt256.ofNat 10)
  checkIndex v207 16
  let v208 := m[v207.toNat]!
  let v209 := (UInt256.ofNat 14)
  checkIndex v209 16
  let v210 := m[v209.toNat]!
  let (v211, v211_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v208 v210
  v := v211_0
  let v217 := (UInt256.ofNat 3)
  checkIndex v217 16
  let v218 := m[v217.toNat]!
  let v219 := (UInt256.ofNat 6)
  checkIndex v219 16
  ten_rounds_k10 m v218 v219 v

def ten_rounds_k8 (m : (Vector UInt256 16)) (v177 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v177 16
  let v178 := m[v177.toNat]!
  let v179 := (UInt256.ofNat 0)
  checkIndex v179 16
  let v180 := m[v179.toNat]!
  let (v181, v181_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v178 v180
  v := v181_0
  let v187 := (UInt256.ofNat 5)
  checkIndex v187 16
  let v188 := m[v187.toNat]!
  let v189 := (UInt256.ofNat 2)
  checkIndex v189 16
  let v190 := m[v189.toNat]!
  let (v191, v191_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v188 v190
  v := v191_0
  let v197 := (UInt256.ofNat 15)
  checkIndex v197 16
  let v198 := m[v197.toNat]!
  ten_rounds_k9 m v198 v

def ten_rounds_k7 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v157 := (UInt256.ofNat 5)
  checkIndex v157 16
  let v158 := m[v157.toNat]!
  let v159 := (UInt256.ofNat 3)
  checkIndex v159 16
  let v160 := m[v159.toNat]!
  let (v161, v161_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v158 v160
  v := v161_0
  let v167 := (UInt256.ofNat 11)
  checkIndex v167 16
  let v168 := m[v167.toNat]!
  let v169 := (UInt256.ofNat 8)
  checkIndex v169 16
  let v170 := m[v169.toNat]!
  let (v171, v171_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v168 v170
  v := v171_0
  let v177 := (UInt256.ofNat 12)
  ten_rounds_k8 m v177 v

def ten_rounds_k6 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v137 := (UInt256.ofNat 0)
  checkIndex v137 16
  let v138 := m[v137.toNat]!
  let v139 := (UInt256.ofNat 2)
  checkIndex v139 16
  let v140 := m[v139.toNat]!
  let (v141, v141_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v138 v140
  v := v141_0
  let v147 := (UInt256.ofNat 11)
  checkIndex v147 16
  let v148 := m[v147.toNat]!
  let v149 := (UInt256.ofNat 7)
  checkIndex v149 16
  let v150 := m[v149.toNat]!
  let (v151, v151_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v148 v150
  v := v151_0
  ten_rounds_k7 m v

def ten_rounds_k5 (m : (Vector UInt256 16)) (v108 : UInt256) (v110 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let (v111, v111_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v108 v110
  v := v111_0
  let v117 := (UInt256.ofNat 13)
  checkIndex v117 16
  let v118 := m[v117.toNat]!
  let v119 := (UInt256.ofNat 6)
  checkIndex v119 16
  let v120 := m[v119.toNat]!
  let (v121, v121_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v118 v120
  v := v121_0
  let v127 := (UInt256.ofNat 1)
  checkIndex v127 16
  let v128 := m[v127.toNat]!
  let v129 := (UInt256.ofNat 12)
  checkIndex v129 16
  let v130 := m[v129.toNat]!
  let (v131, v131_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v128 v130
  v := v131_0
  ten_rounds_k6 m v

def ten_rounds_k4 (m : (Vector UInt256 16)) (v88 : UInt256) (v89 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  checkIndex v89 16
  let v90 := m[v89.toNat]!
  let (v91, v91_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v88 v90
  v := v91_0
  let v97 := (UInt256.ofNat 4)
  checkIndex v97 16
  let v98 := m[v97.toNat]!
  let v99 := (UInt256.ofNat 8)
  checkIndex v99 16
  let v100 := m[v99.toNat]!
  let (v101, v101_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v98 v100
  v := v101_0
  let v107 := (UInt256.ofNat 9)
  checkIndex v107 16
  let v108 := m[v107.toNat]!
  let v109 := (UInt256.ofNat 15)
  checkIndex v109 16
  let v110 := m[v109.toNat]!
  ten_rounds_k5 m v108 v110 v

def ten_rounds_k3 (m : (Vector UInt256 16)) (v67 : UInt256) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v68 := m[v67.toNat]!
  let v69 := (UInt256.ofNat 13)
  checkIndex v69 16
  let v70 := m[v69.toNat]!
  let (v71, v71_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v68 v70
  v := v71_0
  let v77 := (UInt256.ofNat 14)
  checkIndex v77 16
  let v78 := m[v77.toNat]!
  let v79 := (UInt256.ofNat 15)
  checkIndex v79 16
  let v80 := m[v79.toNat]!
  let (v81, v81_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v78 v80
  v := v81_0
  let v87 := (UInt256.ofNat 14)
  checkIndex v87 16
  let v88 := m[v87.toNat]!
  let v89 := (UInt256.ofNat 10)
  ten_rounds_k4 m v88 v89 v

def ten_rounds_k2 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v47 := (UInt256.ofNat 8)
  checkIndex v47 16
  let v48 := m[v47.toNat]!
  let v49 := (UInt256.ofNat 9)
  checkIndex v49 16
  let v50 := m[v49.toNat]!
  let (v51, v51_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v48 v50
  v := v51_0
  let v57 := (UInt256.ofNat 10)
  checkIndex v57 16
  let v58 := m[v57.toNat]!
  let v59 := (UInt256.ofNat 11)
  checkIndex v59 16
  let v60 := m[v59.toNat]!
  let (v61, v61_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v58 v60
  v := v61_0
  let v67 := (UInt256.ofNat 12)
  checkIndex v67 16
  ten_rounds_k3 m v67 v

def ten_rounds_k1 (m : (Vector UInt256 16)) (v : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v27 := (UInt256.ofNat 4)
  checkIndex v27 16
  let v28 := m[v27.toNat]!
  let v29 := (UInt256.ofNat 5)
  checkIndex v29 16
  let v30 := m[v29.toNat]!
  let (v31, v31_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v28 v30
  v := v31_0
  let v37 := (UInt256.ofNat 6)
  checkIndex v37 16
  let v38 := m[v37.toNat]!
  let v39 := (UInt256.ofNat 7)
  checkIndex v39 16
  let v40 := m[v39.toNat]!
  let (v41, v41_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v38 v40
  v := v41_0
  ten_rounds_k2 m v

/-- `lib::blake2f::ten_rounds` -/
def ten_rounds (v : (Vector UInt256 16)) (m : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v := v
  let v7 := (UInt256.ofNat 0)
  checkIndex v7 16
  let v8 := m[v7.toNat]!
  let v9 := (UInt256.ofNat 1)
  checkIndex v9 16
  let v10 := m[v9.toNat]!
  let (v11, v11_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v8 v10
  v := v11_0
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  let v18 := m[v17.toNat]!
  let v19 := (UInt256.ofNat 3)
  checkIndex v19 16
  let v20 := m[v19.toNat]!
  let (v21, v21_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v18 v20
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
  let v6 ← MemBuffer_len_1 input
  let v7 ← INPUT_LEN
  let v8 := decide (v6 = v7)
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
    let _ ← «assert» v1
    let h_19 := Vector.replicate 8 (UInt256.ofNat 0)
    h := h_19
    let m_22 := Vector.replicate 16 (UInt256.ofNat 0)
    m := m_22
    let v27 := (Range_Known_0_Known_8.mk () ())
    r4 := v27
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
      let v32 ← Range_has_0_8 r4 v2
      if v32 then
        let v34 ← Range_at_0_8 r4 v2
        let i_35 := v34
        let v38 ← checkedMul i_35 (UInt256.ofNat 8)
        let v39 ← checkedAdd (UInt256.ofNat 4) v38
        let v40 ← MemBuffer_lane_at input v39
        checkIndex i_35 8
        h := (h.set! i_35.toNat v40)
        let v43 ← Range_next_0_8 r4 v2
        loop h m v round r4 r5 r6 r7 r8 v43
      else
        let v48 := (Range_Known_0_Known_16.mk () ())
        r5 := v48
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
          let v53 ← Range_has_0_16 r5 v3
          if v53 then
            let v55 ← Range_at_0_16 r5 v3
            let i_56 := v55
            let v59 ← checkedMul i_56 (UInt256.ofNat 8)
            let v60 ← checkedAdd (UInt256.ofNat 68) v59
            let v61 ← MemBuffer_lane_at input v60
            checkIndex i_56 16
            m := (m.set! i_56.toNat v61)
            let v64 ← Range_next_0_16 r5 v3
            loop_1 h m v round r4 r5 r6 r7 r8 v64
          else
            let v_67 := Vector.replicate 16 (UInt256.ofNat 0)
            v := v_67
            let v72 := (Range_Known_0_Known_8.mk () ())
            r6 := v72
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
              let v77 ← Range_has_0_8 r6 v4
              if v77 then
                let v79 ← Range_at_0_8 r6 v4
                let i_80 := v79
                checkIndex i_80 8
                let v81 := h[i_80.toNat]!
                checkIndex i_80 16
                v := (v.set! i_80.toNat v81)
                let v83 ← IV
                checkIndex i_80 8
                let v84 := v83[i_80.toNat]!
                let v86 ← checkedAdd i_80 (UInt256.ofNat 8)
                checkIndex v86 16
                v := (v.set! v86.toNat v84)
                let v89 ← Range_next_0_8 r6 v4
                loop_2 h m v round r4 r5 r6 r7 r8 v89
              else
                let v92 ← MemBuffer_lane_at input (UInt256.ofNat 196)
                let v93 := (UInt256.ofNat 12)
                checkIndex v93 16
                let v94 := v[v93.toNat]!
                let v95 := v94 ^^^ v92
                checkIndex v93 16
                v := (v.set! v93.toNat v95)
                let v98 ← MemBuffer_lane_at input (UInt256.ofNat 204)
                let v99 := (UInt256.ofNat 13)
                checkIndex v99 16
                let v100 := v[v99.toNat]!
                let v101 := v100 ^^^ v98
                checkIndex v99 16
                v := (v.set! v99.toNat v101)
                let v104 ← MemBuffer_byte_at input (UInt256.ofNat 212)
                let v106 := decide (v104 = (UInt256.ofNat 1))
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
                  let total ← rounds_MemBuffer input
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
                    let round_119 := round
                    let v120 ← IntTy.sub (.u 64) total round_119
                    let v122 := decide ((UInt256.ofNat 10).toNat ≤ v120.toNat)
                    if v122 then
                      let m_124 := m
                      r7 := m_124
                      let (v127, v127_0) ← ten_rounds v r7
                      v := v127_0
                      let round_129 := round
                      let round_130 ← IntTy.add (.u 64) round_129 (UInt256.ofNat 10)
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
                        let round_133 := round
                        let v135 := decide (round_133.toNat < total.toNat)
                        if v135 then
                          let round_136 := round
                          let v139 ← checkedRem round_136 (UInt256.ofNat 10)
                          let v141 ← IntTy.mul (.u 64) v139 (UInt256.ofNat 16)
                          let s ← downcast_truncate_u64_usize v141
                          let v149 ← SIGMA
                          checkIndex s 160
                          let v150 := v149[s.toNat]!
                          let v151 := v150
                          checkIndex v151 16
                          let v152 := m[v151.toNat]!
                          let v153 ← SIGMA
                          let v155 ← checkedAdd s (UInt256.ofNat 1)
                          checkIndex v155 160
                          let v156 := v153[v155.toNat]!
                          let v157 := v156
                          checkIndex v157 16
                          let v158 := m[v157.toNat]!
                          let (v159, v159_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 4) (UInt256.ofNat 8) (UInt256.ofNat 12) v152 v158
                          v := v159_0
                          let v165 ← SIGMA
                          let v167 ← checkedAdd s (UInt256.ofNat 2)
                          checkIndex v167 160
                          let v168 := v165[v167.toNat]!
                          let v169 := v168
                          checkIndex v169 16
                          let v170 := m[v169.toNat]!
                          let v171 ← SIGMA
                          let v173 ← checkedAdd s (UInt256.ofNat 3)
                          checkIndex v173 160
                          let v174 := v171[v173.toNat]!
                          let v175 := v174
                          checkIndex v175 16
                          let v176 := m[v175.toNat]!
                          let (v177, v177_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 5) (UInt256.ofNat 9) (UInt256.ofNat 13) v170 v176
                          v := v177_0
                          let v183 ← SIGMA
                          let v185 ← checkedAdd s (UInt256.ofNat 4)
                          checkIndex v185 160
                          let v186 := v183[v185.toNat]!
                          let v187 := v186
                          checkIndex v187 16
                          let v188 := m[v187.toNat]!
                          let v189 ← SIGMA
                          let v191 ← checkedAdd s (UInt256.ofNat 5)
                          checkIndex v191 160
                          let v192 := v189[v191.toNat]!
                          let v193 := v192
                          checkIndex v193 16
                          let v194 := m[v193.toNat]!
                          let (v195, v195_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 6) (UInt256.ofNat 10) (UInt256.ofNat 14) v188 v194
                          v := v195_0
                          let v201 ← SIGMA
                          let v203 ← checkedAdd s (UInt256.ofNat 6)
                          checkIndex v203 160
                          let v204 := v201[v203.toNat]!
                          let v205 := v204
                          checkIndex v205 16
                          let v206 := m[v205.toNat]!
                          let v207 ← SIGMA
                          let v209 ← checkedAdd s (UInt256.ofNat 7)
                          checkIndex v209 160
                          let v210 := v207[v209.toNat]!
                          let v211 := v210
                          checkIndex v211 16
                          let v212 := m[v211.toNat]!
                          let (v213, v213_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 7) (UInt256.ofNat 11) (UInt256.ofNat 15) v206 v212
                          v := v213_0
                          let v219 ← SIGMA
                          let v221 ← checkedAdd s (UInt256.ofNat 8)
                          checkIndex v221 160
                          let v222 := v219[v221.toNat]!
                          let v223 := v222
                          checkIndex v223 16
                          let v224 := m[v223.toNat]!
                          let v225 ← SIGMA
                          let v227 ← checkedAdd s (UInt256.ofNat 9)
                          checkIndex v227 160
                          let v228 := v225[v227.toNat]!
                          let v229 := v228
                          checkIndex v229 16
                          let v230 := m[v229.toNat]!
                          let (v231, v231_0) ← mix v (UInt256.ofNat 0) (UInt256.ofNat 5) (UInt256.ofNat 10) (UInt256.ofNat 15) v224 v230
                          v := v231_0
                          let v237 ← SIGMA
                          let v239 ← checkedAdd s (UInt256.ofNat 10)
                          checkIndex v239 160
                          let v240 := v237[v239.toNat]!
                          let v241 := v240
                          checkIndex v241 16
                          let v242 := m[v241.toNat]!
                          let v243 ← SIGMA
                          let v245 ← checkedAdd s (UInt256.ofNat 11)
                          checkIndex v245 160
                          let v246 := v243[v245.toNat]!
                          let v247 := v246
                          checkIndex v247 16
                          let v248 := m[v247.toNat]!
                          let (v249, v249_0) ← mix v (UInt256.ofNat 1) (UInt256.ofNat 6) (UInt256.ofNat 11) (UInt256.ofNat 12) v242 v248
                          v := v249_0
                          let v255 ← SIGMA
                          let v257 ← checkedAdd s (UInt256.ofNat 12)
                          checkIndex v257 160
                          let v258 := v255[v257.toNat]!
                          let v259 := v258
                          checkIndex v259 16
                          let v260 := m[v259.toNat]!
                          let v261 ← SIGMA
                          let v263 ← checkedAdd s (UInt256.ofNat 13)
                          checkIndex v263 160
                          let v264 := v261[v263.toNat]!
                          let v265 := v264
                          checkIndex v265 16
                          let v266 := m[v265.toNat]!
                          let (v267, v267_0) ← mix v (UInt256.ofNat 2) (UInt256.ofNat 7) (UInt256.ofNat 8) (UInt256.ofNat 13) v260 v266
                          v := v267_0
                          let v273 ← SIGMA
                          let v275 ← checkedAdd s (UInt256.ofNat 14)
                          checkIndex v275 160
                          let v276 := v273[v275.toNat]!
                          let v277 := v276
                          checkIndex v277 16
                          let v278 := m[v277.toNat]!
                          let v279 ← SIGMA
                          let v281 ← checkedAdd s (UInt256.ofNat 15)
                          checkIndex v281 160
                          let v282 := v279[v281.toNat]!
                          let v283 := v282
                          checkIndex v283 16
                          let v284 := m[v283.toNat]!
                          let (v285, v285_0) ← mix v (UInt256.ofNat 3) (UInt256.ofNat 4) (UInt256.ofNat 9) (UInt256.ofNat 14) v278 v284
                          v := v285_0
                          let round_287 := round
                          let round_288 ← IntTy.add (.u 64) round_287 (UInt256.ofNat 1)
                          round := round_288
                          loop_4 h m v round r4 r5 r6 r7 r8
                        else
                          let v294 := (Range_Known_0_Known_8.mk () ())
                          r8 := v294
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
                            let v299 ← Range_has_0_8 r8 v5
                            if v299 then
                              let v301 ← Range_at_0_8 r8 v5
                              let i_302 := v301
                              checkIndex i_302 16
                              let v303 := v[i_302.toNat]!
                              let v305 ← checkedAdd i_302 (UInt256.ofNat 8)
                              checkIndex v305 16
                              let v306 := v[v305.toNat]!
                              let v307 := v303 ^^^ v306
                              checkIndex i_302 8
                              let v308 := h[i_302.toNat]!
                              let v309 := v308 ^^^ v307
                              checkIndex i_302 8
                              h := (h.set! i_302.toNat v309)
                              let v312 ← Range_next_0_8 r8 v5
                              loop_5 h m v round r4 r5 r6 r7 r8 v312
                            else
                              let h_314 := h
                              return h_314
                          partial_fixpoint
                          loop_5 h m v round r4 r5 r6 r7 r8 v296
                      partial_fixpoint
                      loop_4 h m v round r4 r5 r6 r7 r8
                  partial_fixpoint
                  loop_3 h m v round r4 r5 r6 r7 r8
                if v106 then
                  let v107 := (UInt256.ofNat 14)
                  checkIndex v107 16
                  let v108 := v[v107.toNat]!
                  let v109 := IntTy.not (.u 64) v108
                  let v110 := (UInt256.ofNat 14)
                  checkIndex v110 16
                  v := (v.set! v110.toNat v109)
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
  if v8 then
    let v10 ← MemBuffer_byte_at input (UInt256.ofNat 212)
    let v12 := decide (v10.toNat ≤ (UInt256.ofNat 1).toNat)
    if v12 then
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
  let input_1 ← input_Evm_1
  input := input_1
  let v4 ← MemBuffer_len input
  let v5 ← INPUT_LEN
  let v6 := decide (v4 ≠ v5)
  let join : MemBuffer → MemBuffer → Range_Known_0_Known_8 → FeM Empty := fun input output r3 => do
    let mut input := input
    let mut output := output
    let mut r3 := r3
    let v46 ← MemSlice_empty_u8
    let v49 ← Evm_revert v46
    nomatch v49
  if v6 then
    join input output r3
  else
    let v9 ← MemBuffer_index_ref input (UInt256.ofNat 212)
    let v10 ← mloadInt (.u 8) v9
    let v12 := decide ((UInt256.ofNat 1).toNat < v10.toNat)
    if v12 then
      join input output r3
    else
      let state ← compress_MemBuffer input
      let output_19 ← MemBuffer_new (UInt256.ofNat 64)
      output := output_19
      let v24 := (Range_Known_0_Known_8.mk () ())
      r3 := v24
      let v26 ← Range_start_0_8 r3
      let rec loop (input : MemBuffer) (output : MemBuffer) (r3 : Range_Known_0_Known_8) (v0 : UInt256) : FeM Empty := do
        let mut input := input
        let mut output := output
        let mut r3 := r3
        let v29 ← Range_has_0_8 r3 v0
        if v29 then
          let v31 ← Range_at_0_8 r3 v0
          let i := v31
          let v34 ← checkedMul i (UInt256.ofNat 8)
          checkIndex i 8
          let v35 := state[i.toNat]!
          let (v37, v37_0) ← MemBuffer_write_le_u64 output v34 v35
          output := v37_0
          let v39 ← Range_next_0_8 r3 v0
          loop input output r3 v39
        else
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


export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_index_ref MemBuffer_len MemBuffer_new MemBuffer_span_ref MemBuffer_write_le_u64 MemSlice_empty_u8 Range_Known_0_Known_16 Range_Known_0_Known_16.mk Range_Known_0_Known_8 Range_Known_0_Known_8.mk Range_at_0_16 Range_at_0_8 Range_has_0_16 Range_has_0_8 Range_next_0_16 Range_next_0_8 Range_start_0_16 Range_start_0_8 assert downcast_truncate_u64_usize downcast_unchecked_u256_usize u64_wrapping_add)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1)

def init : FeM Unit := do
  return ()

def input_Evm_1 : FeM MemBuffer := do
  let mut r1 : MemBuffer := default
  let v1 ← Evm_calldatasize
  let v2 := v1
  let v3 ← downcast_unchecked_u256_usize v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 0)
  let (v10, v10_0) ← Evm_calldatacopy r1 v7 v8 v2
  r1 := v10_0
  let v11 := r1
  return v11

def INPUT_LEN : FeM UInt256 := do
  let v0 := (UInt256.ofNat 213)
  return v0

def IV : FeM (Vector UInt256 8) := do
  let v0 := (UInt256.ofNat 7640891576956012808)
  let v1 := (UInt256.ofNat 13503953896175478587)
  let v2 := (UInt256.ofNat 4354685564936845355)
  let v3 := (UInt256.ofNat 11912009170470909681)
  let v4 := (UInt256.ofNat 5840696475078001361)
  let v5 := (UInt256.ofNat 11170449401992604703)
  let v6 := (UInt256.ofNat 2270897969802886507)
  let v7 := (UInt256.ofNat 6620516959819538809)
  let v8 := #v[v0, v1, v2, v3, v4, v5, v6, v7]
  return v8

def rounds_MemBuffer (v0 : MemBuffer) : FeM UInt256 := do
  let v1 ← MemBuffer_len_1 v0
  let v2 ← INPUT_LEN
  let v3 := decide (v1 = v2)
  let v4 ← assert v3
  let v5 := (UInt256.ofNat 0)
  let v6 ← MemBuffer_byte_at v0 v5
  let v7 := v6
  let v8 := (UInt256.ofNat 24)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := (UInt256.ofNat 1)
  let v11 ← MemBuffer_byte_at v0 v10
  let v12 := v11
  let v13 := (UInt256.ofNat 16)
  let v14 := IntTy.shl (.u 64) v12 v13
  let v15 := v9 ||| v14
  let v16 := (UInt256.ofNat 2)
  let v17 ← MemBuffer_byte_at v0 v16
  let v18 := v17
  let v19 := (UInt256.ofNat 8)
  let v20 := IntTy.shl (.u 64) v18 v19
  let v21 := v15 ||| v20
  let v22 := (UInt256.ofNat 3)
  let v23 ← MemBuffer_byte_at v0 v22
  let v24 := v23
  let v25 := v21 ||| v24
  return v25

def rotate_right (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := v1
  let v4 := v2 >>> v3
  let v5 := v0
  let v6 := (UInt256.ofNat 64)
  let v7 := v1
  let v8 ← IntTy.sub (.u 64) v6 v7
  let v9 := IntTy.shl (.u 64) v5 v8
  let v10 := v4 ||| v9
  return v10

def mix_k4 (v2 : UInt256) (v72 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v73 := v2
  checkIndex v73 16
  v0 := (v0.set! v73.toNat v72)
  let v74 := ()
  return ((), v0)

def mix_k3 (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v56 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v57 := v4
  checkIndex v57 16
  v0 := (v0.set! v57.toNat v56)
  let v58 := ()
  let v59 := v4
  checkIndex v59 16
  let v60 := v0[v59.toNat]!
  let v61 := v3
  checkIndex v61 16
  let v62 := v0[v61.toNat]!
  let v63 ← u64_wrapping_add v62 v60
  let v64 := v3
  checkIndex v64 16
  v0 := (v0.set! v64.toNat v63)
  let v65 := ()
  let v66 := v2
  checkIndex v66 16
  let v67 := v0[v66.toNat]!
  let v68 := v3
  checkIndex v68 16
  let v69 := v0[v68.toNat]!
  let v70 := v67 ^^^ v69
  let v71 := (UInt256.ofNat 63)
  let v72 ← rotate_right v70 v71
  mix_k4 v2 v72 v0

def mix_k2 (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v6 : UInt256) (v38 : UInt256) (v39 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  v0 := (v0.set! v39.toNat v38)
  let v40 := ()
  let v41 := v2
  checkIndex v41 16
  let v42 := v0[v41.toNat]!
  let v43 := v1
  checkIndex v43 16
  let v44 := v0[v43.toNat]!
  let v45 ← u64_wrapping_add v44 v42
  let v46 := v6
  let v47 ← u64_wrapping_add v45 v46
  let v48 := v1
  checkIndex v48 16
  v0 := (v0.set! v48.toNat v47)
  let v49 := ()
  let v50 := v4
  checkIndex v50 16
  let v51 := v0[v50.toNat]!
  let v52 := v1
  checkIndex v52 16
  let v53 := v0[v52.toNat]!
  let v54 := v51 ^^^ v53
  let v55 := (UInt256.ofNat 16)
  let v56 ← rotate_right v54 v55
  mix_k3 v2 v3 v4 v56 v0

def mix_k1 (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v6 : UInt256) (v22 : UInt256) (v23 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  v0 := (v0.set! v23.toNat v22)
  let v24 := ()
  let v25 := v4
  checkIndex v25 16
  let v26 := v0[v25.toNat]!
  let v27 := v3
  checkIndex v27 16
  let v28 := v0[v27.toNat]!
  let v29 ← u64_wrapping_add v28 v26
  let v30 := v3
  checkIndex v30 16
  v0 := (v0.set! v30.toNat v29)
  let v31 := ()
  let v32 := v2
  checkIndex v32 16
  let v33 := v0[v32.toNat]!
  let v34 := v3
  checkIndex v34 16
  let v35 := v0[v34.toNat]!
  let v36 := v33 ^^^ v35
  let v37 := (UInt256.ofNat 24)
  let v38 ← rotate_right v36 v37
  let v39 := v2
  checkIndex v39 16
  mix_k2 v1 v2 v3 v4 v6 v38 v39 v0

def mix (v0 : (Vector UInt256 16)) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v7 := v2
  checkIndex v7 16
  let v8 := v0[v7.toNat]!
  let v9 := v1
  checkIndex v9 16
  let v10 := v0[v9.toNat]!
  let v11 ← u64_wrapping_add v10 v8
  let v12 := v5
  let v13 ← u64_wrapping_add v11 v12
  let v14 := v1
  checkIndex v14 16
  v0 := (v0.set! v14.toNat v13)
  let v15 := ()
  let v16 := v4
  checkIndex v16 16
  let v17 := v0[v16.toNat]!
  let v18 := v1
  checkIndex v18 16
  let v19 := v0[v18.toNat]!
  let v20 := v17 ^^^ v19
  let v21 := (UInt256.ofNat 32)
  let v22 ← rotate_right v20 v21
  let v23 := v4
  checkIndex v23 16
  mix_k1 v1 v2 v3 v4 v6 v22 v23 v0

def ten_rounds_k36 (v1 : (Vector UInt256 16)) (v783 : UInt256) (v784 : UInt256) (v785 : UInt256) (v786 : UInt256) (v787 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v788 := v1[v787.toNat]!
  let v789 := (UInt256.ofNat 12)
  checkIndex v789 16
  let v790 := v1[v789.toNat]!
  let (v791, v791_0) ← mix v0 v783 v784 v785 v786 v788 v790
  v0 := v791_0
  let v793 := (UInt256.ofNat 3)
  let v794 := (UInt256.ofNat 4)
  let v795 := (UInt256.ofNat 9)
  let v796 := (UInt256.ofNat 14)
  let v797 := (UInt256.ofNat 13)
  checkIndex v797 16
  let v798 := v1[v797.toNat]!
  let v799 := (UInt256.ofNat 0)
  checkIndex v799 16
  let v800 := v1[v799.toNat]!
  let (v801, v801_0) ← mix v0 v793 v794 v795 v796 v798 v800
  v0 := v801_0
  return ((), v0)

def ten_rounds_k35 (v1 : (Vector UInt256 16)) (v763 : UInt256) (v764 : UInt256) (v765 : UInt256) (v766 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v767 := (UInt256.ofNat 15)
  checkIndex v767 16
  let v768 := v1[v767.toNat]!
  let v769 := (UInt256.ofNat 11)
  checkIndex v769 16
  let v770 := v1[v769.toNat]!
  let (v771, v771_0) ← mix v0 v763 v764 v765 v766 v768 v770
  v0 := v771_0
  let v773 := (UInt256.ofNat 1)
  let v774 := (UInt256.ofNat 6)
  let v775 := (UInt256.ofNat 11)
  let v776 := (UInt256.ofNat 12)
  let v777 := (UInt256.ofNat 9)
  checkIndex v777 16
  let v778 := v1[v777.toNat]!
  let v779 := (UInt256.ofNat 14)
  checkIndex v779 16
  let v780 := v1[v779.toNat]!
  let (v781, v781_0) ← mix v0 v773 v774 v775 v776 v778 v780
  v0 := v781_0
  let v783 := (UInt256.ofNat 2)
  let v784 := (UInt256.ofNat 7)
  let v785 := (UInt256.ofNat 8)
  let v786 := (UInt256.ofNat 13)
  let v787 := (UInt256.ofNat 3)
  checkIndex v787 16
  ten_rounds_k36 v1 v783 v784 v785 v786 v787 v0

def ten_rounds_k34 (v1 : (Vector UInt256 16)) (v743 : UInt256) (v744 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v745 := (UInt256.ofNat 10)
  let v746 := (UInt256.ofNat 14)
  let v747 := (UInt256.ofNat 7)
  checkIndex v747 16
  let v748 := v1[v747.toNat]!
  let v749 := (UInt256.ofNat 6)
  checkIndex v749 16
  let v750 := v1[v749.toNat]!
  let (v751, v751_0) ← mix v0 v743 v744 v745 v746 v748 v750
  v0 := v751_0
  let v753 := (UInt256.ofNat 3)
  let v754 := (UInt256.ofNat 7)
  let v755 := (UInt256.ofNat 11)
  let v756 := (UInt256.ofNat 15)
  let v757 := (UInt256.ofNat 1)
  checkIndex v757 16
  let v758 := v1[v757.toNat]!
  let v759 := (UInt256.ofNat 5)
  checkIndex v759 16
  let v760 := v1[v759.toNat]!
  let (v761, v761_0) ← mix v0 v753 v754 v755 v756 v758 v760
  v0 := v761_0
  let v763 := (UInt256.ofNat 0)
  let v764 := (UInt256.ofNat 5)
  let v765 := (UInt256.ofNat 10)
  let v766 := (UInt256.ofNat 15)
  ten_rounds_k35 v1 v763 v764 v765 v766 v0

def ten_rounds_k33 (v1 : (Vector UInt256 16)) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v723 := (UInt256.ofNat 0)
  let v724 := (UInt256.ofNat 4)
  let v725 := (UInt256.ofNat 8)
  let v726 := (UInt256.ofNat 12)
  let v727 := (UInt256.ofNat 10)
  checkIndex v727 16
  let v728 := v1[v727.toNat]!
  let v729 := (UInt256.ofNat 2)
  checkIndex v729 16
  let v730 := v1[v729.toNat]!
  let (v731, v731_0) ← mix v0 v723 v724 v725 v726 v728 v730
  v0 := v731_0
  let v733 := (UInt256.ofNat 1)
  let v734 := (UInt256.ofNat 5)
  let v735 := (UInt256.ofNat 9)
  let v736 := (UInt256.ofNat 13)
  let v737 := (UInt256.ofNat 8)
  checkIndex v737 16
  let v738 := v1[v737.toNat]!
  let v739 := (UInt256.ofNat 4)
  checkIndex v739 16
  let v740 := v1[v739.toNat]!
  let (v741, v741_0) ← mix v0 v733 v734 v735 v736 v738 v740
  v0 := v741_0
  let v743 := (UInt256.ofNat 2)
  let v744 := (UInt256.ofNat 6)
  ten_rounds_k34 v1 v743 v744 v0

def ten_rounds_k32 (v1 : (Vector UInt256 16)) (v693 : UInt256) (v694 : UInt256) (v695 : UInt256) (v696 : UInt256) (v698 : UInt256) (v699 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v700 := v1[v699.toNat]!
  let (v701, v701_0) ← mix v0 v693 v694 v695 v696 v698 v700
  v0 := v701_0
  let v703 := (UInt256.ofNat 2)
  let v704 := (UInt256.ofNat 7)
  let v705 := (UInt256.ofNat 8)
  let v706 := (UInt256.ofNat 13)
  let v707 := (UInt256.ofNat 1)
  checkIndex v707 16
  let v708 := v1[v707.toNat]!
  let v709 := (UInt256.ofNat 4)
  checkIndex v709 16
  let v710 := v1[v709.toNat]!
  let (v711, v711_0) ← mix v0 v703 v704 v705 v706 v708 v710
  v0 := v711_0
  let v713 := (UInt256.ofNat 3)
  let v714 := (UInt256.ofNat 4)
  let v715 := (UInt256.ofNat 9)
  let v716 := (UInt256.ofNat 14)
  let v717 := (UInt256.ofNat 10)
  checkIndex v717 16
  let v718 := v1[v717.toNat]!
  let v719 := (UInt256.ofNat 5)
  checkIndex v719 16
  let v720 := v1[v719.toNat]!
  let (v721, v721_0) ← mix v0 v713 v714 v715 v716 v718 v720
  v0 := v721_0
  ten_rounds_k33 v1 v0

def ten_rounds_k31 (v1 : (Vector UInt256 16)) (v673 : UInt256) (v674 : UInt256) (v675 : UInt256) (v676 : UInt256) (v678 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v679 := (UInt256.ofNat 8)
  checkIndex v679 16
  let v680 := v1[v679.toNat]!
  let (v681, v681_0) ← mix v0 v673 v674 v675 v676 v678 v680
  v0 := v681_0
  let v683 := (UInt256.ofNat 0)
  let v684 := (UInt256.ofNat 5)
  let v685 := (UInt256.ofNat 10)
  let v686 := (UInt256.ofNat 15)
  let v687 := (UInt256.ofNat 12)
  checkIndex v687 16
  let v688 := v1[v687.toNat]!
  let v689 := (UInt256.ofNat 2)
  checkIndex v689 16
  let v690 := v1[v689.toNat]!
  let (v691, v691_0) ← mix v0 v683 v684 v685 v686 v688 v690
  v0 := v691_0
  let v693 := (UInt256.ofNat 1)
  let v694 := (UInt256.ofNat 6)
  let v695 := (UInt256.ofNat 11)
  let v696 := (UInt256.ofNat 12)
  let v697 := (UInt256.ofNat 13)
  checkIndex v697 16
  let v698 := v1[v697.toNat]!
  let v699 := (UInt256.ofNat 7)
  checkIndex v699 16
  ten_rounds_k32 v1 v693 v694 v695 v696 v698 v699 v0

def ten_rounds_k30 (v1 : (Vector UInt256 16)) (v653 : UInt256) (v654 : UInt256) (v655 : UInt256) (v656 : UInt256) (v657 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  checkIndex v657 16
  let v658 := v1[v657.toNat]!
  let v659 := (UInt256.ofNat 9)
  checkIndex v659 16
  let v660 := v1[v659.toNat]!
  let (v661, v661_0) ← mix v0 v653 v654 v655 v656 v658 v660
  v0 := v661_0
  let v663 := (UInt256.ofNat 2)
  let v664 := (UInt256.ofNat 6)
  let v665 := (UInt256.ofNat 10)
  let v666 := (UInt256.ofNat 14)
  let v667 := (UInt256.ofNat 11)
  checkIndex v667 16
  let v668 := v1[v667.toNat]!
  let v669 := (UInt256.ofNat 3)
  checkIndex v669 16
  let v670 := v1[v669.toNat]!
  let (v671, v671_0) ← mix v0 v663 v664 v665 v666 v668 v670
  v0 := v671_0
  let v673 := (UInt256.ofNat 3)
  let v674 := (UInt256.ofNat 7)
  let v675 := (UInt256.ofNat 11)
  let v676 := (UInt256.ofNat 15)
  let v677 := (UInt256.ofNat 0)
  checkIndex v677 16
  let v678 := v1[v677.toNat]!
  ten_rounds_k31 v1 v673 v674 v675 v676 v678 v0

def ten_rounds_k29 (v1 : (Vector UInt256 16)) (v633 : UInt256) (v634 : UInt256) (v635 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v636 := (UInt256.ofNat 14)
  let v637 := (UInt256.ofNat 2)
  checkIndex v637 16
  let v638 := v1[v637.toNat]!
  let v639 := (UInt256.ofNat 10)
  checkIndex v639 16
  let v640 := v1[v639.toNat]!
  let (v641, v641_0) ← mix v0 v633 v634 v635 v636 v638 v640
  v0 := v641_0
  let v643 := (UInt256.ofNat 0)
  let v644 := (UInt256.ofNat 4)
  let v645 := (UInt256.ofNat 8)
  let v646 := (UInt256.ofNat 12)
  let v647 := (UInt256.ofNat 6)
  checkIndex v647 16
  let v648 := v1[v647.toNat]!
  let v649 := (UInt256.ofNat 15)
  checkIndex v649 16
  let v650 := v1[v649.toNat]!
  let (v651, v651_0) ← mix v0 v643 v644 v645 v646 v648 v650
  v0 := v651_0
  let v653 := (UInt256.ofNat 1)
  let v654 := (UInt256.ofNat 5)
  let v655 := (UInt256.ofNat 9)
  let v656 := (UInt256.ofNat 13)
  let v657 := (UInt256.ofNat 14)
  ten_rounds_k30 v1 v653 v654 v655 v656 v657 v0

def ten_rounds_k28 (v1 : (Vector UInt256 16)) (v613 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v614 := (UInt256.ofNat 6)
  let v615 := (UInt256.ofNat 11)
  let v616 := (UInt256.ofNat 12)
  let v617 := (UInt256.ofNat 15)
  checkIndex v617 16
  let v618 := v1[v617.toNat]!
  let v619 := (UInt256.ofNat 4)
  checkIndex v619 16
  let v620 := v1[v619.toNat]!
  let (v621, v621_0) ← mix v0 v613 v614 v615 v616 v618 v620
  v0 := v621_0
  let v623 := (UInt256.ofNat 2)
  let v624 := (UInt256.ofNat 7)
  let v625 := (UInt256.ofNat 8)
  let v626 := (UInt256.ofNat 13)
  let v627 := (UInt256.ofNat 8)
  checkIndex v627 16
  let v628 := v1[v627.toNat]!
  let v629 := (UInt256.ofNat 6)
  checkIndex v629 16
  let v630 := v1[v629.toNat]!
  let (v631, v631_0) ← mix v0 v623 v624 v625 v626 v628 v630
  v0 := v631_0
  let v633 := (UInt256.ofNat 3)
  let v634 := (UInt256.ofNat 4)
  let v635 := (UInt256.ofNat 9)
  ten_rounds_k29 v1 v633 v634 v635 v0

def ten_rounds_k27 (v1 : (Vector UInt256 16)) (v583 : UInt256) (v584 : UInt256) (v585 : UInt256) (v586 : UInt256) (v588 : UInt256) (v590 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let (v591, v591_0) ← mix v0 v583 v584 v585 v586 v588 v590
  v0 := v591_0
  let v593 := (UInt256.ofNat 3)
  let v594 := (UInt256.ofNat 7)
  let v595 := (UInt256.ofNat 11)
  let v596 := (UInt256.ofNat 15)
  let v597 := (UInt256.ofNat 3)
  checkIndex v597 16
  let v598 := v1[v597.toNat]!
  let v599 := (UInt256.ofNat 9)
  checkIndex v599 16
  let v600 := v1[v599.toNat]!
  let (v601, v601_0) ← mix v0 v593 v594 v595 v596 v598 v600
  v0 := v601_0
  let v603 := (UInt256.ofNat 0)
  let v604 := (UInt256.ofNat 5)
  let v605 := (UInt256.ofNat 10)
  let v606 := (UInt256.ofNat 15)
  let v607 := (UInt256.ofNat 5)
  checkIndex v607 16
  let v608 := v1[v607.toNat]!
  let v609 := (UInt256.ofNat 0)
  checkIndex v609 16
  let v610 := v1[v609.toNat]!
  let (v611, v611_0) ← mix v0 v603 v604 v605 v606 v608 v610
  v0 := v611_0
  let v613 := (UInt256.ofNat 1)
  ten_rounds_k28 v1 v613 v0

def ten_rounds_k26 (v1 : (Vector UInt256 16)) (v563 : UInt256) (v564 : UInt256) (v565 : UInt256) (v566 : UInt256) (v568 : UInt256) (v569 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  checkIndex v569 16
  let v570 := v1[v569.toNat]!
  let (v571, v571_0) ← mix v0 v563 v564 v565 v566 v568 v570
  v0 := v571_0
  let v573 := (UInt256.ofNat 1)
  let v574 := (UInt256.ofNat 5)
  let v575 := (UInt256.ofNat 9)
  let v576 := (UInt256.ofNat 13)
  let v577 := (UInt256.ofNat 7)
  checkIndex v577 16
  let v578 := v1[v577.toNat]!
  let v579 := (UInt256.ofNat 14)
  checkIndex v579 16
  let v580 := v1[v579.toNat]!
  let (v581, v581_0) ← mix v0 v573 v574 v575 v576 v578 v580
  v0 := v581_0
  let v583 := (UInt256.ofNat 2)
  let v584 := (UInt256.ofNat 6)
  let v585 := (UInt256.ofNat 10)
  let v586 := (UInt256.ofNat 14)
  let v587 := (UInt256.ofNat 12)
  checkIndex v587 16
  let v588 := v1[v587.toNat]!
  let v589 := (UInt256.ofNat 1)
  checkIndex v589 16
  let v590 := v1[v589.toNat]!
  ten_rounds_k27 v1 v583 v584 v585 v586 v588 v590 v0

def ten_rounds_k25 (v1 : (Vector UInt256 16)) (v543 : UInt256) (v544 : UInt256) (v545 : UInt256) (v546 : UInt256) (v547 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v548 := v1[v547.toNat]!
  let v549 := (UInt256.ofNat 2)
  checkIndex v549 16
  let v550 := v1[v549.toNat]!
  let (v551, v551_0) ← mix v0 v543 v544 v545 v546 v548 v550
  v0 := v551_0
  let v553 := (UInt256.ofNat 3)
  let v554 := (UInt256.ofNat 4)
  let v555 := (UInt256.ofNat 9)
  let v556 := (UInt256.ofNat 14)
  let v557 := (UInt256.ofNat 8)
  checkIndex v557 16
  let v558 := v1[v557.toNat]!
  let v559 := (UInt256.ofNat 11)
  checkIndex v559 16
  let v560 := v1[v559.toNat]!
  let (v561, v561_0) ← mix v0 v553 v554 v555 v556 v558 v560
  v0 := v561_0
  let v563 := (UInt256.ofNat 0)
  let v564 := (UInt256.ofNat 4)
  let v565 := (UInt256.ofNat 8)
  let v566 := (UInt256.ofNat 12)
  let v567 := (UInt256.ofNat 13)
  checkIndex v567 16
  let v568 := v1[v567.toNat]!
  let v569 := (UInt256.ofNat 11)
  ten_rounds_k26 v1 v563 v564 v565 v566 v568 v569 v0

def ten_rounds_k24 (v1 : (Vector UInt256 16)) (v523 : UInt256) (v524 : UInt256) (v525 : UInt256) (v526 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v527 := (UInt256.ofNat 0)
  checkIndex v527 16
  let v528 := v1[v527.toNat]!
  let v529 := (UInt256.ofNat 7)
  checkIndex v529 16
  let v530 := v1[v529.toNat]!
  let (v531, v531_0) ← mix v0 v523 v524 v525 v526 v528 v530
  v0 := v531_0
  let v533 := (UInt256.ofNat 1)
  let v534 := (UInt256.ofNat 6)
  let v535 := (UInt256.ofNat 11)
  let v536 := (UInt256.ofNat 12)
  let v537 := (UInt256.ofNat 6)
  checkIndex v537 16
  let v538 := v1[v537.toNat]!
  let v539 := (UInt256.ofNat 3)
  checkIndex v539 16
  let v540 := v1[v539.toNat]!
  let (v541, v541_0) ← mix v0 v533 v534 v535 v536 v538 v540
  v0 := v541_0
  let v543 := (UInt256.ofNat 2)
  let v544 := (UInt256.ofNat 7)
  let v545 := (UInt256.ofNat 8)
  let v546 := (UInt256.ofNat 13)
  let v547 := (UInt256.ofNat 9)
  checkIndex v547 16
  ten_rounds_k25 v1 v543 v544 v545 v546 v547 v0

def ten_rounds_k23 (v1 : (Vector UInt256 16)) (v503 : UInt256) (v504 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v505 := (UInt256.ofNat 10)
  let v506 := (UInt256.ofNat 14)
  let v507 := (UInt256.ofNat 14)
  checkIndex v507 16
  let v508 := v1[v507.toNat]!
  let v509 := (UInt256.ofNat 13)
  checkIndex v509 16
  let v510 := v1[v509.toNat]!
  let (v511, v511_0) ← mix v0 v503 v504 v505 v506 v508 v510
  v0 := v511_0
  let v513 := (UInt256.ofNat 3)
  let v514 := (UInt256.ofNat 7)
  let v515 := (UInt256.ofNat 11)
  let v516 := (UInt256.ofNat 15)
  let v517 := (UInt256.ofNat 4)
  checkIndex v517 16
  let v518 := v1[v517.toNat]!
  let v519 := (UInt256.ofNat 10)
  checkIndex v519 16
  let v520 := v1[v519.toNat]!
  let (v521, v521_0) ← mix v0 v513 v514 v515 v516 v518 v520
  v0 := v521_0
  let v523 := (UInt256.ofNat 0)
  let v524 := (UInt256.ofNat 5)
  let v525 := (UInt256.ofNat 10)
  let v526 := (UInt256.ofNat 15)
  ten_rounds_k24 v1 v523 v524 v525 v526 v0

def ten_rounds_k22 (v1 : (Vector UInt256 16)) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v483 := (UInt256.ofNat 0)
  let v484 := (UInt256.ofNat 4)
  let v485 := (UInt256.ofNat 8)
  let v486 := (UInt256.ofNat 12)
  let v487 := (UInt256.ofNat 12)
  checkIndex v487 16
  let v488 := v1[v487.toNat]!
  let v489 := (UInt256.ofNat 5)
  checkIndex v489 16
  let v490 := v1[v489.toNat]!
  let (v491, v491_0) ← mix v0 v483 v484 v485 v486 v488 v490
  v0 := v491_0
  let v493 := (UInt256.ofNat 1)
  let v494 := (UInt256.ofNat 5)
  let v495 := (UInt256.ofNat 9)
  let v496 := (UInt256.ofNat 13)
  let v497 := (UInt256.ofNat 1)
  checkIndex v497 16
  let v498 := v1[v497.toNat]!
  let v499 := (UInt256.ofNat 15)
  checkIndex v499 16
  let v500 := v1[v499.toNat]!
  let (v501, v501_0) ← mix v0 v493 v494 v495 v496 v498 v500
  v0 := v501_0
  let v503 := (UInt256.ofNat 2)
  let v504 := (UInt256.ofNat 6)
  ten_rounds_k23 v1 v503 v504 v0

def ten_rounds_k21 (v1 : (Vector UInt256 16)) (v453 : UInt256) (v454 : UInt256) (v455 : UInt256) (v456 : UInt256) (v458 : UInt256) (v459 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v460 := v1[v459.toNat]!
  let (v461, v461_0) ← mix v0 v453 v454 v455 v456 v458 v460
  v0 := v461_0
  let v463 := (UInt256.ofNat 2)
  let v464 := (UInt256.ofNat 7)
  let v465 := (UInt256.ofNat 8)
  let v466 := (UInt256.ofNat 13)
  let v467 := (UInt256.ofNat 15)
  checkIndex v467 16
  let v468 := v1[v467.toNat]!
  let v469 := (UInt256.ofNat 14)
  checkIndex v469 16
  let v470 := v1[v469.toNat]!
  let (v471, v471_0) ← mix v0 v463 v464 v465 v466 v468 v470
  v0 := v471_0
  let v473 := (UInt256.ofNat 3)
  let v474 := (UInt256.ofNat 4)
  let v475 := (UInt256.ofNat 9)
  let v476 := (UInt256.ofNat 14)
  let v477 := (UInt256.ofNat 1)
  checkIndex v477 16
  let v478 := v1[v477.toNat]!
  let v479 := (UInt256.ofNat 9)
  checkIndex v479 16
  let v480 := v1[v479.toNat]!
  let (v481, v481_0) ← mix v0 v473 v474 v475 v476 v478 v480
  v0 := v481_0
  ten_rounds_k22 v1 v0

def ten_rounds_k20 (v1 : (Vector UInt256 16)) (v433 : UInt256) (v434 : UInt256) (v435 : UInt256) (v436 : UInt256) (v438 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v439 := (UInt256.ofNat 3)
  checkIndex v439 16
  let v440 := v1[v439.toNat]!
  let (v441, v441_0) ← mix v0 v433 v434 v435 v436 v438 v440
  v0 := v441_0
  let v443 := (UInt256.ofNat 0)
  let v444 := (UInt256.ofNat 5)
  let v445 := (UInt256.ofNat 10)
  let v446 := (UInt256.ofNat 15)
  let v447 := (UInt256.ofNat 4)
  checkIndex v447 16
  let v448 := v1[v447.toNat]!
  let v449 := (UInt256.ofNat 13)
  checkIndex v449 16
  let v450 := v1[v449.toNat]!
  let (v451, v451_0) ← mix v0 v443 v444 v445 v446 v448 v450
  v0 := v451_0
  let v453 := (UInt256.ofNat 1)
  let v454 := (UInt256.ofNat 6)
  let v455 := (UInt256.ofNat 11)
  let v456 := (UInt256.ofNat 12)
  let v457 := (UInt256.ofNat 7)
  checkIndex v457 16
  let v458 := v1[v457.toNat]!
  let v459 := (UInt256.ofNat 5)
  checkIndex v459 16
  ten_rounds_k21 v1 v453 v454 v455 v456 v458 v459 v0

def ten_rounds_k19 (v1 : (Vector UInt256 16)) (v413 : UInt256) (v414 : UInt256) (v415 : UInt256) (v416 : UInt256) (v417 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  checkIndex v417 16
  let v418 := v1[v417.toNat]!
  let v419 := (UInt256.ofNat 10)
  checkIndex v419 16
  let v420 := v1[v419.toNat]!
  let (v421, v421_0) ← mix v0 v413 v414 v415 v416 v418 v420
  v0 := v421_0
  let v423 := (UInt256.ofNat 2)
  let v424 := (UInt256.ofNat 6)
  let v425 := (UInt256.ofNat 10)
  let v426 := (UInt256.ofNat 14)
  let v427 := (UInt256.ofNat 0)
  checkIndex v427 16
  let v428 := v1[v427.toNat]!
  let v429 := (UInt256.ofNat 11)
  checkIndex v429 16
  let v430 := v1[v429.toNat]!
  let (v431, v431_0) ← mix v0 v423 v424 v425 v426 v428 v430
  v0 := v431_0
  let v433 := (UInt256.ofNat 3)
  let v434 := (UInt256.ofNat 7)
  let v435 := (UInt256.ofNat 11)
  let v436 := (UInt256.ofNat 15)
  let v437 := (UInt256.ofNat 8)
  checkIndex v437 16
  let v438 := v1[v437.toNat]!
  ten_rounds_k20 v1 v433 v434 v435 v436 v438 v0

def ten_rounds_k18 (v1 : (Vector UInt256 16)) (v393 : UInt256) (v394 : UInt256) (v395 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v396 := (UInt256.ofNat 14)
  let v397 := (UInt256.ofNat 3)
  checkIndex v397 16
  let v398 := v1[v397.toNat]!
  let v399 := (UInt256.ofNat 13)
  checkIndex v399 16
  let v400 := v1[v399.toNat]!
  let (v401, v401_0) ← mix v0 v393 v394 v395 v396 v398 v400
  v0 := v401_0
  let v403 := (UInt256.ofNat 0)
  let v404 := (UInt256.ofNat 4)
  let v405 := (UInt256.ofNat 8)
  let v406 := (UInt256.ofNat 12)
  let v407 := (UInt256.ofNat 2)
  checkIndex v407 16
  let v408 := v1[v407.toNat]!
  let v409 := (UInt256.ofNat 12)
  checkIndex v409 16
  let v410 := v1[v409.toNat]!
  let (v411, v411_0) ← mix v0 v403 v404 v405 v406 v408 v410
  v0 := v411_0
  let v413 := (UInt256.ofNat 1)
  let v414 := (UInt256.ofNat 5)
  let v415 := (UInt256.ofNat 9)
  let v416 := (UInt256.ofNat 13)
  let v417 := (UInt256.ofNat 6)
  ten_rounds_k19 v1 v413 v414 v415 v416 v417 v0

def ten_rounds_k17 (v1 : (Vector UInt256 16)) (v373 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v374 := (UInt256.ofNat 6)
  let v375 := (UInt256.ofNat 11)
  let v376 := (UInt256.ofNat 12)
  let v377 := (UInt256.ofNat 11)
  checkIndex v377 16
  let v378 := v1[v377.toNat]!
  let v379 := (UInt256.ofNat 12)
  checkIndex v379 16
  let v380 := v1[v379.toNat]!
  let (v381, v381_0) ← mix v0 v373 v374 v375 v376 v378 v380
  v0 := v381_0
  let v383 := (UInt256.ofNat 2)
  let v384 := (UInt256.ofNat 7)
  let v385 := (UInt256.ofNat 8)
  let v386 := (UInt256.ofNat 13)
  let v387 := (UInt256.ofNat 6)
  checkIndex v387 16
  let v388 := v1[v387.toNat]!
  let v389 := (UInt256.ofNat 8)
  checkIndex v389 16
  let v390 := v1[v389.toNat]!
  let (v391, v391_0) ← mix v0 v383 v384 v385 v386 v388 v390
  v0 := v391_0
  let v393 := (UInt256.ofNat 3)
  let v394 := (UInt256.ofNat 4)
  let v395 := (UInt256.ofNat 9)
  ten_rounds_k18 v1 v393 v394 v395 v0

def ten_rounds_k16 (v1 : (Vector UInt256 16)) (v343 : UInt256) (v344 : UInt256) (v345 : UInt256) (v346 : UInt256) (v348 : UInt256) (v350 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let (v351, v351_0) ← mix v0 v343 v344 v345 v346 v348 v350
  v0 := v351_0
  let v353 := (UInt256.ofNat 3)
  let v354 := (UInt256.ofNat 7)
  let v355 := (UInt256.ofNat 11)
  let v356 := (UInt256.ofNat 15)
  let v357 := (UInt256.ofNat 10)
  checkIndex v357 16
  let v358 := v1[v357.toNat]!
  let v359 := (UInt256.ofNat 15)
  checkIndex v359 16
  let v360 := v1[v359.toNat]!
  let (v361, v361_0) ← mix v0 v353 v354 v355 v356 v358 v360
  v0 := v361_0
  let v363 := (UInt256.ofNat 0)
  let v364 := (UInt256.ofNat 5)
  let v365 := (UInt256.ofNat 10)
  let v366 := (UInt256.ofNat 15)
  let v367 := (UInt256.ofNat 14)
  checkIndex v367 16
  let v368 := v1[v367.toNat]!
  let v369 := (UInt256.ofNat 1)
  checkIndex v369 16
  let v370 := v1[v369.toNat]!
  let (v371, v371_0) ← mix v0 v363 v364 v365 v366 v368 v370
  v0 := v371_0
  let v373 := (UInt256.ofNat 1)
  ten_rounds_k17 v1 v373 v0

def ten_rounds_k15 (v1 : (Vector UInt256 16)) (v323 : UInt256) (v324 : UInt256) (v325 : UInt256) (v326 : UInt256) (v328 : UInt256) (v329 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  checkIndex v329 16
  let v330 := v1[v329.toNat]!
  let (v331, v331_0) ← mix v0 v323 v324 v325 v326 v328 v330
  v0 := v331_0
  let v333 := (UInt256.ofNat 1)
  let v334 := (UInt256.ofNat 5)
  let v335 := (UInt256.ofNat 9)
  let v336 := (UInt256.ofNat 13)
  let v337 := (UInt256.ofNat 5)
  checkIndex v337 16
  let v338 := v1[v337.toNat]!
  let v339 := (UInt256.ofNat 7)
  checkIndex v339 16
  let v340 := v1[v339.toNat]!
  let (v341, v341_0) ← mix v0 v333 v334 v335 v336 v338 v340
  v0 := v341_0
  let v343 := (UInt256.ofNat 2)
  let v344 := (UInt256.ofNat 6)
  let v345 := (UInt256.ofNat 10)
  let v346 := (UInt256.ofNat 14)
  let v347 := (UInt256.ofNat 2)
  checkIndex v347 16
  let v348 := v1[v347.toNat]!
  let v349 := (UInt256.ofNat 4)
  checkIndex v349 16
  let v350 := v1[v349.toNat]!
  ten_rounds_k16 v1 v343 v344 v345 v346 v348 v350 v0

def ten_rounds_k14 (v1 : (Vector UInt256 16)) (v303 : UInt256) (v304 : UInt256) (v305 : UInt256) (v306 : UInt256) (v307 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v308 := v1[v307.toNat]!
  let v309 := (UInt256.ofNat 0)
  checkIndex v309 16
  let v310 := v1[v309.toNat]!
  let (v311, v311_0) ← mix v0 v303 v304 v305 v306 v308 v310
  v0 := v311_0
  let v313 := (UInt256.ofNat 3)
  let v314 := (UInt256.ofNat 4)
  let v315 := (UInt256.ofNat 9)
  let v316 := (UInt256.ofNat 14)
  let v317 := (UInt256.ofNat 15)
  checkIndex v317 16
  let v318 := v1[v317.toNat]!
  let v319 := (UInt256.ofNat 8)
  checkIndex v319 16
  let v320 := v1[v319.toNat]!
  let (v321, v321_0) ← mix v0 v313 v314 v315 v316 v318 v320
  v0 := v321_0
  let v323 := (UInt256.ofNat 0)
  let v324 := (UInt256.ofNat 4)
  let v325 := (UInt256.ofNat 8)
  let v326 := (UInt256.ofNat 12)
  let v327 := (UInt256.ofNat 9)
  checkIndex v327 16
  let v328 := v1[v327.toNat]!
  let v329 := (UInt256.ofNat 0)
  ten_rounds_k15 v1 v323 v324 v325 v326 v328 v329 v0

def ten_rounds_k13 (v1 : (Vector UInt256 16)) (v283 : UInt256) (v284 : UInt256) (v285 : UInt256) (v286 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v287 := (UInt256.ofNat 2)
  checkIndex v287 16
  let v288 := v1[v287.toNat]!
  let v289 := (UInt256.ofNat 6)
  checkIndex v289 16
  let v290 := v1[v289.toNat]!
  let (v291, v291_0) ← mix v0 v283 v284 v285 v286 v288 v290
  v0 := v291_0
  let v293 := (UInt256.ofNat 1)
  let v294 := (UInt256.ofNat 6)
  let v295 := (UInt256.ofNat 11)
  let v296 := (UInt256.ofNat 12)
  let v297 := (UInt256.ofNat 5)
  checkIndex v297 16
  let v298 := v1[v297.toNat]!
  let v299 := (UInt256.ofNat 10)
  checkIndex v299 16
  let v300 := v1[v299.toNat]!
  let (v301, v301_0) ← mix v0 v293 v294 v295 v296 v298 v300
  v0 := v301_0
  let v303 := (UInt256.ofNat 2)
  let v304 := (UInt256.ofNat 7)
  let v305 := (UInt256.ofNat 8)
  let v306 := (UInt256.ofNat 13)
  let v307 := (UInt256.ofNat 4)
  checkIndex v307 16
  ten_rounds_k14 v1 v303 v304 v305 v306 v307 v0

def ten_rounds_k12 (v1 : (Vector UInt256 16)) (v263 : UInt256) (v264 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v265 := (UInt256.ofNat 10)
  let v266 := (UInt256.ofNat 14)
  let v267 := (UInt256.ofNat 13)
  checkIndex v267 16
  let v268 := v1[v267.toNat]!
  let v269 := (UInt256.ofNat 12)
  checkIndex v269 16
  let v270 := v1[v269.toNat]!
  let (v271, v271_0) ← mix v0 v263 v264 v265 v266 v268 v270
  v0 := v271_0
  let v273 := (UInt256.ofNat 3)
  let v274 := (UInt256.ofNat 7)
  let v275 := (UInt256.ofNat 11)
  let v276 := (UInt256.ofNat 15)
  let v277 := (UInt256.ofNat 11)
  checkIndex v277 16
  let v278 := v1[v277.toNat]!
  let v279 := (UInt256.ofNat 14)
  checkIndex v279 16
  let v280 := v1[v279.toNat]!
  let (v281, v281_0) ← mix v0 v273 v274 v275 v276 v278 v280
  v0 := v281_0
  let v283 := (UInt256.ofNat 0)
  let v284 := (UInt256.ofNat 5)
  let v285 := (UInt256.ofNat 10)
  let v286 := (UInt256.ofNat 15)
  ten_rounds_k13 v1 v283 v284 v285 v286 v0

def ten_rounds_k11 (v1 : (Vector UInt256 16)) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v243 := (UInt256.ofNat 0)
  let v244 := (UInt256.ofNat 4)
  let v245 := (UInt256.ofNat 8)
  let v246 := (UInt256.ofNat 12)
  let v247 := (UInt256.ofNat 7)
  checkIndex v247 16
  let v248 := v1[v247.toNat]!
  let v249 := (UInt256.ofNat 9)
  checkIndex v249 16
  let v250 := v1[v249.toNat]!
  let (v251, v251_0) ← mix v0 v243 v244 v245 v246 v248 v250
  v0 := v251_0
  let v253 := (UInt256.ofNat 1)
  let v254 := (UInt256.ofNat 5)
  let v255 := (UInt256.ofNat 9)
  let v256 := (UInt256.ofNat 13)
  let v257 := (UInt256.ofNat 3)
  checkIndex v257 16
  let v258 := v1[v257.toNat]!
  let v259 := (UInt256.ofNat 1)
  checkIndex v259 16
  let v260 := v1[v259.toNat]!
  let (v261, v261_0) ← mix v0 v253 v254 v255 v256 v258 v260
  v0 := v261_0
  let v263 := (UInt256.ofNat 2)
  let v264 := (UInt256.ofNat 6)
  ten_rounds_k12 v1 v263 v264 v0

def ten_rounds_k10 (v1 : (Vector UInt256 16)) (v213 : UInt256) (v214 : UInt256) (v215 : UInt256) (v216 : UInt256) (v218 : UInt256) (v219 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v220 := v1[v219.toNat]!
  let (v221, v221_0) ← mix v0 v213 v214 v215 v216 v218 v220
  v0 := v221_0
  let v223 := (UInt256.ofNat 2)
  let v224 := (UInt256.ofNat 7)
  let v225 := (UInt256.ofNat 8)
  let v226 := (UInt256.ofNat 13)
  let v227 := (UInt256.ofNat 7)
  checkIndex v227 16
  let v228 := v1[v227.toNat]!
  let v229 := (UInt256.ofNat 1)
  checkIndex v229 16
  let v230 := v1[v229.toNat]!
  let (v231, v231_0) ← mix v0 v223 v224 v225 v226 v228 v230
  v0 := v231_0
  let v233 := (UInt256.ofNat 3)
  let v234 := (UInt256.ofNat 4)
  let v235 := (UInt256.ofNat 9)
  let v236 := (UInt256.ofNat 14)
  let v237 := (UInt256.ofNat 9)
  checkIndex v237 16
  let v238 := v1[v237.toNat]!
  let v239 := (UInt256.ofNat 4)
  checkIndex v239 16
  let v240 := v1[v239.toNat]!
  let (v241, v241_0) ← mix v0 v233 v234 v235 v236 v238 v240
  v0 := v241_0
  ten_rounds_k11 v1 v0

def ten_rounds_k9 (v1 : (Vector UInt256 16)) (v193 : UInt256) (v194 : UInt256) (v195 : UInt256) (v196 : UInt256) (v198 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v199 := (UInt256.ofNat 13)
  checkIndex v199 16
  let v200 := v1[v199.toNat]!
  let (v201, v201_0) ← mix v0 v193 v194 v195 v196 v198 v200
  v0 := v201_0
  let v203 := (UInt256.ofNat 0)
  let v204 := (UInt256.ofNat 5)
  let v205 := (UInt256.ofNat 10)
  let v206 := (UInt256.ofNat 15)
  let v207 := (UInt256.ofNat 10)
  checkIndex v207 16
  let v208 := v1[v207.toNat]!
  let v209 := (UInt256.ofNat 14)
  checkIndex v209 16
  let v210 := v1[v209.toNat]!
  let (v211, v211_0) ← mix v0 v203 v204 v205 v206 v208 v210
  v0 := v211_0
  let v213 := (UInt256.ofNat 1)
  let v214 := (UInt256.ofNat 6)
  let v215 := (UInt256.ofNat 11)
  let v216 := (UInt256.ofNat 12)
  let v217 := (UInt256.ofNat 3)
  checkIndex v217 16
  let v218 := v1[v217.toNat]!
  let v219 := (UInt256.ofNat 6)
  checkIndex v219 16
  ten_rounds_k10 v1 v213 v214 v215 v216 v218 v219 v0

def ten_rounds_k8 (v1 : (Vector UInt256 16)) (v173 : UInt256) (v174 : UInt256) (v175 : UInt256) (v176 : UInt256) (v177 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  checkIndex v177 16
  let v178 := v1[v177.toNat]!
  let v179 := (UInt256.ofNat 0)
  checkIndex v179 16
  let v180 := v1[v179.toNat]!
  let (v181, v181_0) ← mix v0 v173 v174 v175 v176 v178 v180
  v0 := v181_0
  let v183 := (UInt256.ofNat 2)
  let v184 := (UInt256.ofNat 6)
  let v185 := (UInt256.ofNat 10)
  let v186 := (UInt256.ofNat 14)
  let v187 := (UInt256.ofNat 5)
  checkIndex v187 16
  let v188 := v1[v187.toNat]!
  let v189 := (UInt256.ofNat 2)
  checkIndex v189 16
  let v190 := v1[v189.toNat]!
  let (v191, v191_0) ← mix v0 v183 v184 v185 v186 v188 v190
  v0 := v191_0
  let v193 := (UInt256.ofNat 3)
  let v194 := (UInt256.ofNat 7)
  let v195 := (UInt256.ofNat 11)
  let v196 := (UInt256.ofNat 15)
  let v197 := (UInt256.ofNat 15)
  checkIndex v197 16
  let v198 := v1[v197.toNat]!
  ten_rounds_k9 v1 v193 v194 v195 v196 v198 v0

def ten_rounds_k7 (v1 : (Vector UInt256 16)) (v153 : UInt256) (v154 : UInt256) (v155 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v156 := (UInt256.ofNat 14)
  let v157 := (UInt256.ofNat 5)
  checkIndex v157 16
  let v158 := v1[v157.toNat]!
  let v159 := (UInt256.ofNat 3)
  checkIndex v159 16
  let v160 := v1[v159.toNat]!
  let (v161, v161_0) ← mix v0 v153 v154 v155 v156 v158 v160
  v0 := v161_0
  let v163 := (UInt256.ofNat 0)
  let v164 := (UInt256.ofNat 4)
  let v165 := (UInt256.ofNat 8)
  let v166 := (UInt256.ofNat 12)
  let v167 := (UInt256.ofNat 11)
  checkIndex v167 16
  let v168 := v1[v167.toNat]!
  let v169 := (UInt256.ofNat 8)
  checkIndex v169 16
  let v170 := v1[v169.toNat]!
  let (v171, v171_0) ← mix v0 v163 v164 v165 v166 v168 v170
  v0 := v171_0
  let v173 := (UInt256.ofNat 1)
  let v174 := (UInt256.ofNat 5)
  let v175 := (UInt256.ofNat 9)
  let v176 := (UInt256.ofNat 13)
  let v177 := (UInt256.ofNat 12)
  ten_rounds_k8 v1 v173 v174 v175 v176 v177 v0

def ten_rounds_k6 (v1 : (Vector UInt256 16)) (v133 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v134 := (UInt256.ofNat 6)
  let v135 := (UInt256.ofNat 11)
  let v136 := (UInt256.ofNat 12)
  let v137 := (UInt256.ofNat 0)
  checkIndex v137 16
  let v138 := v1[v137.toNat]!
  let v139 := (UInt256.ofNat 2)
  checkIndex v139 16
  let v140 := v1[v139.toNat]!
  let (v141, v141_0) ← mix v0 v133 v134 v135 v136 v138 v140
  v0 := v141_0
  let v143 := (UInt256.ofNat 2)
  let v144 := (UInt256.ofNat 7)
  let v145 := (UInt256.ofNat 8)
  let v146 := (UInt256.ofNat 13)
  let v147 := (UInt256.ofNat 11)
  checkIndex v147 16
  let v148 := v1[v147.toNat]!
  let v149 := (UInt256.ofNat 7)
  checkIndex v149 16
  let v150 := v1[v149.toNat]!
  let (v151, v151_0) ← mix v0 v143 v144 v145 v146 v148 v150
  v0 := v151_0
  let v153 := (UInt256.ofNat 3)
  let v154 := (UInt256.ofNat 4)
  let v155 := (UInt256.ofNat 9)
  ten_rounds_k7 v1 v153 v154 v155 v0

def ten_rounds_k5 (v1 : (Vector UInt256 16)) (v103 : UInt256) (v104 : UInt256) (v105 : UInt256) (v106 : UInt256) (v108 : UInt256) (v110 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let (v111, v111_0) ← mix v0 v103 v104 v105 v106 v108 v110
  v0 := v111_0
  let v113 := (UInt256.ofNat 3)
  let v114 := (UInt256.ofNat 7)
  let v115 := (UInt256.ofNat 11)
  let v116 := (UInt256.ofNat 15)
  let v117 := (UInt256.ofNat 13)
  checkIndex v117 16
  let v118 := v1[v117.toNat]!
  let v119 := (UInt256.ofNat 6)
  checkIndex v119 16
  let v120 := v1[v119.toNat]!
  let (v121, v121_0) ← mix v0 v113 v114 v115 v116 v118 v120
  v0 := v121_0
  let v123 := (UInt256.ofNat 0)
  let v124 := (UInt256.ofNat 5)
  let v125 := (UInt256.ofNat 10)
  let v126 := (UInt256.ofNat 15)
  let v127 := (UInt256.ofNat 1)
  checkIndex v127 16
  let v128 := v1[v127.toNat]!
  let v129 := (UInt256.ofNat 12)
  checkIndex v129 16
  let v130 := v1[v129.toNat]!
  let (v131, v131_0) ← mix v0 v123 v124 v125 v126 v128 v130
  v0 := v131_0
  let v133 := (UInt256.ofNat 1)
  ten_rounds_k6 v1 v133 v0

def ten_rounds_k4 (v1 : (Vector UInt256 16)) (v83 : UInt256) (v84 : UInt256) (v85 : UInt256) (v86 : UInt256) (v88 : UInt256) (v89 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  checkIndex v89 16
  let v90 := v1[v89.toNat]!
  let (v91, v91_0) ← mix v0 v83 v84 v85 v86 v88 v90
  v0 := v91_0
  let v93 := (UInt256.ofNat 1)
  let v94 := (UInt256.ofNat 5)
  let v95 := (UInt256.ofNat 9)
  let v96 := (UInt256.ofNat 13)
  let v97 := (UInt256.ofNat 4)
  checkIndex v97 16
  let v98 := v1[v97.toNat]!
  let v99 := (UInt256.ofNat 8)
  checkIndex v99 16
  let v100 := v1[v99.toNat]!
  let (v101, v101_0) ← mix v0 v93 v94 v95 v96 v98 v100
  v0 := v101_0
  let v103 := (UInt256.ofNat 2)
  let v104 := (UInt256.ofNat 6)
  let v105 := (UInt256.ofNat 10)
  let v106 := (UInt256.ofNat 14)
  let v107 := (UInt256.ofNat 9)
  checkIndex v107 16
  let v108 := v1[v107.toNat]!
  let v109 := (UInt256.ofNat 15)
  checkIndex v109 16
  let v110 := v1[v109.toNat]!
  ten_rounds_k5 v1 v103 v104 v105 v106 v108 v110 v0

def ten_rounds_k3 (v1 : (Vector UInt256 16)) (v63 : UInt256) (v64 : UInt256) (v65 : UInt256) (v66 : UInt256) (v67 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v68 := v1[v67.toNat]!
  let v69 := (UInt256.ofNat 13)
  checkIndex v69 16
  let v70 := v1[v69.toNat]!
  let (v71, v71_0) ← mix v0 v63 v64 v65 v66 v68 v70
  v0 := v71_0
  let v73 := (UInt256.ofNat 3)
  let v74 := (UInt256.ofNat 4)
  let v75 := (UInt256.ofNat 9)
  let v76 := (UInt256.ofNat 14)
  let v77 := (UInt256.ofNat 14)
  checkIndex v77 16
  let v78 := v1[v77.toNat]!
  let v79 := (UInt256.ofNat 15)
  checkIndex v79 16
  let v80 := v1[v79.toNat]!
  let (v81, v81_0) ← mix v0 v73 v74 v75 v76 v78 v80
  v0 := v81_0
  let v83 := (UInt256.ofNat 0)
  let v84 := (UInt256.ofNat 4)
  let v85 := (UInt256.ofNat 8)
  let v86 := (UInt256.ofNat 12)
  let v87 := (UInt256.ofNat 14)
  checkIndex v87 16
  let v88 := v1[v87.toNat]!
  let v89 := (UInt256.ofNat 10)
  ten_rounds_k4 v1 v83 v84 v85 v86 v88 v89 v0

def ten_rounds_k2 (v1 : (Vector UInt256 16)) (v43 : UInt256) (v44 : UInt256) (v45 : UInt256) (v46 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v47 := (UInt256.ofNat 8)
  checkIndex v47 16
  let v48 := v1[v47.toNat]!
  let v49 := (UInt256.ofNat 9)
  checkIndex v49 16
  let v50 := v1[v49.toNat]!
  let (v51, v51_0) ← mix v0 v43 v44 v45 v46 v48 v50
  v0 := v51_0
  let v53 := (UInt256.ofNat 1)
  let v54 := (UInt256.ofNat 6)
  let v55 := (UInt256.ofNat 11)
  let v56 := (UInt256.ofNat 12)
  let v57 := (UInt256.ofNat 10)
  checkIndex v57 16
  let v58 := v1[v57.toNat]!
  let v59 := (UInt256.ofNat 11)
  checkIndex v59 16
  let v60 := v1[v59.toNat]!
  let (v61, v61_0) ← mix v0 v53 v54 v55 v56 v58 v60
  v0 := v61_0
  let v63 := (UInt256.ofNat 2)
  let v64 := (UInt256.ofNat 7)
  let v65 := (UInt256.ofNat 8)
  let v66 := (UInt256.ofNat 13)
  let v67 := (UInt256.ofNat 12)
  checkIndex v67 16
  ten_rounds_k3 v1 v63 v64 v65 v66 v67 v0

def ten_rounds_k1 (v1 : (Vector UInt256 16)) (v23 : UInt256) (v24 : UInt256) (v0 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v25 := (UInt256.ofNat 10)
  let v26 := (UInt256.ofNat 14)
  let v27 := (UInt256.ofNat 4)
  checkIndex v27 16
  let v28 := v1[v27.toNat]!
  let v29 := (UInt256.ofNat 5)
  checkIndex v29 16
  let v30 := v1[v29.toNat]!
  let (v31, v31_0) ← mix v0 v23 v24 v25 v26 v28 v30
  v0 := v31_0
  let v33 := (UInt256.ofNat 3)
  let v34 := (UInt256.ofNat 7)
  let v35 := (UInt256.ofNat 11)
  let v36 := (UInt256.ofNat 15)
  let v37 := (UInt256.ofNat 6)
  checkIndex v37 16
  let v38 := v1[v37.toNat]!
  let v39 := (UInt256.ofNat 7)
  checkIndex v39 16
  let v40 := v1[v39.toNat]!
  let (v41, v41_0) ← mix v0 v33 v34 v35 v36 v38 v40
  v0 := v41_0
  let v43 := (UInt256.ofNat 0)
  let v44 := (UInt256.ofNat 5)
  let v45 := (UInt256.ofNat 10)
  let v46 := (UInt256.ofNat 15)
  ten_rounds_k2 v1 v43 v44 v45 v46 v0

def ten_rounds (v0 : (Vector UInt256 16)) (v1 : (Vector UInt256 16)) : FeM (Unit × (Vector UInt256 16)) := do
  let mut v0 := v0
  let v3 := (UInt256.ofNat 0)
  let v4 := (UInt256.ofNat 4)
  let v5 := (UInt256.ofNat 8)
  let v6 := (UInt256.ofNat 12)
  let v7 := (UInt256.ofNat 0)
  checkIndex v7 16
  let v8 := v1[v7.toNat]!
  let v9 := (UInt256.ofNat 1)
  checkIndex v9 16
  let v10 := v1[v9.toNat]!
  let (v11, v11_0) ← mix v0 v3 v4 v5 v6 v8 v10
  v0 := v11_0
  let v13 := (UInt256.ofNat 1)
  let v14 := (UInt256.ofNat 5)
  let v15 := (UInt256.ofNat 9)
  let v16 := (UInt256.ofNat 13)
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  let v18 := v1[v17.toNat]!
  let v19 := (UInt256.ofNat 3)
  checkIndex v19 16
  let v20 := v1[v19.toNat]!
  let (v21, v21_0) ← mix v0 v13 v14 v15 v16 v18 v20
  v0 := v21_0
  let v23 := (UInt256.ofNat 2)
  let v24 := (UInt256.ofNat 6)
  ten_rounds_k1 v1 v23 v24 v0

def SIGMA_k6 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) (v10 : UInt256) (v11 : UInt256) (v12 : UInt256) (v13 : UInt256) (v14 : UInt256) (v15 : UInt256) (v16 : UInt256) (v17 : UInt256) (v18 : UInt256) (v19 : UInt256) (v20 : UInt256) (v21 : UInt256) (v22 : UInt256) (v23 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) (v29 : UInt256) (v30 : UInt256) (v31 : UInt256) (v32 : UInt256) (v33 : UInt256) (v34 : UInt256) (v35 : UInt256) (v36 : UInt256) (v37 : UInt256) (v38 : UInt256) (v39 : UInt256) (v40 : UInt256) (v41 : UInt256) (v42 : UInt256) (v43 : UInt256) (v44 : UInt256) (v45 : UInt256) (v46 : UInt256) (v47 : UInt256) (v48 : UInt256) (v49 : UInt256) (v50 : UInt256) (v51 : UInt256) (v52 : UInt256) (v53 : UInt256) (v54 : UInt256) (v55 : UInt256) (v56 : UInt256) (v57 : UInt256) (v58 : UInt256) (v59 : UInt256) (v60 : UInt256) (v61 : UInt256) (v62 : UInt256) (v63 : UInt256) (v64 : UInt256) (v65 : UInt256) (v66 : UInt256) (v67 : UInt256) (v68 : UInt256) (v69 : UInt256) (v70 : UInt256) (v71 : UInt256) (v72 : UInt256) (v73 : UInt256) (v74 : UInt256) (v75 : UInt256) (v76 : UInt256) (v77 : UInt256) (v78 : UInt256) (v79 : UInt256) (v80 : UInt256) (v81 : UInt256) (v82 : UInt256) (v83 : UInt256) (v84 : UInt256) (v85 : UInt256) (v86 : UInt256) (v87 : UInt256) (v88 : UInt256) (v89 : UInt256) (v90 : UInt256) (v91 : UInt256) (v92 : UInt256) (v93 : UInt256) (v94 : UInt256) (v95 : UInt256) (v96 : UInt256) (v97 : UInt256) (v98 : UInt256) (v99 : UInt256) (v100 : UInt256) (v101 : UInt256) (v102 : UInt256) (v103 : UInt256) (v104 : UInt256) (v105 : UInt256) (v106 : UInt256) (v107 : UInt256) (v108 : UInt256) (v109 : UInt256) (v110 : UInt256) (v111 : UInt256) (v112 : UInt256) (v113 : UInt256) (v114 : UInt256) (v115 : UInt256) (v116 : UInt256) (v117 : UInt256) (v118 : UInt256) (v119 : UInt256) (v120 : UInt256) (v121 : UInt256) (v122 : UInt256) (v123 : UInt256) (v124 : UInt256) (v125 : UInt256) (v126 : UInt256) (v127 : UInt256) (v128 : UInt256) (v129 : UInt256) (v130 : UInt256) (v131 : UInt256) (v132 : UInt256) (v133 : UInt256) (v134 : UInt256) (v135 : UInt256) (v136 : UInt256) (v137 : UInt256) (v138 : UInt256) (v139 : UInt256) (v140 : UInt256) (v141 : UInt256) (v142 : UInt256) (v143 : UInt256) : FeM (Vector UInt256 160) := do
  let v144 := (UInt256.ofNat 10)
  let v145 := (UInt256.ofNat 2)
  let v146 := (UInt256.ofNat 8)
  let v147 := (UInt256.ofNat 4)
  let v148 := (UInt256.ofNat 7)
  let v149 := (UInt256.ofNat 6)
  let v150 := (UInt256.ofNat 1)
  let v151 := (UInt256.ofNat 5)
  let v152 := (UInt256.ofNat 15)
  let v153 := (UInt256.ofNat 11)
  let v154 := (UInt256.ofNat 9)
  let v155 := (UInt256.ofNat 14)
  let v156 := (UInt256.ofNat 3)
  let v157 := (UInt256.ofNat 12)
  let v158 := (UInt256.ofNat 13)
  let v159 := (UInt256.ofNat 0)
  let v160 := #v[v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137, v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152, v153, v154, v155, v156, v157, v158, v159]
  return v160

def SIGMA_k5 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) (v10 : UInt256) (v11 : UInt256) (v12 : UInt256) (v13 : UInt256) (v14 : UInt256) (v15 : UInt256) (v16 : UInt256) (v17 : UInt256) (v18 : UInt256) (v19 : UInt256) (v20 : UInt256) (v21 : UInt256) (v22 : UInt256) (v23 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) (v29 : UInt256) (v30 : UInt256) (v31 : UInt256) (v32 : UInt256) (v33 : UInt256) (v34 : UInt256) (v35 : UInt256) (v36 : UInt256) (v37 : UInt256) (v38 : UInt256) (v39 : UInt256) (v40 : UInt256) (v41 : UInt256) (v42 : UInt256) (v43 : UInt256) (v44 : UInt256) (v45 : UInt256) (v46 : UInt256) (v47 : UInt256) (v48 : UInt256) (v49 : UInt256) (v50 : UInt256) (v51 : UInt256) (v52 : UInt256) (v53 : UInt256) (v54 : UInt256) (v55 : UInt256) (v56 : UInt256) (v57 : UInt256) (v58 : UInt256) (v59 : UInt256) (v60 : UInt256) (v61 : UInt256) (v62 : UInt256) (v63 : UInt256) (v64 : UInt256) (v65 : UInt256) (v66 : UInt256) (v67 : UInt256) (v68 : UInt256) (v69 : UInt256) (v70 : UInt256) (v71 : UInt256) (v72 : UInt256) (v73 : UInt256) (v74 : UInt256) (v75 : UInt256) (v76 : UInt256) (v77 : UInt256) (v78 : UInt256) (v79 : UInt256) (v80 : UInt256) (v81 : UInt256) (v82 : UInt256) (v83 : UInt256) (v84 : UInt256) (v85 : UInt256) (v86 : UInt256) (v87 : UInt256) (v88 : UInt256) (v89 : UInt256) (v90 : UInt256) (v91 : UInt256) (v92 : UInt256) (v93 : UInt256) (v94 : UInt256) (v95 : UInt256) (v96 : UInt256) (v97 : UInt256) (v98 : UInt256) (v99 : UInt256) (v100 : UInt256) (v101 : UInt256) (v102 : UInt256) (v103 : UInt256) (v104 : UInt256) (v105 : UInt256) (v106 : UInt256) (v107 : UInt256) (v108 : UInt256) (v109 : UInt256) (v110 : UInt256) (v111 : UInt256) (v112 : UInt256) (v113 : UInt256) (v114 : UInt256) (v115 : UInt256) (v116 : UInt256) (v117 : UInt256) (v118 : UInt256) (v119 : UInt256) : FeM (Vector UInt256 160) := do
  let v120 := (UInt256.ofNat 5)
  let v121 := (UInt256.ofNat 0)
  let v122 := (UInt256.ofNat 15)
  let v123 := (UInt256.ofNat 4)
  let v124 := (UInt256.ofNat 8)
  let v125 := (UInt256.ofNat 6)
  let v126 := (UInt256.ofNat 2)
  let v127 := (UInt256.ofNat 10)
  let v128 := (UInt256.ofNat 6)
  let v129 := (UInt256.ofNat 15)
  let v130 := (UInt256.ofNat 14)
  let v131 := (UInt256.ofNat 9)
  let v132 := (UInt256.ofNat 11)
  let v133 := (UInt256.ofNat 3)
  let v134 := (UInt256.ofNat 0)
  let v135 := (UInt256.ofNat 8)
  let v136 := (UInt256.ofNat 12)
  let v137 := (UInt256.ofNat 2)
  let v138 := (UInt256.ofNat 13)
  let v139 := (UInt256.ofNat 7)
  let v140 := (UInt256.ofNat 1)
  let v141 := (UInt256.ofNat 4)
  let v142 := (UInt256.ofNat 10)
  let v143 := (UInt256.ofNat 5)
  SIGMA_k6 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83 v84 v85 v86 v87 v88 v89 v90 v91 v92 v93 v94 v95 v96 v97 v98 v99 v100 v101 v102 v103 v104 v105 v106 v107 v108 v109 v110 v111 v112 v113 v114 v115 v116 v117 v118 v119 v120 v121 v122 v123 v124 v125 v126 v127 v128 v129 v130 v131 v132 v133 v134 v135 v136 v137 v138 v139 v140 v141 v142 v143

def SIGMA_k4 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) (v10 : UInt256) (v11 : UInt256) (v12 : UInt256) (v13 : UInt256) (v14 : UInt256) (v15 : UInt256) (v16 : UInt256) (v17 : UInt256) (v18 : UInt256) (v19 : UInt256) (v20 : UInt256) (v21 : UInt256) (v22 : UInt256) (v23 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) (v29 : UInt256) (v30 : UInt256) (v31 : UInt256) (v32 : UInt256) (v33 : UInt256) (v34 : UInt256) (v35 : UInt256) (v36 : UInt256) (v37 : UInt256) (v38 : UInt256) (v39 : UInt256) (v40 : UInt256) (v41 : UInt256) (v42 : UInt256) (v43 : UInt256) (v44 : UInt256) (v45 : UInt256) (v46 : UInt256) (v47 : UInt256) (v48 : UInt256) (v49 : UInt256) (v50 : UInt256) (v51 : UInt256) (v52 : UInt256) (v53 : UInt256) (v54 : UInt256) (v55 : UInt256) (v56 : UInt256) (v57 : UInt256) (v58 : UInt256) (v59 : UInt256) (v60 : UInt256) (v61 : UInt256) (v62 : UInt256) (v63 : UInt256) (v64 : UInt256) (v65 : UInt256) (v66 : UInt256) (v67 : UInt256) (v68 : UInt256) (v69 : UInt256) (v70 : UInt256) (v71 : UInt256) (v72 : UInt256) (v73 : UInt256) (v74 : UInt256) (v75 : UInt256) (v76 : UInt256) (v77 : UInt256) (v78 : UInt256) (v79 : UInt256) (v80 : UInt256) (v81 : UInt256) (v82 : UInt256) (v83 : UInt256) (v84 : UInt256) (v85 : UInt256) (v86 : UInt256) (v87 : UInt256) (v88 : UInt256) (v89 : UInt256) (v90 : UInt256) (v91 : UInt256) (v92 : UInt256) (v93 : UInt256) (v94 : UInt256) (v95 : UInt256) : FeM (Vector UInt256 160) := do
  let v96 := (UInt256.ofNat 12)
  let v97 := (UInt256.ofNat 5)
  let v98 := (UInt256.ofNat 1)
  let v99 := (UInt256.ofNat 15)
  let v100 := (UInt256.ofNat 14)
  let v101 := (UInt256.ofNat 13)
  let v102 := (UInt256.ofNat 4)
  let v103 := (UInt256.ofNat 10)
  let v104 := (UInt256.ofNat 0)
  let v105 := (UInt256.ofNat 7)
  let v106 := (UInt256.ofNat 6)
  let v107 := (UInt256.ofNat 3)
  let v108 := (UInt256.ofNat 9)
  let v109 := (UInt256.ofNat 2)
  let v110 := (UInt256.ofNat 8)
  let v111 := (UInt256.ofNat 11)
  let v112 := (UInt256.ofNat 13)
  let v113 := (UInt256.ofNat 11)
  let v114 := (UInt256.ofNat 7)
  let v115 := (UInt256.ofNat 14)
  let v116 := (UInt256.ofNat 12)
  let v117 := (UInt256.ofNat 1)
  let v118 := (UInt256.ofNat 3)
  let v119 := (UInt256.ofNat 9)
  SIGMA_k5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83 v84 v85 v86 v87 v88 v89 v90 v91 v92 v93 v94 v95 v96 v97 v98 v99 v100 v101 v102 v103 v104 v105 v106 v107 v108 v109 v110 v111 v112 v113 v114 v115 v116 v117 v118 v119

def SIGMA_k3 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) (v10 : UInt256) (v11 : UInt256) (v12 : UInt256) (v13 : UInt256) (v14 : UInt256) (v15 : UInt256) (v16 : UInt256) (v17 : UInt256) (v18 : UInt256) (v19 : UInt256) (v20 : UInt256) (v21 : UInt256) (v22 : UInt256) (v23 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) (v29 : UInt256) (v30 : UInt256) (v31 : UInt256) (v32 : UInt256) (v33 : UInt256) (v34 : UInt256) (v35 : UInt256) (v36 : UInt256) (v37 : UInt256) (v38 : UInt256) (v39 : UInt256) (v40 : UInt256) (v41 : UInt256) (v42 : UInt256) (v43 : UInt256) (v44 : UInt256) (v45 : UInt256) (v46 : UInt256) (v47 : UInt256) (v48 : UInt256) (v49 : UInt256) (v50 : UInt256) (v51 : UInt256) (v52 : UInt256) (v53 : UInt256) (v54 : UInt256) (v55 : UInt256) (v56 : UInt256) (v57 : UInt256) (v58 : UInt256) (v59 : UInt256) (v60 : UInt256) (v61 : UInt256) (v62 : UInt256) (v63 : UInt256) (v64 : UInt256) (v65 : UInt256) (v66 : UInt256) (v67 : UInt256) (v68 : UInt256) (v69 : UInt256) (v70 : UInt256) (v71 : UInt256) : FeM (Vector UInt256 160) := do
  let v72 := (UInt256.ofNat 14)
  let v73 := (UInt256.ofNat 1)
  let v74 := (UInt256.ofNat 11)
  let v75 := (UInt256.ofNat 12)
  let v76 := (UInt256.ofNat 6)
  let v77 := (UInt256.ofNat 8)
  let v78 := (UInt256.ofNat 3)
  let v79 := (UInt256.ofNat 13)
  let v80 := (UInt256.ofNat 2)
  let v81 := (UInt256.ofNat 12)
  let v82 := (UInt256.ofNat 6)
  let v83 := (UInt256.ofNat 10)
  let v84 := (UInt256.ofNat 0)
  let v85 := (UInt256.ofNat 11)
  let v86 := (UInt256.ofNat 8)
  let v87 := (UInt256.ofNat 3)
  let v88 := (UInt256.ofNat 4)
  let v89 := (UInt256.ofNat 13)
  let v90 := (UInt256.ofNat 7)
  let v91 := (UInt256.ofNat 5)
  let v92 := (UInt256.ofNat 15)
  let v93 := (UInt256.ofNat 14)
  let v94 := (UInt256.ofNat 1)
  let v95 := (UInt256.ofNat 9)
  SIGMA_k4 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83 v84 v85 v86 v87 v88 v89 v90 v91 v92 v93 v94 v95

def SIGMA_k2 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) (v10 : UInt256) (v11 : UInt256) (v12 : UInt256) (v13 : UInt256) (v14 : UInt256) (v15 : UInt256) (v16 : UInt256) (v17 : UInt256) (v18 : UInt256) (v19 : UInt256) (v20 : UInt256) (v21 : UInt256) (v22 : UInt256) (v23 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) (v29 : UInt256) (v30 : UInt256) (v31 : UInt256) (v32 : UInt256) (v33 : UInt256) (v34 : UInt256) (v35 : UInt256) (v36 : UInt256) (v37 : UInt256) (v38 : UInt256) (v39 : UInt256) (v40 : UInt256) (v41 : UInt256) (v42 : UInt256) (v43 : UInt256) (v44 : UInt256) (v45 : UInt256) (v46 : UInt256) (v47 : UInt256) : FeM (Vector UInt256 160) := do
  let v48 := (UInt256.ofNat 7)
  let v49 := (UInt256.ofNat 9)
  let v50 := (UInt256.ofNat 3)
  let v51 := (UInt256.ofNat 1)
  let v52 := (UInt256.ofNat 13)
  let v53 := (UInt256.ofNat 12)
  let v54 := (UInt256.ofNat 11)
  let v55 := (UInt256.ofNat 14)
  let v56 := (UInt256.ofNat 2)
  let v57 := (UInt256.ofNat 6)
  let v58 := (UInt256.ofNat 5)
  let v59 := (UInt256.ofNat 10)
  let v60 := (UInt256.ofNat 4)
  let v61 := (UInt256.ofNat 0)
  let v62 := (UInt256.ofNat 15)
  let v63 := (UInt256.ofNat 8)
  let v64 := (UInt256.ofNat 9)
  let v65 := (UInt256.ofNat 0)
  let v66 := (UInt256.ofNat 5)
  let v67 := (UInt256.ofNat 7)
  let v68 := (UInt256.ofNat 2)
  let v69 := (UInt256.ofNat 4)
  let v70 := (UInt256.ofNat 10)
  let v71 := (UInt256.ofNat 15)
  SIGMA_k3 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71

def SIGMA_k1 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v6 : UInt256) (v7 : UInt256) (v8 : UInt256) (v9 : UInt256) (v10 : UInt256) (v11 : UInt256) (v12 : UInt256) (v13 : UInt256) (v14 : UInt256) (v15 : UInt256) (v16 : UInt256) (v17 : UInt256) (v18 : UInt256) (v19 : UInt256) (v20 : UInt256) (v21 : UInt256) (v22 : UInt256) (v23 : UInt256) : FeM (Vector UInt256 160) := do
  let v24 := (UInt256.ofNat 1)
  let v25 := (UInt256.ofNat 12)
  let v26 := (UInt256.ofNat 0)
  let v27 := (UInt256.ofNat 2)
  let v28 := (UInt256.ofNat 11)
  let v29 := (UInt256.ofNat 7)
  let v30 := (UInt256.ofNat 5)
  let v31 := (UInt256.ofNat 3)
  let v32 := (UInt256.ofNat 11)
  let v33 := (UInt256.ofNat 8)
  let v34 := (UInt256.ofNat 12)
  let v35 := (UInt256.ofNat 0)
  let v36 := (UInt256.ofNat 5)
  let v37 := (UInt256.ofNat 2)
  let v38 := (UInt256.ofNat 15)
  let v39 := (UInt256.ofNat 13)
  let v40 := (UInt256.ofNat 10)
  let v41 := (UInt256.ofNat 14)
  let v42 := (UInt256.ofNat 3)
  let v43 := (UInt256.ofNat 6)
  let v44 := (UInt256.ofNat 7)
  let v45 := (UInt256.ofNat 1)
  let v46 := (UInt256.ofNat 9)
  let v47 := (UInt256.ofNat 4)
  SIGMA_k2 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47

def SIGMA : FeM (Vector UInt256 160) := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (UInt256.ofNat 1)
  let v2 := (UInt256.ofNat 2)
  let v3 := (UInt256.ofNat 3)
  let v4 := (UInt256.ofNat 4)
  let v5 := (UInt256.ofNat 5)
  let v6 := (UInt256.ofNat 6)
  let v7 := (UInt256.ofNat 7)
  let v8 := (UInt256.ofNat 8)
  let v9 := (UInt256.ofNat 9)
  let v10 := (UInt256.ofNat 10)
  let v11 := (UInt256.ofNat 11)
  let v12 := (UInt256.ofNat 12)
  let v13 := (UInt256.ofNat 13)
  let v14 := (UInt256.ofNat 14)
  let v15 := (UInt256.ofNat 15)
  let v16 := (UInt256.ofNat 14)
  let v17 := (UInt256.ofNat 10)
  let v18 := (UInt256.ofNat 4)
  let v19 := (UInt256.ofNat 8)
  let v20 := (UInt256.ofNat 9)
  let v21 := (UInt256.ofNat 15)
  let v22 := (UInt256.ofNat 13)
  let v23 := (UInt256.ofNat 6)
  SIGMA_k1 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23

def compress_MemBuffer (v0 : MemBuffer) : FeM (Vector UInt256 8) := do
  let mut r0 : (Vector UInt256 8) := default
  let mut r1 : (Vector UInt256 16) := default
  let mut r2 : (Vector UInt256 16) := default
  let mut r3 : UInt256 := default
  let mut r4 : Range_Known_0_Known_8 := default
  let mut r5 : Range_Known_0_Known_16 := default
  let mut r6 : Range_Known_0_Known_8 := default
  let mut r7 : (Vector UInt256 16) := default
  let mut r8 : Range_Known_0_Known_8 := default
  let v6 ← MemBuffer_len_1 v0
  let v7 ← INPUT_LEN
  let v8 := decide (v6 = v7)
  let b3 : (Vector UInt256 8) → (Vector UInt256 16) → (Vector UInt256 16) → UInt256 → Range_Known_0_Known_8 → Range_Known_0_Known_16 → Range_Known_0_Known_8 → (Vector UInt256 16) → Range_Known_0_Known_8 → Bool → FeM (Vector UInt256 8) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    let v17 ← assert v1
    let v18 := (UInt256.ofNat 0)
    let v19 := Vector.replicate 8 v18
    let v20 := v19
    r0 := v20
    let v21 := (UInt256.ofNat 0)
    let v22 := Vector.replicate 16 v21
    let v23 := v22
    r1 := v23
    let v24 := (UInt256.ofNat 0)
    let v25 := (UInt256.ofNat 8)
    let v26 := ()
    let v27 := (Range_Known_0_Known_8.mk v26 v26)
    r4 := v27
    let v29 ← Range_start_0_8 r4
    let v30 := v29
    let rec b5 (r0 : (Vector UInt256 8)) (r1 : (Vector UInt256 16)) (r2 : (Vector UInt256 16)) (r3 : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v2 : UInt256) : FeM (Vector UInt256 8) := do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let mut r7 := r7
      let mut r8 := r8
      let v32 ← Range_has_0_8 r4 v2
      if v32 then
        let v34 ← Range_at_0_8 r4 v2
        let v35 := v34
        let v36 := (UInt256.ofNat 4)
        let v37 := (UInt256.ofNat 8)
        let v38 ← checkedMul v35 v37
        let v39 ← checkedAdd v36 v38
        let v40 ← MemBuffer_lane_at v0 v39
        checkIndex v35 8
        r0 := (r0.set! v35.toNat v40)
        let v41 := ()
        let v43 ← Range_next_0_8 r4 v2
        let v44 := v43
        b5 r0 r1 r2 r3 r4 r5 r6 r7 r8 v44
      else
        let v45 := (UInt256.ofNat 0)
        let v46 := (UInt256.ofNat 16)
        let v47 := ()
        let v48 := (Range_Known_0_Known_16.mk v47 v47)
        r5 := v48
        let v50 ← Range_start_0_16 r5
        let v51 := v50
        let rec b9 (r0 : (Vector UInt256 8)) (r1 : (Vector UInt256 16)) (r2 : (Vector UInt256 16)) (r3 : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v3 : UInt256) : FeM (Vector UInt256 8) := do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut r7 := r7
          let mut r8 := r8
          let v53 ← Range_has_0_16 r5 v3
          if v53 then
            let v55 ← Range_at_0_16 r5 v3
            let v56 := v55
            let v57 := (UInt256.ofNat 68)
            let v58 := (UInt256.ofNat 8)
            let v59 ← checkedMul v56 v58
            let v60 ← checkedAdd v57 v59
            let v61 ← MemBuffer_lane_at v0 v60
            checkIndex v56 16
            r1 := (r1.set! v56.toNat v61)
            let v62 := ()
            let v64 ← Range_next_0_16 r5 v3
            let v65 := v64
            b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 v65
          else
            let v66 := (UInt256.ofNat 0)
            let v67 := Vector.replicate 16 v66
            let v68 := v67
            r2 := v68
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 8)
            let v71 := ()
            let v72 := (Range_Known_0_Known_8.mk v71 v71)
            r6 := v72
            let v74 ← Range_start_0_8 r6
            let v75 := v74
            let rec b13 (r0 : (Vector UInt256 8)) (r1 : (Vector UInt256 16)) (r2 : (Vector UInt256 16)) (r3 : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v4 : UInt256) : FeM (Vector UInt256 8) := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let mut r6 := r6
              let mut r7 := r7
              let mut r8 := r8
              let v77 ← Range_has_0_8 r6 v4
              if v77 then
                let v79 ← Range_at_0_8 r6 v4
                let v80 := v79
                checkIndex v80 8
                let v81 := r0[v80.toNat]!
                checkIndex v80 16
                r2 := (r2.set! v80.toNat v81)
                let v82 := ()
                let v83 ← IV
                checkIndex v80 8
                let v84 := v83[v80.toNat]!
                let v85 := (UInt256.ofNat 8)
                let v86 ← checkedAdd v80 v85
                checkIndex v86 16
                r2 := (r2.set! v86.toNat v84)
                let v87 := ()
                let v89 ← Range_next_0_8 r6 v4
                let v90 := v89
                b13 r0 r1 r2 r3 r4 r5 r6 r7 r8 v90
              else
                let v91 := (UInt256.ofNat 196)
                let v92 ← MemBuffer_lane_at v0 v91
                let v93 := (UInt256.ofNat 12)
                checkIndex v93 16
                let v94 := r2[v93.toNat]!
                let v95 := v94 ^^^ v92
                checkIndex v93 16
                r2 := (r2.set! v93.toNat v95)
                let v96 := ()
                let v97 := (UInt256.ofNat 204)
                let v98 ← MemBuffer_lane_at v0 v97
                let v99 := (UInt256.ofNat 13)
                checkIndex v99 16
                let v100 := r2[v99.toNat]!
                let v101 := v100 ^^^ v98
                checkIndex v99 16
                r2 := (r2.set! v99.toNat v101)
                let v102 := ()
                let v103 := (UInt256.ofNat 212)
                let v104 ← MemBuffer_byte_at v0 v103
                let v105 := (UInt256.ofNat 1)
                let v106 := decide (v104 = v105)
                let b19 : (Vector UInt256 8) → (Vector UInt256 16) → (Vector UInt256 16) → UInt256 → Range_Known_0_Known_8 → Range_Known_0_Known_16 → Range_Known_0_Known_8 → (Vector UInt256 16) → Range_Known_0_Known_8 → FeM (Vector UInt256 8) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 => do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let mut r6 := r6
                  let mut r7 := r7
                  let mut r8 := r8
                  let v115 ← rounds_MemBuffer v0
                  let v116 := v115
                  let v117 := (UInt256.ofNat 0)
                  let v118 := v117
                  r3 := v118
                  let rec b20 (r0 : (Vector UInt256 8)) (r1 : (Vector UInt256 16)) (r2 : (Vector UInt256 16)) (r3 : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) : FeM (Vector UInt256 8) := do
                    let mut r0 := r0
                    let mut r1 := r1
                    let mut r2 := r2
                    let mut r3 := r3
                    let mut r4 := r4
                    let mut r5 := r5
                    let mut r6 := r6
                    let mut r7 := r7
                    let mut r8 := r8
                    let v119 := r3
                    let v120 ← IntTy.sub (.u 64) v116 v119
                    let v121 := (UInt256.ofNat 10)
                    let v122 := decide (v121.toNat ≤ v120.toNat)
                    if v122 then
                      let v124 := r1
                      let v125 := v124
                      r7 := v125
                      let (v127, v127_0) ← ten_rounds r2 r7
                      r2 := v127_0
                      let v128 := (UInt256.ofNat 10)
                      let v129 := r3
                      let v130 ← IntTy.add (.u 64) v129 v128
                      let v131 := v130
                      r3 := v131
                      let v132 := ()
                      b20 r0 r1 r2 r3 r4 r5 r6 r7 r8
                    else
                      let rec b23 (r0 : (Vector UInt256 8)) (r1 : (Vector UInt256 16)) (r2 : (Vector UInt256 16)) (r3 : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) : FeM (Vector UInt256 8) := do
                        let mut r0 := r0
                        let mut r1 := r1
                        let mut r2 := r2
                        let mut r3 := r3
                        let mut r4 := r4
                        let mut r5 := r5
                        let mut r6 := r6
                        let mut r7 := r7
                        let mut r8 := r8
                        let v133 := r3
                        let v134 := v133
                        let v135 := decide (v134.toNat < v116.toNat)
                        if v135 then
                          let v136 := r3
                          let v137 := v136
                          let v138 := (UInt256.ofNat 10)
                          let v139 ← checkedRem v137 v138
                          let v140 := (UInt256.ofNat 16)
                          let v141 ← IntTy.mul (.u 64) v139 v140
                          let v142 ← downcast_truncate_u64_usize v141
                          let v143 := v142
                          let v145 := (UInt256.ofNat 0)
                          let v146 := (UInt256.ofNat 4)
                          let v147 := (UInt256.ofNat 8)
                          let v148 := (UInt256.ofNat 12)
                          let v149 ← SIGMA
                          checkIndex v143 160
                          let v150 := v149[v143.toNat]!
                          let v151 := v150
                          checkIndex v151 16
                          let v152 := r1[v151.toNat]!
                          let v153 ← SIGMA
                          let v154 := (UInt256.ofNat 1)
                          let v155 ← checkedAdd v143 v154
                          checkIndex v155 160
                          let v156 := v153[v155.toNat]!
                          let v157 := v156
                          checkIndex v157 16
                          let v158 := r1[v157.toNat]!
                          let (v159, v159_0) ← mix r2 v145 v146 v147 v148 v152 v158
                          r2 := v159_0
                          let v161 := (UInt256.ofNat 1)
                          let v162 := (UInt256.ofNat 5)
                          let v163 := (UInt256.ofNat 9)
                          let v164 := (UInt256.ofNat 13)
                          let v165 ← SIGMA
                          let v166 := (UInt256.ofNat 2)
                          let v167 ← checkedAdd v143 v166
                          checkIndex v167 160
                          let v168 := v165[v167.toNat]!
                          let v169 := v168
                          checkIndex v169 16
                          let v170 := r1[v169.toNat]!
                          let v171 ← SIGMA
                          let v172 := (UInt256.ofNat 3)
                          let v173 ← checkedAdd v143 v172
                          checkIndex v173 160
                          let v174 := v171[v173.toNat]!
                          let v175 := v174
                          checkIndex v175 16
                          let v176 := r1[v175.toNat]!
                          let (v177, v177_0) ← mix r2 v161 v162 v163 v164 v170 v176
                          r2 := v177_0
                          let v179 := (UInt256.ofNat 2)
                          let v180 := (UInt256.ofNat 6)
                          let v181 := (UInt256.ofNat 10)
                          let v182 := (UInt256.ofNat 14)
                          let v183 ← SIGMA
                          let v184 := (UInt256.ofNat 4)
                          let v185 ← checkedAdd v143 v184
                          checkIndex v185 160
                          let v186 := v183[v185.toNat]!
                          let v187 := v186
                          checkIndex v187 16
                          let v188 := r1[v187.toNat]!
                          let v189 ← SIGMA
                          let v190 := (UInt256.ofNat 5)
                          let v191 ← checkedAdd v143 v190
                          checkIndex v191 160
                          let v192 := v189[v191.toNat]!
                          let v193 := v192
                          checkIndex v193 16
                          let v194 := r1[v193.toNat]!
                          let (v195, v195_0) ← mix r2 v179 v180 v181 v182 v188 v194
                          r2 := v195_0
                          let v197 := (UInt256.ofNat 3)
                          let v198 := (UInt256.ofNat 7)
                          let v199 := (UInt256.ofNat 11)
                          let v200 := (UInt256.ofNat 15)
                          let v201 ← SIGMA
                          let v202 := (UInt256.ofNat 6)
                          let v203 ← checkedAdd v143 v202
                          checkIndex v203 160
                          let v204 := v201[v203.toNat]!
                          let v205 := v204
                          checkIndex v205 16
                          let v206 := r1[v205.toNat]!
                          let v207 ← SIGMA
                          let v208 := (UInt256.ofNat 7)
                          let v209 ← checkedAdd v143 v208
                          checkIndex v209 160
                          let v210 := v207[v209.toNat]!
                          let v211 := v210
                          checkIndex v211 16
                          let v212 := r1[v211.toNat]!
                          let (v213, v213_0) ← mix r2 v197 v198 v199 v200 v206 v212
                          r2 := v213_0
                          let v215 := (UInt256.ofNat 0)
                          let v216 := (UInt256.ofNat 5)
                          let v217 := (UInt256.ofNat 10)
                          let v218 := (UInt256.ofNat 15)
                          let v219 ← SIGMA
                          let v220 := (UInt256.ofNat 8)
                          let v221 ← checkedAdd v143 v220
                          checkIndex v221 160
                          let v222 := v219[v221.toNat]!
                          let v223 := v222
                          checkIndex v223 16
                          let v224 := r1[v223.toNat]!
                          let v225 ← SIGMA
                          let v226 := (UInt256.ofNat 9)
                          let v227 ← checkedAdd v143 v226
                          checkIndex v227 160
                          let v228 := v225[v227.toNat]!
                          let v229 := v228
                          checkIndex v229 16
                          let v230 := r1[v229.toNat]!
                          let (v231, v231_0) ← mix r2 v215 v216 v217 v218 v224 v230
                          r2 := v231_0
                          let v233 := (UInt256.ofNat 1)
                          let v234 := (UInt256.ofNat 6)
                          let v235 := (UInt256.ofNat 11)
                          let v236 := (UInt256.ofNat 12)
                          let v237 ← SIGMA
                          let v238 := (UInt256.ofNat 10)
                          let v239 ← checkedAdd v143 v238
                          checkIndex v239 160
                          let v240 := v237[v239.toNat]!
                          let v241 := v240
                          checkIndex v241 16
                          let v242 := r1[v241.toNat]!
                          let v243 ← SIGMA
                          let v244 := (UInt256.ofNat 11)
                          let v245 ← checkedAdd v143 v244
                          checkIndex v245 160
                          let v246 := v243[v245.toNat]!
                          let v247 := v246
                          checkIndex v247 16
                          let v248 := r1[v247.toNat]!
                          let (v249, v249_0) ← mix r2 v233 v234 v235 v236 v242 v248
                          r2 := v249_0
                          let v251 := (UInt256.ofNat 2)
                          let v252 := (UInt256.ofNat 7)
                          let v253 := (UInt256.ofNat 8)
                          let v254 := (UInt256.ofNat 13)
                          let v255 ← SIGMA
                          let v256 := (UInt256.ofNat 12)
                          let v257 ← checkedAdd v143 v256
                          checkIndex v257 160
                          let v258 := v255[v257.toNat]!
                          let v259 := v258
                          checkIndex v259 16
                          let v260 := r1[v259.toNat]!
                          let v261 ← SIGMA
                          let v262 := (UInt256.ofNat 13)
                          let v263 ← checkedAdd v143 v262
                          checkIndex v263 160
                          let v264 := v261[v263.toNat]!
                          let v265 := v264
                          checkIndex v265 16
                          let v266 := r1[v265.toNat]!
                          let (v267, v267_0) ← mix r2 v251 v252 v253 v254 v260 v266
                          r2 := v267_0
                          let v269 := (UInt256.ofNat 3)
                          let v270 := (UInt256.ofNat 4)
                          let v271 := (UInt256.ofNat 9)
                          let v272 := (UInt256.ofNat 14)
                          let v273 ← SIGMA
                          let v274 := (UInt256.ofNat 14)
                          let v275 ← checkedAdd v143 v274
                          checkIndex v275 160
                          let v276 := v273[v275.toNat]!
                          let v277 := v276
                          checkIndex v277 16
                          let v278 := r1[v277.toNat]!
                          let v279 ← SIGMA
                          let v280 := (UInt256.ofNat 15)
                          let v281 ← checkedAdd v143 v280
                          checkIndex v281 160
                          let v282 := v279[v281.toNat]!
                          let v283 := v282
                          checkIndex v283 16
                          let v284 := r1[v283.toNat]!
                          let (v285, v285_0) ← mix r2 v269 v270 v271 v272 v278 v284
                          r2 := v285_0
                          let v286 := (UInt256.ofNat 1)
                          let v287 := r3
                          let v288 ← IntTy.add (.u 64) v287 v286
                          let v289 := v288
                          r3 := v289
                          let v290 := ()
                          b23 r0 r1 r2 r3 r4 r5 r6 r7 r8
                        else
                          let v291 := (UInt256.ofNat 0)
                          let v292 := (UInt256.ofNat 8)
                          let v293 := ()
                          let v294 := (Range_Known_0_Known_8.mk v293 v293)
                          r8 := v294
                          let v296 ← Range_start_0_8 r8
                          let v297 := v296
                          let rec b26 (r0 : (Vector UInt256 8)) (r1 : (Vector UInt256 16)) (r2 : (Vector UInt256 16)) (r3 : UInt256) (r4 : Range_Known_0_Known_8) (r5 : Range_Known_0_Known_16) (r6 : Range_Known_0_Known_8) (r7 : (Vector UInt256 16)) (r8 : Range_Known_0_Known_8) (v5 : UInt256) : FeM (Vector UInt256 8) := do
                            let mut r0 := r0
                            let mut r1 := r1
                            let mut r2 := r2
                            let mut r3 := r3
                            let mut r4 := r4
                            let mut r5 := r5
                            let mut r6 := r6
                            let mut r7 := r7
                            let mut r8 := r8
                            let v299 ← Range_has_0_8 r8 v5
                            if v299 then
                              let v301 ← Range_at_0_8 r8 v5
                              let v302 := v301
                              checkIndex v302 16
                              let v303 := r2[v302.toNat]!
                              let v304 := (UInt256.ofNat 8)
                              let v305 ← checkedAdd v302 v304
                              checkIndex v305 16
                              let v306 := r2[v305.toNat]!
                              let v307 := v303 ^^^ v306
                              checkIndex v302 8
                              let v308 := r0[v302.toNat]!
                              let v309 := v308 ^^^ v307
                              checkIndex v302 8
                              r0 := (r0.set! v302.toNat v309)
                              let v310 := ()
                              let v312 ← Range_next_0_8 r8 v5
                              let v313 := v312
                              b26 r0 r1 r2 r3 r4 r5 r6 r7 r8 v313
                            else
                              let v314 := r0
                              return v314
                          partial_fixpoint
                          b26 r0 r1 r2 r3 r4 r5 r6 r7 r8 v297
                      partial_fixpoint
                      b23 r0 r1 r2 r3 r4 r5 r6 r7 r8
                  partial_fixpoint
                  b20 r0 r1 r2 r3 r4 r5 r6 r7 r8
                if v106 then
                  let v107 := (UInt256.ofNat 14)
                  checkIndex v107 16
                  let v108 := r2[v107.toNat]!
                  let v109 := IntTy.not (.u 64) v108
                  let v110 := (UInt256.ofNat 14)
                  checkIndex v110 16
                  r2 := (r2.set! v110.toNat v109)
                  let v111 := ()
                  let v112 := v111
                  b19 r0 r1 r2 r3 r4 r5 r6 r7 r8
                else
                  let v113 := ()
                  let v114 := v113
                  b19 r0 r1 r2 r3 r4 r5 r6 r7 r8
            partial_fixpoint
            b13 r0 r1 r2 r3 r4 r5 r6 r7 r8 v75
        partial_fixpoint
        b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 v51
    partial_fixpoint
    b5 r0 r1 r2 r3 r4 r5 r6 r7 r8 v30
  let b2 : (Vector UInt256 8) → (Vector UInt256 16) → (Vector UInt256 16) → UInt256 → Range_Known_0_Known_8 → Range_Known_0_Known_16 → Range_Known_0_Known_8 → (Vector UInt256 16) → Range_Known_0_Known_8 → FeM (Vector UInt256 8) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    let v15 := false
    let v16 := v15
    b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 v16
  if v8 then
    let v9 := (UInt256.ofNat 212)
    let v10 ← MemBuffer_byte_at v0 v9
    let v11 := (UInt256.ofNat 1)
    let v12 := decide (v10.toNat ≤ v11.toNat)
    if v12 then
      let v13 := true
      let v14 := v13
      b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 v14
    else
      b2 r0 r1 r2 r3 r4 r5 r6 r7 r8
  else
    b2 r0 r1 r2 r3 r4 r5 r6 r7 r8

def recv0_arm0 : FeM Empty := do
  let mut r1 : MemBuffer := default
  let mut r2 : MemBuffer := default
  let mut r3 : Range_Known_0_Known_8 := default
  let v1 ← input_Evm_1
  let v2 := v1
  r1 := v2
  let v4 ← MemBuffer_len r1
  let v5 ← INPUT_LEN
  let v6 := decide (v4 ≠ v5)
  let b1 : MemBuffer → MemBuffer → Range_Known_0_Known_8 → FeM Empty := fun r1 r2 r3 => do
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let v46 ← MemSlice_empty_u8
    let v49 ← Evm_revert v46
    nomatch v49
  if v6 then
    b1 r1 r2 r3
  else
    let v7 := (UInt256.ofNat 212)
    let v9 ← MemBuffer_index_ref r1 v7
    let v10 ← mloadInt (.u 8) v9
    let v11 := (UInt256.ofNat 1)
    let v12 := decide (v11.toNat < v10.toNat)
    if v12 then
      b1 r1 r2 r3
    else
      let v13 := ()
      let v14 := v13
      let v16 ← compress_MemBuffer r1
      let v17 := v16
      let v18 := (UInt256.ofNat 64)
      let v19 ← MemBuffer_new v18
      let v20 := v19
      r2 := v20
      let v21 := (UInt256.ofNat 0)
      let v22 := (UInt256.ofNat 8)
      let v23 := ()
      let v24 := (Range_Known_0_Known_8.mk v23 v23)
      r3 := v24
      let v26 ← Range_start_0_8 r3
      let v27 := v26
      let rec b5 (r1 : MemBuffer) (r2 : MemBuffer) (r3 : Range_Known_0_Known_8) (v0 : UInt256) : FeM Empty := do
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let v29 ← Range_has_0_8 r3 v0
        if v29 then
          let v31 ← Range_at_0_8 r3 v0
          let v32 := v31
          let v33 := (UInt256.ofNat 8)
          let v34 ← checkedMul v32 v33
          checkIndex v32 8
          let v35 := v17[v32.toNat]!
          let (v37, v37_0) ← MemBuffer_write_le_u64 r2 v34 v35
          r2 := v37_0
          let v39 ← Range_next_0_8 r3 v0
          let v40 := v39
          b5 r1 r2 r3 v40
        else
          let v42 ← MemBuffer_span_ref r2
          let v45 ← Evm_return_data v42
          nomatch v45
      partial_fixpoint
      b5 r1 r2 r3 v27

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let _ ← recv0_arm0
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv0_arm0_entry
  else
    let selector ← calldataSelector
    recv0_arm0_entry

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


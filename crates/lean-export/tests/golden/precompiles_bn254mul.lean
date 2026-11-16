export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_new MemBuffer_span_ref MemBuffer_write_word MemSlice_empty_u8 Option_array_u256_2 Option_array_u256_2.None Option_array_u256_2.Some downcast_truncate_u256_u64 downcast_unchecked_u256_usize u256_wrapping_sub)
export Shared (Mont_Bn254Base Mont_add_Bn254Base Mont_double_Bn254Base Mont_is_zero_Bn254Base Mont_mul_Bn254Base Mont_square_Bn254Base Mont_sub_Bn254Base Mont_zero_Bn254Base Point_Mont_Bn254Base Point_Mont_Bn254Base.mk Point_affine_Mont_Bn254Base Point_double_Mont_Bn254Base Point_infinity_Mont_Bn254Base Point_is_infinity_Mont_Bn254Base Point_to_affine_Mont_Bn254Base encode fp g1_point_MemBuffer word_at_MemBuffer)

def init : FeM Unit := do
  return ()

def input_Evm_3 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let v1 ← Evm_calldatasize
  let len := v1
  let v3 ← downcast_unchecked_u256_usize len
  let v4 ← MemBuffer_new v3
  let input_5 := v4
  input := input_5
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 0)
  let (v10, v10_0) ← Evm_calldatacopy input v7 v8 len
  input := v10_0
  let input_11 := input
  return input_11

def ORDER : FeM UInt256 := do
  let v0 := (UInt256.ofNat 21888242871839275222246405745257275088548364400416034343698204186575808495617)
  return v0

def GLV_G1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 52538187511802934231)
  return v0

def GLV_G2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 782660544089080853078787955015628534157)
  return v0

def GLV_A1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 9931322734385697763)
  return v0

def GLV_A2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 147946756881789319010696353538189108491)
  return v0

def magnitude (word_0 : UInt256) : FeM (UInt256 × Bool) := do
  let v2 := (UInt256.ofNat 0)
  let zero := v2
  let word_4 := word_0
  let v5 := (UInt256.ofNat 255)
  let v6 := word_4 >>> v5
  let v7 := (UInt256.ofNat 1)
  let v8 := decide (v6 = v7)
  let b3 : (UInt256 × Bool) → FeM (UInt256 × Bool) := fun v1 => do
    return v1
  if v8 then
    let word_9 := word_0
    let v10 ← u256_wrapping_sub zero word_9
    let v11 := true
    let v12 := (v10, v11)
    let v13 := v12
    b3 v13
  else
    let word_14 := word_0
    let v15 := false
    let v16 := (word_14, v15)
    let v17 := v16
    b3 v17

def GLV_B1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 147946756881789319000765030803803410728)
  return v0

def GLV_B2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 9931322734385697763)
  return v0

def BETA : FeM UInt256 := do
  let v0 := (UInt256.ofNat 2203960485148121921418603742825762020974279258880205651966)
  return v0

def Point_add_affine_Mont_Bn254Base (self : Point_Mont_Bn254Base) (x2 : Mont_Bn254Base) (y2 : Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let mut z1z1 : Mont_Bn254Base := default
  let mut u2 : Mont_Bn254Base := default
  let mut s2 : Mont_Bn254Base := default
  let mut h : Mont_Bn254Base := default
  let mut r : Mont_Bn254Base := default
  let mut hh : Mont_Bn254Base := default
  let mut i : Mont_Bn254Base := default
  let mut j : Mont_Bn254Base := default
  let mut v : Mont_Bn254Base := default
  let mut x : Mont_Bn254Base := default
  let mut r10 : Mont_Bn254Base := default
  let mut r11 : Mont_Bn254Base := default
  let mut r12 : Mont_Bn254Base := default
  let mut r13 : Mont_Bn254Base := default
  let mut r14 : Mont_Bn254Base := default
  let mut r15 : Mont_Bn254Base := default
  let mut r16 : Mont_Bn254Base := default
  let mut r17 : Mont_Bn254Base := default
  let mut r18 : Mont_Bn254Base := default
  let mut r19 : Mont_Bn254Base := default
  let mut r20 : Mont_Bn254Base := default
  let mut r21 : Mont_Bn254Base := default
  let mut r22 : Mont_Bn254Base := default
  let mut r23 : Mont_Bn254Base := default
  let mut r24 : Mont_Bn254Base := default
  let mut r25 : Mont_Bn254Base := default
  let mut r26 : Mont_Bn254Base := default
  let mut r27 : Mont_Bn254Base := default
  let mut r28 : Mont_Bn254Base := default
  let mut r29 : Mont_Bn254Base := default
  let v4 ← Point_is_infinity_Mont_Bn254Base self
  if v4 then
    let v5 ← Point_affine_Mont_Bn254Base x2 y2
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := self.z
    r10 := v8
    let v10 ← Mont_square_Bn254Base r10
    let z1z1_11 := v10
    z1z1 := z1z1_11
    let v13 ← Mont_mul_Bn254Base x2 z1z1
    let u2_14 := v13
    u2 := u2_14
    let v15 := self.z
    r11 := v15
    let v17 ← Mont_mul_Bn254Base y2 r11
    r12 := v17
    let v20 ← Mont_mul_Bn254Base r12 z1z1
    let s2_21 := v20
    s2 := s2_21
    let v22 := self.x
    r13 := v22
    let v25 ← Mont_sub_Bn254Base u2 r13
    let h_26 := v25
    h := h_26
    let v27 := self.y
    r14 := v27
    let v30 ← Mont_sub_Bn254Base s2 r14
    r15 := v30
    let v32 ← Mont_double_Bn254Base r15
    let r_33 := v32
    r := r_33
    let v35 ← Mont_is_zero_Bn254Base h
    if v35 then
      let v37 ← Mont_is_zero_Bn254Base r
      let b9 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v3 => do
        let mut z1z1 := z1z1
        let mut u2 := u2
        let mut s2 := s2
        let mut h := h
        let mut r := r
        let mut hh := hh
        let mut i := i
        let mut j := j
        let mut v := v
        let mut x := x
        let mut r10 := r10
        let mut r11 := r11
        let mut r12 := r12
        let mut r13 := r13
        let mut r14 := r14
        let mut r15 := r15
        let mut r16 := r16
        let mut r17 := r17
        let mut r18 := r18
        let mut r19 := r19
        let mut r20 := r20
        let mut r21 := r21
        let mut r22 := r22
        let mut r23 := r23
        let mut r24 := r24
        let mut r25 := r25
        let mut r26 := r26
        let mut r27 := r27
        let mut r28 := r28
        let mut r29 := r29
        return v3
      if v37 then
        let v38 ← Point_double_Mont_Bn254Base self
        let v39 := v38
        b9 z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v39
      else
        let v40 ← Point_infinity_Mont_Bn254Base
        let v41 := v40
        b9 z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v41
    else
      let v42 := ()
      let v43 := v42
      let v45 ← Mont_square_Bn254Base h
      let hh_46 := v45
      hh := hh_46
      let v48 ← Mont_double_Bn254Base hh
      r16 := v48
      let v50 ← Mont_double_Bn254Base r16
      let i_51 := v50
      i := i_51
      let v54 ← Mont_mul_Bn254Base h i
      let j_55 := v54
      j := j_55
      let v56 := self.x
      r17 := v56
      let v59 ← Mont_mul_Bn254Base r17 i
      let v_60 := v59
      v := v_60
      let v62 ← Mont_square_Bn254Base r
      r18 := v62
      let v65 ← Mont_sub_Bn254Base r18 j
      r19 := v65
      let v67 ← Mont_double_Bn254Base v
      r20 := v67
      let v70 ← Mont_sub_Bn254Base r19 r20
      let x_71 := v70
      x := x_71
      let v74 ← Mont_sub_Bn254Base v x
      r21 := v74
      let v77 ← Mont_mul_Bn254Base r r21
      r22 := v77
      let v78 := self.y
      r23 := v78
      let v81 ← Mont_mul_Bn254Base r23 j
      r24 := v81
      let v83 ← Mont_double_Bn254Base r24
      r25 := v83
      let v86 ← Mont_sub_Bn254Base r22 r25
      let y := v86
      let v88 := self.z
      r26 := v88
      let v91 ← Mont_add_Bn254Base r26 h
      r27 := v91
      let v93 ← Mont_square_Bn254Base r27
      r28 := v93
      let v96 ← Mont_sub_Bn254Base r28 z1z1
      r29 := v96
      let v99 ← Mont_sub_Bn254Base r29 hh
      let x_100 := x
      let v101 := (Point_Mont_Bn254Base.mk x_100 y v99)
      return v101

def Mont_neg_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let v1 ← Mont_zero_Bn254Base
  r0 := v1
  let v3 ← Mont_sub_Bn254Base r0 self
  return v3

def multiply_g1_k1 (point : Point_Mont_Bn254Base) (c1 : UInt256) (c2 : UInt256) (v29 : (UInt256 × Bool)) (x1 : Mont_Bn254Base) (y1 : Mont_Bn254Base) (x2 : Mont_Bn254Base) (y2 : Mont_Bn254Base) (x3 : Mont_Bn254Base) (y3 : Mont_Bn254Base) (result : Point_Mont_Bn254Base) (bit : UInt256) (r8 : Mont_Bn254Base) (r9 : Mont_Bn254Base) (r10 : Mont_Bn254Base) (r11 : Mont_Bn254Base) (r12 : Point_Mont_Bn254Base) (r13 : Point_Mont_Bn254Base) (r14 : Point_Mont_Bn254Base) (r15 : Point_Mont_Bn254Base) (r16 : Point_Mont_Bn254Base) (r17 : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let mut x1 := x1
  let mut y1 := y1
  let mut x2 := x2
  let mut y2 := y2
  let mut x3 := x3
  let mut y3 := y3
  let mut result := result
  let mut bit := bit
  let mut r8 := r8
  let mut r9 := r9
  let mut r10 := r10
  let mut r11 := r11
  let mut r12 := r12
  let mut r13 := r13
  let mut r14 := r14
  let mut r15 := r15
  let mut r16 := r16
  let mut r17 := r17
  let v30 := v29.fst
  let k1 := v30
  let v32 := v29.snd
  let negative1 := v32
  let v34 ← GLV_B1
  let v35 ← checkedMul c1 v34
  let v36 ← GLV_B2
  let v37 ← checkedMul c2 v36
  let v38 ← u256_wrapping_sub v35 v37
  let v39 ← magnitude v38
  let v40 := v39.fst
  let k2 := v40
  let v42 := v39.snd
  let negative2 := v42
  let v44 := point.x
  let x1_45 := v44
  x1 := x1_45
  let b3 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v2 => do
    let mut x1 := x1
    let mut y1 := y1
    let mut x2 := x2
    let mut y2 := y2
    let mut x3 := x3
    let mut y3 := y3
    let mut result := result
    let mut bit := bit
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut r11 := r11
    let mut r12 := r12
    let mut r13 := r13
    let mut r14 := r14
    let mut r15 := r15
    let mut r16 := r16
    let mut r17 := r17
    let y1_52 := v2
    y1 := y1_52
    let v53 ← BETA
    let v54 ← fp v53
    r9 := v54
    let v55 := point.x
    r10 := v55
    let v58 ← Mont_mul_Bn254Base r10 r9
    let x2_59 := v58
    x2 := x2_59
    let b6 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v3 => do
      let mut x1 := x1
      let mut y1 := y1
      let mut x2 := x2
      let mut y2 := y2
      let mut x3 := x3
      let mut y3 := y3
      let mut result := result
      let mut bit := bit
      let mut r8 := r8
      let mut r9 := r9
      let mut r10 := r10
      let mut r11 := r11
      let mut r12 := r12
      let mut r13 := r13
      let mut r14 := r14
      let mut r15 := r15
      let mut r16 := r16
      let mut r17 := r17
      let y2_66 := v3
      y2 := y2_66
      let v69 ← Point_affine_Mont_Bn254Base x1 y1
      r12 := v69
      let v73 ← Point_add_affine_Mont_Bn254Base r12 x2 y2
      r13 := v73
      let v75 ← Point_to_affine_Mont_Bn254Base r13
      let v76 := v75.fst
      let x3_77 := v76
      x3 := x3_77
      let v78 := v75.snd
      let y3_79 := v78
      y3 := y3_79
      let v80 ← Point_infinity_Mont_Bn254Base
      let result_81 := v80
      result := result_81
      let v82 := (UInt256.ofNat 256)
      let v83 := k1 ||| k2
      let v84 := IntTy.leadingZeros (.u 256) v83
      let v85 ← checkedSub v82 v84
      let v86 ← downcast_truncate_u256_u64 v85
      let bit_87 := v86
      bit := bit_87
      let rec b7 (x1 : Mont_Bn254Base) (y1 : Mont_Bn254Base) (x2 : Mont_Bn254Base) (y2 : Mont_Bn254Base) (x3 : Mont_Bn254Base) (y3 : Mont_Bn254Base) (result : Point_Mont_Bn254Base) (bit : UInt256) (r8 : Mont_Bn254Base) (r9 : Mont_Bn254Base) (r10 : Mont_Bn254Base) (r11 : Mont_Bn254Base) (r12 : Point_Mont_Bn254Base) (r13 : Point_Mont_Bn254Base) (r14 : Point_Mont_Bn254Base) (r15 : Point_Mont_Bn254Base) (r16 : Point_Mont_Bn254Base) (r17 : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
        let mut x1 := x1
        let mut y1 := y1
        let mut x2 := x2
        let mut y2 := y2
        let mut x3 := x3
        let mut y3 := y3
        let mut result := result
        let mut bit := bit
        let mut r8 := r8
        let mut r9 := r9
        let mut r10 := r10
        let mut r11 := r11
        let mut r12 := r12
        let mut r13 := r13
        let mut r14 := r14
        let mut r15 := r15
        let mut r16 := r16
        let mut r17 := r17
        let bit_88 := bit
        let bit_89 := bit_88
        let v90 := (UInt256.ofNat 0)
        let v91 := decide (v90.toNat < bit_89.toNat)
        if v91 then
          let v92 := (UInt256.ofNat 1)
          let bit_93 := bit
          let v94 ← IntTy.sub (.u 64) bit_93 v92
          let bit_95 := v94
          bit := bit_95
          let v96 := ()
          let result_97 := result
          let result_98 := result_97
          r14 := result_98
          let v100 ← Point_double_Mont_Bn254Base r14
          let result_101 := v100
          result := result_101
          let v102 := ()
          let bit_103 := bit
          let v104 := bit_103
          let v105 := k1 >>> v104
          let v106 := (UInt256.ofNat 1)
          let v107 := v105 &&& v106
          let v108 := (UInt256.ofNat 1)
          let v109 := decide (v107 = v108)
          let in1 := v109
          let bit_111 := bit
          let v112 := bit_111
          let v113 := k2 >>> v112
          let v114 := (UInt256.ofNat 1)
          let v115 := v113 &&& v114
          let v116 := (UInt256.ofNat 1)
          let v117 := decide (v115 = v116)
          let in2 := v117
          let b12 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 => do
            let mut x1 := x1
            let mut y1 := y1
            let mut x2 := x2
            let mut y2 := y2
            let mut x3 := x3
            let mut y3 := y3
            let mut result := result
            let mut bit := bit
            let mut r8 := r8
            let mut r9 := r9
            let mut r10 := r10
            let mut r11 := r11
            let mut r12 := r12
            let mut r13 := r13
            let mut r14 := r14
            let mut r15 := r15
            let mut r16 := r16
            let mut r17 := r17
            b7 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
          let b11 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 => do
            let mut x1 := x1
            let mut y1 := y1
            let mut x2 := x2
            let mut y2 := y2
            let mut x3 := x3
            let mut y3 := y3
            let mut result := result
            let mut bit := bit
            let mut r8 := r8
            let mut r9 := r9
            let mut r10 := r10
            let mut r11 := r11
            let mut r12 := r12
            let mut r13 := r13
            let mut r14 := r14
            let mut r15 := r15
            let mut r16 := r16
            let mut r17 := r17
            let b16 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Unit → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v4 => do
              let mut x1 := x1
              let mut y1 := y1
              let mut x2 := x2
              let mut y2 := y2
              let mut x3 := x3
              let mut y3 := y3
              let mut result := result
              let mut bit := bit
              let mut r8 := r8
              let mut r9 := r9
              let mut r10 := r10
              let mut r11 := r11
              let mut r12 := r12
              let mut r13 := r13
              let mut r14 := r14
              let mut r15 := r15
              let mut r16 := r16
              let mut r17 := r17
              let v149 := v4
              b12 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
            if in1 then
              let result_128 := result
              let result_129 := result_128
              r16 := result_129
              let v133 ← Point_add_affine_Mont_Bn254Base r16 x1 y1
              let result_134 := v133
              result := result_134
              let v135 := ()
              let v136 := v135
              b16 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v136
            else
              let b19 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Unit → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v5 => do
                let mut x1 := x1
                let mut y1 := y1
                let mut x2 := x2
                let mut y2 := y2
                let mut x3 := x3
                let mut y3 := y3
                let mut result := result
                let mut bit := bit
                let mut r8 := r8
                let mut r9 := r9
                let mut r10 := r10
                let mut r11 := r11
                let mut r12 := r12
                let mut r13 := r13
                let mut r14 := r14
                let mut r15 := r15
                let mut r16 := r16
                let mut r17 := r17
                let v148 := v5
                b16 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v148
              if in2 then
                let result_137 := result
                let result_138 := result_137
                r17 := result_138
                let v142 ← Point_add_affine_Mont_Bn254Base r17 x2 y2
                let result_143 := v142
                result := result_143
                let v144 := ()
                let v145 := v144
                b19 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v145
              else
                let v146 := ()
                let v147 := v146
                b19 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v147
          if in1 then
            if in2 then
              let result_119 := result
              let result_120 := result_119
              r15 := result_120
              let v124 ← Point_add_affine_Mont_Bn254Base r15 x3 y3
              let result_125 := v124
              result := result_125
              let v126 := ()
              let v127 := v126
              b12 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
            else
              b11 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
          else
            b11 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
        else
          let result_150 := result
          return result_150
      partial_fixpoint
      b7 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
    if negative2 then
      let v60 := point.y
      r11 := v60
      let v62 ← Mont_neg_Bn254Base r11
      let v63 := v62
      b6 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v63
    else
      let v64 := point.y
      let v65 := v64
      b6 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v65
  if negative1 then
    let v46 := point.y
    r8 := v46
    let v48 ← Mont_neg_Bn254Base r8
    let v49 := v48
    b3 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v49
  else
    let v50 := point.y
    let v51 := v50
    b3 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v51

def multiply_g1 (point : Point_Mont_Bn254Base) (scalar_1 : UInt256) : FeM Point_Mont_Bn254Base := do
  let mut x1 : Mont_Bn254Base := default
  let mut y1 : Mont_Bn254Base := default
  let mut x2 : Mont_Bn254Base := default
  let mut y2 : Mont_Bn254Base := default
  let mut x3 : Mont_Bn254Base := default
  let mut y3 : Mont_Bn254Base := default
  let mut result : Point_Mont_Bn254Base := default
  let mut bit : UInt256 := default
  let mut r8 : Mont_Bn254Base := default
  let mut r9 : Mont_Bn254Base := default
  let mut r10 : Mont_Bn254Base := default
  let mut r11 : Mont_Bn254Base := default
  let mut r12 : Point_Mont_Bn254Base := default
  let mut r13 : Point_Mont_Bn254Base := default
  let mut r14 : Point_Mont_Bn254Base := default
  let mut r15 : Point_Mont_Bn254Base := default
  let mut r16 : Point_Mont_Bn254Base := default
  let mut r17 : Point_Mont_Bn254Base := default
  let scalar_6 := scalar_1
  let v7 ← ORDER
  let v8 ← checkedRem scalar_6 v7
  let k := v8
  let v10 := (UInt256.ofNat 128)
  let v11 := k >>> v10
  let high := v11
  let v13 ← GLV_G1
  let v14 ← checkedMul high v13
  let v15 := (UInt256.ofNat 128)
  let v16 := v14 >>> v15
  let c1 := v16
  let v18 ← GLV_G2
  let v19 ← checkedMul high v18
  let v20 := (UInt256.ofNat 128)
  let v21 := v19 >>> v20
  let c2 := v21
  let v23 ← GLV_A1
  let v24 ← checkedMul c1 v23
  let v25 ← u256_wrapping_sub k v24
  let v26 ← GLV_A2
  let v27 ← checkedMul c2 v26
  let v28 ← u256_wrapping_sub v25 v27
  let v29 ← magnitude v28
  multiply_g1_k1 point c1 c2 v29 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17

def mul_MemBuffer (input : MemBuffer) : FeM Option_array_u256_2 := do
  let mut point : Point_Mont_Bn254Base := default
  let mut r1 : Point_Mont_Bn254Base := default
  let v2 := (UInt256.ofNat 0)
  let v3 ← g1_point_MemBuffer input v2
  let b1 : Point_Mont_Bn254Base → Point_Mont_Bn254Base → Option_array_u256_2 → FeM Option_array_u256_2 := fun point r1 v1 => do
    let mut point := point
    let mut r1 := r1
    return v1
  match v3 with
  | .Some .. =>
    let v4 := v3.Some_0
    let point_5 := v4
    point := point_5
    let v6 := (UInt256.ofNat 64)
    let v7 ← word_at_MemBuffer input v6
    let scalar := v7
    let v10 ← Point_is_infinity_Mont_Bn254Base point
    let b4 : Point_Mont_Bn254Base → Point_Mont_Bn254Base → FeM Option_array_u256_2 := fun point r1 => do
      let mut point := point
      let mut r1 := r1
      let v21 := (UInt256.ofNat 0)
      let v22 := (UInt256.ofNat 0)
      let v23 := #v[v21, v22]
      let v24 := Option_array_u256_2.Some v23
      return v24
    if v10 then
      b4 point r1
    else
      let v11 := (UInt256.ofNat 0)
      let v12 := decide (scalar = v11)
      if v12 then
        b4 point r1
      else
        let v13 := ()
        let v14 := v13
        let v16 ← multiply_g1 point scalar
        r1 := v16
        let v18 ← encode r1
        let v19 := Option_array_u256_2.Some v18
        let v20 := v19
        b1 point r1 v20
  | .None =>
    let v25 := Option_array_u256_2.None
    let v26 := v25
    b1 point r1 v26

def recv0_arm0 : FeM Empty := do
  let mut output : MemBuffer := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_3
  r2 := v0
  let v2 ← mul_MemBuffer r2
  match v2 with
  | .Some .. =>
    let v3 := v2.Some_0
    let product := v3
    let v5 := (UInt256.ofNat 64)
    let v6 ← MemBuffer_new v5
    let output_7 := v6
    output := output_7
    let v8 := (UInt256.ofNat 0)
    let v9 := (UInt256.ofNat 0)
    checkIndex v9 2
    let v10 := product[v9.toNat]!
    let (v12, v12_0) ← MemBuffer_write_word output v8 v10
    output := v12_0
    let v13 := (UInt256.ofNat 32)
    let v14 := (UInt256.ofNat 1)
    checkIndex v14 2
    let v15 := product[v14.toNat]!
    let (v17, v17_0) ← MemBuffer_write_word output v13 v15
    output := v17_0
    let v19 ← MemBuffer_span_ref output
    let v22 ← Evm_return_data v19
    nomatch v22
  | .None =>
    let v23 ← MemSlice_empty_u8
    let v26 ← Evm_revert v23
    nomatch v26

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


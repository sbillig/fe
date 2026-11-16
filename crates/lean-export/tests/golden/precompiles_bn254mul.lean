export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_new MemBuffer_span_ref MemBuffer_write_word MemSlice_empty_u8 Option_array_u256_2 Option_array_u256_2.None Option_array_u256_2.Some downcast_truncate_u256_u64 downcast_unchecked_u256_usize u256_wrapping_sub)
export Shared (Mont_Bn254Base Mont_add_Bn254Base Mont_double_Bn254Base Mont_is_zero_Bn254Base Mont_mul_Bn254Base Mont_square_Bn254Base Mont_sub_Bn254Base Mont_zero_Bn254Base Point_Mont_Bn254Base Point_Mont_Bn254Base.mk Point_affine_Mont_Bn254Base Point_double_Mont_Bn254Base Point_infinity_Mont_Bn254Base Point_is_infinity_Mont_Bn254Base Point_to_affine_Mont_Bn254Base encode fp g1_point_MemBuffer word_at_MemBuffer)

/-- The `init` of contract `Bn254Mul`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_3 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let len ← Evm_calldatasize
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  let input_11 := input
  return input_11

/-- `lib::bn254::ORDER` -/
def ORDER : FeM UInt256 := do
  return UInt256.ofNat 21888242871839275222246405745257275088548364400416034343698204186575808495617

/-- `lib::bn254::GLV_G1` -/
def GLV_G1 : FeM UInt256 := do
  return UInt256.ofNat 52538187511802934231

/-- `lib::bn254::GLV_G2` -/
def GLV_G2 : FeM UInt256 := do
  return UInt256.ofNat 782660544089080853078787955015628534157

/-- `lib::bn254::GLV_A1` -/
def GLV_A1 : FeM UInt256 := do
  return UInt256.ofNat 9931322734385697763

/-- `lib::bn254::GLV_A2` -/
def GLV_A2 : FeM UInt256 := do
  return UInt256.ofNat 147946756881789319010696353538189108491

/-- `lib::bn254::magnitude` -/
def magnitude (word : UInt256) : FeM (UInt256 × Bool) := do
  let v8 := decide ((word >>> (UInt256.ofNat 255)) = (UInt256.ofNat 1))
  let join : (UInt256 × Bool) → FeM (UInt256 × Bool) := fun v1 => do
    return v1
  if v8 then
    let v10 ← u256_wrapping_sub (UInt256.ofNat 0) word
    join (v10, true)
  else
    join (word, false)

/-- `lib::bn254::GLV_B1` -/
def GLV_B1 : FeM UInt256 := do
  return UInt256.ofNat 147946756881789319000765030803803410728

/-- `lib::bn254::GLV_B2` -/
def GLV_B2 : FeM UInt256 := do
  return UInt256.ofNat 9931322734385697763

/-- `lib::bn254::BETA` -/
def BETA : FeM UInt256 := do
  return UInt256.ofNat 2203960485148121921418603742825762020974279258880205651966

/-- `Point<Mont<Bn254Base>>::add_affine` -/
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
    r10 := self.z
    let z1z1_10 ← Mont_square_Bn254Base r10
    z1z1 := z1z1_10
    let u2_13 ← Mont_mul_Bn254Base x2 z1z1
    u2 := u2_13
    r11 := self.z
    let v17 ← Mont_mul_Bn254Base y2 r11
    r12 := v17
    let s2_20 ← Mont_mul_Bn254Base r12 z1z1
    s2 := s2_20
    r13 := self.x
    let h_25 ← Mont_sub_Bn254Base u2 r13
    h := h_25
    r14 := self.y
    let v30 ← Mont_sub_Bn254Base s2 r14
    r15 := v30
    let r_32 ← Mont_double_Bn254Base r15
    r := r_32
    let v35 ← Mont_is_zero_Bn254Base h
    if v35 then
      let v37 ← Mont_is_zero_Bn254Base r
      let join : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v3 => do
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
        join z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v38
      else
        let v40 ← Point_infinity_Mont_Bn254Base
        join z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v40
    else
      let hh_45 ← Mont_square_Bn254Base h
      hh := hh_45
      let v48 ← Mont_double_Bn254Base hh
      r16 := v48
      let i_50 ← Mont_double_Bn254Base r16
      i := i_50
      let j_54 ← Mont_mul_Bn254Base h i
      j := j_54
      r17 := self.x
      let v_59 ← Mont_mul_Bn254Base r17 i
      v := v_59
      let v62 ← Mont_square_Bn254Base r
      r18 := v62
      let v65 ← Mont_sub_Bn254Base r18 j
      r19 := v65
      let v67 ← Mont_double_Bn254Base v
      r20 := v67
      let x_70 ← Mont_sub_Bn254Base r19 r20
      x := x_70
      let v74 ← Mont_sub_Bn254Base v x
      r21 := v74
      let v77 ← Mont_mul_Bn254Base r r21
      r22 := v77
      r23 := self.y
      let v81 ← Mont_mul_Bn254Base r23 j
      r24 := v81
      let v83 ← Mont_double_Bn254Base r24
      r25 := v83
      let y ← Mont_sub_Bn254Base r22 r25
      r26 := self.z
      let v91 ← Mont_add_Bn254Base r26 h
      r27 := v91
      let v93 ← Mont_square_Bn254Base r27
      r28 := v93
      let v96 ← Mont_sub_Bn254Base r28 z1z1
      r29 := v96
      let v99 ← Mont_sub_Bn254Base r29 hh
      let x_100 := x
      return Point_Mont_Bn254Base.mk x_100 y v99

/-- `<Mont<Bn254Base> as Field>::neg` -/
def Mont_neg_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let v1 ← Mont_zero_Bn254Base
  r0 := v1
  let v3 ← Mont_sub_Bn254Base r0 self
  return v3

/-- `lib::bn254::multiply_g1` -/
def multiply_g1 (point : Point_Mont_Bn254Base) (scalar : UInt256) : FeM Point_Mont_Bn254Base := do
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
  let v7 ← ORDER
  let k ← checkedRem scalar v7
  let high := k >>> (UInt256.ofNat 128)
  let v13 ← GLV_G1
  let v14 ← checkedMul high v13
  let c1 := v14 >>> (UInt256.ofNat 128)
  let v18 ← GLV_G2
  let v19 ← checkedMul high v18
  let c2 := v19 >>> (UInt256.ofNat 128)
  let v23 ← GLV_A1
  let v24 ← checkedMul c1 v23
  let v25 ← u256_wrapping_sub k v24
  let v26 ← GLV_A2
  let v27 ← checkedMul c2 v26
  let v28 ← u256_wrapping_sub v25 v27
  let v29 ← magnitude v28
  let k1 := v29.fst
  let v34 ← GLV_B1
  let v35 ← checkedMul c1 v34
  let v36 ← GLV_B2
  let v37 ← checkedMul c2 v36
  let v38 ← u256_wrapping_sub v35 v37
  let v39 ← magnitude v38
  let k2 := v39.fst
  let negative2 := v39.snd
  x1 := point.x
  let join : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 y1_2 => do
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
    y1 := y1_2
    let v53 ← BETA
    let v54 ← fp v53
    r9 := v54
    r10 := point.x
    let x2_58 ← Mont_mul_Bn254Base r10 r9
    x2 := x2_58
    let join_1 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 y2_3 => do
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
      y2 := y2_3
      let v69 ← Point_affine_Mont_Bn254Base x1 y1
      r12 := v69
      let v73 ← Point_add_affine_Mont_Bn254Base r12 x2 y2
      r13 := v73
      let v75 ← Point_to_affine_Mont_Bn254Base r13
      x3 := v75.fst
      y3 := v75.snd
      let result_80 ← Point_infinity_Mont_Bn254Base
      result := result_80
      let v85 ← checkedSub (UInt256.ofNat 256) (IntTy.leadingZeros (.u 256) (k1 ||| k2))
      let bit_86 ← downcast_truncate_u256_u64 v85
      bit := bit_86
      let rec loop (x1 : Mont_Bn254Base) (y1 : Mont_Bn254Base) (x2 : Mont_Bn254Base) (y2 : Mont_Bn254Base) (x3 : Mont_Bn254Base) (y3 : Mont_Bn254Base) (result : Point_Mont_Bn254Base) (bit : UInt256) (r8 : Mont_Bn254Base) (r9 : Mont_Bn254Base) (r10 : Mont_Bn254Base) (r11 : Mont_Bn254Base) (r12 : Point_Mont_Bn254Base) (r13 : Point_Mont_Bn254Base) (r14 : Point_Mont_Bn254Base) (r15 : Point_Mont_Bn254Base) (r16 : Point_Mont_Bn254Base) (r17 : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
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
        if decide ((UInt256.ofNat 0).toNat < bit_88.toNat) then
          let bit_93 := bit
          let bit_94 ← IntTy.sub (.u 64) bit_93 (UInt256.ofNat 1)
          bit := bit_94
          let result_97 := result
          r14 := result_97
          let result_100 ← Point_double_Mont_Bn254Base r14
          result := result_100
          let bit_103 := bit
          let in1 := decide (((k1 >>> bit_103) &&& (UInt256.ofNat 1)) = (UInt256.ofNat 1))
          let bit_111 := bit
          let in2 := decide (((k2 >>> bit_111) &&& (UInt256.ofNat 1)) = (UInt256.ofNat 1))
          let join_2 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 => do
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
            loop x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
          let join_3 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 => do
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
            let join_4 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Unit → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v4 => do
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
              join_2 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
            if in1 then
              let result_128 := result
              r16 := result_128
              let result_133 ← Point_add_affine_Mont_Bn254Base r16 x1 y1
              result := result_133
              join_4 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 ()
            else
              let join_5 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → UInt256 → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Unit → FeM Point_Mont_Bn254Base := fun x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v5 => do
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
                join_4 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v5
              if in2 then
                let result_137 := result
                r17 := result_137
                let result_142 ← Point_add_affine_Mont_Bn254Base r17 x2 y2
                result := result_142
                join_5 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 ()
              else
                join_5 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 ()
          if in1 then
            if in2 then
              let result_119 := result
              r15 := result_119
              let result_124 ← Point_add_affine_Mont_Bn254Base r15 x3 y3
              result := result_124
              join_2 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
            else
              join_3 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
          else
            join_3 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
        else
          let result_150 := result
          return result_150
      partial_fixpoint
      loop x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17
    if negative2 then
      r11 := point.y
      let v62 ← Mont_neg_Bn254Base r11
      join_1 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v62
    else
      join_1 x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 point.y
  if v29.snd then
    r8 := point.y
    let v48 ← Mont_neg_Bn254Base r8
    join x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 v48
  else
    join x1 y1 x2 y2 x3 y3 result bit r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 point.y

/-- `lib::bn254::mul<MemBuffer>` -/
def mul_MemBuffer (input : MemBuffer) : FeM Option_array_u256_2 := do
  let mut point : Point_Mont_Bn254Base := default
  let mut r1 : Point_Mont_Bn254Base := default
  let v3 ← g1_point_MemBuffer input (UInt256.ofNat 0)
  let join : Point_Mont_Bn254Base → Point_Mont_Bn254Base → Option_array_u256_2 → FeM Option_array_u256_2 := fun point r1 v1 => do
    let mut point := point
    let mut r1 := r1
    return v1
  match v3 with
  | .Some .. =>
    point := v3.Some_0
    let scalar ← word_at_MemBuffer input (UInt256.ofNat 64)
    let v10 ← Point_is_infinity_Mont_Bn254Base point
    let join_1 : Point_Mont_Bn254Base → Point_Mont_Bn254Base → FeM Option_array_u256_2 := fun point r1 => do
      let mut point := point
      let mut r1 := r1
      let v24 := Option_array_u256_2.Some (#v[(UInt256.ofNat 0), (UInt256.ofNat 0)])
      return v24
    if v10 then
      join_1 point r1
    else
      if decide (scalar = (UInt256.ofNat 0)) then
        join_1 point r1
      else
        let v16 ← multiply_g1 point scalar
        r1 := v16
        let v18 ← encode r1
        join point r1 (Option_array_u256_2.Some v18)
  | .None =>
    join point r1 Option_array_u256_2.None

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Empty := do
  let mut output : MemBuffer := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_3
  r2 := v0
  let v2 ← mul_MemBuffer r2
  match v2 with
  | .Some .. =>
    let product := v2.Some_0
    let output_6 ← MemBuffer_new (UInt256.ofNat 64)
    output := output_6
    let v9 := (UInt256.ofNat 0)
    checkIndex v9 2
    let (v12, v12_0) ← MemBuffer_write_word output (UInt256.ofNat 0) product[v9.toNat]!
    output := v12_0
    let v14 := (UInt256.ofNat 1)
    checkIndex v14 2
    let (v17, v17_0) ← MemBuffer_write_word output (UInt256.ofNat 32) product[v14.toNat]!
    output := v17_0
    let v19 ← MemBuffer_span_ref output
    let v22 ← Evm_return_data v19
    nomatch v22
  | .None =>
    let v23 ← MemSlice_empty_u8
    let v26 ← Evm_revert v23
    nomatch v26

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  nonPayable
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `Bn254Mul`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
  else
    let selector ← calldataSelector
    recv_fallback_entry

/-- The deployment entry of contract `Bn254Mul`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


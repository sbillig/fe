export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_new MemBuffer_span_ref MemBuffer_write_word MemSlice_empty_u8 Option_array_u256_2 Option_array_u256_2.None Option_array_u256_2.Some downcast_unchecked_u256_usize)
export Shared (Mont_Bn254Base Mont_add_Bn254Base Mont_double_Bn254Base Mont_is_zero_Bn254Base Mont_mul_Bn254Base Mont_square_Bn254Base Mont_sub_Bn254Base Point_Mont_Bn254Base Point_Mont_Bn254Base.mk Point_double_Mont_Bn254Base Point_infinity_Mont_Bn254Base Point_is_infinity_Mont_Bn254Base encode g1_point_MemBuffer)

def init : FeM Unit := do
  return ()

def input_Evm : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let len ← Evm_calldatasize
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  let input_11 := input
  return input_11

def Point_add_Mont_Bn254Base (self : Point_Mont_Bn254Base) (other : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let mut z1z1 : Mont_Bn254Base := default
  let mut z2z2 : Mont_Bn254Base := default
  let mut u1 : Mont_Bn254Base := default
  let mut u2 : Mont_Bn254Base := default
  let mut s1 : Mont_Bn254Base := default
  let mut s2 : Mont_Bn254Base := default
  let mut h : Mont_Bn254Base := default
  let mut r : Mont_Bn254Base := default
  let mut i : Mont_Bn254Base := default
  let mut j : Mont_Bn254Base := default
  let mut v : Mont_Bn254Base := default
  let mut x : Mont_Bn254Base := default
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
  let mut r30 : Mont_Bn254Base := default
  let mut r31 : Mont_Bn254Base := default
  let mut r32 : Mont_Bn254Base := default
  let mut r33 : Mont_Bn254Base := default
  let mut r34 : Mont_Bn254Base := default
  let mut r35 : Mont_Bn254Base := default
  let mut r36 : Mont_Bn254Base := default
  let v3 ← Point_is_infinity_Mont_Bn254Base self
  if v3 then
    let v4 := other
    return v4
  else
    let v7 ← Point_is_infinity_Mont_Bn254Base other
    if v7 then
      let v8 := self
      return v8
    else
      let v11 := self.z
      r12 := v11
      let z1z1_13 ← Mont_square_Bn254Base r12
      z1z1 := z1z1_13
      let v15 := other.z
      r13 := v15
      let z2z2_17 ← Mont_square_Bn254Base r13
      z2z2 := z2z2_17
      let v19 := self.x
      r14 := v19
      let u1_22 ← Mont_mul_Bn254Base r14 z2z2
      u1 := u1_22
      let v24 := other.x
      r15 := v24
      let u2_27 ← Mont_mul_Bn254Base r15 z1z1
      u2 := u2_27
      let v29 := other.z
      r16 := v29
      let v30 := self.y
      r17 := v30
      let v33 ← Mont_mul_Bn254Base r17 r16
      r18 := v33
      let s1_36 ← Mont_mul_Bn254Base r18 z2z2
      s1 := s1_36
      let v38 := self.z
      r19 := v38
      let v39 := other.y
      r20 := v39
      let v42 ← Mont_mul_Bn254Base r20 r19
      r21 := v42
      let s2_45 ← Mont_mul_Bn254Base r21 z1z1
      s2 := s2_45
      let h_49 ← Mont_sub_Bn254Base u2 u1
      h := h_49
      let v53 ← Mont_sub_Bn254Base s2 s1
      r22 := v53
      let r_55 ← Mont_double_Bn254Base r22
      r := r_55
      let v58 ← Mont_is_zero_Bn254Base h
      if v58 then
        let v60 ← Mont_is_zero_Bn254Base r
        let b12 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v2 => do
          let mut z1z1 := z1z1
          let mut z2z2 := z2z2
          let mut u1 := u1
          let mut u2 := u2
          let mut s1 := s1
          let mut s2 := s2
          let mut h := h
          let mut r := r
          let mut i := i
          let mut j := j
          let mut v := v
          let mut x := x
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
          let mut r30 := r30
          let mut r31 := r31
          let mut r32 := r32
          let mut r33 := r33
          let mut r34 := r34
          let mut r35 := r35
          let mut r36 := r36
          return v2
        if v60 then
          let v61 ← Point_double_Mont_Bn254Base self
          b12 z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v61
        else
          let v63 ← Point_infinity_Mont_Bn254Base
          b12 z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v63
      else
        let v68 ← Mont_double_Bn254Base h
        r23 := v68
        let i_70 ← Mont_square_Bn254Base r23
        i := i_70
        let j_74 ← Mont_mul_Bn254Base h i
        j := j_74
        let v_78 ← Mont_mul_Bn254Base u1 i
        v := v_78
        let v81 ← Mont_square_Bn254Base r
        r24 := v81
        let v84 ← Mont_sub_Bn254Base r24 j
        r25 := v84
        let v86 ← Mont_double_Bn254Base v
        r26 := v86
        let x_89 ← Mont_sub_Bn254Base r25 r26
        x := x_89
        let v93 ← Mont_sub_Bn254Base v x
        r27 := v93
        let v96 ← Mont_mul_Bn254Base r r27
        r28 := v96
        let v99 ← Mont_mul_Bn254Base s1 j
        r29 := v99
        let v101 ← Mont_double_Bn254Base r29
        r30 := v101
        let y ← Mont_sub_Bn254Base r28 r30
        let v106 := other.z
        r31 := v106
        let v107 := self.z
        r32 := v107
        let v110 ← Mont_add_Bn254Base r32 r31
        r33 := v110
        let v112 ← Mont_square_Bn254Base r33
        r34 := v112
        let v115 ← Mont_sub_Bn254Base r34 z1z1
        r35 := v115
        let v118 ← Mont_sub_Bn254Base r35 z2z2
        r36 := v118
        let v121 ← Mont_mul_Bn254Base r36 h
        let x_122 := x
        let v123 := (Point_Mont_Bn254Base.mk x_122 y v121)
        return v123

def add_MemBuffer (input : MemBuffer) : FeM Option_array_u256_2 := do
  let mut a : Point_Mont_Bn254Base := default
  let mut b : Point_Mont_Bn254Base := default
  let mut r2 : Point_Mont_Bn254Base := default
  let v3 ← g1_point_MemBuffer input (UInt256.ofNat 0)
  let v5 ← g1_point_MemBuffer input (UInt256.ofNat 64)
  let v6 := (v3, v5)
  let v7 := v6.snd
  let b1 : Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Option_array_u256_2 → FeM Option_array_u256_2 := fun a b r2 v1 => do
    let mut a := a
    let mut b := b
    let mut r2 := r2
    return v1
  match v7 with
  | .Some .. =>
    let v8 := v6.fst
    match v8 with
    | .Some .. =>
      let a_9 := v8.Some_0
      a := a_9
      let b_11 := v7.Some_0
      b := b_11
      let v15 ← Point_add_Mont_Bn254Base a b
      r2 := v15
      let v17 ← encode r2
      let v18 := Option_array_u256_2.Some v17
      b1 a b r2 v18
    | _ =>
      let v20 := Option_array_u256_2.None
      b1 a b r2 v20
  | _ =>
    let v22 := Option_array_u256_2.None
    b1 a b r2 v22

def recv0_arm0 : FeM Empty := do
  let mut output : MemBuffer := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm
  r2 := v0
  let v2 ← add_MemBuffer r2
  match v2 with
  | .Some .. =>
    let sum := v2.Some_0
    let output_6 ← MemBuffer_new (UInt256.ofNat 64)
    output := output_6
    let v9 := (UInt256.ofNat 0)
    checkIndex v9 2
    let v10 := sum[v9.toNat]!
    let (v12, v12_0) ← MemBuffer_write_word output (UInt256.ofNat 0) v10
    output := v12_0
    let v14 := (UInt256.ofNat 1)
    checkIndex v14 2
    let v15 := sum[v14.toNat]!
    let (v17, v17_0) ← MemBuffer_write_word output (UInt256.ofNat 32) v15
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


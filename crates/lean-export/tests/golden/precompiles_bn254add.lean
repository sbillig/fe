export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_new MemBuffer_span_ref MemBuffer_write_word MemSlice_empty_u8 Option_array_u256_2 Option_array_u256_2.None Option_array_u256_2.Some downcast_unchecked_u256_usize)
export Shared (Mont_Bn254Base Mont_add_Bn254Base Mont_double_Bn254Base Mont_is_zero_Bn254Base Mont_mul_Bn254Base Mont_square_Bn254Base Mont_sub_Bn254Base Point_Mont_Bn254Base Point_Mont_Bn254Base.mk Point_double_Mont_Bn254Base Point_infinity_Mont_Bn254Base Point_is_infinity_Mont_Bn254Base encode g1_point_MemBuffer)

def init : FeM Unit := do
  return ()

def input_Evm : FeM MemBuffer := do
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

def Point_add_Mont_Bn254Base (v0 : Point_Mont_Bn254Base) (v1 : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let mut r5 : Mont_Bn254Base := default
  let mut r6 : Mont_Bn254Base := default
  let mut r7 : Mont_Bn254Base := default
  let mut r8 : Mont_Bn254Base := default
  let mut r9 : Mont_Bn254Base := default
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
  let mut r30 : Mont_Bn254Base := default
  let mut r31 : Mont_Bn254Base := default
  let mut r32 : Mont_Bn254Base := default
  let mut r33 : Mont_Bn254Base := default
  let mut r34 : Mont_Bn254Base := default
  let mut r35 : Mont_Bn254Base := default
  let mut r36 : Mont_Bn254Base := default
  let v3 ← Point_is_infinity_Mont_Bn254Base v0
  if v3 then
    let v4 := v1
    return v4
  else
    let v5 := ()
    let v6 := v5
    let v7 ← Point_is_infinity_Mont_Bn254Base v1
    if v7 then
      let v8 := v0
      return v8
    else
      let v9 := ()
      let v10 := v9
      let v11 := v0.z
      r12 := v11
      let v13 ← Mont_square_Bn254Base r12
      let v14 := v13
      r0 := v14
      let v15 := v1.z
      r13 := v15
      let v17 ← Mont_square_Bn254Base r13
      let v18 := v17
      r1 := v18
      let v19 := v0.x
      r14 := v19
      let v22 ← Mont_mul_Bn254Base r14 r1
      let v23 := v22
      r2 := v23
      let v24 := v1.x
      r15 := v24
      let v27 ← Mont_mul_Bn254Base r15 r0
      let v28 := v27
      r3 := v28
      let v29 := v1.z
      r16 := v29
      let v30 := v0.y
      r17 := v30
      let v33 ← Mont_mul_Bn254Base r17 r16
      r18 := v33
      let v36 ← Mont_mul_Bn254Base r18 r1
      let v37 := v36
      r4 := v37
      let v38 := v0.z
      r19 := v38
      let v39 := v1.y
      r20 := v39
      let v42 ← Mont_mul_Bn254Base r20 r19
      r21 := v42
      let v45 ← Mont_mul_Bn254Base r21 r0
      let v46 := v45
      r5 := v46
      let v49 ← Mont_sub_Bn254Base r3 r2
      let v50 := v49
      r6 := v50
      let v53 ← Mont_sub_Bn254Base r5 r4
      r22 := v53
      let v55 ← Mont_double_Bn254Base r22
      let v56 := v55
      r7 := v56
      let v58 ← Mont_is_zero_Bn254Base r6
      if v58 then
        let v60 ← Mont_is_zero_Bn254Base r7
        let b12 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Point_Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v2 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut r7 := r7
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
          let v61 ← Point_double_Mont_Bn254Base v0
          let v62 := v61
          b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v62
        else
          let v63 ← Point_infinity_Mont_Bn254Base
          let v64 := v63
          b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v64
      else
        let v65 := ()
        let v66 := v65
        let v68 ← Mont_double_Bn254Base r6
        r23 := v68
        let v70 ← Mont_square_Bn254Base r23
        let v71 := v70
        r8 := v71
        let v74 ← Mont_mul_Bn254Base r6 r8
        let v75 := v74
        r9 := v75
        let v78 ← Mont_mul_Bn254Base r2 r8
        let v79 := v78
        r10 := v79
        let v81 ← Mont_square_Bn254Base r7
        r24 := v81
        let v84 ← Mont_sub_Bn254Base r24 r9
        r25 := v84
        let v86 ← Mont_double_Bn254Base r10
        r26 := v86
        let v89 ← Mont_sub_Bn254Base r25 r26
        let v90 := v89
        r11 := v90
        let v93 ← Mont_sub_Bn254Base r10 r11
        r27 := v93
        let v96 ← Mont_mul_Bn254Base r7 r27
        r28 := v96
        let v99 ← Mont_mul_Bn254Base r4 r9
        r29 := v99
        let v101 ← Mont_double_Bn254Base r29
        r30 := v101
        let v104 ← Mont_sub_Bn254Base r28 r30
        let v105 := v104
        let v106 := v1.z
        r31 := v106
        let v107 := v0.z
        r32 := v107
        let v110 ← Mont_add_Bn254Base r32 r31
        r33 := v110
        let v112 ← Mont_square_Bn254Base r33
        r34 := v112
        let v115 ← Mont_sub_Bn254Base r34 r0
        r35 := v115
        let v118 ← Mont_sub_Bn254Base r35 r1
        r36 := v118
        let v121 ← Mont_mul_Bn254Base r36 r6
        let v122 := r11
        let v123 := (Point_Mont_Bn254Base.mk v122 v105 v121)
        return v123

def add_MemBuffer (v0 : MemBuffer) : FeM Option_array_u256_2 := do
  let mut r0 : Point_Mont_Bn254Base := default
  let mut r1 : Point_Mont_Bn254Base := default
  let mut r2 : Point_Mont_Bn254Base := default
  let v2 := (UInt256.ofNat 0)
  let v3 ← g1_point_MemBuffer v0 v2
  let v4 := (UInt256.ofNat 64)
  let v5 ← g1_point_MemBuffer v0 v4
  let v6 := (v3, v5)
  let v7 := v6.snd
  let b1 : Point_Mont_Bn254Base → Point_Mont_Bn254Base → Point_Mont_Bn254Base → Option_array_u256_2 → FeM Option_array_u256_2 := fun r0 r1 r2 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    return v1
  match v7 with
  | .Some .. =>
    let v8 := v6.fst
    match v8 with
    | .Some .. =>
      let v9 := v8.Some_0
      let v10 := v9
      r0 := v10
      let v11 := v7.Some_0
      let v12 := v11
      r1 := v12
      let v15 ← Point_add_Mont_Bn254Base r0 r1
      r2 := v15
      let v17 ← encode r2
      let v18 := Option_array_u256_2.Some v17
      let v19 := v18
      b1 r0 r1 r2 v19
    | _ =>
      let v20 := Option_array_u256_2.None
      let v21 := v20
      b1 r0 r1 r2 v21
  | _ =>
    let v22 := Option_array_u256_2.None
    let v23 := v22
    b1 r0 r1 r2 v23

def recv0_arm0 : FeM Empty := do
  let mut r1 : MemBuffer := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm
  r2 := v0
  let v2 ← add_MemBuffer r2
  match v2 with
  | .Some .. =>
    let v3 := v2.Some_0
    let v4 := v3
    let v5 := (UInt256.ofNat 64)
    let v6 ← MemBuffer_new v5
    let v7 := v6
    r1 := v7
    let v8 := (UInt256.ofNat 0)
    let v9 := (UInt256.ofNat 0)
    checkIndex v9 2
    let v10 := v4[v9.toNat]!
    let (v12, v12_0) ← MemBuffer_write_word r1 v8 v10
    r1 := v12_0
    let v13 := (UInt256.ofNat 32)
    let v14 := (UInt256.ofNat 1)
    checkIndex v14 2
    let v15 := v4[v14.toNat]!
    let (v17, v17_0) ← MemBuffer_write_word r1 v13 v15
    r1 := v17_0
    let v19 ← MemBuffer_span_ref r1
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


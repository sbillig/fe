export Lib (CallData Evm_input Evm_return_value_Sol_tuple_bool_u256_u256 Range_Known_1_Known_16 Range_Known_1_Known_16.mk Range_Known_1_Known_4 Range_Known_1_Known_4.mk Range_at_1_16 Range_at_1_4 Range_has_1_16 Range_has_1_4 Range_next_1_16 Range_next_1_4 Range_start_1_16 Range_start_1_4 Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_u256 array_at_ref_usize_3 array_has_usize_3 array_next_usize_3 array_start_usize_3 assert decode_msg_field_from_Sol_u256_CallData downcast_truncate_u256_u64 downcast_truncate_u256_usize len_CallData u64_wrapping_mul)
export Shared (adc invert_256 mul_wide sbb)

structure Mont_Base where
  l0 : UInt256
  l1 : UInt256
  l2 : UInt256
  l3 : UInt256
deriving Inhabited

structure Mont_Order where
  l0 : UInt256
  l1 : UInt256
  l2 : UInt256
  l3 : UInt256
deriving Inhabited

structure Point_Mont_Base where
  x : Mont_Base
  y : Mont_Base
  z : Mont_Base
deriving Inhabited

structure PublicKey where
  x : UInt256
  y : UInt256
deriving Inhabited

inductive Option_PublicKey where
  | None
  | Some (f0 : PublicKey)
deriving Inhabited

def Option_PublicKey.tag : Option_PublicKey → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_PublicKey.Some_0 : Option_PublicKey → PublicKey
  | .Some f0 => f0
  | _ => default

def Option_PublicKey.set_Some_0 (x : Option_PublicKey) (v : PublicKey) : Option_PublicKey :=
  match x with
  | .Some f0 => .Some v
  | x => x

structure Recover where
  hash : UInt256
  recovery_id : UInt256
  r : UInt256
  s : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def N : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907852837564279074904382605163141518161494337)
  return v0

def P : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007908834671663)
  return v0

def Base_VALUE : FeM UInt256 := do
  let v0 ← P
  return v0

def raw_Base (v0 : UInt256) : FeM Mont_Base := do
  let v1 ← downcast_truncate_u256_u64 v0
  let v2 := v0
  let v3 := (UInt256.ofNat 64)
  let v4 := v2 >>> v3
  let v5 ← downcast_truncate_u256_u64 v4
  let v6 := v0
  let v7 := (UInt256.ofNat 128)
  let v8 := v6 >>> v7
  let v9 ← downcast_truncate_u256_u64 v8
  let v10 := v0
  let v11 := (UInt256.ofNat 192)
  let v12 := v10 >>> v11
  let v13 ← downcast_truncate_u256_u64 v12
  let v14 := (Mont_Base.mk v1 v5 v9 v13)
  return v14

def Base_R2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446752466076602529)
  return v0

def Base_INV : FeM UInt256 := do
  let v0 := (UInt256.ofNat 15580212934572586289)
  return v0

def Base_P0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744069414583343)
  return v0

def Base_P1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def Base_P2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def Base_P3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def Mont_shift_reduced_Base_k3 (v5 : UInt256) (v34 : UInt256) (v50 : UInt256) (v66 : UInt256) (v73 : UInt256) (v75 : (UInt256 × UInt256)) (v77 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v78 := v75.snd
  let v79 := v78
  let v80 ← adc v5 v73 v79
  let v81 := v80.fst
  let v82 := v81
  let v83 := (Mont_Base.mk v34 v50 v66 v77)
  let v84 := (v83, v82)
  return v84

def Mont_shift_reduced_Base_k2 (v4 : UInt256) (v5 : UInt256) (v9 : UInt256) (v34 : UInt256) (v50 : UInt256) (v52 : UInt256) (v53 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  let v54 := v53.fst
  let v55 := v54
  let v56 := v53.snd
  let v57 := v56
  let v58 ← Base_P3
  let v59 ← mul_wide v9 v58
  let v60 := v59.fst
  let v61 := v60
  let v62 := v59.snd
  let v63 := v62
  let v64 ← adc v55 v61 v52
  let v65 := v64.fst
  let v66 := v65
  let v67 := v64.snd
  let v68 := v67
  let v69 ← adc v4 v63 v57
  let v70 := v69.fst
  let v71 := v70
  let v72 := v69.snd
  let v73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc v71 v74 v68
  let v76 := v75.fst
  let v77 := v76
  Mont_shift_reduced_Base_k3 v5 v34 v50 v66 v73 v75 v77

def Mont_shift_reduced_Base_k1 (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v9 : UInt256) (v19 : UInt256) (v23 : UInt256) (v25 : UInt256) (v27 : (UInt256 × UInt256)) (v29 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v30 := v27.snd
  let v31 := v30
  let v32 ← adc v23 v29 v19
  let v33 := v32.fst
  let v34 := v33
  let v35 := v32.snd
  let v36 := v35
  let v37 ← adc v2 v31 v25
  let v38 := v37.fst
  let v39 := v38
  let v40 := v37.snd
  let v41 := v40
  let v42 ← Base_P2
  let v43 ← mul_wide v9 v42
  let v44 := v43.fst
  let v45 := v44
  let v46 := v43.snd
  let v47 := v46
  let v48 ← adc v39 v45 v36
  let v49 := v48.fst
  let v50 := v49
  let v51 := v48.snd
  let v52 := v51
  let v53 ← adc v3 v47 v41
  Mont_shift_reduced_Base_k2 v4 v5 v9 v34 v50 v52 v53

def Mont_shift_reduced_Base (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v6 ← Base_INV
  let v7 := v0
  let v8 ← u64_wrapping_mul v7 v6
  let v9 := v8
  let v10 ← Base_P0
  let v11 ← mul_wide v9 v10
  let v12 := v11.fst
  let v13 := v12
  let v14 := v11.snd
  let v15 := v14
  let v16 := (UInt256.ofNat 0)
  let v17 ← adc v0 v13 v16
  let v18 := v17.snd
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc v1 v15 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  let v26 ← Base_P1
  let v27 ← mul_wide v9 v26
  let v28 := v27.fst
  let v29 := v28
  Mont_shift_reduced_Base_k1 v2 v3 v4 v5 v9 v19 v23 v25 v27 v29

def Mont_first_row_Base_k1 (v0 : Mont_Base) (v1 : UInt256) (v5 : UInt256) (v13 : UInt256) (v17 : UInt256) (v19 : UInt256) (v23 : UInt256) (v25 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v26 ← adc v13 v23 v19
  let v27 := v26.fst
  let v28 := v27
  let v29 := v26.snd
  let v30 := v29
  let v31 := v0.l3
  let v32 ← mul_wide v1 v31
  let v33 := v32.fst
  let v34 := v33
  let v35 := v32.snd
  let v36 := v35
  let v37 ← adc v25 v34 v30
  let v38 := v37.fst
  let v39 := v38
  let v40 := v37.snd
  let v41 := v40
  let v42 := (UInt256.ofNat 0)
  let v43 ← adc v36 v42 v41
  let v44 := v43.fst
  let v45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 ← Mont_shift_reduced_Base v5 v17 v28 v39 v45 v46
  return v47

def Mont_first_row_Base (v0 : Mont_Base) (v1 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v2 := v0.l0
  let v3 ← mul_wide v1 v2
  let v4 := v3.fst
  let v5 := v4
  let v6 := v3.snd
  let v7 := v6
  let v8 := v0.l1
  let v9 ← mul_wide v1 v8
  let v10 := v9.fst
  let v11 := v10
  let v12 := v9.snd
  let v13 := v12
  let v14 := (UInt256.ofNat 0)
  let v15 ← adc v7 v11 v14
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v0.l2
  let v21 ← mul_wide v1 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  Mont_first_row_Base_k1 v0 v1 v5 v13 v17 v19 v23 v25

def Mont_row_Base_k3 (v14 : UInt256) (v32 : UInt256) (v49 : UInt256) (v66 : UInt256) (v73 : UInt256) (v75 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  let v76 := v75.fst
  let v77 := v76
  let v78 := v75.snd
  let v79 := v78
  let v80 := (UInt256.ofNat 0)
  let v81 ← adc v73 v80 v79
  let v82 := v81.fst
  let v83 := v82
  let v84 ← Mont_shift_reduced_Base v14 v32 v49 v66 v77 v83
  return v84

def Mont_row_Base_k2 (v0 : Mont_Base) (v1 : UInt256) (v2 : Mont_Base) (v3 : UInt256) (v14 : UInt256) (v32 : UInt256) (v40 : UInt256) (v46 : UInt256) (v49 : UInt256) (v51 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v52 := v2.l3
  let v53 ← adc v52 v46 v40
  let v54 := v53.fst
  let v55 := v54
  let v56 := v53.snd
  let v57 := v56
  let v58 := v0.l3
  let v59 ← mul_wide v1 v58
  let v60 := v59.fst
  let v61 := v60
  let v62 := v59.snd
  let v63 := v62
  let v64 ← adc v55 v61 v51
  let v65 := v64.fst
  let v66 := v65
  let v67 := v64.snd
  let v68 := v67
  let v69 ← adc v3 v63 v57
  let v70 := v69.fst
  let v71 := v70
  let v72 := v69.snd
  let v73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc v71 v74 v68
  Mont_row_Base_k3 v14 v32 v49 v66 v73 v75

def Mont_row_Base_k1 (v0 : Mont_Base) (v1 : UInt256) (v2 : Mont_Base) (v3 : UInt256) (v14 : UInt256) (v16 : UInt256) (v21 : UInt256) (v23 : UInt256) (v25 : (UInt256 × UInt256)) (v27 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v28 := v25.snd
  let v29 := v28
  let v30 ← adc v21 v27 v16
  let v31 := v30.fst
  let v32 := v31
  let v33 := v30.snd
  let v34 := v33
  let v35 := v2.l2
  let v36 ← adc v35 v29 v23
  let v37 := v36.fst
  let v38 := v37
  let v39 := v36.snd
  let v40 := v39
  let v41 := v0.l2
  let v42 ← mul_wide v1 v41
  let v43 := v42.fst
  let v44 := v43
  let v45 := v42.snd
  let v46 := v45
  let v47 ← adc v38 v44 v34
  let v48 := v47.fst
  let v49 := v48
  let v50 := v47.snd
  let v51 := v50
  Mont_row_Base_k2 v0 v1 v2 v3 v14 v32 v40 v46 v49 v51

def Mont_row_Base (v0 : Mont_Base) (v1 : UInt256) (v2 : Mont_Base) (v3 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v4 := v0.l0
  let v5 ← mul_wide v1 v4
  let v6 := v5.fst
  let v7 := v6
  let v8 := v5.snd
  let v9 := v8
  let v10 := v2.l0
  let v11 := (UInt256.ofNat 0)
  let v12 ← adc v10 v7 v11
  let v13 := v12.fst
  let v14 := v13
  let v15 := v12.snd
  let v16 := v15
  let v17 := v2.l1
  let v18 := (UInt256.ofNat 0)
  let v19 ← adc v17 v9 v18
  let v20 := v19.fst
  let v21 := v20
  let v22 := v19.snd
  let v23 := v22
  let v24 := v0.l1
  let v25 ← mul_wide v1 v24
  let v26 := v25.fst
  let v27 := v26
  Mont_row_Base_k1 v0 v1 v2 v3 v14 v16 v21 v23 v25 v27

def Mont_reduce_Base (v0 : Mont_Base) (v1 : UInt256) : FeM Mont_Base := do
  let v3 := v0.l0
  let v4 ← Base_P0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let v8 := v7
  let v9 := v6.snd
  let v10 := v9
  let v11 := v0.l1
  let v12 ← Base_P1
  let v13 ← sbb v11 v12 v10
  let v14 := v13.fst
  let v15 := v14
  let v16 := v13.snd
  let v17 := v16
  let v18 := v0.l2
  let v19 ← Base_P2
  let v20 ← sbb v18 v19 v17
  let v21 := v20.fst
  let v22 := v21
  let v23 := v20.snd
  let v24 := v23
  let v25 := v0.l3
  let v26 ← Base_P3
  let v27 ← sbb v25 v26 v24
  let v28 := v27.fst
  let v29 := v28
  let v30 := v27.snd
  let v31 := v30
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb v1 v32 v31
  let v34 := v33.snd
  let v35 := v34
  let v36 := (UInt256.ofNat 1)
  let v37 := decide (v35 = v36)
  let b3 : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  if v37 then
    let v38 := v0
    let v39 := v38
    b3 v39
  else
    let v40 := (Mont_Base.mk v8 v15 v22 v29)
    let v41 := v40
    b3 v41

def Mont_mul_Base (v0 : Mont_Base) (v1 : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let v2 := v1.l0
  let v3 ← Mont_first_row_Base v0 v2
  let v4 := v3.fst
  let v5 := v4
  r0 := v5
  let v6 := v3.snd
  let v7 := v6
  let v8 := v1.l1
  let v10 ← Mont_row_Base v0 v8 r0 v7
  let v11 := v10.fst
  let v12 := v11
  r1 := v12
  let v13 := v10.snd
  let v14 := v13
  let v15 := v1.l2
  let v17 ← Mont_row_Base v0 v15 r1 v14
  let v18 := v17.fst
  let v19 := v18
  r2 := v19
  let v20 := v17.snd
  let v21 := v20
  let v22 := v1.l3
  let v24 ← Mont_row_Base v0 v22 r2 v21
  let v25 := v24.fst
  let v26 := v25
  r3 := v26
  let v27 := v24.snd
  let v28 := v27
  let v30 ← Mont_reduce_Base r3 v28
  return v30

def Mont_new_Base (v0 : UInt256) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let v1 := v0
  let v2 ← Base_VALUE
  let v3 := decide (v1.toNat < v2.toNat)
  let v4 ← assert v3
  let v5 ← raw_Base v0
  r0 := v5
  let v6 ← Base_R2
  let v7 ← raw_Base v6
  r1 := v7
  let v10 ← Mont_mul_Base r0 r1
  return v10

def Mont_square_Base (v0 : Mont_Base) : FeM Mont_Base := do
  let v1 ← Mont_mul_Base v0 v0
  return v1

def Mont_add_Base (v0 : Mont_Base) (v1 : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let v2 := v0.l0
  let v3 := v1.l0
  let v4 := (UInt256.ofNat 0)
  let v5 ← adc v2 v3 v4
  let v6 := v5.fst
  let v7 := v6
  let v8 := v5.snd
  let v9 := v8
  let v10 := v0.l1
  let v11 := v1.l1
  let v12 ← adc v10 v11 v9
  let v13 := v12.fst
  let v14 := v13
  let v15 := v12.snd
  let v16 := v15
  let v17 := v0.l2
  let v18 := v1.l2
  let v19 ← adc v17 v18 v16
  let v20 := v19.fst
  let v21 := v20
  let v22 := v19.snd
  let v23 := v22
  let v24 := v0.l3
  let v25 := v1.l3
  let v26 ← adc v24 v25 v23
  let v27 := v26.fst
  let v28 := v27
  let v29 := v26.snd
  let v30 := v29
  let v31 := (Mont_Base.mk v7 v14 v21 v28)
  r0 := v31
  let v33 ← Mont_reduce_Base r0 v30
  return v33

def Mont_one_Base : FeM Mont_Base := do
  let v0 := (UInt256.ofNat 1)
  let v1 ← Mont_new_Base v0
  return v1

def Mont_pow_Base (v0 : Mont_Base) (v1 : UInt256) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : UInt256 := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let v2 ← Mont_one_Base
  let v3 := v2
  r0 := v3
  let v4 := (UInt256.ofNat 256)
  let v5 := v4
  r1 := v5
  let rec b1 (r0 : Mont_Base) (r1 : UInt256) (r2 : Mont_Base) (r3 : Mont_Base) : FeM Mont_Base := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let v6 := r1
    let v7 := v6
    let v8 := (UInt256.ofNat 0)
    let v9 := decide (v8.toNat < v7.toNat)
    if v9 then
      let v10 := (UInt256.ofNat 1)
      let v11 := r1
      let v12 ← IntTy.sub (.u 64) v11 v10
      let v13 := v12
      r1 := v13
      let v14 := ()
      let v15 := r0
      let v16 := v15
      r2 := v16
      let v18 ← Mont_square_Base r2
      let v19 := v18
      r0 := v19
      let v20 := ()
      let v21 := v1
      let v22 := r1
      let v23 := v22
      let v24 := v21 >>> v23
      let v25 := (UInt256.ofNat 1)
      let v26 := v24 &&& v25
      let v27 := (UInt256.ofNat 1)
      let v28 := decide (v26 = v27)
      let b6 : Mont_Base → UInt256 → Mont_Base → Mont_Base → FeM Mont_Base := fun r0 r1 r2 r3 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        b1 r0 r1 r2 r3
      if v28 then
        let v29 := r0
        let v30 := v29
        r3 := v30
        let v32 ← Mont_mul_Base r3 v0
        let v33 := v32
        r0 := v33
        let v34 := ()
        let v35 := v34
        b6 r0 r1 r2 r3
      else
        let v36 := ()
        let v37 := v36
        b6 r0 r1 r2 r3
    else
      let v38 := r0
      return v38
  partial_fixpoint
  b1 r0 r1 r2 r3

def Mont_equals_Base (v0 : Mont_Base) (v1 : Mont_Base) : FeM Bool := do
  let v3 := v0.l0
  let v4 := v1.l0
  let v5 := decide (v3 = v4)
  let b3 : Bool → FeM Bool := fun v2 => do
    return v2
  let b2 : FeM Bool := do
    let v17 := false
    let v18 := v17
    b3 v18
  if v5 then
    let v6 := v0.l1
    let v7 := v1.l1
    let v8 := decide (v6 = v7)
    if v8 then
      let v9 := v0.l2
      let v10 := v1.l2
      let v11 := decide (v9 = v10)
      if v11 then
        let v12 := v0.l3
        let v13 := v1.l3
        let v14 := decide (v12 = v13)
        if v14 then
          let v15 := true
          let v16 := v15
          b3 v16
        else
          b2
      else
        b2
    else
      b2
  else
    b2

def Mont_value_Base (v0 : Mont_Base) : FeM UInt256 := do
  let mut r0 : Mont_Base := default
  let v1 := (UInt256.ofNat 1)
  let v2 ← raw_Base v1
  r0 := v2
  let v4 ← Mont_mul_Base v0 r0
  let v5 := v4
  let v6 := v5.l0
  let v7 := v6
  let v8 := v5.l1
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := v9 <<< v10
  let v12 := v7 ||| v11
  let v13 := v5.l2
  let v14 := v13
  let v15 := (UInt256.ofNat 128)
  let v16 := v14 <<< v15
  let v17 := v12 ||| v16
  let v18 := v5.l3
  let v19 := v18
  let v20 := (UInt256.ofNat 192)
  let v21 := v19 <<< v20
  let v22 := v17 ||| v21
  return v22

def Order_VALUE : FeM UInt256 := do
  let v0 ← N
  return v0

def raw_Order (v0 : UInt256) : FeM Mont_Order := do
  let v1 ← downcast_truncate_u256_u64 v0
  let v2 := v0
  let v3 := (UInt256.ofNat 64)
  let v4 := v2 >>> v3
  let v5 ← downcast_truncate_u256_u64 v4
  let v6 := v0
  let v7 := (UInt256.ofNat 128)
  let v8 := v6 >>> v7
  let v9 ← downcast_truncate_u256_u64 v8
  let v10 := v0
  let v11 := (UInt256.ofNat 192)
  let v12 := v10 >>> v11
  let v13 ← downcast_truncate_u256_u64 v12
  let v14 := (Mont_Order.mk v1 v5 v9 v13)
  return v14

def Order_R2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 71195301480278335217902614543643724933430614355449737089222010364394701574464)
  return v0

def Order_INV : FeM UInt256 := do
  let v0 := (UInt256.ofNat 5408259542528602431)
  return v0

def Order_P0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 13822214165235122497)
  return v0

def Order_P1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 13451932020343611451)
  return v0

def Order_P2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551614)
  return v0

def Order_P3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551615)
  return v0

def Mont_shift_reduced_Order_k3 (v5 : UInt256) (v34 : UInt256) (v50 : UInt256) (v66 : UInt256) (v73 : UInt256) (v75 : (UInt256 × UInt256)) (v77 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v78 := v75.snd
  let v79 := v78
  let v80 ← adc v5 v73 v79
  let v81 := v80.fst
  let v82 := v81
  let v83 := (Mont_Order.mk v34 v50 v66 v77)
  let v84 := (v83, v82)
  return v84

def Mont_shift_reduced_Order_k2 (v4 : UInt256) (v5 : UInt256) (v9 : UInt256) (v34 : UInt256) (v50 : UInt256) (v52 : UInt256) (v53 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  let v54 := v53.fst
  let v55 := v54
  let v56 := v53.snd
  let v57 := v56
  let v58 ← Order_P3
  let v59 ← mul_wide v9 v58
  let v60 := v59.fst
  let v61 := v60
  let v62 := v59.snd
  let v63 := v62
  let v64 ← adc v55 v61 v52
  let v65 := v64.fst
  let v66 := v65
  let v67 := v64.snd
  let v68 := v67
  let v69 ← adc v4 v63 v57
  let v70 := v69.fst
  let v71 := v70
  let v72 := v69.snd
  let v73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc v71 v74 v68
  let v76 := v75.fst
  let v77 := v76
  Mont_shift_reduced_Order_k3 v5 v34 v50 v66 v73 v75 v77

def Mont_shift_reduced_Order_k1 (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v9 : UInt256) (v19 : UInt256) (v23 : UInt256) (v25 : UInt256) (v27 : (UInt256 × UInt256)) (v29 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v30 := v27.snd
  let v31 := v30
  let v32 ← adc v23 v29 v19
  let v33 := v32.fst
  let v34 := v33
  let v35 := v32.snd
  let v36 := v35
  let v37 ← adc v2 v31 v25
  let v38 := v37.fst
  let v39 := v38
  let v40 := v37.snd
  let v41 := v40
  let v42 ← Order_P2
  let v43 ← mul_wide v9 v42
  let v44 := v43.fst
  let v45 := v44
  let v46 := v43.snd
  let v47 := v46
  let v48 ← adc v39 v45 v36
  let v49 := v48.fst
  let v50 := v49
  let v51 := v48.snd
  let v52 := v51
  let v53 ← adc v3 v47 v41
  Mont_shift_reduced_Order_k2 v4 v5 v9 v34 v50 v52 v53

def Mont_shift_reduced_Order (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v6 ← Order_INV
  let v7 := v0
  let v8 ← u64_wrapping_mul v7 v6
  let v9 := v8
  let v10 ← Order_P0
  let v11 ← mul_wide v9 v10
  let v12 := v11.fst
  let v13 := v12
  let v14 := v11.snd
  let v15 := v14
  let v16 := (UInt256.ofNat 0)
  let v17 ← adc v0 v13 v16
  let v18 := v17.snd
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc v1 v15 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  let v26 ← Order_P1
  let v27 ← mul_wide v9 v26
  let v28 := v27.fst
  let v29 := v28
  Mont_shift_reduced_Order_k1 v2 v3 v4 v5 v9 v19 v23 v25 v27 v29

def Mont_first_row_Order_k1 (v0 : Mont_Order) (v1 : UInt256) (v5 : UInt256) (v13 : UInt256) (v17 : UInt256) (v19 : UInt256) (v23 : UInt256) (v25 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v26 ← adc v13 v23 v19
  let v27 := v26.fst
  let v28 := v27
  let v29 := v26.snd
  let v30 := v29
  let v31 := v0.l3
  let v32 ← mul_wide v1 v31
  let v33 := v32.fst
  let v34 := v33
  let v35 := v32.snd
  let v36 := v35
  let v37 ← adc v25 v34 v30
  let v38 := v37.fst
  let v39 := v38
  let v40 := v37.snd
  let v41 := v40
  let v42 := (UInt256.ofNat 0)
  let v43 ← adc v36 v42 v41
  let v44 := v43.fst
  let v45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 ← Mont_shift_reduced_Order v5 v17 v28 v39 v45 v46
  return v47

def Mont_first_row_Order (v0 : Mont_Order) (v1 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v2 := v0.l0
  let v3 ← mul_wide v1 v2
  let v4 := v3.fst
  let v5 := v4
  let v6 := v3.snd
  let v7 := v6
  let v8 := v0.l1
  let v9 ← mul_wide v1 v8
  let v10 := v9.fst
  let v11 := v10
  let v12 := v9.snd
  let v13 := v12
  let v14 := (UInt256.ofNat 0)
  let v15 ← adc v7 v11 v14
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v0.l2
  let v21 ← mul_wide v1 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  Mont_first_row_Order_k1 v0 v1 v5 v13 v17 v19 v23 v25

def Mont_row_Order_k3 (v14 : UInt256) (v32 : UInt256) (v49 : UInt256) (v66 : UInt256) (v73 : UInt256) (v75 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  let v76 := v75.fst
  let v77 := v76
  let v78 := v75.snd
  let v79 := v78
  let v80 := (UInt256.ofNat 0)
  let v81 ← adc v73 v80 v79
  let v82 := v81.fst
  let v83 := v82
  let v84 ← Mont_shift_reduced_Order v14 v32 v49 v66 v77 v83
  return v84

def Mont_row_Order_k2 (v0 : Mont_Order) (v1 : UInt256) (v2 : Mont_Order) (v3 : UInt256) (v14 : UInt256) (v32 : UInt256) (v40 : UInt256) (v46 : UInt256) (v49 : UInt256) (v51 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v52 := v2.l3
  let v53 ← adc v52 v46 v40
  let v54 := v53.fst
  let v55 := v54
  let v56 := v53.snd
  let v57 := v56
  let v58 := v0.l3
  let v59 ← mul_wide v1 v58
  let v60 := v59.fst
  let v61 := v60
  let v62 := v59.snd
  let v63 := v62
  let v64 ← adc v55 v61 v51
  let v65 := v64.fst
  let v66 := v65
  let v67 := v64.snd
  let v68 := v67
  let v69 ← adc v3 v63 v57
  let v70 := v69.fst
  let v71 := v70
  let v72 := v69.snd
  let v73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc v71 v74 v68
  Mont_row_Order_k3 v14 v32 v49 v66 v73 v75

def Mont_row_Order_k1 (v0 : Mont_Order) (v1 : UInt256) (v2 : Mont_Order) (v3 : UInt256) (v14 : UInt256) (v16 : UInt256) (v21 : UInt256) (v23 : UInt256) (v25 : (UInt256 × UInt256)) (v27 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v28 := v25.snd
  let v29 := v28
  let v30 ← adc v21 v27 v16
  let v31 := v30.fst
  let v32 := v31
  let v33 := v30.snd
  let v34 := v33
  let v35 := v2.l2
  let v36 ← adc v35 v29 v23
  let v37 := v36.fst
  let v38 := v37
  let v39 := v36.snd
  let v40 := v39
  let v41 := v0.l2
  let v42 ← mul_wide v1 v41
  let v43 := v42.fst
  let v44 := v43
  let v45 := v42.snd
  let v46 := v45
  let v47 ← adc v38 v44 v34
  let v48 := v47.fst
  let v49 := v48
  let v50 := v47.snd
  let v51 := v50
  Mont_row_Order_k2 v0 v1 v2 v3 v14 v32 v40 v46 v49 v51

def Mont_row_Order (v0 : Mont_Order) (v1 : UInt256) (v2 : Mont_Order) (v3 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v4 := v0.l0
  let v5 ← mul_wide v1 v4
  let v6 := v5.fst
  let v7 := v6
  let v8 := v5.snd
  let v9 := v8
  let v10 := v2.l0
  let v11 := (UInt256.ofNat 0)
  let v12 ← adc v10 v7 v11
  let v13 := v12.fst
  let v14 := v13
  let v15 := v12.snd
  let v16 := v15
  let v17 := v2.l1
  let v18 := (UInt256.ofNat 0)
  let v19 ← adc v17 v9 v18
  let v20 := v19.fst
  let v21 := v20
  let v22 := v19.snd
  let v23 := v22
  let v24 := v0.l1
  let v25 ← mul_wide v1 v24
  let v26 := v25.fst
  let v27 := v26
  Mont_row_Order_k1 v0 v1 v2 v3 v14 v16 v21 v23 v25 v27

def Mont_reduce_Order (v0 : Mont_Order) (v1 : UInt256) : FeM Mont_Order := do
  let v3 := v0.l0
  let v4 ← Order_P0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let v8 := v7
  let v9 := v6.snd
  let v10 := v9
  let v11 := v0.l1
  let v12 ← Order_P1
  let v13 ← sbb v11 v12 v10
  let v14 := v13.fst
  let v15 := v14
  let v16 := v13.snd
  let v17 := v16
  let v18 := v0.l2
  let v19 ← Order_P2
  let v20 ← sbb v18 v19 v17
  let v21 := v20.fst
  let v22 := v21
  let v23 := v20.snd
  let v24 := v23
  let v25 := v0.l3
  let v26 ← Order_P3
  let v27 ← sbb v25 v26 v24
  let v28 := v27.fst
  let v29 := v28
  let v30 := v27.snd
  let v31 := v30
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb v1 v32 v31
  let v34 := v33.snd
  let v35 := v34
  let v36 := (UInt256.ofNat 1)
  let v37 := decide (v35 = v36)
  let b3 : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  if v37 then
    let v38 := v0
    let v39 := v38
    b3 v39
  else
    let v40 := (Mont_Order.mk v8 v15 v22 v29)
    let v41 := v40
    b3 v41

def Mont_mul_Order (v0 : Mont_Order) (v1 : Mont_Order) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let mut r1 : Mont_Order := default
  let mut r2 : Mont_Order := default
  let mut r3 : Mont_Order := default
  let v2 := v1.l0
  let v3 ← Mont_first_row_Order v0 v2
  let v4 := v3.fst
  let v5 := v4
  r0 := v5
  let v6 := v3.snd
  let v7 := v6
  let v8 := v1.l1
  let v10 ← Mont_row_Order v0 v8 r0 v7
  let v11 := v10.fst
  let v12 := v11
  r1 := v12
  let v13 := v10.snd
  let v14 := v13
  let v15 := v1.l2
  let v17 ← Mont_row_Order v0 v15 r1 v14
  let v18 := v17.fst
  let v19 := v18
  r2 := v19
  let v20 := v17.snd
  let v21 := v20
  let v22 := v1.l3
  let v24 ← Mont_row_Order v0 v22 r2 v21
  let v25 := v24.fst
  let v26 := v25
  r3 := v26
  let v27 := v24.snd
  let v28 := v27
  let v30 ← Mont_reduce_Order r3 v28
  return v30

def Mont_new_Order (v0 : UInt256) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let mut r1 : Mont_Order := default
  let v1 := v0
  let v2 ← Order_VALUE
  let v3 := decide (v1.toNat < v2.toNat)
  let v4 ← assert v3
  let v5 ← raw_Order v0
  r0 := v5
  let v6 ← Order_R2
  let v7 ← raw_Order v6
  r1 := v7
  let v10 ← Mont_mul_Order r0 r1
  return v10

def Order_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v0 := (UInt256.ofNat 4598842128380346689)
  let v1 := (UInt256.ofNat 3079181878673178862)
  let v2 := (UInt256.ofNat 4611686018427387883)
  let v3 := (UInt256.ofNat 4611686018427387903)
  let v4 := (UInt256.ofNat 255)
  let v5 := #v[v0, v1, v2, v3, v4]
  return v5

def Order_INV62 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 3815112494326173377)
  return v0

def Order_R3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 38611792656279623856209604949727965891812033543773964974832218758873530450413)
  return v0

def Mont_inverse_Order (v0 : Mont_Order) : FeM Mont_Order := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Order := default
  let mut r3 : Mont_Order := default
  let v1 := v0.l0
  let v2 := v0.l1
  let v3 := v0.l2
  let v4 := v0.l3
  let v5 := #v[v1, v2, v3, v4]
  r0 := v5
  let v6 ← Order_SIGNED_62
  r1 := v6
  let v7 ← Order_INV62
  let v10 ← invert_256 r0 r1 v7
  let v11 := v10
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v13 := v11[v12.toNat]!
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v15 := v11[v14.toNat]!
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v17 := v11[v16.toNat]!
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v19 := v11[v18.toNat]!
  let v20 := (Mont_Order.mk v13 v15 v17 v19)
  r2 := v20
  let v21 ← Order_R3
  let v22 ← raw_Order v21
  r3 := v22
  let v25 ← Mont_mul_Order r2 r3
  return v25

def Mont_zero_Order : FeM Mont_Order := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (UInt256.ofNat 0)
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 0)
  let v4 := (Mont_Order.mk v0 v1 v2 v3)
  return v4

def Mont_sub_Order_k2 (v8 : UInt256) (v15 : UInt256) (v22 : UInt256) (v29 : UInt256) (v31 : UInt256) (v36 : UInt256) (v42 : UInt256) (v48 : UInt256) (v50 : UInt256) : FeM Mont_Order := do
  let v51 ← Order_P3
  let v52 ← adc v29 v51 v50
  let v53 := v52.fst
  let v54 := v53
  let v55 := (UInt256.ofNat 1)
  let v56 := decide (v31 = v55)
  let b3 : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Order.mk v36 v42 v48 v54)
    let v58 := v57
    b3 v58
  else
    let v59 := (Mont_Order.mk v8 v15 v22 v29)
    let v60 := v59
    b3 v60

def Mont_sub_Order_k1 (v8 : UInt256) (v15 : UInt256) (v22 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Mont_Order := do
  let v27 ← sbb v25 v26 v24
  let v28 := v27.fst
  let v29 := v28
  let v30 := v27.snd
  let v31 := v30
  let v32 ← Order_P0
  let v33 := (UInt256.ofNat 0)
  let v34 ← adc v8 v32 v33
  let v35 := v34.fst
  let v36 := v35
  let v37 := v34.snd
  let v38 := v37
  let v39 ← Order_P1
  let v40 ← adc v15 v39 v38
  let v41 := v40.fst
  let v42 := v41
  let v43 := v40.snd
  let v44 := v43
  let v45 ← Order_P2
  let v46 ← adc v22 v45 v44
  let v47 := v46.fst
  let v48 := v47
  let v49 := v46.snd
  let v50 := v49
  Mont_sub_Order_k2 v8 v15 v22 v29 v31 v36 v42 v48 v50

def Mont_sub_Order (v0 : Mont_Order) (v1 : Mont_Order) : FeM Mont_Order := do
  let v3 := v0.l0
  let v4 := v1.l0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let v8 := v7
  let v9 := v6.snd
  let v10 := v9
  let v11 := v0.l1
  let v12 := v1.l1
  let v13 ← sbb v11 v12 v10
  let v14 := v13.fst
  let v15 := v14
  let v16 := v13.snd
  let v17 := v16
  let v18 := v0.l2
  let v19 := v1.l2
  let v20 ← sbb v18 v19 v17
  let v21 := v20.fst
  let v22 := v21
  let v23 := v20.snd
  let v24 := v23
  let v25 := v0.l3
  let v26 := v1.l3
  Mont_sub_Order_k1 v8 v15 v22 v24 v25 v26

def Mont_neg_Order (v0 : Mont_Order) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let v1 ← Mont_zero_Order
  r0 := v1
  let v3 ← Mont_sub_Order r0 v0
  return v3

def Mont_value_Order (v0 : Mont_Order) : FeM UInt256 := do
  let mut r0 : Mont_Order := default
  let v1 := (UInt256.ofNat 1)
  let v2 ← raw_Order v1
  r0 := v2
  let v4 ← Mont_mul_Order v0 r0
  let v5 := v4
  let v6 := v5.l0
  let v7 := v6
  let v8 := v5.l1
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := v9 <<< v10
  let v12 := v7 ||| v11
  let v13 := v5.l2
  let v14 := v13
  let v15 := (UInt256.ofNat 128)
  let v16 := v14 <<< v15
  let v17 := v12 ||| v16
  let v18 := v5.l3
  let v19 := v18
  let v20 := (UInt256.ofNat 192)
  let v21 := v19 <<< v20
  let v22 := v17 ||| v21
  return v22

def Point_affine_Mont_Base (v0 : Mont_Base) (v1 : Mont_Base) : FeM Point_Mont_Base := do
  let v2 := v0
  let v3 := v1
  let v4 ← Mont_one_Base
  let v5 := (Point_Mont_Base.mk v2 v3 v4)
  return v5

def GX : FeM UInt256 := do
  let v0 := (UInt256.ofNat 55066263022277343669578718895168534326250603453777594175500187360389116729240)
  return v0

def GY : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32670510020758816978083085130507043184471273380659243275938904335757337482424)
  return v0

def generator : FeM Point_Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let v0 ← GX
  let v1 ← Mont_new_Base v0
  r0 := v1
  let v2 ← GY
  let v3 ← Mont_new_Base v2
  r1 := v3
  let v6 ← Point_affine_Mont_Base r0 r1
  return v6

def Mont_zero_Base : FeM Mont_Base := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (UInt256.ofNat 0)
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 0)
  let v4 := (Mont_Base.mk v0 v1 v2 v3)
  return v4

def Point_infinity_Mont_Base : FeM Point_Mont_Base := do
  let v0 ← Mont_one_Base
  let v1 ← Mont_one_Base
  let v2 ← Mont_zero_Base
  let v3 := (Point_Mont_Base.mk v0 v1 v2)
  return v3

def Mont_is_zero_Base (v0 : Mont_Base) : FeM Bool := do
  let v2 := v0.l0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (v2 = v3)
  let b3 : Bool → FeM Bool := fun v1 => do
    return v1
  let b2 : FeM Bool := do
    let v16 := false
    let v17 := v16
    b3 v17
  if v4 then
    let v5 := v0.l1
    let v6 := (UInt256.ofNat 0)
    let v7 := decide (v5 = v6)
    if v7 then
      let v8 := v0.l2
      let v9 := (UInt256.ofNat 0)
      let v10 := decide (v8 = v9)
      if v10 then
        let v11 := v0.l3
        let v12 := (UInt256.ofNat 0)
        let v13 := decide (v11 = v12)
        if v13 then
          let v14 := true
          let v15 := v14
          b3 v15
        else
          b2
      else
        b2
    else
      b2
  else
    b2

def Point_is_infinity_Mont_Base (v0 : Point_Mont_Base) : FeM Bool := do
  let mut r0 : Mont_Base := default
  let v1 := v0.z
  r0 := v1
  let v3 ← Mont_is_zero_Base r0
  return v3

def Mont_sub_Base_k2 (v8 : UInt256) (v15 : UInt256) (v22 : UInt256) (v29 : UInt256) (v31 : UInt256) (v36 : UInt256) (v42 : UInt256) (v48 : UInt256) (v50 : UInt256) : FeM Mont_Base := do
  let v51 ← Base_P3
  let v52 ← adc v29 v51 v50
  let v53 := v52.fst
  let v54 := v53
  let v55 := (UInt256.ofNat 1)
  let v56 := decide (v31 = v55)
  let b3 : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Base.mk v36 v42 v48 v54)
    let v58 := v57
    b3 v58
  else
    let v59 := (Mont_Base.mk v8 v15 v22 v29)
    let v60 := v59
    b3 v60

def Mont_sub_Base_k1 (v8 : UInt256) (v15 : UInt256) (v22 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Mont_Base := do
  let v27 ← sbb v25 v26 v24
  let v28 := v27.fst
  let v29 := v28
  let v30 := v27.snd
  let v31 := v30
  let v32 ← Base_P0
  let v33 := (UInt256.ofNat 0)
  let v34 ← adc v8 v32 v33
  let v35 := v34.fst
  let v36 := v35
  let v37 := v34.snd
  let v38 := v37
  let v39 ← Base_P1
  let v40 ← adc v15 v39 v38
  let v41 := v40.fst
  let v42 := v41
  let v43 := v40.snd
  let v44 := v43
  let v45 ← Base_P2
  let v46 ← adc v22 v45 v44
  let v47 := v46.fst
  let v48 := v47
  let v49 := v46.snd
  let v50 := v49
  Mont_sub_Base_k2 v8 v15 v22 v29 v31 v36 v42 v48 v50

def Mont_sub_Base (v0 : Mont_Base) (v1 : Mont_Base) : FeM Mont_Base := do
  let v3 := v0.l0
  let v4 := v1.l0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let v8 := v7
  let v9 := v6.snd
  let v10 := v9
  let v11 := v0.l1
  let v12 := v1.l1
  let v13 ← sbb v11 v12 v10
  let v14 := v13.fst
  let v15 := v14
  let v16 := v13.snd
  let v17 := v16
  let v18 := v0.l2
  let v19 := v1.l2
  let v20 ← sbb v18 v19 v17
  let v21 := v20.fst
  let v22 := v21
  let v23 := v20.snd
  let v24 := v23
  let v25 := v0.l3
  let v26 := v1.l3
  Mont_sub_Base_k1 v8 v15 v22 v24 v25 v26

def Mont_double_Base (v0 : Mont_Base) : FeM Mont_Base := do
  let v1 ← Mont_add_Base v0 v0
  return v1

def Point_double_Mont_Base (v0 : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  let mut r6 : Mont_Base := default
  let mut r7 : Mont_Base := default
  let mut r8 : Mont_Base := default
  let mut r9 : Mont_Base := default
  let mut r10 : Mont_Base := default
  let mut r11 : Mont_Base := default
  let mut r12 : Mont_Base := default
  let mut r13 : Mont_Base := default
  let mut r14 : Mont_Base := default
  let mut r15 : Mont_Base := default
  let mut r16 : Mont_Base := default
  let mut r17 : Mont_Base := default
  let mut r18 : Mont_Base := default
  let mut r19 : Mont_Base := default
  let mut r20 : Mont_Base := default
  let mut r21 : Mont_Base := default
  let mut r22 : Mont_Base := default
  let mut r23 : Mont_Base := default
  let mut r24 : Mont_Base := default
  let v1 ← Point_is_infinity_Mont_Base v0
  let b1 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → FeM Point_Mont_Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 => do
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
    let v72 ← Point_infinity_Mont_Base
    return v72
  if v1 then
    b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    let v2 := v0.y
    r6 := v2
    let v4 ← Mont_is_zero_Base r6
    if v4 then
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      let v5 := ()
      let v6 := v5
      let v7 := v0.x
      r7 := v7
      let v9 ← Mont_square_Base r7
      let v10 := v9
      r0 := v10
      let v11 := v0.y
      r8 := v11
      let v13 ← Mont_square_Base r8
      let v14 := v13
      r1 := v14
      let v16 ← Mont_square_Base r1
      let v17 := v16
      r2 := v17
      let v18 := v0.x
      r9 := v18
      let v21 ← Mont_add_Base r9 r1
      r10 := v21
      let v23 ← Mont_square_Base r10
      r11 := v23
      let v26 ← Mont_sub_Base r11 r0
      r12 := v26
      let v29 ← Mont_sub_Base r12 r2
      r13 := v29
      let v31 ← Mont_double_Base r13
      let v32 := v31
      r3 := v32
      let v34 ← Mont_double_Base r0
      r14 := v34
      let v37 ← Mont_add_Base r14 r0
      let v38 := v37
      r4 := v38
      let v40 ← Mont_square_Base r4
      r15 := v40
      let v42 ← Mont_double_Base r3
      r16 := v42
      let v45 ← Mont_sub_Base r15 r16
      let v46 := v45
      r5 := v46
      let v49 ← Mont_sub_Base r3 r5
      r17 := v49
      let v52 ← Mont_mul_Base r4 r17
      r18 := v52
      let v54 ← Mont_double_Base r2
      r19 := v54
      let v56 ← Mont_double_Base r19
      r20 := v56
      let v58 ← Mont_double_Base r20
      r21 := v58
      let v61 ← Mont_sub_Base r18 r21
      let v62 := v61
      let v63 := v0.z
      r22 := v63
      let v64 := v0.y
      r23 := v64
      let v67 ← Mont_mul_Base r23 r22
      r24 := v67
      let v69 ← Mont_double_Base r24
      let v70 := r5
      let v71 := (Point_Mont_Base.mk v70 v62 v69)
      return v71

def Point_add_Mont_Base (v0 : Point_Mont_Base) (v1 : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  let mut r6 : Mont_Base := default
  let mut r7 : Mont_Base := default
  let mut r8 : Mont_Base := default
  let mut r9 : Mont_Base := default
  let mut r10 : Mont_Base := default
  let mut r11 : Mont_Base := default
  let mut r12 : Mont_Base := default
  let mut r13 : Mont_Base := default
  let mut r14 : Mont_Base := default
  let mut r15 : Mont_Base := default
  let mut r16 : Mont_Base := default
  let mut r17 : Mont_Base := default
  let mut r18 : Mont_Base := default
  let mut r19 : Mont_Base := default
  let mut r20 : Mont_Base := default
  let mut r21 : Mont_Base := default
  let mut r22 : Mont_Base := default
  let mut r23 : Mont_Base := default
  let mut r24 : Mont_Base := default
  let mut r25 : Mont_Base := default
  let mut r26 : Mont_Base := default
  let mut r27 : Mont_Base := default
  let mut r28 : Mont_Base := default
  let mut r29 : Mont_Base := default
  let mut r30 : Mont_Base := default
  let mut r31 : Mont_Base := default
  let mut r32 : Mont_Base := default
  let mut r33 : Mont_Base := default
  let mut r34 : Mont_Base := default
  let mut r35 : Mont_Base := default
  let mut r36 : Mont_Base := default
  let v3 ← Point_is_infinity_Mont_Base v0
  if v3 then
    let v4 := v1
    return v4
  else
    let v5 := ()
    let v6 := v5
    let v7 ← Point_is_infinity_Mont_Base v1
    if v7 then
      let v8 := v0
      return v8
    else
      let v9 := ()
      let v10 := v9
      let v11 := v0.z
      r12 := v11
      let v13 ← Mont_square_Base r12
      let v14 := v13
      r0 := v14
      let v15 := v1.z
      r13 := v15
      let v17 ← Mont_square_Base r13
      let v18 := v17
      r1 := v18
      let v19 := v0.x
      r14 := v19
      let v22 ← Mont_mul_Base r14 r1
      let v23 := v22
      r2 := v23
      let v24 := v1.x
      r15 := v24
      let v27 ← Mont_mul_Base r15 r0
      let v28 := v27
      r3 := v28
      let v29 := v1.z
      r16 := v29
      let v30 := v0.y
      r17 := v30
      let v33 ← Mont_mul_Base r17 r16
      r18 := v33
      let v36 ← Mont_mul_Base r18 r1
      let v37 := v36
      r4 := v37
      let v38 := v0.z
      r19 := v38
      let v39 := v1.y
      r20 := v39
      let v42 ← Mont_mul_Base r20 r19
      r21 := v42
      let v45 ← Mont_mul_Base r21 r0
      let v46 := v45
      r5 := v46
      let v49 ← Mont_sub_Base r3 r2
      let v50 := v49
      r6 := v50
      let v53 ← Mont_sub_Base r5 r4
      r22 := v53
      let v55 ← Mont_double_Base r22
      let v56 := v55
      r7 := v56
      let v58 ← Mont_is_zero_Base r6
      if v58 then
        let v60 ← Mont_is_zero_Base r7
        let b12 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v2 => do
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
          let v61 ← Point_double_Mont_Base v0
          let v62 := v61
          b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v62
        else
          let v63 ← Point_infinity_Mont_Base
          let v64 := v63
          b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v64
      else
        let v65 := ()
        let v66 := v65
        let v68 ← Mont_double_Base r6
        r23 := v68
        let v70 ← Mont_square_Base r23
        let v71 := v70
        r8 := v71
        let v74 ← Mont_mul_Base r6 r8
        let v75 := v74
        r9 := v75
        let v78 ← Mont_mul_Base r2 r8
        let v79 := v78
        r10 := v79
        let v81 ← Mont_square_Base r7
        r24 := v81
        let v84 ← Mont_sub_Base r24 r9
        r25 := v84
        let v86 ← Mont_double_Base r10
        r26 := v86
        let v89 ← Mont_sub_Base r25 r26
        let v90 := v89
        r11 := v90
        let v93 ← Mont_sub_Base r10 r11
        r27 := v93
        let v96 ← Mont_mul_Base r7 r27
        r28 := v96
        let v99 ← Mont_mul_Base r4 r9
        r29 := v99
        let v101 ← Mont_double_Base r29
        r30 := v101
        let v104 ← Mont_sub_Base r28 r30
        let v105 := v104
        let v106 := v1.z
        r31 := v106
        let v107 := v0.z
        r32 := v107
        let v110 ← Mont_add_Base r32 r31
        r33 := v110
        let v112 ← Mont_square_Base r33
        r34 := v112
        let v115 ← Mont_sub_Base r34 r0
        r35 := v115
        let v118 ← Mont_sub_Base r35 r1
        r36 := v118
        let v121 ← Mont_mul_Base r36 r6
        let v122 := r11
        let v123 := (Point_Mont_Base.mk v122 v105 v121)
        return v123

def Base_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v0 := (UInt256.ofNat 4611686014132419631)
  let v1 := (UInt256.ofNat 4611686018427387903)
  let v2 := (UInt256.ofNat 4611686018427387903)
  let v3 := (UInt256.ofNat 4611686018427387903)
  let v4 := (UInt256.ofNat 255)
  let v5 := #v[v0, v1, v2, v3, v4]
  return v5

def Base_INV62 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 2866531139136965327)
  return v0

def Base_R3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 79228216581683516649686562417)
  return v0

def Mont_inverse_Base (v0 : Mont_Base) : FeM Mont_Base := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let v1 := v0.l0
  let v2 := v0.l1
  let v3 := v0.l2
  let v4 := v0.l3
  let v5 := #v[v1, v2, v3, v4]
  r0 := v5
  let v6 ← Base_SIGNED_62
  r1 := v6
  let v7 ← Base_INV62
  let v10 ← invert_256 r0 r1 v7
  let v11 := v10
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v13 := v11[v12.toNat]!
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v15 := v11[v14.toNat]!
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v17 := v11[v16.toNat]!
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v19 := v11[v18.toNat]!
  let v20 := (Mont_Base.mk v13 v15 v17 v19)
  r2 := v20
  let v21 ← Base_R3
  let v22 ← raw_Base v21
  r3 := v22
  let v25 ← Mont_mul_Base r2 r3
  return v25

def Point_add_affine_Mont_Base (v0 : Point_Mont_Base) (v1 : Mont_Base) (v2 : Mont_Base) : FeM Point_Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  let mut r6 : Mont_Base := default
  let mut r7 : Mont_Base := default
  let mut r8 : Mont_Base := default
  let mut r9 : Mont_Base := default
  let mut r10 : Mont_Base := default
  let mut r11 : Mont_Base := default
  let mut r12 : Mont_Base := default
  let mut r13 : Mont_Base := default
  let mut r14 : Mont_Base := default
  let mut r15 : Mont_Base := default
  let mut r16 : Mont_Base := default
  let mut r17 : Mont_Base := default
  let mut r18 : Mont_Base := default
  let mut r19 : Mont_Base := default
  let mut r20 : Mont_Base := default
  let mut r21 : Mont_Base := default
  let mut r22 : Mont_Base := default
  let mut r23 : Mont_Base := default
  let mut r24 : Mont_Base := default
  let mut r25 : Mont_Base := default
  let mut r26 : Mont_Base := default
  let mut r27 : Mont_Base := default
  let mut r28 : Mont_Base := default
  let mut r29 : Mont_Base := default
  let v4 ← Point_is_infinity_Mont_Base v0
  if v4 then
    let v5 ← Point_affine_Mont_Base v1 v2
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := v0.z
    r10 := v8
    let v10 ← Mont_square_Base r10
    let v11 := v10
    r0 := v11
    let v13 ← Mont_mul_Base v1 r0
    let v14 := v13
    r1 := v14
    let v15 := v0.z
    r11 := v15
    let v17 ← Mont_mul_Base v2 r11
    r12 := v17
    let v20 ← Mont_mul_Base r12 r0
    let v21 := v20
    r2 := v21
    let v22 := v0.x
    r13 := v22
    let v25 ← Mont_sub_Base r1 r13
    let v26 := v25
    r3 := v26
    let v27 := v0.y
    r14 := v27
    let v30 ← Mont_sub_Base r2 r14
    r15 := v30
    let v32 ← Mont_double_Base r15
    let v33 := v32
    r4 := v33
    let v35 ← Mont_is_zero_Base r3
    if v35 then
      let v37 ← Mont_is_zero_Base r4
      let b9 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v3 => do
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
        return v3
      if v37 then
        let v38 ← Point_double_Mont_Base v0
        let v39 := v38
        b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v39
      else
        let v40 ← Point_infinity_Mont_Base
        let v41 := v40
        b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v41
    else
      let v42 := ()
      let v43 := v42
      let v45 ← Mont_square_Base r3
      let v46 := v45
      r5 := v46
      let v48 ← Mont_double_Base r5
      r16 := v48
      let v50 ← Mont_double_Base r16
      let v51 := v50
      r6 := v51
      let v54 ← Mont_mul_Base r3 r6
      let v55 := v54
      r7 := v55
      let v56 := v0.x
      r17 := v56
      let v59 ← Mont_mul_Base r17 r6
      let v60 := v59
      r8 := v60
      let v62 ← Mont_square_Base r4
      r18 := v62
      let v65 ← Mont_sub_Base r18 r7
      r19 := v65
      let v67 ← Mont_double_Base r8
      r20 := v67
      let v70 ← Mont_sub_Base r19 r20
      let v71 := v70
      r9 := v71
      let v74 ← Mont_sub_Base r8 r9
      r21 := v74
      let v77 ← Mont_mul_Base r4 r21
      r22 := v77
      let v78 := v0.y
      r23 := v78
      let v81 ← Mont_mul_Base r23 r7
      r24 := v81
      let v83 ← Mont_double_Base r24
      r25 := v83
      let v86 ← Mont_sub_Base r22 r25
      let v87 := v86
      let v88 := v0.z
      r26 := v88
      let v91 ← Mont_add_Base r26 r3
      r27 := v91
      let v93 ← Mont_square_Base r27
      r28 := v93
      let v96 ← Mont_sub_Base r28 r0
      r29 := v96
      let v99 ← Mont_sub_Base r29 r5
      let v100 := r9
      let v101 := (Point_Mont_Base.mk v100 v87 v99)
      return v101

def combination_k2 (v0 : UInt256) (v1 : UInt256) (v38 : UInt256) (v39 : UInt256) (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
  let v40 := (UInt256.ofNat 12)
  let v41 := #v[v38, v39, v40]
  r12 := v41
  let v43 ← array_start_usize_3 r12
  let v44 := v43
  let rec b1 (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v3 : UInt256) : FeM Point_Mont_Base := do
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
    let v46 ← array_has_usize_3 r12 v3
    if v46 then
      let v48 ← array_at_ref_usize_3 r12 v3
      let v49 := v48
      let v50 := (UInt256.ofNat 1)
      let v51 := (UInt256.ofNat 4)
      let v52 := ()
      let v53 := (Range_Known_1_Known_4.mk v52 v52)
      r13 := v53
      let v55 ← Range_start_1_4 r13
      let v56 := v55
      let rec b5 (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v4 : UInt256) : FeM Point_Mont_Base := do
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
        let v58 ← Range_has_1_4 r13 v4
        if v58 then
          let v60 ← Range_at_1_4 r13 v4
          let v61 := v60
          checkIndex v61 16
          let v62 := r1[v61.toNat]!
          r14 := v62
          checkIndex v49 16
          let v63 := r1[v49.toNat]!
          r15 := v63
          let v66 ← Point_add_Mont_Base r15 r14
          let v67 ← checkedAdd v49 v61
          checkIndex v67 16
          r1 := (r1.set! v67.toNat v66)
          let v68 := ()
          let v70 ← Range_next_1_4 r13 v4
          let v71 := v70
          b5 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v71
        else
          let v72 := ()
          let v74 ← array_next_usize_3 r12 v3
          let v75 := v74
          b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v75
      partial_fixpoint
      b5 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v56
    else
      let v76 ← Mont_one_Base
      let v77 := Vector.replicate 16 v76
      let v78 := v77
      r2 := v78
      let v79 ← Mont_one_Base
      let v80 := v79
      r3 := v80
      let v81 := (UInt256.ofNat 1)
      let v82 := (UInt256.ofNat 16)
      let v83 := ()
      let v84 := (Range_Known_1_Known_16.mk v83 v83)
      r16 := v84
      let v86 ← Range_start_1_16 r16
      let v87 := v86
      let rec b9 (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v5 : UInt256) : FeM Point_Mont_Base := do
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
        let v89 ← Range_has_1_16 r16 v5
        if v89 then
          let v91 ← Range_at_1_16 r16 v5
          let v92 := v91
          let v93 := r3
          let v94 := v93
          checkIndex v92 16
          r2 := (r2.set! v92.toNat v94)
          let v95 := ()
          checkIndex v92 16
          let v96 := r1[v92.toNat]!
          r17 := v96
          let v98 ← Point_is_infinity_Mont_Base r17
          let v99 := !v98
          let b15 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
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
            let v112 ← Range_next_1_16 r16 v5
            let v113 := v112
            b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v113
          if v99 then
            checkIndex v92 16
            let v100 := r1[v92.toNat]!.z
            r18 := v100
            let v101 := r3
            let v102 := v101
            r19 := v102
            let v105 ← Mont_mul_Base r19 r18
            let v106 := v105
            r3 := v106
            let v107 := ()
            let v108 := v107
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          else
            let v109 := ()
            let v110 := v109
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
        else
          let v114 := r3
          let v115 := v114
          r20 := v115
          let v117 ← Mont_inverse_Base r20
          let v118 := v117
          r4 := v118
          let v119 := (UInt256.ofNat 16)
          let v120 := v119
          r5 := v120
          let rec b16 (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
            let v121 := r5
            let v122 := v121
            let v123 := (UInt256.ofNat 1)
            let v124 := decide (v123.toNat < v122.toNat)
            if v124 then
              let v125 := (UInt256.ofNat 1)
              let v126 := r5
              let v127 ← checkedSub v126 v125
              let v128 := v127
              r5 := v128
              let v129 := ()
              let v130 := r5
              checkIndex v130 16
              let v131 := r1[v130.toNat]!
              r21 := v131
              let v132 := r5
              let v134 ← Point_is_infinity_Mont_Base r21
              if v134 then
                b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
              else
                let v135 := ()
                let v136 := v135
                let v137 := r5
                checkIndex v137 16
                let v138 := r2[v137.toNat]!
                r22 := v138
                let v139 := r4
                let v140 := v139
                r23 := v140
                let v142 := r5
                let v144 ← Mont_mul_Base r23 r22
                let v145 := v144
                r6 := v145
                let v146 := r5
                checkIndex v146 16
                let v147 := r1[v146.toNat]!.z
                r24 := v147
                let v148 := r4
                let v149 := v148
                r25 := v149
                let v151 := r5
                let v153 ← Mont_mul_Base r25 r24
                let v154 := v153
                r4 := v154
                let v155 := ()
                let v157 ← Mont_square_Base r6
                let v158 := v157
                r7 := v158
                let v159 := r5
                checkIndex v159 16
                let v160 := r1[v159.toNat]!.x
                r26 := v160
                let v161 := r5
                let v164 ← Mont_mul_Base r26 r7
                r27 := v164
                let v167 ← Mont_mul_Base r7 r6
                r28 := v167
                let v168 := r5
                checkIndex v168 16
                let v169 := r1[v168.toNat]!.y
                r29 := v169
                let v170 := r5
                let v173 ← Mont_mul_Base r29 r28
                r30 := v173
                let v176 ← Point_affine_Mont_Base r27 r30
                let v177 := r5
                checkIndex v177 16
                r1 := (r1.set! v177.toNat v176)
                let v178 := ()
                b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
            else
              let v179 ← Point_infinity_Mont_Base
              let v180 := v179
              r8 := v180
              let v181 := (UInt256.ofNat 256)
              let v182 := v181
              r9 := v182
              let rec b22 (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
                let v183 := r9
                let v184 := v183
                let v185 := (UInt256.ofNat 0)
                let v186 := decide (v185.toNat < v184.toNat)
                if v186 then
                  let v187 := (UInt256.ofNat 2)
                  let v188 := r9
                  let v189 ← IntTy.sub (.u 64) v188 v187
                  let v190 := v189
                  r9 := v190
                  let v191 := ()
                  let v192 := r8
                  let v193 := v192
                  r31 := v193
                  let v195 ← Point_double_Mont_Base r31
                  r32 := v195
                  let v197 ← Point_double_Mont_Base r32
                  let v198 := v197
                  r8 := v198
                  let v199 := ()
                  let v200 := v0
                  let v201 := r9
                  let v202 := v201
                  let v203 := v200 >>> v202
                  let v204 := (UInt256.ofNat 3)
                  let v205 := v203 &&& v204
                  let v206 := (UInt256.ofNat 2)
                  let v207 := v205 <<< v206
                  let v208 := v1
                  let v209 := r9
                  let v210 := v209
                  let v211 := v208 >>> v210
                  let v212 := (UInt256.ofNat 3)
                  let v213 := v211 &&& v212
                  let v214 := v207 ||| v213
                  let v215 ← downcast_truncate_u256_usize v214
                  let v216 := v215
                  let v217 := (UInt256.ofNat 0)
                  let v218 := decide (v216 ≠ v217)
                  let b27 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
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
                    b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  let b26 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
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
                    let v234 := ()
                    let v235 := v234
                    b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  if v218 then
                    checkIndex v216 16
                    let v219 := r1[v216.toNat]!
                    r33 := v219
                    let v221 ← Point_is_infinity_Mont_Base r33
                    let v222 := !v221
                    if v222 then
                      checkIndex v216 16
                      let v223 := r1[v216.toNat]!.x
                      r34 := v223
                      checkIndex v216 16
                      let v224 := r1[v216.toNat]!.y
                      r35 := v224
                      let v225 := r8
                      let v226 := v225
                      r36 := v226
                      let v230 ← Point_add_affine_Mont_Base r36 r34 r35
                      let v231 := v230
                      r8 := v231
                      let v232 := ()
                      let v233 := v232
                      b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                    else
                      b26 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  else
                    b26 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                else
                  let v236 := r8
                  return v236
              partial_fixpoint
              b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          partial_fixpoint
          b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
      partial_fixpoint
      b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v87
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v44

def combination_k1 (v0 : UInt256) (v1 : UInt256) (v20 : Point_Mont_Base) (v21 : UInt256) (r0 : Point_Mont_Base) (r1 : (Vector Point_Mont_Base 16)) (r2 : (Vector Mont_Base 16)) (r3 : Mont_Base) (r4 : Mont_Base) (r5 : UInt256) (r6 : Mont_Base) (r7 : Mont_Base) (r8 : Point_Mont_Base) (r9 : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
  r1 := (r1.set! v21.toNat v20)
  let v22 := ()
  let v23 := (UInt256.ofNat 4)
  let v24 := r0
  checkIndex v23 16
  r1 := (r1.set! v23.toNat v24)
  let v25 := ()
  let v27 ← Point_double_Mont_Base r0
  let v28 := (UInt256.ofNat 8)
  checkIndex v28 16
  r1 := (r1.set! v28.toNat v27)
  let v29 := ()
  let v30 := (UInt256.ofNat 8)
  checkIndex v30 16
  let v31 := r1[v30.toNat]!
  r11 := v31
  let v34 ← Point_add_Mont_Base r11 r0
  let v35 := (UInt256.ofNat 12)
  checkIndex v35 16
  r1 := (r1.set! v35.toNat v34)
  let v36 := ()
  let v37 := (UInt256.ofNat 4)
  let v38 := v37
  let v39 := (UInt256.ofNat 8)
  combination_k2 v0 v1 v38 v39 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36

def combination (v0 : UInt256) (v1 : UInt256) (v2 : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut r0 : Point_Mont_Base := default
  let mut r1 : (Vector Point_Mont_Base 16) := default
  let mut r2 : (Vector Mont_Base 16) := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : UInt256 := default
  let mut r6 : Mont_Base := default
  let mut r7 : Mont_Base := default
  let mut r8 : Point_Mont_Base := default
  let mut r9 : UInt256 := default
  let mut r10 : Point_Mont_Base := default
  let mut r11 : Point_Mont_Base := default
  let mut r12 : (Vector UInt256 3) := default
  let mut r13 : Range_Known_1_Known_4 := default
  let mut r14 : Point_Mont_Base := default
  let mut r15 : Point_Mont_Base := default
  let mut r16 : Range_Known_1_Known_16 := default
  let mut r17 : Point_Mont_Base := default
  let mut r18 : Mont_Base := default
  let mut r19 : Mont_Base := default
  let mut r20 : Mont_Base := default
  let mut r21 : Point_Mont_Base := default
  let mut r22 : Mont_Base := default
  let mut r23 : Mont_Base := default
  let mut r24 : Mont_Base := default
  let mut r25 : Mont_Base := default
  let mut r26 : Mont_Base := default
  let mut r27 : Mont_Base := default
  let mut r28 : Mont_Base := default
  let mut r29 : Mont_Base := default
  let mut r30 : Mont_Base := default
  let mut r31 : Point_Mont_Base := default
  let mut r32 : Point_Mont_Base := default
  let mut r33 : Point_Mont_Base := default
  let mut r34 : Mont_Base := default
  let mut r35 : Mont_Base := default
  let mut r36 : Point_Mont_Base := default
  let v6 ← generator
  let v7 := v6
  r0 := v7
  let v8 ← Point_infinity_Mont_Base
  let v9 := Vector.replicate 16 v8
  let v10 := v9
  r1 := v10
  let v11 := v2
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 16
  r1 := (r1.set! v12.toNat v11)
  let v13 := ()
  let v14 ← Point_double_Mont_Base v2
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 16
  r1 := (r1.set! v15.toNat v14)
  let v16 := ()
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  let v18 := r1[v17.toNat]!
  r10 := v18
  let v20 ← Point_add_Mont_Base r10 v2
  let v21 := (UInt256.ofNat 3)
  checkIndex v21 16
  combination_k1 v0 v1 v20 v21 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36

def Point_to_affine_Mont_Base (v0 : Point_Mont_Base) : FeM (Mont_Base × Mont_Base) := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  let v1 := v0.z
  r2 := v1
  let v3 ← Mont_inverse_Base r2
  let v4 := v3
  r0 := v4
  let v6 ← Mont_square_Base r0
  let v7 := v6
  r1 := v7
  let v8 := v0.x
  r3 := v8
  let v11 ← Mont_mul_Base r3 r1
  let v14 ← Mont_mul_Base r1 r0
  r4 := v14
  let v15 := v0.y
  r5 := v15
  let v18 ← Mont_mul_Base r5 r4
  let v19 := (v11, v18)
  return v19

def Mont_neg_Base (v0 : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let v1 ← Mont_zero_Base
  r0 := v1
  let v3 ← Mont_sub_Base r0 v0
  return v3

def recover (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM Option_PublicKey := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Order := default
  let mut r5 : Point_Mont_Base := default
  let mut r6 : Mont_Base := default
  let mut r7 : Mont_Base := default
  let mut r8 : Mont_Base := default
  let mut r9 : Mont_Base := default
  let mut r10 : Mont_Base := default
  let mut r11 : Mont_Base := default
  let mut r12 : Mont_Order := default
  let mut r13 : Mont_Order := default
  let mut r14 : Mont_Order := default
  let mut r15 : Mont_Order := default
  let mut r16 : Mont_Order := default
  let mut r17 : Mont_Order := default
  let mut r18 : Point_Mont_Base := default
  let v6 := v1
  let v7 := (UInt256.ofNat 1)
  let v8 := decide (v6.toNat ≤ v7.toNat)
  let b3 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → Bool → FeM Option_PublicKey := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v4 => do
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
    let v25 ← assert v4
    let v26 ← Mont_new_Base v2
    let v27 := v26
    r0 := v27
    let v29 ← Mont_square_Base r0
    r8 := v29
    let v32 ← Mont_mul_Base r8 r0
    r9 := v32
    let v33 := (UInt256.ofNat 7)
    let v34 ← Mont_new_Base v33
    r10 := v34
    let v37 ← Mont_add_Base r9 r10
    let v38 := v37
    r1 := v38
    let v39 ← P
    let v40 := (UInt256.ofNat 1)
    let v41 ← checkedAdd v39 v40
    let v42 := (UInt256.ofNat 4)
    let v43 ← checkedDiv v41 v42
    let v45 ← Mont_pow_Base r1 v43
    let v46 := v45
    r2 := v46
    let v48 ← Mont_square_Base r2
    r11 := v48
    let v51 ← Mont_equals_Base r11 r1
    let v52 := !v51
    if v52 then
      let v53 := Option_PublicKey.None
      return v53
    else
      let v54 := ()
      let v55 := v54
      let v57 ← Mont_value_Base r2
      let v58 := (UInt256.ofNat 1)
      let v59 := v57 &&& v58
      let v60 := v1
      let v61 := decide (v59 = v60)
      let b13 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → Mont_Base → FeM Option_PublicKey := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v5 => do
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
        let v67 := v5
        r3 := v67
        let v68 ← Mont_new_Order v2
        r12 := v68
        let v70 ← Mont_inverse_Order r12
        let v71 := v70
        r4 := v71
        let v72 := v0
        let v73 ← N
        let v74 ← checkedRem v72 v73
        let v75 ← Mont_new_Order v74
        r13 := v75
        let v77 ← Mont_neg_Order r13
        r14 := v77
        let v80 ← Mont_mul_Order r14 r4
        r15 := v80
        let v82 ← Mont_value_Order r15
        let v83 := v82
        let v84 ← Mont_new_Order v3
        r16 := v84
        let v87 ← Mont_mul_Order r16 r4
        r17 := v87
        let v89 ← Mont_value_Order r17
        let v90 := v89
        let v93 ← Point_affine_Mont_Base r0 r3
        r18 := v93
        let v95 ← combination v83 v90 r18
        let v96 := v95
        r5 := v96
        let v98 ← Point_is_infinity_Mont_Base r5
        if v98 then
          let v99 := Option_PublicKey.None
          return v99
        else
          let v100 := ()
          let v101 := v100
          let v103 ← Point_to_affine_Mont_Base r5
          let v104 := v103.fst
          let v105 := v104
          r6 := v105
          let v106 := v103.snd
          let v107 := v106
          r7 := v107
          let v109 ← Mont_value_Base r6
          let v111 ← Mont_value_Base r7
          let v112 := (PublicKey.mk v109 v111)
          let v113 := Option_PublicKey.Some v112
          return v113
      if v61 then
        let v62 := r2
        let v63 := v62
        b13 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v63
      else
        let v65 ← Mont_neg_Base r2
        let v66 := v65
        b13 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v66
  let b2 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → FeM Option_PublicKey := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 => do
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
    let v23 := false
    let v24 := v23
    b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v24
  if v8 then
    let v9 := v2
    let v10 := (UInt256.ofNat 0)
    let v11 := decide (v9 ≠ v10)
    if v11 then
      let v12 := v2
      let v13 ← N
      let v14 := decide (v12.toNat < v13.toNat)
      if v14 then
        let v15 := v3
        let v16 := (UInt256.ofNat 0)
        let v17 := decide (v15 ≠ v16)
        if v17 then
          let v18 := v3
          let v19 ← N
          let v20 := decide (v18.toNat < v19.toNat)
          if v20 then
            let v21 := true
            let v22 := v21
            b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v22
          else
            b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
        else
          b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
      else
        b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
    else
      b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
  else
    b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18

def recv0_arm0 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM (Bool × UInt256 × UInt256) := do
  let v5 := (UInt256.ofNat 1)
  let v6 := decide (v5.toNat < v1.toNat)
  let b1 : FeM (Bool × UInt256 × UInt256) := do
    let v30 := false
    let v31 := (UInt256.ofNat 0)
    let v32 := (UInt256.ofNat 0)
    let v33 := (v30, v31, v32)
    return v33
  if v6 then
    b1
  else
    let v7 := (UInt256.ofNat 0)
    let v8 := decide (v2 = v7)
    if v8 then
      b1
    else
      let v9 ← N
      let v10 := decide (v9.toNat ≤ v2.toNat)
      if v10 then
        b1
      else
        let v11 := (UInt256.ofNat 0)
        let v12 := decide (v3 = v11)
        if v12 then
          b1
        else
          let v13 ← N
          let v14 := decide (v13.toNat ≤ v3.toNat)
          if v14 then
            b1
          else
            let v15 := ()
            let v16 := v15
            let v17 ← recover v0 v1 v2 v3
            let b8 : (Bool × UInt256 × UInt256) → FeM (Bool × UInt256 × UInt256) := fun v4 => do
              return v4
            match v17 with
            | .Some .. =>
              let v18 := v17.Some_0
              let v19 := v18
              let v20 := true
              let v21 := v19.x
              let v22 := v19.y
              let v23 := (v20, v21, v22)
              let v24 := v23
              b8 v24
            | .None =>
              let v25 := false
              let v26 := (UInt256.ofNat 0)
              let v27 := (UInt256.ofNat 0)
              let v28 := (v25, v26, v27)
              let v29 := v28
              b8 v29

def Recover_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_u256
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_u256
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_u256
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  return v8

def Recover_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Recover := do
  let v2 ← len_CallData v0
  let v3 := v2
  let v4 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v1 v3
  let v5 := v4
  let v6 := v1
  let v7 ← abi_field_head_size_u256
  let v8 ← checkedAdd v6 v7
  let v9 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v8 v3
  let v10 := v9
  let v11 := v1
  let v12 ← abi_field_head_size_u256
  let v13 ← checkedAdd v11 v12
  let v14 ← abi_field_head_size_u256
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v15 v3
  let v17 := v16
  let v18 := v1
  let v19 ← abi_field_head_size_u256
  let v20 ← checkedAdd v18 v19
  let v21 ← abi_field_head_size_u256
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_u256
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_u256_CallData v0 v1 v24 v3
  let v26 := v25
  let v27 := (Recover.mk v5 v10 v17 v26)
  return v27

def decode_from_prechecked_head_Recover_Sol_CallData (v0 : CallData) (v1 : UInt256) (v2 : UInt256) : FeM Recover := do
  let v3 ← Recover_decode_from_CallData v0 v1
  return v3

def decode_runtime_args_Evm_Sol_Recover : FeM Recover := do
  let mut r0 : CallData := default
  let v2 ← Evm_input
  let v3 := v2
  r0 := v3
  let v4 ← Sol_SELECTOR_SIZE
  let v5 := v4
  let v6 := r0
  let v7 ← len_CallData v6
  let v8 := v7
  let v9 ← Recover_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head v8 v5 v9
  let v11 := r0
  let v12 ← decode_from_prechecked_head_Recover_Sol_CallData v11 v5 v8
  return v12

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Recover
  let result ← recv0_arm0 message.hash message.recovery_id message.r message.s
  Evm_return_value_Sol_tuple_bool_u256_u256 result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x2aa6d351 then
      recv0_arm0_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


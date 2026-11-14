export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_new MemBuffer_span_ref MemBuffer_write_word MemSlice_empty_u8 Option_array_u256_2 Option_array_u256_2.None Option_array_u256_2.Some Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start assert downcast_truncate_u256_u64 downcast_unchecked_u256_usize u64_wrapping_mul)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1 adc invert_256 mul_wide sbb)

structure Mont_Bn254Base where
  l0 : UInt256
  l1 : UInt256
  l2 : UInt256
  l3 : UInt256
deriving Inhabited

structure Point_Mont_Bn254Base where
  x : Mont_Bn254Base
  y : Mont_Bn254Base
  z : Mont_Bn254Base
deriving Inhabited

inductive Option_Point_Mont_Bn254Base where
  | None
  | Some (f0 : Point_Mont_Bn254Base)
deriving Inhabited

def Option_Point_Mont_Bn254Base.tag : Option_Point_Mont_Bn254Base → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Point_Mont_Bn254Base.Some_0 : Option_Point_Mont_Bn254Base → Point_Mont_Bn254Base
  | .Some f0 => f0
  | _ => default

def Option_Point_Mont_Bn254Base.set_Some_0 (x : Option_Point_Mont_Bn254Base) (v : Point_Mont_Bn254Base) : Option_Point_Mont_Bn254Base :=
  match x with
  | .Some f0 => .Some v
  | x => x

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

def word_at_MemBuffer (v0 : MemBuffer) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : Range_Unknown_Unknown := default
  let v4 := v1
  let v5 := (UInt256.ofNat 32)
  let v6 ← checkedAdd v4 v5
  let v7 ← MemBuffer_len_1 v0
  let v8 := decide (v6.toNat ≤ v7.toNat)
  if v8 then
    let v9 ← MemBuffer_lane_at v0 v1
    let v10 := IntTy.swapBytes (.u 64) v9
    let v11 := v10
    let v12 := (UInt256.ofNat 192)
    let v13 := v11 <<< v12
    let v14 := v1
    let v15 := (UInt256.ofNat 8)
    let v16 ← checkedAdd v14 v15
    let v17 ← MemBuffer_lane_at v0 v16
    let v18 := IntTy.swapBytes (.u 64) v17
    let v19 := v18
    let v20 := (UInt256.ofNat 128)
    let v21 := v19 <<< v20
    let v22 := v13 ||| v21
    let v23 := v1
    let v24 := (UInt256.ofNat 16)
    let v25 ← checkedAdd v23 v24
    let v26 ← MemBuffer_lane_at v0 v25
    let v27 := IntTy.swapBytes (.u 64) v26
    let v28 := v27
    let v29 := (UInt256.ofNat 64)
    let v30 := v28 <<< v29
    let v31 := v22 ||| v30
    let v32 := v1
    let v33 := (UInt256.ofNat 24)
    let v34 ← checkedAdd v32 v33
    let v35 ← MemBuffer_lane_at v0 v34
    let v36 := IntTy.swapBytes (.u 64) v35
    let v37 := v36
    let v38 := v31 ||| v37
    return v38
  else
    let v39 := ()
    let v40 := v39
    let v41 := (UInt256.ofNat 0)
    let v42 := v41
    r0 := v42
    let v43 := v1
    let v44 := v1
    let v45 := (UInt256.ofNat 32)
    let v46 ← checkedAdd v44 v45
    let v47 := ()
    let v48 := (Range_Unknown_Unknown.mk v43 v46)
    r1 := v48
    let v50 ← Range_Unknown_Unknown_start r1
    let v51 := v50
    let rec b4 (r0 : UInt256) (r1 : Range_Unknown_Unknown) (v2 : UInt256) : FeM UInt256 := do
      let mut r0 := r0
      let mut r1 := r1
      let v53 ← Range_Unknown_Unknown_has r1 v2
      if v53 then
        let v55 ← Range_Unknown_Unknown_at r1 v2
        let v56 := v55
        let v57 := r0
        let v58 := v57
        let v59 := (UInt256.ofNat 8)
        let v60 := v58 <<< v59
        let v61 ← MemBuffer_len_1 v0
        let v62 := decide (v56.toNat < v61.toNat)
        let b10 : UInt256 → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun r0 r1 v3 => do
          let mut r0 := r0
          let mut r1 := r1
          let v68 := v60 ||| v3
          let v69 := v68
          r0 := v69
          let v70 := ()
          let v72 ← Range_Unknown_Unknown_next r1 v2
          let v73 := v72
          b4 r0 r1 v73
        if v62 then
          let v63 ← MemBuffer_byte_at v0 v56
          let v64 := v63
          let v65 := v64
          b10 r0 r1 v65
        else
          let v66 := (UInt256.ofNat 0)
          let v67 := v66
          b10 r0 r1 v67
      else
        let v74 := r0
        return v74
    partial_fixpoint
    b4 r0 r1 v51

def MODULUS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 21888242871839275222246405745257275088696311157297823662689037894645226208583)
  return v0

def Bn254Base_VALUE : FeM UInt256 := do
  let v0 ← MODULUS
  return v0

def raw_Bn254Base (v0 : UInt256) : FeM Mont_Bn254Base := do
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
  let v14 := (Mont_Bn254Base.mk v1 v5 v9 v13)
  return v14

def Bn254Base_R2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 3096616502983703923843567936837374451735540968419076528771170197431451843209)
  return v0

def Bn254Base_INV : FeM UInt256 := do
  let v0 := (UInt256.ofNat 9786893198990664585)
  return v0

def Bn254Base_P0 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4332616871279656263)
  return v0

def Bn254Base_P1 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 10917124144477883021)
  return v0

def Bn254Base_P2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 13281191951274694749)
  return v0

def Bn254Base_P3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 3486998266802970665)
  return v0

def Mont_shift_reduced_Bn254Base_k3 (v5 : UInt256) (v34 : UInt256) (v50 : UInt256) (v66 : UInt256) (v73 : UInt256) (v75 : (UInt256 × UInt256)) (v77 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v78 := v75.snd
  let v79 := v78
  let v80 ← adc v5 v73 v79
  let v81 := v80.fst
  let v82 := v81
  let v83 := (Mont_Bn254Base.mk v34 v50 v66 v77)
  let v84 := (v83, v82)
  return v84

def Mont_shift_reduced_Bn254Base_k2 (v4 : UInt256) (v5 : UInt256) (v9 : UInt256) (v34 : UInt256) (v50 : UInt256) (v52 : UInt256) (v53 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  let v54 := v53.fst
  let v55 := v54
  let v56 := v53.snd
  let v57 := v56
  let v58 ← Bn254Base_P3
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
  Mont_shift_reduced_Bn254Base_k3 v5 v34 v50 v66 v73 v75 v77

def Mont_shift_reduced_Bn254Base_k1 (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) (v9 : UInt256) (v19 : UInt256) (v23 : UInt256) (v25 : UInt256) (v27 : (UInt256 × UInt256)) (v29 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
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
  let v42 ← Bn254Base_P2
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
  Mont_shift_reduced_Bn254Base_k2 v4 v5 v9 v34 v50 v52 v53

def Mont_shift_reduced_Bn254Base (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v6 ← Bn254Base_INV
  let v7 := v0
  let v8 ← u64_wrapping_mul v7 v6
  let v9 := v8
  let v10 ← Bn254Base_P0
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
  let v26 ← Bn254Base_P1
  let v27 ← mul_wide v9 v26
  let v28 := v27.fst
  let v29 := v28
  Mont_shift_reduced_Bn254Base_k1 v2 v3 v4 v5 v9 v19 v23 v25 v27 v29

def Mont_first_row_Bn254Base_k1 (v0 : Mont_Bn254Base) (v1 : UInt256) (v5 : UInt256) (v13 : UInt256) (v17 : UInt256) (v19 : UInt256) (v23 : UInt256) (v25 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
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
  let v47 ← Mont_shift_reduced_Bn254Base v5 v17 v28 v39 v45 v46
  return v47

def Mont_first_row_Bn254Base (v0 : Mont_Bn254Base) (v1 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
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
  Mont_first_row_Bn254Base_k1 v0 v1 v5 v13 v17 v19 v23 v25

def Mont_row_Bn254Base_k3 (v14 : UInt256) (v32 : UInt256) (v49 : UInt256) (v66 : UInt256) (v73 : UInt256) (v75 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  let v76 := v75.fst
  let v77 := v76
  let v78 := v75.snd
  let v79 := v78
  let v80 := (UInt256.ofNat 0)
  let v81 ← adc v73 v80 v79
  let v82 := v81.fst
  let v83 := v82
  let v84 ← Mont_shift_reduced_Bn254Base v14 v32 v49 v66 v77 v83
  return v84

def Mont_row_Bn254Base_k2 (v0 : Mont_Bn254Base) (v1 : UInt256) (v2 : Mont_Bn254Base) (v3 : UInt256) (v14 : UInt256) (v32 : UInt256) (v40 : UInt256) (v46 : UInt256) (v49 : UInt256) (v51 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
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
  Mont_row_Bn254Base_k3 v14 v32 v49 v66 v73 v75

def Mont_row_Bn254Base_k1 (v0 : Mont_Bn254Base) (v1 : UInt256) (v2 : Mont_Bn254Base) (v3 : UInt256) (v14 : UInt256) (v16 : UInt256) (v21 : UInt256) (v23 : UInt256) (v25 : (UInt256 × UInt256)) (v27 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
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
  Mont_row_Bn254Base_k2 v0 v1 v2 v3 v14 v32 v40 v46 v49 v51

def Mont_row_Bn254Base (v0 : Mont_Bn254Base) (v1 : UInt256) (v2 : Mont_Bn254Base) (v3 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
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
  Mont_row_Bn254Base_k1 v0 v1 v2 v3 v14 v16 v21 v23 v25 v27

def Mont_reduce_Bn254Base (v0 : Mont_Bn254Base) (v1 : UInt256) : FeM Mont_Bn254Base := do
  let v3 := v0.l0
  let v4 ← Bn254Base_P0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let v8 := v7
  let v9 := v6.snd
  let v10 := v9
  let v11 := v0.l1
  let v12 ← Bn254Base_P1
  let v13 ← sbb v11 v12 v10
  let v14 := v13.fst
  let v15 := v14
  let v16 := v13.snd
  let v17 := v16
  let v18 := v0.l2
  let v19 ← Bn254Base_P2
  let v20 ← sbb v18 v19 v17
  let v21 := v20.fst
  let v22 := v21
  let v23 := v20.snd
  let v24 := v23
  let v25 := v0.l3
  let v26 ← Bn254Base_P3
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
  let b3 : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  if v37 then
    let v38 := v0
    let v39 := v38
    b3 v39
  else
    let v40 := (Mont_Bn254Base.mk v8 v15 v22 v29)
    let v41 := v40
    b3 v41

def Mont_mul_Bn254Base (v0 : Mont_Bn254Base) (v1 : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let v2 := v1.l0
  let v3 ← Mont_first_row_Bn254Base v0 v2
  let v4 := v3.fst
  let v5 := v4
  r0 := v5
  let v6 := v3.snd
  let v7 := v6
  let v8 := v1.l1
  let v10 ← Mont_row_Bn254Base v0 v8 r0 v7
  let v11 := v10.fst
  let v12 := v11
  r1 := v12
  let v13 := v10.snd
  let v14 := v13
  let v15 := v1.l2
  let v17 ← Mont_row_Bn254Base v0 v15 r1 v14
  let v18 := v17.fst
  let v19 := v18
  r2 := v19
  let v20 := v17.snd
  let v21 := v20
  let v22 := v1.l3
  let v24 ← Mont_row_Bn254Base v0 v22 r2 v21
  let v25 := v24.fst
  let v26 := v25
  r3 := v26
  let v27 := v24.snd
  let v28 := v27
  let v30 ← Mont_reduce_Bn254Base r3 v28
  return v30

def Mont_new_Bn254Base (v0 : UInt256) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let v1 := v0
  let v2 ← Bn254Base_VALUE
  let v3 := decide (v1.toNat < v2.toNat)
  let v4 ← assert v3
  let v5 ← raw_Bn254Base v0
  r0 := v5
  let v6 ← Bn254Base_R2
  let v7 ← raw_Bn254Base v6
  r1 := v7
  let v10 ← Mont_mul_Bn254Base r0 r1
  return v10

def fp (v0 : UInt256) : FeM Mont_Bn254Base := do
  let v1 ← Mont_new_Bn254Base v0
  return v1

def Mont_one_Bn254Base : FeM Mont_Bn254Base := do
  let v0 := (UInt256.ofNat 1)
  let v1 ← Mont_new_Bn254Base v0
  return v1

def Point_affine_Mont_Bn254Base (v0 : Mont_Bn254Base) (v1 : Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let v2 := v0
  let v3 := v1
  let v4 ← Mont_one_Bn254Base
  let v5 := (Point_Mont_Bn254Base.mk v2 v3 v4)
  return v5

def Mont_square_Bn254Base (v0 : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v1 ← Mont_mul_Bn254Base v0 v0
  return v1

def Mont_add_Bn254Base (v0 : Mont_Bn254Base) (v1 : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
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
  let v31 := (Mont_Bn254Base.mk v7 v14 v21 v28)
  r0 := v31
  let v33 ← Mont_reduce_Bn254Base r0 v30
  return v33

def Mont_equals_Bn254Base (v0 : Mont_Bn254Base) (v1 : Mont_Bn254Base) : FeM Bool := do
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

def on_curve_Mont_Bn254Base (v0 : Mont_Bn254Base) (v1 : Mont_Bn254Base) (v2 : Mont_Bn254Base) : FeM Bool := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let v3 ← Mont_square_Bn254Base v1
  r0 := v3
  let v4 ← Mont_square_Bn254Base v0
  r1 := v4
  let v6 ← Mont_mul_Bn254Base r1 v0
  r2 := v6
  let v8 ← Mont_add_Bn254Base r2 v2
  r3 := v8
  let v11 ← Mont_equals_Bn254Base r0 r3
  return v11

def Mont_zero_Bn254Base : FeM Mont_Bn254Base := do
  let v0 := (UInt256.ofNat 0)
  let v1 := (UInt256.ofNat 0)
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 0)
  let v4 := (Mont_Bn254Base.mk v0 v1 v2 v3)
  return v4

def Point_infinity_Mont_Bn254Base : FeM Point_Mont_Bn254Base := do
  let v0 ← Mont_one_Bn254Base
  let v1 ← Mont_one_Bn254Base
  let v2 ← Mont_zero_Bn254Base
  let v3 := (Point_Mont_Bn254Base.mk v0 v1 v2)
  return v3

def g1_point_MemBuffer (v0 : MemBuffer) (v1 : UInt256) : FeM Option_Point_Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let v2 ← word_at_MemBuffer v0 v1
  let v3 := v2
  let v4 := v1
  let v5 := (UInt256.ofNat 32)
  let v6 ← checkedAdd v4 v5
  let v7 ← word_at_MemBuffer v0 v6
  let v8 := v7
  let v9 ← MODULUS
  let v10 := decide (v9.toNat ≤ v3.toNat)
  let b1 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let v42 := Option_Point_Mont_Bn254Base.None
    return v42
  if v10 then
    b1 r0 r1 r2 r3 r4
  else
    let v11 ← MODULUS
    let v12 := decide (v11.toNat ≤ v8.toNat)
    if v12 then
      b1 r0 r1 r2 r3 r4
    else
      let v13 := ()
      let v14 := v13
      let v15 := (UInt256.ofNat 0)
      let v16 := decide (v3 = v15)
      let b6 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let v21 := ()
        let v22 := v21
        let v23 ← fp v3
        r0 := v23
        let v24 ← fp v8
        r1 := v24
        let v27 ← Point_affine_Mont_Bn254Base r0 r1
        let v28 := v27
        let v29 := v28.x
        r2 := v29
        let v30 := v28.y
        r3 := v30
        let v31 := (UInt256.ofNat 3)
        let v32 ← fp v31
        r4 := v32
        let v36 ← on_curve_Mont_Bn254Base r2 r3 r4
        let v37 := !v36
        if v37 then
          let v38 := Option_Point_Mont_Bn254Base.None
          return v38
        else
          let v39 := ()
          let v40 := v39
          let v41 := Option_Point_Mont_Bn254Base.Some v28
          return v41
      if v16 then
        let v17 := (UInt256.ofNat 0)
        let v18 := decide (v8 = v17)
        if v18 then
          let v19 ← Point_infinity_Mont_Bn254Base
          let v20 := Option_Point_Mont_Bn254Base.Some v19
          return v20
        else
          b6 r0 r1 r2 r3 r4
      else
        b6 r0 r1 r2 r3 r4

def Mont_is_zero_Bn254Base (v0 : Mont_Bn254Base) : FeM Bool := do
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

def Point_is_infinity_Mont_Bn254Base (v0 : Point_Mont_Bn254Base) : FeM Bool := do
  let mut r0 : Mont_Bn254Base := default
  let v1 := v0.z
  r0 := v1
  let v3 ← Mont_is_zero_Bn254Base r0
  return v3

def Mont_sub_Bn254Base_k2 (v8 : UInt256) (v15 : UInt256) (v22 : UInt256) (v29 : UInt256) (v31 : UInt256) (v36 : UInt256) (v42 : UInt256) (v48 : UInt256) (v50 : UInt256) : FeM Mont_Bn254Base := do
  let v51 ← Bn254Base_P3
  let v52 ← adc v29 v51 v50
  let v53 := v52.fst
  let v54 := v53
  let v55 := (UInt256.ofNat 1)
  let v56 := decide (v31 = v55)
  let b3 : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Bn254Base.mk v36 v42 v48 v54)
    let v58 := v57
    b3 v58
  else
    let v59 := (Mont_Bn254Base.mk v8 v15 v22 v29)
    let v60 := v59
    b3 v60

def Mont_sub_Bn254Base_k1 (v8 : UInt256) (v15 : UInt256) (v22 : UInt256) (v24 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Mont_Bn254Base := do
  let v27 ← sbb v25 v26 v24
  let v28 := v27.fst
  let v29 := v28
  let v30 := v27.snd
  let v31 := v30
  let v32 ← Bn254Base_P0
  let v33 := (UInt256.ofNat 0)
  let v34 ← adc v8 v32 v33
  let v35 := v34.fst
  let v36 := v35
  let v37 := v34.snd
  let v38 := v37
  let v39 ← Bn254Base_P1
  let v40 ← adc v15 v39 v38
  let v41 := v40.fst
  let v42 := v41
  let v43 := v40.snd
  let v44 := v43
  let v45 ← Bn254Base_P2
  let v46 ← adc v22 v45 v44
  let v47 := v46.fst
  let v48 := v47
  let v49 := v46.snd
  let v50 := v49
  Mont_sub_Bn254Base_k2 v8 v15 v22 v29 v31 v36 v42 v48 v50

def Mont_sub_Bn254Base (v0 : Mont_Bn254Base) (v1 : Mont_Bn254Base) : FeM Mont_Bn254Base := do
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
  Mont_sub_Bn254Base_k1 v8 v15 v22 v24 v25 v26

def Mont_double_Bn254Base (v0 : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v1 ← Mont_add_Bn254Base v0 v0
  return v1

def Point_double_Mont_Bn254Base (v0 : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
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
  let v1 ← Point_is_infinity_Mont_Bn254Base v0
  let b1 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 => do
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
    let v72 ← Point_infinity_Mont_Bn254Base
    return v72
  if v1 then
    b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    let v2 := v0.y
    r6 := v2
    let v4 ← Mont_is_zero_Bn254Base r6
    if v4 then
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      let v5 := ()
      let v6 := v5
      let v7 := v0.x
      r7 := v7
      let v9 ← Mont_square_Bn254Base r7
      let v10 := v9
      r0 := v10
      let v11 := v0.y
      r8 := v11
      let v13 ← Mont_square_Bn254Base r8
      let v14 := v13
      r1 := v14
      let v16 ← Mont_square_Bn254Base r1
      let v17 := v16
      r2 := v17
      let v18 := v0.x
      r9 := v18
      let v21 ← Mont_add_Bn254Base r9 r1
      r10 := v21
      let v23 ← Mont_square_Bn254Base r10
      r11 := v23
      let v26 ← Mont_sub_Bn254Base r11 r0
      r12 := v26
      let v29 ← Mont_sub_Bn254Base r12 r2
      r13 := v29
      let v31 ← Mont_double_Bn254Base r13
      let v32 := v31
      r3 := v32
      let v34 ← Mont_double_Bn254Base r0
      r14 := v34
      let v37 ← Mont_add_Bn254Base r14 r0
      let v38 := v37
      r4 := v38
      let v40 ← Mont_square_Bn254Base r4
      r15 := v40
      let v42 ← Mont_double_Bn254Base r3
      r16 := v42
      let v45 ← Mont_sub_Bn254Base r15 r16
      let v46 := v45
      r5 := v46
      let v49 ← Mont_sub_Bn254Base r3 r5
      r17 := v49
      let v52 ← Mont_mul_Bn254Base r4 r17
      r18 := v52
      let v54 ← Mont_double_Bn254Base r2
      r19 := v54
      let v56 ← Mont_double_Bn254Base r19
      r20 := v56
      let v58 ← Mont_double_Bn254Base r20
      r21 := v58
      let v61 ← Mont_sub_Bn254Base r18 r21
      let v62 := v61
      let v63 := v0.z
      r22 := v63
      let v64 := v0.y
      r23 := v64
      let v67 ← Mont_mul_Bn254Base r23 r22
      r24 := v67
      let v69 ← Mont_double_Bn254Base r24
      let v70 := r5
      let v71 := (Point_Mont_Bn254Base.mk v70 v62 v69)
      return v71

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

def Bn254Base_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v0 := (UInt256.ofNat 4332616871279656263)
  let v1 := (UInt256.ofNat 2163322412065040948)
  let v2 := (UInt256.ofNat 361514372735272409)
  let v3 := (UInt256.ofNat 1806960190875503214)
  let v4 := (UInt256.ofNat 48)
  let v5 := #v[v0, v1, v2, v3, v4]
  return v5

def Bn254Base_INV62 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4048164856291499127)
  return v0

def Bn254Base_R3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 14921786541159648185948152738563080959093619838510245177710943249661917737183)
  return v0

def Mont_inverse_Bn254Base (v0 : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let v1 := v0.l0
  let v2 := v0.l1
  let v3 := v0.l2
  let v4 := v0.l3
  let v5 := #v[v1, v2, v3, v4]
  r0 := v5
  let v6 ← Bn254Base_SIGNED_62
  r1 := v6
  let v7 ← Bn254Base_INV62
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
  let v20 := (Mont_Bn254Base.mk v13 v15 v17 v19)
  r2 := v20
  let v21 ← Bn254Base_R3
  let v22 ← raw_Bn254Base v21
  r3 := v22
  let v25 ← Mont_mul_Bn254Base r2 r3
  return v25

def Point_to_affine_Mont_Bn254Base (v0 : Point_Mont_Bn254Base) : FeM (Mont_Bn254Base × Mont_Bn254Base) := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let mut r5 : Mont_Bn254Base := default
  let v1 := v0.z
  r2 := v1
  let v3 ← Mont_inverse_Bn254Base r2
  let v4 := v3
  r0 := v4
  let v6 ← Mont_square_Bn254Base r0
  let v7 := v6
  r1 := v7
  let v8 := v0.x
  r3 := v8
  let v11 ← Mont_mul_Bn254Base r3 r1
  let v14 ← Mont_mul_Bn254Base r1 r0
  r4 := v14
  let v15 := v0.y
  r5 := v15
  let v18 ← Mont_mul_Bn254Base r5 r4
  let v19 := (v11, v18)
  return v19

def Mont_value_Bn254Base (v0 : Mont_Bn254Base) : FeM UInt256 := do
  let mut r0 : Mont_Bn254Base := default
  let v1 := (UInt256.ofNat 1)
  let v2 ← raw_Bn254Base v1
  r0 := v2
  let v4 ← Mont_mul_Bn254Base v0 r0
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

def fp_value (v0 : Mont_Bn254Base) : FeM UInt256 := do
  let v1 ← Mont_value_Bn254Base v0
  return v1

def encode (v0 : Point_Mont_Bn254Base) : FeM (Vector UInt256 2) := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let v1 ← Point_is_infinity_Mont_Bn254Base v0
  if v1 then
    let v2 := (UInt256.ofNat 0)
    let v3 := (UInt256.ofNat 0)
    let v4 := #v[v2, v3]
    return v4
  else
    let v5 := ()
    let v6 := v5
    let v7 ← Point_to_affine_Mont_Bn254Base v0
    let v8 := v7.fst
    let v9 := v8
    r0 := v9
    let v10 := v7.snd
    let v11 := v10
    r1 := v11
    let v13 ← fp_value r0
    let v15 ← fp_value r1
    let v16 := #v[v13, v15]
    return v16

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


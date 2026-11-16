export Lib (MemBuffer MemBuffer_empty MemBuffer_index_ref MemBuffer_len MemBuffer_read_le_u64 Range_Known_1_Known_16 Range_Known_1_Known_16.mk Range_Known_1_Known_4 Range_Known_1_Known_4.mk Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start Range_at_1_16 Range_at_1_4 Range_has_1_16 Range_has_1_4 Range_next_1_16 Range_next_1_4 Range_start_1_16 Range_start_1_4 array_at_ref_usize_3 array_has_usize_3 array_next_usize_3 array_start_usize_3 downcast_truncate_i128_i64 downcast_truncate_i128_u64 downcast_truncate_i64_u64 downcast_truncate_u128_u64 downcast_truncate_u256_u64 downcast_truncate_u256_usize downcast_truncate_u64_i64 u128_wrapping_sub u64_wrapping_mul «assert»)

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
  | Some (f0_1 : PublicKey)
deriving Inhabited

def Option_PublicKey.tag : Option_PublicKey → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_PublicKey.Some_0 : Option_PublicKey → PublicKey
  | .Some f0_1 => f0_1
  | _ => default

def Option_PublicKey.set_Some_0 (x : Option_PublicKey) (v : PublicKey) : Option_PublicKey :=
  match x with
  | .Some f0_1 => .Some v
  | x => x

structure Transition where
  u : UInt256
  v : UInt256
  q : UInt256
  r : UInt256
deriving Inhabited

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
  | Some (f0_1 : Point_Mont_Bn254Base)
deriving Inhabited

def Option_Point_Mont_Bn254Base.tag : Option_Point_Mont_Bn254Base → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Point_Mont_Bn254Base.Some_0 : Option_Point_Mont_Bn254Base → Point_Mont_Bn254Base
  | .Some f0_1 => f0_1
  | _ => default

def Option_Point_Mont_Bn254Base.set_Some_0 (x : Option_Point_Mont_Bn254Base) (v : Point_Mont_Bn254Base) : Option_Point_Mont_Bn254Base :=
  match x with
  | .Some f0_1 => .Some v
  | x => x

structure Output where
  status : UInt256
  gas_used : UInt256
  output : MemBuffer
deriving Inhabited

def N : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907852837564279074904382605163141518161494337)
  return v0

def P : FeM UInt256 := do
  let v0 := (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007908834671663)
  return v0

def Base_VALUE : FeM UInt256 := do
  let v0 ← P
  return v0

def raw_Base (value_0 : UInt256) : FeM Mont_Base := do
  let v1 ← downcast_truncate_u256_u64 value_0
  let value_2 := value_0
  let v3 := (UInt256.ofNat 64)
  let v4 := value_2 >>> v3
  let v5 ← downcast_truncate_u256_u64 v4
  let value_6 := value_0
  let v7 := (UInt256.ofNat 128)
  let v8 := value_6 >>> v7
  let v9 ← downcast_truncate_u256_u64 v8
  let value_10 := value_0
  let v11 := (UInt256.ofNat 192)
  let v12 := value_10 >>> v11
  let v13 ← downcast_truncate_u256_u64 v12
  let v14 := (Mont_Base.mk v1 v5 v9 v13)
  return v14

def Base_R2 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446752466076602529)
  return v0

def mul_wide (a_0 : UInt256) (b_1 : UInt256) : FeM (UInt256 × UInt256) := do
  let a_2 := a_0
  let v3 := a_2
  let b_4 := b_1
  let v5 := b_4
  let v6 ← IntTy.mul (.u 128) v3 v5
  let t := v6
  let v8 ← downcast_truncate_u128_u64 t
  let v9 := (UInt256.ofNat 64)
  let v10 := t >>> v9
  let v11 ← downcast_truncate_u128_u64 v10
  let v12 := (v8, v11)
  return v12

def adc (a_0 : UInt256) (b_1 : UInt256) (carry_2 : UInt256) : FeM (UInt256 × UInt256) := do
  let a_3 := a_0
  let v4 := a_3
  let b_5 := b_1
  let v6 := b_5
  let v7 ← IntTy.add (.u 128) v4 v6
  let carry_8 := carry_2
  let v9 := carry_8
  let v10 ← IntTy.add (.u 128) v7 v9
  let t := v10
  let v12 ← downcast_truncate_u128_u64 t
  let v13 := (UInt256.ofNat 64)
  let v14 := t >>> v13
  let v15 ← downcast_truncate_u128_u64 v14
  let v16 := (v12, v15)
  return v16

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

def Mont_shift_reduced_Base_k3 (s5 : UInt256) (u1_34 : UInt256) (u2_50 : UInt256) (u3_66 : UInt256) (cb_73 : UInt256) (v75 : (UInt256 × UInt256)) (u4_77 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v78 := v75.snd
  let ca_79 := v78
  let v80 ← adc s5 cb_73 ca_79
  let v81 := v80.fst
  let u5 := v81
  let v83 := (Mont_Base.mk u1_34 u2_50 u3_66 u4_77)
  let v84 := (v83, u5)
  return v84

def Mont_shift_reduced_Base_k2 (s4 : UInt256) (s5 : UInt256) (m : UInt256) (u1_34 : UInt256) (u2_50 : UInt256) (ca_52 : UInt256) (v53 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  let v54 := v53.fst
  let u3_55 := v54
  let v56 := v53.snd
  let cb_57 := v56
  let v58 ← Base_P3
  let v59 ← mul_wide m v58
  let v60 := v59.fst
  let k3 := v60
  let v62 := v59.snd
  let j3 := v62
  let v64 ← adc u3_55 k3 ca_52
  let v65 := v64.fst
  let u3_66 := v65
  let v67 := v64.snd
  let ca_68 := v67
  let v69 ← adc s4 j3 cb_57
  let v70 := v69.fst
  let u4_71 := v70
  let v72 := v69.snd
  let cb_73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc u4_71 v74 ca_68
  let v76 := v75.fst
  let u4_77 := v76
  Mont_shift_reduced_Base_k3 s5 u1_34 u2_50 u3_66 cb_73 v75 u4_77

def Mont_shift_reduced_Base_k1 (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (ca_19 : UInt256) (u1_23 : UInt256) (cb_25 : UInt256) (v27 : (UInt256 × UInt256)) (k1 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v30 := v27.snd
  let j1 := v30
  let v32 ← adc u1_23 k1 ca_19
  let v33 := v32.fst
  let u1_34 := v33
  let v35 := v32.snd
  let ca_36 := v35
  let v37 ← adc s2 j1 cb_25
  let v38 := v37.fst
  let u2_39 := v38
  let v40 := v37.snd
  let cb_41 := v40
  let v42 ← Base_P2
  let v43 ← mul_wide m v42
  let v44 := v43.fst
  let k2 := v44
  let v46 := v43.snd
  let j2 := v46
  let v48 ← adc u2_39 k2 ca_36
  let v49 := v48.fst
  let u2_50 := v49
  let v51 := v48.snd
  let ca_52 := v51
  let v53 ← adc s3 j2 cb_41
  Mont_shift_reduced_Base_k2 s4 s5 m u1_34 u2_50 ca_52 v53

def Mont_shift_reduced_Base (s0_0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v6 ← Base_INV
  let s0_7 := s0_0
  let v8 ← u64_wrapping_mul s0_7 v6
  let m := v8
  let v10 ← Base_P0
  let v11 ← mul_wide m v10
  let v12 := v11.fst
  let k0 := v12
  let v14 := v11.snd
  let j0 := v14
  let v16 := (UInt256.ofNat 0)
  let v17 ← adc s0_0 k0 v16
  let v18 := v17.snd
  let ca_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc s1 j0 v20
  let v22 := v21.fst
  let u1_23 := v22
  let v24 := v21.snd
  let cb_25 := v24
  let v26 ← Base_P1
  let v27 ← mul_wide m v26
  let v28 := v27.fst
  let k1 := v28
  Mont_shift_reduced_Base_k1 s2 s3 s4 s5 m ca_19 u1_23 cb_25 v27 k1

def Mont_first_row_Base_k1 (self : Mont_Base) (b : UInt256) (s0 : UInt256) (h1 : UInt256) (s1 : UInt256) (c_19 : UInt256) (l2 : UInt256) (h2 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v26 ← adc h1 l2 c_19
  let v27 := v26.fst
  let s2 := v27
  let v29 := v26.snd
  let c_30 := v29
  let v31 := self.l3
  let v32 ← mul_wide b v31
  let v33 := v32.fst
  let l3 := v33
  let v35 := v32.snd
  let h3 := v35
  let v37 ← adc h2 l3 c_30
  let v38 := v37.fst
  let s3 := v38
  let v40 := v37.snd
  let c_41 := v40
  let v42 := (UInt256.ofNat 0)
  let v43 ← adc h3 v42 c_41
  let v44 := v43.fst
  let s4 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 ← Mont_shift_reduced_Base s0 s1 s2 s3 s4 v46
  return v47

def Mont_first_row_Base (self : Mont_Base) (b : UInt256) : FeM (Mont_Base × UInt256) := do
  let v2 := self.l0
  let v3 ← mul_wide b v2
  let v4 := v3.fst
  let s0 := v4
  let v6 := v3.snd
  let h0 := v6
  let v8 := self.l1
  let v9 ← mul_wide b v8
  let v10 := v9.fst
  let l1 := v10
  let v12 := v9.snd
  let h1 := v12
  let v14 := (UInt256.ofNat 0)
  let v15 ← adc h0 l1 v14
  let v16 := v15.fst
  let s1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := self.l2
  let v21 ← mul_wide b v20
  let v22 := v21.fst
  let l2 := v22
  let v24 := v21.snd
  let h2 := v24
  Mont_first_row_Base_k1 self b s0 h1 s1 c_19 l2 h2

def Mont_row_Base_k3 (s0 : UInt256) (s1_32 : UInt256) (s2_49 : UInt256) (s3_66 : UInt256) (cb_73 : UInt256) (v75 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  let v76 := v75.fst
  let s4_77 := v76
  let v78 := v75.snd
  let ca_79 := v78
  let v80 := (UInt256.ofNat 0)
  let v81 ← adc cb_73 v80 ca_79
  let v82 := v81.fst
  let s5 := v82
  let v84 ← Mont_shift_reduced_Base s0 s1_32 s2_49 s3_66 s4_77 s5
  return v84

def Mont_row_Base_k2 (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) (s0 : UInt256) (s1_32 : UInt256) (cb_40 : UInt256) (h2 : UInt256) (s2_49 : UInt256) (ca_51 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v52 := t.l3
  let v53 ← adc v52 h2 cb_40
  let v54 := v53.fst
  let s3_55 := v54
  let v56 := v53.snd
  let cb_57 := v56
  let v58 := self.l3
  let v59 ← mul_wide b v58
  let v60 := v59.fst
  let l3 := v60
  let v62 := v59.snd
  let h3 := v62
  let v64 ← adc s3_55 l3 ca_51
  let v65 := v64.fst
  let s3_66 := v65
  let v67 := v64.snd
  let ca_68 := v67
  let v69 ← adc top h3 cb_57
  let v70 := v69.fst
  let s4_71 := v70
  let v72 := v69.snd
  let cb_73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc s4_71 v74 ca_68
  Mont_row_Base_k3 s0 s1_32 s2_49 s3_66 cb_73 v75

def Mont_row_Base_k1 (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) (s0 : UInt256) (ca_16 : UInt256) (s1_21 : UInt256) (cb_23 : UInt256) (v25 : (UInt256 × UInt256)) (l1 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v28 := v25.snd
  let h1 := v28
  let v30 ← adc s1_21 l1 ca_16
  let v31 := v30.fst
  let s1_32 := v31
  let v33 := v30.snd
  let ca_34 := v33
  let v35 := t.l2
  let v36 ← adc v35 h1 cb_23
  let v37 := v36.fst
  let s2_38 := v37
  let v39 := v36.snd
  let cb_40 := v39
  let v41 := self.l2
  let v42 ← mul_wide b v41
  let v43 := v42.fst
  let l2 := v43
  let v45 := v42.snd
  let h2 := v45
  let v47 ← adc s2_38 l2 ca_34
  let v48 := v47.fst
  let s2_49 := v48
  let v50 := v47.snd
  let ca_51 := v50
  Mont_row_Base_k2 self b t top s0 s1_32 cb_40 h2 s2_49 ca_51

def Mont_row_Base (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) : FeM (Mont_Base × UInt256) := do
  let v4 := self.l0
  let v5 ← mul_wide b v4
  let v6 := v5.fst
  let l0 := v6
  let v8 := v5.snd
  let h0 := v8
  let v10 := t.l0
  let v11 := (UInt256.ofNat 0)
  let v12 ← adc v10 l0 v11
  let v13 := v12.fst
  let s0 := v13
  let v15 := v12.snd
  let ca_16 := v15
  let v17 := t.l1
  let v18 := (UInt256.ofNat 0)
  let v19 ← adc v17 h0 v18
  let v20 := v19.fst
  let s1_21 := v20
  let v22 := v19.snd
  let cb_23 := v22
  let v24 := self.l1
  let v25 ← mul_wide b v24
  let v26 := v25.fst
  let l1 := v26
  Mont_row_Base_k1 self b t top s0 ca_16 s1_21 cb_23 v25 l1

def sbb (a_0 : UInt256) (b_1 : UInt256) (borrow_2 : UInt256) : FeM (UInt256 × UInt256) := do
  let a_3 := a_0
  let v4 := a_3
  let b_5 := b_1
  let v6 := b_5
  let v7 ← u128_wrapping_sub v4 v6
  let borrow_8 := borrow_2
  let v9 := borrow_8
  let v10 ← u128_wrapping_sub v7 v9
  let t := v10
  let v12 ← downcast_truncate_u128_u64 t
  let v13 := (UInt256.ofNat 127)
  let v14 := t >>> v13
  let v15 ← downcast_truncate_u128_u64 v14
  let v16 := (v12, v15)
  return v16

def Mont_reduce_Base (self : Mont_Base) (top : UInt256) : FeM Mont_Base := do
  let v3 := self.l0
  let v4 ← Base_P0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let d0 := v7
  let v9 := v6.snd
  let b_10 := v9
  let v11 := self.l1
  let v12 ← Base_P1
  let v13 ← sbb v11 v12 b_10
  let v14 := v13.fst
  let d1 := v14
  let v16 := v13.snd
  let b_17 := v16
  let v18 := self.l2
  let v19 ← Base_P2
  let v20 ← sbb v18 v19 b_17
  let v21 := v20.fst
  let d2 := v21
  let v23 := v20.snd
  let b_24 := v23
  let v25 := self.l3
  let v26 ← Base_P3
  let v27 ← sbb v25 v26 b_24
  let v28 := v27.fst
  let d3 := v28
  let v30 := v27.snd
  let b_31 := v30
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb top v32 b_31
  let v34 := v33.snd
  let b_35 := v34
  let v36 := (UInt256.ofNat 1)
  let v37 := decide (b_35 = v36)
  let b3 : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  if v37 then
    let v38 := self
    let v39 := v38
    b3 v39
  else
    let v40 := (Mont_Base.mk d0 d1 d2 d3)
    let v41 := v40
    b3 v41

def Mont_mul_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let mut t_r0 : Mont_Base := default
  let mut t_r1 : Mont_Base := default
  let mut t_r2 : Mont_Base := default
  let mut t_r3 : Mont_Base := default
  let v2 := other.l0
  let v3 ← Mont_first_row_Base self v2
  let v4 := v3.fst
  let t_5 := v4
  t_r0 := t_5
  let v6 := v3.snd
  let top_7 := v6
  let v8 := other.l1
  let v10 ← Mont_row_Base self v8 t_r0 top_7
  let v11 := v10.fst
  let t_12 := v11
  t_r1 := t_12
  let v13 := v10.snd
  let top_14 := v13
  let v15 := other.l2
  let v17 ← Mont_row_Base self v15 t_r1 top_14
  let v18 := v17.fst
  let t_19 := v18
  t_r2 := t_19
  let v20 := v17.snd
  let top_21 := v20
  let v22 := other.l3
  let v24 ← Mont_row_Base self v22 t_r2 top_21
  let v25 := v24.fst
  let t_26 := v25
  t_r3 := t_26
  let v27 := v24.snd
  let top_28 := v27
  let v30 ← Mont_reduce_Base t_r3 top_28
  return v30

def Mont_new_Base (value_0 : UInt256) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let value_1 := value_0
  let v2 ← Base_VALUE
  let v3 := decide (value_1.toNat < v2.toNat)
  let v4 ← «assert» v3
  let v5 ← raw_Base value_0
  r0 := v5
  let v6 ← Base_R2
  let v7 ← raw_Base v6
  r1 := v7
  let v10 ← Mont_mul_Base r0 r1
  return v10

def Mont_square_Base (self : Mont_Base) : FeM Mont_Base := do
  let v1 ← Mont_mul_Base self self
  return v1

def Mont_add_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let v2 := self.l0
  let v3 := other.l0
  let v4 := (UInt256.ofNat 0)
  let v5 ← adc v2 v3 v4
  let v6 := v5.fst
  let s0 := v6
  let v8 := v5.snd
  let c_9 := v8
  let v10 := self.l1
  let v11 := other.l1
  let v12 ← adc v10 v11 c_9
  let v13 := v12.fst
  let s1 := v13
  let v15 := v12.snd
  let c_16 := v15
  let v17 := self.l2
  let v18 := other.l2
  let v19 ← adc v17 v18 c_16
  let v20 := v19.fst
  let s2 := v20
  let v22 := v19.snd
  let c_23 := v22
  let v24 := self.l3
  let v25 := other.l3
  let v26 ← adc v24 v25 c_23
  let v27 := v26.fst
  let s3 := v27
  let v29 := v26.snd
  let c_30 := v29
  let v31 := (Mont_Base.mk s0 s1 s2 s3)
  r0 := v31
  let v33 ← Mont_reduce_Base r0 c_30
  return v33

def Mont_one_Base : FeM Mont_Base := do
  let v0 := (UInt256.ofNat 1)
  let v1 ← Mont_new_Base v0
  return v1

def Mont_pow_Base (self : Mont_Base) (exponent_1 : UInt256) : FeM Mont_Base := do
  let mut result : Mont_Base := default
  let mut bit : UInt256 := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let v2 ← Mont_one_Base
  let result_3 := v2
  result := result_3
  let v4 := (UInt256.ofNat 256)
  let bit_5 := v4
  bit := bit_5
  let rec b1 (result : Mont_Base) (bit : UInt256) (r2 : Mont_Base) (r3 : Mont_Base) : FeM Mont_Base := do
    let mut result := result
    let mut bit := bit
    let mut r2 := r2
    let mut r3 := r3
    let bit_6 := bit
    let bit_7 := bit_6
    let v8 := (UInt256.ofNat 0)
    let v9 := decide (v8.toNat < bit_7.toNat)
    if v9 then
      let v10 := (UInt256.ofNat 1)
      let bit_11 := bit
      let v12 ← IntTy.sub (.u 64) bit_11 v10
      let bit_13 := v12
      bit := bit_13
      let v14 := ()
      let result_15 := result
      let result_16 := result_15
      r2 := result_16
      let v18 ← Mont_square_Base r2
      let result_19 := v18
      result := result_19
      let v20 := ()
      let exponent_21 := exponent_1
      let bit_22 := bit
      let v23 := bit_22
      let v24 := exponent_21 >>> v23
      let v25 := (UInt256.ofNat 1)
      let v26 := v24 &&& v25
      let v27 := (UInt256.ofNat 1)
      let v28 := decide (v26 = v27)
      let b6 : Mont_Base → UInt256 → Mont_Base → Mont_Base → FeM Mont_Base := fun result bit r2 r3 => do
        let mut result := result
        let mut bit := bit
        let mut r2 := r2
        let mut r3 := r3
        b1 result bit r2 r3
      if v28 then
        let result_29 := result
        let result_30 := result_29
        r3 := result_30
        let v32 ← Mont_mul_Base r3 self
        let result_33 := v32
        result := result_33
        let v34 := ()
        let v35 := v34
        b6 result bit r2 r3
      else
        let v36 := ()
        let v37 := v36
        b6 result bit r2 r3
    else
      let result_38 := result
      return result_38
  partial_fixpoint
  b1 result bit r2 r3

def Mont_equals_Base (self : Mont_Base) (other : Mont_Base) : FeM Bool := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := decide (v3 = v4)
  let b3 : Bool → FeM Bool := fun v2 => do
    return v2
  let b2 : FeM Bool := do
    let v17 := false
    let v18 := v17
    b3 v18
  if v5 then
    let v6 := self.l1
    let v7 := other.l1
    let v8 := decide (v6 = v7)
    if v8 then
      let v9 := self.l2
      let v10 := other.l2
      let v11 := decide (v9 = v10)
      if v11 then
        let v12 := self.l3
        let v13 := other.l3
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

def Mont_value_Base (self : Mont_Base) : FeM UInt256 := do
  let mut r0 : Mont_Base := default
  let v1 := (UInt256.ofNat 1)
  let v2 ← raw_Base v1
  r0 := v2
  let v4 ← Mont_mul_Base self r0
  let value := v4
  let v6 := value.l0
  let v7 := v6
  let v8 := value.l1
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := v9 <<< v10
  let v12 := v7 ||| v11
  let v13 := value.l2
  let v14 := v13
  let v15 := (UInt256.ofNat 128)
  let v16 := v14 <<< v15
  let v17 := v12 ||| v16
  let v18 := value.l3
  let v19 := v18
  let v20 := (UInt256.ofNat 192)
  let v21 := v19 <<< v20
  let v22 := v17 ||| v21
  return v22

def Order_VALUE : FeM UInt256 := do
  let v0 ← N
  return v0

def raw_Order (value_0 : UInt256) : FeM Mont_Order := do
  let v1 ← downcast_truncate_u256_u64 value_0
  let value_2 := value_0
  let v3 := (UInt256.ofNat 64)
  let v4 := value_2 >>> v3
  let v5 ← downcast_truncate_u256_u64 v4
  let value_6 := value_0
  let v7 := (UInt256.ofNat 128)
  let v8 := value_6 >>> v7
  let v9 ← downcast_truncate_u256_u64 v8
  let value_10 := value_0
  let v11 := (UInt256.ofNat 192)
  let v12 := value_10 >>> v11
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

def Mont_shift_reduced_Order_k3 (s5 : UInt256) (u1_34 : UInt256) (u2_50 : UInt256) (u3_66 : UInt256) (cb_73 : UInt256) (v75 : (UInt256 × UInt256)) (u4_77 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v78 := v75.snd
  let ca_79 := v78
  let v80 ← adc s5 cb_73 ca_79
  let v81 := v80.fst
  let u5 := v81
  let v83 := (Mont_Order.mk u1_34 u2_50 u3_66 u4_77)
  let v84 := (v83, u5)
  return v84

def Mont_shift_reduced_Order_k2 (s4 : UInt256) (s5 : UInt256) (m : UInt256) (u1_34 : UInt256) (u2_50 : UInt256) (ca_52 : UInt256) (v53 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  let v54 := v53.fst
  let u3_55 := v54
  let v56 := v53.snd
  let cb_57 := v56
  let v58 ← Order_P3
  let v59 ← mul_wide m v58
  let v60 := v59.fst
  let k3 := v60
  let v62 := v59.snd
  let j3 := v62
  let v64 ← adc u3_55 k3 ca_52
  let v65 := v64.fst
  let u3_66 := v65
  let v67 := v64.snd
  let ca_68 := v67
  let v69 ← adc s4 j3 cb_57
  let v70 := v69.fst
  let u4_71 := v70
  let v72 := v69.snd
  let cb_73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc u4_71 v74 ca_68
  let v76 := v75.fst
  let u4_77 := v76
  Mont_shift_reduced_Order_k3 s5 u1_34 u2_50 u3_66 cb_73 v75 u4_77

def Mont_shift_reduced_Order_k1 (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (ca_19 : UInt256) (u1_23 : UInt256) (cb_25 : UInt256) (v27 : (UInt256 × UInt256)) (k1 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v30 := v27.snd
  let j1 := v30
  let v32 ← adc u1_23 k1 ca_19
  let v33 := v32.fst
  let u1_34 := v33
  let v35 := v32.snd
  let ca_36 := v35
  let v37 ← adc s2 j1 cb_25
  let v38 := v37.fst
  let u2_39 := v38
  let v40 := v37.snd
  let cb_41 := v40
  let v42 ← Order_P2
  let v43 ← mul_wide m v42
  let v44 := v43.fst
  let k2 := v44
  let v46 := v43.snd
  let j2 := v46
  let v48 ← adc u2_39 k2 ca_36
  let v49 := v48.fst
  let u2_50 := v49
  let v51 := v48.snd
  let ca_52 := v51
  let v53 ← adc s3 j2 cb_41
  Mont_shift_reduced_Order_k2 s4 s5 m u1_34 u2_50 ca_52 v53

def Mont_shift_reduced_Order (s0_0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v6 ← Order_INV
  let s0_7 := s0_0
  let v8 ← u64_wrapping_mul s0_7 v6
  let m := v8
  let v10 ← Order_P0
  let v11 ← mul_wide m v10
  let v12 := v11.fst
  let k0 := v12
  let v14 := v11.snd
  let j0 := v14
  let v16 := (UInt256.ofNat 0)
  let v17 ← adc s0_0 k0 v16
  let v18 := v17.snd
  let ca_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc s1 j0 v20
  let v22 := v21.fst
  let u1_23 := v22
  let v24 := v21.snd
  let cb_25 := v24
  let v26 ← Order_P1
  let v27 ← mul_wide m v26
  let v28 := v27.fst
  let k1 := v28
  Mont_shift_reduced_Order_k1 s2 s3 s4 s5 m ca_19 u1_23 cb_25 v27 k1

def Mont_first_row_Order_k1 (self : Mont_Order) (b : UInt256) (s0 : UInt256) (h1 : UInt256) (s1 : UInt256) (c_19 : UInt256) (l2 : UInt256) (h2 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v26 ← adc h1 l2 c_19
  let v27 := v26.fst
  let s2 := v27
  let v29 := v26.snd
  let c_30 := v29
  let v31 := self.l3
  let v32 ← mul_wide b v31
  let v33 := v32.fst
  let l3 := v33
  let v35 := v32.snd
  let h3 := v35
  let v37 ← adc h2 l3 c_30
  let v38 := v37.fst
  let s3 := v38
  let v40 := v37.snd
  let c_41 := v40
  let v42 := (UInt256.ofNat 0)
  let v43 ← adc h3 v42 c_41
  let v44 := v43.fst
  let s4 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 ← Mont_shift_reduced_Order s0 s1 s2 s3 s4 v46
  return v47

def Mont_first_row_Order (self : Mont_Order) (b : UInt256) : FeM (Mont_Order × UInt256) := do
  let v2 := self.l0
  let v3 ← mul_wide b v2
  let v4 := v3.fst
  let s0 := v4
  let v6 := v3.snd
  let h0 := v6
  let v8 := self.l1
  let v9 ← mul_wide b v8
  let v10 := v9.fst
  let l1 := v10
  let v12 := v9.snd
  let h1 := v12
  let v14 := (UInt256.ofNat 0)
  let v15 ← adc h0 l1 v14
  let v16 := v15.fst
  let s1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := self.l2
  let v21 ← mul_wide b v20
  let v22 := v21.fst
  let l2 := v22
  let v24 := v21.snd
  let h2 := v24
  Mont_first_row_Order_k1 self b s0 h1 s1 c_19 l2 h2

def Mont_row_Order_k3 (s0 : UInt256) (s1_32 : UInt256) (s2_49 : UInt256) (s3_66 : UInt256) (cb_73 : UInt256) (v75 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  let v76 := v75.fst
  let s4_77 := v76
  let v78 := v75.snd
  let ca_79 := v78
  let v80 := (UInt256.ofNat 0)
  let v81 ← adc cb_73 v80 ca_79
  let v82 := v81.fst
  let s5 := v82
  let v84 ← Mont_shift_reduced_Order s0 s1_32 s2_49 s3_66 s4_77 s5
  return v84

def Mont_row_Order_k2 (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) (s0 : UInt256) (s1_32 : UInt256) (cb_40 : UInt256) (h2 : UInt256) (s2_49 : UInt256) (ca_51 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v52 := t.l3
  let v53 ← adc v52 h2 cb_40
  let v54 := v53.fst
  let s3_55 := v54
  let v56 := v53.snd
  let cb_57 := v56
  let v58 := self.l3
  let v59 ← mul_wide b v58
  let v60 := v59.fst
  let l3 := v60
  let v62 := v59.snd
  let h3 := v62
  let v64 ← adc s3_55 l3 ca_51
  let v65 := v64.fst
  let s3_66 := v65
  let v67 := v64.snd
  let ca_68 := v67
  let v69 ← adc top h3 cb_57
  let v70 := v69.fst
  let s4_71 := v70
  let v72 := v69.snd
  let cb_73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc s4_71 v74 ca_68
  Mont_row_Order_k3 s0 s1_32 s2_49 s3_66 cb_73 v75

def Mont_row_Order_k1 (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) (s0 : UInt256) (ca_16 : UInt256) (s1_21 : UInt256) (cb_23 : UInt256) (v25 : (UInt256 × UInt256)) (l1 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v28 := v25.snd
  let h1 := v28
  let v30 ← adc s1_21 l1 ca_16
  let v31 := v30.fst
  let s1_32 := v31
  let v33 := v30.snd
  let ca_34 := v33
  let v35 := t.l2
  let v36 ← adc v35 h1 cb_23
  let v37 := v36.fst
  let s2_38 := v37
  let v39 := v36.snd
  let cb_40 := v39
  let v41 := self.l2
  let v42 ← mul_wide b v41
  let v43 := v42.fst
  let l2 := v43
  let v45 := v42.snd
  let h2 := v45
  let v47 ← adc s2_38 l2 ca_34
  let v48 := v47.fst
  let s2_49 := v48
  let v50 := v47.snd
  let ca_51 := v50
  Mont_row_Order_k2 self b t top s0 s1_32 cb_40 h2 s2_49 ca_51

def Mont_row_Order (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) : FeM (Mont_Order × UInt256) := do
  let v4 := self.l0
  let v5 ← mul_wide b v4
  let v6 := v5.fst
  let l0 := v6
  let v8 := v5.snd
  let h0 := v8
  let v10 := t.l0
  let v11 := (UInt256.ofNat 0)
  let v12 ← adc v10 l0 v11
  let v13 := v12.fst
  let s0 := v13
  let v15 := v12.snd
  let ca_16 := v15
  let v17 := t.l1
  let v18 := (UInt256.ofNat 0)
  let v19 ← adc v17 h0 v18
  let v20 := v19.fst
  let s1_21 := v20
  let v22 := v19.snd
  let cb_23 := v22
  let v24 := self.l1
  let v25 ← mul_wide b v24
  let v26 := v25.fst
  let l1 := v26
  Mont_row_Order_k1 self b t top s0 ca_16 s1_21 cb_23 v25 l1

def Mont_reduce_Order (self : Mont_Order) (top : UInt256) : FeM Mont_Order := do
  let v3 := self.l0
  let v4 ← Order_P0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let d0 := v7
  let v9 := v6.snd
  let b_10 := v9
  let v11 := self.l1
  let v12 ← Order_P1
  let v13 ← sbb v11 v12 b_10
  let v14 := v13.fst
  let d1 := v14
  let v16 := v13.snd
  let b_17 := v16
  let v18 := self.l2
  let v19 ← Order_P2
  let v20 ← sbb v18 v19 b_17
  let v21 := v20.fst
  let d2 := v21
  let v23 := v20.snd
  let b_24 := v23
  let v25 := self.l3
  let v26 ← Order_P3
  let v27 ← sbb v25 v26 b_24
  let v28 := v27.fst
  let d3 := v28
  let v30 := v27.snd
  let b_31 := v30
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb top v32 b_31
  let v34 := v33.snd
  let b_35 := v34
  let v36 := (UInt256.ofNat 1)
  let v37 := decide (b_35 = v36)
  let b3 : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  if v37 then
    let v38 := self
    let v39 := v38
    b3 v39
  else
    let v40 := (Mont_Order.mk d0 d1 d2 d3)
    let v41 := v40
    b3 v41

def Mont_mul_Order (self : Mont_Order) (other : Mont_Order) : FeM Mont_Order := do
  let mut t_r0 : Mont_Order := default
  let mut t_r1 : Mont_Order := default
  let mut t_r2 : Mont_Order := default
  let mut t_r3 : Mont_Order := default
  let v2 := other.l0
  let v3 ← Mont_first_row_Order self v2
  let v4 := v3.fst
  let t_5 := v4
  t_r0 := t_5
  let v6 := v3.snd
  let top_7 := v6
  let v8 := other.l1
  let v10 ← Mont_row_Order self v8 t_r0 top_7
  let v11 := v10.fst
  let t_12 := v11
  t_r1 := t_12
  let v13 := v10.snd
  let top_14 := v13
  let v15 := other.l2
  let v17 ← Mont_row_Order self v15 t_r1 top_14
  let v18 := v17.fst
  let t_19 := v18
  t_r2 := t_19
  let v20 := v17.snd
  let top_21 := v20
  let v22 := other.l3
  let v24 ← Mont_row_Order self v22 t_r2 top_21
  let v25 := v24.fst
  let t_26 := v25
  t_r3 := t_26
  let v27 := v24.snd
  let top_28 := v27
  let v30 ← Mont_reduce_Order t_r3 top_28
  return v30

def Mont_new_Order (value_0 : UInt256) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let mut r1 : Mont_Order := default
  let value_1 := value_0
  let v2 ← Order_VALUE
  let v3 := decide (value_1.toNat < v2.toNat)
  let v4 ← «assert» v3
  let v5 ← raw_Order value_0
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

def M62 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4611686018427387903)
  return v0

def is_zero_5 (r : (Vector UInt256 5)) : FeM Bool := do
  let mut i : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let i_2 := v1
  i := i_2
  let rec b1 (i : UInt256) : FeM Bool := do
    let mut i := i
    let i_3 := i
    let i_4 := i_3
    let v5 := (UInt256.ofNat 5)
    let v6 := decide (i_4.toNat < v5.toNat)
    if v6 then
      let i_7 := i
      checkIndex i_7 5
      let v8 := r[i_7.toNat]!
      let v9 := (UInt256.ofNat 0)
      let v10 := decide (v8 ≠ v9)
      if v10 then
        let v11 := false
        return v11
      else
        let v12 := ()
        let v13 := v12
        let v14 := (UInt256.ofNat 1)
        let i_15 := i
        let v16 ← checkedAdd i_15 v14
        let i_17 := v16
        i := i_17
        let v18 := ()
        b1 i
    else
      let v19 := true
      return v19
  partial_fixpoint
  b1 i

def divsteps (eta_0 : UInt256) (f0_1 : UInt256) (g0_2 : UInt256) : FeM (UInt256 × Transition) := do
  let mut u : UInt256 := default
  let mut v : UInt256 := default
  let mut q : UInt256 := default
  let mut r : UInt256 := default
  let mut f : UInt256 := default
  let mut g : UInt256 := default
  let mut eta : UInt256 := default
  let mut left : UInt256 := default
  let mut bits : UInt256 := default
  let v5 := (UInt256.ofNat 1)
  let u_6 := v5
  u := u_6
  let v7 := (UInt256.ofNat 0)
  let v_8 := v7
  v := v_8
  let v9 := (UInt256.ofNat 0)
  let q_10 := v9
  q := q_10
  let v11 := (UInt256.ofNat 1)
  let r_12 := v11
  r := r_12
  let f0_13 := f0_1
  let f0_14 := f0_13
  f := f0_14
  let g0_15 := g0_2
  let g0_16 := g0_15
  g := g0_16
  let eta_17 := eta_0
  let eta_18 := eta_17
  eta := eta_18
  let v19 := (UInt256.ofNat 62)
  let left_20 := v19
  left := left_20
  let rec b1 (u : UInt256) (v : UInt256) (q : UInt256) (r : UInt256) (f : UInt256) (g : UInt256) (eta : UInt256) (left : UInt256) (bits : UInt256) : FeM (UInt256 × Transition) := do
    let mut u := u
    let mut v := v
    let mut q := q
    let mut r := r
    let mut f := f
    let mut g := g
    let mut eta := eta
    let mut left := left
    let mut bits := bits
    let g_21 := g
    let g_22 := g_21
    let v23 := (UInt256.ofNat 18446744073709551615)
    let left_24 := left
    let v25 := IntTy.shl (.u 64) v23 left_24
    let v26 := g_22 ||| v25
    let v27 := IntTy.trailingZeros (.u 64) v26
    let zeros := v27
    let g_29 := g
    let g_30 := g_29
    let v31 := g_30 >>> zeros
    let g_32 := v31
    g := g_32
    let v33 := ()
    let u_34 := u
    let u_35 := u_34
    let v36 := IntTy.shl (.u 64) u_35 zeros
    let u_37 := v36
    u := u_37
    let v38 := ()
    let v_39 := v
    let v_40 := v_39
    let v41 := IntTy.shl (.u 64) v_40 zeros
    let v_42 := v41
    v := v_42
    let v43 := ()
    let v44 ← downcast_truncate_u64_i64 zeros
    let eta_45 := eta
    let v46 := IntTy.subW (.i 64) eta_45 v44
    let eta_47 := v46
    eta := eta_47
    let v48 := ()
    let left_49 := left
    let v50 := IntTy.subW (.u 64) left_49 zeros
    let left_51 := v50
    left := left_51
    let v52 := ()
    let left_53 := left
    let left_54 := left_53
    let v55 := (UInt256.ofNat 0)
    let v56 := decide (left_54 = v55)
    if v56 then
      let eta_57 := eta
      let eta_58 := eta_57
      let u_59 := u
      let u_60 := u_59
      let v61 ← downcast_truncate_u64_i64 u_60
      let v_62 := v
      let v_63 := v_62
      let v64 ← downcast_truncate_u64_i64 v_63
      let q_65 := q
      let q_66 := q_65
      let v67 ← downcast_truncate_u64_i64 q_66
      let r_68 := r
      let r_69 := r_68
      let v70 ← downcast_truncate_u64_i64 r_69
      let v71 := (Transition.mk v61 v64 v67 v70)
      let v72 := (eta_58, v71)
      return v72
    else
      let v73 := ()
      let v74 := v73
      let v75 := (UInt256.ofNat 15)
      let bits_76 := v75
      bits := bits_76
      let eta_77 := eta
      let eta_78 := eta_77
      let v79 := (UInt256.ofNat 0)
      let v80 := UInt256.sltBool eta_78 v79
      let b9 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun u v q r f g eta left bits => do
        let mut u := u
        let mut v := v
        let mut q := q
        let mut r := r
        let mut f := f
        let mut g := g
        let mut eta := eta
        let mut left := left
        let mut bits := bits
        let eta_122 := eta
        let eta_123 := eta_122
        let v124 := (UInt256.ofNat 1)
        let v125 := IntTy.addW (.i 64) eta_123 v124
        let v126 ← downcast_truncate_i64_u64 v125
        let room_127 := v126
        let left_128 := left
        let v129 := decide (room_127.toNat < left_128.toNat)
        let b12 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun u v q r f g eta left bits v3 => do
          let mut u := u
          let mut v := v
          let mut q := q
          let mut r := r
          let mut f := f
          let mut g := g
          let mut eta := eta
          let mut left := left
          let mut bits := bits
          let limit := v3
          let v135 := (UInt256.ofNat 18446744073709551615)
          let v136 := (UInt256.ofNat 64)
          let v137 := IntTy.subW (.u 64) v136 limit
          let v138 := v135 >>> v137
          let bits_139 := bits
          let v140 := v138 &&& bits_139
          let mask := v140
          let bits_142 := bits
          let bits_143 := bits_142
          let v144 := (UInt256.ofNat 63)
          let v145 := decide (bits_143 = v144)
          let b15 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun u v q r f g eta left bits v4 => do
            let mut u := u
            let mut v := v
            let mut q := q
            let mut r := r
            let mut f := f
            let mut g := g
            let mut eta := eta
            let mut left := left
            let mut bits := bits
            let w := v4
            let f_177 := f
            let f_178 := f_177
            let v179 := IntTy.mulW (.u 64) f_178 w
            let g_180 := g
            let v181 := IntTy.addW (.u 64) g_180 v179
            let g_182 := v181
            g := g_182
            let v183 := ()
            let u_184 := u
            let u_185 := u_184
            let v186 := IntTy.mulW (.u 64) u_185 w
            let q_187 := q
            let v188 := IntTy.addW (.u 64) q_187 v186
            let q_189 := v188
            q := q_189
            let v190 := ()
            let v_191 := v
            let v_192 := v_191
            let v193 := IntTy.mulW (.u 64) v_192 w
            let r_194 := r
            let v195 := IntTy.addW (.u 64) r_194 v193
            let r_196 := v195
            r := r_196
            let v197 := ()
            b1 u v q r f g eta left bits
          if v145 then
            let f_146 := f
            let f_147 := f_146
            let g_148 := g
            let v149 := IntTy.mulW (.u 64) f_147 g_148
            let f_150 := f
            let f_151 := f_150
            let f_152 := f
            let v153 := IntTy.mulW (.u 64) f_151 f_152
            let v154 := (UInt256.ofNat 2)
            let v155 := IntTy.subW (.u 64) v153 v154
            let v156 := IntTy.mulW (.u 64) v149 v155
            let v157 := v156 &&& mask
            let v158 := v157
            b15 u v q r f g eta left bits v158
          else
            let v159 := (UInt256.ofNat 0)
            let f_160 := f
            let f_161 := f_160
            let f_162 := f
            let f_163 := f_162
            let v164 := (UInt256.ofNat 1)
            let v165 := IntTy.addW (.u 64) f_163 v164
            let v166 := (UInt256.ofNat 4)
            let v167 := v165 &&& v166
            let v168 := (UInt256.ofNat 1)
            let v169 := IntTy.shl (.u 64) v167 v168
            let v170 := IntTy.addW (.u 64) f_161 v169
            let v171 := IntTy.subW (.u 64) v159 v170
            let g_172 := g
            let v173 := IntTy.mulW (.u 64) v171 g_172
            let v174 := v173 &&& mask
            let v175 := v174
            b15 u v q r f g eta left bits v175
        if v129 then
          let room_130 := room_127
          b12 u v q r f g eta left bits room_130
        else
          let left_131 := left
          let left_132 := left_131
          let left_133 := left_132
          b12 u v q r f g eta left bits left_133
      if v80 then
        let v81 := (UInt256.ofNat 0)
        let eta_82 := eta
        let v83 := IntTy.subW (.i 64) v81 eta_82
        let eta_84 := v83
        eta := eta_84
        let v85 := ()
        let f_86 := f
        let f_87 := f_86
        let g_88 := g
        let g_89 := g_88
        let g_90 := g_89
        f := g_90
        let v91 := ()
        let v92 := (UInt256.ofNat 0)
        let v93 := IntTy.subW (.u 64) v92 f_87
        let g_94 := v93
        g := g_94
        let v95 := ()
        let u_96 := u
        let u_97 := u_96
        let q_98 := q
        let q_99 := q_98
        let q_100 := q_99
        u := q_100
        let v101 := ()
        let v102 := (UInt256.ofNat 0)
        let v103 := IntTy.subW (.u 64) v102 u_97
        let q_104 := v103
        q := q_104
        let v105 := ()
        let v_106 := v
        let v_107 := v_106
        let r_108 := r
        let r_109 := r_108
        let r_110 := r_109
        v := r_110
        let v111 := ()
        let v112 := (UInt256.ofNat 0)
        let v113 := IntTy.subW (.u 64) v112 v_107
        let r_114 := v113
        r := r_114
        let v115 := ()
        let v116 := (UInt256.ofNat 63)
        let bits_117 := v116
        bits := bits_117
        let v118 := ()
        let v119 := v118
        b9 u v q r f g eta left bits
      else
        let v120 := ()
        let v121 := v120
        b9 u v q r f g eta left bits
  partial_fixpoint
  b1 u v q r f g eta left bits

def M62_SIGNED : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4611686018427387903)
  return v0

def update_de_5_k6 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (v130 : (Vector UInt256 5)) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let next_d_131 := v130
  next_d := next_d_131
  let v132 := (UInt256.ofNat 0)
  let v133 := Vector.replicate 5 v132
  let next_e_134 := v133
  next_e := next_e_134
  let v135 := (UInt256.ofNat 1)
  let i_136 := v135
  i := i_136
  let rec b1 (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut md := md
    let mut me := me
    let mut cd := cd
    let mut ce := ce
    let mut next_d := next_d
    let mut next_e := next_e
    let mut i := i
    let i_137 := i
    let i_138 := i_137
    let v139 := (UInt256.ofNat 5)
    let v140 := decide (i_138.toNat < v139.toNat)
    if v140 then
      let v141 := t.u
      let v142 := v141
      let i_143 := i
      checkIndex i_143 5
      let v144 := d[i_143.toNat]!
      let v145 := v144
      let v146 := IntTy.mulW (.i 128) v142 v145
      let v147 := t.v
      let v148 := v147
      let i_149 := i
      checkIndex i_149 5
      let v150 := e[i_149.toNat]!
      let v151 := v150
      let v152 := IntTy.mulW (.i 128) v148 v151
      let v153 := IntTy.addW (.i 128) v146 v152
      let i_154 := i
      checkIndex i_154 5
      let v155 := modulus[i_154.toNat]!
      let v156 := v155
      let md_157 := md
      let v158 := md_157
      let v159 := IntTy.mulW (.i 128) v156 v158
      let v160 := IntTy.addW (.i 128) v153 v159
      let cd_161 := cd
      let v162 := IntTy.addW (.i 128) cd_161 v160
      let cd_163 := v162
      cd := cd_163
      let v164 := ()
      let v165 := t.q
      let v166 := v165
      let i_167 := i
      checkIndex i_167 5
      let v168 := d[i_167.toNat]!
      let v169 := v168
      let v170 := IntTy.mulW (.i 128) v166 v169
      let v171 := t.r
      let v172 := v171
      let i_173 := i
      checkIndex i_173 5
      let v174 := e[i_173.toNat]!
      let v175 := v174
      let v176 := IntTy.mulW (.i 128) v172 v175
      let v177 := IntTy.addW (.i 128) v170 v176
      let i_178 := i
      checkIndex i_178 5
      let v179 := modulus[i_178.toNat]!
      let v180 := v179
      let me_181 := me
      let v182 := me_181
      let v183 := IntTy.mulW (.i 128) v180 v182
      let v184 := IntTy.addW (.i 128) v177 v183
      let ce_185 := ce
      let v186 := IntTy.addW (.i 128) ce_185 v184
      let ce_187 := v186
      ce := ce_187
      let v188 := ()
      let cd_189 := cd
      let cd_190 := cd_189
      let v191 ← downcast_truncate_i128_i64 cd_190
      let limb_d := v191
      let ce_193 := ce
      let ce_194 := ce_193
      let v195 ← downcast_truncate_i128_i64 ce_194
      let limb_e := v195
      let v197 ← M62_SIGNED
      let v198 := limb_d &&& v197
      let i_199 := i
      let i_200 := i_199
      let v201 := (UInt256.ofNat 1)
      let v202 := i_200 - v201
      checkIndex v202 5
      next_d := (next_d.set! v202.toNat v198)
      let v203 := ()
      let v204 ← M62_SIGNED
      let v205 := limb_e &&& v204
      let i_206 := i
      let i_207 := i_206
      let v208 := (UInt256.ofNat 1)
      let v209 := i_207 - v208
      checkIndex v209 5
      next_e := (next_e.set! v209.toNat v205)
      let v210 := ()
      let cd_211 := cd
      let cd_212 := cd_211
      let v213 := (UInt256.ofNat 62)
      let v214 := IntTy.shr (.i 128) cd_212 v213
      let cd_215 := v214
      cd := cd_215
      let v216 := ()
      let ce_217 := ce
      let ce_218 := ce_217
      let v219 := (UInt256.ofNat 62)
      let v220 := IntTy.shr (.i 128) ce_218 v219
      let ce_221 := v220
      ce := ce_221
      let v222 := ()
      let v223 := (UInt256.ofNat 1)
      let i_224 := i
      let v225 := i_224 + v223
      let i_226 := v225
      i := i_226
      let v227 := ()
      b1 md me cd ce next_d next_e i
    else
      let cd_228 := cd
      let cd_229 := cd_228
      let v230 ← downcast_truncate_i128_i64 cd_229
      let v231 := (UInt256.ofNat 5)
      let v232 := (UInt256.ofNat 1)
      let v233 := v231 - v232
      checkIndex v233 5
      next_d := (next_d.set! v233.toNat v230)
      let v234 := ()
      let ce_235 := ce
      let ce_236 := ce_235
      let v237 ← downcast_truncate_i128_i64 ce_236
      let v238 := (UInt256.ofNat 5)
      let v239 := (UInt256.ofNat 1)
      let v240 := v238 - v239
      checkIndex v240 5
      next_e := (next_e.set! v240.toNat v237)
      let v241 := ()
      let next_d_242 := next_d
      let next_d_243 := next_d_242
      let next_e_244 := next_e
      let v245 := (next_d_243, next_e_244)
      return v245
  partial_fixpoint
  b1 md me cd ce next_d next_e i

def update_de_5_k5 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (v109 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let me_110 := me
  let v111 := me_110
  let v112 := IntTy.mulW (.i 128) v109 v111
  let ce_113 := ce
  let v114 := IntTy.addW (.i 128) ce_113 v112
  let ce_115 := v114
  ce := ce_115
  let v116 := ()
  let cd_117 := cd
  let cd_118 := cd_117
  let v119 := (UInt256.ofNat 62)
  let v120 := IntTy.shr (.i 128) cd_118 v119
  let cd_121 := v120
  cd := cd_121
  let v122 := ()
  let ce_123 := ce
  let ce_124 := ce_123
  let v125 := (UInt256.ofNat 62)
  let v126 := IntTy.shr (.i 128) ce_124 v125
  let ce_127 := v126
  ce := ce_127
  let v128 := ()
  let v129 := (UInt256.ofNat 0)
  let v130 := Vector.replicate 5 v129
  update_de_5_k6 d e t modulus v130 md me cd ce next_d next_e i

def update_de_5_k4 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (correction_e : UInt256) (v90 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let md_91 := v90
  md := md_91
  let v92 := ()
  let me_93 := me
  let v94 := IntTy.subW (.i 64) me_93 correction_e
  let me_95 := v94
  me := me_95
  let v96 := ()
  let v97 := (UInt256.ofNat 0)
  checkIndex v97 5
  let v98 := modulus[v97.toNat]!
  let v99 := v98
  let md_100 := md
  let v101 := md_100
  let v102 := IntTy.mulW (.i 128) v99 v101
  let cd_103 := cd
  let v104 := IntTy.addW (.i 128) cd_103 v102
  let cd_105 := v104
  cd := cd_105
  let v106 := ()
  let v107 := (UInt256.ofNat 0)
  checkIndex v107 5
  let v108 := modulus[v107.toNat]!
  let v109 := v108
  update_de_5_k5 d e t modulus v109 md me cd ce next_d next_e i

def update_de_5_k3 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62_4 : UInt256) (low_d : UInt256) (low_e : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let md_67 := md
  let md_68 := md_67
  let v69 ← downcast_truncate_i64_u64 md_68
  let md_bits := v69
  let me_71 := me
  let me_72 := me_71
  let v73 ← downcast_truncate_i64_u64 me_72
  let me_bits := v73
  let inv62_75 := inv62_4
  let v76 := IntTy.mulW (.u 64) inv62_75 low_d
  let v77 := IntTy.addW (.u 64) v76 md_bits
  let v78 ← M62
  let v79 := v77 &&& v78
  let v80 ← downcast_truncate_u64_i64 v79
  let correction_d := v80
  let inv62_82 := inv62_4
  let v83 := IntTy.mulW (.u 64) inv62_82 low_e
  let v84 := IntTy.addW (.u 64) v83 me_bits
  let v85 ← M62
  let v86 := v84 &&& v85
  let v87 ← downcast_truncate_u64_i64 v86
  let correction_e := v87
  let md_89 := md
  let v90 := IntTy.subW (.i 64) md_89 correction_d
  update_de_5_k4 d e t modulus correction_e v90 md me cd ce next_d next_e i

def update_de_5_k2 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62_4 : UInt256) (v45 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let v46 := v45
  let v47 := (UInt256.ofNat 0)
  checkIndex v47 5
  let v48 := d[v47.toNat]!
  let v49 := v48
  let v50 := IntTy.mulW (.i 128) v46 v49
  let v51 := t.r
  let v52 := v51
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let v54 := e[v53.toNat]!
  let v55 := v54
  let v56 := IntTy.mulW (.i 128) v52 v55
  let v57 := IntTy.addW (.i 128) v50 v56
  let ce_58 := v57
  ce := ce_58
  let cd_59 := cd
  let cd_60 := cd_59
  let v61 ← downcast_truncate_i128_u64 cd_60
  let low_d := v61
  let ce_63 := ce
  let ce_64 := ce_63
  let v65 ← downcast_truncate_i128_u64 ce_64
  let low_e := v65
  update_de_5_k3 d e t modulus inv62_4 low_d low_e md me cd ce next_d next_e i

def update_de_5_k1 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62_4 : UInt256) (sd : UInt256) (se : UInt256) (v25 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let v26 := v25 &&& sd
  let v27 := t.r
  let v28 := v27 &&& se
  let v29 := IntTy.addW (.i 64) v26 v28
  let me_30 := v29
  me := me_30
  let v31 := t.u
  let v32 := v31
  let v33 := (UInt256.ofNat 0)
  checkIndex v33 5
  let v34 := d[v33.toNat]!
  let v35 := v34
  let v36 := IntTy.mulW (.i 128) v32 v35
  let v37 := t.v
  let v38 := v37
  let v39 := (UInt256.ofNat 0)
  checkIndex v39 5
  let v40 := e[v39.toNat]!
  let v41 := v40
  let v42 := IntTy.mulW (.i 128) v38 v41
  let v43 := IntTy.addW (.i 128) v36 v42
  let cd_44 := v43
  cd := cd_44
  let v45 := t.q
  update_de_5_k2 d e t modulus inv62_4 v45 md me cd ce next_d next_e i

def update_de_5 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62_4 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md : UInt256 := default
  let mut me : UInt256 := default
  let mut cd : UInt256 := default
  let mut ce : UInt256 := default
  let mut next_d : (Vector UInt256 5) := default
  let mut next_e : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let v5 := (UInt256.ofNat 5)
  let v6 := (UInt256.ofNat 1)
  let v7 := v5 - v6
  checkIndex v7 5
  let v8 := d[v7.toNat]!
  let v9 := (UInt256.ofNat 63)
  let v10 := IntTy.shr (.i 64) v8 v9
  let sd := v10
  let v12 := (UInt256.ofNat 5)
  let v13 := (UInt256.ofNat 1)
  let v14 := v12 - v13
  checkIndex v14 5
  let v15 := e[v14.toNat]!
  let v16 := (UInt256.ofNat 63)
  let v17 := IntTy.shr (.i 64) v15 v16
  let se := v17
  let v19 := t.u
  let v20 := v19 &&& sd
  let v21 := t.v
  let v22 := v21 &&& se
  let v23 := IntTy.addW (.i 64) v20 v22
  let md_24 := v23
  md := md_24
  let v25 := t.q
  update_de_5_k1 d e t modulus inv62_4 sd se v25 md me cd ce next_d next_e i

def update_fg_5_k2 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf := cf
  let mut cg := cg
  let mut next_f := next_f
  let mut next_g := next_g
  let mut i := i
  let v43 := (UInt256.ofNat 0)
  let v44 := Vector.replicate 5 v43
  let next_f_45 := v44
  next_f := next_f_45
  let v46 := (UInt256.ofNat 0)
  let v47 := Vector.replicate 5 v46
  let next_g_48 := v47
  next_g := next_g_48
  let v49 := (UInt256.ofNat 1)
  let i_50 := v49
  i := i_50
  let rec b1 (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut cf := cf
    let mut cg := cg
    let mut next_f := next_f
    let mut next_g := next_g
    let mut i := i
    let i_51 := i
    let i_52 := i_51
    let v53 := (UInt256.ofNat 5)
    let v54 := decide (i_52.toNat < v53.toNat)
    if v54 then
      let v55 := t.u
      let v56 := v55
      let i_57 := i
      checkIndex i_57 5
      let v58 := f[i_57.toNat]!
      let v59 := v58
      let v60 := IntTy.mulW (.i 128) v56 v59
      let v61 := t.v
      let v62 := v61
      let i_63 := i
      checkIndex i_63 5
      let v64 := g[i_63.toNat]!
      let v65 := v64
      let v66 := IntTy.mulW (.i 128) v62 v65
      let v67 := IntTy.addW (.i 128) v60 v66
      let cf_68 := cf
      let v69 := IntTy.addW (.i 128) cf_68 v67
      let cf_70 := v69
      cf := cf_70
      let v71 := ()
      let v72 := t.q
      let v73 := v72
      let i_74 := i
      checkIndex i_74 5
      let v75 := f[i_74.toNat]!
      let v76 := v75
      let v77 := IntTy.mulW (.i 128) v73 v76
      let v78 := t.r
      let v79 := v78
      let i_80 := i
      checkIndex i_80 5
      let v81 := g[i_80.toNat]!
      let v82 := v81
      let v83 := IntTy.mulW (.i 128) v79 v82
      let v84 := IntTy.addW (.i 128) v77 v83
      let cg_85 := cg
      let v86 := IntTy.addW (.i 128) cg_85 v84
      let cg_87 := v86
      cg := cg_87
      let v88 := ()
      let cf_89 := cf
      let cf_90 := cf_89
      let v91 ← downcast_truncate_i128_i64 cf_90
      let limb_f := v91
      let cg_93 := cg
      let cg_94 := cg_93
      let v95 ← downcast_truncate_i128_i64 cg_94
      let limb_g := v95
      let v97 ← M62_SIGNED
      let v98 := limb_f &&& v97
      let i_99 := i
      let i_100 := i_99
      let v101 := (UInt256.ofNat 1)
      let v102 := i_100 - v101
      checkIndex v102 5
      next_f := (next_f.set! v102.toNat v98)
      let v103 := ()
      let v104 ← M62_SIGNED
      let v105 := limb_g &&& v104
      let i_106 := i
      let i_107 := i_106
      let v108 := (UInt256.ofNat 1)
      let v109 := i_107 - v108
      checkIndex v109 5
      next_g := (next_g.set! v109.toNat v105)
      let v110 := ()
      let cf_111 := cf
      let cf_112 := cf_111
      let v113 := (UInt256.ofNat 62)
      let v114 := IntTy.shr (.i 128) cf_112 v113
      let cf_115 := v114
      cf := cf_115
      let v116 := ()
      let cg_117 := cg
      let cg_118 := cg_117
      let v119 := (UInt256.ofNat 62)
      let v120 := IntTy.shr (.i 128) cg_118 v119
      let cg_121 := v120
      cg := cg_121
      let v122 := ()
      let v123 := (UInt256.ofNat 1)
      let i_124 := i
      let v125 := i_124 + v123
      let i_126 := v125
      i := i_126
      let v127 := ()
      b1 cf cg next_f next_g i
    else
      let cf_128 := cf
      let cf_129 := cf_128
      let v130 ← downcast_truncate_i128_i64 cf_129
      let v131 := (UInt256.ofNat 5)
      let v132 := (UInt256.ofNat 1)
      let v133 := v131 - v132
      checkIndex v133 5
      next_f := (next_f.set! v133.toNat v130)
      let v134 := ()
      let cg_135 := cg
      let cg_136 := cg_135
      let v137 ← downcast_truncate_i128_i64 cg_136
      let v138 := (UInt256.ofNat 5)
      let v139 := (UInt256.ofNat 1)
      let v140 := v138 - v139
      checkIndex v140 5
      next_g := (next_g.set! v140.toNat v137)
      let v141 := ()
      let next_f_142 := next_f
      let next_f_143 := next_f_142
      let next_g_144 := next_g
      let v145 := (next_f_143, next_g_144)
      return v145
  partial_fixpoint
  b1 cf cg next_f next_g i

def update_fg_5_k1 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) (v22 : UInt256) (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf := cf
  let mut cg := cg
  let mut next_f := next_f
  let mut next_g := next_g
  let mut i := i
  let v23 := t.r
  let v24 := v23
  let v25 := (UInt256.ofNat 0)
  checkIndex v25 5
  let v26 := g[v25.toNat]!
  let v27 := v26
  let v28 := IntTy.mulW (.i 128) v24 v27
  let v29 := IntTy.addW (.i 128) v22 v28
  let cg_30 := v29
  cg := cg_30
  let cf_31 := cf
  let cf_32 := cf_31
  let v33 := (UInt256.ofNat 62)
  let v34 := IntTy.shr (.i 128) cf_32 v33
  let cf_35 := v34
  cf := cf_35
  let v36 := ()
  let cg_37 := cg
  let cg_38 := cg_37
  let v39 := (UInt256.ofNat 62)
  let v40 := IntTy.shr (.i 128) cg_38 v39
  let cg_41 := v40
  cg := cg_41
  let v42 := ()
  update_fg_5_k2 f g t cf cg next_f next_g i

def update_fg_5 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf : UInt256 := default
  let mut cg : UInt256 := default
  let mut next_f : (Vector UInt256 5) := default
  let mut next_g : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let v3 := t.u
  let v4 := v3
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 5
  let v6 := f[v5.toNat]!
  let v7 := v6
  let v8 := IntTy.mulW (.i 128) v4 v7
  let v9 := t.v
  let v10 := v9
  let v11 := (UInt256.ofNat 0)
  checkIndex v11 5
  let v12 := g[v11.toNat]!
  let v13 := v12
  let v14 := IntTy.mulW (.i 128) v10 v13
  let v15 := IntTy.addW (.i 128) v8 v14
  let cf_16 := v15
  cf := cf_16
  let v17 := t.q
  let v18 := v17
  let v19 := (UInt256.ofNat 0)
  checkIndex v19 5
  let v20 := f[v19.toNat]!
  let v21 := v20
  let v22 := IntTy.mulW (.i 128) v18 v21
  update_fg_5_k1 f g t v22 cf cg next_f next_g i

def add_if_negative_5 (r_0 : (Vector UInt256 5)) (modulus : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let v2 := r_0
  let r_3 := v2
  r := r_3
  let v4 := (UInt256.ofNat 5)
  let v5 := (UInt256.ofNat 1)
  let v6 := v4 - v5
  checkIndex v6 5
  let v7 := r[v6.toNat]!
  let v8 := (UInt256.ofNat 63)
  let v9 := IntTy.shr (.i 64) v7 v8
  let negative := v9
  let v11 := (UInt256.ofNat 0)
  let i_12 := v11
  i := i_12
  let rec b1 (r : (Vector UInt256 5)) (i : UInt256) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let i_13 := i
    let i_14 := i_13
    let v15 := (UInt256.ofNat 5)
    let v16 := decide (i_14.toNat < v15.toNat)
    if v16 then
      let i_17 := i
      checkIndex i_17 5
      let v18 := modulus[i_17.toNat]!
      let v19 := v18 &&& negative
      let i_20 := i
      checkIndex i_20 5
      let v21 := r[i_20.toNat]!
      let v22 := IntTy.addW (.i 64) v21 v19
      let i_23 := i
      checkIndex i_23 5
      r := (r.set! i_23.toNat v22)
      let v24 := ()
      let v25 := (UInt256.ofNat 1)
      let i_26 := i
      let v27 := i_26 + v25
      let i_28 := v27
      i := i_28
      let v29 := ()
      b1 r i
    else
      let r_30 := r
      return r_30
  partial_fixpoint
  b1 r i

def carry_5 (r_0 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let v1 := r_0
  let r_2 := v1
  r := r_2
  let v3 := (UInt256.ofNat 0)
  let i_4 := v3
  i := i_4
  let rec b1 (r : (Vector UInt256 5)) (i : UInt256) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let i_5 := i
    let i_6 := i_5
    let v7 := (UInt256.ofNat 1)
    let v8 := i_6 + v7
    let v9 := (UInt256.ofNat 5)
    let v10 := decide (v8.toNat < v9.toNat)
    if v10 then
      let i_11 := i
      checkIndex i_11 5
      let v12 := r[i_11.toNat]!
      let v13 := (UInt256.ofNat 62)
      let v14 := IntTy.shr (.i 64) v12 v13
      let i_15 := i
      let i_16 := i_15
      let v17 := (UInt256.ofNat 1)
      let v18 := i_16 + v17
      checkIndex v18 5
      let v19 := r[v18.toNat]!
      let v20 := IntTy.addW (.i 64) v19 v14
      checkIndex v18 5
      r := (r.set! v18.toNat v20)
      let v21 := ()
      let v22 ← M62_SIGNED
      let i_23 := i
      checkIndex i_23 5
      let v24 := r[i_23.toNat]!
      let v25 := v24 &&& v22
      let i_26 := i
      checkIndex i_26 5
      r := (r.set! i_26.toNat v25)
      let v27 := ()
      let v28 := (UInt256.ofNat 1)
      let i_29 := i
      let v30 := i_29 + v28
      let i_31 := v30
      i := i_31
      let v32 := ()
      b1 r i
    else
      let r_33 := r
      return r_33
  partial_fixpoint
  b1 r i

def normalize_5 (r_0 : (Vector UInt256 5)) (sign_1 : UInt256) (modulus : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let mut r2 : (Vector UInt256 5) := default
  let mut r3 : (Vector UInt256 5) := default
  let mut r4 : (Vector UInt256 5) := default
  let v3 ← add_if_negative_5 r_0 modulus
  let r_4 := v3
  r := r_4
  let sign_5 := sign_1
  let v6 := (UInt256.ofNat 63)
  let v7 := IntTy.shr (.i 64) sign_5 v6
  let negate := v7
  let v9 := (UInt256.ofNat 0)
  let i_10 := v9
  i := i_10
  let rec b1 (r : (Vector UInt256 5)) (i : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let i_11 := i
    let i_12 := i_11
    let v13 := (UInt256.ofNat 5)
    let v14 := decide (i_12.toNat < v13.toNat)
    if v14 then
      let i_15 := i
      checkIndex i_15 5
      let v16 := r[i_15.toNat]!
      let v17 := v16 ^^^ negate
      let v18 := IntTy.subW (.i 64) v17 negate
      let i_19 := i
      checkIndex i_19 5
      r := (r.set! i_19.toNat v18)
      let v20 := ()
      let v21 := (UInt256.ofNat 1)
      let i_22 := i
      let v23 := i_22 + v21
      let i_24 := v23
      i := i_24
      let v25 := ()
      b1 r i r2 r3 r4
    else
      let r_26 := r
      let r_27 := r_26
      r2 := r_27
      let v29 ← carry_5 r2
      r3 := v29
      let v31 ← add_if_negative_5 r3 modulus
      r4 := v31
      let v33 ← carry_5 r4
      return v33
  partial_fixpoint
  b1 r i r2 r3 r4

def invert_5 (x : (Vector UInt256 5)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) : FeM (Vector UInt256 5) := do
  let mut d : (Vector UInt256 5) := default
  let mut e : (Vector UInt256 5) := default
  let mut f : (Vector UInt256 5) := default
  let mut g : (Vector UInt256 5) := default
  let mut eta : UInt256 := default
  let mut t : Transition := default
  let mut r6 : (Vector UInt256 5) := default
  let mut r7 : (Vector UInt256 5) := default
  let mut r8 : (Vector UInt256 5) := default
  let mut r9 : (Vector UInt256 5) := default
  let mut r10 : (Vector UInt256 5) := default
  let mut r11 : (Vector UInt256 5) := default
  let v3 := (UInt256.ofNat 0)
  let v4 := Vector.replicate 5 v3
  let d_5 := v4
  d := d_5
  let v6 := (UInt256.ofNat 0)
  let v7 := Vector.replicate 5 v6
  let e_8 := v7
  e := e_8
  let v9 := (UInt256.ofNat 1)
  let v10 := (UInt256.ofNat 0)
  checkIndex v10 5
  e := (e.set! v10.toNat v9)
  let v11 := ()
  let v12 := modulus
  let f_13 := v12
  f := f_13
  let v14 := x
  let g_15 := v14
  g := g_15
  let v16 := (IntTy.wrap (.i 256) (-1))
  let eta_17 := v16
  eta := eta_17
  let rec b1 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (eta : UInt256) (t : Transition) (r6 : (Vector UInt256 5)) (r7 : (Vector UInt256 5)) (r8 : (Vector UInt256 5)) (r9 : (Vector UInt256 5)) (r10 : (Vector UInt256 5)) (r11 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
    let mut d := d
    let mut e := e
    let mut f := f
    let mut g := g
    let mut eta := eta
    let mut t := t
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut r11 := r11
    let g_18 := g
    let g_19 := g_18
    r6 := g_19
    let v21 ← is_zero_5 r6
    let v22 := !v21
    if v22 then
      let eta_23 := eta
      let eta_24 := eta_23
      let v25 := (UInt256.ofNat 0)
      checkIndex v25 5
      let v26 := f[v25.toNat]!
      let v27 ← downcast_truncate_i64_u64 v26
      let v28 := (UInt256.ofNat 0)
      checkIndex v28 5
      let v29 := g[v28.toNat]!
      let v30 ← downcast_truncate_i64_u64 v29
      let v31 ← divsteps eta_24 v27 v30
      let v32 := v31.fst
      let next_33 := v32
      let v34 := v31.snd
      let t_35 := v34
      t := t_35
      let next_36 := next_33
      eta := next_36
      let v37 := ()
      let d_38 := d
      let d_39 := d_38
      r7 := d_39
      let e_40 := e
      let e_41 := e_40
      r8 := e_41
      let v45 ← update_de_5 r7 r8 t modulus inv62
      let v46 := v45.fst
      let next_d_47 := v46
      let v48 := v45.snd
      let next_e_49 := v48
      let next_d_50 := next_d_47
      d := next_d_50
      let v51 := ()
      let next_e_52 := next_e_49
      e := next_e_52
      let v53 := ()
      let f_54 := f
      let f_55 := f_54
      r9 := f_55
      let g_56 := g
      let g_57 := g_56
      r10 := g_57
      let v61 ← update_fg_5 r9 r10 t
      let v62 := v61.fst
      let next_f_63 := v62
      let v64 := v61.snd
      let next_g_65 := v64
      let next_f_66 := next_f_63
      f := next_f_66
      let v67 := ()
      let next_g_68 := next_g_65
      g := next_g_68
      let v69 := ()
      b1 d e f g eta t r6 r7 r8 r9 r10 r11
    else
      let d_70 := d
      let d_71 := d_70
      r11 := d_71
      let v72 := (UInt256.ofNat 5)
      let v73 := (UInt256.ofNat 1)
      let v74 := v72 - v73
      checkIndex v74 5
      let v75 := f[v74.toNat]!
      let v77 ← normalize_5 r11 v75 modulus
      return v77
  partial_fixpoint
  b1 d e f g eta t r6 r7 r8 r9 r10 r11

def invert_256_k5 (v76 : UInt256) (v85 : UInt256) (v94 : UInt256) (v98 : UInt256) (v100 : UInt256) (v101 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v102 := IntTy.shl (.u 64) v100 v101
  let v103 := v98 ||| v102
  let v104 := #v[v76, v85, v94, v103]
  return v104

def invert_256_k4 (w : (Vector UInt256 5)) (v76 : UInt256) (v80 : UInt256) (v81 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v82 := w[v81.toNat]!
  let v83 := (UInt256.ofNat 60)
  let v84 := IntTy.shl (.u 64) v82 v83
  let v85 := v80 ||| v84
  let v86 := (UInt256.ofNat 2)
  checkIndex v86 5
  let v87 := w[v86.toNat]!
  let v88 := (UInt256.ofNat 4)
  let v89 := v87 >>> v88
  let v90 := (UInt256.ofNat 3)
  checkIndex v90 5
  let v91 := w[v90.toNat]!
  let v92 := (UInt256.ofNat 58)
  let v93 := IntTy.shl (.u 64) v91 v92
  let v94 := v89 ||| v93
  let v95 := (UInt256.ofNat 3)
  checkIndex v95 5
  let v96 := w[v95.toNat]!
  let v97 := (UInt256.ofNat 6)
  let v98 := v96 >>> v97
  let v99 := (UInt256.ofNat 4)
  checkIndex v99 5
  let v100 := w[v99.toNat]!
  let v101 := (UInt256.ofNat 56)
  invert_256_k5 v76 v85 v94 v98 v100 v101 r0

def invert_256_k3 (s : (Vector UInt256 5)) (v55 : UInt256) (v58 : UInt256) (v61 : UInt256) (v62 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v63 := s[v62.toNat]!
  let v64 ← downcast_truncate_i64_u64 v63
  let v65 := (UInt256.ofNat 4)
  checkIndex v65 5
  let v66 := s[v65.toNat]!
  let v67 ← downcast_truncate_i64_u64 v66
  let v68 := #v[v55, v58, v61, v64, v67]
  let w := v68
  let v70 := (UInt256.ofNat 0)
  checkIndex v70 5
  let v71 := w[v70.toNat]!
  let v72 := (UInt256.ofNat 1)
  checkIndex v72 5
  let v73 := w[v72.toNat]!
  let v74 := (UInt256.ofNat 62)
  let v75 := IntTy.shl (.u 64) v73 v74
  let v76 := v71 ||| v75
  let v77 := (UInt256.ofNat 1)
  checkIndex v77 5
  let v78 := w[v77.toNat]!
  let v79 := (UInt256.ofNat 2)
  let v80 := v78 >>> v79
  let v81 := (UInt256.ofNat 2)
  checkIndex v81 5
  invert_256_k4 w v76 v80 v81 r0

def invert_256_k2 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v7 : UInt256) (v19 : UInt256) (v31 : UInt256) (v43 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v44 := (UInt256.ofNat 3)
  checkIndex v44 4
  let v45 := x[v44.toNat]!
  let v46 := (UInt256.ofNat 56)
  let v47 := v45 >>> v46
  let v48 ← downcast_truncate_u64_i64 v47
  let v49 := #v[v7, v19, v31, v43, v48]
  r0 := v49
  let v51 ← invert_5 r0 modulus inv62
  let s := v51
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let v54 := s[v53.toNat]!
  let v55 ← downcast_truncate_i64_u64 v54
  let v56 := (UInt256.ofNat 1)
  checkIndex v56 5
  let v57 := s[v56.toNat]!
  let v58 ← downcast_truncate_i64_u64 v57
  let v59 := (UInt256.ofNat 2)
  checkIndex v59 5
  let v60 := s[v59.toNat]!
  let v61 ← downcast_truncate_i64_u64 v60
  let v62 := (UInt256.ofNat 3)
  checkIndex v62 5
  invert_256_k3 s v55 v58 v61 v62 r0

def invert_256_k1 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v7 : UInt256) (v19 : UInt256) (v21 : UInt256) (v22 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v23 := v21 >>> v22
  let v24 := (UInt256.ofNat 2)
  checkIndex v24 4
  let v25 := x[v24.toNat]!
  let v26 := (UInt256.ofNat 4)
  let v27 := IntTy.shl (.u 64) v25 v26
  let v28 := v23 ||| v27
  let v29 ← M62
  let v30 := v28 &&& v29
  let v31 ← downcast_truncate_u64_i64 v30
  let v32 := (UInt256.ofNat 2)
  checkIndex v32 4
  let v33 := x[v32.toNat]!
  let v34 := (UInt256.ofNat 58)
  let v35 := v33 >>> v34
  let v36 := (UInt256.ofNat 3)
  checkIndex v36 4
  let v37 := x[v36.toNat]!
  let v38 := (UInt256.ofNat 6)
  let v39 := IntTy.shl (.u 64) v37 v38
  let v40 := v35 ||| v39
  let v41 ← M62
  let v42 := v40 &&& v41
  let v43 ← downcast_truncate_u64_i64 v42
  invert_256_k2 x modulus inv62 v7 v19 v31 v43 r0

def invert_256 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) : FeM (Vector UInt256 4) := do
  let mut r0 : (Vector UInt256 5) := default
  let v3 := (UInt256.ofNat 0)
  checkIndex v3 4
  let v4 := x[v3.toNat]!
  let v5 ← M62
  let v6 := v4 &&& v5
  let v7 ← downcast_truncate_u64_i64 v6
  let v8 := (UInt256.ofNat 0)
  checkIndex v8 4
  let v9 := x[v8.toNat]!
  let v10 := (UInt256.ofNat 62)
  let v11 := v9 >>> v10
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 4
  let v13 := x[v12.toNat]!
  let v14 := (UInt256.ofNat 2)
  let v15 := IntTy.shl (.u 64) v13 v14
  let v16 := v11 ||| v15
  let v17 ← M62
  let v18 := v16 &&& v17
  let v19 ← downcast_truncate_u64_i64 v18
  let v20 := (UInt256.ofNat 1)
  checkIndex v20 4
  let v21 := x[v20.toNat]!
  let v22 := (UInt256.ofNat 60)
  invert_256_k1 x modulus inv62 v7 v19 v21 v22 r0

def Order_R3 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 38611792656279623856209604949727965891812033543773964974832218758873530450413)
  return v0

def Mont_inverse_Order (self : Mont_Order) : FeM Mont_Order := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Order := default
  let mut r3 : Mont_Order := default
  let v1 := self.l0
  let v2 := self.l1
  let v3 := self.l2
  let v4 := self.l3
  let v5 := #v[v1, v2, v3, v4]
  r0 := v5
  let v6 ← Order_SIGNED_62
  r1 := v6
  let v7 ← Order_INV62
  let v10 ← invert_256 r0 r1 v7
  let limbs := v10
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v13 := limbs[v12.toNat]!
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v15 := limbs[v14.toNat]!
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v17 := limbs[v16.toNat]!
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v19 := limbs[v18.toNat]!
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

def Mont_sub_Order_k2 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (b_31 : UInt256) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (c_50 : UInt256) : FeM Mont_Order := do
  let v51 ← Order_P3
  let v52 ← adc d3 v51 c_50
  let v53 := v52.fst
  let s3 := v53
  let v55 := (UInt256.ofNat 1)
  let v56 := decide (b_31 = v55)
  let b3 : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Order.mk s0 s1 s2 s3)
    let v58 := v57
    b3 v58
  else
    let v59 := (Mont_Order.mk d0 d1 d2 d3)
    let v60 := v59
    b3 v60

def Mont_sub_Order_k1 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (b_24 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Mont_Order := do
  let v27 ← sbb v25 v26 b_24
  let v28 := v27.fst
  let d3 := v28
  let v30 := v27.snd
  let b_31 := v30
  let v32 ← Order_P0
  let v33 := (UInt256.ofNat 0)
  let v34 ← adc d0 v32 v33
  let v35 := v34.fst
  let s0 := v35
  let v37 := v34.snd
  let c_38 := v37
  let v39 ← Order_P1
  let v40 ← adc d1 v39 c_38
  let v41 := v40.fst
  let s1 := v41
  let v43 := v40.snd
  let c_44 := v43
  let v45 ← Order_P2
  let v46 ← adc d2 v45 c_44
  let v47 := v46.fst
  let s2 := v47
  let v49 := v46.snd
  let c_50 := v49
  Mont_sub_Order_k2 d0 d1 d2 d3 b_31 s0 s1 s2 c_50

def Mont_sub_Order (self : Mont_Order) (other : Mont_Order) : FeM Mont_Order := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let d0 := v7
  let v9 := v6.snd
  let b_10 := v9
  let v11 := self.l1
  let v12 := other.l1
  let v13 ← sbb v11 v12 b_10
  let v14 := v13.fst
  let d1 := v14
  let v16 := v13.snd
  let b_17 := v16
  let v18 := self.l2
  let v19 := other.l2
  let v20 ← sbb v18 v19 b_17
  let v21 := v20.fst
  let d2 := v21
  let v23 := v20.snd
  let b_24 := v23
  let v25 := self.l3
  let v26 := other.l3
  Mont_sub_Order_k1 d0 d1 d2 b_24 v25 v26

def Mont_neg_Order (self : Mont_Order) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let v1 ← Mont_zero_Order
  r0 := v1
  let v3 ← Mont_sub_Order r0 self
  return v3

def Mont_value_Order (self : Mont_Order) : FeM UInt256 := do
  let mut r0 : Mont_Order := default
  let v1 := (UInt256.ofNat 1)
  let v2 ← raw_Order v1
  r0 := v2
  let v4 ← Mont_mul_Order self r0
  let value := v4
  let v6 := value.l0
  let v7 := v6
  let v8 := value.l1
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := v9 <<< v10
  let v12 := v7 ||| v11
  let v13 := value.l2
  let v14 := v13
  let v15 := (UInt256.ofNat 128)
  let v16 := v14 <<< v15
  let v17 := v12 ||| v16
  let v18 := value.l3
  let v19 := v18
  let v20 := (UInt256.ofNat 192)
  let v21 := v19 <<< v20
  let v22 := v17 ||| v21
  return v22

def Point_affine_Mont_Base (x : Mont_Base) (y : Mont_Base) : FeM Point_Mont_Base := do
  let v2 := x
  let v3 := y
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

def Mont_is_zero_Base (self : Mont_Base) : FeM Bool := do
  let v2 := self.l0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (v2 = v3)
  let b3 : Bool → FeM Bool := fun v1 => do
    return v1
  let b2 : FeM Bool := do
    let v16 := false
    let v17 := v16
    b3 v17
  if v4 then
    let v5 := self.l1
    let v6 := (UInt256.ofNat 0)
    let v7 := decide (v5 = v6)
    if v7 then
      let v8 := self.l2
      let v9 := (UInt256.ofNat 0)
      let v10 := decide (v8 = v9)
      if v10 then
        let v11 := self.l3
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

def Point_is_infinity_Mont_Base (self : Point_Mont_Base) : FeM Bool := do
  let mut r0 : Mont_Base := default
  let v1 := self.z
  r0 := v1
  let v3 ← Mont_is_zero_Base r0
  return v3

def Mont_sub_Base_k2 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (b_31 : UInt256) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (c_50 : UInt256) : FeM Mont_Base := do
  let v51 ← Base_P3
  let v52 ← adc d3 v51 c_50
  let v53 := v52.fst
  let s3 := v53
  let v55 := (UInt256.ofNat 1)
  let v56 := decide (b_31 = v55)
  let b3 : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Base.mk s0 s1 s2 s3)
    let v58 := v57
    b3 v58
  else
    let v59 := (Mont_Base.mk d0 d1 d2 d3)
    let v60 := v59
    b3 v60

def Mont_sub_Base_k1 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (b_24 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Mont_Base := do
  let v27 ← sbb v25 v26 b_24
  let v28 := v27.fst
  let d3 := v28
  let v30 := v27.snd
  let b_31 := v30
  let v32 ← Base_P0
  let v33 := (UInt256.ofNat 0)
  let v34 ← adc d0 v32 v33
  let v35 := v34.fst
  let s0 := v35
  let v37 := v34.snd
  let c_38 := v37
  let v39 ← Base_P1
  let v40 ← adc d1 v39 c_38
  let v41 := v40.fst
  let s1 := v41
  let v43 := v40.snd
  let c_44 := v43
  let v45 ← Base_P2
  let v46 ← adc d2 v45 c_44
  let v47 := v46.fst
  let s2 := v47
  let v49 := v46.snd
  let c_50 := v49
  Mont_sub_Base_k2 d0 d1 d2 d3 b_31 s0 s1 s2 c_50

def Mont_sub_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let d0 := v7
  let v9 := v6.snd
  let b_10 := v9
  let v11 := self.l1
  let v12 := other.l1
  let v13 ← sbb v11 v12 b_10
  let v14 := v13.fst
  let d1 := v14
  let v16 := v13.snd
  let b_17 := v16
  let v18 := self.l2
  let v19 := other.l2
  let v20 ← sbb v18 v19 b_17
  let v21 := v20.fst
  let d2 := v21
  let v23 := v20.snd
  let b_24 := v23
  let v25 := self.l3
  let v26 := other.l3
  Mont_sub_Base_k1 d0 d1 d2 b_24 v25 v26

def Mont_double_Base (self : Mont_Base) : FeM Mont_Base := do
  let v1 ← Mont_add_Base self self
  return v1

def Point_double_Mont_Base (self : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut a : Mont_Base := default
  let mut b : Mont_Base := default
  let mut c : Mont_Base := default
  let mut d : Mont_Base := default
  let mut e : Mont_Base := default
  let mut x : Mont_Base := default
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
  let v1 ← Point_is_infinity_Mont_Base self
  let b1 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → FeM Point_Mont_Base := fun a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 => do
    let mut a := a
    let mut b := b
    let mut c := c
    let mut d := d
    let mut e := e
    let mut x := x
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
    b1 a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    let v2 := self.y
    r6 := v2
    let v4 ← Mont_is_zero_Base r6
    if v4 then
      b1 a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      let v5 := ()
      let v6 := v5
      let v7 := self.x
      r7 := v7
      let v9 ← Mont_square_Base r7
      let a_10 := v9
      a := a_10
      let v11 := self.y
      r8 := v11
      let v13 ← Mont_square_Base r8
      let b_14 := v13
      b := b_14
      let v16 ← Mont_square_Base b
      let c_17 := v16
      c := c_17
      let v18 := self.x
      r9 := v18
      let v21 ← Mont_add_Base r9 b
      r10 := v21
      let v23 ← Mont_square_Base r10
      r11 := v23
      let v26 ← Mont_sub_Base r11 a
      r12 := v26
      let v29 ← Mont_sub_Base r12 c
      r13 := v29
      let v31 ← Mont_double_Base r13
      let d_32 := v31
      d := d_32
      let v34 ← Mont_double_Base a
      r14 := v34
      let v37 ← Mont_add_Base r14 a
      let e_38 := v37
      e := e_38
      let v40 ← Mont_square_Base e
      r15 := v40
      let v42 ← Mont_double_Base d
      r16 := v42
      let v45 ← Mont_sub_Base r15 r16
      let x_46 := v45
      x := x_46
      let v49 ← Mont_sub_Base d x
      r17 := v49
      let v52 ← Mont_mul_Base e r17
      r18 := v52
      let v54 ← Mont_double_Base c
      r19 := v54
      let v56 ← Mont_double_Base r19
      r20 := v56
      let v58 ← Mont_double_Base r20
      r21 := v58
      let v61 ← Mont_sub_Base r18 r21
      let y := v61
      let v63 := self.z
      r22 := v63
      let v64 := self.y
      r23 := v64
      let v67 ← Mont_mul_Base r23 r22
      r24 := v67
      let v69 ← Mont_double_Base r24
      let x_70 := x
      let v71 := (Point_Mont_Base.mk x_70 y v69)
      return v71

def Point_add_Mont_Base (self : Point_Mont_Base) (other : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut z1z1 : Mont_Base := default
  let mut z2z2 : Mont_Base := default
  let mut u1 : Mont_Base := default
  let mut u2 : Mont_Base := default
  let mut s1 : Mont_Base := default
  let mut s2 : Mont_Base := default
  let mut h : Mont_Base := default
  let mut r : Mont_Base := default
  let mut i : Mont_Base := default
  let mut j : Mont_Base := default
  let mut v : Mont_Base := default
  let mut x : Mont_Base := default
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
  let v3 ← Point_is_infinity_Mont_Base self
  if v3 then
    let v4 := other
    return v4
  else
    let v5 := ()
    let v6 := v5
    let v7 ← Point_is_infinity_Mont_Base other
    if v7 then
      let v8 := self
      return v8
    else
      let v9 := ()
      let v10 := v9
      let v11 := self.z
      r12 := v11
      let v13 ← Mont_square_Base r12
      let z1z1_14 := v13
      z1z1 := z1z1_14
      let v15 := other.z
      r13 := v15
      let v17 ← Mont_square_Base r13
      let z2z2_18 := v17
      z2z2 := z2z2_18
      let v19 := self.x
      r14 := v19
      let v22 ← Mont_mul_Base r14 z2z2
      let u1_23 := v22
      u1 := u1_23
      let v24 := other.x
      r15 := v24
      let v27 ← Mont_mul_Base r15 z1z1
      let u2_28 := v27
      u2 := u2_28
      let v29 := other.z
      r16 := v29
      let v30 := self.y
      r17 := v30
      let v33 ← Mont_mul_Base r17 r16
      r18 := v33
      let v36 ← Mont_mul_Base r18 z2z2
      let s1_37 := v36
      s1 := s1_37
      let v38 := self.z
      r19 := v38
      let v39 := other.y
      r20 := v39
      let v42 ← Mont_mul_Base r20 r19
      r21 := v42
      let v45 ← Mont_mul_Base r21 z1z1
      let s2_46 := v45
      s2 := s2_46
      let v49 ← Mont_sub_Base u2 u1
      let h_50 := v49
      h := h_50
      let v53 ← Mont_sub_Base s2 s1
      r22 := v53
      let v55 ← Mont_double_Base r22
      let r_56 := v55
      r := r_56
      let v58 ← Mont_is_zero_Base h
      if v58 then
        let v60 ← Mont_is_zero_Base r
        let b12 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v2 => do
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
          let v61 ← Point_double_Mont_Base self
          let v62 := v61
          b12 z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v62
        else
          let v63 ← Point_infinity_Mont_Base
          let v64 := v63
          b12 z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v64
      else
        let v65 := ()
        let v66 := v65
        let v68 ← Mont_double_Base h
        r23 := v68
        let v70 ← Mont_square_Base r23
        let i_71 := v70
        i := i_71
        let v74 ← Mont_mul_Base h i
        let j_75 := v74
        j := j_75
        let v78 ← Mont_mul_Base u1 i
        let v_79 := v78
        v := v_79
        let v81 ← Mont_square_Base r
        r24 := v81
        let v84 ← Mont_sub_Base r24 j
        r25 := v84
        let v86 ← Mont_double_Base v
        r26 := v86
        let v89 ← Mont_sub_Base r25 r26
        let x_90 := v89
        x := x_90
        let v93 ← Mont_sub_Base v x
        r27 := v93
        let v96 ← Mont_mul_Base r r27
        r28 := v96
        let v99 ← Mont_mul_Base s1 j
        r29 := v99
        let v101 ← Mont_double_Base r29
        r30 := v101
        let v104 ← Mont_sub_Base r28 r30
        let y := v104
        let v106 := other.z
        r31 := v106
        let v107 := self.z
        r32 := v107
        let v110 ← Mont_add_Base r32 r31
        r33 := v110
        let v112 ← Mont_square_Base r33
        r34 := v112
        let v115 ← Mont_sub_Base r34 z1z1
        r35 := v115
        let v118 ← Mont_sub_Base r35 z2z2
        r36 := v118
        let v121 ← Mont_mul_Base r36 h
        let x_122 := x
        let v123 := (Point_Mont_Base.mk x_122 y v121)
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

def Mont_inverse_Base (self : Mont_Base) : FeM Mont_Base := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let v1 := self.l0
  let v2 := self.l1
  let v3 := self.l2
  let v4 := self.l3
  let v5 := #v[v1, v2, v3, v4]
  r0 := v5
  let v6 ← Base_SIGNED_62
  r1 := v6
  let v7 ← Base_INV62
  let v10 ← invert_256 r0 r1 v7
  let limbs := v10
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v13 := limbs[v12.toNat]!
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v15 := limbs[v14.toNat]!
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v17 := limbs[v16.toNat]!
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v19 := limbs[v18.toNat]!
  let v20 := (Mont_Base.mk v13 v15 v17 v19)
  r2 := v20
  let v21 ← Base_R3
  let v22 ← raw_Base v21
  r3 := v22
  let v25 ← Mont_mul_Base r2 r3
  return v25

def Point_add_affine_Mont_Base (self : Point_Mont_Base) (x2 : Mont_Base) (y2 : Mont_Base) : FeM Point_Mont_Base := do
  let mut z1z1 : Mont_Base := default
  let mut u2 : Mont_Base := default
  let mut s2 : Mont_Base := default
  let mut h : Mont_Base := default
  let mut r : Mont_Base := default
  let mut hh : Mont_Base := default
  let mut i : Mont_Base := default
  let mut j : Mont_Base := default
  let mut v : Mont_Base := default
  let mut x : Mont_Base := default
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
  let v4 ← Point_is_infinity_Mont_Base self
  if v4 then
    let v5 ← Point_affine_Mont_Base x2 y2
    return v5
  else
    let v6 := ()
    let v7 := v6
    let v8 := self.z
    r10 := v8
    let v10 ← Mont_square_Base r10
    let z1z1_11 := v10
    z1z1 := z1z1_11
    let v13 ← Mont_mul_Base x2 z1z1
    let u2_14 := v13
    u2 := u2_14
    let v15 := self.z
    r11 := v15
    let v17 ← Mont_mul_Base y2 r11
    r12 := v17
    let v20 ← Mont_mul_Base r12 z1z1
    let s2_21 := v20
    s2 := s2_21
    let v22 := self.x
    r13 := v22
    let v25 ← Mont_sub_Base u2 r13
    let h_26 := v25
    h := h_26
    let v27 := self.y
    r14 := v27
    let v30 ← Mont_sub_Base s2 r14
    r15 := v30
    let v32 ← Mont_double_Base r15
    let r_33 := v32
    r := r_33
    let v35 ← Mont_is_zero_Base h
    if v35 then
      let v37 ← Mont_is_zero_Base r
      let b9 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v3 => do
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
        let v38 ← Point_double_Mont_Base self
        let v39 := v38
        b9 z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v39
      else
        let v40 ← Point_infinity_Mont_Base
        let v41 := v40
        b9 z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v41
    else
      let v42 := ()
      let v43 := v42
      let v45 ← Mont_square_Base h
      let hh_46 := v45
      hh := hh_46
      let v48 ← Mont_double_Base hh
      r16 := v48
      let v50 ← Mont_double_Base r16
      let i_51 := v50
      i := i_51
      let v54 ← Mont_mul_Base h i
      let j_55 := v54
      j := j_55
      let v56 := self.x
      r17 := v56
      let v59 ← Mont_mul_Base r17 i
      let v_60 := v59
      v := v_60
      let v62 ← Mont_square_Base r
      r18 := v62
      let v65 ← Mont_sub_Base r18 j
      r19 := v65
      let v67 ← Mont_double_Base v
      r20 := v67
      let v70 ← Mont_sub_Base r19 r20
      let x_71 := v70
      x := x_71
      let v74 ← Mont_sub_Base v x
      r21 := v74
      let v77 ← Mont_mul_Base r r21
      r22 := v77
      let v78 := self.y
      r23 := v78
      let v81 ← Mont_mul_Base r23 j
      r24 := v81
      let v83 ← Mont_double_Base r24
      r25 := v83
      let v86 ← Mont_sub_Base r22 r25
      let y := v86
      let v88 := self.z
      r26 := v88
      let v91 ← Mont_add_Base r26 h
      r27 := v91
      let v93 ← Mont_square_Base r27
      r28 := v93
      let v96 ← Mont_sub_Base r28 z1z1
      r29 := v96
      let v99 ← Mont_sub_Base r29 hh
      let x_100 := x
      let v101 := (Point_Mont_Base.mk x_100 y v99)
      return v101

def combination_k2 (u_0 : UInt256) (v_1 : UInt256) (v38 : UInt256) (v39 : UInt256) (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut g := g
  let mut table := table
  let mut «prefix» := «prefix»
  let mut running := running
  let mut inverse := inverse
  let mut k := k
  let mut z_inverse := z_inverse
  let mut z_inverse2 := z_inverse2
  let mut result := result
  let mut bit := bit
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
  let rec b1 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v3 : UInt256) : FeM Point_Mont_Base := do
    let mut g := g
    let mut table := table
    let mut «prefix» := «prefix»
    let mut running := running
    let mut inverse := inverse
    let mut k := k
    let mut z_inverse := z_inverse
    let mut z_inverse2 := z_inverse2
    let mut result := result
    let mut bit := bit
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
      let i := v48
      let v50 := (UInt256.ofNat 1)
      let v51 := (UInt256.ofNat 4)
      let v52 := ()
      let v53 := (Range_Known_1_Known_4.mk v52 v52)
      r13 := v53
      let v55 ← Range_start_1_4 r13
      let v56 := v55
      let rec b5 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v4 : UInt256) : FeM Point_Mont_Base := do
        let mut g := g
        let mut table := table
        let mut «prefix» := «prefix»
        let mut running := running
        let mut inverse := inverse
        let mut k := k
        let mut z_inverse := z_inverse
        let mut z_inverse2 := z_inverse2
        let mut result := result
        let mut bit := bit
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
          let j := v60
          checkIndex j 16
          let v62 := table[j.toNat]!
          r14 := v62
          checkIndex i 16
          let v63 := table[i.toNat]!
          r15 := v63
          let v66 ← Point_add_Mont_Base r15 r14
          let v67 ← checkedAdd i j
          checkIndex v67 16
          table := (table.set! v67.toNat v66)
          let v68 := ()
          let v70 ← Range_next_1_4 r13 v4
          let v71 := v70
          b5 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v71
        else
          let v72 := ()
          let v74 ← array_next_usize_3 r12 v3
          let v75 := v74
          b1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v75
      partial_fixpoint
      b5 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v56
    else
      let v76 ← Mont_one_Base
      let v77 := Vector.replicate 16 v76
      let prefix_78 := v77
      «prefix» := prefix_78
      let v79 ← Mont_one_Base
      let running_80 := v79
      running := running_80
      let v81 := (UInt256.ofNat 1)
      let v82 := (UInt256.ofNat 16)
      let v83 := ()
      let v84 := (Range_Known_1_Known_16.mk v83 v83)
      r16 := v84
      let v86 ← Range_start_1_16 r16
      let v87 := v86
      let rec b9 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v5 : UInt256) : FeM Point_Mont_Base := do
        let mut g := g
        let mut table := table
        let mut «prefix» := «prefix»
        let mut running := running
        let mut inverse := inverse
        let mut k := k
        let mut z_inverse := z_inverse
        let mut z_inverse2 := z_inverse2
        let mut result := result
        let mut bit := bit
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
          let k_92 := v91
          let running_93 := running
          let running_94 := running_93
          checkIndex k_92 16
          «prefix» := («prefix».set! k_92.toNat running_94)
          let v95 := ()
          checkIndex k_92 16
          let v96 := table[k_92.toNat]!
          r17 := v96
          let v98 ← Point_is_infinity_Mont_Base r17
          let v99 := !v98
          let b15 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
            let mut g := g
            let mut table := table
            let mut «prefix» := «prefix»
            let mut running := running
            let mut inverse := inverse
            let mut k := k
            let mut z_inverse := z_inverse
            let mut z_inverse2 := z_inverse2
            let mut result := result
            let mut bit := bit
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
            b9 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v113
          if v99 then
            checkIndex k_92 16
            let v100 := table[k_92.toNat]!.z
            r18 := v100
            let running_101 := running
            let running_102 := running_101
            r19 := running_102
            let v105 ← Mont_mul_Base r19 r18
            let running_106 := v105
            running := running_106
            let v107 := ()
            let v108 := v107
            b15 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          else
            let v109 := ()
            let v110 := v109
            b15 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
        else
          let running_114 := running
          let running_115 := running_114
          r20 := running_115
          let v117 ← Mont_inverse_Base r20
          let inverse_118 := v117
          inverse := inverse_118
          let v119 := (UInt256.ofNat 16)
          let k_120 := v119
          k := k_120
          let rec b16 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
            let mut g := g
            let mut table := table
            let mut «prefix» := «prefix»
            let mut running := running
            let mut inverse := inverse
            let mut k := k
            let mut z_inverse := z_inverse
            let mut z_inverse2 := z_inverse2
            let mut result := result
            let mut bit := bit
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
            let k_121 := k
            let k_122 := k_121
            let v123 := (UInt256.ofNat 1)
            let v124 := decide (v123.toNat < k_122.toNat)
            if v124 then
              let v125 := (UInt256.ofNat 1)
              let k_126 := k
              let v127 ← checkedSub k_126 v125
              let k_128 := v127
              k := k_128
              let v129 := ()
              let k_130 := k
              checkIndex k_130 16
              let v131 := table[k_130.toNat]!
              r21 := v131
              let k_132 := k
              let v134 ← Point_is_infinity_Mont_Base r21
              if v134 then
                b16 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
              else
                let v135 := ()
                let v136 := v135
                let k_137 := k
                checkIndex k_137 16
                let v138 := «prefix»[k_137.toNat]!
                r22 := v138
                let inverse_139 := inverse
                let inverse_140 := inverse_139
                r23 := inverse_140
                let k_142 := k
                let v144 ← Mont_mul_Base r23 r22
                let z_inverse_145 := v144
                z_inverse := z_inverse_145
                let k_146 := k
                checkIndex k_146 16
                let v147 := table[k_146.toNat]!.z
                r24 := v147
                let inverse_148 := inverse
                let inverse_149 := inverse_148
                r25 := inverse_149
                let k_151 := k
                let v153 ← Mont_mul_Base r25 r24
                let inverse_154 := v153
                inverse := inverse_154
                let v155 := ()
                let v157 ← Mont_square_Base z_inverse
                let z_inverse2_158 := v157
                z_inverse2 := z_inverse2_158
                let k_159 := k
                checkIndex k_159 16
                let v160 := table[k_159.toNat]!.x
                r26 := v160
                let k_161 := k
                let v164 ← Mont_mul_Base r26 z_inverse2
                r27 := v164
                let v167 ← Mont_mul_Base z_inverse2 z_inverse
                r28 := v167
                let k_168 := k
                checkIndex k_168 16
                let v169 := table[k_168.toNat]!.y
                r29 := v169
                let k_170 := k
                let v173 ← Mont_mul_Base r29 r28
                r30 := v173
                let v176 ← Point_affine_Mont_Base r27 r30
                let k_177 := k
                checkIndex k_177 16
                table := (table.set! k_177.toNat v176)
                let v178 := ()
                b16 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
            else
              let v179 ← Point_infinity_Mont_Base
              let result_180 := v179
              result := result_180
              let v181 := (UInt256.ofNat 256)
              let bit_182 := v181
              bit := bit_182
              let rec b22 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
                let mut g := g
                let mut table := table
                let mut «prefix» := «prefix»
                let mut running := running
                let mut inverse := inverse
                let mut k := k
                let mut z_inverse := z_inverse
                let mut z_inverse2 := z_inverse2
                let mut result := result
                let mut bit := bit
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
                let bit_183 := bit
                let bit_184 := bit_183
                let v185 := (UInt256.ofNat 0)
                let v186 := decide (v185.toNat < bit_184.toNat)
                if v186 then
                  let v187 := (UInt256.ofNat 2)
                  let bit_188 := bit
                  let v189 ← IntTy.sub (.u 64) bit_188 v187
                  let bit_190 := v189
                  bit := bit_190
                  let v191 := ()
                  let result_192 := result
                  let result_193 := result_192
                  r31 := result_193
                  let v195 ← Point_double_Mont_Base r31
                  r32 := v195
                  let v197 ← Point_double_Mont_Base r32
                  let result_198 := v197
                  result := result_198
                  let v199 := ()
                  let u_200 := u_0
                  let bit_201 := bit
                  let v202 := bit_201
                  let v203 := u_200 >>> v202
                  let v204 := (UInt256.ofNat 3)
                  let v205 := v203 &&& v204
                  let v206 := (UInt256.ofNat 2)
                  let v207 := v205 <<< v206
                  let v_208 := v_1
                  let bit_209 := bit
                  let v210 := bit_209
                  let v211 := v_208 >>> v210
                  let v212 := (UInt256.ofNat 3)
                  let v213 := v211 &&& v212
                  let v214 := v207 ||| v213
                  let v215 ← downcast_truncate_u256_usize v214
                  let entry := v215
                  let v217 := (UInt256.ofNat 0)
                  let v218 := decide (entry ≠ v217)
                  let b27 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
                    let mut g := g
                    let mut table := table
                    let mut «prefix» := «prefix»
                    let mut running := running
                    let mut inverse := inverse
                    let mut k := k
                    let mut z_inverse := z_inverse
                    let mut z_inverse2 := z_inverse2
                    let mut result := result
                    let mut bit := bit
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
                    b22 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  let b26 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
                    let mut g := g
                    let mut table := table
                    let mut «prefix» := «prefix»
                    let mut running := running
                    let mut inverse := inverse
                    let mut k := k
                    let mut z_inverse := z_inverse
                    let mut z_inverse2 := z_inverse2
                    let mut result := result
                    let mut bit := bit
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
                    b27 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  if v218 then
                    checkIndex entry 16
                    let v219 := table[entry.toNat]!
                    r33 := v219
                    let v221 ← Point_is_infinity_Mont_Base r33
                    let v222 := !v221
                    if v222 then
                      checkIndex entry 16
                      let v223 := table[entry.toNat]!.x
                      r34 := v223
                      checkIndex entry 16
                      let v224 := table[entry.toNat]!.y
                      r35 := v224
                      let result_225 := result
                      let result_226 := result_225
                      r36 := result_226
                      let v230 ← Point_add_affine_Mont_Base r36 r34 r35
                      let result_231 := v230
                      result := result_231
                      let v232 := ()
                      let v233 := v232
                      b27 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                    else
                      b26 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  else
                    b26 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                else
                  let result_236 := result
                  return result_236
              partial_fixpoint
              b22 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          partial_fixpoint
          b16 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
      partial_fixpoint
      b9 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v87
  partial_fixpoint
  b1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v44

def combination_k1 (u_0 : UInt256) (v_1 : UInt256) (v20 : Point_Mont_Base) (v21 : UInt256) (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut g := g
  let mut table := table
  let mut «prefix» := «prefix»
  let mut running := running
  let mut inverse := inverse
  let mut k := k
  let mut z_inverse := z_inverse
  let mut z_inverse2 := z_inverse2
  let mut result := result
  let mut bit := bit
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
  table := (table.set! v21.toNat v20)
  let v22 := ()
  let v23 := (UInt256.ofNat 4)
  let g_24 := g
  checkIndex v23 16
  table := (table.set! v23.toNat g_24)
  let v25 := ()
  let v27 ← Point_double_Mont_Base g
  let v28 := (UInt256.ofNat 8)
  checkIndex v28 16
  table := (table.set! v28.toNat v27)
  let v29 := ()
  let v30 := (UInt256.ofNat 8)
  checkIndex v30 16
  let v31 := table[v30.toNat]!
  r11 := v31
  let v34 ← Point_add_Mont_Base r11 g
  let v35 := (UInt256.ofNat 12)
  checkIndex v35 16
  table := (table.set! v35.toNat v34)
  let v36 := ()
  let v37 := (UInt256.ofNat 4)
  let v38 := v37
  let v39 := (UInt256.ofNat 8)
  combination_k2 u_0 v_1 v38 v39 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36

def combination (u_0 : UInt256) (v_1 : UInt256) (q : Point_Mont_Base) : FeM Point_Mont_Base := do
  let mut g : Point_Mont_Base := default
  let mut table : (Vector Point_Mont_Base 16) := default
  let mut «prefix» : (Vector Mont_Base 16) := default
  let mut running : Mont_Base := default
  let mut inverse : Mont_Base := default
  let mut k : UInt256 := default
  let mut z_inverse : Mont_Base := default
  let mut z_inverse2 : Mont_Base := default
  let mut result : Point_Mont_Base := default
  let mut bit : UInt256 := default
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
  let g_7 := v6
  g := g_7
  let v8 ← Point_infinity_Mont_Base
  let v9 := Vector.replicate 16 v8
  let table_10 := v9
  table := table_10
  let v11 := q
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 16
  table := (table.set! v12.toNat v11)
  let v13 := ()
  let v14 ← Point_double_Mont_Base q
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 16
  table := (table.set! v15.toNat v14)
  let v16 := ()
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  let v18 := table[v17.toNat]!
  r10 := v18
  let v20 ← Point_add_Mont_Base r10 q
  let v21 := (UInt256.ofNat 3)
  checkIndex v21 16
  combination_k1 u_0 v_1 v20 v21 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36

def Point_to_affine_Mont_Base (self : Point_Mont_Base) : FeM (Mont_Base × Mont_Base) := do
  let mut z_inv : Mont_Base := default
  let mut z_inv2 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  let v1 := self.z
  r2 := v1
  let v3 ← Mont_inverse_Base r2
  let z_inv_4 := v3
  z_inv := z_inv_4
  let v6 ← Mont_square_Base z_inv
  let z_inv2_7 := v6
  z_inv2 := z_inv2_7
  let v8 := self.x
  r3 := v8
  let v11 ← Mont_mul_Base r3 z_inv2
  let v14 ← Mont_mul_Base z_inv2 z_inv
  r4 := v14
  let v15 := self.y
  r5 := v15
  let v18 ← Mont_mul_Base r5 r4
  let v19 := (v11, v18)
  return v19

def Mont_neg_Base (self : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let v1 ← Mont_zero_Base
  r0 := v1
  let v3 ← Mont_sub_Base r0 self
  return v3

def recover (hash_0 : UInt256) (recovery_id_1 : UInt256) (r_2 : UInt256) (s_3 : UInt256) : FeM Option_PublicKey := do
  let mut x : Mont_Base := default
  let mut alpha : Mont_Base := default
  let mut beta : Mont_Base := default
  let mut y : Mont_Base := default
  let mut r_inverse : Mont_Order := default
  let mut key : Point_Mont_Base := default
  let mut key_x : Mont_Base := default
  let mut key_y : Mont_Base := default
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
  let recovery_id_6 := recovery_id_1
  let v7 := (UInt256.ofNat 1)
  let v8 := decide (recovery_id_6.toNat ≤ v7.toNat)
  let b3 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → Bool → FeM Option_PublicKey := fun x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v4 => do
    let mut x := x
    let mut alpha := alpha
    let mut beta := beta
    let mut y := y
    let mut r_inverse := r_inverse
    let mut key := key
    let mut key_x := key_x
    let mut key_y := key_y
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
    let v25 ← «assert» v4
    let v26 ← Mont_new_Base r_2
    let x_27 := v26
    x := x_27
    let v29 ← Mont_square_Base x
    r8 := v29
    let v32 ← Mont_mul_Base r8 x
    r9 := v32
    let v33 := (UInt256.ofNat 7)
    let v34 ← Mont_new_Base v33
    r10 := v34
    let v37 ← Mont_add_Base r9 r10
    let alpha_38 := v37
    alpha := alpha_38
    let v39 ← P
    let v40 := (UInt256.ofNat 1)
    let v41 ← checkedAdd v39 v40
    let v42 := (UInt256.ofNat 4)
    let v43 ← checkedDiv v41 v42
    let v45 ← Mont_pow_Base alpha v43
    let beta_46 := v45
    beta := beta_46
    let v48 ← Mont_square_Base beta
    r11 := v48
    let v51 ← Mont_equals_Base r11 alpha
    let v52 := !v51
    if v52 then
      let v53 := Option_PublicKey.None
      return v53
    else
      let v54 := ()
      let v55 := v54
      let v57 ← Mont_value_Base beta
      let v58 := (UInt256.ofNat 1)
      let v59 := v57 &&& v58
      let recovery_id_60 := recovery_id_1
      let v61 := decide (v59 = recovery_id_60)
      let b13 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → Mont_Base → FeM Option_PublicKey := fun x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v5 => do
        let mut x := x
        let mut alpha := alpha
        let mut beta := beta
        let mut y := y
        let mut r_inverse := r_inverse
        let mut key := key
        let mut key_x := key_x
        let mut key_y := key_y
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
        let y_67 := v5
        y := y_67
        let v68 ← Mont_new_Order r_2
        r12 := v68
        let v70 ← Mont_inverse_Order r12
        let r_inverse_71 := v70
        r_inverse := r_inverse_71
        let hash_72 := hash_0
        let v73 ← N
        let v74 ← checkedRem hash_72 v73
        let v75 ← Mont_new_Order v74
        r13 := v75
        let v77 ← Mont_neg_Order r13
        r14 := v77
        let v80 ← Mont_mul_Order r14 r_inverse
        r15 := v80
        let v82 ← Mont_value_Order r15
        let u := v82
        let v84 ← Mont_new_Order s_3
        r16 := v84
        let v87 ← Mont_mul_Order r16 r_inverse
        r17 := v87
        let v89 ← Mont_value_Order r17
        let v := v89
        let v93 ← Point_affine_Mont_Base x y
        r18 := v93
        let v95 ← combination u v r18
        let key_96 := v95
        key := key_96
        let v98 ← Point_is_infinity_Mont_Base key
        if v98 then
          let v99 := Option_PublicKey.None
          return v99
        else
          let v100 := ()
          let v101 := v100
          let v103 ← Point_to_affine_Mont_Base key
          let v104 := v103.fst
          let key_x_105 := v104
          key_x := key_x_105
          let v106 := v103.snd
          let key_y_107 := v106
          key_y := key_y_107
          let v109 ← Mont_value_Base key_x
          let v111 ← Mont_value_Base key_y
          let v112 := (PublicKey.mk v109 v111)
          let v113 := Option_PublicKey.Some v112
          return v113
      if v61 then
        let beta_62 := beta
        let beta_63 := beta_62
        b13 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 beta_63
      else
        let v65 ← Mont_neg_Base beta
        let v66 := v65
        b13 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v66
  let b2 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → FeM Option_PublicKey := fun x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 => do
    let mut x := x
    let mut alpha := alpha
    let mut beta := beta
    let mut y := y
    let mut r_inverse := r_inverse
    let mut key := key
    let mut key_x := key_x
    let mut key_y := key_y
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
    b3 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v24
  if v8 then
    let r_9 := r_2
    let v10 := (UInt256.ofNat 0)
    let v11 := decide (r_9 ≠ v10)
    if v11 then
      let r_12 := r_2
      let v13 ← N
      let v14 := decide (r_12.toNat < v13.toNat)
      if v14 then
        let s_15 := s_3
        let v16 := (UInt256.ofNat 0)
        let v17 := decide (s_15 ≠ v16)
        if v17 then
          let s_18 := s_3
          let v19 ← N
          let v20 := decide (s_18.toNat < v19.toNat)
          if v20 then
            let v21 := true
            let v22 := v21
            b3 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v22
          else
            b2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
        else
          b2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
      else
        b2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
    else
      b2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
  else
    b2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18

def MemBuffer_len_1 (self : MemBuffer) : FeM UInt256 := do
  let v1 ← MemBuffer_len self
  return v1

def MemBuffer_lane_at (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_read_le_u64 self index
  return v2

def MemBuffer_byte_at (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_index_ref self index
  let v3 ← mloadInt (.u 8) v2
  return v3

def word_at_MemBuffer (input : MemBuffer) (offset_1 : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut r1 : Range_Unknown_Unknown := default
  let offset_4 := offset_1
  let v5 := (UInt256.ofNat 32)
  let v6 ← checkedAdd offset_4 v5
  let v7 ← MemBuffer_len_1 input
  let v8 := decide (v6.toNat ≤ v7.toNat)
  if v8 then
    let v9 ← MemBuffer_lane_at input offset_1
    let v10 := IntTy.swapBytes (.u 64) v9
    let v11 := v10
    let v12 := (UInt256.ofNat 192)
    let v13 := v11 <<< v12
    let offset_14 := offset_1
    let v15 := (UInt256.ofNat 8)
    let v16 ← checkedAdd offset_14 v15
    let v17 ← MemBuffer_lane_at input v16
    let v18 := IntTy.swapBytes (.u 64) v17
    let v19 := v18
    let v20 := (UInt256.ofNat 128)
    let v21 := v19 <<< v20
    let v22 := v13 ||| v21
    let offset_23 := offset_1
    let v24 := (UInt256.ofNat 16)
    let v25 ← checkedAdd offset_23 v24
    let v26 ← MemBuffer_lane_at input v25
    let v27 := IntTy.swapBytes (.u 64) v26
    let v28 := v27
    let v29 := (UInt256.ofNat 64)
    let v30 := v28 <<< v29
    let v31 := v22 ||| v30
    let offset_32 := offset_1
    let v33 := (UInt256.ofNat 24)
    let v34 ← checkedAdd offset_32 v33
    let v35 ← MemBuffer_lane_at input v34
    let v36 := IntTy.swapBytes (.u 64) v35
    let v37 := v36
    let v38 := v31 ||| v37
    return v38
  else
    let v39 := ()
    let v40 := v39
    let v41 := (UInt256.ofNat 0)
    let value_42 := v41
    value := value_42
    let offset_43 := offset_1
    let offset_44 := offset_1
    let v45 := (UInt256.ofNat 32)
    let v46 ← checkedAdd offset_44 v45
    let v47 := ()
    let v48 := (Range_Unknown_Unknown.mk offset_43 v46)
    r1 := v48
    let v50 ← Range_Unknown_Unknown_start r1
    let v51 := v50
    let rec b4 (value : UInt256) (r1 : Range_Unknown_Unknown) (v2 : UInt256) : FeM UInt256 := do
      let mut value := value
      let mut r1 := r1
      let v53 ← Range_Unknown_Unknown_has r1 v2
      if v53 then
        let v55 ← Range_Unknown_Unknown_at r1 v2
        let index := v55
        let value_57 := value
        let value_58 := value_57
        let v59 := (UInt256.ofNat 8)
        let v60 := value_58 <<< v59
        let v61 ← MemBuffer_len_1 input
        let v62 := decide (index.toNat < v61.toNat)
        let b10 : UInt256 → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun value r1 v3 => do
          let mut value := value
          let mut r1 := r1
          let v68 := v60 ||| v3
          let value_69 := v68
          value := value_69
          let v70 := ()
          let v72 ← Range_Unknown_Unknown_next r1 v2
          let v73 := v72
          b4 value r1 v73
        if v62 then
          let v63 ← MemBuffer_byte_at input index
          let v64 := v63
          let v65 := v64
          b10 value r1 v65
        else
          let v66 := (UInt256.ofNat 0)
          let v67 := v66
          b10 value r1 v67
      else
        let value_74 := value
        return value_74
    partial_fixpoint
    b4 value r1 v51

def MODULUS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 21888242871839275222246405745257275088696311157297823662689037894645226208583)
  return v0

def Bn254Base_VALUE : FeM UInt256 := do
  let v0 ← MODULUS
  return v0

def raw_Bn254Base (value_0 : UInt256) : FeM Mont_Bn254Base := do
  let v1 ← downcast_truncate_u256_u64 value_0
  let value_2 := value_0
  let v3 := (UInt256.ofNat 64)
  let v4 := value_2 >>> v3
  let v5 ← downcast_truncate_u256_u64 v4
  let value_6 := value_0
  let v7 := (UInt256.ofNat 128)
  let v8 := value_6 >>> v7
  let v9 ← downcast_truncate_u256_u64 v8
  let value_10 := value_0
  let v11 := (UInt256.ofNat 192)
  let v12 := value_10 >>> v11
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

def Mont_shift_reduced_Bn254Base_k3 (s5 : UInt256) (u1_34 : UInt256) (u2_50 : UInt256) (u3_66 : UInt256) (cb_73 : UInt256) (v75 : (UInt256 × UInt256)) (u4_77 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v78 := v75.snd
  let ca_79 := v78
  let v80 ← adc s5 cb_73 ca_79
  let v81 := v80.fst
  let u5 := v81
  let v83 := (Mont_Bn254Base.mk u1_34 u2_50 u3_66 u4_77)
  let v84 := (v83, u5)
  return v84

def Mont_shift_reduced_Bn254Base_k2 (s4 : UInt256) (s5 : UInt256) (m : UInt256) (u1_34 : UInt256) (u2_50 : UInt256) (ca_52 : UInt256) (v53 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  let v54 := v53.fst
  let u3_55 := v54
  let v56 := v53.snd
  let cb_57 := v56
  let v58 ← Bn254Base_P3
  let v59 ← mul_wide m v58
  let v60 := v59.fst
  let k3 := v60
  let v62 := v59.snd
  let j3 := v62
  let v64 ← adc u3_55 k3 ca_52
  let v65 := v64.fst
  let u3_66 := v65
  let v67 := v64.snd
  let ca_68 := v67
  let v69 ← adc s4 j3 cb_57
  let v70 := v69.fst
  let u4_71 := v70
  let v72 := v69.snd
  let cb_73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc u4_71 v74 ca_68
  let v76 := v75.fst
  let u4_77 := v76
  Mont_shift_reduced_Bn254Base_k3 s5 u1_34 u2_50 u3_66 cb_73 v75 u4_77

def Mont_shift_reduced_Bn254Base_k1 (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (ca_19 : UInt256) (u1_23 : UInt256) (cb_25 : UInt256) (v27 : (UInt256 × UInt256)) (k1 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v30 := v27.snd
  let j1 := v30
  let v32 ← adc u1_23 k1 ca_19
  let v33 := v32.fst
  let u1_34 := v33
  let v35 := v32.snd
  let ca_36 := v35
  let v37 ← adc s2 j1 cb_25
  let v38 := v37.fst
  let u2_39 := v38
  let v40 := v37.snd
  let cb_41 := v40
  let v42 ← Bn254Base_P2
  let v43 ← mul_wide m v42
  let v44 := v43.fst
  let k2 := v44
  let v46 := v43.snd
  let j2 := v46
  let v48 ← adc u2_39 k2 ca_36
  let v49 := v48.fst
  let u2_50 := v49
  let v51 := v48.snd
  let ca_52 := v51
  let v53 ← adc s3 j2 cb_41
  Mont_shift_reduced_Bn254Base_k2 s4 s5 m u1_34 u2_50 ca_52 v53

def Mont_shift_reduced_Bn254Base (s0_0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v6 ← Bn254Base_INV
  let s0_7 := s0_0
  let v8 ← u64_wrapping_mul s0_7 v6
  let m := v8
  let v10 ← Bn254Base_P0
  let v11 ← mul_wide m v10
  let v12 := v11.fst
  let k0 := v12
  let v14 := v11.snd
  let j0 := v14
  let v16 := (UInt256.ofNat 0)
  let v17 ← adc s0_0 k0 v16
  let v18 := v17.snd
  let ca_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc s1 j0 v20
  let v22 := v21.fst
  let u1_23 := v22
  let v24 := v21.snd
  let cb_25 := v24
  let v26 ← Bn254Base_P1
  let v27 ← mul_wide m v26
  let v28 := v27.fst
  let k1 := v28
  Mont_shift_reduced_Bn254Base_k1 s2 s3 s4 s5 m ca_19 u1_23 cb_25 v27 k1

def Mont_first_row_Bn254Base_k1 (self : Mont_Bn254Base) (b : UInt256) (s0 : UInt256) (h1 : UInt256) (s1 : UInt256) (c_19 : UInt256) (l2 : UInt256) (h2 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v26 ← adc h1 l2 c_19
  let v27 := v26.fst
  let s2 := v27
  let v29 := v26.snd
  let c_30 := v29
  let v31 := self.l3
  let v32 ← mul_wide b v31
  let v33 := v32.fst
  let l3 := v33
  let v35 := v32.snd
  let h3 := v35
  let v37 ← adc h2 l3 c_30
  let v38 := v37.fst
  let s3 := v38
  let v40 := v37.snd
  let c_41 := v40
  let v42 := (UInt256.ofNat 0)
  let v43 ← adc h3 v42 c_41
  let v44 := v43.fst
  let s4 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 ← Mont_shift_reduced_Bn254Base s0 s1 s2 s3 s4 v46
  return v47

def Mont_first_row_Bn254Base (self : Mont_Bn254Base) (b : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v2 := self.l0
  let v3 ← mul_wide b v2
  let v4 := v3.fst
  let s0 := v4
  let v6 := v3.snd
  let h0 := v6
  let v8 := self.l1
  let v9 ← mul_wide b v8
  let v10 := v9.fst
  let l1 := v10
  let v12 := v9.snd
  let h1 := v12
  let v14 := (UInt256.ofNat 0)
  let v15 ← adc h0 l1 v14
  let v16 := v15.fst
  let s1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := self.l2
  let v21 ← mul_wide b v20
  let v22 := v21.fst
  let l2 := v22
  let v24 := v21.snd
  let h2 := v24
  Mont_first_row_Bn254Base_k1 self b s0 h1 s1 c_19 l2 h2

def Mont_row_Bn254Base_k3 (s0 : UInt256) (s1_32 : UInt256) (s2_49 : UInt256) (s3_66 : UInt256) (cb_73 : UInt256) (v75 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  let v76 := v75.fst
  let s4_77 := v76
  let v78 := v75.snd
  let ca_79 := v78
  let v80 := (UInt256.ofNat 0)
  let v81 ← adc cb_73 v80 ca_79
  let v82 := v81.fst
  let s5 := v82
  let v84 ← Mont_shift_reduced_Bn254Base s0 s1_32 s2_49 s3_66 s4_77 s5
  return v84

def Mont_row_Bn254Base_k2 (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) (s0 : UInt256) (s1_32 : UInt256) (cb_40 : UInt256) (h2 : UInt256) (s2_49 : UInt256) (ca_51 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v52 := t.l3
  let v53 ← adc v52 h2 cb_40
  let v54 := v53.fst
  let s3_55 := v54
  let v56 := v53.snd
  let cb_57 := v56
  let v58 := self.l3
  let v59 ← mul_wide b v58
  let v60 := v59.fst
  let l3 := v60
  let v62 := v59.snd
  let h3 := v62
  let v64 ← adc s3_55 l3 ca_51
  let v65 := v64.fst
  let s3_66 := v65
  let v67 := v64.snd
  let ca_68 := v67
  let v69 ← adc top h3 cb_57
  let v70 := v69.fst
  let s4_71 := v70
  let v72 := v69.snd
  let cb_73 := v72
  let v74 := (UInt256.ofNat 0)
  let v75 ← adc s4_71 v74 ca_68
  Mont_row_Bn254Base_k3 s0 s1_32 s2_49 s3_66 cb_73 v75

def Mont_row_Bn254Base_k1 (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) (s0 : UInt256) (ca_16 : UInt256) (s1_21 : UInt256) (cb_23 : UInt256) (v25 : (UInt256 × UInt256)) (l1 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v28 := v25.snd
  let h1 := v28
  let v30 ← adc s1_21 l1 ca_16
  let v31 := v30.fst
  let s1_32 := v31
  let v33 := v30.snd
  let ca_34 := v33
  let v35 := t.l2
  let v36 ← adc v35 h1 cb_23
  let v37 := v36.fst
  let s2_38 := v37
  let v39 := v36.snd
  let cb_40 := v39
  let v41 := self.l2
  let v42 ← mul_wide b v41
  let v43 := v42.fst
  let l2 := v43
  let v45 := v42.snd
  let h2 := v45
  let v47 ← adc s2_38 l2 ca_34
  let v48 := v47.fst
  let s2_49 := v48
  let v50 := v47.snd
  let ca_51 := v50
  Mont_row_Bn254Base_k2 self b t top s0 s1_32 cb_40 h2 s2_49 ca_51

def Mont_row_Bn254Base (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v4 := self.l0
  let v5 ← mul_wide b v4
  let v6 := v5.fst
  let l0 := v6
  let v8 := v5.snd
  let h0 := v8
  let v10 := t.l0
  let v11 := (UInt256.ofNat 0)
  let v12 ← adc v10 l0 v11
  let v13 := v12.fst
  let s0 := v13
  let v15 := v12.snd
  let ca_16 := v15
  let v17 := t.l1
  let v18 := (UInt256.ofNat 0)
  let v19 ← adc v17 h0 v18
  let v20 := v19.fst
  let s1_21 := v20
  let v22 := v19.snd
  let cb_23 := v22
  let v24 := self.l1
  let v25 ← mul_wide b v24
  let v26 := v25.fst
  let l1 := v26
  Mont_row_Bn254Base_k1 self b t top s0 ca_16 s1_21 cb_23 v25 l1

def Mont_reduce_Bn254Base (self : Mont_Bn254Base) (top : UInt256) : FeM Mont_Bn254Base := do
  let v3 := self.l0
  let v4 ← Bn254Base_P0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let d0 := v7
  let v9 := v6.snd
  let b_10 := v9
  let v11 := self.l1
  let v12 ← Bn254Base_P1
  let v13 ← sbb v11 v12 b_10
  let v14 := v13.fst
  let d1 := v14
  let v16 := v13.snd
  let b_17 := v16
  let v18 := self.l2
  let v19 ← Bn254Base_P2
  let v20 ← sbb v18 v19 b_17
  let v21 := v20.fst
  let d2 := v21
  let v23 := v20.snd
  let b_24 := v23
  let v25 := self.l3
  let v26 ← Bn254Base_P3
  let v27 ← sbb v25 v26 b_24
  let v28 := v27.fst
  let d3 := v28
  let v30 := v27.snd
  let b_31 := v30
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb top v32 b_31
  let v34 := v33.snd
  let b_35 := v34
  let v36 := (UInt256.ofNat 1)
  let v37 := decide (b_35 = v36)
  let b3 : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  if v37 then
    let v38 := self
    let v39 := v38
    b3 v39
  else
    let v40 := (Mont_Bn254Base.mk d0 d1 d2 d3)
    let v41 := v40
    b3 v41

def Mont_mul_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut t_r0 : Mont_Bn254Base := default
  let mut t_r1 : Mont_Bn254Base := default
  let mut t_r2 : Mont_Bn254Base := default
  let mut t_r3 : Mont_Bn254Base := default
  let v2 := other.l0
  let v3 ← Mont_first_row_Bn254Base self v2
  let v4 := v3.fst
  let t_5 := v4
  t_r0 := t_5
  let v6 := v3.snd
  let top_7 := v6
  let v8 := other.l1
  let v10 ← Mont_row_Bn254Base self v8 t_r0 top_7
  let v11 := v10.fst
  let t_12 := v11
  t_r1 := t_12
  let v13 := v10.snd
  let top_14 := v13
  let v15 := other.l2
  let v17 ← Mont_row_Bn254Base self v15 t_r1 top_14
  let v18 := v17.fst
  let t_19 := v18
  t_r2 := t_19
  let v20 := v17.snd
  let top_21 := v20
  let v22 := other.l3
  let v24 ← Mont_row_Bn254Base self v22 t_r2 top_21
  let v25 := v24.fst
  let t_26 := v25
  t_r3 := t_26
  let v27 := v24.snd
  let top_28 := v27
  let v30 ← Mont_reduce_Bn254Base t_r3 top_28
  return v30

def Mont_new_Bn254Base (value_0 : UInt256) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let value_1 := value_0
  let v2 ← Bn254Base_VALUE
  let v3 := decide (value_1.toNat < v2.toNat)
  let v4 ← «assert» v3
  let v5 ← raw_Bn254Base value_0
  r0 := v5
  let v6 ← Bn254Base_R2
  let v7 ← raw_Bn254Base v6
  r1 := v7
  let v10 ← Mont_mul_Bn254Base r0 r1
  return v10

def fp (value : UInt256) : FeM Mont_Bn254Base := do
  let v1 ← Mont_new_Bn254Base value
  return v1

def Mont_one_Bn254Base : FeM Mont_Bn254Base := do
  let v0 := (UInt256.ofNat 1)
  let v1 ← Mont_new_Bn254Base v0
  return v1

def Point_affine_Mont_Bn254Base (x : Mont_Bn254Base) (y : Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let v2 := x
  let v3 := y
  let v4 ← Mont_one_Bn254Base
  let v5 := (Point_Mont_Bn254Base.mk v2 v3 v4)
  return v5

def Mont_square_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v1 ← Mont_mul_Bn254Base self self
  return v1

def Mont_add_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let v2 := self.l0
  let v3 := other.l0
  let v4 := (UInt256.ofNat 0)
  let v5 ← adc v2 v3 v4
  let v6 := v5.fst
  let s0 := v6
  let v8 := v5.snd
  let c_9 := v8
  let v10 := self.l1
  let v11 := other.l1
  let v12 ← adc v10 v11 c_9
  let v13 := v12.fst
  let s1 := v13
  let v15 := v12.snd
  let c_16 := v15
  let v17 := self.l2
  let v18 := other.l2
  let v19 ← adc v17 v18 c_16
  let v20 := v19.fst
  let s2 := v20
  let v22 := v19.snd
  let c_23 := v22
  let v24 := self.l3
  let v25 := other.l3
  let v26 ← adc v24 v25 c_23
  let v27 := v26.fst
  let s3 := v27
  let v29 := v26.snd
  let c_30 := v29
  let v31 := (Mont_Bn254Base.mk s0 s1 s2 s3)
  r0 := v31
  let v33 ← Mont_reduce_Bn254Base r0 c_30
  return v33

def Mont_equals_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Bool := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := decide (v3 = v4)
  let b3 : Bool → FeM Bool := fun v2 => do
    return v2
  let b2 : FeM Bool := do
    let v17 := false
    let v18 := v17
    b3 v18
  if v5 then
    let v6 := self.l1
    let v7 := other.l1
    let v8 := decide (v6 = v7)
    if v8 then
      let v9 := self.l2
      let v10 := other.l2
      let v11 := decide (v9 = v10)
      if v11 then
        let v12 := self.l3
        let v13 := other.l3
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

def on_curve_Mont_Bn254Base (x : Mont_Bn254Base) (y : Mont_Bn254Base) (b : Mont_Bn254Base) : FeM Bool := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let v3 ← Mont_square_Bn254Base y
  r0 := v3
  let v4 ← Mont_square_Bn254Base x
  r1 := v4
  let v6 ← Mont_mul_Bn254Base r1 x
  r2 := v6
  let v8 ← Mont_add_Bn254Base r2 b
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

def g1_point_MemBuffer (input : MemBuffer) (offset_1 : UInt256) : FeM Option_Point_Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let v2 ← word_at_MemBuffer input offset_1
  let x := v2
  let offset_4 := offset_1
  let v5 := (UInt256.ofNat 32)
  let v6 ← checkedAdd offset_4 v5
  let v7 ← word_at_MemBuffer input v6
  let y := v7
  let v9 ← MODULUS
  let v10 := decide (v9.toNat ≤ x.toNat)
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
    let v12 := decide (v11.toNat ≤ y.toNat)
    if v12 then
      b1 r0 r1 r2 r3 r4
    else
      let v13 := ()
      let v14 := v13
      let v15 := (UInt256.ofNat 0)
      let v16 := decide (x = v15)
      let b6 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let v21 := ()
        let v22 := v21
        let v23 ← fp x
        r0 := v23
        let v24 ← fp y
        r1 := v24
        let v27 ← Point_affine_Mont_Bn254Base r0 r1
        let point := v27
        let v29 := point.x
        r2 := v29
        let v30 := point.y
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
          let v41 := Option_Point_Mont_Bn254Base.Some point
          return v41
      if v16 then
        let v17 := (UInt256.ofNat 0)
        let v18 := decide (y = v17)
        if v18 then
          let v19 ← Point_infinity_Mont_Bn254Base
          let v20 := Option_Point_Mont_Bn254Base.Some v19
          return v20
        else
          b6 r0 r1 r2 r3 r4
      else
        b6 r0 r1 r2 r3 r4

def Mont_is_zero_Bn254Base (self : Mont_Bn254Base) : FeM Bool := do
  let v2 := self.l0
  let v3 := (UInt256.ofNat 0)
  let v4 := decide (v2 = v3)
  let b3 : Bool → FeM Bool := fun v1 => do
    return v1
  let b2 : FeM Bool := do
    let v16 := false
    let v17 := v16
    b3 v17
  if v4 then
    let v5 := self.l1
    let v6 := (UInt256.ofNat 0)
    let v7 := decide (v5 = v6)
    if v7 then
      let v8 := self.l2
      let v9 := (UInt256.ofNat 0)
      let v10 := decide (v8 = v9)
      if v10 then
        let v11 := self.l3
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

def Point_is_infinity_Mont_Bn254Base (self : Point_Mont_Bn254Base) : FeM Bool := do
  let mut r0 : Mont_Bn254Base := default
  let v1 := self.z
  r0 := v1
  let v3 ← Mont_is_zero_Bn254Base r0
  return v3

def Mont_sub_Bn254Base_k2 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (b_31 : UInt256) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (c_50 : UInt256) : FeM Mont_Bn254Base := do
  let v51 ← Bn254Base_P3
  let v52 ← adc d3 v51 c_50
  let v53 := v52.fst
  let s3 := v53
  let v55 := (UInt256.ofNat 1)
  let v56 := decide (b_31 = v55)
  let b3 : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Bn254Base.mk s0 s1 s2 s3)
    let v58 := v57
    b3 v58
  else
    let v59 := (Mont_Bn254Base.mk d0 d1 d2 d3)
    let v60 := v59
    b3 v60

def Mont_sub_Bn254Base_k1 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (b_24 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Mont_Bn254Base := do
  let v27 ← sbb v25 v26 b_24
  let v28 := v27.fst
  let d3 := v28
  let v30 := v27.snd
  let b_31 := v30
  let v32 ← Bn254Base_P0
  let v33 := (UInt256.ofNat 0)
  let v34 ← adc d0 v32 v33
  let v35 := v34.fst
  let s0 := v35
  let v37 := v34.snd
  let c_38 := v37
  let v39 ← Bn254Base_P1
  let v40 ← adc d1 v39 c_38
  let v41 := v40.fst
  let s1 := v41
  let v43 := v40.snd
  let c_44 := v43
  let v45 ← Bn254Base_P2
  let v46 ← adc d2 v45 c_44
  let v47 := v46.fst
  let s2 := v47
  let v49 := v46.snd
  let c_50 := v49
  Mont_sub_Bn254Base_k2 d0 d1 d2 d3 b_31 s0 s1 s2 c_50

def Mont_sub_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := (UInt256.ofNat 0)
  let v6 ← sbb v3 v4 v5
  let v7 := v6.fst
  let d0 := v7
  let v9 := v6.snd
  let b_10 := v9
  let v11 := self.l1
  let v12 := other.l1
  let v13 ← sbb v11 v12 b_10
  let v14 := v13.fst
  let d1 := v14
  let v16 := v13.snd
  let b_17 := v16
  let v18 := self.l2
  let v19 := other.l2
  let v20 ← sbb v18 v19 b_17
  let v21 := v20.fst
  let d2 := v21
  let v23 := v20.snd
  let b_24 := v23
  let v25 := self.l3
  let v26 := other.l3
  Mont_sub_Bn254Base_k1 d0 d1 d2 b_24 v25 v26

def Mont_double_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v1 ← Mont_add_Bn254Base self self
  return v1

def Point_double_Mont_Bn254Base (self : Point_Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let mut a : Mont_Bn254Base := default
  let mut b : Mont_Bn254Base := default
  let mut c : Mont_Bn254Base := default
  let mut d : Mont_Bn254Base := default
  let mut e : Mont_Bn254Base := default
  let mut x : Mont_Bn254Base := default
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
  let v1 ← Point_is_infinity_Mont_Bn254Base self
  let b1 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 => do
    let mut a := a
    let mut b := b
    let mut c := c
    let mut d := d
    let mut e := e
    let mut x := x
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
    b1 a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    let v2 := self.y
    r6 := v2
    let v4 ← Mont_is_zero_Bn254Base r6
    if v4 then
      b1 a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      let v5 := ()
      let v6 := v5
      let v7 := self.x
      r7 := v7
      let v9 ← Mont_square_Bn254Base r7
      let a_10 := v9
      a := a_10
      let v11 := self.y
      r8 := v11
      let v13 ← Mont_square_Bn254Base r8
      let b_14 := v13
      b := b_14
      let v16 ← Mont_square_Bn254Base b
      let c_17 := v16
      c := c_17
      let v18 := self.x
      r9 := v18
      let v21 ← Mont_add_Bn254Base r9 b
      r10 := v21
      let v23 ← Mont_square_Bn254Base r10
      r11 := v23
      let v26 ← Mont_sub_Bn254Base r11 a
      r12 := v26
      let v29 ← Mont_sub_Bn254Base r12 c
      r13 := v29
      let v31 ← Mont_double_Bn254Base r13
      let d_32 := v31
      d := d_32
      let v34 ← Mont_double_Bn254Base a
      r14 := v34
      let v37 ← Mont_add_Bn254Base r14 a
      let e_38 := v37
      e := e_38
      let v40 ← Mont_square_Bn254Base e
      r15 := v40
      let v42 ← Mont_double_Bn254Base d
      r16 := v42
      let v45 ← Mont_sub_Bn254Base r15 r16
      let x_46 := v45
      x := x_46
      let v49 ← Mont_sub_Bn254Base d x
      r17 := v49
      let v52 ← Mont_mul_Bn254Base e r17
      r18 := v52
      let v54 ← Mont_double_Bn254Base c
      r19 := v54
      let v56 ← Mont_double_Bn254Base r19
      r20 := v56
      let v58 ← Mont_double_Bn254Base r20
      r21 := v58
      let v61 ← Mont_sub_Bn254Base r18 r21
      let y := v61
      let v63 := self.z
      r22 := v63
      let v64 := self.y
      r23 := v64
      let v67 ← Mont_mul_Bn254Base r23 r22
      r24 := v67
      let v69 ← Mont_double_Bn254Base r24
      let x_70 := x
      let v71 := (Point_Mont_Bn254Base.mk x_70 y v69)
      return v71

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

def Mont_inverse_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let v1 := self.l0
  let v2 := self.l1
  let v3 := self.l2
  let v4 := self.l3
  let v5 := #v[v1, v2, v3, v4]
  r0 := v5
  let v6 ← Bn254Base_SIGNED_62
  r1 := v6
  let v7 ← Bn254Base_INV62
  let v10 ← invert_256 r0 r1 v7
  let limbs := v10
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v13 := limbs[v12.toNat]!
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v15 := limbs[v14.toNat]!
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v17 := limbs[v16.toNat]!
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v19 := limbs[v18.toNat]!
  let v20 := (Mont_Bn254Base.mk v13 v15 v17 v19)
  r2 := v20
  let v21 ← Bn254Base_R3
  let v22 ← raw_Bn254Base v21
  r3 := v22
  let v25 ← Mont_mul_Bn254Base r2 r3
  return v25

def Point_to_affine_Mont_Bn254Base (self : Point_Mont_Bn254Base) : FeM (Mont_Bn254Base × Mont_Bn254Base) := do
  let mut z_inv : Mont_Bn254Base := default
  let mut z_inv2 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let mut r5 : Mont_Bn254Base := default
  let v1 := self.z
  r2 := v1
  let v3 ← Mont_inverse_Bn254Base r2
  let z_inv_4 := v3
  z_inv := z_inv_4
  let v6 ← Mont_square_Bn254Base z_inv
  let z_inv2_7 := v6
  z_inv2 := z_inv2_7
  let v8 := self.x
  r3 := v8
  let v11 ← Mont_mul_Bn254Base r3 z_inv2
  let v14 ← Mont_mul_Bn254Base z_inv2 z_inv
  r4 := v14
  let v15 := self.y
  r5 := v15
  let v18 ← Mont_mul_Bn254Base r5 r4
  let v19 := (v11, v18)
  return v19

def Mont_value_Bn254Base (self : Mont_Bn254Base) : FeM UInt256 := do
  let mut r0 : Mont_Bn254Base := default
  let v1 := (UInt256.ofNat 1)
  let v2 ← raw_Bn254Base v1
  r0 := v2
  let v4 ← Mont_mul_Bn254Base self r0
  let value := v4
  let v6 := value.l0
  let v7 := v6
  let v8 := value.l1
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := v9 <<< v10
  let v12 := v7 ||| v11
  let v13 := value.l2
  let v14 := v13
  let v15 := (UInt256.ofNat 128)
  let v16 := v14 <<< v15
  let v17 := v12 ||| v16
  let v18 := value.l3
  let v19 := v18
  let v20 := (UInt256.ofNat 192)
  let v21 := v19 <<< v20
  let v22 := v17 ||| v21
  return v22

def fp_value (a : Mont_Bn254Base) : FeM UInt256 := do
  let v1 ← Mont_value_Bn254Base a
  return v1

def encode (point : Point_Mont_Bn254Base) : FeM (Vector UInt256 2) := do
  let mut x : Mont_Bn254Base := default
  let mut y : Mont_Bn254Base := default
  let v1 ← Point_is_infinity_Mont_Bn254Base point
  if v1 then
    let v2 := (UInt256.ofNat 0)
    let v3 := (UInt256.ofNat 0)
    let v4 := #v[v2, v3]
    return v4
  else
    let v5 := ()
    let v6 := v5
    let v7 ← Point_to_affine_Mont_Bn254Base point
    let v8 := v7.fst
    let x_9 := v8
    x := x_9
    let v10 := v7.snd
    let y_11 := v10
    y := y_11
    let v13 ← fp_value x
    let v15 ← fp_value y
    let v16 := #v[v13, v15]
    return v16

def SUCCESS : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def HOST_ALLOCATION : FeM UInt256 := do
  let v0 := (UInt256.ofNat 2)
  return v0

def host_allocation (gas_used_0 : UInt256) : FeM Output := do
  let v1 ← HOST_ALLOCATION
  let gas_used_2 := gas_used_0
  let v3 ← MemBuffer_empty
  let v4 := (Output.mk v1 gas_used_2 v3)
  return v4


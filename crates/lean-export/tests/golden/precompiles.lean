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

/-- `lib::secp256k1::N` -/
def N : FeM UInt256 := do
  return (UInt256.ofNat 115792089237316195423570985008687907852837564279074904382605163141518161494337)

/-- `lib::secp256k1::P` -/
def P : FeM UInt256 := do
  return (UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007908834671663)

/-- `<Base as Modulus>::VALUE` -/
def Base_VALUE : FeM UInt256 := do
  let v0 ← P
  return v0

/-- `lib::montgomery::raw<Base>` -/
def raw_Base (value : UInt256) : FeM Mont_Base := do
  let v1 ← downcast_truncate_u256_u64 value
  let v4 := value >>> (UInt256.ofNat 64)
  let v5 ← downcast_truncate_u256_u64 v4
  let v8 := value >>> (UInt256.ofNat 128)
  let v9 ← downcast_truncate_u256_u64 v8
  let v12 := value >>> (UInt256.ofNat 192)
  let v13 ← downcast_truncate_u256_u64 v12
  let v14 := (Mont_Base.mk v1 v5 v9 v13)
  return v14

/-- `<Base as Modulus>::R2` -/
def Base_R2 : FeM UInt256 := do
  return (UInt256.ofNat 18446752466076602529)

/-- `lib::montgomery::mul_wide` -/
def mul_wide (a : UInt256) (b : UInt256) : FeM (UInt256 × UInt256) := do
  let t ← IntTy.mul (.u 128) a b
  let v8 ← downcast_truncate_u128_u64 t
  let v10 := t >>> (UInt256.ofNat 64)
  let v11 ← downcast_truncate_u128_u64 v10
  let v12 := (v8, v11)
  return v12

/-- `lib::montgomery::adc` -/
def adc (a : UInt256) (b : UInt256) (carry : UInt256) : FeM (UInt256 × UInt256) := do
  let v7 ← IntTy.add (.u 128) a b
  let t ← IntTy.add (.u 128) v7 carry
  let v12 ← downcast_truncate_u128_u64 t
  let v14 := t >>> (UInt256.ofNat 64)
  let v15 ← downcast_truncate_u128_u64 v14
  let v16 := (v12, v15)
  return v16

/-- `<Base as Modulus>::INV` -/
def Base_INV : FeM UInt256 := do
  return (UInt256.ofNat 15580212934572586289)

/-- `<Base as Modulus>::P0` -/
def Base_P0 : FeM UInt256 := do
  return (UInt256.ofNat 18446744069414583343)

/-- `<Base as Modulus>::P1` -/
def Base_P1 : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551615)

/-- `<Base as Modulus>::P2` -/
def Base_P2 : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551615)

/-- `<Base as Modulus>::P3` -/
def Base_P3 : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551615)

def Mont_shift_reduced_Base_k1 (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (u1_33 : UInt256) (ca_35 : UInt256) (u2_38 : UInt256) (cb_40 : UInt256) (v43 : (UInt256 × UInt256)) (k2 : UInt256) : FeM (Mont_Base × UInt256) := do
  let j2 := v43.snd
  let v48 ← adc u2_38 k2 ca_35
  let u2_49 := v48.fst
  let ca_51 := v48.snd
  let v53 ← adc s3 j2 cb_40
  let u3_54 := v53.fst
  let cb_56 := v53.snd
  let v58 ← Base_P3
  let v59 ← mul_wide m v58
  let k3 := v59.fst
  let j3 := v59.snd
  let v64 ← adc u3_54 k3 ca_51
  let u3_65 := v64.fst
  let ca_67 := v64.snd
  let v69 ← adc s4 j3 cb_56
  let u4_70 := v69.fst
  let cb_72 := v69.snd
  let v75 ← adc u4_70 (UInt256.ofNat 0) ca_67
  let u4_76 := v75.fst
  let ca_78 := v75.snd
  let v80 ← adc s5 cb_72 ca_78
  let u5 := v80.fst
  let v83 := (Mont_Base.mk u1_33 u2_49 u3_65 u4_76)
  let v84 := (v83, u5)
  return v84

/-- `Mont<Base>::shift_reduced` -/
def Mont_shift_reduced_Base (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v6 ← Base_INV
  let m ← u64_wrapping_mul s0 v6
  let v10 ← Base_P0
  let v11 ← mul_wide m v10
  let k0 := v11.fst
  let j0 := v11.snd
  let v17 ← adc s0 k0 (UInt256.ofNat 0)
  let ca_18 := v17.snd
  let v21 ← adc s1 j0 (UInt256.ofNat 0)
  let u1_22 := v21.fst
  let cb_24 := v21.snd
  let v26 ← Base_P1
  let v27 ← mul_wide m v26
  let k1 := v27.fst
  let j1 := v27.snd
  let v32 ← adc u1_22 k1 ca_18
  let u1_33 := v32.fst
  let ca_35 := v32.snd
  let v37 ← adc s2 j1 cb_24
  let u2_38 := v37.fst
  let cb_40 := v37.snd
  let v42 ← Base_P2
  let v43 ← mul_wide m v42
  let k2 := v43.fst
  Mont_shift_reduced_Base_k1 s3 s4 s5 m u1_33 ca_35 u2_38 cb_40 v43 k2

/-- `Mont<Base>::first_row` -/
def Mont_first_row_Base (self : Mont_Base) (b : UInt256) : FeM (Mont_Base × UInt256) := do
  let v2 := self.l0
  let v3 ← mul_wide b v2
  let s0 := v3.fst
  let h0 := v3.snd
  let v8 := self.l1
  let v9 ← mul_wide b v8
  let l1 := v9.fst
  let h1 := v9.snd
  let v15 ← adc h0 l1 (UInt256.ofNat 0)
  let s1 := v15.fst
  let c_18 := v15.snd
  let v20 := self.l2
  let v21 ← mul_wide b v20
  let l2 := v21.fst
  let h2 := v21.snd
  let v26 ← adc h1 l2 c_18
  let s2 := v26.fst
  let c_29 := v26.snd
  let v31 := self.l3
  let v32 ← mul_wide b v31
  let l3 := v32.fst
  let h3 := v32.snd
  let v37 ← adc h2 l3 c_29
  let s3 := v37.fst
  let c_40 := v37.snd
  let v43 ← adc h3 (UInt256.ofNat 0) c_40
  let s4 := v43.fst
  let v47 ← Mont_shift_reduced_Base s0 s1 s2 s3 s4 (UInt256.ofNat 0)
  return v47

def Mont_row_Base_k2 (s0 : UInt256) (s1_31 : UInt256) (s2_48 : UInt256) (s3_65 : UInt256) (s4_76 : UInt256) (v81 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  let s5 := v81.fst
  let v84 ← Mont_shift_reduced_Base s0 s1_31 s2_48 s3_65 s4_76 s5
  return v84

def Mont_row_Base_k1 (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) (s0 : UInt256) (s1_31 : UInt256) (ca_33 : UInt256) (s2_37 : UInt256) (cb_39 : UInt256) (v41 : UInt256) : FeM (Mont_Base × UInt256) := do
  let v42 ← mul_wide b v41
  let l2 := v42.fst
  let h2 := v42.snd
  let v47 ← adc s2_37 l2 ca_33
  let s2_48 := v47.fst
  let ca_50 := v47.snd
  let v52 := t.l3
  let v53 ← adc v52 h2 cb_39
  let s3_54 := v53.fst
  let cb_56 := v53.snd
  let v58 := self.l3
  let v59 ← mul_wide b v58
  let l3 := v59.fst
  let h3 := v59.snd
  let v64 ← adc s3_54 l3 ca_50
  let s3_65 := v64.fst
  let ca_67 := v64.snd
  let v69 ← adc top h3 cb_56
  let s4_70 := v69.fst
  let cb_72 := v69.snd
  let v75 ← adc s4_70 (UInt256.ofNat 0) ca_67
  let s4_76 := v75.fst
  let ca_78 := v75.snd
  let v81 ← adc cb_72 (UInt256.ofNat 0) ca_78
  Mont_row_Base_k2 s0 s1_31 s2_48 s3_65 s4_76 v81

/-- `Mont<Base>::row` -/
def Mont_row_Base (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) : FeM (Mont_Base × UInt256) := do
  let v4 := self.l0
  let v5 ← mul_wide b v4
  let l0 := v5.fst
  let h0 := v5.snd
  let v10 := t.l0
  let v12 ← adc v10 l0 (UInt256.ofNat 0)
  let s0 := v12.fst
  let ca_15 := v12.snd
  let v17 := t.l1
  let v19 ← adc v17 h0 (UInt256.ofNat 0)
  let s1_20 := v19.fst
  let cb_22 := v19.snd
  let v24 := self.l1
  let v25 ← mul_wide b v24
  let l1 := v25.fst
  let h1 := v25.snd
  let v30 ← adc s1_20 l1 ca_15
  let s1_31 := v30.fst
  let ca_33 := v30.snd
  let v35 := t.l2
  let v36 ← adc v35 h1 cb_22
  let s2_37 := v36.fst
  let cb_39 := v36.snd
  let v41 := self.l2
  Mont_row_Base_k1 self b t top s0 s1_31 ca_33 s2_37 cb_39 v41

/-- `lib::montgomery::sbb` -/
def sbb (a : UInt256) (b : UInt256) (borrow : UInt256) : FeM (UInt256 × UInt256) := do
  let v7 ← u128_wrapping_sub a b
  let t ← u128_wrapping_sub v7 borrow
  let v12 ← downcast_truncate_u128_u64 t
  let v14 := t >>> (UInt256.ofNat 127)
  let v15 ← downcast_truncate_u128_u64 v14
  let v16 := (v12, v15)
  return v16

/-- `Mont<Base>::reduce` -/
def Mont_reduce_Base (self : Mont_Base) (top : UInt256) : FeM Mont_Base := do
  let v3 := self.l0
  let v4 ← Base_P0
  let v6 ← sbb v3 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  let b_9 := v6.snd
  let v11 := self.l1
  let v12 ← Base_P1
  let v13 ← sbb v11 v12 b_9
  let d1 := v13.fst
  let b_16 := v13.snd
  let v18 := self.l2
  let v19 ← Base_P2
  let v20 ← sbb v18 v19 b_16
  let d2 := v20.fst
  let b_23 := v20.snd
  let v25 := self.l3
  let v26 ← Base_P3
  let v27 ← sbb v25 v26 b_23
  let d3 := v27.fst
  let b_30 := v27.snd
  let v33 ← sbb top (UInt256.ofNat 0) b_30
  let b_34 := v33.snd
  let v37 := decide (b_34 = (UInt256.ofNat 1))
  let join : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  if v37 then
    let v38 := self
    join v38
  else
    let v40 := (Mont_Base.mk d0 d1 d2 d3)
    join v40

/-- `<Mont<Base> as Field>::mul` -/
def Mont_mul_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let mut t_r0 : Mont_Base := default
  let mut t_r1 : Mont_Base := default
  let mut t_r2 : Mont_Base := default
  let mut t_r3 : Mont_Base := default
  let v2 := other.l0
  let v3 ← Mont_first_row_Base self v2
  let t_4 := v3.fst
  t_r0 := t_4
  let top_6 := v3.snd
  let v8 := other.l1
  let v10 ← Mont_row_Base self v8 t_r0 top_6
  let t_11 := v10.fst
  t_r1 := t_11
  let top_13 := v10.snd
  let v15 := other.l2
  let v17 ← Mont_row_Base self v15 t_r1 top_13
  let t_18 := v17.fst
  t_r2 := t_18
  let top_20 := v17.snd
  let v22 := other.l3
  let v24 ← Mont_row_Base self v22 t_r2 top_20
  let t_25 := v24.fst
  t_r3 := t_25
  let top_27 := v24.snd
  let v30 ← Mont_reduce_Base t_r3 top_27
  return v30

/-- `Mont<Base>::new` -/
def Mont_new_Base (value : UInt256) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  let v2 ← Base_VALUE
  let v3 := decide (value.toNat < v2.toNat)
  let _ ← «assert» v3
  let v5 ← raw_Base value
  r0 := v5
  let v6 ← Base_R2
  let v7 ← raw_Base v6
  r1 := v7
  let v10 ← Mont_mul_Base r0 r1
  return v10

/-- `<Mont<Base> as Field>::square` -/
def Mont_square_Base (self : Mont_Base) : FeM Mont_Base := do
  let v1 ← Mont_mul_Base self self
  return v1

/-- `<Mont<Base> as Field>::add` -/
def Mont_add_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let v2 := self.l0
  let v3 := other.l0
  let v5 ← adc v2 v3 (UInt256.ofNat 0)
  let s0 := v5.fst
  let c_8 := v5.snd
  let v10 := self.l1
  let v11 := other.l1
  let v12 ← adc v10 v11 c_8
  let s1 := v12.fst
  let c_15 := v12.snd
  let v17 := self.l2
  let v18 := other.l2
  let v19 ← adc v17 v18 c_15
  let s2 := v19.fst
  let c_22 := v19.snd
  let v24 := self.l3
  let v25 := other.l3
  let v26 ← adc v24 v25 c_22
  let s3 := v26.fst
  let c_29 := v26.snd
  let v31 := (Mont_Base.mk s0 s1 s2 s3)
  r0 := v31
  let v33 ← Mont_reduce_Base r0 c_29
  return v33

/-- `<Mont<Base> as Field>::one` -/
def Mont_one_Base : FeM Mont_Base := do
  let v1 ← Mont_new_Base (UInt256.ofNat 1)
  return v1

/-- `Mont<Base>::pow` -/
def Mont_pow_Base (self : Mont_Base) (exponent : UInt256) : FeM Mont_Base := do
  let mut result : Mont_Base := default
  let mut bit : UInt256 := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let result_2 ← Mont_one_Base
  result := result_2
  bit := (UInt256.ofNat 256)
  let rec loop (result : Mont_Base) (bit : UInt256) (r2 : Mont_Base) (r3 : Mont_Base) : FeM Mont_Base := do
    let mut result := result
    let mut bit := bit
    let mut r2 := r2
    let mut r3 := r3
    let bit_6 := bit
    let v9 := decide ((UInt256.ofNat 0).toNat < bit_6.toNat)
    if v9 then
      let bit_11 := bit
      let bit_12 ← IntTy.sub (.u 64) bit_11 (UInt256.ofNat 1)
      bit := bit_12
      let result_15 := result
      r2 := result_15
      let result_18 ← Mont_square_Base r2
      result := result_18
      let bit_22 := bit
      let v24 := exponent >>> bit_22
      let v26 := v24 &&& (UInt256.ofNat 1)
      let v28 := decide (v26 = (UInt256.ofNat 1))
      let join : Mont_Base → UInt256 → Mont_Base → Mont_Base → FeM Mont_Base := fun result bit r2 r3 => do
        let mut result := result
        let mut bit := bit
        let mut r2 := r2
        let mut r3 := r3
        loop result bit r2 r3
      if v28 then
        let result_29 := result
        r3 := result_29
        let result_32 ← Mont_mul_Base r3 self
        result := result_32
        join result bit r2 r3
      else
        join result bit r2 r3
    else
      let result_38 := result
      return result_38
  partial_fixpoint
  loop result bit r2 r3

/-- `<Mont<Base> as Field>::equals` -/
def Mont_equals_Base (self : Mont_Base) (other : Mont_Base) : FeM Bool := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := decide (v3 = v4)
  let join : Bool → FeM Bool := fun v2 => do
    return v2
  let join_1 : FeM Bool := do
    join false
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
          join true
        else
          join_1
      else
        join_1
    else
      join_1
  else
    join_1

/-- `Mont<Base>::value` -/
def Mont_value_Base (self : Mont_Base) : FeM UInt256 := do
  let mut r0 : Mont_Base := default
  let v2 ← raw_Base (UInt256.ofNat 1)
  r0 := v2
  let value ← Mont_mul_Base self r0
  let v6 := value.l0
  let v8 := value.l1
  let v11 := v8 <<< (UInt256.ofNat 64)
  let v12 := v6 ||| v11
  let v13 := value.l2
  let v16 := v13 <<< (UInt256.ofNat 128)
  let v17 := v12 ||| v16
  let v18 := value.l3
  let v21 := v18 <<< (UInt256.ofNat 192)
  let v22 := v17 ||| v21
  return v22

/-- `<Order as Modulus>::VALUE` -/
def Order_VALUE : FeM UInt256 := do
  let v0 ← N
  return v0

/-- `lib::montgomery::raw<Order>` -/
def raw_Order (value : UInt256) : FeM Mont_Order := do
  let v1 ← downcast_truncate_u256_u64 value
  let v4 := value >>> (UInt256.ofNat 64)
  let v5 ← downcast_truncate_u256_u64 v4
  let v8 := value >>> (UInt256.ofNat 128)
  let v9 ← downcast_truncate_u256_u64 v8
  let v12 := value >>> (UInt256.ofNat 192)
  let v13 ← downcast_truncate_u256_u64 v12
  let v14 := (Mont_Order.mk v1 v5 v9 v13)
  return v14

/-- `<Order as Modulus>::R2` -/
def Order_R2 : FeM UInt256 := do
  return (UInt256.ofNat 71195301480278335217902614543643724933430614355449737089222010364394701574464)

/-- `<Order as Modulus>::INV` -/
def Order_INV : FeM UInt256 := do
  return (UInt256.ofNat 5408259542528602431)

/-- `<Order as Modulus>::P0` -/
def Order_P0 : FeM UInt256 := do
  return (UInt256.ofNat 13822214165235122497)

/-- `<Order as Modulus>::P1` -/
def Order_P1 : FeM UInt256 := do
  return (UInt256.ofNat 13451932020343611451)

/-- `<Order as Modulus>::P2` -/
def Order_P2 : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551614)

/-- `<Order as Modulus>::P3` -/
def Order_P3 : FeM UInt256 := do
  return (UInt256.ofNat 18446744073709551615)

def Mont_shift_reduced_Order_k1 (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (u1_33 : UInt256) (ca_35 : UInt256) (u2_38 : UInt256) (cb_40 : UInt256) (v43 : (UInt256 × UInt256)) (k2 : UInt256) : FeM (Mont_Order × UInt256) := do
  let j2 := v43.snd
  let v48 ← adc u2_38 k2 ca_35
  let u2_49 := v48.fst
  let ca_51 := v48.snd
  let v53 ← adc s3 j2 cb_40
  let u3_54 := v53.fst
  let cb_56 := v53.snd
  let v58 ← Order_P3
  let v59 ← mul_wide m v58
  let k3 := v59.fst
  let j3 := v59.snd
  let v64 ← adc u3_54 k3 ca_51
  let u3_65 := v64.fst
  let ca_67 := v64.snd
  let v69 ← adc s4 j3 cb_56
  let u4_70 := v69.fst
  let cb_72 := v69.snd
  let v75 ← adc u4_70 (UInt256.ofNat 0) ca_67
  let u4_76 := v75.fst
  let ca_78 := v75.snd
  let v80 ← adc s5 cb_72 ca_78
  let u5 := v80.fst
  let v83 := (Mont_Order.mk u1_33 u2_49 u3_65 u4_76)
  let v84 := (v83, u5)
  return v84

/-- `Mont<Order>::shift_reduced` -/
def Mont_shift_reduced_Order (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v6 ← Order_INV
  let m ← u64_wrapping_mul s0 v6
  let v10 ← Order_P0
  let v11 ← mul_wide m v10
  let k0 := v11.fst
  let j0 := v11.snd
  let v17 ← adc s0 k0 (UInt256.ofNat 0)
  let ca_18 := v17.snd
  let v21 ← adc s1 j0 (UInt256.ofNat 0)
  let u1_22 := v21.fst
  let cb_24 := v21.snd
  let v26 ← Order_P1
  let v27 ← mul_wide m v26
  let k1 := v27.fst
  let j1 := v27.snd
  let v32 ← adc u1_22 k1 ca_18
  let u1_33 := v32.fst
  let ca_35 := v32.snd
  let v37 ← adc s2 j1 cb_24
  let u2_38 := v37.fst
  let cb_40 := v37.snd
  let v42 ← Order_P2
  let v43 ← mul_wide m v42
  let k2 := v43.fst
  Mont_shift_reduced_Order_k1 s3 s4 s5 m u1_33 ca_35 u2_38 cb_40 v43 k2

/-- `Mont<Order>::first_row` -/
def Mont_first_row_Order (self : Mont_Order) (b : UInt256) : FeM (Mont_Order × UInt256) := do
  let v2 := self.l0
  let v3 ← mul_wide b v2
  let s0 := v3.fst
  let h0 := v3.snd
  let v8 := self.l1
  let v9 ← mul_wide b v8
  let l1 := v9.fst
  let h1 := v9.snd
  let v15 ← adc h0 l1 (UInt256.ofNat 0)
  let s1 := v15.fst
  let c_18 := v15.snd
  let v20 := self.l2
  let v21 ← mul_wide b v20
  let l2 := v21.fst
  let h2 := v21.snd
  let v26 ← adc h1 l2 c_18
  let s2 := v26.fst
  let c_29 := v26.snd
  let v31 := self.l3
  let v32 ← mul_wide b v31
  let l3 := v32.fst
  let h3 := v32.snd
  let v37 ← adc h2 l3 c_29
  let s3 := v37.fst
  let c_40 := v37.snd
  let v43 ← adc h3 (UInt256.ofNat 0) c_40
  let s4 := v43.fst
  let v47 ← Mont_shift_reduced_Order s0 s1 s2 s3 s4 (UInt256.ofNat 0)
  return v47

def Mont_row_Order_k2 (s0 : UInt256) (s1_31 : UInt256) (s2_48 : UInt256) (s3_65 : UInt256) (s4_76 : UInt256) (v81 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  let s5 := v81.fst
  let v84 ← Mont_shift_reduced_Order s0 s1_31 s2_48 s3_65 s4_76 s5
  return v84

def Mont_row_Order_k1 (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) (s0 : UInt256) (s1_31 : UInt256) (ca_33 : UInt256) (s2_37 : UInt256) (cb_39 : UInt256) (v41 : UInt256) : FeM (Mont_Order × UInt256) := do
  let v42 ← mul_wide b v41
  let l2 := v42.fst
  let h2 := v42.snd
  let v47 ← adc s2_37 l2 ca_33
  let s2_48 := v47.fst
  let ca_50 := v47.snd
  let v52 := t.l3
  let v53 ← adc v52 h2 cb_39
  let s3_54 := v53.fst
  let cb_56 := v53.snd
  let v58 := self.l3
  let v59 ← mul_wide b v58
  let l3 := v59.fst
  let h3 := v59.snd
  let v64 ← adc s3_54 l3 ca_50
  let s3_65 := v64.fst
  let ca_67 := v64.snd
  let v69 ← adc top h3 cb_56
  let s4_70 := v69.fst
  let cb_72 := v69.snd
  let v75 ← adc s4_70 (UInt256.ofNat 0) ca_67
  let s4_76 := v75.fst
  let ca_78 := v75.snd
  let v81 ← adc cb_72 (UInt256.ofNat 0) ca_78
  Mont_row_Order_k2 s0 s1_31 s2_48 s3_65 s4_76 v81

/-- `Mont<Order>::row` -/
def Mont_row_Order (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) : FeM (Mont_Order × UInt256) := do
  let v4 := self.l0
  let v5 ← mul_wide b v4
  let l0 := v5.fst
  let h0 := v5.snd
  let v10 := t.l0
  let v12 ← adc v10 l0 (UInt256.ofNat 0)
  let s0 := v12.fst
  let ca_15 := v12.snd
  let v17 := t.l1
  let v19 ← adc v17 h0 (UInt256.ofNat 0)
  let s1_20 := v19.fst
  let cb_22 := v19.snd
  let v24 := self.l1
  let v25 ← mul_wide b v24
  let l1 := v25.fst
  let h1 := v25.snd
  let v30 ← adc s1_20 l1 ca_15
  let s1_31 := v30.fst
  let ca_33 := v30.snd
  let v35 := t.l2
  let v36 ← adc v35 h1 cb_22
  let s2_37 := v36.fst
  let cb_39 := v36.snd
  let v41 := self.l2
  Mont_row_Order_k1 self b t top s0 s1_31 ca_33 s2_37 cb_39 v41

/-- `Mont<Order>::reduce` -/
def Mont_reduce_Order (self : Mont_Order) (top : UInt256) : FeM Mont_Order := do
  let v3 := self.l0
  let v4 ← Order_P0
  let v6 ← sbb v3 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  let b_9 := v6.snd
  let v11 := self.l1
  let v12 ← Order_P1
  let v13 ← sbb v11 v12 b_9
  let d1 := v13.fst
  let b_16 := v13.snd
  let v18 := self.l2
  let v19 ← Order_P2
  let v20 ← sbb v18 v19 b_16
  let d2 := v20.fst
  let b_23 := v20.snd
  let v25 := self.l3
  let v26 ← Order_P3
  let v27 ← sbb v25 v26 b_23
  let d3 := v27.fst
  let b_30 := v27.snd
  let v33 ← sbb top (UInt256.ofNat 0) b_30
  let b_34 := v33.snd
  let v37 := decide (b_34 = (UInt256.ofNat 1))
  let join : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  if v37 then
    let v38 := self
    join v38
  else
    let v40 := (Mont_Order.mk d0 d1 d2 d3)
    join v40

/-- `<Mont<Order> as Field>::mul` -/
def Mont_mul_Order (self : Mont_Order) (other : Mont_Order) : FeM Mont_Order := do
  let mut t_r0 : Mont_Order := default
  let mut t_r1 : Mont_Order := default
  let mut t_r2 : Mont_Order := default
  let mut t_r3 : Mont_Order := default
  let v2 := other.l0
  let v3 ← Mont_first_row_Order self v2
  let t_4 := v3.fst
  t_r0 := t_4
  let top_6 := v3.snd
  let v8 := other.l1
  let v10 ← Mont_row_Order self v8 t_r0 top_6
  let t_11 := v10.fst
  t_r1 := t_11
  let top_13 := v10.snd
  let v15 := other.l2
  let v17 ← Mont_row_Order self v15 t_r1 top_13
  let t_18 := v17.fst
  t_r2 := t_18
  let top_20 := v17.snd
  let v22 := other.l3
  let v24 ← Mont_row_Order self v22 t_r2 top_20
  let t_25 := v24.fst
  t_r3 := t_25
  let top_27 := v24.snd
  let v30 ← Mont_reduce_Order t_r3 top_27
  return v30

/-- `Mont<Order>::new` -/
def Mont_new_Order (value : UInt256) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let mut r1 : Mont_Order := default
  let v2 ← Order_VALUE
  let v3 := decide (value.toNat < v2.toNat)
  let _ ← «assert» v3
  let v5 ← raw_Order value
  r0 := v5
  let v6 ← Order_R2
  let v7 ← raw_Order v6
  r1 := v7
  let v10 ← Mont_mul_Order r0 r1
  return v10

/-- `<Order as Modulus>::SIGNED_62` -/
def Order_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v5 := #v[(UInt256.ofNat 4598842128380346689), (UInt256.ofNat 3079181878673178862), (UInt256.ofNat 4611686018427387883), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 255)]
  return v5

/-- `<Order as Modulus>::INV62` -/
def Order_INV62 : FeM UInt256 := do
  return (UInt256.ofNat 3815112494326173377)

/-- `lib::inversion::M62` -/
def M62 : FeM UInt256 := do
  return (UInt256.ofNat 4611686018427387903)

/-- `lib::inversion::is_zero<5>` -/
def is_zero_5 (r : (Vector UInt256 5)) : FeM Bool := do
  let mut i : UInt256 := default
  i := (UInt256.ofNat 0)
  let rec loop (i : UInt256) : FeM Bool := do
    let mut i := i
    let i_3 := i
    let v6 := decide (i_3.toNat < (UInt256.ofNat 5).toNat)
    if v6 then
      let i_7 := i
      checkIndex i_7 5
      let v8 := r[i_7.toNat]!
      let v10 := decide (v8 ≠ (UInt256.ofNat 0))
      if v10 then
        return false
      else
        let i_15 := i
        let i_16 ← checkedAdd i_15 (UInt256.ofNat 1)
        i := i_16
        loop i
    else
      return true
  partial_fixpoint
  loop i

/-- `lib::inversion::divsteps` -/
def divsteps (eta_0 : UInt256) (f0_1 : UInt256) (g0 : UInt256) : FeM (UInt256 × Transition) := do
  let mut u : UInt256 := default
  let mut v : UInt256 := default
  let mut q : UInt256 := default
  let mut r : UInt256 := default
  let mut f : UInt256 := default
  let mut g : UInt256 := default
  let mut eta : UInt256 := default
  let mut left : UInt256 := default
  let mut bits : UInt256 := default
  u := (UInt256.ofNat 1)
  v := (UInt256.ofNat 0)
  q := (UInt256.ofNat 0)
  r := (UInt256.ofNat 1)
  f := f0_1
  g := g0
  eta := eta_0
  left := (UInt256.ofNat 62)
  let rec loop (u : UInt256) (v : UInt256) (q : UInt256) (r : UInt256) (f : UInt256) (g : UInt256) (eta : UInt256) (left : UInt256) (bits : UInt256) : FeM (UInt256 × Transition) := do
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
    let left_24 := left
    let v25 := IntTy.shl (.u 64) (UInt256.ofNat 18446744073709551615) left_24
    let v26 := g_21 ||| v25
    let zeros := IntTy.trailingZeros (.u 64) v26
    let g_29 := g
    let g_31 := g_29 >>> zeros
    g := g_31
    let u_34 := u
    let u_36 := IntTy.shl (.u 64) u_34 zeros
    u := u_36
    let v_39 := v
    let v_41 := IntTy.shl (.u 64) v_39 zeros
    v := v_41
    let v44 ← downcast_truncate_u64_i64 zeros
    let eta_45 := eta
    let eta_46 := IntTy.subW (.i 64) eta_45 v44
    eta := eta_46
    let left_49 := left
    let left_50 := IntTy.subW (.u 64) left_49 zeros
    left := left_50
    let left_53 := left
    let v56 := decide (left_53 = (UInt256.ofNat 0))
    if v56 then
      let eta_57 := eta
      let u_59 := u
      let v61 ← downcast_truncate_u64_i64 u_59
      let v_62 := v
      let v64 ← downcast_truncate_u64_i64 v_62
      let q_65 := q
      let v67 ← downcast_truncate_u64_i64 q_65
      let r_68 := r
      let v70 ← downcast_truncate_u64_i64 r_68
      let v71 := (Transition.mk v61 v64 v67 v70)
      let v72 := (eta_57, v71)
      return v72
    else
      bits := (UInt256.ofNat 15)
      let eta_77 := eta
      let v80 := UInt256.sltBool eta_77 (UInt256.ofNat 0)
      let join : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun u v q r f g eta left bits => do
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
        let v125 := IntTy.addW (.i 64) eta_122 (UInt256.ofNat 1)
        let room ← downcast_truncate_i64_u64 v125
        let left_128 := left
        let v129 := decide (room.toNat < left_128.toNat)
        let join_1 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun u v q r f g eta left bits limit => do
          let mut u := u
          let mut v := v
          let mut q := q
          let mut r := r
          let mut f := f
          let mut g := g
          let mut eta := eta
          let mut left := left
          let mut bits := bits
          let v137 := IntTy.subW (.u 64) (UInt256.ofNat 64) limit
          let v138 := (UInt256.ofNat 18446744073709551615) >>> v137
          let bits_139 := bits
          let mask := v138 &&& bits_139
          let bits_142 := bits
          let v145 := decide (bits_142 = (UInt256.ofNat 63))
          let join_2 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun u v q r f g eta left bits w => do
            let mut u := u
            let mut v := v
            let mut q := q
            let mut r := r
            let mut f := f
            let mut g := g
            let mut eta := eta
            let mut left := left
            let mut bits := bits
            let f_177 := f
            let v179 := IntTy.mulW (.u 64) f_177 w
            let g_180 := g
            let g_181 := IntTy.addW (.u 64) g_180 v179
            g := g_181
            let u_184 := u
            let v186 := IntTy.mulW (.u 64) u_184 w
            let q_187 := q
            let q_188 := IntTy.addW (.u 64) q_187 v186
            q := q_188
            let v_191 := v
            let v193 := IntTy.mulW (.u 64) v_191 w
            let r_194 := r
            let r_195 := IntTy.addW (.u 64) r_194 v193
            r := r_195
            loop u v q r f g eta left bits
          if v145 then
            let f_146 := f
            let g_148 := g
            let v149 := IntTy.mulW (.u 64) f_146 g_148
            let f_150 := f
            let f_152 := f
            let v153 := IntTy.mulW (.u 64) f_150 f_152
            let v155 := IntTy.subW (.u 64) v153 (UInt256.ofNat 2)
            let v156 := IntTy.mulW (.u 64) v149 v155
            let v157 := v156 &&& mask
            join_2 u v q r f g eta left bits v157
          else
            let f_160 := f
            let f_162 := f
            let v165 := IntTy.addW (.u 64) f_162 (UInt256.ofNat 1)
            let v167 := v165 &&& (UInt256.ofNat 4)
            let v169 := IntTy.shl (.u 64) v167 (UInt256.ofNat 1)
            let v170 := IntTy.addW (.u 64) f_160 v169
            let v171 := IntTy.subW (.u 64) (UInt256.ofNat 0) v170
            let g_172 := g
            let v173 := IntTy.mulW (.u 64) v171 g_172
            let v174 := v173 &&& mask
            join_2 u v q r f g eta left bits v174
        if v129 then
          join_1 u v q r f g eta left bits room
        else
          let left_131 := left
          join_1 u v q r f g eta left bits left_131
      if v80 then
        let eta_82 := eta
        let eta_83 := IntTy.subW (.i 64) (UInt256.ofNat 0) eta_82
        eta := eta_83
        let f_86 := f
        let g_88 := g
        f := g_88
        let g_93 := IntTy.subW (.u 64) (UInt256.ofNat 0) f_86
        g := g_93
        let u_96 := u
        let q_98 := q
        u := q_98
        let q_103 := IntTy.subW (.u 64) (UInt256.ofNat 0) u_96
        q := q_103
        let v_106 := v
        let r_108 := r
        v := r_108
        let r_113 := IntTy.subW (.u 64) (UInt256.ofNat 0) v_106
        r := r_113
        bits := (UInt256.ofNat 63)
        join u v q r f g eta left bits
      else
        join u v q r f g eta left bits
  partial_fixpoint
  loop u v q r f g eta left bits

/-- `lib::inversion::M62_SIGNED` -/
def M62_SIGNED : FeM UInt256 := do
  return (UInt256.ofNat 4611686018427387903)

def update_de_5_k3 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (v98 : UInt256) (md_100 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let v102 := IntTy.mulW (.i 128) v98 md_100
  let cd_103 := cd
  let cd_104 := IntTy.addW (.i 128) cd_103 v102
  cd := cd_104
  let v107 := (UInt256.ofNat 0)
  checkIndex v107 5
  let v108 := modulus[v107.toNat]!
  let me_110 := me
  let v112 := IntTy.mulW (.i 128) v108 me_110
  let ce_113 := ce
  let ce_114 := IntTy.addW (.i 128) ce_113 v112
  ce := ce_114
  let cd_117 := cd
  let cd_120 := IntTy.shr (.i 128) cd_117 (UInt256.ofNat 62)
  cd := cd_120
  let ce_123 := ce
  let ce_126 := IntTy.shr (.i 128) ce_123 (UInt256.ofNat 62)
  ce := ce_126
  let next_d_130 := Vector.replicate 5 (UInt256.ofNat 0)
  next_d := next_d_130
  let next_e_133 := Vector.replicate 5 (UInt256.ofNat 0)
  next_e := next_e_133
  i := (UInt256.ofNat 1)
  let rec loop (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut md := md
    let mut me := me
    let mut cd := cd
    let mut ce := ce
    let mut next_d := next_d
    let mut next_e := next_e
    let mut i := i
    let i_137 := i
    let v140 := decide (i_137.toNat < (UInt256.ofNat 5).toNat)
    if v140 then
      let v141 := t.u
      let i_143 := i
      checkIndex i_143 5
      let v144 := d[i_143.toNat]!
      let v146 := IntTy.mulW (.i 128) v141 v144
      let v147 := t.v
      let i_149 := i
      checkIndex i_149 5
      let v150 := e[i_149.toNat]!
      let v152 := IntTy.mulW (.i 128) v147 v150
      let v153 := IntTy.addW (.i 128) v146 v152
      let i_154 := i
      checkIndex i_154 5
      let v155 := modulus[i_154.toNat]!
      let md_157 := md
      let v159 := IntTy.mulW (.i 128) v155 md_157
      let v160 := IntTy.addW (.i 128) v153 v159
      let cd_161 := cd
      let cd_162 := IntTy.addW (.i 128) cd_161 v160
      cd := cd_162
      let v165 := t.q
      let i_167 := i
      checkIndex i_167 5
      let v168 := d[i_167.toNat]!
      let v170 := IntTy.mulW (.i 128) v165 v168
      let v171 := t.r
      let i_173 := i
      checkIndex i_173 5
      let v174 := e[i_173.toNat]!
      let v176 := IntTy.mulW (.i 128) v171 v174
      let v177 := IntTy.addW (.i 128) v170 v176
      let i_178 := i
      checkIndex i_178 5
      let v179 := modulus[i_178.toNat]!
      let me_181 := me
      let v183 := IntTy.mulW (.i 128) v179 me_181
      let v184 := IntTy.addW (.i 128) v177 v183
      let ce_185 := ce
      let ce_186 := IntTy.addW (.i 128) ce_185 v184
      ce := ce_186
      let cd_189 := cd
      let limb_d ← downcast_truncate_i128_i64 cd_189
      let ce_193 := ce
      let limb_e ← downcast_truncate_i128_i64 ce_193
      let v197 ← M62_SIGNED
      let v198 := limb_d &&& v197
      let i_199 := i
      let v202 := i_199 - (UInt256.ofNat 1)
      checkIndex v202 5
      next_d := (next_d.set! v202.toNat v198)
      let v204 ← M62_SIGNED
      let v205 := limb_e &&& v204
      let i_206 := i
      let v209 := i_206 - (UInt256.ofNat 1)
      checkIndex v209 5
      next_e := (next_e.set! v209.toNat v205)
      let cd_211 := cd
      let cd_214 := IntTy.shr (.i 128) cd_211 (UInt256.ofNat 62)
      cd := cd_214
      let ce_217 := ce
      let ce_220 := IntTy.shr (.i 128) ce_217 (UInt256.ofNat 62)
      ce := ce_220
      let i_224 := i
      let i_225 := i_224 + (UInt256.ofNat 1)
      i := i_225
      loop md me cd ce next_d next_e i
    else
      let cd_228 := cd
      let v230 ← downcast_truncate_i128_i64 cd_228
      let v233 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v233 5
      next_d := (next_d.set! v233.toNat v230)
      let ce_235 := ce
      let v237 ← downcast_truncate_i128_i64 ce_235
      let v240 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v240 5
      next_e := (next_e.set! v240.toNat v237)
      let next_d_242 := next_d
      let next_e_244 := next_e
      let v245 := (next_d_242, next_e_244)
      return v245
  partial_fixpoint
  loop md me cd ce next_d next_e i

def update_de_5_k2 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (low_d : UInt256) (low_e : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let md_67 := md
  let md_bits ← downcast_truncate_i64_u64 md_67
  let me_71 := me
  let me_bits ← downcast_truncate_i64_u64 me_71
  let v76 := IntTy.mulW (.u 64) inv62 low_d
  let v77 := IntTy.addW (.u 64) v76 md_bits
  let v78 ← M62
  let v79 := v77 &&& v78
  let correction_d ← downcast_truncate_u64_i64 v79
  let v83 := IntTy.mulW (.u 64) inv62 low_e
  let v84 := IntTy.addW (.u 64) v83 me_bits
  let v85 ← M62
  let v86 := v84 &&& v85
  let correction_e ← downcast_truncate_u64_i64 v86
  let md_89 := md
  let md_90 := IntTy.subW (.i 64) md_89 correction_d
  md := md_90
  let me_93 := me
  let me_94 := IntTy.subW (.i 64) me_93 correction_e
  me := me_94
  let v97 := (UInt256.ofNat 0)
  checkIndex v97 5
  let v98 := modulus[v97.toNat]!
  let md_100 := md
  update_de_5_k3 d e t modulus v98 md_100 md me cd ce next_d next_e i

def update_de_5_k1 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v31 : UInt256) (v34 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let v36 := IntTy.mulW (.i 128) v31 v34
  let v37 := t.v
  let v39 := (UInt256.ofNat 0)
  checkIndex v39 5
  let v40 := e[v39.toNat]!
  let v42 := IntTy.mulW (.i 128) v37 v40
  let cd_43 := IntTy.addW (.i 128) v36 v42
  cd := cd_43
  let v45 := t.q
  let v47 := (UInt256.ofNat 0)
  checkIndex v47 5
  let v48 := d[v47.toNat]!
  let v50 := IntTy.mulW (.i 128) v45 v48
  let v51 := t.r
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let v54 := e[v53.toNat]!
  let v56 := IntTy.mulW (.i 128) v51 v54
  let ce_57 := IntTy.addW (.i 128) v50 v56
  ce := ce_57
  let cd_59 := cd
  let low_d ← downcast_truncate_i128_u64 cd_59
  let ce_63 := ce
  let low_e ← downcast_truncate_i128_u64 ce_63
  update_de_5_k2 d e t modulus inv62 low_d low_e md me cd ce next_d next_e i

/-- `lib::inversion::update_de<5>` -/
def update_de_5 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md : UInt256 := default
  let mut me : UInt256 := default
  let mut cd : UInt256 := default
  let mut ce : UInt256 := default
  let mut next_d : (Vector UInt256 5) := default
  let mut next_e : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let v7 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
  checkIndex v7 5
  let v8 := d[v7.toNat]!
  let sd := IntTy.shr (.i 64) v8 (UInt256.ofNat 63)
  let v14 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
  checkIndex v14 5
  let v15 := e[v14.toNat]!
  let se := IntTy.shr (.i 64) v15 (UInt256.ofNat 63)
  let v19 := t.u
  let v20 := v19 &&& sd
  let v21 := t.v
  let v22 := v21 &&& se
  let md_23 := IntTy.addW (.i 64) v20 v22
  md := md_23
  let v25 := t.q
  let v26 := v25 &&& sd
  let v27 := t.r
  let v28 := v27 &&& se
  let me_29 := IntTy.addW (.i 64) v26 v28
  me := me_29
  let v31 := t.u
  let v33 := (UInt256.ofNat 0)
  checkIndex v33 5
  let v34 := d[v33.toNat]!
  update_de_5_k1 d e t modulus inv62 v31 v34 md me cd ce next_d next_e i

/-- `lib::inversion::update_fg<5>` -/
def update_fg_5 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf : UInt256 := default
  let mut cg : UInt256 := default
  let mut next_f : (Vector UInt256 5) := default
  let mut next_g : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let v3 := t.u
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 5
  let v6 := f[v5.toNat]!
  let v8 := IntTy.mulW (.i 128) v3 v6
  let v9 := t.v
  let v11 := (UInt256.ofNat 0)
  checkIndex v11 5
  let v12 := g[v11.toNat]!
  let v14 := IntTy.mulW (.i 128) v9 v12
  let cf_15 := IntTy.addW (.i 128) v8 v14
  cf := cf_15
  let v17 := t.q
  let v19 := (UInt256.ofNat 0)
  checkIndex v19 5
  let v20 := f[v19.toNat]!
  let v22 := IntTy.mulW (.i 128) v17 v20
  let v23 := t.r
  let v25 := (UInt256.ofNat 0)
  checkIndex v25 5
  let v26 := g[v25.toNat]!
  let v28 := IntTy.mulW (.i 128) v23 v26
  let cg_29 := IntTy.addW (.i 128) v22 v28
  cg := cg_29
  let cf_31 := cf
  let cf_34 := IntTy.shr (.i 128) cf_31 (UInt256.ofNat 62)
  cf := cf_34
  let cg_37 := cg
  let cg_40 := IntTy.shr (.i 128) cg_37 (UInt256.ofNat 62)
  cg := cg_40
  let next_f_44 := Vector.replicate 5 (UInt256.ofNat 0)
  next_f := next_f_44
  let next_g_47 := Vector.replicate 5 (UInt256.ofNat 0)
  next_g := next_g_47
  i := (UInt256.ofNat 1)
  let rec loop (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut cf := cf
    let mut cg := cg
    let mut next_f := next_f
    let mut next_g := next_g
    let mut i := i
    let i_51 := i
    let v54 := decide (i_51.toNat < (UInt256.ofNat 5).toNat)
    if v54 then
      let v55 := t.u
      let i_57 := i
      checkIndex i_57 5
      let v58 := f[i_57.toNat]!
      let v60 := IntTy.mulW (.i 128) v55 v58
      let v61 := t.v
      let i_63 := i
      checkIndex i_63 5
      let v64 := g[i_63.toNat]!
      let v66 := IntTy.mulW (.i 128) v61 v64
      let v67 := IntTy.addW (.i 128) v60 v66
      let cf_68 := cf
      let cf_69 := IntTy.addW (.i 128) cf_68 v67
      cf := cf_69
      let v72 := t.q
      let i_74 := i
      checkIndex i_74 5
      let v75 := f[i_74.toNat]!
      let v77 := IntTy.mulW (.i 128) v72 v75
      let v78 := t.r
      let i_80 := i
      checkIndex i_80 5
      let v81 := g[i_80.toNat]!
      let v83 := IntTy.mulW (.i 128) v78 v81
      let v84 := IntTy.addW (.i 128) v77 v83
      let cg_85 := cg
      let cg_86 := IntTy.addW (.i 128) cg_85 v84
      cg := cg_86
      let cf_89 := cf
      let limb_f ← downcast_truncate_i128_i64 cf_89
      let cg_93 := cg
      let limb_g ← downcast_truncate_i128_i64 cg_93
      let v97 ← M62_SIGNED
      let v98 := limb_f &&& v97
      let i_99 := i
      let v102 := i_99 - (UInt256.ofNat 1)
      checkIndex v102 5
      next_f := (next_f.set! v102.toNat v98)
      let v104 ← M62_SIGNED
      let v105 := limb_g &&& v104
      let i_106 := i
      let v109 := i_106 - (UInt256.ofNat 1)
      checkIndex v109 5
      next_g := (next_g.set! v109.toNat v105)
      let cf_111 := cf
      let cf_114 := IntTy.shr (.i 128) cf_111 (UInt256.ofNat 62)
      cf := cf_114
      let cg_117 := cg
      let cg_120 := IntTy.shr (.i 128) cg_117 (UInt256.ofNat 62)
      cg := cg_120
      let i_124 := i
      let i_125 := i_124 + (UInt256.ofNat 1)
      i := i_125
      loop cf cg next_f next_g i
    else
      let cf_128 := cf
      let v130 ← downcast_truncate_i128_i64 cf_128
      let v133 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v133 5
      next_f := (next_f.set! v133.toNat v130)
      let cg_135 := cg
      let v137 ← downcast_truncate_i128_i64 cg_135
      let v140 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v140 5
      next_g := (next_g.set! v140.toNat v137)
      let next_f_142 := next_f
      let next_g_144 := next_g
      let v145 := (next_f_142, next_g_144)
      return v145
  partial_fixpoint
  loop cf cg next_f next_g i

/-- `lib::inversion::add_if_negative<5>` -/
def add_if_negative_5 (r_0 : (Vector UInt256 5)) (modulus : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let r_2 := r_0
  r := r_2
  let v6 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
  checkIndex v6 5
  let v7 := r[v6.toNat]!
  let negative := IntTy.shr (.i 64) v7 (UInt256.ofNat 63)
  i := (UInt256.ofNat 0)
  let rec loop (r : (Vector UInt256 5)) (i : UInt256) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let i_13 := i
    let v16 := decide (i_13.toNat < (UInt256.ofNat 5).toNat)
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
      let i_26 := i
      let i_27 := i_26 + (UInt256.ofNat 1)
      i := i_27
      loop r i
    else
      let r_30 := r
      return r_30
  partial_fixpoint
  loop r i

/-- `lib::inversion::carry<5>` -/
def carry_5 (r_0 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let r_1 := r_0
  r := r_1
  i := (UInt256.ofNat 0)
  let rec loop (r : (Vector UInt256 5)) (i : UInt256) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let i_5 := i
    let v8 := i_5 + (UInt256.ofNat 1)
    let v10 := decide (v8.toNat < (UInt256.ofNat 5).toNat)
    if v10 then
      let i_11 := i
      checkIndex i_11 5
      let v12 := r[i_11.toNat]!
      let v14 := IntTy.shr (.i 64) v12 (UInt256.ofNat 62)
      let i_15 := i
      let v18 := i_15 + (UInt256.ofNat 1)
      checkIndex v18 5
      let v19 := r[v18.toNat]!
      let v20 := IntTy.addW (.i 64) v19 v14
      checkIndex v18 5
      r := (r.set! v18.toNat v20)
      let v22 ← M62_SIGNED
      let i_23 := i
      checkIndex i_23 5
      let v24 := r[i_23.toNat]!
      let v25 := v24 &&& v22
      let i_26 := i
      checkIndex i_26 5
      r := (r.set! i_26.toNat v25)
      let i_29 := i
      let i_30 := i_29 + (UInt256.ofNat 1)
      i := i_30
      loop r i
    else
      let r_33 := r
      return r_33
  partial_fixpoint
  loop r i

/-- `lib::inversion::normalize<5>` -/
def normalize_5 (r_0 : (Vector UInt256 5)) (sign : UInt256) (modulus : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let mut r2 : (Vector UInt256 5) := default
  let mut r3 : (Vector UInt256 5) := default
  let mut r4 : (Vector UInt256 5) := default
  let r_3 ← add_if_negative_5 r_0 modulus
  r := r_3
  let negate := IntTy.shr (.i 64) sign (UInt256.ofNat 63)
  i := (UInt256.ofNat 0)
  let rec loop (r : (Vector UInt256 5)) (i : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let i_11 := i
    let v14 := decide (i_11.toNat < (UInt256.ofNat 5).toNat)
    if v14 then
      let i_15 := i
      checkIndex i_15 5
      let v16 := r[i_15.toNat]!
      let v17 := v16 ^^^ negate
      let v18 := IntTy.subW (.i 64) v17 negate
      let i_19 := i
      checkIndex i_19 5
      r := (r.set! i_19.toNat v18)
      let i_22 := i
      let i_23 := i_22 + (UInt256.ofNat 1)
      i := i_23
      loop r i r2 r3 r4
    else
      let r_26 := r
      r2 := r_26
      let v29 ← carry_5 r2
      r3 := v29
      let v31 ← add_if_negative_5 r3 modulus
      r4 := v31
      let v33 ← carry_5 r4
      return v33
  partial_fixpoint
  loop r i r2 r3 r4

/-- `lib::inversion::invert<5>` -/
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
  let d_4 := Vector.replicate 5 (UInt256.ofNat 0)
  d := d_4
  let e_7 := Vector.replicate 5 (UInt256.ofNat 0)
  e := e_7
  let v10 := (UInt256.ofNat 0)
  checkIndex v10 5
  e := (e.set! v10.toNat (UInt256.ofNat 1))
  let f_12 := modulus
  f := f_12
  let g_14 := x
  g := g_14
  eta := (IntTy.wrap (.i 256) (-1))
  let rec loop (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (eta : UInt256) (t : Transition) (r6 : (Vector UInt256 5)) (r7 : (Vector UInt256 5)) (r8 : (Vector UInt256 5)) (r9 : (Vector UInt256 5)) (r10 : (Vector UInt256 5)) (r11 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
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
    r6 := g_18
    let v21 ← is_zero_5 r6
    let v22 := !v21
    if v22 then
      let eta_23 := eta
      let v25 := (UInt256.ofNat 0)
      checkIndex v25 5
      let v26 := f[v25.toNat]!
      let v27 ← downcast_truncate_i64_u64 v26
      let v28 := (UInt256.ofNat 0)
      checkIndex v28 5
      let v29 := g[v28.toNat]!
      let v30 ← downcast_truncate_i64_u64 v29
      let v31 ← divsteps eta_23 v27 v30
      let next := v31.fst
      let t_34 := v31.snd
      t := t_34
      eta := next
      let d_38 := d
      r7 := d_38
      let e_40 := e
      r8 := e_40
      let v45 ← update_de_5 r7 r8 t modulus inv62
      let next_d := v45.fst
      let next_e := v45.snd
      d := next_d
      e := next_e
      let f_54 := f
      r9 := f_54
      let g_56 := g
      r10 := g_56
      let v61 ← update_fg_5 r9 r10 t
      let next_f := v61.fst
      let next_g := v61.snd
      f := next_f
      g := next_g
      loop d e f g eta t r6 r7 r8 r9 r10 r11
    else
      let d_70 := d
      r11 := d_70
      let v74 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v74 5
      let v75 := f[v74.toNat]!
      let v77 ← normalize_5 r11 v75 modulus
      return v77
  partial_fixpoint
  loop d e f g eta t r6 r7 r8 r9 r10 r11

def invert_256_k4 (w : (Vector UInt256 5)) (v76 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v94 := v89 ||| v93
  let v95 := (UInt256.ofNat 3)
  checkIndex v95 5
  let v96 := w[v95.toNat]!
  let v98 := v96 >>> (UInt256.ofNat 6)
  let v99 := (UInt256.ofNat 4)
  checkIndex v99 5
  let v100 := w[v99.toNat]!
  let v102 := IntTy.shl (.u 64) v100 (UInt256.ofNat 56)
  let v103 := v98 ||| v102
  let v104 := #v[v76, v85, v94, v103]
  return v104

def invert_256_k3 (w : (Vector UInt256 5)) (v70 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  checkIndex v70 5
  let v71 := w[v70.toNat]!
  let v72 := (UInt256.ofNat 1)
  checkIndex v72 5
  let v73 := w[v72.toNat]!
  let v75 := IntTy.shl (.u 64) v73 (UInt256.ofNat 62)
  let v76 := v71 ||| v75
  let v77 := (UInt256.ofNat 1)
  checkIndex v77 5
  let v78 := w[v77.toNat]!
  let v80 := v78 >>> (UInt256.ofNat 2)
  let v81 := (UInt256.ofNat 2)
  checkIndex v81 5
  let v82 := w[v81.toNat]!
  let v84 := IntTy.shl (.u 64) v82 (UInt256.ofNat 60)
  let v85 := v80 ||| v84
  let v86 := (UInt256.ofNat 2)
  checkIndex v86 5
  let v87 := w[v86.toNat]!
  let v89 := v87 >>> (UInt256.ofNat 4)
  let v90 := (UInt256.ofNat 3)
  checkIndex v90 5
  let v91 := w[v90.toNat]!
  let v93 := IntTy.shl (.u 64) v91 (UInt256.ofNat 58)
  invert_256_k4 w v76 v85 v89 v93 r0

def invert_256_k2 (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v49 : (Vector UInt256 5)) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  r0 := v49
  let s ← invert_5 r0 modulus inv62
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
  let v63 := s[v62.toNat]!
  let v64 ← downcast_truncate_i64_u64 v63
  let v65 := (UInt256.ofNat 4)
  checkIndex v65 5
  let v66 := s[v65.toNat]!
  let v67 ← downcast_truncate_i64_u64 v66
  let w := #v[v55, v58, v61, v64, v67]
  let v70 := (UInt256.ofNat 0)
  invert_256_k3 w v70 r0

def invert_256_k1 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v7 : UInt256) (v19 : UInt256) (v23 : UInt256) (v24 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v25 := x[v24.toNat]!
  let v27 := IntTy.shl (.u 64) v25 (UInt256.ofNat 4)
  let v28 := v23 ||| v27
  let v29 ← M62
  let v30 := v28 &&& v29
  let v31 ← downcast_truncate_u64_i64 v30
  let v32 := (UInt256.ofNat 2)
  checkIndex v32 4
  let v33 := x[v32.toNat]!
  let v35 := v33 >>> (UInt256.ofNat 58)
  let v36 := (UInt256.ofNat 3)
  checkIndex v36 4
  let v37 := x[v36.toNat]!
  let v39 := IntTy.shl (.u 64) v37 (UInt256.ofNat 6)
  let v40 := v35 ||| v39
  let v41 ← M62
  let v42 := v40 &&& v41
  let v43 ← downcast_truncate_u64_i64 v42
  let v44 := (UInt256.ofNat 3)
  checkIndex v44 4
  let v45 := x[v44.toNat]!
  let v47 := v45 >>> (UInt256.ofNat 56)
  let v48 ← downcast_truncate_u64_i64 v47
  let v49 := #v[v7, v19, v31, v43, v48]
  invert_256_k2 modulus inv62 v49 r0

/-- `lib::inversion::invert_256` -/
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
  let v11 := v9 >>> (UInt256.ofNat 62)
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 4
  let v13 := x[v12.toNat]!
  let v15 := IntTy.shl (.u 64) v13 (UInt256.ofNat 2)
  let v16 := v11 ||| v15
  let v17 ← M62
  let v18 := v16 &&& v17
  let v19 ← downcast_truncate_u64_i64 v18
  let v20 := (UInt256.ofNat 1)
  checkIndex v20 4
  let v21 := x[v20.toNat]!
  let v23 := v21 >>> (UInt256.ofNat 60)
  let v24 := (UInt256.ofNat 2)
  checkIndex v24 4
  invert_256_k1 x modulus inv62 v7 v19 v23 v24 r0

/-- `<Order as Modulus>::R3` -/
def Order_R3 : FeM UInt256 := do
  return (UInt256.ofNat 38611792656279623856209604949727965891812033543773964974832218758873530450413)

/-- `<Mont<Order> as Field>::inverse` -/
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
  let limbs ← invert_256 r0 r1 v7
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

/-- `<Mont<Order> as Field>::zero` -/
def Mont_zero_Order : FeM Mont_Order := do
  let v4 := (Mont_Order.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  return v4

/-- `<Mont<Order> as Field>::sub` -/
def Mont_sub_Order (self : Mont_Order) (other : Mont_Order) : FeM Mont_Order := do
  let v3 := self.l0
  let v4 := other.l0
  let v6 ← sbb v3 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  let b_9 := v6.snd
  let v11 := self.l1
  let v12 := other.l1
  let v13 ← sbb v11 v12 b_9
  let d1 := v13.fst
  let b_16 := v13.snd
  let v18 := self.l2
  let v19 := other.l2
  let v20 ← sbb v18 v19 b_16
  let d2 := v20.fst
  let b_23 := v20.snd
  let v25 := self.l3
  let v26 := other.l3
  let v27 ← sbb v25 v26 b_23
  let d3 := v27.fst
  let b_30 := v27.snd
  let v32 ← Order_P0
  let v34 ← adc d0 v32 (UInt256.ofNat 0)
  let s0 := v34.fst
  let c_37 := v34.snd
  let v39 ← Order_P1
  let v40 ← adc d1 v39 c_37
  let s1 := v40.fst
  let c_43 := v40.snd
  let v45 ← Order_P2
  let v46 ← adc d2 v45 c_43
  let s2 := v46.fst
  let c_49 := v46.snd
  let v51 ← Order_P3
  let v52 ← adc d3 v51 c_49
  let s3 := v52.fst
  let v56 := decide (b_30 = (UInt256.ofNat 1))
  let join : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Order.mk s0 s1 s2 s3)
    join v57
  else
    let v59 := (Mont_Order.mk d0 d1 d2 d3)
    join v59

/-- `<Mont<Order> as Field>::neg` -/
def Mont_neg_Order (self : Mont_Order) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let v1 ← Mont_zero_Order
  r0 := v1
  let v3 ← Mont_sub_Order r0 self
  return v3

/-- `Mont<Order>::value` -/
def Mont_value_Order (self : Mont_Order) : FeM UInt256 := do
  let mut r0 : Mont_Order := default
  let v2 ← raw_Order (UInt256.ofNat 1)
  r0 := v2
  let value ← Mont_mul_Order self r0
  let v6 := value.l0
  let v8 := value.l1
  let v11 := v8 <<< (UInt256.ofNat 64)
  let v12 := v6 ||| v11
  let v13 := value.l2
  let v16 := v13 <<< (UInt256.ofNat 128)
  let v17 := v12 ||| v16
  let v18 := value.l3
  let v21 := v18 <<< (UInt256.ofNat 192)
  let v22 := v17 ||| v21
  return v22

/-- `Point<Mont<Base>>::affine` -/
def Point_affine_Mont_Base (x : Mont_Base) (y : Mont_Base) : FeM Point_Mont_Base := do
  let v2 := x
  let v3 := y
  let v4 ← Mont_one_Base
  let v5 := (Point_Mont_Base.mk v2 v3 v4)
  return v5

/-- `lib::secp256k1::GX` -/
def GX : FeM UInt256 := do
  return (UInt256.ofNat 55066263022277343669578718895168534326250603453777594175500187360389116729240)

/-- `lib::secp256k1::GY` -/
def GY : FeM UInt256 := do
  return (UInt256.ofNat 32670510020758816978083085130507043184471273380659243275938904335757337482424)

/-- `lib::secp256k1::generator` -/
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

/-- `<Mont<Base> as Field>::zero` -/
def Mont_zero_Base : FeM Mont_Base := do
  let v4 := (Mont_Base.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  return v4

/-- `Point<Mont<Base>>::infinity` -/
def Point_infinity_Mont_Base : FeM Point_Mont_Base := do
  let v0 ← Mont_one_Base
  let v1 ← Mont_one_Base
  let v2 ← Mont_zero_Base
  let v3 := (Point_Mont_Base.mk v0 v1 v2)
  return v3

/-- `<Mont<Base> as Field>::is_zero` -/
def Mont_is_zero_Base (self : Mont_Base) : FeM Bool := do
  let v2 := self.l0
  let v4 := decide (v2 = (UInt256.ofNat 0))
  let join : Bool → FeM Bool := fun v1 => do
    return v1
  let join_1 : FeM Bool := do
    join false
  if v4 then
    let v5 := self.l1
    let v7 := decide (v5 = (UInt256.ofNat 0))
    if v7 then
      let v8 := self.l2
      let v10 := decide (v8 = (UInt256.ofNat 0))
      if v10 then
        let v11 := self.l3
        let v13 := decide (v11 = (UInt256.ofNat 0))
        if v13 then
          join true
        else
          join_1
      else
        join_1
    else
      join_1
  else
    join_1

/-- `Point<Mont<Base>>::is_infinity` -/
def Point_is_infinity_Mont_Base (self : Point_Mont_Base) : FeM Bool := do
  let mut r0 : Mont_Base := default
  let v1 := self.z
  r0 := v1
  let v3 ← Mont_is_zero_Base r0
  return v3

/-- `<Mont<Base> as Field>::sub` -/
def Mont_sub_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let v3 := self.l0
  let v4 := other.l0
  let v6 ← sbb v3 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  let b_9 := v6.snd
  let v11 := self.l1
  let v12 := other.l1
  let v13 ← sbb v11 v12 b_9
  let d1 := v13.fst
  let b_16 := v13.snd
  let v18 := self.l2
  let v19 := other.l2
  let v20 ← sbb v18 v19 b_16
  let d2 := v20.fst
  let b_23 := v20.snd
  let v25 := self.l3
  let v26 := other.l3
  let v27 ← sbb v25 v26 b_23
  let d3 := v27.fst
  let b_30 := v27.snd
  let v32 ← Base_P0
  let v34 ← adc d0 v32 (UInt256.ofNat 0)
  let s0 := v34.fst
  let c_37 := v34.snd
  let v39 ← Base_P1
  let v40 ← adc d1 v39 c_37
  let s1 := v40.fst
  let c_43 := v40.snd
  let v45 ← Base_P2
  let v46 ← adc d2 v45 c_43
  let s2 := v46.fst
  let c_49 := v46.snd
  let v51 ← Base_P3
  let v52 ← adc d3 v51 c_49
  let s3 := v52.fst
  let v56 := decide (b_30 = (UInt256.ofNat 1))
  let join : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Base.mk s0 s1 s2 s3)
    join v57
  else
    let v59 := (Mont_Base.mk d0 d1 d2 d3)
    join v59

/-- `<Mont<Base> as Field>::double` -/
def Mont_double_Base (self : Mont_Base) : FeM Mont_Base := do
  let v1 ← Mont_add_Base self self
  return v1

/-- `Point<Mont<Base>>::double` -/
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
  let join : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → FeM Point_Mont_Base := fun a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 => do
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
    join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    let v2 := self.y
    r6 := v2
    let v4 ← Mont_is_zero_Base r6
    if v4 then
      join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      let v7 := self.x
      r7 := v7
      let a_9 ← Mont_square_Base r7
      a := a_9
      let v11 := self.y
      r8 := v11
      let b_13 ← Mont_square_Base r8
      b := b_13
      let c_16 ← Mont_square_Base b
      c := c_16
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
      let d_31 ← Mont_double_Base r13
      d := d_31
      let v34 ← Mont_double_Base a
      r14 := v34
      let e_37 ← Mont_add_Base r14 a
      e := e_37
      let v40 ← Mont_square_Base e
      r15 := v40
      let v42 ← Mont_double_Base d
      r16 := v42
      let x_45 ← Mont_sub_Base r15 r16
      x := x_45
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
      let y ← Mont_sub_Base r18 r21
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

/-- `Point<Mont<Base>>::add` -/
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
    let v7 ← Point_is_infinity_Mont_Base other
    if v7 then
      let v8 := self
      return v8
    else
      let v11 := self.z
      r12 := v11
      let z1z1_13 ← Mont_square_Base r12
      z1z1 := z1z1_13
      let v15 := other.z
      r13 := v15
      let z2z2_17 ← Mont_square_Base r13
      z2z2 := z2z2_17
      let v19 := self.x
      r14 := v19
      let u1_22 ← Mont_mul_Base r14 z2z2
      u1 := u1_22
      let v24 := other.x
      r15 := v24
      let u2_27 ← Mont_mul_Base r15 z1z1
      u2 := u2_27
      let v29 := other.z
      r16 := v29
      let v30 := self.y
      r17 := v30
      let v33 ← Mont_mul_Base r17 r16
      r18 := v33
      let s1_36 ← Mont_mul_Base r18 z2z2
      s1 := s1_36
      let v38 := self.z
      r19 := v38
      let v39 := other.y
      r20 := v39
      let v42 ← Mont_mul_Base r20 r19
      r21 := v42
      let s2_45 ← Mont_mul_Base r21 z1z1
      s2 := s2_45
      let h_49 ← Mont_sub_Base u2 u1
      h := h_49
      let v53 ← Mont_sub_Base s2 s1
      r22 := v53
      let r_55 ← Mont_double_Base r22
      r := r_55
      let v58 ← Mont_is_zero_Base h
      if v58 then
        let v60 ← Mont_is_zero_Base r
        let join : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v2 => do
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
          join z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v61
        else
          let v63 ← Point_infinity_Mont_Base
          join z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v63
      else
        let v68 ← Mont_double_Base h
        r23 := v68
        let i_70 ← Mont_square_Base r23
        i := i_70
        let j_74 ← Mont_mul_Base h i
        j := j_74
        let v_78 ← Mont_mul_Base u1 i
        v := v_78
        let v81 ← Mont_square_Base r
        r24 := v81
        let v84 ← Mont_sub_Base r24 j
        r25 := v84
        let v86 ← Mont_double_Base v
        r26 := v86
        let x_89 ← Mont_sub_Base r25 r26
        x := x_89
        let v93 ← Mont_sub_Base v x
        r27 := v93
        let v96 ← Mont_mul_Base r r27
        r28 := v96
        let v99 ← Mont_mul_Base s1 j
        r29 := v99
        let v101 ← Mont_double_Base r29
        r30 := v101
        let y ← Mont_sub_Base r28 r30
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

/-- `<Base as Modulus>::SIGNED_62` -/
def Base_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v5 := #v[(UInt256.ofNat 4611686014132419631), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 255)]
  return v5

/-- `<Base as Modulus>::INV62` -/
def Base_INV62 : FeM UInt256 := do
  return (UInt256.ofNat 2866531139136965327)

/-- `<Base as Modulus>::R3` -/
def Base_R3 : FeM UInt256 := do
  return (UInt256.ofNat 79228216581683516649686562417)

/-- `<Mont<Base> as Field>::inverse` -/
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
  let limbs ← invert_256 r0 r1 v7
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

/-- `Point<Mont<Base>>::add_affine` -/
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
    let v8 := self.z
    r10 := v8
    let z1z1_10 ← Mont_square_Base r10
    z1z1 := z1z1_10
    let u2_13 ← Mont_mul_Base x2 z1z1
    u2 := u2_13
    let v15 := self.z
    r11 := v15
    let v17 ← Mont_mul_Base y2 r11
    r12 := v17
    let s2_20 ← Mont_mul_Base r12 z1z1
    s2 := s2_20
    let v22 := self.x
    r13 := v22
    let h_25 ← Mont_sub_Base u2 r13
    h := h_25
    let v27 := self.y
    r14 := v27
    let v30 ← Mont_sub_Base s2 r14
    r15 := v30
    let r_32 ← Mont_double_Base r15
    r := r_32
    let v35 ← Mont_is_zero_Base h
    if v35 then
      let v37 ← Mont_is_zero_Base r
      let join : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v3 => do
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
        join z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v38
      else
        let v40 ← Point_infinity_Mont_Base
        join z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v40
    else
      let hh_45 ← Mont_square_Base h
      hh := hh_45
      let v48 ← Mont_double_Base hh
      r16 := v48
      let i_50 ← Mont_double_Base r16
      i := i_50
      let j_54 ← Mont_mul_Base h i
      j := j_54
      let v56 := self.x
      r17 := v56
      let v_59 ← Mont_mul_Base r17 i
      v := v_59
      let v62 ← Mont_square_Base r
      r18 := v62
      let v65 ← Mont_sub_Base r18 j
      r19 := v65
      let v67 ← Mont_double_Base v
      r20 := v67
      let x_70 ← Mont_sub_Base r19 r20
      x := x_70
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
      let y ← Mont_sub_Base r22 r25
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

/-- `lib::secp256k1::combination` -/
def combination (u : UInt256) (v_1 : UInt256) (q : Point_Mont_Base) : FeM Point_Mont_Base := do
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
  let g_6 ← generator
  g := g_6
  let v8 ← Point_infinity_Mont_Base
  let table_9 := Vector.replicate 16 v8
  table := table_9
  let v11 := q
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 16
  table := (table.set! v12.toNat v11)
  let v14 ← Point_double_Mont_Base q
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 16
  table := (table.set! v15.toNat v14)
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  let v18 := table[v17.toNat]!
  r10 := v18
  let v20 ← Point_add_Mont_Base r10 q
  let v21 := (UInt256.ofNat 3)
  checkIndex v21 16
  table := (table.set! v21.toNat v20)
  let v23 := (UInt256.ofNat 4)
  let g_24 := g
  checkIndex v23 16
  table := (table.set! v23.toNat g_24)
  let v27 ← Point_double_Mont_Base g
  let v28 := (UInt256.ofNat 8)
  checkIndex v28 16
  table := (table.set! v28.toNat v27)
  let v30 := (UInt256.ofNat 8)
  checkIndex v30 16
  let v31 := table[v30.toNat]!
  r11 := v31
  let v34 ← Point_add_Mont_Base r11 g
  let v35 := (UInt256.ofNat 12)
  checkIndex v35 16
  table := (table.set! v35.toNat v34)
  let v41 := #v[(UInt256.ofNat 4), (UInt256.ofNat 8), (UInt256.ofNat 12)]
  r12 := v41
  let v43 ← array_start_usize_3 r12
  let rec loop (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v3 : UInt256) : FeM Point_Mont_Base := do
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
      let v53 := (Range_Known_1_Known_4.mk () ())
      r13 := v53
      let v55 ← Range_start_1_4 r13
      let rec loop_1 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v4 : UInt256) : FeM Point_Mont_Base := do
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
          let v70 ← Range_next_1_4 r13 v4
          loop_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v70
        else
          let v74 ← array_next_usize_3 r12 v3
          loop g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v74
      partial_fixpoint
      loop_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v55
    else
      let v76 ← Mont_one_Base
      let prefix_77 := Vector.replicate 16 v76
      «prefix» := prefix_77
      let running_79 ← Mont_one_Base
      running := running_79
      let v84 := (Range_Known_1_Known_16.mk () ())
      r16 := v84
      let v86 ← Range_start_1_16 r16
      let rec loop_2 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) (v5 : UInt256) : FeM Point_Mont_Base := do
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
          checkIndex k_92 16
          «prefix» := («prefix».set! k_92.toNat running_93)
          checkIndex k_92 16
          let v96 := table[k_92.toNat]!
          r17 := v96
          let v98 ← Point_is_infinity_Mont_Base r17
          let v99 := !v98
          let join : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
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
            loop_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v112
          if v99 then
            checkIndex k_92 16
            let v100 := table[k_92.toNat]!.z
            r18 := v100
            let running_101 := running
            r19 := running_101
            let running_105 ← Mont_mul_Base r19 r18
            running := running_105
            join g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          else
            join g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
        else
          let running_114 := running
          r20 := running_114
          let inverse_117 ← Mont_inverse_Base r20
          inverse := inverse_117
          k := (UInt256.ofNat 16)
          let rec loop_3 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
            let v124 := decide ((UInt256.ofNat 1).toNat < k_121.toNat)
            if v124 then
              let k_126 := k
              let k_127 ← checkedSub k_126 (UInt256.ofNat 1)
              k := k_127
              let k_130 := k
              checkIndex k_130 16
              let v131 := table[k_130.toNat]!
              r21 := v131
              let v134 ← Point_is_infinity_Mont_Base r21
              if v134 then
                loop_3 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
              else
                let k_137 := k
                checkIndex k_137 16
                let v138 := «prefix»[k_137.toNat]!
                r22 := v138
                let inverse_139 := inverse
                r23 := inverse_139
                let z_inverse_144 ← Mont_mul_Base r23 r22
                z_inverse := z_inverse_144
                let k_146 := k
                checkIndex k_146 16
                let v147 := table[k_146.toNat]!.z
                r24 := v147
                let inverse_148 := inverse
                r25 := inverse_148
                let inverse_153 ← Mont_mul_Base r25 r24
                inverse := inverse_153
                let z_inverse2_157 ← Mont_square_Base z_inverse
                z_inverse2 := z_inverse2_157
                let k_159 := k
                checkIndex k_159 16
                let v160 := table[k_159.toNat]!.x
                r26 := v160
                let v164 ← Mont_mul_Base r26 z_inverse2
                r27 := v164
                let v167 ← Mont_mul_Base z_inverse2 z_inverse
                r28 := v167
                let k_168 := k
                checkIndex k_168 16
                let v169 := table[k_168.toNat]!.y
                r29 := v169
                let v173 ← Mont_mul_Base r29 r28
                r30 := v173
                let v176 ← Point_affine_Mont_Base r27 r30
                let k_177 := k
                checkIndex k_177 16
                table := (table.set! k_177.toNat v176)
                loop_3 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
            else
              let result_179 ← Point_infinity_Mont_Base
              result := result_179
              bit := (UInt256.ofNat 256)
              let rec loop_4 (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
                let v186 := decide ((UInt256.ofNat 0).toNat < bit_183.toNat)
                if v186 then
                  let bit_188 := bit
                  let bit_189 ← IntTy.sub (.u 64) bit_188 (UInt256.ofNat 2)
                  bit := bit_189
                  let result_192 := result
                  r31 := result_192
                  let v195 ← Point_double_Mont_Base r31
                  r32 := v195
                  let result_197 ← Point_double_Mont_Base r32
                  result := result_197
                  let bit_201 := bit
                  let v203 := u >>> bit_201
                  let v205 := v203 &&& (UInt256.ofNat 3)
                  let v207 := v205 <<< (UInt256.ofNat 2)
                  let bit_209 := bit
                  let v211 := v_1 >>> bit_209
                  let v213 := v211 &&& (UInt256.ofNat 3)
                  let v214 := v207 ||| v213
                  let entry ← downcast_truncate_u256_usize v214
                  let v218 := decide (entry ≠ (UInt256.ofNat 0))
                  let join_1 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
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
                    loop_4 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  let join_2 : Point_Mont_Base → (Vector Point_Mont_Base 16) → (Vector Mont_Base 16) → Mont_Base → Mont_Base → UInt256 → Mont_Base → Mont_Base → Point_Mont_Base → UInt256 → Point_Mont_Base → Point_Mont_Base → (Vector UInt256 3) → Range_Known_1_Known_4 → Point_Mont_Base → Point_Mont_Base → Range_Known_1_Known_16 → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → Point_Mont_Base → Point_Mont_Base → Mont_Base → Mont_Base → Point_Mont_Base → FeM Point_Mont_Base := fun g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 => do
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
                    join_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
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
                      r36 := result_225
                      let result_230 ← Point_add_affine_Mont_Base r36 r34 r35
                      result := result_230
                      join_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                    else
                      join_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  else
                    join_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                else
                  let result_236 := result
                  return result_236
              partial_fixpoint
              loop_4 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          partial_fixpoint
          loop_3 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
      partial_fixpoint
      loop_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v86
  partial_fixpoint
  loop g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v43

/-- `Point<Mont<Base>>::to_affine` -/
def Point_to_affine_Mont_Base (self : Point_Mont_Base) : FeM (Mont_Base × Mont_Base) := do
  let mut z_inv : Mont_Base := default
  let mut z_inv2 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  let v1 := self.z
  r2 := v1
  let z_inv_3 ← Mont_inverse_Base r2
  z_inv := z_inv_3
  let z_inv2_6 ← Mont_square_Base z_inv
  z_inv2 := z_inv2_6
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

/-- `<Mont<Base> as Field>::neg` -/
def Mont_neg_Base (self : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let v1 ← Mont_zero_Base
  r0 := v1
  let v3 ← Mont_sub_Base r0 self
  return v3

/-- `lib::secp256k1::recover` -/
def recover (hash : UInt256) (recovery_id : UInt256) (r : UInt256) (s : UInt256) : FeM Option_PublicKey := do
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
  let v8 := decide (recovery_id.toNat ≤ (UInt256.ofNat 1).toNat)
  let join : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → Bool → FeM Option_PublicKey := fun x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v4 => do
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
    let _ ← «assert» v4
    let x_26 ← Mont_new_Base r
    x := x_26
    let v29 ← Mont_square_Base x
    r8 := v29
    let v32 ← Mont_mul_Base r8 x
    r9 := v32
    let v34 ← Mont_new_Base (UInt256.ofNat 7)
    r10 := v34
    let alpha_37 ← Mont_add_Base r9 r10
    alpha := alpha_37
    let v39 ← P
    let v41 ← checkedAdd v39 (UInt256.ofNat 1)
    let v43 ← checkedDiv v41 (UInt256.ofNat 4)
    let beta_45 ← Mont_pow_Base alpha v43
    beta := beta_45
    let v48 ← Mont_square_Base beta
    r11 := v48
    let v51 ← Mont_equals_Base r11 alpha
    let v52 := !v51
    if v52 then
      let v53 := Option_PublicKey.None
      return v53
    else
      let v57 ← Mont_value_Base beta
      let v59 := v57 &&& (UInt256.ofNat 1)
      let v61 := decide (v59 = recovery_id)
      let join_1 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → Mont_Base → FeM Option_PublicKey := fun x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 y_5 => do
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
        y := y_5
        let v68 ← Mont_new_Order r
        r12 := v68
        let r_inverse_70 ← Mont_inverse_Order r12
        r_inverse := r_inverse_70
        let v73 ← N
        let v74 ← checkedRem hash v73
        let v75 ← Mont_new_Order v74
        r13 := v75
        let v77 ← Mont_neg_Order r13
        r14 := v77
        let v80 ← Mont_mul_Order r14 r_inverse
        r15 := v80
        let u ← Mont_value_Order r15
        let v84 ← Mont_new_Order s
        r16 := v84
        let v87 ← Mont_mul_Order r16 r_inverse
        r17 := v87
        let v ← Mont_value_Order r17
        let v93 ← Point_affine_Mont_Base x y
        r18 := v93
        let key_95 ← combination u v r18
        key := key_95
        let v98 ← Point_is_infinity_Mont_Base key
        if v98 then
          let v99 := Option_PublicKey.None
          return v99
        else
          let v103 ← Point_to_affine_Mont_Base key
          let key_x_104 := v103.fst
          key_x := key_x_104
          let key_y_106 := v103.snd
          key_y := key_y_106
          let v109 ← Mont_value_Base key_x
          let v111 ← Mont_value_Base key_y
          let v112 := (PublicKey.mk v109 v111)
          let v113 := Option_PublicKey.Some v112
          return v113
      if v61 then
        let beta_62 := beta
        join_1 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 beta_62
      else
        let v65 ← Mont_neg_Base beta
        join_1 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 v65
  let join_2 : Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Point_Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Base → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Mont_Order → Point_Mont_Base → FeM Option_PublicKey := fun x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 => do
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
    join x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 false
  if v8 then
    let v11 := decide (r ≠ (UInt256.ofNat 0))
    if v11 then
      let v13 ← N
      let v14 := decide (r.toNat < v13.toNat)
      if v14 then
        let v17 := decide (s ≠ (UInt256.ofNat 0))
        if v17 then
          let v19 ← N
          let v20 := decide (s.toNat < v19.toNat)
          if v20 then
            join x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 true
          else
            join_2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
        else
          join_2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
      else
        join_2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
    else
      join_2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18
  else
    join_2 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18

/-- `<MemBuffer as ByteSource>::len` -/
def MemBuffer_len_1 (self : MemBuffer) : FeM UInt256 := do
  let v1 ← MemBuffer_len self
  return v1

/-- `<MemBuffer as ByteSource>::lane_at` -/
def MemBuffer_lane_at (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_read_le_u64 self index
  return v2

/-- `<MemBuffer as ByteSource>::byte_at` -/
def MemBuffer_byte_at (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_index_ref self index
  let v3 ← mloadInt (.u 8) v2
  return v3

/-- `lib::word_at<MemBuffer>` -/
def word_at_MemBuffer (input : MemBuffer) (offset : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut r1 : Range_Unknown_Unknown := default
  let v6 ← checkedAdd offset (UInt256.ofNat 32)
  let v7 ← MemBuffer_len_1 input
  let v8 := decide (v6.toNat ≤ v7.toNat)
  if v8 then
    let v9 ← MemBuffer_lane_at input offset
    let v10 := IntTy.swapBytes (.u 64) v9
    let v13 := v10 <<< (UInt256.ofNat 192)
    let v16 ← checkedAdd offset (UInt256.ofNat 8)
    let v17 ← MemBuffer_lane_at input v16
    let v18 := IntTy.swapBytes (.u 64) v17
    let v21 := v18 <<< (UInt256.ofNat 128)
    let v22 := v13 ||| v21
    let v25 ← checkedAdd offset (UInt256.ofNat 16)
    let v26 ← MemBuffer_lane_at input v25
    let v27 := IntTy.swapBytes (.u 64) v26
    let v30 := v27 <<< (UInt256.ofNat 64)
    let v31 := v22 ||| v30
    let v34 ← checkedAdd offset (UInt256.ofNat 24)
    let v35 ← MemBuffer_lane_at input v34
    let v36 := IntTy.swapBytes (.u 64) v35
    let v38 := v31 ||| v36
    return v38
  else
    value := (UInt256.ofNat 0)
    let v46 ← checkedAdd offset (UInt256.ofNat 32)
    let v48 := (Range_Unknown_Unknown.mk offset v46)
    r1 := v48
    let v50 ← Range_Unknown_Unknown_start r1
    let rec loop (value : UInt256) (r1 : Range_Unknown_Unknown) (v2 : UInt256) : FeM UInt256 := do
      let mut value := value
      let mut r1 := r1
      let v53 ← Range_Unknown_Unknown_has r1 v2
      if v53 then
        let v55 ← Range_Unknown_Unknown_at r1 v2
        let index := v55
        let value_57 := value
        let v60 := value_57 <<< (UInt256.ofNat 8)
        let v61 ← MemBuffer_len_1 input
        let v62 := decide (index.toNat < v61.toNat)
        let join : UInt256 → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun value r1 v3 => do
          let mut value := value
          let mut r1 := r1
          let value_68 := v60 ||| v3
          value := value_68
          let v72 ← Range_Unknown_Unknown_next r1 v2
          loop value r1 v72
        if v62 then
          let v63 ← MemBuffer_byte_at input index
          join value r1 v63
        else
          join value r1 (UInt256.ofNat 0)
      else
        let value_74 := value
        return value_74
    partial_fixpoint
    loop value r1 v50

/-- `lib::bn254_field::MODULUS` -/
def MODULUS : FeM UInt256 := do
  return (UInt256.ofNat 21888242871839275222246405745257275088696311157297823662689037894645226208583)

/-- `<Bn254Base as Modulus>::VALUE` -/
def Bn254Base_VALUE : FeM UInt256 := do
  let v0 ← MODULUS
  return v0

/-- `lib::montgomery::raw<Bn254Base>` -/
def raw_Bn254Base (value : UInt256) : FeM Mont_Bn254Base := do
  let v1 ← downcast_truncate_u256_u64 value
  let v4 := value >>> (UInt256.ofNat 64)
  let v5 ← downcast_truncate_u256_u64 v4
  let v8 := value >>> (UInt256.ofNat 128)
  let v9 ← downcast_truncate_u256_u64 v8
  let v12 := value >>> (UInt256.ofNat 192)
  let v13 ← downcast_truncate_u256_u64 v12
  let v14 := (Mont_Bn254Base.mk v1 v5 v9 v13)
  return v14

/-- `<Bn254Base as Modulus>::R2` -/
def Bn254Base_R2 : FeM UInt256 := do
  return (UInt256.ofNat 3096616502983703923843567936837374451735540968419076528771170197431451843209)

/-- `<Bn254Base as Modulus>::INV` -/
def Bn254Base_INV : FeM UInt256 := do
  return (UInt256.ofNat 9786893198990664585)

/-- `<Bn254Base as Modulus>::P0` -/
def Bn254Base_P0 : FeM UInt256 := do
  return (UInt256.ofNat 4332616871279656263)

/-- `<Bn254Base as Modulus>::P1` -/
def Bn254Base_P1 : FeM UInt256 := do
  return (UInt256.ofNat 10917124144477883021)

/-- `<Bn254Base as Modulus>::P2` -/
def Bn254Base_P2 : FeM UInt256 := do
  return (UInt256.ofNat 13281191951274694749)

/-- `<Bn254Base as Modulus>::P3` -/
def Bn254Base_P3 : FeM UInt256 := do
  return (UInt256.ofNat 3486998266802970665)

def Mont_shift_reduced_Bn254Base_k1 (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (u1_33 : UInt256) (ca_35 : UInt256) (u2_38 : UInt256) (cb_40 : UInt256) (v43 : (UInt256 × UInt256)) (k2 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let j2 := v43.snd
  let v48 ← adc u2_38 k2 ca_35
  let u2_49 := v48.fst
  let ca_51 := v48.snd
  let v53 ← adc s3 j2 cb_40
  let u3_54 := v53.fst
  let cb_56 := v53.snd
  let v58 ← Bn254Base_P3
  let v59 ← mul_wide m v58
  let k3 := v59.fst
  let j3 := v59.snd
  let v64 ← adc u3_54 k3 ca_51
  let u3_65 := v64.fst
  let ca_67 := v64.snd
  let v69 ← adc s4 j3 cb_56
  let u4_70 := v69.fst
  let cb_72 := v69.snd
  let v75 ← adc u4_70 (UInt256.ofNat 0) ca_67
  let u4_76 := v75.fst
  let ca_78 := v75.snd
  let v80 ← adc s5 cb_72 ca_78
  let u5 := v80.fst
  let v83 := (Mont_Bn254Base.mk u1_33 u2_49 u3_65 u4_76)
  let v84 := (v83, u5)
  return v84

/-- `Mont<Bn254Base>::shift_reduced` -/
def Mont_shift_reduced_Bn254Base (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v6 ← Bn254Base_INV
  let m ← u64_wrapping_mul s0 v6
  let v10 ← Bn254Base_P0
  let v11 ← mul_wide m v10
  let k0 := v11.fst
  let j0 := v11.snd
  let v17 ← adc s0 k0 (UInt256.ofNat 0)
  let ca_18 := v17.snd
  let v21 ← adc s1 j0 (UInt256.ofNat 0)
  let u1_22 := v21.fst
  let cb_24 := v21.snd
  let v26 ← Bn254Base_P1
  let v27 ← mul_wide m v26
  let k1 := v27.fst
  let j1 := v27.snd
  let v32 ← adc u1_22 k1 ca_18
  let u1_33 := v32.fst
  let ca_35 := v32.snd
  let v37 ← adc s2 j1 cb_24
  let u2_38 := v37.fst
  let cb_40 := v37.snd
  let v42 ← Bn254Base_P2
  let v43 ← mul_wide m v42
  let k2 := v43.fst
  Mont_shift_reduced_Bn254Base_k1 s3 s4 s5 m u1_33 ca_35 u2_38 cb_40 v43 k2

/-- `Mont<Bn254Base>::first_row` -/
def Mont_first_row_Bn254Base (self : Mont_Bn254Base) (b : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v2 := self.l0
  let v3 ← mul_wide b v2
  let s0 := v3.fst
  let h0 := v3.snd
  let v8 := self.l1
  let v9 ← mul_wide b v8
  let l1 := v9.fst
  let h1 := v9.snd
  let v15 ← adc h0 l1 (UInt256.ofNat 0)
  let s1 := v15.fst
  let c_18 := v15.snd
  let v20 := self.l2
  let v21 ← mul_wide b v20
  let l2 := v21.fst
  let h2 := v21.snd
  let v26 ← adc h1 l2 c_18
  let s2 := v26.fst
  let c_29 := v26.snd
  let v31 := self.l3
  let v32 ← mul_wide b v31
  let l3 := v32.fst
  let h3 := v32.snd
  let v37 ← adc h2 l3 c_29
  let s3 := v37.fst
  let c_40 := v37.snd
  let v43 ← adc h3 (UInt256.ofNat 0) c_40
  let s4 := v43.fst
  let v47 ← Mont_shift_reduced_Bn254Base s0 s1 s2 s3 s4 (UInt256.ofNat 0)
  return v47

def Mont_row_Bn254Base_k2 (s0 : UInt256) (s1_31 : UInt256) (s2_48 : UInt256) (s3_65 : UInt256) (s4_76 : UInt256) (v81 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  let s5 := v81.fst
  let v84 ← Mont_shift_reduced_Bn254Base s0 s1_31 s2_48 s3_65 s4_76 s5
  return v84

def Mont_row_Bn254Base_k1 (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) (s0 : UInt256) (s1_31 : UInt256) (ca_33 : UInt256) (s2_37 : UInt256) (cb_39 : UInt256) (v41 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v42 ← mul_wide b v41
  let l2 := v42.fst
  let h2 := v42.snd
  let v47 ← adc s2_37 l2 ca_33
  let s2_48 := v47.fst
  let ca_50 := v47.snd
  let v52 := t.l3
  let v53 ← adc v52 h2 cb_39
  let s3_54 := v53.fst
  let cb_56 := v53.snd
  let v58 := self.l3
  let v59 ← mul_wide b v58
  let l3 := v59.fst
  let h3 := v59.snd
  let v64 ← adc s3_54 l3 ca_50
  let s3_65 := v64.fst
  let ca_67 := v64.snd
  let v69 ← adc top h3 cb_56
  let s4_70 := v69.fst
  let cb_72 := v69.snd
  let v75 ← adc s4_70 (UInt256.ofNat 0) ca_67
  let s4_76 := v75.fst
  let ca_78 := v75.snd
  let v81 ← adc cb_72 (UInt256.ofNat 0) ca_78
  Mont_row_Bn254Base_k2 s0 s1_31 s2_48 s3_65 s4_76 v81

/-- `Mont<Bn254Base>::row` -/
def Mont_row_Bn254Base (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  let v4 := self.l0
  let v5 ← mul_wide b v4
  let l0 := v5.fst
  let h0 := v5.snd
  let v10 := t.l0
  let v12 ← adc v10 l0 (UInt256.ofNat 0)
  let s0 := v12.fst
  let ca_15 := v12.snd
  let v17 := t.l1
  let v19 ← adc v17 h0 (UInt256.ofNat 0)
  let s1_20 := v19.fst
  let cb_22 := v19.snd
  let v24 := self.l1
  let v25 ← mul_wide b v24
  let l1 := v25.fst
  let h1 := v25.snd
  let v30 ← adc s1_20 l1 ca_15
  let s1_31 := v30.fst
  let ca_33 := v30.snd
  let v35 := t.l2
  let v36 ← adc v35 h1 cb_22
  let s2_37 := v36.fst
  let cb_39 := v36.snd
  let v41 := self.l2
  Mont_row_Bn254Base_k1 self b t top s0 s1_31 ca_33 s2_37 cb_39 v41

/-- `Mont<Bn254Base>::reduce` -/
def Mont_reduce_Bn254Base (self : Mont_Bn254Base) (top : UInt256) : FeM Mont_Bn254Base := do
  let v3 := self.l0
  let v4 ← Bn254Base_P0
  let v6 ← sbb v3 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  let b_9 := v6.snd
  let v11 := self.l1
  let v12 ← Bn254Base_P1
  let v13 ← sbb v11 v12 b_9
  let d1 := v13.fst
  let b_16 := v13.snd
  let v18 := self.l2
  let v19 ← Bn254Base_P2
  let v20 ← sbb v18 v19 b_16
  let d2 := v20.fst
  let b_23 := v20.snd
  let v25 := self.l3
  let v26 ← Bn254Base_P3
  let v27 ← sbb v25 v26 b_23
  let d3 := v27.fst
  let b_30 := v27.snd
  let v33 ← sbb top (UInt256.ofNat 0) b_30
  let b_34 := v33.snd
  let v37 := decide (b_34 = (UInt256.ofNat 1))
  let join : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  if v37 then
    let v38 := self
    join v38
  else
    let v40 := (Mont_Bn254Base.mk d0 d1 d2 d3)
    join v40

/-- `<Mont<Bn254Base> as Field>::mul` -/
def Mont_mul_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut t_r0 : Mont_Bn254Base := default
  let mut t_r1 : Mont_Bn254Base := default
  let mut t_r2 : Mont_Bn254Base := default
  let mut t_r3 : Mont_Bn254Base := default
  let v2 := other.l0
  let v3 ← Mont_first_row_Bn254Base self v2
  let t_4 := v3.fst
  t_r0 := t_4
  let top_6 := v3.snd
  let v8 := other.l1
  let v10 ← Mont_row_Bn254Base self v8 t_r0 top_6
  let t_11 := v10.fst
  t_r1 := t_11
  let top_13 := v10.snd
  let v15 := other.l2
  let v17 ← Mont_row_Bn254Base self v15 t_r1 top_13
  let t_18 := v17.fst
  t_r2 := t_18
  let top_20 := v17.snd
  let v22 := other.l3
  let v24 ← Mont_row_Bn254Base self v22 t_r2 top_20
  let t_25 := v24.fst
  t_r3 := t_25
  let top_27 := v24.snd
  let v30 ← Mont_reduce_Bn254Base t_r3 top_27
  return v30

/-- `Mont<Bn254Base>::new` -/
def Mont_new_Bn254Base (value : UInt256) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let v2 ← Bn254Base_VALUE
  let v3 := decide (value.toNat < v2.toNat)
  let _ ← «assert» v3
  let v5 ← raw_Bn254Base value
  r0 := v5
  let v6 ← Bn254Base_R2
  let v7 ← raw_Bn254Base v6
  r1 := v7
  let v10 ← Mont_mul_Bn254Base r0 r1
  return v10

/-- `lib::bn254_field::fp` -/
def fp (value : UInt256) : FeM Mont_Bn254Base := do
  let v1 ← Mont_new_Bn254Base value
  return v1

/-- `<Mont<Bn254Base> as Field>::one` -/
def Mont_one_Bn254Base : FeM Mont_Bn254Base := do
  let v1 ← Mont_new_Bn254Base (UInt256.ofNat 1)
  return v1

/-- `Point<Mont<Bn254Base>>::affine` -/
def Point_affine_Mont_Bn254Base (x : Mont_Bn254Base) (y : Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  let v2 := x
  let v3 := y
  let v4 ← Mont_one_Bn254Base
  let v5 := (Point_Mont_Bn254Base.mk v2 v3 v4)
  return v5

/-- `<Mont<Bn254Base> as Field>::square` -/
def Mont_square_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v1 ← Mont_mul_Bn254Base self self
  return v1

/-- `<Mont<Bn254Base> as Field>::add` -/
def Mont_add_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let v2 := self.l0
  let v3 := other.l0
  let v5 ← adc v2 v3 (UInt256.ofNat 0)
  let s0 := v5.fst
  let c_8 := v5.snd
  let v10 := self.l1
  let v11 := other.l1
  let v12 ← adc v10 v11 c_8
  let s1 := v12.fst
  let c_15 := v12.snd
  let v17 := self.l2
  let v18 := other.l2
  let v19 ← adc v17 v18 c_15
  let s2 := v19.fst
  let c_22 := v19.snd
  let v24 := self.l3
  let v25 := other.l3
  let v26 ← adc v24 v25 c_22
  let s3 := v26.fst
  let c_29 := v26.snd
  let v31 := (Mont_Bn254Base.mk s0 s1 s2 s3)
  r0 := v31
  let v33 ← Mont_reduce_Bn254Base r0 c_29
  return v33

/-- `<Mont<Bn254Base> as Field>::equals` -/
def Mont_equals_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Bool := do
  let v3 := self.l0
  let v4 := other.l0
  let v5 := decide (v3 = v4)
  let join : Bool → FeM Bool := fun v2 => do
    return v2
  let join_1 : FeM Bool := do
    join false
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
          join true
        else
          join_1
      else
        join_1
    else
      join_1
  else
    join_1

/-- `lib::bn254_curve::on_curve<Mont<Bn254Base>>` -/
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

/-- `<Mont<Bn254Base> as Field>::zero` -/
def Mont_zero_Bn254Base : FeM Mont_Bn254Base := do
  let v4 := (Mont_Bn254Base.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  return v4

/-- `Point<Mont<Bn254Base>>::infinity` -/
def Point_infinity_Mont_Bn254Base : FeM Point_Mont_Bn254Base := do
  let v0 ← Mont_one_Bn254Base
  let v1 ← Mont_one_Bn254Base
  let v2 ← Mont_zero_Bn254Base
  let v3 := (Point_Mont_Bn254Base.mk v0 v1 v2)
  return v3

/-- `lib::bn254::g1_point<MemBuffer>` -/
def g1_point_MemBuffer (input : MemBuffer) (offset : UInt256) : FeM Option_Point_Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let x ← word_at_MemBuffer input offset
  let v6 ← checkedAdd offset (UInt256.ofNat 32)
  let y ← word_at_MemBuffer input v6
  let v9 ← MODULUS
  let v10 := decide (v9.toNat ≤ x.toNat)
  let join : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let v42 := Option_Point_Mont_Bn254Base.None
    return v42
  if v10 then
    join r0 r1 r2 r3 r4
  else
    let v11 ← MODULUS
    let v12 := decide (v11.toNat ≤ y.toNat)
    if v12 then
      join r0 r1 r2 r3 r4
    else
      let v16 := decide (x = (UInt256.ofNat 0))
      let join_1 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let v23 ← fp x
        r0 := v23
        let v24 ← fp y
        r1 := v24
        let point ← Point_affine_Mont_Bn254Base r0 r1
        let v29 := point.x
        r2 := v29
        let v30 := point.y
        r3 := v30
        let v32 ← fp (UInt256.ofNat 3)
        r4 := v32
        let v36 ← on_curve_Mont_Bn254Base r2 r3 r4
        let v37 := !v36
        if v37 then
          let v38 := Option_Point_Mont_Bn254Base.None
          return v38
        else
          let v41 := Option_Point_Mont_Bn254Base.Some point
          return v41
      if v16 then
        let v18 := decide (y = (UInt256.ofNat 0))
        if v18 then
          let v19 ← Point_infinity_Mont_Bn254Base
          let v20 := Option_Point_Mont_Bn254Base.Some v19
          return v20
        else
          join_1 r0 r1 r2 r3 r4
      else
        join_1 r0 r1 r2 r3 r4

/-- `<Mont<Bn254Base> as Field>::is_zero` -/
def Mont_is_zero_Bn254Base (self : Mont_Bn254Base) : FeM Bool := do
  let v2 := self.l0
  let v4 := decide (v2 = (UInt256.ofNat 0))
  let join : Bool → FeM Bool := fun v1 => do
    return v1
  let join_1 : FeM Bool := do
    join false
  if v4 then
    let v5 := self.l1
    let v7 := decide (v5 = (UInt256.ofNat 0))
    if v7 then
      let v8 := self.l2
      let v10 := decide (v8 = (UInt256.ofNat 0))
      if v10 then
        let v11 := self.l3
        let v13 := decide (v11 = (UInt256.ofNat 0))
        if v13 then
          join true
        else
          join_1
      else
        join_1
    else
      join_1
  else
    join_1

/-- `Point<Mont<Bn254Base>>::is_infinity` -/
def Point_is_infinity_Mont_Bn254Base (self : Point_Mont_Bn254Base) : FeM Bool := do
  let mut r0 : Mont_Bn254Base := default
  let v1 := self.z
  r0 := v1
  let v3 ← Mont_is_zero_Bn254Base r0
  return v3

/-- `<Mont<Bn254Base> as Field>::sub` -/
def Mont_sub_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v3 := self.l0
  let v4 := other.l0
  let v6 ← sbb v3 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  let b_9 := v6.snd
  let v11 := self.l1
  let v12 := other.l1
  let v13 ← sbb v11 v12 b_9
  let d1 := v13.fst
  let b_16 := v13.snd
  let v18 := self.l2
  let v19 := other.l2
  let v20 ← sbb v18 v19 b_16
  let d2 := v20.fst
  let b_23 := v20.snd
  let v25 := self.l3
  let v26 := other.l3
  let v27 ← sbb v25 v26 b_23
  let d3 := v27.fst
  let b_30 := v27.snd
  let v32 ← Bn254Base_P0
  let v34 ← adc d0 v32 (UInt256.ofNat 0)
  let s0 := v34.fst
  let c_37 := v34.snd
  let v39 ← Bn254Base_P1
  let v40 ← adc d1 v39 c_37
  let s1 := v40.fst
  let c_43 := v40.snd
  let v45 ← Bn254Base_P2
  let v46 ← adc d2 v45 c_43
  let s2 := v46.fst
  let c_49 := v46.snd
  let v51 ← Bn254Base_P3
  let v52 ← adc d3 v51 c_49
  let s3 := v52.fst
  let v56 := decide (b_30 = (UInt256.ofNat 1))
  let join : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  if v56 then
    let v57 := (Mont_Bn254Base.mk s0 s1 s2 s3)
    join v57
  else
    let v59 := (Mont_Bn254Base.mk d0 d1 d2 d3)
    join v59

/-- `<Mont<Bn254Base> as Field>::double` -/
def Mont_double_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let v1 ← Mont_add_Bn254Base self self
  return v1

/-- `Point<Mont<Bn254Base>>::double` -/
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
  let join : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Point_Mont_Bn254Base := fun a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 => do
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
    join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    let v2 := self.y
    r6 := v2
    let v4 ← Mont_is_zero_Bn254Base r6
    if v4 then
      join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      let v7 := self.x
      r7 := v7
      let a_9 ← Mont_square_Bn254Base r7
      a := a_9
      let v11 := self.y
      r8 := v11
      let b_13 ← Mont_square_Bn254Base r8
      b := b_13
      let c_16 ← Mont_square_Bn254Base b
      c := c_16
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
      let d_31 ← Mont_double_Bn254Base r13
      d := d_31
      let v34 ← Mont_double_Bn254Base a
      r14 := v34
      let e_37 ← Mont_add_Bn254Base r14 a
      e := e_37
      let v40 ← Mont_square_Bn254Base e
      r15 := v40
      let v42 ← Mont_double_Bn254Base d
      r16 := v42
      let x_45 ← Mont_sub_Bn254Base r15 r16
      x := x_45
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
      let y ← Mont_sub_Bn254Base r18 r21
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

/-- `<Bn254Base as Modulus>::SIGNED_62` -/
def Bn254Base_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v5 := #v[(UInt256.ofNat 4332616871279656263), (UInt256.ofNat 2163322412065040948), (UInt256.ofNat 361514372735272409), (UInt256.ofNat 1806960190875503214), (UInt256.ofNat 48)]
  return v5

/-- `<Bn254Base as Modulus>::INV62` -/
def Bn254Base_INV62 : FeM UInt256 := do
  return (UInt256.ofNat 4048164856291499127)

/-- `<Bn254Base as Modulus>::R3` -/
def Bn254Base_R3 : FeM UInt256 := do
  return (UInt256.ofNat 14921786541159648185948152738563080959093619838510245177710943249661917737183)

/-- `<Mont<Bn254Base> as Field>::inverse` -/
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
  let limbs ← invert_256 r0 r1 v7
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

/-- `Point<Mont<Bn254Base>>::to_affine` -/
def Point_to_affine_Mont_Bn254Base (self : Point_Mont_Bn254Base) : FeM (Mont_Bn254Base × Mont_Bn254Base) := do
  let mut z_inv : Mont_Bn254Base := default
  let mut z_inv2 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  let mut r5 : Mont_Bn254Base := default
  let v1 := self.z
  r2 := v1
  let z_inv_3 ← Mont_inverse_Bn254Base r2
  z_inv := z_inv_3
  let z_inv2_6 ← Mont_square_Bn254Base z_inv
  z_inv2 := z_inv2_6
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

/-- `Mont<Bn254Base>::value` -/
def Mont_value_Bn254Base (self : Mont_Bn254Base) : FeM UInt256 := do
  let mut r0 : Mont_Bn254Base := default
  let v2 ← raw_Bn254Base (UInt256.ofNat 1)
  r0 := v2
  let value ← Mont_mul_Bn254Base self r0
  let v6 := value.l0
  let v8 := value.l1
  let v11 := v8 <<< (UInt256.ofNat 64)
  let v12 := v6 ||| v11
  let v13 := value.l2
  let v16 := v13 <<< (UInt256.ofNat 128)
  let v17 := v12 ||| v16
  let v18 := value.l3
  let v21 := v18 <<< (UInt256.ofNat 192)
  let v22 := v17 ||| v21
  return v22

/-- `lib::bn254_field::fp_value` -/
def fp_value (a : Mont_Bn254Base) : FeM UInt256 := do
  let v1 ← Mont_value_Bn254Base a
  return v1

/-- `lib::bn254::encode` -/
def encode (point : Point_Mont_Bn254Base) : FeM (Vector UInt256 2) := do
  let mut x : Mont_Bn254Base := default
  let mut y : Mont_Bn254Base := default
  let v1 ← Point_is_infinity_Mont_Bn254Base point
  if v1 then
    let v4 := #v[(UInt256.ofNat 0), (UInt256.ofNat 0)]
    return v4
  else
    let v7 ← Point_to_affine_Mont_Bn254Base point
    let x_8 := v7.fst
    x := x_8
    let y_10 := v7.snd
    y := y_10
    let v13 ← fp_value x
    let v15 ← fp_value y
    let v16 := #v[v13, v15]
    return v16

/-- `lib::SUCCESS` -/
def SUCCESS : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `lib::HOST_ALLOCATION` -/
def HOST_ALLOCATION : FeM UInt256 := do
  return (UInt256.ofNat 2)

/-- `lib::host_allocation` -/
def host_allocation (gas_used : UInt256) : FeM Output := do
  let v1 ← HOST_ALLOCATION
  let v3 ← MemBuffer_empty
  let v4 := (Output.mk v1 gas_used v3)
  return v4


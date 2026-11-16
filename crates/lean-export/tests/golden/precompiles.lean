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
  return UInt256.ofNat 115792089237316195423570985008687907852837564279074904382605163141518161494337

/-- `lib::secp256k1::P` -/
def P : FeM UInt256 := do
  return UInt256.ofNat 115792089237316195423570985008687907853269984665640564039457584007908834671663

/-- `<Base as Modulus>::VALUE` -/
def Base_VALUE : FeM UInt256 := do
  let v0 ← P
  return v0

/-- `lib::montgomery::raw<Base>` -/
def raw_Base (value : UInt256) : FeM Mont_Base := do
  -- Mont {
  let v1 ← downcast_truncate_u256_u64 value
  let v5 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 64))
  let v9 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 128))
  let v13 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 192))
  return Mont_Base.mk v1 v5 v9 v13

/-- `<Base as Modulus>::R2` -/
def Base_R2 : FeM UInt256 := do
  return UInt256.ofNat 18446752466076602529

/-- `lib::montgomery::mul_wide` -/
def mul_wide (a : UInt256) (b : UInt256) : FeM (UInt256 × UInt256) := do
  -- let t = (a as u128) * (b as u128)
  let t ← IntTy.mul (.u 128) a b
  -- (t.downcast_truncate(), (t >> 64).downcast_truncate())
  let v8 ← downcast_truncate_u128_u64 t
  let v11 ← downcast_truncate_u128_u64 (t >>> (UInt256.ofNat 64))
  return (v8, v11)

/-- `lib::montgomery::adc` -/
def adc (a : UInt256) (b : UInt256) (carry : UInt256) : FeM (UInt256 × UInt256) := do
  -- let t = (a as u128) + (b as u128) + (carry as u128)
  let v7 ← IntTy.add (.u 128) a b
  let t ← IntTy.add (.u 128) v7 carry
  -- (t.downcast_truncate(), (t >> 64).downcast_truncate())
  let v12 ← downcast_truncate_u128_u64 t
  let v15 ← downcast_truncate_u128_u64 (t >>> (UInt256.ofNat 64))
  return (v12, v15)

/-- `<Base as Modulus>::INV` -/
def Base_INV : FeM UInt256 := do
  return UInt256.ofNat 15580212934572586289

/-- `<Base as Modulus>::P0` -/
def Base_P0 : FeM UInt256 := do
  return UInt256.ofNat 18446744069414583343

/-- `<Base as Modulus>::P1` -/
def Base_P1 : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551615

/-- `<Base as Modulus>::P2` -/
def Base_P2 : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551615

/-- `<Base as Modulus>::P3` -/
def Base_P3 : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551615

def Mont_shift_reduced_Base_k3 (s5 : UInt256) (v32 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) (v64 : (UInt256 × UInt256)) (v69 : (UInt256 × UInt256)) (v75 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (u5, _) = adc(s5, cb, ca)
  let v80 ← adc s5 v69.snd v75.snd
  -- (Mont { l0: u1, l1: u2, l2: u3, l3: u4 }, u5)
  let v84 := ((Mont_Base.mk v32.fst v48.fst v64.fst v75.fst), v80.fst)
  return v84

def Mont_shift_reduced_Base_k2 (s4 : UInt256) (s5 : UInt256) (m : UInt256) (v32 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) (v53 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (k3, j3) = mul_wide(m, M::P3)
  let v58 ← Base_P3
  let v59 ← mul_wide m v58
  -- let (u3, ca) = adc(u3, k3, ca)
  let v64 ← adc v53.fst v59.fst v48.snd
  -- let (u4, cb) = adc(s4, j3, cb)
  let v69 ← adc s4 v59.snd v53.snd
  -- let (u4, ca) = adc(u4, 0, ca)
  let v75 ← adc v69.fst (UInt256.ofNat 0) v64.snd
  Mont_shift_reduced_Base_k3 s5 v32 v48 v64 v69 v75

def Mont_shift_reduced_Base_k1 (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (v17 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (v27 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (u1, ca) = adc(u1, k1, ca)
  let v32 ← adc v21.fst v27.fst v17.snd
  -- let (u2, cb) = adc(s2, j1, cb)
  let v37 ← adc s2 v27.snd v21.snd
  -- let (k2, j2) = mul_wide(m, M::P2)
  let v42 ← Base_P2
  let v43 ← mul_wide m v42
  -- let (u2, ca) = adc(u2, k2, ca)
  let v48 ← adc v37.fst v43.fst v32.snd
  -- let (u3, cb) = adc(s3, j2, cb)
  let v53 ← adc s3 v43.snd v37.snd
  Mont_shift_reduced_Base_k2 s4 s5 m v32 v48 v53

/-- `Mont<Base>::shift_reduced` -/
def Mont_shift_reduced_Base (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Base × UInt256) := do
  -- let m = s0.wrapping_mul(M::INV)
  let v6 ← Base_INV
  let m ← u64_wrapping_mul s0 v6
  -- let (k0, j0) = mul_wide(m, M::P0)
  let v10 ← Base_P0
  let v11 ← mul_wide m v10
  -- let (_, ca) = adc(s0, k0, 0)
  let v17 ← adc s0 v11.fst (UInt256.ofNat 0)
  -- let (u1, cb) = adc(s1, j0, 0)
  let v21 ← adc s1 v11.snd (UInt256.ofNat 0)
  -- let (k1, j1) = mul_wide(m, M::P1)
  let v26 ← Base_P1
  let v27 ← mul_wide m v26
  Mont_shift_reduced_Base_k1 s2 s3 s4 s5 m v17 v21 v27

def Mont_first_row_Base_k1 (self : Mont_Base) (b : UInt256) (v3 : (UInt256 × UInt256)) (v9 : (UInt256 × UInt256)) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (s2, c) = adc(h1, l2, c)
  let v26 ← adc v9.snd v21.fst v15.snd
  -- let (l3, h3) = mul_wide(b, self.l3)
  let v32 ← mul_wide b self.l3
  -- let (s3, c) = adc(h2, l3, c)
  let v37 ← adc v21.snd v32.fst v26.snd
  -- let (s4, _) = adc(h3, 0, c)
  let v43 ← adc v32.snd (UInt256.ofNat 0) v37.snd
  -- Self::shift_reduced(s0, s1, s2, s3, s4, 0)
  let v47 ← Mont_shift_reduced_Base v3.fst v15.fst v26.fst v37.fst v43.fst (UInt256.ofNat 0)
  return v47

/-- `Mont<Base>::first_row` -/
def Mont_first_row_Base (self : Mont_Base) (b : UInt256) : FeM (Mont_Base × UInt256) := do
  -- let (s0, h0) = mul_wide(b, self.l0)
  let v3 ← mul_wide b self.l0
  -- let (l1, h1) = mul_wide(b, self.l1)
  let v9 ← mul_wide b self.l1
  -- let (s1, c) = adc(h0, l1, 0)
  let v15 ← adc v3.snd v9.fst (UInt256.ofNat 0)
  -- let (l2, h2) = mul_wide(b, self.l2)
  let v21 ← mul_wide b self.l2
  Mont_first_row_Base_k1 self b v3 v9 v15 v21

def Mont_row_Base_k3 (v12 : (UInt256 × UInt256)) (v30 : (UInt256 × UInt256)) (v47 : (UInt256 × UInt256)) (v64 : (UInt256 × UInt256)) (v69 : (UInt256 × UInt256)) (v75 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (s5, _) = adc(cb, 0, ca)
  let v81 ← adc v69.snd (UInt256.ofNat 0) v75.snd
  -- Self::shift_reduced(s0, s1, s2, s3, s4, s5)
  let v84 ← Mont_shift_reduced_Base v12.fst v30.fst v47.fst v64.fst v75.fst v81.fst
  return v84

def Mont_row_Base_k2 (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) (v12 : (UInt256 × UInt256)) (v30 : (UInt256 × UInt256)) (v36 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v47 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (s3, cb) = adc(t.l3, h2, cb)
  let v53 ← adc t.l3 v42.snd v36.snd
  -- let (l3, h3) = mul_wide(b, self.l3)
  let v59 ← mul_wide b self.l3
  -- let (s3, ca) = adc(s3, l3, ca)
  let v64 ← adc v53.fst v59.fst v47.snd
  -- let (s4, cb) = adc(top, h3, cb)
  let v69 ← adc top v59.snd v53.snd
  -- let (s4, ca) = adc(s4, 0, ca)
  let v75 ← adc v69.fst (UInt256.ofNat 0) v64.snd
  Mont_row_Base_k3 v12 v30 v47 v64 v69 v75

def Mont_row_Base_k1 (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) (v12 : (UInt256 × UInt256)) (v19 : (UInt256 × UInt256)) (v25 : (UInt256 × UInt256)) : FeM (Mont_Base × UInt256) := do
  -- let (s1, ca) = adc(s1, l1, ca)
  let v30 ← adc v19.fst v25.fst v12.snd
  -- let (s2, cb) = adc(t.l2, h1, cb)
  let v36 ← adc t.l2 v25.snd v19.snd
  -- let (l2, h2) = mul_wide(b, self.l2)
  let v42 ← mul_wide b self.l2
  -- let (s2, ca) = adc(s2, l2, ca)
  let v47 ← adc v36.fst v42.fst v30.snd
  Mont_row_Base_k2 self b t top v12 v30 v36 v42 v47

/-- `Mont<Base>::row` -/
def Mont_row_Base (self : Mont_Base) (b : UInt256) (t : Mont_Base) (top : UInt256) : FeM (Mont_Base × UInt256) := do
  -- let (l0, h0) = mul_wide(b, self.l0)
  let v5 ← mul_wide b self.l0
  -- let (s0, ca) = adc(t.l0, l0, 0)
  let v12 ← adc t.l0 v5.fst (UInt256.ofNat 0)
  -- let (s1, cb) = adc(t.l1, h0, 0)
  let v19 ← adc t.l1 v5.snd (UInt256.ofNat 0)
  -- let (l1, h1) = mul_wide(b, self.l1)
  let v25 ← mul_wide b self.l1
  Mont_row_Base_k1 self b t top v12 v19 v25

/-- `lib::montgomery::sbb` -/
def sbb (a : UInt256) (b : UInt256) (borrow : UInt256) : FeM (UInt256 × UInt256) := do
  -- let t = (a as u128).wrapping_sub(b as u128).wrapping_sub(borrow as u128)
  let v7 ← u128_wrapping_sub a b
  let t ← u128_wrapping_sub v7 borrow
  -- (t.downcast_truncate(), (t >> 127).downcast_truncate())
  let v12 ← downcast_truncate_u128_u64 t
  let v15 ← downcast_truncate_u128_u64 (t >>> (UInt256.ofNat 127))
  return (v12, v15)

/-- `Mont<Base>::reduce` -/
def Mont_reduce_Base (self : Mont_Base) (top : UInt256) : FeM Mont_Base := do
  -- let (d0, b) = sbb(self.l0, M::P0, 0)
  let v4 ← Base_P0
  let v6 ← sbb self.l0 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  -- let (d1, b) = sbb(self.l1, M::P1, b)
  let v12 ← Base_P1
  let v13 ← sbb self.l1 v12 v6.snd
  let d1 := v13.fst
  -- let (d2, b) = sbb(self.l2, M::P2, b)
  let v19 ← Base_P2
  let v20 ← sbb self.l2 v19 v13.snd
  let d2 := v20.fst
  -- let (d3, b) = sbb(self.l3, M::P3, b)
  let v26 ← Base_P3
  let v27 ← sbb self.l3 v26 v20.snd
  let d3 := v27.fst
  -- let (_, b) = sbb(top, 0, b)
  let v33 ← sbb top (UInt256.ofNat 0) v27.snd
  let join : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  -- if b == 1 { self } else { Mont { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v33.snd = (UInt256.ofNat 1)) then
    join self
  else
    join (Mont_Base.mk d0 d1 d2 d3)

/-- `<Mont<Base> as Field>::mul` -/
def Mont_mul_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let mut t_r0 : Mont_Base := default
  let mut t_r1 : Mont_Base := default
  let mut t_r2 : Mont_Base := default
  let mut t_r3 : Mont_Base := default
  -- let (t, top) = self.first_row(other.l0)
  let v3 ← Mont_first_row_Base self other.l0
  t_r0 := v3.fst
  -- let (t, top) = self.row(other.l1, t, top)
  let v10 ← Mont_row_Base self other.l1 t_r0 v3.snd
  t_r1 := v10.fst
  -- let (t, top) = self.row(other.l2, t, top)
  let v17 ← Mont_row_Base self other.l2 t_r1 v10.snd
  t_r2 := v17.fst
  -- let (t, top) = self.row(other.l3, t, top)
  let v24 ← Mont_row_Base self other.l3 t_r2 v17.snd
  t_r3 := v24.fst
  -- t.reduce(top)
  let v30 ← Mont_reduce_Base t_r3 v24.snd
  return v30

/-- `Mont<Base>::new` -/
def Mont_new_Base (value : UInt256) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  -- core::assert(value < M::VALUE)
  let v2 ← Base_VALUE
  let _ ← «assert» (decide (value.toNat < v2.toNat))
  -- raw<M>(value).mul(raw<M>(M::R2))
  let v5 ← raw_Base value
  r0 := v5
  let v6 ← Base_R2
  let v7 ← raw_Base v6
  r1 := v7
  let v10 ← Mont_mul_Base r0 r1
  return v10

/-- `<Mont<Base> as Field>::square` -/
def Mont_square_Base (self : Mont_Base) : FeM Mont_Base := do
  -- self.mul(self)
  let v1 ← Mont_mul_Base self self
  return v1

/-- `<Mont<Base> as Field>::add` -/
def Mont_add_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  -- let (s0, c) = adc(self.l0, other.l0, 0)
  let v5 ← adc self.l0 other.l0 (UInt256.ofNat 0)
  -- let (s1, c) = adc(self.l1, other.l1, c)
  let v12 ← adc self.l1 other.l1 v5.snd
  -- let (s2, c) = adc(self.l2, other.l2, c)
  let v19 ← adc self.l2 other.l2 v12.snd
  -- let (s3, c) = adc(self.l3, other.l3, c)
  let v26 ← adc self.l3 other.l3 v19.snd
  -- Mont { l0: s0, l1: s1, l2: s2, l3: s3 }.reduce(c)
  r0 := (Mont_Base.mk v5.fst v12.fst v19.fst v26.fst)
  let v33 ← Mont_reduce_Base r0 v26.snd
  return v33

/-- `<Mont<Base> as Field>::one` -/
def Mont_one_Base : FeM Mont_Base := do
  -- Self::new(1)
  let v1 ← Mont_new_Base (UInt256.ofNat 1)
  return v1

/-- `Mont<Base>::pow` -/
def Mont_pow_Base (self : Mont_Base) (exponent : UInt256) : FeM Mont_Base := do
  let mut result : Mont_Base := default
  let mut bit : UInt256 := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  -- var result = Self::one()
  let result_2 ← Mont_one_Base
  result := result_2
  -- var bit: u64 = 256
  bit := (UInt256.ofNat 256)
  let rec loop (result : Mont_Base) (bit : UInt256) (r2 : Mont_Base) (r3 : Mont_Base) : FeM Mont_Base := do
    let mut result := result
    let mut bit := bit
    let mut r2 := r2
    let mut r3 := r3
    -- while bit > 0 {
    if decide ((UInt256.ofNat 0).toNat < bit.toNat) then
      -- bit -= 1
      let bit_12 ← IntTy.sub (.u 64) bit (UInt256.ofNat 1)
      bit := bit_12
      -- result = result.square()
      r2 := result
      let result_18 ← Mont_square_Base r2
      result := result_18
      -- if (exponent >> bit as u256) & 1 == 1 { result = result.mul(self) }
      let v28 := decide (((exponent >>> bit) &&& (UInt256.ofNat 1)) = (UInt256.ofNat 1))
      let join : Mont_Base → UInt256 → Mont_Base → Mont_Base → FeM Mont_Base := fun result bit r2 r3 => do
        let mut result := result
        let mut bit := bit
        let mut r2 := r2
        let mut r3 := r3
        loop result bit r2 r3
      -- if (exponent >> bit as u256) & 1 == 1 { result = result.mul(self) }
      if v28 then
        -- result = result.mul(self)
        r3 := result
        let result_32 ← Mont_mul_Base r3 self
        result := result_32
        join result bit r2 r3
      else
        join result bit r2 r3
    else
      return result
  partial_fixpoint
  loop result bit r2 r3

/-- `<Mont<Base> as Field>::equals` -/
def Mont_equals_Base (self : Mont_Base) (other : Mont_Base) : FeM Bool := do
  let join : Bool → FeM Bool := fun v2 => do
    return v2
  let join_1 : FeM Bool := do
    join false
  -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
  if decide (self.l0 = other.l0) then
    -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
    if decide (self.l1 = other.l1) then
      -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
      if decide (self.l2 = other.l2) then
        -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
        if decide (self.l3 = other.l3) then
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
  -- let value = self.mul(raw<M>(1))
  let v2 ← raw_Base (UInt256.ofNat 1)
  r0 := v2
  let value ← Mont_mul_Base self r0
  -- value.l0 as u256 | (value.l1 as u256) << 64 | (value.l2 as u256) << 128 | (value.l3 as u256) << 192
  let v17 := (value.l0 ||| (value.l1 <<< (UInt256.ofNat 64))) ||| (value.l2 <<< (UInt256.ofNat 128))
  return v17 ||| (value.l3 <<< (UInt256.ofNat 192))

/-- `<Order as Modulus>::VALUE` -/
def Order_VALUE : FeM UInt256 := do
  let v0 ← N
  return v0

/-- `lib::montgomery::raw<Order>` -/
def raw_Order (value : UInt256) : FeM Mont_Order := do
  -- Mont {
  let v1 ← downcast_truncate_u256_u64 value
  let v5 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 64))
  let v9 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 128))
  let v13 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 192))
  return Mont_Order.mk v1 v5 v9 v13

/-- `<Order as Modulus>::R2` -/
def Order_R2 : FeM UInt256 := do
  return UInt256.ofNat 71195301480278335217902614543643724933430614355449737089222010364394701574464

/-- `<Order as Modulus>::INV` -/
def Order_INV : FeM UInt256 := do
  return UInt256.ofNat 5408259542528602431

/-- `<Order as Modulus>::P0` -/
def Order_P0 : FeM UInt256 := do
  return UInt256.ofNat 13822214165235122497

/-- `<Order as Modulus>::P1` -/
def Order_P1 : FeM UInt256 := do
  return UInt256.ofNat 13451932020343611451

/-- `<Order as Modulus>::P2` -/
def Order_P2 : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551614

/-- `<Order as Modulus>::P3` -/
def Order_P3 : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551615

def Mont_shift_reduced_Order_k3 (s5 : UInt256) (v32 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) (v64 : (UInt256 × UInt256)) (v69 : (UInt256 × UInt256)) (v75 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (u5, _) = adc(s5, cb, ca)
  let v80 ← adc s5 v69.snd v75.snd
  -- (Mont { l0: u1, l1: u2, l2: u3, l3: u4 }, u5)
  let v84 := ((Mont_Order.mk v32.fst v48.fst v64.fst v75.fst), v80.fst)
  return v84

def Mont_shift_reduced_Order_k2 (s4 : UInt256) (s5 : UInt256) (m : UInt256) (v32 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) (v53 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (k3, j3) = mul_wide(m, M::P3)
  let v58 ← Order_P3
  let v59 ← mul_wide m v58
  -- let (u3, ca) = adc(u3, k3, ca)
  let v64 ← adc v53.fst v59.fst v48.snd
  -- let (u4, cb) = adc(s4, j3, cb)
  let v69 ← adc s4 v59.snd v53.snd
  -- let (u4, ca) = adc(u4, 0, ca)
  let v75 ← adc v69.fst (UInt256.ofNat 0) v64.snd
  Mont_shift_reduced_Order_k3 s5 v32 v48 v64 v69 v75

def Mont_shift_reduced_Order_k1 (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (v17 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (v27 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (u1, ca) = adc(u1, k1, ca)
  let v32 ← adc v21.fst v27.fst v17.snd
  -- let (u2, cb) = adc(s2, j1, cb)
  let v37 ← adc s2 v27.snd v21.snd
  -- let (k2, j2) = mul_wide(m, M::P2)
  let v42 ← Order_P2
  let v43 ← mul_wide m v42
  -- let (u2, ca) = adc(u2, k2, ca)
  let v48 ← adc v37.fst v43.fst v32.snd
  -- let (u3, cb) = adc(s3, j2, cb)
  let v53 ← adc s3 v43.snd v37.snd
  Mont_shift_reduced_Order_k2 s4 s5 m v32 v48 v53

/-- `Mont<Order>::shift_reduced` -/
def Mont_shift_reduced_Order (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Order × UInt256) := do
  -- let m = s0.wrapping_mul(M::INV)
  let v6 ← Order_INV
  let m ← u64_wrapping_mul s0 v6
  -- let (k0, j0) = mul_wide(m, M::P0)
  let v10 ← Order_P0
  let v11 ← mul_wide m v10
  -- let (_, ca) = adc(s0, k0, 0)
  let v17 ← adc s0 v11.fst (UInt256.ofNat 0)
  -- let (u1, cb) = adc(s1, j0, 0)
  let v21 ← adc s1 v11.snd (UInt256.ofNat 0)
  -- let (k1, j1) = mul_wide(m, M::P1)
  let v26 ← Order_P1
  let v27 ← mul_wide m v26
  Mont_shift_reduced_Order_k1 s2 s3 s4 s5 m v17 v21 v27

def Mont_first_row_Order_k1 (self : Mont_Order) (b : UInt256) (v3 : (UInt256 × UInt256)) (v9 : (UInt256 × UInt256)) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (s2, c) = adc(h1, l2, c)
  let v26 ← adc v9.snd v21.fst v15.snd
  -- let (l3, h3) = mul_wide(b, self.l3)
  let v32 ← mul_wide b self.l3
  -- let (s3, c) = adc(h2, l3, c)
  let v37 ← adc v21.snd v32.fst v26.snd
  -- let (s4, _) = adc(h3, 0, c)
  let v43 ← adc v32.snd (UInt256.ofNat 0) v37.snd
  -- Self::shift_reduced(s0, s1, s2, s3, s4, 0)
  let v47 ← Mont_shift_reduced_Order v3.fst v15.fst v26.fst v37.fst v43.fst (UInt256.ofNat 0)
  return v47

/-- `Mont<Order>::first_row` -/
def Mont_first_row_Order (self : Mont_Order) (b : UInt256) : FeM (Mont_Order × UInt256) := do
  -- let (s0, h0) = mul_wide(b, self.l0)
  let v3 ← mul_wide b self.l0
  -- let (l1, h1) = mul_wide(b, self.l1)
  let v9 ← mul_wide b self.l1
  -- let (s1, c) = adc(h0, l1, 0)
  let v15 ← adc v3.snd v9.fst (UInt256.ofNat 0)
  -- let (l2, h2) = mul_wide(b, self.l2)
  let v21 ← mul_wide b self.l2
  Mont_first_row_Order_k1 self b v3 v9 v15 v21

def Mont_row_Order_k3 (v12 : (UInt256 × UInt256)) (v30 : (UInt256 × UInt256)) (v47 : (UInt256 × UInt256)) (v64 : (UInt256 × UInt256)) (v69 : (UInt256 × UInt256)) (v75 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (s5, _) = adc(cb, 0, ca)
  let v81 ← adc v69.snd (UInt256.ofNat 0) v75.snd
  -- Self::shift_reduced(s0, s1, s2, s3, s4, s5)
  let v84 ← Mont_shift_reduced_Order v12.fst v30.fst v47.fst v64.fst v75.fst v81.fst
  return v84

def Mont_row_Order_k2 (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) (v12 : (UInt256 × UInt256)) (v30 : (UInt256 × UInt256)) (v36 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v47 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (s3, cb) = adc(t.l3, h2, cb)
  let v53 ← adc t.l3 v42.snd v36.snd
  -- let (l3, h3) = mul_wide(b, self.l3)
  let v59 ← mul_wide b self.l3
  -- let (s3, ca) = adc(s3, l3, ca)
  let v64 ← adc v53.fst v59.fst v47.snd
  -- let (s4, cb) = adc(top, h3, cb)
  let v69 ← adc top v59.snd v53.snd
  -- let (s4, ca) = adc(s4, 0, ca)
  let v75 ← adc v69.fst (UInt256.ofNat 0) v64.snd
  Mont_row_Order_k3 v12 v30 v47 v64 v69 v75

def Mont_row_Order_k1 (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) (v12 : (UInt256 × UInt256)) (v19 : (UInt256 × UInt256)) (v25 : (UInt256 × UInt256)) : FeM (Mont_Order × UInt256) := do
  -- let (s1, ca) = adc(s1, l1, ca)
  let v30 ← adc v19.fst v25.fst v12.snd
  -- let (s2, cb) = adc(t.l2, h1, cb)
  let v36 ← adc t.l2 v25.snd v19.snd
  -- let (l2, h2) = mul_wide(b, self.l2)
  let v42 ← mul_wide b self.l2
  -- let (s2, ca) = adc(s2, l2, ca)
  let v47 ← adc v36.fst v42.fst v30.snd
  Mont_row_Order_k2 self b t top v12 v30 v36 v42 v47

/-- `Mont<Order>::row` -/
def Mont_row_Order (self : Mont_Order) (b : UInt256) (t : Mont_Order) (top : UInt256) : FeM (Mont_Order × UInt256) := do
  -- let (l0, h0) = mul_wide(b, self.l0)
  let v5 ← mul_wide b self.l0
  -- let (s0, ca) = adc(t.l0, l0, 0)
  let v12 ← adc t.l0 v5.fst (UInt256.ofNat 0)
  -- let (s1, cb) = adc(t.l1, h0, 0)
  let v19 ← adc t.l1 v5.snd (UInt256.ofNat 0)
  -- let (l1, h1) = mul_wide(b, self.l1)
  let v25 ← mul_wide b self.l1
  Mont_row_Order_k1 self b t top v12 v19 v25

/-- `Mont<Order>::reduce` -/
def Mont_reduce_Order (self : Mont_Order) (top : UInt256) : FeM Mont_Order := do
  -- let (d0, b) = sbb(self.l0, M::P0, 0)
  let v4 ← Order_P0
  let v6 ← sbb self.l0 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  -- let (d1, b) = sbb(self.l1, M::P1, b)
  let v12 ← Order_P1
  let v13 ← sbb self.l1 v12 v6.snd
  let d1 := v13.fst
  -- let (d2, b) = sbb(self.l2, M::P2, b)
  let v19 ← Order_P2
  let v20 ← sbb self.l2 v19 v13.snd
  let d2 := v20.fst
  -- let (d3, b) = sbb(self.l3, M::P3, b)
  let v26 ← Order_P3
  let v27 ← sbb self.l3 v26 v20.snd
  let d3 := v27.fst
  -- let (_, b) = sbb(top, 0, b)
  let v33 ← sbb top (UInt256.ofNat 0) v27.snd
  let join : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  -- if b == 1 { self } else { Mont { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v33.snd = (UInt256.ofNat 1)) then
    join self
  else
    join (Mont_Order.mk d0 d1 d2 d3)

/-- `<Mont<Order> as Field>::mul` -/
def Mont_mul_Order (self : Mont_Order) (other : Mont_Order) : FeM Mont_Order := do
  let mut t_r0 : Mont_Order := default
  let mut t_r1 : Mont_Order := default
  let mut t_r2 : Mont_Order := default
  let mut t_r3 : Mont_Order := default
  -- let (t, top) = self.first_row(other.l0)
  let v3 ← Mont_first_row_Order self other.l0
  t_r0 := v3.fst
  -- let (t, top) = self.row(other.l1, t, top)
  let v10 ← Mont_row_Order self other.l1 t_r0 v3.snd
  t_r1 := v10.fst
  -- let (t, top) = self.row(other.l2, t, top)
  let v17 ← Mont_row_Order self other.l2 t_r1 v10.snd
  t_r2 := v17.fst
  -- let (t, top) = self.row(other.l3, t, top)
  let v24 ← Mont_row_Order self other.l3 t_r2 v17.snd
  t_r3 := v24.fst
  -- t.reduce(top)
  let v30 ← Mont_reduce_Order t_r3 v24.snd
  return v30

/-- `Mont<Order>::new` -/
def Mont_new_Order (value : UInt256) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  let mut r1 : Mont_Order := default
  -- core::assert(value < M::VALUE)
  let v2 ← Order_VALUE
  let _ ← «assert» (decide (value.toNat < v2.toNat))
  -- raw<M>(value).mul(raw<M>(M::R2))
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
  return UInt256.ofNat 3815112494326173377

/-- `lib::inversion::M62` -/
def M62 : FeM UInt256 := do
  return UInt256.ofNat 4611686018427387903

/-- `lib::inversion::is_zero<5>` -/
def is_zero_5 (r : (Vector UInt256 5)) : FeM Bool := do
  let mut i : UInt256 := default
  -- var i: usize = 0
  i := (UInt256.ofNat 0)
  let rec loop (i : UInt256) : FeM Bool := do
    let mut i := i
    -- while i < L {
    if decide (i.toNat < (UInt256.ofNat 5).toNat) then
      -- if r[i] != 0 { return false }
      let i_7 := i
      checkIndex i_7 5
      if decide (r[i_7.toNat]! ≠ (UInt256.ofNat 0)) then
        return false
      else
        -- i += 1
        let i_16 ← checkedAdd i (UInt256.ofNat 1)
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
  -- var u: u64 = 1
  u := (UInt256.ofNat 1)
  -- var v: u64 = 0
  v := (UInt256.ofNat 0)
  -- var q: u64 = 0
  q := (UInt256.ofNat 0)
  -- var r: u64 = 1
  r := (UInt256.ofNat 1)
  -- var f = f0
  f := f0_1
  -- var g = g0
  g := g0
  -- var eta = eta
  eta := eta_0
  -- var left: u64 = 62
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
    -- let zeros = trailing_zeros(g | (0xffffffffffffffff << left))
    let v25 := IntTy.shl (.u 64) (UInt256.ofNat 18446744073709551615) left
    let zeros := IntTy.trailingZeros (.u 64) (g ||| v25)
    -- g = g >> zeros
    g := (g >>> zeros)
    -- u = u << zeros
    u := (IntTy.shl (.u 64) u zeros)
    -- v = v << zeros
    v := (IntTy.shl (.u 64) v zeros)
    -- eta -= zeros.downcast_truncate()
    let v44 ← downcast_truncate_u64_i64 zeros
    eta := (IntTy.subW (.i 64) eta v44)
    -- left -= zeros
    left := (IntTy.subW (.u 64) left zeros)
    -- if left == 0 { break }
    if decide (left = (UInt256.ofNat 0)) then
      -- (eta, Transition { u: u.downcast_truncate(), v: v.downcast_truncate(), q: q.downcast_truncate(), r: r.downcast_truncate() })
      let eta_57 := eta
      let v61 ← downcast_truncate_u64_i64 u
      let v64 ← downcast_truncate_u64_i64 v
      let v67 ← downcast_truncate_u64_i64 q
      let v70 ← downcast_truncate_u64_i64 r
      return (eta_57, (Transition.mk v61 v64 v67 v70))
    else
      -- var bits: u64 = 15
      bits := (UInt256.ofNat 15)
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
        -- let room: u64 = (eta + 1).downcast_truncate()
        let room ← downcast_truncate_i64_u64 (IntTy.addW (.i 64) eta (UInt256.ofNat 1))
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
          -- let mask = (0xffffffffffffffff >> (64 - limit)) & bits
          let v138 := (UInt256.ofNat 18446744073709551615) >>> (IntTy.subW (.u 64) (UInt256.ofNat 64) limit)
          let mask := v138 &&& bits
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
            -- g += f * w
            g := (IntTy.addW (.u 64) g (IntTy.mulW (.u 64) f w))
            -- q += u * w
            q := (IntTy.addW (.u 64) q (IntTy.mulW (.u 64) u w))
            -- r += v * w
            r := (IntTy.addW (.u 64) r (IntTy.mulW (.u 64) v w))
            loop u v q r f g eta left bits
          -- let w = if bits == 63 {
          if decide (bits = (UInt256.ofNat 63)) then
            -- (f * g * (f * f - 2)) & mask
            let v155 := IntTy.subW (.u 64) (IntTy.mulW (.u 64) f f) (UInt256.ofNat 2)
            let v157 := (IntTy.mulW (.u 64) (IntTy.mulW (.u 64) f g) v155) &&& mask
            join_2 u v q r f g eta left bits v157
          else
            -- ((0 - (f + (((f + 1) & 4) << 1))) * g) & mask
            let v167 := (IntTy.addW (.u 64) f (UInt256.ofNat 1)) &&& (UInt256.ofNat 4)
            let v170 := IntTy.addW (.u 64) f (IntTy.shl (.u 64) v167 (UInt256.ofNat 1))
            let v173 := IntTy.mulW (.u 64) (IntTy.subW (.u 64) (UInt256.ofNat 0) v170) g
            join_2 u v q r f g eta left bits (v173 &&& mask)
        -- let limit = if room < left { room } else { left }
        if decide (room.toNat < left.toNat) then
          join_1 u v q r f g eta left bits room
        else
          join_1 u v q r f g eta left bits left
      -- if eta < 0 {
      if UInt256.sltBool eta (UInt256.ofNat 0) then
        -- eta = 0 - eta
        eta := (IntTy.subW (.i 64) (UInt256.ofNat 0) eta)
        -- let f_old = f
        let f_86 := f
        -- f = g
        f := g
        -- g = 0 - f_old
        g := (IntTy.subW (.u 64) (UInt256.ofNat 0) f_86)
        -- let u_old = u
        let u_96 := u
        -- u = q
        u := q
        -- q = 0 - u_old
        q := (IntTy.subW (.u 64) (UInt256.ofNat 0) u_96)
        -- let v_old = v
        let v_106 := v
        -- v = r
        v := r
        -- r = 0 - v_old
        r := (IntTy.subW (.u 64) (UInt256.ofNat 0) v_106)
        -- bits = 63
        bits := (UInt256.ofNat 63)
        join u v q r f g eta left bits
      else
        join u v q r f g eta left bits
  partial_fixpoint
  loop u v q r f g eta left bits

/-- `lib::inversion::M62_SIGNED` -/
def M62_SIGNED : FeM UInt256 := do
  return UInt256.ofNat 4611686018427387903

def update_de_5_k6 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  -- var next_d: [i64; L] = [0; L]
  next_d := (Vector.replicate 5 (UInt256.ofNat 0))
  -- var next_e: [i64; L] = [0; L]
  next_e := (Vector.replicate 5 (UInt256.ofNat 0))
  -- var i: usize = 1
  i := (UInt256.ofNat 1)
  let rec loop (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut md := md
    let mut me := me
    let mut cd := cd
    let mut ce := ce
    let mut next_d := next_d
    let mut next_e := next_e
    let mut i := i
    -- while i < L {
    if decide (i.toNat < (UInt256.ofNat 5).toNat) then
      -- cd += (t.u as i128) * (d[i] as i128) + (t.v as i128) * (e[i] as i128) + (modulus[i] as i128) * (md as i128)
      let i_143 := i
      checkIndex i_143 5
      let i_149 := i
      checkIndex i_149 5
      let v153 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.u d[i_143.toNat]!) (IntTy.mulW (.i 128) t.v e[i_149.toNat]!)
      let i_154 := i
      checkIndex i_154 5
      let v160 := IntTy.addW (.i 128) v153 (IntTy.mulW (.i 128) modulus[i_154.toNat]! md)
      cd := (IntTy.addW (.i 128) cd v160)
      -- ce += (t.q as i128) * (d[i] as i128) + (t.r as i128) * (e[i] as i128) + (modulus[i] as i128) * (me as i128)
      let i_167 := i
      checkIndex i_167 5
      let i_173 := i
      checkIndex i_173 5
      let v177 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.q d[i_167.toNat]!) (IntTy.mulW (.i 128) t.r e[i_173.toNat]!)
      let i_178 := i
      checkIndex i_178 5
      let v184 := IntTy.addW (.i 128) v177 (IntTy.mulW (.i 128) modulus[i_178.toNat]! me)
      ce := (IntTy.addW (.i 128) ce v184)
      -- let limb_d: i64 = cd.downcast_truncate()
      let limb_d ← downcast_truncate_i128_i64 cd
      -- let limb_e: i64 = ce.downcast_truncate()
      let limb_e ← downcast_truncate_i128_i64 ce
      -- next_d[i - 1] = limb_d & M62_SIGNED
      let v197 ← M62_SIGNED
      let v202 := i - (UInt256.ofNat 1)
      checkIndex v202 5
      next_d := (next_d.set! v202.toNat (limb_d &&& v197))
      -- next_e[i - 1] = limb_e & M62_SIGNED
      let v204 ← M62_SIGNED
      let v209 := i - (UInt256.ofNat 1)
      checkIndex v209 5
      next_e := (next_e.set! v209.toNat (limb_e &&& v204))
      -- cd = cd >> 62
      cd := (IntTy.shr (.i 128) cd (UInt256.ofNat 62))
      -- ce = ce >> 62
      ce := (IntTy.shr (.i 128) ce (UInt256.ofNat 62))
      -- i += 1
      i := (i + (UInt256.ofNat 1))
      loop md me cd ce next_d next_e i
    else
      -- next_d[L - 1] = cd.downcast_truncate()
      let v230 ← downcast_truncate_i128_i64 cd
      let v233 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v233 5
      next_d := (next_d.set! v233.toNat v230)
      -- next_e[L - 1] = ce.downcast_truncate()
      let v237 ← downcast_truncate_i128_i64 ce
      let v240 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v240 5
      next_e := (next_e.set! v240.toNat v237)
      return (next_d, next_e)
  partial_fixpoint
  loop md me cd ce next_d next_e i

def update_de_5_k5 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (v107 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  let ce_114 := IntTy.addW (.i 128) ce (IntTy.mulW (.i 128) modulus[v107.toNat]! me)
  ce := ce_114
  -- cd = cd >> 62
  cd := (IntTy.shr (.i 128) cd (UInt256.ofNat 62))
  -- ce = ce >> 62
  ce := (IntTy.shr (.i 128) ce (UInt256.ofNat 62))
  update_de_5_k6 d e t modulus md me cd ce next_d next_e i

def update_de_5_k4 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (correction_d : UInt256) (correction_e : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  -- md -= correction_d
  md := (IntTy.subW (.i 64) md correction_d)
  -- me -= correction_e
  me := (IntTy.subW (.i 64) me correction_e)
  -- cd += (modulus[0] as i128) * (md as i128)
  let v97 := (UInt256.ofNat 0)
  checkIndex v97 5
  let cd_104 := IntTy.addW (.i 128) cd (IntTy.mulW (.i 128) modulus[v97.toNat]! md)
  cd := cd_104
  -- ce += (modulus[0] as i128) * (me as i128)
  let v107 := (UInt256.ofNat 0)
  checkIndex v107 5
  update_de_5_k5 d e t modulus v107 md me cd ce next_d next_e i

def update_de_5_k3 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (low_d : UInt256) (low_e : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  -- let md_bits: u64 = md.downcast_truncate()
  let md_bits ← downcast_truncate_i64_u64 md
  -- let me_bits: u64 = me.downcast_truncate()
  let me_bits ← downcast_truncate_i64_u64 me
  -- let correction_d: i64 = ((inv62 * low_d + md_bits) & M62).downcast_truncate()
  let v77 := IntTy.addW (.u 64) (IntTy.mulW (.u 64) inv62 low_d) md_bits
  let v78 ← M62
  let correction_d ← downcast_truncate_u64_i64 (v77 &&& v78)
  -- let correction_e: i64 = ((inv62 * low_e + me_bits) & M62).downcast_truncate()
  let v84 := IntTy.addW (.u 64) (IntTy.mulW (.u 64) inv62 low_e) me_bits
  let v85 ← M62
  let correction_e ← downcast_truncate_u64_i64 (v84 &&& v85)
  update_de_5_k4 d e t modulus correction_d correction_e md me cd ce next_d next_e i

def update_de_5_k2 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  -- var ce = (t.q as i128) * (d[0] as i128) + (t.r as i128) * (e[0] as i128)
  let v47 := (UInt256.ofNat 0)
  checkIndex v47 5
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let ce_57 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.q d[v47.toNat]!) (IntTy.mulW (.i 128) t.r e[v53.toNat]!)
  ce := ce_57
  -- let low_d: u64 = cd.downcast_truncate()
  let low_d ← downcast_truncate_i128_u64 cd
  -- let low_e: u64 = ce.downcast_truncate()
  let low_e ← downcast_truncate_i128_u64 ce
  update_de_5_k3 d e t modulus inv62 low_d low_e md me cd ce next_d next_e i

def update_de_5_k1 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (sd : UInt256) (se : UInt256) (md : UInt256) (me : UInt256) (cd : UInt256) (ce : UInt256) (next_d : (Vector UInt256 5)) (next_e : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md := md
  let mut me := me
  let mut cd := cd
  let mut ce := ce
  let mut next_d := next_d
  let mut next_e := next_e
  let mut i := i
  -- var me = (t.q & sd) + (t.r & se)
  me := (IntTy.addW (.i 64) (t.q &&& sd) (t.r &&& se))
  -- var cd = (t.u as i128) * (d[0] as i128) + (t.v as i128) * (e[0] as i128)
  let v33 := (UInt256.ofNat 0)
  checkIndex v33 5
  let v39 := (UInt256.ofNat 0)
  checkIndex v39 5
  let cd_43 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.u d[v33.toNat]!) (IntTy.mulW (.i 128) t.v e[v39.toNat]!)
  cd := cd_43
  update_de_5_k2 d e t modulus inv62 md me cd ce next_d next_e i

/-- `lib::inversion::update_de<5>` -/
def update_de_5 (d : (Vector UInt256 5)) (e : (Vector UInt256 5)) (t : Transition) (modulus : (Vector UInt256 5)) (inv62 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut md : UInt256 := default
  let mut me : UInt256 := default
  let mut cd : UInt256 := default
  let mut ce : UInt256 := default
  let mut next_d : (Vector UInt256 5) := default
  let mut next_e : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  -- let sd = d[L - 1] >> 63
  let v7 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
  checkIndex v7 5
  let sd := IntTy.shr (.i 64) d[v7.toNat]! (UInt256.ofNat 63)
  -- let se = e[L - 1] >> 63
  let v14 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
  checkIndex v14 5
  let se := IntTy.shr (.i 64) e[v14.toNat]! (UInt256.ofNat 63)
  -- var md = (t.u & sd) + (t.v & se)
  md := (IntTy.addW (.i 64) (t.u &&& sd) (t.v &&& se))
  update_de_5_k1 d e t modulus inv62 sd se md me cd ce next_d next_e i

def update_fg_5_k2 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf := cf
  let mut cg := cg
  let mut next_f := next_f
  let mut next_g := next_g
  let mut i := i
  -- var next_f: [i64; L] = [0; L]
  next_f := (Vector.replicate 5 (UInt256.ofNat 0))
  -- var next_g: [i64; L] = [0; L]
  next_g := (Vector.replicate 5 (UInt256.ofNat 0))
  -- var i: usize = 1
  i := (UInt256.ofNat 1)
  let rec loop (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut cf := cf
    let mut cg := cg
    let mut next_f := next_f
    let mut next_g := next_g
    let mut i := i
    -- while i < L {
    if decide (i.toNat < (UInt256.ofNat 5).toNat) then
      -- cf += (t.u as i128) * (f[i] as i128) + (t.v as i128) * (g[i] as i128)
      let i_57 := i
      checkIndex i_57 5
      let i_63 := i
      checkIndex i_63 5
      let v67 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.u f[i_57.toNat]!) (IntTy.mulW (.i 128) t.v g[i_63.toNat]!)
      cf := (IntTy.addW (.i 128) cf v67)
      -- cg += (t.q as i128) * (f[i] as i128) + (t.r as i128) * (g[i] as i128)
      let i_74 := i
      checkIndex i_74 5
      let i_80 := i
      checkIndex i_80 5
      let v84 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.q f[i_74.toNat]!) (IntTy.mulW (.i 128) t.r g[i_80.toNat]!)
      cg := (IntTy.addW (.i 128) cg v84)
      -- let limb_f: i64 = cf.downcast_truncate()
      let limb_f ← downcast_truncate_i128_i64 cf
      -- let limb_g: i64 = cg.downcast_truncate()
      let limb_g ← downcast_truncate_i128_i64 cg
      -- next_f[i - 1] = limb_f & M62_SIGNED
      let v97 ← M62_SIGNED
      let v102 := i - (UInt256.ofNat 1)
      checkIndex v102 5
      next_f := (next_f.set! v102.toNat (limb_f &&& v97))
      -- next_g[i - 1] = limb_g & M62_SIGNED
      let v104 ← M62_SIGNED
      let v109 := i - (UInt256.ofNat 1)
      checkIndex v109 5
      next_g := (next_g.set! v109.toNat (limb_g &&& v104))
      -- cf = cf >> 62
      cf := (IntTy.shr (.i 128) cf (UInt256.ofNat 62))
      -- cg = cg >> 62
      cg := (IntTy.shr (.i 128) cg (UInt256.ofNat 62))
      -- i += 1
      i := (i + (UInt256.ofNat 1))
      loop cf cg next_f next_g i
    else
      -- next_f[L - 1] = cf.downcast_truncate()
      let v130 ← downcast_truncate_i128_i64 cf
      let v133 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v133 5
      next_f := (next_f.set! v133.toNat v130)
      -- next_g[L - 1] = cg.downcast_truncate()
      let v137 ← downcast_truncate_i128_i64 cg
      let v140 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v140 5
      next_g := (next_g.set! v140.toNat v137)
      return (next_f, next_g)
  partial_fixpoint
  loop cf cg next_f next_g i

def update_fg_5_k1 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) (v19 : UInt256) (cf : UInt256) (cg : UInt256) (next_f : (Vector UInt256 5)) (next_g : (Vector UInt256 5)) (i : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf := cf
  let mut cg := cg
  let mut next_f := next_f
  let mut next_g := next_g
  let mut i := i
  let v25 := (UInt256.ofNat 0)
  checkIndex v25 5
  let cg_29 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.q f[v19.toNat]!) (IntTy.mulW (.i 128) t.r g[v25.toNat]!)
  cg := cg_29
  -- cf = cf >> 62
  cf := (IntTy.shr (.i 128) cf (UInt256.ofNat 62))
  -- cg = cg >> 62
  cg := (IntTy.shr (.i 128) cg (UInt256.ofNat 62))
  update_fg_5_k2 f g t cf cg next_f next_g i

/-- `lib::inversion::update_fg<5>` -/
def update_fg_5 (f : (Vector UInt256 5)) (g : (Vector UInt256 5)) (t : Transition) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut cf : UInt256 := default
  let mut cg : UInt256 := default
  let mut next_f : (Vector UInt256 5) := default
  let mut next_g : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  -- var cf = (t.u as i128) * (f[0] as i128) + (t.v as i128) * (g[0] as i128)
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 5
  let v11 := (UInt256.ofNat 0)
  checkIndex v11 5
  let cf_15 := IntTy.addW (.i 128) (IntTy.mulW (.i 128) t.u f[v5.toNat]!) (IntTy.mulW (.i 128) t.v g[v11.toNat]!)
  cf := cf_15
  -- var cg = (t.q as i128) * (f[0] as i128) + (t.r as i128) * (g[0] as i128)
  let v19 := (UInt256.ofNat 0)
  checkIndex v19 5
  update_fg_5_k1 f g t v19 cf cg next_f next_g i

/-- `lib::inversion::add_if_negative<5>` -/
def add_if_negative_5 (r_0 : (Vector UInt256 5)) (modulus : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  -- var r = r
  r := r_0
  -- let negative = r[L - 1] >> 63
  let v6 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
  checkIndex v6 5
  let negative := IntTy.shr (.i 64) r[v6.toNat]! (UInt256.ofNat 63)
  -- var i: usize = 0
  i := (UInt256.ofNat 0)
  let rec loop (r : (Vector UInt256 5)) (i : UInt256) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    -- while i < L {
    if decide (i.toNat < (UInt256.ofNat 5).toNat) then
      -- r[i] += modulus[i] & negative
      let i_17 := i
      checkIndex i_17 5
      let i_20 := i
      checkIndex i_20 5
      let v22 := IntTy.addW (.i 64) r[i_20.toNat]! (modulus[i_17.toNat]! &&& negative)
      let i_23 := i
      checkIndex i_23 5
      r := (r.set! i_23.toNat v22)
      -- i += 1
      i := (i + (UInt256.ofNat 1))
      loop r i
    else
      return r
  partial_fixpoint
  loop r i

/-- `lib::inversion::carry<5>` -/
def carry_5 (r_0 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  -- var r = r
  r := r_0
  -- var i: usize = 0
  i := (UInt256.ofNat 0)
  let rec loop (r : (Vector UInt256 5)) (i : UInt256) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    -- while i + 1 < L {
    let v10 := decide ((i + (UInt256.ofNat 1)).toNat < (UInt256.ofNat 5).toNat)
    if v10 then
      -- r[i + 1] += r[i] >> 62
      let i_11 := i
      checkIndex i_11 5
      let v14 := IntTy.shr (.i 64) r[i_11.toNat]! (UInt256.ofNat 62)
      let v18 := i + (UInt256.ofNat 1)
      checkIndex v18 5
      checkIndex v18 5
      r := (r.set! v18.toNat (IntTy.addW (.i 64) r[v18.toNat]! v14))
      -- r[i] &= M62_SIGNED
      let v22 ← M62_SIGNED
      let i_23 := i
      checkIndex i_23 5
      let i_26 := i
      checkIndex i_26 5
      r := (r.set! i_26.toNat (r[i_23.toNat]! &&& v22))
      -- i += 1
      i := (i + (UInt256.ofNat 1))
      loop r i
    else
      return r
  partial_fixpoint
  loop r i

/-- `lib::inversion::normalize<5>` -/
def normalize_5 (r_0 : (Vector UInt256 5)) (sign : UInt256) (modulus : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r : (Vector UInt256 5) := default
  let mut i : UInt256 := default
  let mut r2 : (Vector UInt256 5) := default
  let mut r3 : (Vector UInt256 5) := default
  let mut r4 : (Vector UInt256 5) := default
  -- var r = add_if_negative(r, modulus)
  let r_3 ← add_if_negative_5 r_0 modulus
  r := r_3
  -- let negate = sign >> 63
  let negate := IntTy.shr (.i 64) sign (UInt256.ofNat 63)
  -- var i: usize = 0
  i := (UInt256.ofNat 0)
  let rec loop (r : (Vector UInt256 5)) (i : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
    let mut r := r
    let mut i := i
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    -- while i < L {
    if decide (i.toNat < (UInt256.ofNat 5).toNat) then
      -- r[i] = (r[i] ^ negate) - negate
      let i_15 := i
      checkIndex i_15 5
      let v18 := IntTy.subW (.i 64) (r[i_15.toNat]! ^^^ negate) negate
      let i_19 := i
      checkIndex i_19 5
      r := (r.set! i_19.toNat v18)
      -- i += 1
      i := (i + (UInt256.ofNat 1))
      loop r i r2 r3 r4
    else
      -- carry(add_if_negative(carry(r), modulus))
      r2 := r
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
  -- var d: [i64; L] = [0; L]
  d := (Vector.replicate 5 (UInt256.ofNat 0))
  -- var e: [i64; L] = [0; L]
  e := (Vector.replicate 5 (UInt256.ofNat 0))
  -- e[0] = 1
  let v10 := (UInt256.ofNat 0)
  checkIndex v10 5
  e := (e.set! v10.toNat (UInt256.ofNat 1))
  -- var f = modulus
  f := modulus
  -- var g = x
  g := x
  -- var eta: i64 = -1
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
    -- while !is_zero(g) {
    r6 := g
    let v21 ← is_zero_5 r6
    if !v21 then
      -- let (next, t) = divsteps(eta, f[0].downcast_truncate(), g[0].downcast_truncate())
      let eta_23 := eta
      let v25 := (UInt256.ofNat 0)
      checkIndex v25 5
      let v27 ← downcast_truncate_i64_u64 f[v25.toNat]!
      let v28 := (UInt256.ofNat 0)
      checkIndex v28 5
      let v30 ← downcast_truncate_i64_u64 g[v28.toNat]!
      let v31 ← divsteps eta_23 v27 v30
      t := v31.snd
      -- eta = next
      eta := v31.fst
      -- let (next_d, next_e) = update_de(d, e, t, modulus, inv62)
      r7 := d
      r8 := e
      let v45 ← update_de_5 r7 r8 t modulus inv62
      -- d = next_d
      d := v45.fst
      -- e = next_e
      e := v45.snd
      -- let (next_f, next_g) = update_fg(f, g, t)
      r9 := f
      r10 := g
      let v61 ← update_fg_5 r9 r10 t
      -- f = next_f
      f := v61.fst
      -- g = next_g
      g := v61.snd
      loop d e f g eta t r6 r7 r8 r9 r10 r11
    else
      -- normalize(d, f[L - 1], modulus)
      r11 := d
      let v74 := (UInt256.ofNat 5) - (UInt256.ofNat 1)
      checkIndex v74 5
      let v75 := f[v74.toNat]!
      let v77 ← normalize_5 r11 v75 modulus
      return v77
  partial_fixpoint
  loop d e f g eta t r6 r7 r8 r9 r10 r11

def invert_256_k5 (w : (Vector UInt256 5)) (v76 : UInt256) (v85 : UInt256) (v94 : UInt256) (v95 : UInt256) (v99 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v103 := (w[v95.toNat]! >>> (UInt256.ofNat 6)) ||| (IntTy.shl (.u 64) w[v99.toNat]! (UInt256.ofNat 56))
  return #v[v76, v85, v94, v103]

def invert_256_k4 (w : (Vector UInt256 5)) (v76 : UInt256) (v77 : UInt256) (v81 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v85 := (w[v77.toNat]! >>> (UInt256.ofNat 2)) ||| (IntTy.shl (.u 64) w[v81.toNat]! (UInt256.ofNat 60))
  let v86 := (UInt256.ofNat 2)
  checkIndex v86 5
  let v90 := (UInt256.ofNat 3)
  checkIndex v90 5
  let v94 := (w[v86.toNat]! >>> (UInt256.ofNat 4)) ||| (IntTy.shl (.u 64) w[v90.toNat]! (UInt256.ofNat 58))
  let v95 := (UInt256.ofNat 3)
  checkIndex v95 5
  let v99 := (UInt256.ofNat 4)
  checkIndex v99 5
  invert_256_k5 w v76 v85 v94 v95 v99 r0

def invert_256_k3 (s : (Vector UInt256 5)) (v55 : UInt256) (v58 : UInt256) (v61 : UInt256) (v62 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v64 ← downcast_truncate_i64_u64 s[v62.toNat]!
  let v65 := (UInt256.ofNat 4)
  checkIndex v65 5
  let v67 ← downcast_truncate_i64_u64 s[v65.toNat]!
  let w := #v[v55, v58, v61, v64, v67]
  -- [w[0] | (w[1] << 62), (w[1] >> 2) | (w[2] << 60), (w[2] >> 4) | (w[3] << 58), (w[3] >> 6) | (w[4] << 56)]
  let v70 := (UInt256.ofNat 0)
  checkIndex v70 5
  let v72 := (UInt256.ofNat 1)
  checkIndex v72 5
  let v76 := w[v70.toNat]! ||| (IntTy.shl (.u 64) w[v72.toNat]! (UInt256.ofNat 62))
  let v77 := (UInt256.ofNat 1)
  checkIndex v77 5
  let v81 := (UInt256.ofNat 2)
  checkIndex v81 5
  invert_256_k4 w v76 v77 v81 r0

def invert_256_k2 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v7 : UInt256) (v19 : UInt256) (v31 : UInt256) (v43 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v44 := (UInt256.ofNat 3)
  checkIndex v44 4
  let v48 ← downcast_truncate_u64_i64 (x[v44.toNat]! >>> (UInt256.ofNat 56))
  r0 := (#v[v7, v19, v31, v43, v48])
  let s ← invert_5 r0 modulus inv62
  -- let w: [u64; 5] = [s[0].downcast_truncate(), s[1].downcast_truncate(), s[2].downcast_truncate(), s[3].downcast_truncate(), s[4].downcast_truncate()]
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let v55 ← downcast_truncate_i64_u64 s[v53.toNat]!
  let v56 := (UInt256.ofNat 1)
  checkIndex v56 5
  let v58 ← downcast_truncate_i64_u64 s[v56.toNat]!
  let v59 := (UInt256.ofNat 2)
  checkIndex v59 5
  let v61 ← downcast_truncate_i64_u64 s[v59.toNat]!
  let v62 := (UInt256.ofNat 3)
  checkIndex v62 5
  invert_256_k3 s v55 v58 v61 v62 r0

def invert_256_k1 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) (v7 : UInt256) (v19 : UInt256) (v20 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v24 := (UInt256.ofNat 2)
  checkIndex v24 4
  let v28 := (x[v20.toNat]! >>> (UInt256.ofNat 60)) ||| (IntTy.shl (.u 64) x[v24.toNat]! (UInt256.ofNat 4))
  let v29 ← M62
  let v31 ← downcast_truncate_u64_i64 (v28 &&& v29)
  let v32 := (UInt256.ofNat 2)
  checkIndex v32 4
  let v36 := (UInt256.ofNat 3)
  checkIndex v36 4
  let v40 := (x[v32.toNat]! >>> (UInt256.ofNat 58)) ||| (IntTy.shl (.u 64) x[v36.toNat]! (UInt256.ofNat 6))
  let v41 ← M62
  let v43 ← downcast_truncate_u64_i64 (v40 &&& v41)
  invert_256_k2 x modulus inv62 v7 v19 v31 v43 r0

/-- `lib::inversion::invert_256` -/
def invert_256 (x : (Vector UInt256 4)) (modulus : (Vector UInt256 5)) (inv62 : UInt256) : FeM (Vector UInt256 4) := do
  let mut r0 : (Vector UInt256 5) := default
  -- let s = invert(
  let v3 := (UInt256.ofNat 0)
  checkIndex v3 4
  let v5 ← M62
  let v7 ← downcast_truncate_u64_i64 (x[v3.toNat]! &&& v5)
  let v8 := (UInt256.ofNat 0)
  checkIndex v8 4
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 4
  let v16 := (x[v8.toNat]! >>> (UInt256.ofNat 62)) ||| (IntTy.shl (.u 64) x[v12.toNat]! (UInt256.ofNat 2))
  let v17 ← M62
  let v19 ← downcast_truncate_u64_i64 (v16 &&& v17)
  let v20 := (UInt256.ofNat 1)
  checkIndex v20 4
  invert_256_k1 x modulus inv62 v7 v19 v20 r0

/-- `<Order as Modulus>::R3` -/
def Order_R3 : FeM UInt256 := do
  return UInt256.ofNat 38611792656279623856209604949727965891812033543773964974832218758873530450413

/-- `<Mont<Order> as Field>::inverse` -/
def Mont_inverse_Order (self : Mont_Order) : FeM Mont_Order := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Order := default
  let mut r3 : Mont_Order := default
  -- let limbs = invert_256([self.l0, self.l1, self.l2, self.l3], M::SIGNED_62, M::INV62)
  r0 := (#v[self.l0, self.l1, self.l2, self.l3])
  let v6 ← Order_SIGNED_62
  r1 := v6
  let v7 ← Order_INV62
  let limbs ← invert_256 r0 r1 v7
  -- Mont { l0: limbs[0], l1: limbs[1], l2: limbs[2], l3: limbs[3] }.mul(raw<M>(M::R3))
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v20 := (Mont_Order.mk limbs[v12.toNat]! limbs[v14.toNat]! limbs[v16.toNat]! limbs[v18.toNat]!)
  r2 := v20
  let v21 ← Order_R3
  let v22 ← raw_Order v21
  r3 := v22
  let v25 ← Mont_mul_Order r2 r3
  return v25

/-- `<Mont<Order> as Field>::zero` -/
def Mont_zero_Order : FeM Mont_Order := do
  -- Mont { l0: 0, l1: 0, l2: 0, l3: 0 }
  let v4 := (Mont_Order.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  return v4

def Mont_sub_Order_k2 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (v27 : (UInt256 × UInt256)) (d3 : UInt256) (s0 : UInt256) (s1 : UInt256) (v46 : (UInt256 × UInt256)) (s2 : UInt256) : FeM Mont_Order := do
  -- let (s3, _) = adc(d3, M::P3, c)
  let v51 ← Order_P3
  let v52 ← adc d3 v51 v46.snd
  let s3 := v52.fst
  let join : Mont_Order → FeM Mont_Order := fun v2 => do
    return v2
  -- if b == 1 { Mont { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Mont { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v27.snd = (UInt256.ofNat 1)) then
    join (Mont_Order.mk s0 s1 s2 s3)
  else
    join (Mont_Order.mk d0 d1 d2 d3)

def Mont_sub_Order_k1 (self : Mont_Order) (other : Mont_Order) (d0 : UInt256) (d1 : UInt256) (v20 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Mont_Order := do
  -- let (d3, b) = sbb(self.l3, other.l3, b)
  let v27 ← sbb self.l3 other.l3 v20.snd
  let d3 := v27.fst
  -- let (s0, c) = adc(d0, M::P0, 0)
  let v32 ← Order_P0
  let v34 ← adc d0 v32 (UInt256.ofNat 0)
  let s0 := v34.fst
  -- let (s1, c) = adc(d1, M::P1, c)
  let v39 ← Order_P1
  let v40 ← adc d1 v39 v34.snd
  let s1 := v40.fst
  -- let (s2, c) = adc(d2, M::P2, c)
  let v45 ← Order_P2
  let v46 ← adc d2 v45 v40.snd
  let s2 := v46.fst
  Mont_sub_Order_k2 d0 d1 d2 v27 d3 s0 s1 v46 s2

/-- `<Mont<Order> as Field>::sub` -/
def Mont_sub_Order (self : Mont_Order) (other : Mont_Order) : FeM Mont_Order := do
  -- let (d0, b) = sbb(self.l0, other.l0, 0)
  let v6 ← sbb self.l0 other.l0 (UInt256.ofNat 0)
  let d0 := v6.fst
  -- let (d1, b) = sbb(self.l1, other.l1, b)
  let v13 ← sbb self.l1 other.l1 v6.snd
  let d1 := v13.fst
  -- let (d2, b) = sbb(self.l2, other.l2, b)
  let v20 ← sbb self.l2 other.l2 v13.snd
  let d2 := v20.fst
  Mont_sub_Order_k1 self other d0 d1 v20 d2

/-- `<Mont<Order> as Field>::neg` -/
def Mont_neg_Order (self : Mont_Order) : FeM Mont_Order := do
  let mut r0 : Mont_Order := default
  -- Self::zero().sub(self)
  let v1 ← Mont_zero_Order
  r0 := v1
  let v3 ← Mont_sub_Order r0 self
  return v3

/-- `Mont<Order>::value` -/
def Mont_value_Order (self : Mont_Order) : FeM UInt256 := do
  let mut r0 : Mont_Order := default
  -- let value = self.mul(raw<M>(1))
  let v2 ← raw_Order (UInt256.ofNat 1)
  r0 := v2
  let value ← Mont_mul_Order self r0
  -- value.l0 as u256 | (value.l1 as u256) << 64 | (value.l2 as u256) << 128 | (value.l3 as u256) << 192
  let v17 := (value.l0 ||| (value.l1 <<< (UInt256.ofNat 64))) ||| (value.l2 <<< (UInt256.ofNat 128))
  return v17 ||| (value.l3 <<< (UInt256.ofNat 192))

/-- `Point<Mont<Base>>::affine` -/
def Point_affine_Mont_Base (x : Mont_Base) (y : Mont_Base) : FeM Point_Mont_Base := do
  -- Point { x, y, z: F::one() }
  let v4 ← Mont_one_Base
  return Point_Mont_Base.mk x y v4

/-- `lib::secp256k1::GX` -/
def GX : FeM UInt256 := do
  return UInt256.ofNat 55066263022277343669578718895168534326250603453777594175500187360389116729240

/-- `lib::secp256k1::GY` -/
def GY : FeM UInt256 := do
  return UInt256.ofNat 32670510020758816978083085130507043184471273380659243275938904335757337482424

/-- `lib::secp256k1::generator` -/
def generator : FeM Point_Mont_Base := do
  let mut r0 : Mont_Base := default
  let mut r1 : Mont_Base := default
  -- Point::affine(x: Fp::new(GX), y: Fp::new(GY))
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
  -- Mont { l0: 0, l1: 0, l2: 0, l3: 0 }
  let v4 := (Mont_Base.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  return v4

/-- `Point<Mont<Base>>::infinity` -/
def Point_infinity_Mont_Base : FeM Point_Mont_Base := do
  -- Point { x: F::one(), y: F::one(), z: F::zero() }
  let v0 ← Mont_one_Base
  let v1 ← Mont_one_Base
  let v2 ← Mont_zero_Base
  return Point_Mont_Base.mk v0 v1 v2

/-- `<Mont<Base> as Field>::is_zero` -/
def Mont_is_zero_Base (self : Mont_Base) : FeM Bool := do
  let join : Bool → FeM Bool := fun v1 => do
    return v1
  let join_1 : FeM Bool := do
    join false
  -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
  if decide (self.l0 = (UInt256.ofNat 0)) then
    -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
    if decide (self.l1 = (UInt256.ofNat 0)) then
      -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
      if decide (self.l2 = (UInt256.ofNat 0)) then
        -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
        if decide (self.l3 = (UInt256.ofNat 0)) then
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
  -- self.z.is_zero()
  r0 := self.z
  let v3 ← Mont_is_zero_Base r0
  return v3

def Mont_sub_Base_k2 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (v27 : (UInt256 × UInt256)) (d3 : UInt256) (s0 : UInt256) (s1 : UInt256) (v46 : (UInt256 × UInt256)) (s2 : UInt256) : FeM Mont_Base := do
  -- let (s3, _) = adc(d3, M::P3, c)
  let v51 ← Base_P3
  let v52 ← adc d3 v51 v46.snd
  let s3 := v52.fst
  let join : Mont_Base → FeM Mont_Base := fun v2 => do
    return v2
  -- if b == 1 { Mont { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Mont { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v27.snd = (UInt256.ofNat 1)) then
    join (Mont_Base.mk s0 s1 s2 s3)
  else
    join (Mont_Base.mk d0 d1 d2 d3)

def Mont_sub_Base_k1 (self : Mont_Base) (other : Mont_Base) (d0 : UInt256) (d1 : UInt256) (v20 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Mont_Base := do
  -- let (d3, b) = sbb(self.l3, other.l3, b)
  let v27 ← sbb self.l3 other.l3 v20.snd
  let d3 := v27.fst
  -- let (s0, c) = adc(d0, M::P0, 0)
  let v32 ← Base_P0
  let v34 ← adc d0 v32 (UInt256.ofNat 0)
  let s0 := v34.fst
  -- let (s1, c) = adc(d1, M::P1, c)
  let v39 ← Base_P1
  let v40 ← adc d1 v39 v34.snd
  let s1 := v40.fst
  -- let (s2, c) = adc(d2, M::P2, c)
  let v45 ← Base_P2
  let v46 ← adc d2 v45 v40.snd
  let s2 := v46.fst
  Mont_sub_Base_k2 d0 d1 d2 v27 d3 s0 s1 v46 s2

/-- `<Mont<Base> as Field>::sub` -/
def Mont_sub_Base (self : Mont_Base) (other : Mont_Base) : FeM Mont_Base := do
  -- let (d0, b) = sbb(self.l0, other.l0, 0)
  let v6 ← sbb self.l0 other.l0 (UInt256.ofNat 0)
  let d0 := v6.fst
  -- let (d1, b) = sbb(self.l1, other.l1, b)
  let v13 ← sbb self.l1 other.l1 v6.snd
  let d1 := v13.fst
  -- let (d2, b) = sbb(self.l2, other.l2, b)
  let v20 ← sbb self.l2 other.l2 v13.snd
  let d2 := v20.fst
  Mont_sub_Base_k1 self other d0 d1 v20 d2

/-- `<Mont<Base> as Field>::double` -/
def Mont_double_Base (self : Mont_Base) : FeM Mont_Base := do
  -- self.add(self)
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
  -- if self.is_infinity() || self.y.is_zero() { return Self::infinity() }
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
    -- return Self::infinity()
    let v72 ← Point_infinity_Mont_Base
    return v72
  -- if self.is_infinity() || self.y.is_zero() { return Self::infinity() }
  if v1 then
    join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    -- if self.is_infinity() || self.y.is_zero() { return Self::infinity() }
    r6 := self.y
    let v4 ← Mont_is_zero_Base r6
    if v4 then
      join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      -- let a = self.x.square()
      r7 := self.x
      let a_9 ← Mont_square_Base r7
      a := a_9
      -- let b = self.y.square()
      r8 := self.y
      let b_13 ← Mont_square_Base r8
      b := b_13
      -- let c = b.square()
      let c_16 ← Mont_square_Base b
      c := c_16
      -- let d = self.x.add(b).square().sub(a).sub(c).double()
      r9 := self.x
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
      -- let e = a.double().add(a)
      let v34 ← Mont_double_Base a
      r14 := v34
      let e_37 ← Mont_add_Base r14 a
      e := e_37
      -- let x = e.square().sub(d.double())
      let v40 ← Mont_square_Base e
      r15 := v40
      let v42 ← Mont_double_Base d
      r16 := v42
      let x_45 ← Mont_sub_Base r15 r16
      x := x_45
      -- let y = e.mul(d.sub(x)).sub(c.double().double().double())
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
      -- Point { x, y, z: self.y.mul(self.z).double() }
      r22 := self.z
      r23 := self.y
      let v67 ← Mont_mul_Base r23 r22
      r24 := v67
      let v69 ← Mont_double_Base r24
      return Point_Mont_Base.mk x y v69

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
  -- if self.is_infinity() { return other }
  let v3 ← Point_is_infinity_Mont_Base self
  if v3 then
    return other
  else
    -- if other.is_infinity() { return self }
    let v7 ← Point_is_infinity_Mont_Base other
    if v7 then
      return self
    else
      -- let z1z1 = self.z.square()
      r12 := self.z
      let z1z1_13 ← Mont_square_Base r12
      z1z1 := z1z1_13
      -- let z2z2 = other.z.square()
      r13 := other.z
      let z2z2_17 ← Mont_square_Base r13
      z2z2 := z2z2_17
      -- let u1 = self.x.mul(z2z2)
      r14 := self.x
      let u1_22 ← Mont_mul_Base r14 z2z2
      u1 := u1_22
      -- let u2 = other.x.mul(z1z1)
      r15 := other.x
      let u2_27 ← Mont_mul_Base r15 z1z1
      u2 := u2_27
      -- let s1 = self.y.mul(other.z).mul(z2z2)
      r16 := other.z
      r17 := self.y
      let v33 ← Mont_mul_Base r17 r16
      r18 := v33
      let s1_36 ← Mont_mul_Base r18 z2z2
      s1 := s1_36
      -- let s2 = other.y.mul(self.z).mul(z1z1)
      r19 := self.z
      r20 := other.y
      let v42 ← Mont_mul_Base r20 r19
      r21 := v42
      let s2_45 ← Mont_mul_Base r21 z1z1
      s2 := s2_45
      -- let h = u2.sub(u1)
      let h_49 ← Mont_sub_Base u2 u1
      h := h_49
      -- let r = s2.sub(s1).double()
      let v53 ← Mont_sub_Base s2 s1
      r22 := v53
      let r_55 ← Mont_double_Base r22
      r := r_55
      -- if h.is_zero() { return if r.is_zero() { self.double() } else { Self::infinity() } }
      let v58 ← Mont_is_zero_Base h
      if v58 then
        -- return if r.is_zero() { self.double() } else { Self::infinity() }
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
        -- return if r.is_zero() { self.double() } else { Self::infinity() }
        if v60 then
          -- self.double()
          let v61 ← Point_double_Mont_Base self
          join z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v61
        else
          -- Self::infinity()
          let v63 ← Point_infinity_Mont_Base
          join z1z1 z2z2 u1 u2 s1 s2 h r i j v x r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v63
      else
        -- let i = h.double().square()
        let v68 ← Mont_double_Base h
        r23 := v68
        let i_70 ← Mont_square_Base r23
        i := i_70
        -- let j = h.mul(i)
        let j_74 ← Mont_mul_Base h i
        j := j_74
        -- let v = u1.mul(i)
        let v_78 ← Mont_mul_Base u1 i
        v := v_78
        -- let x = r.square().sub(j).sub(v.double())
        let v81 ← Mont_square_Base r
        r24 := v81
        let v84 ← Mont_sub_Base r24 j
        r25 := v84
        let v86 ← Mont_double_Base v
        r26 := v86
        let x_89 ← Mont_sub_Base r25 r26
        x := x_89
        -- let y = r.mul(v.sub(x)).sub(s1.mul(j).double())
        let v93 ← Mont_sub_Base v x
        r27 := v93
        let v96 ← Mont_mul_Base r r27
        r28 := v96
        let v99 ← Mont_mul_Base s1 j
        r29 := v99
        let v101 ← Mont_double_Base r29
        r30 := v101
        let y ← Mont_sub_Base r28 r30
        -- Point { x, y, z: self.z.add(other.z).square().sub(z1z1).sub(z2z2).mul(h) }
        r31 := other.z
        r32 := self.z
        let v110 ← Mont_add_Base r32 r31
        r33 := v110
        let v112 ← Mont_square_Base r33
        r34 := v112
        let v115 ← Mont_sub_Base r34 z1z1
        r35 := v115
        let v118 ← Mont_sub_Base r35 z2z2
        r36 := v118
        let v121 ← Mont_mul_Base r36 h
        return Point_Mont_Base.mk x y v121

/-- `<Base as Modulus>::SIGNED_62` -/
def Base_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v5 := #v[(UInt256.ofNat 4611686014132419631), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 4611686018427387903), (UInt256.ofNat 255)]
  return v5

/-- `<Base as Modulus>::INV62` -/
def Base_INV62 : FeM UInt256 := do
  return UInt256.ofNat 2866531139136965327

/-- `<Base as Modulus>::R3` -/
def Base_R3 : FeM UInt256 := do
  return UInt256.ofNat 79228216581683516649686562417

/-- `<Mont<Base> as Field>::inverse` -/
def Mont_inverse_Base (self : Mont_Base) : FeM Mont_Base := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  -- let limbs = invert_256([self.l0, self.l1, self.l2, self.l3], M::SIGNED_62, M::INV62)
  r0 := (#v[self.l0, self.l1, self.l2, self.l3])
  let v6 ← Base_SIGNED_62
  r1 := v6
  let v7 ← Base_INV62
  let limbs ← invert_256 r0 r1 v7
  -- Mont { l0: limbs[0], l1: limbs[1], l2: limbs[2], l3: limbs[3] }.mul(raw<M>(M::R3))
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v20 := (Mont_Base.mk limbs[v12.toNat]! limbs[v14.toNat]! limbs[v16.toNat]! limbs[v18.toNat]!)
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
  -- if self.is_infinity() { return Point::affine(x: x2, y: y2) }
  let v4 ← Point_is_infinity_Mont_Base self
  if v4 then
    -- return Point::affine(x: x2, y: y2)
    let v5 ← Point_affine_Mont_Base x2 y2
    return v5
  else
    -- let z1z1 = self.z.square()
    r10 := self.z
    let z1z1_10 ← Mont_square_Base r10
    z1z1 := z1z1_10
    -- let u2 = x2.mul(z1z1)
    let u2_13 ← Mont_mul_Base x2 z1z1
    u2 := u2_13
    -- let s2 = y2.mul(self.z).mul(z1z1)
    r11 := self.z
    let v17 ← Mont_mul_Base y2 r11
    r12 := v17
    let s2_20 ← Mont_mul_Base r12 z1z1
    s2 := s2_20
    -- let h = u2.sub(self.x)
    r13 := self.x
    let h_25 ← Mont_sub_Base u2 r13
    h := h_25
    -- let r = s2.sub(self.y).double()
    r14 := self.y
    let v30 ← Mont_sub_Base s2 r14
    r15 := v30
    let r_32 ← Mont_double_Base r15
    r := r_32
    -- if h.is_zero() { return if r.is_zero() { self.double() } else { Self::infinity() } }
    let v35 ← Mont_is_zero_Base h
    if v35 then
      -- return if r.is_zero() { self.double() } else { Self::infinity() }
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
      -- return if r.is_zero() { self.double() } else { Self::infinity() }
      if v37 then
        -- self.double()
        let v38 ← Point_double_Mont_Base self
        join z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v38
      else
        -- Self::infinity()
        let v40 ← Point_infinity_Mont_Base
        join z1z1 u2 s2 h r hh i j v x r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 v40
    else
      -- let hh = h.square()
      let hh_45 ← Mont_square_Base h
      hh := hh_45
      -- let i = hh.double().double()
      let v48 ← Mont_double_Base hh
      r16 := v48
      let i_50 ← Mont_double_Base r16
      i := i_50
      -- let j = h.mul(i)
      let j_54 ← Mont_mul_Base h i
      j := j_54
      -- let v = self.x.mul(i)
      r17 := self.x
      let v_59 ← Mont_mul_Base r17 i
      v := v_59
      -- let x = r.square().sub(j).sub(v.double())
      let v62 ← Mont_square_Base r
      r18 := v62
      let v65 ← Mont_sub_Base r18 j
      r19 := v65
      let v67 ← Mont_double_Base v
      r20 := v67
      let x_70 ← Mont_sub_Base r19 r20
      x := x_70
      -- let y = r.mul(v.sub(x)).sub(self.y.mul(j).double())
      let v74 ← Mont_sub_Base v x
      r21 := v74
      let v77 ← Mont_mul_Base r r21
      r22 := v77
      r23 := self.y
      let v81 ← Mont_mul_Base r23 j
      r24 := v81
      let v83 ← Mont_double_Base r24
      r25 := v83
      let y ← Mont_sub_Base r22 r25
      -- Point { x, y, z: self.z.add(h).square().sub(z1z1).sub(hh) }
      r26 := self.z
      let v91 ← Mont_add_Base r26 h
      r27 := v91
      let v93 ← Mont_square_Base r27
      r28 := v93
      let v96 ← Mont_sub_Base r28 z1z1
      r29 := v96
      let v99 ← Mont_sub_Base r29 hh
      return Point_Mont_Base.mk x y v99

def combination_k2 (u : UInt256) (v_1 : UInt256) (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
  -- for i in [4 as usize, 8, 12] {
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
    -- for i in [4 as usize, 8, 12] {
    let v46 ← array_has_usize_3 r12 v3
    if v46 then
      -- for i in [4 as usize, 8, 12] {
      let v48 ← array_at_ref_usize_3 r12 v3
      let i := v48
      -- for j in 1..4 { table[i + j] = table[i].add(table[j]) }
      r13 := (Range_Known_1_Known_4.mk () ())
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
        -- for j in 1..4 { table[i + j] = table[i].add(table[j]) }
        let v58 ← Range_has_1_4 r13 v4
        if v58 then
          -- for j in 1..4 { table[i + j] = table[i].add(table[j]) }
          let v60 ← Range_at_1_4 r13 v4
          let j := v60
          -- table[i + j] = table[i].add(table[j])
          checkIndex j 16
          r14 := table[j.toNat]!
          checkIndex i 16
          r15 := table[i.toNat]!
          let v66 ← Point_add_Mont_Base r15 r14
          let v67 ← checkedAdd i j
          checkIndex v67 16
          table := (table.set! v67.toNat v66)
          -- for j in 1..4 { table[i + j] = table[i].add(table[j]) }
          let v70 ← Range_next_1_4 r13 v4
          loop_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v70
        else
          -- for i in [4 as usize, 8, 12] {
          let v74 ← array_next_usize_3 r12 v3
          loop g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v74
      partial_fixpoint
      loop_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v55
    else
      -- var prefix = [Fp::one(); 16]
      let v76 ← Mont_one_Base
      «prefix» := (Vector.replicate 16 v76)
      -- var running = Fp::one()
      let running_79 ← Mont_one_Base
      running := running_79
      -- for k in 1..16 {
      r16 := (Range_Known_1_Known_16.mk () ())
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
        -- for k in 1..16 {
        let v89 ← Range_has_1_16 r16 v5
        if v89 then
          -- for k in 1..16 {
          let v91 ← Range_at_1_16 r16 v5
          let k_92 := v91
          -- prefix[k] = running
          checkIndex k_92 16
          «prefix» := («prefix».set! k_92.toNat running)
          -- if !table[k].is_infinity() { running = running.mul(table[k].z) }
          checkIndex k_92 16
          r17 := table[k_92.toNat]!
          let v98 ← Point_is_infinity_Mont_Base r17
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
            -- for k in 1..16 {
            let v112 ← Range_next_1_16 r16 v5
            loop_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v112
          -- if !table[k].is_infinity() { running = running.mul(table[k].z) }
          if !v98 then
            -- running = running.mul(table[k].z)
            checkIndex k_92 16
            r18 := table[k_92.toNat]!.z
            r19 := running
            let running_105 ← Mont_mul_Base r19 r18
            running := running_105
            join g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          else
            join g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
        else
          -- var inverse = running.inverse()
          r20 := running
          let inverse_117 ← Mont_inverse_Base r20
          inverse := inverse_117
          -- var k: usize = 16
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
            -- while k > 1 {
            if decide ((UInt256.ofNat 1).toNat < k.toNat) then
              -- k -= 1
              let k_127 ← checkedSub k (UInt256.ofNat 1)
              k := k_127
              -- if table[k].is_infinity() { continue }
              let k_130 := k
              checkIndex k_130 16
              r21 := table[k_130.toNat]!
              let v134 ← Point_is_infinity_Mont_Base r21
              if v134 then
                loop_3 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
              else
                -- let z_inverse = inverse.mul(prefix[k])
                let k_137 := k
                checkIndex k_137 16
                r22 := «prefix»[k_137.toNat]!
                r23 := inverse
                let z_inverse_144 ← Mont_mul_Base r23 r22
                z_inverse := z_inverse_144
                -- inverse = inverse.mul(table[k].z)
                let k_146 := k
                checkIndex k_146 16
                r24 := table[k_146.toNat]!.z
                r25 := inverse
                let inverse_153 ← Mont_mul_Base r25 r24
                inverse := inverse_153
                -- let z_inverse2 = z_inverse.square()
                let z_inverse2_157 ← Mont_square_Base z_inverse
                z_inverse2 := z_inverse2_157
                -- table[k] = Point::affine(x: table[k].x.mul(z_inverse2), y: table[k].y.mul(z_inverse2.mul(z_inverse)))
                let k_159 := k
                checkIndex k_159 16
                r26 := table[k_159.toNat]!.x
                let v164 ← Mont_mul_Base r26 z_inverse2
                r27 := v164
                let v167 ← Mont_mul_Base z_inverse2 z_inverse
                r28 := v167
                let k_168 := k
                checkIndex k_168 16
                r29 := table[k_168.toNat]!.y
                let v173 ← Mont_mul_Base r29 r28
                r30 := v173
                let v176 ← Point_affine_Mont_Base r27 r30
                let k_177 := k
                checkIndex k_177 16
                table := (table.set! k_177.toNat v176)
                loop_3 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
            else
              -- var result = Point<Fp>::infinity()
              let result_179 ← Point_infinity_Mont_Base
              result := result_179
              -- var bit: u64 = 256
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
                -- while bit > 0 {
                if decide ((UInt256.ofNat 0).toNat < bit.toNat) then
                  -- bit -= 2
                  let bit_189 ← IntTy.sub (.u 64) bit (UInt256.ofNat 2)
                  bit := bit_189
                  -- result = result.double().double()
                  r31 := result
                  let v195 ← Point_double_Mont_Base r31
                  r32 := v195
                  let result_197 ← Point_double_Mont_Base r32
                  result := result_197
                  -- let entry: usize = ((((u >> bit as u256) & 3) << 2) | ((v >> bit as u256) & 3)).downcast_truncate()
                  let v207 := ((u >>> bit) &&& (UInt256.ofNat 3)) <<< (UInt256.ofNat 2)
                  let entry ← downcast_truncate_u256_usize (v207 ||| ((v_1 >>> bit) &&& (UInt256.ofNat 3)))
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
                  -- if entry != 0 && !table[entry].is_infinity() {
                  if decide (entry ≠ (UInt256.ofNat 0)) then
                    -- if entry != 0 && !table[entry].is_infinity() {
                    checkIndex entry 16
                    r33 := table[entry.toNat]!
                    let v221 ← Point_is_infinity_Mont_Base r33
                    if !v221 then
                      -- result = result.add_affine(x2: table[entry].x, y2: table[entry].y)
                      checkIndex entry 16
                      r34 := table[entry.toNat]!.x
                      checkIndex entry 16
                      r35 := table[entry.toNat]!.y
                      r36 := result
                      let result_230 ← Point_add_affine_Mont_Base r36 r34 r35
                      result := result_230
                      join_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                    else
                      join_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                  else
                    join_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
                else
                  return result
              partial_fixpoint
              loop_4 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
          partial_fixpoint
          loop_3 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36
      partial_fixpoint
      loop_2 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v86
  partial_fixpoint
  loop g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 v43

def combination_k1 (u : UInt256) (v_1 : UInt256) (v20 : Point_Mont_Base) (v21 : UInt256) (g : Point_Mont_Base) (table : (Vector Point_Mont_Base 16)) («prefix» : (Vector Mont_Base 16)) (running : Mont_Base) (inverse : Mont_Base) (k : UInt256) (z_inverse : Mont_Base) (z_inverse2 : Mont_Base) (result : Point_Mont_Base) (bit : UInt256) (r10 : Point_Mont_Base) (r11 : Point_Mont_Base) (r12 : (Vector UInt256 3)) (r13 : Range_Known_1_Known_4) (r14 : Point_Mont_Base) (r15 : Point_Mont_Base) (r16 : Range_Known_1_Known_16) (r17 : Point_Mont_Base) (r18 : Mont_Base) (r19 : Mont_Base) (r20 : Mont_Base) (r21 : Point_Mont_Base) (r22 : Mont_Base) (r23 : Mont_Base) (r24 : Mont_Base) (r25 : Mont_Base) (r26 : Mont_Base) (r27 : Mont_Base) (r28 : Mont_Base) (r29 : Mont_Base) (r30 : Mont_Base) (r31 : Point_Mont_Base) (r32 : Point_Mont_Base) (r33 : Point_Mont_Base) (r34 : Mont_Base) (r35 : Mont_Base) (r36 : Point_Mont_Base) : FeM Point_Mont_Base := do
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
  -- table[4] = g
  let v23 := (UInt256.ofNat 4)
  checkIndex v23 16
  table := (table.set! v23.toNat g)
  -- table[8] = g.double()
  let v27 ← Point_double_Mont_Base g
  let v28 := (UInt256.ofNat 8)
  checkIndex v28 16
  table := (table.set! v28.toNat v27)
  -- table[12] = table[8].add(g)
  let v30 := (UInt256.ofNat 8)
  checkIndex v30 16
  r11 := table[v30.toNat]!
  let v34 ← Point_add_Mont_Base r11 g
  let v35 := (UInt256.ofNat 12)
  checkIndex v35 16
  table := (table.set! v35.toNat v34)
  combination_k2 u v_1 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36

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
  -- let g = generator()
  let g_6 ← generator
  g := g_6
  -- var table = [Point<Fp>::infinity(); 16]
  let v8 ← Point_infinity_Mont_Base
  table := (Vector.replicate 16 v8)
  -- table[1] = q
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 16
  table := (table.set! v12.toNat q)
  -- table[2] = q.double()
  let v14 ← Point_double_Mont_Base q
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 16
  table := (table.set! v15.toNat v14)
  -- table[3] = table[2].add(q)
  let v17 := (UInt256.ofNat 2)
  checkIndex v17 16
  r10 := table[v17.toNat]!
  let v20 ← Point_add_Mont_Base r10 q
  let v21 := (UInt256.ofNat 3)
  checkIndex v21 16
  combination_k1 u v_1 v20 v21 g table «prefix» running inverse k z_inverse z_inverse2 result bit r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36

/-- `Point<Mont<Base>>::to_affine` -/
def Point_to_affine_Mont_Base (self : Point_Mont_Base) : FeM (Mont_Base × Mont_Base) := do
  let mut z_inv : Mont_Base := default
  let mut z_inv2 : Mont_Base := default
  let mut r2 : Mont_Base := default
  let mut r3 : Mont_Base := default
  let mut r4 : Mont_Base := default
  let mut r5 : Mont_Base := default
  -- let z_inv = self.z.inverse()
  r2 := self.z
  let z_inv_3 ← Mont_inverse_Base r2
  z_inv := z_inv_3
  -- let z_inv2 = z_inv.square()
  let z_inv2_6 ← Mont_square_Base z_inv
  z_inv2 := z_inv2_6
  -- (self.x.mul(z_inv2), self.y.mul(z_inv2.mul(z_inv)))
  r3 := self.x
  let v11 ← Mont_mul_Base r3 z_inv2
  let v14 ← Mont_mul_Base z_inv2 z_inv
  r4 := v14
  r5 := self.y
  let v18 ← Mont_mul_Base r5 r4
  return (v11, v18)

/-- `<Mont<Base> as Field>::neg` -/
def Mont_neg_Base (self : Mont_Base) : FeM Mont_Base := do
  let mut r0 : Mont_Base := default
  -- Self::zero().sub(self)
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
  -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
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
    -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
    let _ ← «assert» v4
    -- let x = Fp::new(r)
    let x_26 ← Mont_new_Base r
    x := x_26
    -- let alpha = x.square().mul(x).add(Fp::new(7))
    let v29 ← Mont_square_Base x
    r8 := v29
    let v32 ← Mont_mul_Base r8 x
    r9 := v32
    let v34 ← Mont_new_Base (UInt256.ofNat 7)
    r10 := v34
    let alpha_37 ← Mont_add_Base r9 r10
    alpha := alpha_37
    -- let beta = alpha.pow((P + 1) / 4)
    let v39 ← P
    let v41 ← checkedAdd v39 (UInt256.ofNat 1)
    let v43 ← checkedDiv v41 (UInt256.ofNat 4)
    let beta_45 ← Mont_pow_Base alpha v43
    beta := beta_45
    -- if !beta.square().equals(alpha) { return Option::None }
    let v48 ← Mont_square_Base beta
    r11 := v48
    let v51 ← Mont_equals_Base r11 alpha
    if !v51 then
      return Option_PublicKey.None
    else
      -- let y = if beta.value() & 1 == recovery_id { beta } else { beta.neg() }
      let v57 ← Mont_value_Base beta
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
        -- let y = if beta.value() & 1 == recovery_id { beta } else { beta.neg() }
        y := y_5
        -- let r_inverse = Scalar::new(r).inverse()
        let v68 ← Mont_new_Order r
        r12 := v68
        let r_inverse_70 ← Mont_inverse_Order r12
        r_inverse := r_inverse_70
        -- let u = Scalar::new(hash % N).neg().mul(r_inverse).value()
        let v73 ← N
        let v74 ← checkedRem hash v73
        let v75 ← Mont_new_Order v74
        r13 := v75
        let v77 ← Mont_neg_Order r13
        r14 := v77
        let v80 ← Mont_mul_Order r14 r_inverse
        r15 := v80
        let u ← Mont_value_Order r15
        -- let v = Scalar::new(s).mul(r_inverse).value()
        let v84 ← Mont_new_Order s
        r16 := v84
        let v87 ← Mont_mul_Order r16 r_inverse
        r17 := v87
        let v ← Mont_value_Order r17
        -- let key = combination(u, v, q: Point::affine(x, y))
        let v93 ← Point_affine_Mont_Base x y
        r18 := v93
        let key_95 ← combination u v r18
        key := key_95
        -- if key.is_infinity() { return Option::None }
        let v98 ← Point_is_infinity_Mont_Base key
        if v98 then
          return Option_PublicKey.None
        else
          -- let (key_x, key_y) = key.to_affine()
          let v103 ← Point_to_affine_Mont_Base key
          key_x := v103.fst
          key_y := v103.snd
          -- Option::Some(PublicKey { x: key_x.value(), y: key_y.value() })
          let v109 ← Mont_value_Base key_x
          let v111 ← Mont_value_Base key_y
          return Option_PublicKey.Some (PublicKey.mk v109 v111)
      -- let y = if beta.value() & 1 == recovery_id { beta } else { beta.neg() }
      if decide ((v57 &&& (UInt256.ofNat 1)) = recovery_id) then
        join_1 x alpha beta y r_inverse key key_x key_y r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 beta
      else
        -- beta.neg()
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
  -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
  if v8 then
    -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
    if decide (r ≠ (UInt256.ofNat 0)) then
      -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
      let v13 ← N
      if decide (r.toNat < v13.toNat) then
        -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
        if decide (s ≠ (UInt256.ofNat 0)) then
          -- core::assert(recovery_id <= 1 && r != 0 && r < N && s != 0 && s < N)
          let v19 ← N
          if decide (s.toNat < v19.toNat) then
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
  -- MemBuffer::len(self)
  let v1 ← MemBuffer_len self
  return v1

/-- `<MemBuffer as ByteSource>::lane_at` -/
def MemBuffer_lane_at (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  -- self.read_le<u64>(index)
  let v2 ← MemBuffer_read_le_u64 self index
  return v2

/-- `<MemBuffer as ByteSource>::byte_at` -/
def MemBuffer_byte_at (self : MemBuffer) (index : UInt256) : FeM UInt256 := do
  -- self[index]
  let v2 ← MemBuffer_index_ref self index
  let v3 ← mloadInt (.u 8) v2
  return v3

/-- `lib::word_at<MemBuffer>` -/
def word_at_MemBuffer (input : MemBuffer) (offset : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut r1 : Range_Unknown_Unknown := default
  -- if offset + 32 <= input.len() {
  let v6 ← checkedAdd offset (UInt256.ofNat 32)
  let v7 ← MemBuffer_len_1 input
  if decide (v6.toNat ≤ v7.toNat) then
    -- return (swap_bytes(input.lane_at(offset)) as u256) << 192
    let v9 ← MemBuffer_lane_at input offset
    let v13 := (IntTy.swapBytes (.u 64) v9) <<< (UInt256.ofNat 192)
    let v16 ← checkedAdd offset (UInt256.ofNat 8)
    let v17 ← MemBuffer_lane_at input v16
    let v21 := (IntTy.swapBytes (.u 64) v17) <<< (UInt256.ofNat 128)
    let v25 ← checkedAdd offset (UInt256.ofNat 16)
    let v26 ← MemBuffer_lane_at input v25
    let v30 := (IntTy.swapBytes (.u 64) v26) <<< (UInt256.ofNat 64)
    let v34 ← checkedAdd offset (UInt256.ofNat 24)
    let v35 ← MemBuffer_lane_at input v34
    let v38 := ((v13 ||| v21) ||| v30) ||| (IntTy.swapBytes (.u 64) v35)
    return v38
  else
    -- var value: u256 = 0
    value := (UInt256.ofNat 0)
    -- for index in offset..offset + 32 {
    let v46 ← checkedAdd offset (UInt256.ofNat 32)
    r1 := (Range_Unknown_Unknown.mk offset v46)
    let v50 ← Range_Unknown_Unknown_start r1
    let rec loop (value : UInt256) (r1 : Range_Unknown_Unknown) (v2 : UInt256) : FeM UInt256 := do
      let mut value := value
      let mut r1 := r1
      -- for index in offset..offset + 32 {
      let v53 ← Range_Unknown_Unknown_has r1 v2
      if v53 then
        -- for index in offset..offset + 32 {
        let v55 ← Range_Unknown_Unknown_at r1 v2
        let index := v55
        -- value = (value << 8) | if index < input.len() { input.byte_at(index) as u256 } else { 0 }
        let v60 := value <<< (UInt256.ofNat 8)
        let v61 ← MemBuffer_len_1 input
        let join : UInt256 → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun value r1 v3 => do
          let mut value := value
          let mut r1 := r1
          -- value = (value << 8) | if index < input.len() { input.byte_at(index) as u256 } else { 0 }
          value := (v60 ||| v3)
          -- for index in offset..offset + 32 {
          let v72 ← Range_Unknown_Unknown_next r1 v2
          loop value r1 v72
        -- value = (value << 8) | if index < input.len() { input.byte_at(index) as u256 } else { 0 }
        if decide (index.toNat < v61.toNat) then
          -- input.byte_at(index) as u256
          let v63 ← MemBuffer_byte_at input index
          join value r1 v63
        else
          join value r1 (UInt256.ofNat 0)
      else
        return value
    partial_fixpoint
    loop value r1 v50

/-- `lib::bn254_field::MODULUS` -/
def MODULUS : FeM UInt256 := do
  return UInt256.ofNat 21888242871839275222246405745257275088696311157297823662689037894645226208583

/-- `<Bn254Base as Modulus>::VALUE` -/
def Bn254Base_VALUE : FeM UInt256 := do
  let v0 ← MODULUS
  return v0

/-- `lib::montgomery::raw<Bn254Base>` -/
def raw_Bn254Base (value : UInt256) : FeM Mont_Bn254Base := do
  -- Mont {
  let v1 ← downcast_truncate_u256_u64 value
  let v5 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 64))
  let v9 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 128))
  let v13 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 192))
  return Mont_Bn254Base.mk v1 v5 v9 v13

/-- `<Bn254Base as Modulus>::R2` -/
def Bn254Base_R2 : FeM UInt256 := do
  return UInt256.ofNat 3096616502983703923843567936837374451735540968419076528771170197431451843209

/-- `<Bn254Base as Modulus>::INV` -/
def Bn254Base_INV : FeM UInt256 := do
  return UInt256.ofNat 9786893198990664585

/-- `<Bn254Base as Modulus>::P0` -/
def Bn254Base_P0 : FeM UInt256 := do
  return UInt256.ofNat 4332616871279656263

/-- `<Bn254Base as Modulus>::P1` -/
def Bn254Base_P1 : FeM UInt256 := do
  return UInt256.ofNat 10917124144477883021

/-- `<Bn254Base as Modulus>::P2` -/
def Bn254Base_P2 : FeM UInt256 := do
  return UInt256.ofNat 13281191951274694749

/-- `<Bn254Base as Modulus>::P3` -/
def Bn254Base_P3 : FeM UInt256 := do
  return UInt256.ofNat 3486998266802970665

def Mont_shift_reduced_Bn254Base_k3 (s5 : UInt256) (v32 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) (v64 : (UInt256 × UInt256)) (v69 : (UInt256 × UInt256)) (v75 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (u5, _) = adc(s5, cb, ca)
  let v80 ← adc s5 v69.snd v75.snd
  -- (Mont { l0: u1, l1: u2, l2: u3, l3: u4 }, u5)
  let v83 := (Mont_Bn254Base.mk v32.fst v48.fst v64.fst v75.fst)
  return (v83, v80.fst)

def Mont_shift_reduced_Bn254Base_k2 (s4 : UInt256) (s5 : UInt256) (m : UInt256) (v32 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) (v53 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (k3, j3) = mul_wide(m, M::P3)
  let v58 ← Bn254Base_P3
  let v59 ← mul_wide m v58
  -- let (u3, ca) = adc(u3, k3, ca)
  let v64 ← adc v53.fst v59.fst v48.snd
  -- let (u4, cb) = adc(s4, j3, cb)
  let v69 ← adc s4 v59.snd v53.snd
  -- let (u4, ca) = adc(u4, 0, ca)
  let v75 ← adc v69.fst (UInt256.ofNat 0) v64.snd
  Mont_shift_reduced_Bn254Base_k3 s5 v32 v48 v64 v69 v75

def Mont_shift_reduced_Bn254Base_k1 (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) (m : UInt256) (v17 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (v27 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (u1, ca) = adc(u1, k1, ca)
  let v32 ← adc v21.fst v27.fst v17.snd
  -- let (u2, cb) = adc(s2, j1, cb)
  let v37 ← adc s2 v27.snd v21.snd
  -- let (k2, j2) = mul_wide(m, M::P2)
  let v42 ← Bn254Base_P2
  let v43 ← mul_wide m v42
  -- let (u2, ca) = adc(u2, k2, ca)
  let v48 ← adc v37.fst v43.fst v32.snd
  -- let (u3, cb) = adc(s3, j2, cb)
  let v53 ← adc s3 v43.snd v37.snd
  Mont_shift_reduced_Bn254Base_k2 s4 s5 m v32 v48 v53

/-- `Mont<Bn254Base>::shift_reduced` -/
def Mont_shift_reduced_Bn254Base (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (s4 : UInt256) (s5 : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  -- let m = s0.wrapping_mul(M::INV)
  let v6 ← Bn254Base_INV
  let m ← u64_wrapping_mul s0 v6
  -- let (k0, j0) = mul_wide(m, M::P0)
  let v10 ← Bn254Base_P0
  let v11 ← mul_wide m v10
  -- let (_, ca) = adc(s0, k0, 0)
  let v17 ← adc s0 v11.fst (UInt256.ofNat 0)
  -- let (u1, cb) = adc(s1, j0, 0)
  let v21 ← adc s1 v11.snd (UInt256.ofNat 0)
  -- let (k1, j1) = mul_wide(m, M::P1)
  let v26 ← Bn254Base_P1
  let v27 ← mul_wide m v26
  Mont_shift_reduced_Bn254Base_k1 s2 s3 s4 s5 m v17 v21 v27

def Mont_first_row_Bn254Base_k1 (self : Mont_Bn254Base) (b : UInt256) (v3 : (UInt256 × UInt256)) (v9 : (UInt256 × UInt256)) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (s2, c) = adc(h1, l2, c)
  let v26 ← adc v9.snd v21.fst v15.snd
  -- let (l3, h3) = mul_wide(b, self.l3)
  let v32 ← mul_wide b self.l3
  -- let (s3, c) = adc(h2, l3, c)
  let v37 ← adc v21.snd v32.fst v26.snd
  -- let (s4, _) = adc(h3, 0, c)
  let v43 ← adc v32.snd (UInt256.ofNat 0) v37.snd
  -- Self::shift_reduced(s0, s1, s2, s3, s4, 0)
  let v47 ← Mont_shift_reduced_Bn254Base v3.fst v15.fst v26.fst v37.fst v43.fst (UInt256.ofNat 0)
  return v47

/-- `Mont<Bn254Base>::first_row` -/
def Mont_first_row_Bn254Base (self : Mont_Bn254Base) (b : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (s0, h0) = mul_wide(b, self.l0)
  let v3 ← mul_wide b self.l0
  -- let (l1, h1) = mul_wide(b, self.l1)
  let v9 ← mul_wide b self.l1
  -- let (s1, c) = adc(h0, l1, 0)
  let v15 ← adc v3.snd v9.fst (UInt256.ofNat 0)
  -- let (l2, h2) = mul_wide(b, self.l2)
  let v21 ← mul_wide b self.l2
  Mont_first_row_Bn254Base_k1 self b v3 v9 v15 v21

def Mont_row_Bn254Base_k3 (v12 : (UInt256 × UInt256)) (v30 : (UInt256 × UInt256)) (v47 : (UInt256 × UInt256)) (v64 : (UInt256 × UInt256)) (v69 : (UInt256 × UInt256)) (v75 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (s5, _) = adc(cb, 0, ca)
  let v81 ← adc v69.snd (UInt256.ofNat 0) v75.snd
  -- Self::shift_reduced(s0, s1, s2, s3, s4, s5)
  let v84 ← Mont_shift_reduced_Bn254Base v12.fst v30.fst v47.fst v64.fst v75.fst v81.fst
  return v84

def Mont_row_Bn254Base_k2 (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) (v12 : (UInt256 × UInt256)) (v30 : (UInt256 × UInt256)) (v36 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v47 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (s3, cb) = adc(t.l3, h2, cb)
  let v53 ← adc t.l3 v42.snd v36.snd
  -- let (l3, h3) = mul_wide(b, self.l3)
  let v59 ← mul_wide b self.l3
  -- let (s3, ca) = adc(s3, l3, ca)
  let v64 ← adc v53.fst v59.fst v47.snd
  -- let (s4, cb) = adc(top, h3, cb)
  let v69 ← adc top v59.snd v53.snd
  -- let (s4, ca) = adc(s4, 0, ca)
  let v75 ← adc v69.fst (UInt256.ofNat 0) v64.snd
  Mont_row_Bn254Base_k3 v12 v30 v47 v64 v69 v75

def Mont_row_Bn254Base_k1 (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) (v12 : (UInt256 × UInt256)) (v19 : (UInt256 × UInt256)) (v25 : (UInt256 × UInt256)) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (s1, ca) = adc(s1, l1, ca)
  let v30 ← adc v19.fst v25.fst v12.snd
  -- let (s2, cb) = adc(t.l2, h1, cb)
  let v36 ← adc t.l2 v25.snd v19.snd
  -- let (l2, h2) = mul_wide(b, self.l2)
  let v42 ← mul_wide b self.l2
  -- let (s2, ca) = adc(s2, l2, ca)
  let v47 ← adc v36.fst v42.fst v30.snd
  Mont_row_Bn254Base_k2 self b t top v12 v30 v36 v42 v47

/-- `Mont<Bn254Base>::row` -/
def Mont_row_Bn254Base (self : Mont_Bn254Base) (b : UInt256) (t : Mont_Bn254Base) (top : UInt256) : FeM (Mont_Bn254Base × UInt256) := do
  -- let (l0, h0) = mul_wide(b, self.l0)
  let v5 ← mul_wide b self.l0
  -- let (s0, ca) = adc(t.l0, l0, 0)
  let v12 ← adc t.l0 v5.fst (UInt256.ofNat 0)
  -- let (s1, cb) = adc(t.l1, h0, 0)
  let v19 ← adc t.l1 v5.snd (UInt256.ofNat 0)
  -- let (l1, h1) = mul_wide(b, self.l1)
  let v25 ← mul_wide b self.l1
  Mont_row_Bn254Base_k1 self b t top v12 v19 v25

/-- `Mont<Bn254Base>::reduce` -/
def Mont_reduce_Bn254Base (self : Mont_Bn254Base) (top : UInt256) : FeM Mont_Bn254Base := do
  -- let (d0, b) = sbb(self.l0, M::P0, 0)
  let v4 ← Bn254Base_P0
  let v6 ← sbb self.l0 v4 (UInt256.ofNat 0)
  let d0 := v6.fst
  -- let (d1, b) = sbb(self.l1, M::P1, b)
  let v12 ← Bn254Base_P1
  let v13 ← sbb self.l1 v12 v6.snd
  let d1 := v13.fst
  -- let (d2, b) = sbb(self.l2, M::P2, b)
  let v19 ← Bn254Base_P2
  let v20 ← sbb self.l2 v19 v13.snd
  let d2 := v20.fst
  -- let (d3, b) = sbb(self.l3, M::P3, b)
  let v26 ← Bn254Base_P3
  let v27 ← sbb self.l3 v26 v20.snd
  let d3 := v27.fst
  -- let (_, b) = sbb(top, 0, b)
  let v33 ← sbb top (UInt256.ofNat 0) v27.snd
  let join : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  -- if b == 1 { self } else { Mont { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v33.snd = (UInt256.ofNat 1)) then
    join self
  else
    join (Mont_Bn254Base.mk d0 d1 d2 d3)

/-- `<Mont<Bn254Base> as Field>::mul` -/
def Mont_mul_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut t_r0 : Mont_Bn254Base := default
  let mut t_r1 : Mont_Bn254Base := default
  let mut t_r2 : Mont_Bn254Base := default
  let mut t_r3 : Mont_Bn254Base := default
  -- let (t, top) = self.first_row(other.l0)
  let v3 ← Mont_first_row_Bn254Base self other.l0
  t_r0 := v3.fst
  -- let (t, top) = self.row(other.l1, t, top)
  let v10 ← Mont_row_Bn254Base self other.l1 t_r0 v3.snd
  t_r1 := v10.fst
  -- let (t, top) = self.row(other.l2, t, top)
  let v17 ← Mont_row_Bn254Base self other.l2 t_r1 v10.snd
  t_r2 := v17.fst
  -- let (t, top) = self.row(other.l3, t, top)
  let v24 ← Mont_row_Bn254Base self other.l3 t_r2 v17.snd
  t_r3 := v24.fst
  -- t.reduce(top)
  let v30 ← Mont_reduce_Bn254Base t_r3 v24.snd
  return v30

/-- `Mont<Bn254Base>::new` -/
def Mont_new_Bn254Base (value : UInt256) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  -- core::assert(value < M::VALUE)
  let v2 ← Bn254Base_VALUE
  let _ ← «assert» (decide (value.toNat < v2.toNat))
  -- raw<M>(value).mul(raw<M>(M::R2))
  let v5 ← raw_Bn254Base value
  r0 := v5
  let v6 ← Bn254Base_R2
  let v7 ← raw_Bn254Base v6
  r1 := v7
  let v10 ← Mont_mul_Bn254Base r0 r1
  return v10

/-- `lib::bn254_field::fp` -/
def fp (value : UInt256) : FeM Mont_Bn254Base := do
  -- Fp::new(value)
  let v1 ← Mont_new_Bn254Base value
  return v1

/-- `<Mont<Bn254Base> as Field>::one` -/
def Mont_one_Bn254Base : FeM Mont_Bn254Base := do
  -- Self::new(1)
  let v1 ← Mont_new_Bn254Base (UInt256.ofNat 1)
  return v1

/-- `Point<Mont<Bn254Base>>::affine` -/
def Point_affine_Mont_Bn254Base (x : Mont_Bn254Base) (y : Mont_Bn254Base) : FeM Point_Mont_Bn254Base := do
  -- Point { x, y, z: F::one() }
  let v4 ← Mont_one_Bn254Base
  return Point_Mont_Bn254Base.mk x y v4

/-- `<Mont<Bn254Base> as Field>::square` -/
def Mont_square_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  -- self.mul(self)
  let v1 ← Mont_mul_Bn254Base self self
  return v1

/-- `<Mont<Bn254Base> as Field>::add` -/
def Mont_add_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  -- let (s0, c) = adc(self.l0, other.l0, 0)
  let v5 ← adc self.l0 other.l0 (UInt256.ofNat 0)
  -- let (s1, c) = adc(self.l1, other.l1, c)
  let v12 ← adc self.l1 other.l1 v5.snd
  -- let (s2, c) = adc(self.l2, other.l2, c)
  let v19 ← adc self.l2 other.l2 v12.snd
  -- let (s3, c) = adc(self.l3, other.l3, c)
  let v26 ← adc self.l3 other.l3 v19.snd
  -- Mont { l0: s0, l1: s1, l2: s2, l3: s3 }.reduce(c)
  r0 := (Mont_Bn254Base.mk v5.fst v12.fst v19.fst v26.fst)
  let v33 ← Mont_reduce_Bn254Base r0 v26.snd
  return v33

/-- `<Mont<Bn254Base> as Field>::equals` -/
def Mont_equals_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Bool := do
  let join : Bool → FeM Bool := fun v2 => do
    return v2
  let join_1 : FeM Bool := do
    join false
  -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
  if decide (self.l0 = other.l0) then
    -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
    if decide (self.l1 = other.l1) then
      -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
      if decide (self.l2 = other.l2) then
        -- self.l0 == other.l0 && self.l1 == other.l1 && self.l2 == other.l2 && self.l3 == other.l3
        if decide (self.l3 = other.l3) then
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
  -- y.square().equals(x.square().mul(x).add(b))
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
  -- Mont { l0: 0, l1: 0, l2: 0, l3: 0 }
  let v4 := (Mont_Bn254Base.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  return v4

/-- `Point<Mont<Bn254Base>>::infinity` -/
def Point_infinity_Mont_Bn254Base : FeM Point_Mont_Bn254Base := do
  -- Point { x: F::one(), y: F::one(), z: F::zero() }
  let v0 ← Mont_one_Bn254Base
  let v1 ← Mont_one_Bn254Base
  let v2 ← Mont_zero_Bn254Base
  return Point_Mont_Bn254Base.mk v0 v1 v2

/-- `lib::bn254::g1_point<MemBuffer>` -/
def g1_point_MemBuffer (input : MemBuffer) (offset : UInt256) : FeM Option_Point_Mont_Bn254Base := do
  let mut r0 : Mont_Bn254Base := default
  let mut r1 : Mont_Bn254Base := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  let mut r4 : Mont_Bn254Base := default
  -- let x = word_at(input, offset)
  let x ← word_at_MemBuffer input offset
  -- let y = word_at(input, offset: offset + 32)
  let v6 ← checkedAdd offset (UInt256.ofNat 32)
  let y ← word_at_MemBuffer input v6
  -- if x >= MODULUS || y >= MODULUS { return Option::None }
  let v9 ← MODULUS
  let join : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    return Option_Point_Mont_Bn254Base.None
  -- if x >= MODULUS || y >= MODULUS { return Option::None }
  if decide (v9.toNat ≤ x.toNat) then
    join r0 r1 r2 r3 r4
  else
    -- if x >= MODULUS || y >= MODULUS { return Option::None }
    let v11 ← MODULUS
    if decide (v11.toNat ≤ y.toNat) then
      join r0 r1 r2 r3 r4
    else
      let join_1 : Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → Mont_Bn254Base → FeM Option_Point_Mont_Bn254Base := fun r0 r1 r2 r3 r4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        -- let point = Point::affine(x: fp(x), y: fp(y))
        let v23 ← fp x
        r0 := v23
        let v24 ← fp y
        r1 := v24
        let point ← Point_affine_Mont_Bn254Base r0 r1
        -- if !on_curve(x: point.x, y: point.y, b: fp(3)) { return Option::None }
        r2 := point.x
        r3 := point.y
        let v32 ← fp (UInt256.ofNat 3)
        r4 := v32
        let v36 ← on_curve_Mont_Bn254Base r2 r3 r4
        if !v36 then
          return Option_Point_Mont_Bn254Base.None
        else
          return Option_Point_Mont_Bn254Base.Some point
      -- if x == 0 && y == 0 { return Option::Some(Point::infinity()) }
      if decide (x = (UInt256.ofNat 0)) then
        -- if x == 0 && y == 0 { return Option::Some(Point::infinity()) }
        if decide (y = (UInt256.ofNat 0)) then
          -- return Option::Some(Point::infinity())
          let v19 ← Point_infinity_Mont_Bn254Base
          return Option_Point_Mont_Bn254Base.Some v19
        else
          join_1 r0 r1 r2 r3 r4
      else
        join_1 r0 r1 r2 r3 r4

/-- `<Mont<Bn254Base> as Field>::is_zero` -/
def Mont_is_zero_Bn254Base (self : Mont_Bn254Base) : FeM Bool := do
  let join : Bool → FeM Bool := fun v1 => do
    return v1
  let join_1 : FeM Bool := do
    join false
  -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
  if decide (self.l0 = (UInt256.ofNat 0)) then
    -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
    if decide (self.l1 = (UInt256.ofNat 0)) then
      -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
      if decide (self.l2 = (UInt256.ofNat 0)) then
        -- self.l0 == 0 && self.l1 == 0 && self.l2 == 0 && self.l3 == 0
        if decide (self.l3 = (UInt256.ofNat 0)) then
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
  -- self.z.is_zero()
  r0 := self.z
  let v3 ← Mont_is_zero_Bn254Base r0
  return v3

def Mont_sub_Bn254Base_k2 (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (v27 : (UInt256 × UInt256)) (d3 : UInt256) (s0 : UInt256) (s1 : UInt256) (v46 : (UInt256 × UInt256)) (s2 : UInt256) : FeM Mont_Bn254Base := do
  -- let (s3, _) = adc(d3, M::P3, c)
  let v51 ← Bn254Base_P3
  let v52 ← adc d3 v51 v46.snd
  let s3 := v52.fst
  let join : Mont_Bn254Base → FeM Mont_Bn254Base := fun v2 => do
    return v2
  -- if b == 1 { Mont { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Mont { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v27.snd = (UInt256.ofNat 1)) then
    join (Mont_Bn254Base.mk s0 s1 s2 s3)
  else
    join (Mont_Bn254Base.mk d0 d1 d2 d3)

def Mont_sub_Bn254Base_k1 (self : Mont_Bn254Base) (other : Mont_Bn254Base) (d0 : UInt256) (d1 : UInt256) (v20 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Mont_Bn254Base := do
  -- let (d3, b) = sbb(self.l3, other.l3, b)
  let v27 ← sbb self.l3 other.l3 v20.snd
  let d3 := v27.fst
  -- let (s0, c) = adc(d0, M::P0, 0)
  let v32 ← Bn254Base_P0
  let v34 ← adc d0 v32 (UInt256.ofNat 0)
  let s0 := v34.fst
  -- let (s1, c) = adc(d1, M::P1, c)
  let v39 ← Bn254Base_P1
  let v40 ← adc d1 v39 v34.snd
  let s1 := v40.fst
  -- let (s2, c) = adc(d2, M::P2, c)
  let v45 ← Bn254Base_P2
  let v46 ← adc d2 v45 v40.snd
  let s2 := v46.fst
  Mont_sub_Bn254Base_k2 d0 d1 d2 v27 d3 s0 s1 v46 s2

/-- `<Mont<Bn254Base> as Field>::sub` -/
def Mont_sub_Bn254Base (self : Mont_Bn254Base) (other : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  -- let (d0, b) = sbb(self.l0, other.l0, 0)
  let v6 ← sbb self.l0 other.l0 (UInt256.ofNat 0)
  let d0 := v6.fst
  -- let (d1, b) = sbb(self.l1, other.l1, b)
  let v13 ← sbb self.l1 other.l1 v6.snd
  let d1 := v13.fst
  -- let (d2, b) = sbb(self.l2, other.l2, b)
  let v20 ← sbb self.l2 other.l2 v13.snd
  let d2 := v20.fst
  Mont_sub_Bn254Base_k1 self other d0 d1 v20 d2

/-- `<Mont<Bn254Base> as Field>::double` -/
def Mont_double_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  -- self.add(self)
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
  -- if self.is_infinity() || self.y.is_zero() { return Self::infinity() }
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
    -- return Self::infinity()
    let v72 ← Point_infinity_Mont_Bn254Base
    return v72
  -- if self.is_infinity() || self.y.is_zero() { return Self::infinity() }
  if v1 then
    join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
  else
    -- if self.is_infinity() || self.y.is_zero() { return Self::infinity() }
    r6 := self.y
    let v4 ← Mont_is_zero_Bn254Base r6
    if v4 then
      join a b c d e x r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24
    else
      -- let a = self.x.square()
      r7 := self.x
      let a_9 ← Mont_square_Bn254Base r7
      a := a_9
      -- let b = self.y.square()
      r8 := self.y
      let b_13 ← Mont_square_Bn254Base r8
      b := b_13
      -- let c = b.square()
      let c_16 ← Mont_square_Bn254Base b
      c := c_16
      -- let d = self.x.add(b).square().sub(a).sub(c).double()
      r9 := self.x
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
      -- let e = a.double().add(a)
      let v34 ← Mont_double_Bn254Base a
      r14 := v34
      let e_37 ← Mont_add_Bn254Base r14 a
      e := e_37
      -- let x = e.square().sub(d.double())
      let v40 ← Mont_square_Bn254Base e
      r15 := v40
      let v42 ← Mont_double_Bn254Base d
      r16 := v42
      let x_45 ← Mont_sub_Bn254Base r15 r16
      x := x_45
      -- let y = e.mul(d.sub(x)).sub(c.double().double().double())
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
      -- Point { x, y, z: self.y.mul(self.z).double() }
      r22 := self.z
      r23 := self.y
      let v67 ← Mont_mul_Bn254Base r23 r22
      r24 := v67
      let v69 ← Mont_double_Bn254Base r24
      return Point_Mont_Bn254Base.mk x y v69

/-- `<Bn254Base as Modulus>::SIGNED_62` -/
def Bn254Base_SIGNED_62 : FeM (Vector UInt256 5) := do
  let v5 := #v[(UInt256.ofNat 4332616871279656263), (UInt256.ofNat 2163322412065040948), (UInt256.ofNat 361514372735272409), (UInt256.ofNat 1806960190875503214), (UInt256.ofNat 48)]
  return v5

/-- `<Bn254Base as Modulus>::INV62` -/
def Bn254Base_INV62 : FeM UInt256 := do
  return UInt256.ofNat 4048164856291499127

/-- `<Bn254Base as Modulus>::R3` -/
def Bn254Base_R3 : FeM UInt256 := do
  return UInt256.ofNat 14921786541159648185948152738563080959093619838510245177710943249661917737183

/-- `<Mont<Bn254Base> as Field>::inverse` -/
def Mont_inverse_Bn254Base (self : Mont_Bn254Base) : FeM Mont_Bn254Base := do
  let mut r0 : (Vector UInt256 4) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : Mont_Bn254Base := default
  let mut r3 : Mont_Bn254Base := default
  -- let limbs = invert_256([self.l0, self.l1, self.l2, self.l3], M::SIGNED_62, M::INV62)
  r0 := (#v[self.l0, self.l1, self.l2, self.l3])
  let v6 ← Bn254Base_SIGNED_62
  r1 := v6
  let v7 ← Bn254Base_INV62
  let limbs ← invert_256 r0 r1 v7
  -- Mont { l0: limbs[0], l1: limbs[1], l2: limbs[2], l3: limbs[3] }.mul(raw<M>(M::R3))
  let v12 := (UInt256.ofNat 0)
  checkIndex v12 4
  let v14 := (UInt256.ofNat 1)
  checkIndex v14 4
  let v16 := (UInt256.ofNat 2)
  checkIndex v16 4
  let v18 := (UInt256.ofNat 3)
  checkIndex v18 4
  let v20 := (Mont_Bn254Base.mk limbs[v12.toNat]! limbs[v14.toNat]! limbs[v16.toNat]! limbs[v18.toNat]!)
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
  -- let z_inv = self.z.inverse()
  r2 := self.z
  let z_inv_3 ← Mont_inverse_Bn254Base r2
  z_inv := z_inv_3
  -- let z_inv2 = z_inv.square()
  let z_inv2_6 ← Mont_square_Bn254Base z_inv
  z_inv2 := z_inv2_6
  -- (self.x.mul(z_inv2), self.y.mul(z_inv2.mul(z_inv)))
  r3 := self.x
  let v11 ← Mont_mul_Bn254Base r3 z_inv2
  let v14 ← Mont_mul_Bn254Base z_inv2 z_inv
  r4 := v14
  r5 := self.y
  let v18 ← Mont_mul_Bn254Base r5 r4
  return (v11, v18)

/-- `Mont<Bn254Base>::value` -/
def Mont_value_Bn254Base (self : Mont_Bn254Base) : FeM UInt256 := do
  let mut r0 : Mont_Bn254Base := default
  -- let value = self.mul(raw<M>(1))
  let v2 ← raw_Bn254Base (UInt256.ofNat 1)
  r0 := v2
  let value ← Mont_mul_Bn254Base self r0
  -- value.l0 as u256 | (value.l1 as u256) << 64 | (value.l2 as u256) << 128 | (value.l3 as u256) << 192
  let v17 := (value.l0 ||| (value.l1 <<< (UInt256.ofNat 64))) ||| (value.l2 <<< (UInt256.ofNat 128))
  return v17 ||| (value.l3 <<< (UInt256.ofNat 192))

/-- `lib::bn254_field::fp_value` -/
def fp_value (a : Mont_Bn254Base) : FeM UInt256 := do
  -- a.value()
  let v1 ← Mont_value_Bn254Base a
  return v1

/-- `lib::bn254::encode` -/
def encode (point : Point_Mont_Bn254Base) : FeM (Vector UInt256 2) := do
  let mut x : Mont_Bn254Base := default
  let mut y : Mont_Bn254Base := default
  -- if point.is_infinity() { return [0, 0] }
  let v1 ← Point_is_infinity_Mont_Bn254Base point
  if v1 then
    return #v[(UInt256.ofNat 0), (UInt256.ofNat 0)]
  else
    -- let (x, y) = point.to_affine()
    let v7 ← Point_to_affine_Mont_Bn254Base point
    x := v7.fst
    y := v7.snd
    -- [fp_value(x), fp_value(y)]
    let v13 ← fp_value x
    let v15 ← fp_value y
    return #v[v13, v15]

/-- `lib::SUCCESS` -/
def SUCCESS : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `lib::HOST_ALLOCATION` -/
def HOST_ALLOCATION : FeM UInt256 := do
  return UInt256.ofNat 2

/-- `lib::host_allocation` -/
def host_allocation (gas_used : UInt256) : FeM Output := do
  -- Output { status: HOST_ALLOCATION, gas_used, output: MemBuffer::empty() }
  let v1 ← HOST_ALLOCATION
  let v3 ← MemBuffer_empty
  return Output.mk v1 gas_used v3


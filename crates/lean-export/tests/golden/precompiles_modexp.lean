export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemArray_as_slice_mut_u64 MemArray_as_slice_ref_u64 MemArray_drop_u64 MemArray_index_mut_u64 MemArray_index_ref_u64 MemArray_len_u64 MemArray_ptr_u64 MemArray_try_new_zeroed_u64 MemArray_u64 MemArray_zero_u64 MemBuffer MemBuffer_drop MemBuffer_empty MemBuffer_len MemBuffer_new MemBuffer_ptr MemBuffer_span_ref MemBuffer_try_new MemSlice_copy_from_u64 MemSlice_empty_u8 MemSlice_head_mut_u64 MemSlice_head_ref_u64 MemSlice_len_u64 MemSlice_ptr_u64 MemSlice_slice_mut_u64 MemSlice_slice_ref_u64 MemSlice_u64 Option_MemArray_u64 Option_MemArray_u64.None Option_MemArray_u64.Some Range_Known_0_Known_5 Range_Known_0_Known_5.mk Range_Known_0_Known_6 Range_Known_0_Known_6.mk Range_Known_0_Known_8 Range_Known_0_Known_8.mk Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_Known_1_Unknown Range_Known_1_Unknown.mk Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start Range_at_0 Range_at_0_5 Range_at_0_6 Range_at_0_8 Range_at_1 Range_has_0 Range_has_0_5 Range_has_0_6 Range_has_0_8 Range_has_1 Range_next_0 Range_next_0_5 Range_next_0_6 Range_next_0_8 Range_next_1 Range_start_0 Range_start_0_5 Range_start_0_6 Range_start_0_8 Range_start_1 cast_u8_u64 downcast_truncate_u128_u64 downcast_truncate_u256_u64 downcast_truncate_u256_usize downcast_truncate_u64_u8 downcast_truncate_u64_usize downcast_truncate_usize_u64 downcast_truncate_usize_usize downcast_unchecked_u256_usize «assert»)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1 Output Output.mk SUCCESS adc host_allocation sbb)

structure Exponent where
  limbs : MemArray_u64
  top : UInt256
  total : UInt256
deriving Inhabited

inductive Option_Exponent where
  | None
  | Some (f0_1 : Exponent)
deriving Inhabited

def Option_Exponent.tag : Option_Exponent → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Exponent.Some_0 : Option_Exponent → Exponent
  | .Some f0_1 => f0_1
  | _ => default

def Option_Exponent.set_Some_0 (x : Option_Exponent) (v : Exponent) : Option_Exponent :=
  match x with
  | .Some f0_1 => .Some v
  | x => x

structure Modulus where
  limbs : MemArray_u64
  len : UInt256
  shift : UInt256
  reciprocal : UInt256
deriving Inhabited

inductive Option_Modulus where
  | None
  | Some (f0_1 : Modulus)
deriving Inhabited

def Option_Modulus.tag : Option_Modulus → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Modulus.Some_0 : Option_Modulus → Modulus
  | .Some f0_1 => f0_1
  | _ => default

def Option_Modulus.set_Some_0 (x : Option_Modulus) (v : Modulus) : Option_Modulus :=
  match x with
  | .Some f0_1 => .Some v
  | x => x

structure Wide where
  l0 : UInt256
  l1 : UInt256
  l2 : UInt256
  l3 : UInt256
deriving Inhabited

structure Small_1 where
  m : Wide
  inverse : UInt256
deriving Inhabited

structure Window where
  «end» : UInt256
  entry : UInt256
  odd : Bool
deriving Inhabited

structure Small_2 where
  m : Wide
  inverse : UInt256
deriving Inhabited

structure Small_3 where
  m : Wide
  inverse : UInt256
deriving Inhabited

structure Small_4 where
  m : Wide
  inverse : UInt256
deriving Inhabited

structure Montgomery where
  limbs : MemArray_u64
  len : UInt256
  inverse : UInt256
deriving Inhabited

/-- The `init` of contract `Modexp`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_6 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  -- let len = raw.calldatasize()
  let len ← Evm_calldatasize
  -- var input = MemBuffer::new(len.downcast_unchecked())
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  -- raw.calldatacopy(dest: mut input, dest_offset: 0, src_offset: 0, len)
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  return input

/-- `lib::modexp::input_word<MemBuffer>` -/
def input_word_MemBuffer (input : MemBuffer) (offset : UInt256) (len : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut i : UInt256 := default
  -- let end = input.len()
  let «end» ← MemBuffer_len_1 input
  let join : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun value i present => do
    let mut value := value
    let mut i := i
    -- if present == 0 { return 0 }
    if decide (present = (UInt256.ofNat 0)) then
      return UInt256.ofNat 0
    else
      -- var value: u256 = 0
      value := (UInt256.ofNat 0)
      -- var i: usize = 0
      i := (UInt256.ofNat 0)
      let rec loop (value : UInt256) (i : UInt256) : FeM UInt256 := do
        let mut value := value
        let mut i := i
        -- while i + 8 <= present {
        let v35 := decide ((i + (UInt256.ofNat 8)).toNat ≤ present.toNat)
        if v35 then
          -- value = (value << 64) | swap_bytes(input.lane_at(offset + i)) as u256
          let v39 := value <<< (UInt256.ofNat 64)
          let v43 ← MemBuffer_lane_at input (offset + i)
          value := (v39 ||| (IntTy.swapBytes (.u 64) v43))
          -- i += 8
          i := (i + (UInt256.ofNat 8))
          loop value i
        else
          let rec loop_1 (value : UInt256) (i : UInt256) : FeM UInt256 := do
            let mut value := value
            let mut i := i
            -- while i < present {
            if decide (i.toNat < present.toNat) then
              -- value = (value << 8) | input.byte_at(offset + i) as u256
              let v60 := value <<< (UInt256.ofNat 8)
              let v64 ← MemBuffer_byte_at input (offset + i)
              value := (v60 ||| v64)
              -- i += 1
              i := (i + (UInt256.ofNat 1))
              loop_1 value i
            else
              return value <<< ((len - present) * (UInt256.ofNat 8))
          partial_fixpoint
          loop_1 value i
      partial_fixpoint
      loop value i
  -- let present = if offset >= end { 0 } else if end - offset < len { end - offset } else { len }
  if decide («end».toNat ≤ offset.toNat) then
    join value i (UInt256.ofNat 0)
  else
    let join_1 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun value i v4 => do
      let mut value := value
      let mut i := i
      join value i v4
    if decide ((«end» - offset).toNat < len.toNat) then
      join_1 value i («end» - offset)
    else
      join_1 value i len

/-- `lib::modexp::MAX_LENGTH` -/
def MAX_LENGTH : FeM UInt256 := do
  return UInt256.ofNat 1024

/-- `lib::FAILURE` -/
def FAILURE : FeM UInt256 := do
  return UInt256.ofNat 1

/-- `lib::modexp::bit_length` -/
def bit_length (value : UInt256) : FeM UInt256 := do
  -- (256 - leading_zeros(value)).downcast_truncate()
  let v3 ← checkedSub (UInt256.ofNat 256) (IntTy.leadingZeros (.u 256) value)
  let v4 ← downcast_truncate_u256_u64 v3
  return v4

/-- `lib::modexp::gas_cost<MemBuffer>` -/
def gas_cost_MemBuffer (input : MemBuffer) (base : UInt256) (exponent : UInt256) (modulus : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v4 => do
    -- let longest: u64 = (if base > modulus { base } else { modulus }).downcast_truncate()
    let longest ← downcast_truncate_usize_u64 v4
    -- let words = (longest + 7) / 8
    let v21 ← IntTy.add (.u 64) longest (UInt256.ofNat 7)
    let words ← checkedDiv v21 (UInt256.ofNat 8)
    let join_1 : UInt256 → FeM UInt256 := fun complexity => do
      -- let head = input_word(input, offset: 96 + base, len: if exponent < 32 { exponent } else { 32 })
      let v36 ← checkedAdd (UInt256.ofNat 96) base
      let join_2 : UInt256 → FeM UInt256 := fun v6 => do
        -- let head = input_word(input, offset: 96 + base, len: if exponent < 32 { exponent } else { 32 })
        let head ← input_word_MemBuffer input v36 v6
        -- let head_bits = bit_length(head)
        let head_bits ← bit_length head
        let join_3 : UInt256 → FeM UInt256 := fun top_bits => do
          let join_4 : UInt256 → FeM UInt256 := fun extra => do
            -- let count = 16 * extra + top_bits
            let v68 ← IntTy.mul (.u 64) (UInt256.ofNat 16) extra
            let count ← IntTy.add (.u 64) v68 top_bits
            let join_5 : UInt256 → FeM UInt256 := fun v9 => do
              -- let cost = complexity * if count == 0 { 1 } else { count }
              let cost ← IntTy.mul (.u 64) complexity v9
              let join_6 : UInt256 → FeM UInt256 := fun v10 => do
                return v10
              -- if cost < 500 { 500 } else { cost }
              if decide (cost.toNat < (UInt256.ofNat 500).toNat) then
                join_6 (UInt256.ofNat 500)
              else
                join_6 cost
            -- let cost = complexity * if count == 0 { 1 } else { count }
            if decide (count = (UInt256.ofNat 0)) then
              join_5 (UInt256.ofNat 1)
            else
              join_5 count
          -- let extra: u64 = if exponent > 32 { (exponent - 32).downcast_truncate() } else { 0 }
          if decide ((UInt256.ofNat 32).toNat < exponent.toNat) then
            -- (exponent - 32).downcast_truncate()
            let v61 ← checkedSub exponent (UInt256.ofNat 32)
            let v62 ← downcast_truncate_usize_u64 v61
            join_4 v62
          else
            join_4 (UInt256.ofNat 0)
        -- let top_bits = if head_bits > 0 { head_bits - 1 } else { 0 }
        if decide ((UInt256.ofNat 0).toNat < head_bits.toNat) then
          -- head_bits - 1
          let v51 ← IntTy.sub (.u 64) head_bits (UInt256.ofNat 1)
          join_3 v51
        else
          join_3 (UInt256.ofNat 0)
      if decide (exponent.toNat < (UInt256.ofNat 32).toNat) then
        join_2 exponent
      else
        join_2 (UInt256.ofNat 32)
    -- let complexity = if longest > 32 { 2 * words * words } else { 16 }
    if decide ((UInt256.ofNat 32).toNat < longest.toNat) then
      -- 2 * words * words
      let v28 ← IntTy.mul (.u 64) (UInt256.ofNat 2) words
      let v29 ← IntTy.mul (.u 64) v28 words
      join_1 v29
    else
      join_1 (UInt256.ofNat 16)
  -- let longest: u64 = (if base > modulus { base } else { modulus }).downcast_truncate()
  if decide (modulus.toNat < base.toNat) then
    join base
  else
    join modulus

/-- `lib::modexp::input_byte<MemBuffer>` -/
def input_byte_MemBuffer (input : MemBuffer) (index : UInt256) : FeM UInt256 := do
  -- if index < input.len() { input.byte_at(index) } else { 0 }
  let v4 ← MemBuffer_len_1 input
  let join : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  if decide (index.toNat < v4.toNat) then
    -- input.byte_at(index)
    let v6 ← MemBuffer_byte_at input index
    join v6
  else
    join (UInt256.ofNat 0)

/-- `lib::modexp::input_limb<MemBuffer>` -/
def input_limb_MemBuffer (input : MemBuffer) (offset : UInt256) (len : UInt256) (index : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut byte : UInt256 := default
  -- let end = offset + len - index * 8
  let «end» := (offset + len) - (index * (UInt256.ofNat 8))
  -- if index * 8 + 8 <= len && end <= input.len() { return swap_bytes(input.lane_at(end - 8)) }
  let v18 := decide (((index * (UInt256.ofNat 8)) + (UInt256.ofNat 8)).toNat ≤ len.toNat)
  let join : UInt256 → UInt256 → FeM UInt256 := fun value byte => do
    let mut value := value
    let mut byte := byte
    -- var value: u64 = 0
    value := (UInt256.ofNat 0)
    -- var byte: usize = 8
    byte := (UInt256.ofNat 8)
    let rec loop (value : UInt256) (byte : UInt256) : FeM UInt256 := do
      let mut value := value
      let mut byte := byte
      -- while byte > 0 {
      if decide ((UInt256.ofNat 0).toNat < byte.toNat) then
        -- byte -= 1
        byte := (byte - (UInt256.ofNat 1))
        -- let position = index * 8 + byte
        let position := (index * (UInt256.ofNat 8)) + byte
        -- value = value << 8
        value := (IntTy.shl (.u 64) value (UInt256.ofNat 8))
        let join_1 : UInt256 → UInt256 → FeM UInt256 := fun value byte => do
          let mut value := value
          let mut byte := byte
          loop value byte
        -- if position < len { value |= input_byte(input, offset + len - 1 - position) as u64 }
        if decide (position.toNat < len.toNat) then
          -- value |= input_byte(input, offset + len - 1 - position) as u64
          let v60 ← input_byte_MemBuffer input (((offset + len) - (UInt256.ofNat 1)) - position)
          value := (value ||| v60)
          join_1 value byte
        else
          join_1 value byte
      else
        return value
    partial_fixpoint
    loop value byte
  if v18 then
    let v19 ← MemBuffer_len_1 input
    if decide («end».toNat ≤ v19.toNat) then
      -- return swap_bytes(input.lane_at(end - 8))
      let v23 ← MemBuffer_lane_at input («end» - (UInt256.ofNat 8))
      return IntTy.swapBytes (.u 64) v23
    else
      join value byte
  else
    join value byte

/-- `lib::modexp::input_limbs<MemBuffer>` -/
def input_limbs_MemBuffer (input : MemBuffer) (offset : UInt256) (len : UInt256) (count : UInt256) : FeM Option_MemArray_u64 := do
  let mut limbs : MemArray_u64 := default
  let mut r1 : Range_Known_0_Unknown := default
  -- var limbs = MemArray<u64>::try_new_zeroed(count)?
  let v6 ← MemArray_try_new_zeroed_u64 count
  if match v6 with | .Some .. => true | _ => false then
    let limbs_8 := v6.Some_0
    limbs := limbs_8
    -- let present = (len + 7) / 8
    let v12 ← checkedAdd len (UInt256.ofNat 7)
    let present ← checkedDiv v12 (UInt256.ofNat 8)
    let join : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_MemArray_u64 := fun limbs r1 read => do
      let mut limbs := limbs
      let mut r1 := r1
      -- for i in 0..read { limbs[i] = input_limb(input, offset, len, i) }
      r1 := (Range_Known_0_Unknown.mk () read)
      let v26 ← Range_start_0 r1
      let rec loop (limbs : MemArray_u64) (r1 : Range_Known_0_Unknown) (v5 : UInt256) : FeM Option_MemArray_u64 := do
        let mut limbs := limbs
        let mut r1 := r1
        -- for i in 0..read { limbs[i] = input_limb(input, offset, len, i) }
        let v29 ← Range_has_0 r1 v5
        if v29 then
          let v31 ← Range_at_0 r1 v5
          let i := v31
          -- limbs[i] = input_limb(input, offset, len, i)
          let v33 ← input_limb_MemBuffer input offset len i
          let (v35, v35_k) ← MemArray_index_mut_u64 limbs i
          mstoreInt (.u 64) v35 v33
          let v35_0 ← v35_k v35
          limbs := v35_0
          -- for i in 0..read { limbs[i] = input_limb(input, offset, len, i) }
          let v38 ← Range_next_0 r1 v5
          loop limbs r1 v38
        else
          -- Option::Some(limbs)
          let limbs_40 := limbs
          return Option_MemArray_u64.Some limbs_40
      partial_fixpoint
      loop limbs r1 v26
    -- let read = if present < count { present } else { count }
    if decide (present.toNat < count.toNat) then
      join limbs r1 present
    else
      join limbs r1 count
  else
    match (← pure v6) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v6.Some_0
      pure ()
    | _ => pure ()
    return Option_MemArray_u64.None

/-- `Exponent::try_read<MemBuffer>` -/
def Exponent_try_read_MemBuffer (input : MemBuffer) (start : UInt256) (len : UInt256) : FeM Option_Exponent := do
  let mut limbs : MemArray_u64 := default
  let mut count : UInt256 := default
  -- let total = len * 8
  let total ← checkedMul len (UInt256.ofNat 8)
  -- let limbs = input_limbs(input, offset: start, len, count: (len + 7) / 8)?
  let v10 ← checkedAdd len (UInt256.ofNat 7)
  let v12 ← checkedDiv v10 (UInt256.ofNat 8)
  let v13 ← input_limbs_MemBuffer input start len v12
  if match v13 with | .Some .. => true | _ => false then
    let limbs_15 := v13.Some_0
    limbs := limbs_15
    -- var count = limbs.len()
    let count_18 ← MemArray_len_u64 limbs
    count := count_18
    let rec loop (limbs : MemArray_u64) (count : UInt256) : FeM Option_Exponent := do
      let mut limbs := limbs
      let mut count := count
      let join : MemArray_u64 → UInt256 → FeM Option_Exponent := fun limbs count => do
        let mut limbs := limbs
        let mut count := count
        let join_1 : MemArray_u64 → UInt256 → UInt256 → FeM Option_Exponent := fun limbs count top => do
          let mut limbs := limbs
          let mut count := count
          -- Option::Some(Self { limbs, top, total })
          let limbs_68 := limbs
          let v70 := (Exponent.mk limbs_68 top total)
          return Option_Exponent.Some v70
        -- let top = if count == 0 { total } else {
        if decide (count = (UInt256.ofNat 0)) then
          join_1 limbs count total
        else
          -- let highest: usize = (255 - leading_zeros(limbs[count - 1] as u256)).downcast_truncate()
          let v47 ← checkedSub count (UInt256.ofNat 1)
          let v49 ← MemArray_index_ref_u64 limbs v47
          let v50 ← mloadInt (.u 64) v49
          let v53 ← checkedSub (UInt256.ofNat 255) (IntTy.leadingZeros (.u 256) v50)
          let highest ← downcast_truncate_u256_usize v53
          -- total - 1 - ((count - 1) * 64 + highest)
          let v57 ← checkedSub total (UInt256.ofNat 1)
          let v61 ← checkedSub count (UInt256.ofNat 1)
          let v63 ← checkedMul v61 (UInt256.ofNat 64)
          let v64 ← checkedAdd v63 highest
          let v65 ← checkedSub v57 v64
          join_1 limbs count v65
      -- while count > 0 && limbs[count - 1] == 0 { count -= 1 }
      if decide ((UInt256.ofNat 0).toNat < count.toNat) then
        let v27 ← checkedSub count (UInt256.ofNat 1)
        let v29 ← MemArray_index_ref_u64 limbs v27
        let v30 ← mloadInt (.u 64) v29
        if decide (v30 = (UInt256.ofNat 0)) then
          -- count -= 1
          let count_35 ← checkedSub count (UInt256.ofNat 1)
          count := count_35
          loop limbs count
        else
          join limbs count
      else
        join limbs count
    partial_fixpoint
    loop limbs count
  else
    match (← pure v13) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v13.Some_0
      pure ()
    | _ => pure ()
    return Option_Exponent.None

/-- `lib::bignum::LIMB_BASE` -/
def LIMB_BASE : FeM UInt256 := do
  return UInt256.ofNat 18446744073709551616

/-- `lib::bignum::reciprocal_2by1` -/
def reciprocal_2by1 (d : UInt256) : FeM UInt256 := do
  -- (all / d as u128 - LIMB_BASE).downcast_truncate()
  let v5 ← checkedDiv (UInt256.ofNat 340282366920938463463374607431768211455) d
  let v6 ← LIMB_BASE
  let v7 ← IntTy.sub (.u 128) v5 v6
  let v8 ← downcast_truncate_u128_u64 v7
  return v8

/-- `lib::bignum::reciprocal_3by2` -/
def reciprocal_3by2 (d1 : UInt256) (d0 : UInt256) : FeM UInt256 := do
  let mut v : UInt256 := default
  let mut p : UInt256 := default
  -- var v = reciprocal_2by1(d1)
  let v_3 ← reciprocal_2by1 d1
  v := v_3
  -- var p = d1 * v + d0
  p := (IntTy.addW (.u 64) (IntTy.mulW (.u 64) d1 v) d0)
  let join : UInt256 → UInt256 → FeM UInt256 := fun v p => do
    let mut v := v
    let mut p := p
    -- let t = v as u128 * d0 as u128
    let t := IntTy.mulW (.u 128) v d0
    -- let t1: u64 = (t >> 64).downcast_truncate()
    let t1 ← downcast_truncate_u128_u64 (t >>> (UInt256.ofNat 64))
    -- let t0: u64 = t.downcast_truncate()
    let t0 ← downcast_truncate_u128_u64 t
    -- p += t1
    p := (IntTy.addW (.u 64) p t1)
    let join_1 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
      let mut v := v
      let mut p := p
      return v
    -- if p < t1 {
    if decide (p.toNat < t1.toNat) then
      -- v -= 1
      v := (IntTy.subW (.u 64) v (UInt256.ofNat 1))
      let join_2 : UInt256 → UInt256 → Unit → FeM UInt256 := fun v p v2 => do
        let mut v := v
        let mut p := p
        join_1 v p
      let join_3 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
        let mut v := v
        let mut p := p
        -- v -= 1
        v := (IntTy.subW (.u 64) v (UInt256.ofNat 1))
        join_2 v p ()
      -- if p > d1 || (p == d1 && t0 >= d0) { v -= 1 }
      if decide (d1.toNat < p.toNat) then
        join_3 v p
      else
        let join_4 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
          let mut v := v
          let mut p := p
          join_2 v p ()
        if decide (p = d1) then
          if decide (d0.toNat ≤ t0.toNat) then
            join_3 v p
          else
            join_4 v p
        else
          join_4 v p
    else
      join_1 v p
  -- if p < d0 {
  if decide (p.toNat < d0.toNat) then
    -- v -= 1
    v := (IntTy.subW (.u 64) v (UInt256.ofNat 1))
    let join_5 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
      let mut v := v
      let mut p := p
      -- p -= d1
      p := (IntTy.subW (.u 64) p d1)
      join v p
    -- if p >= d1 {
    if decide (d1.toNat ≤ p.toNat) then
      -- v -= 1
      v := (IntTy.subW (.u 64) v (UInt256.ofNat 1))
      -- p -= d1
      p := (IntTy.subW (.u 64) p d1)
      join_5 v p
    else
      join_5 v p
  else
    join v p

/-- `Modulus::try_new` -/
def Modulus_try_new (value : MemArray_u64) (len : UInt256) : FeM Option_Modulus := do
  let mut limbs : MemArray_u64 := default
  let mut r1 : Range_Known_0_Unknown := default
  -- let count: usize = len.downcast_truncate()
  let count ← downcast_truncate_u64_usize len
  let join : MemArray_u64 → Range_Known_0_Unknown → Bool → FeM Option_Modulus := fun limbs r1 v2 => do
    let mut limbs := limbs
    let mut r1 := r1
    -- core::assert(len > 0 && count <= value.len() && value[(len - 1).downcast_truncate()] != 0)
    let _ ← «assert» v2
    -- let shift: u64 = (leading_zeros(value[(len - 1).downcast_truncate()] as u256) - 192).downcast_truncate()
    let v28 ← IntTy.sub (.u 64) len (UInt256.ofNat 1)
    let v29 ← downcast_truncate_u64_usize v28
    let v30 ← MemArray_index_ref_u64 value v29
    let v31 ← mloadInt (.u 64) v30
    let v35 ← checkedSub (IntTy.leadingZeros (.u 256) v31) (UInt256.ofNat 192)
    let shift ← downcast_truncate_u256_u64 v35
    -- var limbs = MemArray<u64>::try_new_zeroed(len.downcast_truncate())?
    let v38 ← downcast_truncate_u64_usize len
    let v39 ← MemArray_try_new_zeroed_u64 v38
    if match v39 with | .Some .. => true | _ => false then
      let limbs_41 := v39.Some_0
      limbs := limbs_41
      -- for i in 0..len.downcast_truncate() {
      let v44 ← downcast_truncate_u64_usize len
      r1 := (Range_Known_0_Unknown.mk () v44)
      let v48 ← Range_start_0 r1
      let rec loop (limbs : MemArray_u64) (r1 : Range_Known_0_Unknown) (v3 : UInt256) : FeM Option_Modulus := do
        let mut limbs := limbs
        let mut r1 := r1
        -- for i in 0..len.downcast_truncate() {
        let v51 ← Range_has_0 r1 v3
        if v51 then
          let v53 ← Range_at_0 r1 v3
          let i := v53
          -- let high = value[i] << shift
          let v55 ← MemArray_index_ref_u64 value i
          let v56 ← mloadInt (.u 64) v55
          let high := IntTy.shl (.u 64) v56 shift
          let join_1 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_Modulus := fun limbs r1 low => do
            let mut limbs := limbs
            let mut r1 := r1
            -- limbs[i] = high | low
            let (v76, v76_k) ← MemArray_index_mut_u64 limbs i
            mstoreInt (.u 64) v76 (high ||| low)
            let v76_0 ← v76_k v76
            limbs := v76_0
            -- for i in 0..len.downcast_truncate() {
            let v79 ← Range_next_0 r1 v3
            loop limbs r1 v79
          let join_2 : MemArray_u64 → Range_Known_0_Unknown → FeM Option_Modulus := fun limbs r1 => do
            let mut limbs := limbs
            let mut r1 := r1
            join_1 limbs r1 (UInt256.ofNat 0)
          -- let low = if shift == 0 || i == 0 { 0 } else { value[i - 1] >> (64 - shift) }
          if decide (shift = (UInt256.ofNat 0)) then
            join_2 limbs r1
          else
            if decide (i = (UInt256.ofNat 0)) then
              join_2 limbs r1
            else
              -- value[i - 1] >> (64 - shift)
              let v64 ← checkedSub i (UInt256.ofNat 1)
              let v65 ← MemArray_index_ref_u64 value v64
              let v66 ← mloadInt (.u 64) v65
              let v68 ← IntTy.sub (.u 64) (UInt256.ofNat 64) shift
              join_1 limbs r1 (v66 >>> v68)
        else
          let join_3 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_Modulus := fun limbs r1 reciprocal => do
            let mut limbs := limbs
            let mut r1 := r1
            -- Option::Some(Self { limbs, len, shift, reciprocal })
            let limbs_102 := limbs
            let v105 := (Modulus.mk limbs_102 len shift reciprocal)
            return Option_Modulus.Some v105
          -- let reciprocal = if count == 1 { reciprocal_2by1(limbs[0]) } else {
          if decide (count = (UInt256.ofNat 1)) then
            -- reciprocal_2by1(limbs[0])
            let v85 ← MemArray_index_ref_u64 limbs (UInt256.ofNat 0)
            let v86 ← mloadInt (.u 64) v85
            let v87 ← reciprocal_2by1 v86
            join_3 limbs r1 v87
          else
            -- reciprocal_3by2(limbs[count - 1], limbs[count - 2])
            let v90 ← checkedSub count (UInt256.ofNat 1)
            let v92 ← MemArray_index_ref_u64 limbs v90
            let v94 ← checkedSub count (UInt256.ofNat 2)
            let v96 ← MemArray_index_ref_u64 limbs v94
            let v97 ← mloadInt (.u 64) v92
            let v98 ← mloadInt (.u 64) v96
            let v99 ← reciprocal_3by2 v97 v98
            join_3 limbs r1 v99
      partial_fixpoint
      loop limbs r1 v48
    else
      match (← pure v39) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v39.Some_0
        pure ()
      | _ => pure ()
      return Option_Modulus.None
  let join_4 : MemArray_u64 → Range_Known_0_Unknown → FeM Option_Modulus := fun limbs r1 => do
    let mut limbs := limbs
    let mut r1 := r1
    join limbs r1 false
  -- core::assert(len > 0 && count <= value.len() && value[(len - 1).downcast_truncate()] != 0)
  if decide ((UInt256.ofNat 0).toNat < len.toNat) then
    let v11 ← MemArray_len_u64 value
    if decide (count.toNat ≤ v11.toNat) then
      let v15 ← IntTy.sub (.u 64) len (UInt256.ofNat 1)
      let v16 ← downcast_truncate_u64_usize v15
      let v17 ← MemArray_index_ref_u64 value v16
      let v18 ← mloadInt (.u 64) v17
      if decide (v18 ≠ (UInt256.ofNat 0)) then
        join limbs r1 true
      else
        join_4 limbs r1
    else
      join_4 limbs r1
  else
    join_4 limbs r1

/-- `Wide::of` -/
def Wide_of (limbs : MemArray_u64) (n : UInt256) : FeM Wide := do
  -- Wide {
  let v6 ← MemArray_index_ref_u64 limbs (UInt256.ofNat 0)
  let v7 ← mloadInt (.u 64) v6
  let join : UInt256 → FeM Wide := fun v2 => do
    let join_1 : UInt256 → FeM Wide := fun v3 => do
      let join_2 : UInt256 → FeM Wide := fun v4 => do
        return Wide.mk v7 v2 v3 v4
      -- Wide {
      if decide ((UInt256.ofNat 3).toNat < n.toNat) then
        -- limbs[3]
        let v30 ← MemArray_index_ref_u64 limbs (UInt256.ofNat 3)
        -- Wide {
        let v31 ← mloadInt (.u 64) v30
        join_2 v31
      else
        join_2 (UInt256.ofNat 0)
    -- Wide {
    if decide ((UInt256.ofNat 2).toNat < n.toNat) then
      -- limbs[2]
      let v21 ← MemArray_index_ref_u64 limbs (UInt256.ofNat 2)
      -- Wide {
      let v22 ← mloadInt (.u 64) v21
      join_1 v22
    else
      join_1 (UInt256.ofNat 0)
  if decide ((UInt256.ofNat 1).toNat < n.toNat) then
    -- limbs[1]
    let v12 ← MemArray_index_ref_u64 limbs (UInt256.ofNat 1)
    -- Wide {
    let v13 ← mloadInt (.u 64) v12
    join v13
  else
    join (UInt256.ofNat 0)

/-- `lib::modexp::window_width` -/
def window_width (bits : UInt256) : FeM UInt256 := do
  let mut best : UInt256 := default
  let mut best_cost : UInt256 := default
  let mut w : UInt256 := default
  -- var best: usize = 1
  best := (UInt256.ofNat 1)
  -- var best_cost = bits / 2
  let best_cost_5 ← checkedDiv bits (UInt256.ofNat 2)
  best_cost := best_cost_5
  -- var w: usize = 2
  w := (UInt256.ofNat 2)
  let rec loop (best : UInt256) (best_cost : UInt256) (w : UInt256) : FeM UInt256 := do
    let mut best := best
    let mut best_cost := best_cost
    let mut w := w
    -- while w <= 7 {
    if decide (w.toNat ≤ (UInt256.ofNat 7).toNat) then
      -- let cost = (1 << (w - 1)) + bits / (w + 1)
      let v17 ← checkedSub w (UInt256.ofNat 1)
      let v23 ← checkedAdd w (UInt256.ofNat 1)
      let v24 ← checkedDiv bits v23
      let cost ← checkedAdd ((UInt256.ofNat 1) <<< v17) v24
      let join : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun best best_cost w => do
        let mut best := best
        let mut best_cost := best_cost
        let mut w := w
        -- w += 1
        let w_40 ← checkedAdd w (UInt256.ofNat 1)
        w := w_40
        loop best best_cost w
      -- if cost < best_cost {
      if decide (cost.toNat < best_cost.toNat) then
        -- best = w
        best := w
        -- best_cost = cost
        best_cost := cost
        join best best_cost w
      else
        join best best_cost w
    else
      return best
  partial_fixpoint
  loop best best_cost w

def Small_double_1_k2 (self : Small_1) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (v45 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Wide := do
  -- let (d3, borrow) = sbb(s3, self.m.l3, borrow)
  let v51 ← sbb s3 self.m.l3 v45.snd
  let d3 := v51.fst
  -- let (_, borrow) = sbb(x.l3 >> 63, 0, borrow)
  let v60 ← sbb (x.l3 >>> (UInt256.ofNat 63)) (UInt256.ofNat 0) v51.snd
  let join : Wide → FeM Wide := fun v2 => do
    return v2
  -- if borrow == 1 { Wide { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v60.snd = (UInt256.ofNat 1)) then
    join (Wide.mk s0 s1 s2 s3)
  else
    join (Wide.mk d0 d1 d2 d3)

def Small_double_1_k1 (self : Small_1) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) : FeM Wide := do
  -- let s3 = (x.l3 << 1) | (x.l2 >> 63)
  let s3 := (IntTy.shl (.u 64) x.l3 (UInt256.ofNat 1)) ||| (x.l2 >>> (UInt256.ofNat 63))
  -- let (d0, borrow) = sbb(s0, self.m.l0, 0)
  let v33 ← sbb s0 self.m.l0 (UInt256.ofNat 0)
  let d0 := v33.fst
  -- let (d1, borrow) = sbb(s1, self.m.l1, borrow)
  let v39 ← sbb s1 self.m.l1 v33.snd
  let d1 := v39.fst
  -- let (d2, borrow) = sbb(s2, self.m.l2, borrow)
  let v45 ← sbb s2 self.m.l2 v39.snd
  let d2 := v45.fst
  Small_double_1_k2 self x s0 s1 s2 s3 d0 d1 v45 d2

/-- `Small<1>::double` -/
def Small_double_1 (self : Small_1) (x : Wide) : FeM Wide := do
  -- let s0 = x.l0 << 1
  let s0 := IntTy.shl (.u 64) x.l0 (UInt256.ofNat 1)
  -- let s1 = (x.l1 << 1) | (x.l0 >> 63)
  let s1 := (IntTy.shl (.u 64) x.l1 (UInt256.ofNat 1)) ||| (x.l0 >>> (UInt256.ofNat 63))
  -- let s2 = (x.l2 << 1) | (x.l1 >> 63)
  let s2 := (IntTy.shl (.u 64) x.l2 (UInt256.ofNat 1)) ||| (x.l1 >>> (UInt256.ofNat 63))
  Small_double_1_k1 self x s0 s1 s2

/-- `lib::montgomery::mac` -/
def mac (a : UInt256) (b : UInt256) (c : UInt256) (carry : UInt256) : FeM (UInt256 × UInt256) := do
  -- let t = (a as u128) * (b as u128) + (c as u128) + (carry as u128)
  let v8 ← IntTy.mul (.u 128) a b
  let v11 ← IntTy.add (.u 128) v8 c
  let t ← IntTy.add (.u 128) v11 carry
  -- (t.downcast_truncate(), (t >> 64).downcast_truncate())
  let v16 ← downcast_truncate_u128_u64 t
  let v19 ← downcast_truncate_u128_u64 (t >>> (UInt256.ofNat 64))
  return (v16, v19)

def Small_row2_1_k1 (self : Small_1) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (m : UInt256) : FeM (Wide × UInt256) := do
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v31 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v35 ← mac m self.m.l1 v15.fst v31.snd
  -- let (r1, c) = adc(t2, 0, c)
  let v41 ← adc v21.fst (UInt256.ofNat 0) v35.snd
  -- (Wide { l0: r0, l1: r1, l2: 0, l3: 0 }, t3 + c)
  let v48 := (Wide.mk v35.fst v41.fst (UInt256.ofNat 0) (UInt256.ofNat 0))
  return (v48, (IntTy.addW (.u 64) v21.snd v41.snd))

/-- `Small<1>::row2` -/
def Small_row2_1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, t3) = adc(top, c, 0)
  let v21 ← adc top v15.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  Small_row2_1_k1 self t0 v15 v21 m

def Small_row3_1_k2 (v28 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r2, c) = adc(t3, 0, c)
  let v54 ← adc v28.fst (UInt256.ofNat 0) v48.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: 0 }, t4 + c)
  let v60 := (Wide.mk v42.fst v48.fst v54.fst (UInt256.ofNat 0))
  return (v60, (IntTy.addW (.u 64) v28.snd v54.snd))

def Small_row3_1_k1 (self : Small_1) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v38 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v42 ← mac m self.m.l1 v15.fst v38.snd
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v48 ← mac m self.m.l2 v22.fst v42.snd
  Small_row3_1_k2 v28 v42 v48

/-- `Small<1>::row3` -/
def Small_row3_1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  -- let (t3, t4) = adc(top, c, 0)
  let v28 ← adc top v22.snd (UInt256.ofNat 0)
  Small_row3_1_k1 self t0 v15 v22 v28

def Small_row4_1_k2 (self : Small_1) (v22 : (UInt256 × UInt256)) (v29 : (UInt256 × UInt256)) (v35 : (UInt256 × UInt256)) (m : UInt256) (v49 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v55 ← mac m self.m.l2 v22.fst v49.snd
  -- let (r2, c) = mac(m, self.m.l3, t3, c)
  let v61 ← mac m self.m.l3 v29.fst v55.snd
  -- let (r3, c) = adc(t4, 0, c)
  let v67 ← adc v35.fst (UInt256.ofNat 0) v61.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: r3 }, t5 + c)
  let v74 := ((Wide.mk v49.fst v55.fst v61.fst v67.fst), (IntTy.addW (.u 64) v35.snd v67.snd))
  return v74

def Small_row4_1_k1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (t3, c) = mac(a.l3, b, t.l3, c)
  let v29 ← mac a.l3 b t.l3 v22.snd
  -- let (t4, t5) = adc(top, c, 0)
  let v35 ← adc top v29.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v45 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v49 ← mac m self.m.l1 v15.fst v45.snd
  Small_row4_1_k2 self v22 v29 v35 m v49

/-- `Small<1>::row4` -/
def Small_row4_1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  Small_row4_1_k1 self a b t top t0 v15 v22

/-- `Small<1>::mul` -/
def Small_mul_1 (self : Small_1) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  -- let zero = Wide { l0: 0, l1: 0, l2: 0, l3: 0 }
  let zero_12 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  zero := zero_12
  -- if L == 1 {
  if decide ((UInt256.ofNat 1) = (UInt256.ofNat 1)) then
    -- let t = a.l0 as u128 * b.l0 as u128
    let t_21 := IntTy.mulW (.u 128) a.l0 b.l0
    -- let low: u64 = t.downcast_truncate()
    let low ← downcast_truncate_u128_u64 t_21
    -- let product = q as u128 * self.m.l0 as u128
    let product := IntTy.mulW (.u 128) (IntTy.mulW (.u 64) low self.inverse) self.m.l0
    -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
    let v37 := IntTy.addW (.u 128) (t_21 >>> (UInt256.ofNat 64)) (product >>> (UInt256.ofNat 64))
    let join : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
      let sum := IntTy.addW (.u 128) v37 v3
      let join_1 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 reduced => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        -- return Wide { l0: reduced.downcast_truncate(), l1: 0, l2: 0, l3: 0 }
        let v55 ← downcast_truncate_u128_u64 reduced
        let v59 := (Wide.mk v55 (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
        return v59
      -- let reduced = if sum >= self.m.l0 as u128 { sum - self.m.l0 as u128 } else { sum }
      if decide (self.m.l0.toNat ≤ sum.toNat) then
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (IntTy.subW (.u 128) sum self.m.l0)
      else
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum
    if decide (low ≠ (UInt256.ofNat 0)) then
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 1)
    else
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 0)
  else
    -- if L == 2 {
    if decide ((UInt256.ofNat 1) = (UInt256.ofNat 2)) then
      -- let (t, top) = self.row2(a, b.l0, zero, 0)
      let v68 ← Small_row2_1 self a b.l0 zero (UInt256.ofNat 0)
      t_r1 := v68.fst
      -- let (t, top) = self.row2(a, b.l1, t, top)
      let v75 ← Small_row2_1 self a b.l1 t_r1 v68.snd
      let t_76 := v75.fst
      -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
      let v83 ← sbb t_76.l0 self.m.l0 (UInt256.ofNat 0)
      let d0_84 := v83.fst
      -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
      let v90 ← sbb t_76.l1 self.m.l1 v83.snd
      let d1_91 := v90.fst
      -- let (_, borrow) = sbb(top, 0, borrow)
      let v96 ← sbb v75.snd (UInt256.ofNat 0) v90.snd
      let join_2 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: 0, l3: 0 } }
      if decide (v96.snd = (UInt256.ofNat 1)) then
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_76
      else
        -- Wide { l0: d0, l1: d1, l2: 0, l3: 0 }
        let v104 := (Wide.mk d0_84 d1_91 (UInt256.ofNat 0) (UInt256.ofNat 0))
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v104
    else
      -- if L == 3 {
      if decide ((UInt256.ofNat 1) = (UInt256.ofNat 3)) then
        -- let (t, top) = self.row3(a, b.l0, zero, 0)
        let v114 ← Small_row3_1 self a b.l0 zero (UInt256.ofNat 0)
        t_r2 := v114.fst
        -- let (t, top) = self.row3(a, b.l1, t, top)
        let v121 ← Small_row3_1 self a b.l1 t_r2 v114.snd
        t_r3 := v121.fst
        -- let (t, top) = self.row3(a, b.l2, t, top)
        let v128 ← Small_row3_1 self a b.l2 t_r3 v121.snd
        let t_129 := v128.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v136 ← sbb t_129.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_137 := v136.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v143 ← sbb t_129.l1 self.m.l1 v136.snd
        let d1_144 := v143.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v150 ← sbb t_129.l2 self.m.l2 v143.snd
        let d2_151 := v150.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v156 ← sbb v128.snd (UInt256.ofNat 0) v150.snd
        let join_3 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: 0 } }
        if decide (v156.snd = (UInt256.ofNat 1)) then
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_129
        else
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_137 d1_144 d2_151 (UInt256.ofNat 0))
      else
        -- let (t, top) = self.row4(a, b.l0, zero, 0)
        let v170 ← Small_row4_1 self a b.l0 zero (UInt256.ofNat 0)
        t_r4 := v170.fst
        -- let (t, top) = self.row4(a, b.l1, t, top)
        let v177 ← Small_row4_1 self a b.l1 t_r4 v170.snd
        t_r5 := v177.fst
        -- let (t, top) = self.row4(a, b.l2, t, top)
        let v184 ← Small_row4_1 self a b.l2 t_r5 v177.snd
        t_r6 := v184.fst
        -- let (t, top) = self.row4(a, b.l3, t, top)
        let v191 ← Small_row4_1 self a b.l3 t_r6 v184.snd
        let t_192 := v191.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v199 ← sbb t_192.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_200 := v199.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v206 ← sbb t_192.l1 self.m.l1 v199.snd
        let d1_207 := v206.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v213 ← sbb t_192.l2 self.m.l2 v206.snd
        let d2_214 := v213.fst
        -- let (d3, borrow) = sbb(t.l3, self.m.l3, borrow)
        let v220 ← sbb t_192.l3 self.m.l3 v213.snd
        let d3 := v220.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v226 ← sbb v191.snd (UInt256.ofNat 0) v220.snd
        let join_4 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        -- if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
        if decide (v226.snd = (UInt256.ofNat 1)) then
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_192
        else
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_200 d1_207 d2_214 d3)

/-- `Small<1>::r2` -/
def Small_r2_1 (self : Small_1) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let join : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 top => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    -- let shift: u64 = (255 - leading_zeros(top as u256)).downcast_truncate()
    let v33 := (UInt256.ofNat 255) - (IntTy.leadingZeros (.u 256) top)
    let shift ← downcast_truncate_u256_u64 v33
    -- let bit: u64 = 1 << shift
    let bit := IntTy.shl (.u 64) (UInt256.ofNat 1) shift
    let join_1 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let join_2 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let join_3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let join_4 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            -- var x = Wide {
            x := (Wide.mk v4 v5 v6 v7)
            -- let doublings: usize = (64 - shift).downcast_truncate()
            let doublings ← downcast_truncate_u64_usize (IntTy.subW (.u 64) (UInt256.ofNat 64) shift)
            -- for _ in 0..doublings + L { x = self.double(x) }
            let v73 := (Range_Known_0_Unknown.mk () (doublings + (UInt256.ofNat 1)))
            r1 := v73
            let v75 ← Range_start_0 r1
            let rec loop (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              -- for _ in 0..doublings + L { x = self.double(x) }
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let _ ← Range_at_0 r1 v8
                -- x = self.double(x)
                r2 := x
                let x_84 ← Small_double_1 self r2
                x := x_84
                -- for _ in 0..doublings + L { x = self.double(x) }
                let v88 ← Range_next_0 r1 v8
                loop x r1 r2 r3 r4 r5 v88
              else
                -- for _ in 0..6 { x = self.mul(x, x) }
                r3 := (Range_Known_0_Known_6.mk () ())
                let v95 ← Range_start_0_6 r3
                let rec loop_1 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  -- for _ in 0..6 { x = self.mul(x, x) }
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let _ ← Range_at_0_6 r3 v9
                    -- x = self.mul(x, x)
                    r4 := x
                    r5 := x
                    let x_107 ← Small_mul_1 self r4 r5
                    x := x_107
                    -- for _ in 0..6 { x = self.mul(x, x) }
                    let v111 ← Range_next_0_6 r3 v9
                    loop_1 x r1 r2 r3 r4 r5 v111
                  else
                    return x
                partial_fixpoint
                loop_1 x r1 r2 r3 r4 r5 v95
            partial_fixpoint
            loop x r1 r2 r3 r4 r5 v75
          -- var x = Wide {
          if decide ((UInt256.ofNat 1) = (UInt256.ofNat 4)) then
            join_4 x r1 r2 r3 r4 r5 bit
          else
            join_4 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
        -- var x = Wide {
        if decide ((UInt256.ofNat 1) = (UInt256.ofNat 3)) then
          join_3 x r1 r2 r3 r4 r5 bit
        else
          join_3 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
      -- var x = Wide {
      if decide ((UInt256.ofNat 1) = (UInt256.ofNat 2)) then
        join_2 x r1 r2 r3 r4 r5 bit
      else
        join_2 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
    -- var x = Wide {
    if decide ((UInt256.ofNat 1) = (UInt256.ofNat 1)) then
      join_1 x r1 r2 r3 r4 r5 bit
    else
      join_1 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
  -- let top = if L == 1 { self.m.l0 } else if L == 2 { self.m.l1 } else if L == 3 { self.m.l2 } else { self.m.l3 }
  if decide ((UInt256.ofNat 1) = (UInt256.ofNat 1)) then
    join x r1 r2 r3 r4 r5 self.m.l0
  else
    let join_5 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      join x r1 r2 r3 r4 r5 v2
    if decide ((UInt256.ofNat 1) = (UInt256.ofNat 2)) then
      join_5 x r1 r2 r3 r4 r5 self.m.l1
    else
      let join_6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        join_5 x r1 r2 r3 r4 r5 v3
      if decide ((UInt256.ofNat 1) = (UInt256.ofNat 3)) then
        join_6 x r1 r2 r3 r4 r5 self.m.l2
      else
        join_6 x r1 r2 r3 r4 r5 self.m.l3

/-- `lib::bignum::limb` -/
def limb (limbs : UInt256) (index : UInt256) : FeM UInt256 := do
  return limbs + (index * (UInt256.ofNat 8))

/-- `Exponent::bit` -/
def Exponent_bit (self : Exponent) (index : UInt256) : FeM UInt256 := do
  -- let position = self.total - 1 - index
  let position := (self.total - (UInt256.ofNat 1)) - index
  -- let shift: u64 = (position & 63).downcast_truncate()
  let shift ← downcast_truncate_usize_u64 (position &&& (UInt256.ofNat 63))
  -- *limb(self.limbs.ptr(), position >> 6)
  let v14 ← MemArray_ptr_u64 self.limbs
  let v17 ← limb v14 (position >>> (UInt256.ofNat 6))
  let v18 ← mloadInt (.u 64) v17
  -- ((unsafe { *limb(self.limbs.ptr(), position >> 6) } >> shift) & 1).downcast_truncate()
  let v22 ← downcast_truncate_u64_usize ((v18 >>> shift) &&& (UInt256.ofNat 1))
  return v22

/-- `Exponent::window` -/
def Exponent_window (self : Exponent) (index : UInt256) (width : UInt256) : FeM Window := do
  let mut «end» : UInt256 := default
  let mut digit : UInt256 := default
  let mut r2 : Range_Unknown_Unknown := default
  -- if self.bit(index) == 0 { return Window { end: index + 1, entry: 0, odd: false } }
  let v5 ← Exponent_bit self index
  if decide (v5 = (UInt256.ofNat 0)) then
    -- return Window { end: index + 1, entry: 0, odd: false }
    let v13 := (Window.mk (index + (UInt256.ofNat 1)) (UInt256.ofNat 0) false)
    return v13
  else
    let join : UInt256 → UInt256 → Range_Unknown_Unknown → UInt256 → FeM Window := fun «end» digit r2 end_3 => do
      let mut «end» := «end»
      let mut digit := digit
      let mut r2 := r2
      -- var end = if self.total - index < width { self.total } else { index + width }
      «end» := end_3
      let rec loop («end» : UInt256) (digit : UInt256) (r2 : Range_Unknown_Unknown) : FeM Window := do
        let mut «end» := «end»
        let mut digit := digit
        let mut r2 := r2
        -- while self.bit(end - 1) == 0 { end -= 1 }
        let v32 ← Exponent_bit self («end» - (UInt256.ofNat 1))
        if decide (v32 = (UInt256.ofNat 0)) then
          -- end -= 1
          «end» := («end» - (UInt256.ofNat 1))
          loop «end» digit r2
        else
          -- var digit: usize = 0
          digit := (UInt256.ofNat 0)
          -- for bit in index..end { digit = (digit << 1) | self.bit(bit) }
          r2 := (Range_Unknown_Unknown.mk index «end»)
          let v47 ← Range_Unknown_Unknown_start r2
          let rec loop_1 («end» : UInt256) (digit : UInt256) (r2 : Range_Unknown_Unknown) (v4 : UInt256) : FeM Window := do
            let mut «end» := «end»
            let mut digit := digit
            let mut r2 := r2
            -- for bit in index..end { digit = (digit << 1) | self.bit(bit) }
            let v50 ← Range_Unknown_Unknown_has r2 v4
            if v50 then
              let v52 ← Range_Unknown_Unknown_at r2 v4
              -- digit = (digit << 1) | self.bit(bit)
              let v57 := digit <<< (UInt256.ofNat 1)
              let v58 ← Exponent_bit self v52
              digit := (v57 ||| v58)
              -- for bit in index..end { digit = (digit << 1) | self.bit(bit) }
              let v63 ← Range_Unknown_Unknown_next r2 v4
              loop_1 «end» digit r2 v63
            else
              -- Window { end, entry: (digit - 1) / 2, odd: true }
              let v72 ← checkedDiv (digit - (UInt256.ofNat 1)) (UInt256.ofNat 2)
              return Window.mk «end» v72 true
          partial_fixpoint
          loop_1 «end» digit r2 v47
      partial_fixpoint
      loop «end» digit r2
    -- var end = if self.total - index < width { self.total } else { index + width }
    if decide ((self.total - index).toNat < width.toNat) then
      join «end» digit r2 self.total
    else
      join «end» digit r2 (index + width)

/-- `lib::modexp::small_power<1>` -/
def small_power_1 (exponent : Exponent) (base : MemArray_u64) (value : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut inverse : UInt256 := default
  let mut small : Small_1 := default
  let mut table : (Vector Wide 64) := default
  let mut squared : Wide := default
  let mut power : Wide := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut result : MemArray_u64 := default
  let mut r8 : Range_Known_0_Known_5 := default
  let mut r9 : Wide := default
  let mut r10 : Wide := default
  let mut r11 : Wide := default
  let mut r12 : Wide := default
  let mut r13 : Range_Known_1_Unknown := default
  let mut r14 : Wide := default
  let mut r15 : Range_Unknown_Unknown := default
  let mut r16 : Wide := default
  let mut r17 : Wide := default
  let mut r18 : Wide := default
  let mut r19 : Wide := default
  let mut r20 : Wide := default
  let mut r21 : Wide := default
  -- let low = value[0]
  let v8 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
  let low ← mloadInt (.u 64) v8
  -- var inverse = low
  inverse := low
  -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
  r8 := (Range_Known_0_Known_5.mk () ())
  let v16 ← Range_start_0_5 r8
  let rec loop (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
    let mut inverse := inverse
    let mut small := small
    let mut table := table
    let mut squared := squared
    let mut power := power
    let mut index := index
    let mut started := started
    let mut result := result
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
    -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let _ ← Range_at_0_5 r8 v3
      -- inverse = inverse * (2 - low * inverse)
      let v27 := IntTy.subW (.u 64) (UInt256.ofNat 2) (IntTy.mulW (.u 64) low inverse)
      inverse := (IntTy.mulW (.u 64) inverse v27)
      -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
      let v32 ← Range_next_0_5 r8 v3
      loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v32
    else
      -- let small = Small<L> { m: Wide::of(value, L), inverse: 0 - inverse }
      let v35 ← Wide_of value (UInt256.ofNat 1)
      let small_39 := (Small_1.mk v35 (IntTy.subW (.u 64) (UInt256.ofNat 0) inverse))
      small := small_39
      -- let width = window_width(exponent.total - exponent.top)
      let width ← window_width (exponent.total - exponent.top)
      -- let entries: usize = (1 << (width - 1)).downcast_truncate()
      let entries ← downcast_truncate_usize_usize ((UInt256.ofNat 1) <<< (width - (UInt256.ofNat 1)))
      -- var table: [Wide; 64] = [Wide { l0: 0, l1: 0, l2: 0, l3: 0 }; 64]
      let v56 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
      table := (Vector.replicate 64 v56)
      -- table[0] = small.mul(Wide::of(base, L), small.r2())
      let v60 ← Wide_of base (UInt256.ofNat 1)
      r9 := v60
      let v62 ← Small_r2_1 small
      r10 := v62
      let v66 ← Small_mul_1 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let join : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
        let mut inverse := inverse
        let mut small := small
        let mut table := table
        let mut squared := squared
        let mut power := power
        let mut index := index
        let mut started := started
        let mut result := result
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
        -- var power = table[0]
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        power := table[v106.toNat]!
        -- var index = exponent.top
        index := exponent.top
        -- var started = false
        started := false
        let rec loop_1 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- while index < exponent.total {
          if decide (index.toNat < exponent.total.toNat) then
            -- let window = exponent.window(index, width)
            let window ← Exponent_window exponent index width
            let rec loop_2 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
              let mut inverse := inverse
              let mut small := small
              let mut table := table
              let mut squared := squared
              let mut power := power
              let mut index := index
              let mut started := started
              let mut result := result
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
              let join_1 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- index = window.end
                index := window.«end»
                loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              -- if window.odd {
              if window.odd then
                let join_2 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 power_6 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                  power := power_6
                  -- started = true
                  started := true
                  join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                if started then
                  -- small.mul(power, table[window.entry])
                  r18 := power
                  let v155 := window.entry
                  checkIndex v155 64
                  r19 := table[v155.toNat]!
                  let v160 ← Small_mul_1 small r18 r19
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v160
                else
                  -- table[window.entry]
                  let v162 := window.entry
                  checkIndex v162 64
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 table[v162.toNat]!
              else
                join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            -- if started {
            if started then
              -- for _ in index..window.end { power = small.mul(power, power) }
              r15 := (Range_Unknown_Unknown.mk index window.«end»)
              let v128 ← Range_Unknown_Unknown_start r15
              let rec loop_3 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- for _ in index..window.end { power = small.mul(power, power) }
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let _ ← Range_Unknown_Unknown_at r15 v5
                  -- power = small.mul(power, power)
                  r16 := power
                  r17 := power
                  let power_141 ← Small_mul_1 small r16 r17
                  power := power_141
                  -- for _ in index..window.end { power = small.mul(power, power) }
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v145
                else
                  loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v128
            else
              loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            -- let out = small.mul(power, Wide { l0: 1, l1: 0, l2: 0, l3: 0 })
            r20 := power
            let v182 := (Wide.mk (UInt256.ofNat 1) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
            r21 := v182
            let out ← Small_mul_1 small r20 r21
            -- var result = MemArray<u64>::try_new_zeroed(L)?
            let v189 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 1)
            if match v189 with | .Some .. => true | _ => false then
              let result_191 := v189.Some_0
              result := result_191
              -- result[0] = out.l0
              let (v196, v196_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 0)
              mstoreInt (.u 64) v196 out.l0
              let v196_0 ← v196_k v196
              result := v196_0
              -- if L > 1 { result[1] = out.l1 }
              let v200 := decide ((UInt256.ofNat 1).toNat < (UInt256.ofNat 1).toNat)
              let join_3 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- if L > 2 { result[2] = out.l2 }
                let v211 := decide ((UInt256.ofNat 2).toNat < (UInt256.ofNat 1).toNat)
                let join_4 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- if L > 3 { result[3] = out.l3 }
                  let v222 := decide ((UInt256.ofNat 3).toNat < (UInt256.ofNat 1).toNat)
                  let join_5 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                    let mut inverse := inverse
                    let mut small := small
                    let mut table := table
                    let mut squared := squared
                    let mut power := power
                    let mut index := index
                    let mut started := started
                    let mut result := result
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
                    -- Option::Some(result)
                    let result_231 := result
                    return Option_MemArray_u64.Some result_231
                  if v222 then
                    -- result[3] = out.l3
                    let (v226, v226_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 3)
                    mstoreInt (.u 64) v226 out.l3
                    let v226_0 ← v226_k v226
                    result := v226_0
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  -- result[2] = out.l2
                  let (v215, v215_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 2)
                  mstoreInt (.u 64) v215 out.l2
                  let v215_0 ← v215_k v215
                  result := v215_0
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                -- result[1] = out.l1
                let (v204, v204_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 1)
                mstoreInt (.u 64) v204 out.l1
                let v204_0 ← v204_k v204
                result := v204_0
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return Option_MemArray_u64.None
        partial_fixpoint
        loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      -- if entries > 1 {
      if decide ((UInt256.ofNat 1).toNat < entries.toNat) then
        -- let squared = small.mul(table[0], table[0])
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        r11 := table[v71.toNat]!
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        r12 := table[v73.toNat]!
        let squared_78 ← Small_mul_1 small r11 r12
        squared := squared_78
        -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
        r13 := (Range_Known_1_Unknown.mk () entries)
        let v84 ← Range_start_1 r13
        let rec loop_4 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            -- table[k] = small.mul(table[k - 1], squared)
            let v92 := k - (UInt256.ofNat 1)
            checkIndex v92 64
            r14 := table[v92.toNat]!
            let v97 ← Small_mul_1 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
            let v100 ← Range_next_1 r13 v4
            loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v100
          else
            join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v84
      else
        join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v16

def Small_double_2_k2 (self : Small_2) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (v45 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Wide := do
  -- let (d3, borrow) = sbb(s3, self.m.l3, borrow)
  let v51 ← sbb s3 self.m.l3 v45.snd
  let d3 := v51.fst
  -- let (_, borrow) = sbb(x.l3 >> 63, 0, borrow)
  let v60 ← sbb (x.l3 >>> (UInt256.ofNat 63)) (UInt256.ofNat 0) v51.snd
  let join : Wide → FeM Wide := fun v2 => do
    return v2
  -- if borrow == 1 { Wide { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v60.snd = (UInt256.ofNat 1)) then
    join (Wide.mk s0 s1 s2 s3)
  else
    join (Wide.mk d0 d1 d2 d3)

def Small_double_2_k1 (self : Small_2) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) : FeM Wide := do
  -- let s3 = (x.l3 << 1) | (x.l2 >> 63)
  let s3 := (IntTy.shl (.u 64) x.l3 (UInt256.ofNat 1)) ||| (x.l2 >>> (UInt256.ofNat 63))
  -- let (d0, borrow) = sbb(s0, self.m.l0, 0)
  let v33 ← sbb s0 self.m.l0 (UInt256.ofNat 0)
  let d0 := v33.fst
  -- let (d1, borrow) = sbb(s1, self.m.l1, borrow)
  let v39 ← sbb s1 self.m.l1 v33.snd
  let d1 := v39.fst
  -- let (d2, borrow) = sbb(s2, self.m.l2, borrow)
  let v45 ← sbb s2 self.m.l2 v39.snd
  let d2 := v45.fst
  Small_double_2_k2 self x s0 s1 s2 s3 d0 d1 v45 d2

/-- `Small<2>::double` -/
def Small_double_2 (self : Small_2) (x : Wide) : FeM Wide := do
  -- let s0 = x.l0 << 1
  let s0 := IntTy.shl (.u 64) x.l0 (UInt256.ofNat 1)
  -- let s1 = (x.l1 << 1) | (x.l0 >> 63)
  let s1 := (IntTy.shl (.u 64) x.l1 (UInt256.ofNat 1)) ||| (x.l0 >>> (UInt256.ofNat 63))
  -- let s2 = (x.l2 << 1) | (x.l1 >> 63)
  let s2 := (IntTy.shl (.u 64) x.l2 (UInt256.ofNat 1)) ||| (x.l1 >>> (UInt256.ofNat 63))
  Small_double_2_k1 self x s0 s1 s2

def Small_row2_2_k1 (self : Small_2) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (m : UInt256) : FeM (Wide × UInt256) := do
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v31 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v35 ← mac m self.m.l1 v15.fst v31.snd
  -- let (r1, c) = adc(t2, 0, c)
  let v41 ← adc v21.fst (UInt256.ofNat 0) v35.snd
  -- (Wide { l0: r0, l1: r1, l2: 0, l3: 0 }, t3 + c)
  let v48 := (Wide.mk v35.fst v41.fst (UInt256.ofNat 0) (UInt256.ofNat 0))
  return (v48, (IntTy.addW (.u 64) v21.snd v41.snd))

/-- `Small<2>::row2` -/
def Small_row2_2 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, t3) = adc(top, c, 0)
  let v21 ← adc top v15.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  Small_row2_2_k1 self t0 v15 v21 m

def Small_row3_2_k2 (v28 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r2, c) = adc(t3, 0, c)
  let v54 ← adc v28.fst (UInt256.ofNat 0) v48.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: 0 }, t4 + c)
  let v60 := (Wide.mk v42.fst v48.fst v54.fst (UInt256.ofNat 0))
  return (v60, (IntTy.addW (.u 64) v28.snd v54.snd))

def Small_row3_2_k1 (self : Small_2) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v38 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v42 ← mac m self.m.l1 v15.fst v38.snd
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v48 ← mac m self.m.l2 v22.fst v42.snd
  Small_row3_2_k2 v28 v42 v48

/-- `Small<2>::row3` -/
def Small_row3_2 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  -- let (t3, t4) = adc(top, c, 0)
  let v28 ← adc top v22.snd (UInt256.ofNat 0)
  Small_row3_2_k1 self t0 v15 v22 v28

def Small_row4_2_k2 (self : Small_2) (v22 : (UInt256 × UInt256)) (v29 : (UInt256 × UInt256)) (v35 : (UInt256 × UInt256)) (m : UInt256) (v49 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v55 ← mac m self.m.l2 v22.fst v49.snd
  -- let (r2, c) = mac(m, self.m.l3, t3, c)
  let v61 ← mac m self.m.l3 v29.fst v55.snd
  -- let (r3, c) = adc(t4, 0, c)
  let v67 ← adc v35.fst (UInt256.ofNat 0) v61.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: r3 }, t5 + c)
  let v74 := ((Wide.mk v49.fst v55.fst v61.fst v67.fst), (IntTy.addW (.u 64) v35.snd v67.snd))
  return v74

def Small_row4_2_k1 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (t3, c) = mac(a.l3, b, t.l3, c)
  let v29 ← mac a.l3 b t.l3 v22.snd
  -- let (t4, t5) = adc(top, c, 0)
  let v35 ← adc top v29.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v45 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v49 ← mac m self.m.l1 v15.fst v45.snd
  Small_row4_2_k2 self v22 v29 v35 m v49

/-- `Small<2>::row4` -/
def Small_row4_2 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  Small_row4_2_k1 self a b t top t0 v15 v22

/-- `Small<2>::mul` -/
def Small_mul_2 (self : Small_2) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  -- let zero = Wide { l0: 0, l1: 0, l2: 0, l3: 0 }
  let zero_12 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  zero := zero_12
  -- if L == 1 {
  if decide ((UInt256.ofNat 2) = (UInt256.ofNat 1)) then
    -- let t = a.l0 as u128 * b.l0 as u128
    let t_21 := IntTy.mulW (.u 128) a.l0 b.l0
    -- let low: u64 = t.downcast_truncate()
    let low ← downcast_truncate_u128_u64 t_21
    -- let product = q as u128 * self.m.l0 as u128
    let product := IntTy.mulW (.u 128) (IntTy.mulW (.u 64) low self.inverse) self.m.l0
    -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
    let v37 := IntTy.addW (.u 128) (t_21 >>> (UInt256.ofNat 64)) (product >>> (UInt256.ofNat 64))
    let join : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
      let sum := IntTy.addW (.u 128) v37 v3
      let join_1 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 reduced => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        -- return Wide { l0: reduced.downcast_truncate(), l1: 0, l2: 0, l3: 0 }
        let v55 ← downcast_truncate_u128_u64 reduced
        let v59 := (Wide.mk v55 (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
        return v59
      -- let reduced = if sum >= self.m.l0 as u128 { sum - self.m.l0 as u128 } else { sum }
      if decide (self.m.l0.toNat ≤ sum.toNat) then
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (IntTy.subW (.u 128) sum self.m.l0)
      else
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum
    if decide (low ≠ (UInt256.ofNat 0)) then
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 1)
    else
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 0)
  else
    -- if L == 2 {
    if decide ((UInt256.ofNat 2) = (UInt256.ofNat 2)) then
      -- let (t, top) = self.row2(a, b.l0, zero, 0)
      let v68 ← Small_row2_2 self a b.l0 zero (UInt256.ofNat 0)
      t_r1 := v68.fst
      -- let (t, top) = self.row2(a, b.l1, t, top)
      let v75 ← Small_row2_2 self a b.l1 t_r1 v68.snd
      let t_76 := v75.fst
      -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
      let v83 ← sbb t_76.l0 self.m.l0 (UInt256.ofNat 0)
      let d0_84 := v83.fst
      -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
      let v90 ← sbb t_76.l1 self.m.l1 v83.snd
      let d1_91 := v90.fst
      -- let (_, borrow) = sbb(top, 0, borrow)
      let v96 ← sbb v75.snd (UInt256.ofNat 0) v90.snd
      let join_2 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: 0, l3: 0 } }
      if decide (v96.snd = (UInt256.ofNat 1)) then
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_76
      else
        -- Wide { l0: d0, l1: d1, l2: 0, l3: 0 }
        let v104 := (Wide.mk d0_84 d1_91 (UInt256.ofNat 0) (UInt256.ofNat 0))
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v104
    else
      -- if L == 3 {
      if decide ((UInt256.ofNat 2) = (UInt256.ofNat 3)) then
        -- let (t, top) = self.row3(a, b.l0, zero, 0)
        let v114 ← Small_row3_2 self a b.l0 zero (UInt256.ofNat 0)
        t_r2 := v114.fst
        -- let (t, top) = self.row3(a, b.l1, t, top)
        let v121 ← Small_row3_2 self a b.l1 t_r2 v114.snd
        t_r3 := v121.fst
        -- let (t, top) = self.row3(a, b.l2, t, top)
        let v128 ← Small_row3_2 self a b.l2 t_r3 v121.snd
        let t_129 := v128.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v136 ← sbb t_129.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_137 := v136.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v143 ← sbb t_129.l1 self.m.l1 v136.snd
        let d1_144 := v143.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v150 ← sbb t_129.l2 self.m.l2 v143.snd
        let d2_151 := v150.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v156 ← sbb v128.snd (UInt256.ofNat 0) v150.snd
        let join_3 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: 0 } }
        if decide (v156.snd = (UInt256.ofNat 1)) then
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_129
        else
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_137 d1_144 d2_151 (UInt256.ofNat 0))
      else
        -- let (t, top) = self.row4(a, b.l0, zero, 0)
        let v170 ← Small_row4_2 self a b.l0 zero (UInt256.ofNat 0)
        t_r4 := v170.fst
        -- let (t, top) = self.row4(a, b.l1, t, top)
        let v177 ← Small_row4_2 self a b.l1 t_r4 v170.snd
        t_r5 := v177.fst
        -- let (t, top) = self.row4(a, b.l2, t, top)
        let v184 ← Small_row4_2 self a b.l2 t_r5 v177.snd
        t_r6 := v184.fst
        -- let (t, top) = self.row4(a, b.l3, t, top)
        let v191 ← Small_row4_2 self a b.l3 t_r6 v184.snd
        let t_192 := v191.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v199 ← sbb t_192.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_200 := v199.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v206 ← sbb t_192.l1 self.m.l1 v199.snd
        let d1_207 := v206.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v213 ← sbb t_192.l2 self.m.l2 v206.snd
        let d2_214 := v213.fst
        -- let (d3, borrow) = sbb(t.l3, self.m.l3, borrow)
        let v220 ← sbb t_192.l3 self.m.l3 v213.snd
        let d3 := v220.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v226 ← sbb v191.snd (UInt256.ofNat 0) v220.snd
        let join_4 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        -- if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
        if decide (v226.snd = (UInt256.ofNat 1)) then
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_192
        else
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_200 d1_207 d2_214 d3)

/-- `Small<2>::r2` -/
def Small_r2_2 (self : Small_2) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let join : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 top => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    -- let shift: u64 = (255 - leading_zeros(top as u256)).downcast_truncate()
    let v33 := (UInt256.ofNat 255) - (IntTy.leadingZeros (.u 256) top)
    let shift ← downcast_truncate_u256_u64 v33
    -- let bit: u64 = 1 << shift
    let bit := IntTy.shl (.u 64) (UInt256.ofNat 1) shift
    let join_1 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let join_2 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let join_3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let join_4 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            -- var x = Wide {
            x := (Wide.mk v4 v5 v6 v7)
            -- let doublings: usize = (64 - shift).downcast_truncate()
            let doublings ← downcast_truncate_u64_usize (IntTy.subW (.u 64) (UInt256.ofNat 64) shift)
            -- for _ in 0..doublings + L { x = self.double(x) }
            let v73 := (Range_Known_0_Unknown.mk () (doublings + (UInt256.ofNat 2)))
            r1 := v73
            let v75 ← Range_start_0 r1
            let rec loop (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              -- for _ in 0..doublings + L { x = self.double(x) }
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let _ ← Range_at_0 r1 v8
                -- x = self.double(x)
                r2 := x
                let x_84 ← Small_double_2 self r2
                x := x_84
                -- for _ in 0..doublings + L { x = self.double(x) }
                let v88 ← Range_next_0 r1 v8
                loop x r1 r2 r3 r4 r5 v88
              else
                -- for _ in 0..6 { x = self.mul(x, x) }
                r3 := (Range_Known_0_Known_6.mk () ())
                let v95 ← Range_start_0_6 r3
                let rec loop_1 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  -- for _ in 0..6 { x = self.mul(x, x) }
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let _ ← Range_at_0_6 r3 v9
                    -- x = self.mul(x, x)
                    r4 := x
                    r5 := x
                    let x_107 ← Small_mul_2 self r4 r5
                    x := x_107
                    -- for _ in 0..6 { x = self.mul(x, x) }
                    let v111 ← Range_next_0_6 r3 v9
                    loop_1 x r1 r2 r3 r4 r5 v111
                  else
                    return x
                partial_fixpoint
                loop_1 x r1 r2 r3 r4 r5 v95
            partial_fixpoint
            loop x r1 r2 r3 r4 r5 v75
          -- var x = Wide {
          if decide ((UInt256.ofNat 2) = (UInt256.ofNat 4)) then
            join_4 x r1 r2 r3 r4 r5 bit
          else
            join_4 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
        -- var x = Wide {
        if decide ((UInt256.ofNat 2) = (UInt256.ofNat 3)) then
          join_3 x r1 r2 r3 r4 r5 bit
        else
          join_3 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
      -- var x = Wide {
      if decide ((UInt256.ofNat 2) = (UInt256.ofNat 2)) then
        join_2 x r1 r2 r3 r4 r5 bit
      else
        join_2 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
    -- var x = Wide {
    if decide ((UInt256.ofNat 2) = (UInt256.ofNat 1)) then
      join_1 x r1 r2 r3 r4 r5 bit
    else
      join_1 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
  -- let top = if L == 1 { self.m.l0 } else if L == 2 { self.m.l1 } else if L == 3 { self.m.l2 } else { self.m.l3 }
  if decide ((UInt256.ofNat 2) = (UInt256.ofNat 1)) then
    join x r1 r2 r3 r4 r5 self.m.l0
  else
    let join_5 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      join x r1 r2 r3 r4 r5 v2
    if decide ((UInt256.ofNat 2) = (UInt256.ofNat 2)) then
      join_5 x r1 r2 r3 r4 r5 self.m.l1
    else
      let join_6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        join_5 x r1 r2 r3 r4 r5 v3
      if decide ((UInt256.ofNat 2) = (UInt256.ofNat 3)) then
        join_6 x r1 r2 r3 r4 r5 self.m.l2
      else
        join_6 x r1 r2 r3 r4 r5 self.m.l3

/-- `lib::modexp::small_power<2>` -/
def small_power_2 (exponent : Exponent) (base : MemArray_u64) (value : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut inverse : UInt256 := default
  let mut small : Small_2 := default
  let mut table : (Vector Wide 64) := default
  let mut squared : Wide := default
  let mut power : Wide := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut result : MemArray_u64 := default
  let mut r8 : Range_Known_0_Known_5 := default
  let mut r9 : Wide := default
  let mut r10 : Wide := default
  let mut r11 : Wide := default
  let mut r12 : Wide := default
  let mut r13 : Range_Known_1_Unknown := default
  let mut r14 : Wide := default
  let mut r15 : Range_Unknown_Unknown := default
  let mut r16 : Wide := default
  let mut r17 : Wide := default
  let mut r18 : Wide := default
  let mut r19 : Wide := default
  let mut r20 : Wide := default
  let mut r21 : Wide := default
  -- let low = value[0]
  let v8 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
  let low ← mloadInt (.u 64) v8
  -- var inverse = low
  inverse := low
  -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
  r8 := (Range_Known_0_Known_5.mk () ())
  let v16 ← Range_start_0_5 r8
  let rec loop (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
    let mut inverse := inverse
    let mut small := small
    let mut table := table
    let mut squared := squared
    let mut power := power
    let mut index := index
    let mut started := started
    let mut result := result
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
    -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let _ ← Range_at_0_5 r8 v3
      -- inverse = inverse * (2 - low * inverse)
      let v27 := IntTy.subW (.u 64) (UInt256.ofNat 2) (IntTy.mulW (.u 64) low inverse)
      inverse := (IntTy.mulW (.u 64) inverse v27)
      -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
      let v32 ← Range_next_0_5 r8 v3
      loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v32
    else
      -- let small = Small<L> { m: Wide::of(value, L), inverse: 0 - inverse }
      let v35 ← Wide_of value (UInt256.ofNat 2)
      let small_39 := (Small_2.mk v35 (IntTy.subW (.u 64) (UInt256.ofNat 0) inverse))
      small := small_39
      -- let width = window_width(exponent.total - exponent.top)
      let width ← window_width (exponent.total - exponent.top)
      -- let entries: usize = (1 << (width - 1)).downcast_truncate()
      let entries ← downcast_truncate_usize_usize ((UInt256.ofNat 1) <<< (width - (UInt256.ofNat 1)))
      -- var table: [Wide; 64] = [Wide { l0: 0, l1: 0, l2: 0, l3: 0 }; 64]
      let v56 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
      table := (Vector.replicate 64 v56)
      -- table[0] = small.mul(Wide::of(base, L), small.r2())
      let v60 ← Wide_of base (UInt256.ofNat 2)
      r9 := v60
      let v62 ← Small_r2_2 small
      r10 := v62
      let v66 ← Small_mul_2 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let join : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
        let mut inverse := inverse
        let mut small := small
        let mut table := table
        let mut squared := squared
        let mut power := power
        let mut index := index
        let mut started := started
        let mut result := result
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
        -- var power = table[0]
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        power := table[v106.toNat]!
        -- var index = exponent.top
        index := exponent.top
        -- var started = false
        started := false
        let rec loop_1 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- while index < exponent.total {
          if decide (index.toNat < exponent.total.toNat) then
            -- let window = exponent.window(index, width)
            let window ← Exponent_window exponent index width
            let rec loop_2 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
              let mut inverse := inverse
              let mut small := small
              let mut table := table
              let mut squared := squared
              let mut power := power
              let mut index := index
              let mut started := started
              let mut result := result
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
              let join_1 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- index = window.end
                index := window.«end»
                loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              -- if window.odd {
              if window.odd then
                let join_2 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 power_6 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                  power := power_6
                  -- started = true
                  started := true
                  join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                if started then
                  -- small.mul(power, table[window.entry])
                  r18 := power
                  let v155 := window.entry
                  checkIndex v155 64
                  r19 := table[v155.toNat]!
                  let v160 ← Small_mul_2 small r18 r19
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v160
                else
                  -- table[window.entry]
                  let v162 := window.entry
                  checkIndex v162 64
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 table[v162.toNat]!
              else
                join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            -- if started {
            if started then
              -- for _ in index..window.end { power = small.mul(power, power) }
              r15 := (Range_Unknown_Unknown.mk index window.«end»)
              let v128 ← Range_Unknown_Unknown_start r15
              let rec loop_3 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- for _ in index..window.end { power = small.mul(power, power) }
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let _ ← Range_Unknown_Unknown_at r15 v5
                  -- power = small.mul(power, power)
                  r16 := power
                  r17 := power
                  let power_141 ← Small_mul_2 small r16 r17
                  power := power_141
                  -- for _ in index..window.end { power = small.mul(power, power) }
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v145
                else
                  loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v128
            else
              loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            -- let out = small.mul(power, Wide { l0: 1, l1: 0, l2: 0, l3: 0 })
            r20 := power
            let v182 := (Wide.mk (UInt256.ofNat 1) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
            r21 := v182
            let out ← Small_mul_2 small r20 r21
            -- var result = MemArray<u64>::try_new_zeroed(L)?
            let v189 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 2)
            if match v189 with | .Some .. => true | _ => false then
              let result_191 := v189.Some_0
              result := result_191
              -- result[0] = out.l0
              let (v196, v196_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 0)
              mstoreInt (.u 64) v196 out.l0
              let v196_0 ← v196_k v196
              result := v196_0
              -- if L > 1 { result[1] = out.l1 }
              let v200 := decide ((UInt256.ofNat 1).toNat < (UInt256.ofNat 2).toNat)
              let join_3 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- if L > 2 { result[2] = out.l2 }
                let v211 := decide ((UInt256.ofNat 2).toNat < (UInt256.ofNat 2).toNat)
                let join_4 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- if L > 3 { result[3] = out.l3 }
                  let v222 := decide ((UInt256.ofNat 3).toNat < (UInt256.ofNat 2).toNat)
                  let join_5 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                    let mut inverse := inverse
                    let mut small := small
                    let mut table := table
                    let mut squared := squared
                    let mut power := power
                    let mut index := index
                    let mut started := started
                    let mut result := result
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
                    -- Option::Some(result)
                    let result_231 := result
                    return Option_MemArray_u64.Some result_231
                  if v222 then
                    -- result[3] = out.l3
                    let (v226, v226_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 3)
                    mstoreInt (.u 64) v226 out.l3
                    let v226_0 ← v226_k v226
                    result := v226_0
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  -- result[2] = out.l2
                  let (v215, v215_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 2)
                  mstoreInt (.u 64) v215 out.l2
                  let v215_0 ← v215_k v215
                  result := v215_0
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                -- result[1] = out.l1
                let (v204, v204_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 1)
                mstoreInt (.u 64) v204 out.l1
                let v204_0 ← v204_k v204
                result := v204_0
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return Option_MemArray_u64.None
        partial_fixpoint
        loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      -- if entries > 1 {
      if decide ((UInt256.ofNat 1).toNat < entries.toNat) then
        -- let squared = small.mul(table[0], table[0])
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        r11 := table[v71.toNat]!
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        r12 := table[v73.toNat]!
        let squared_78 ← Small_mul_2 small r11 r12
        squared := squared_78
        -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
        r13 := (Range_Known_1_Unknown.mk () entries)
        let v84 ← Range_start_1 r13
        let rec loop_4 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            -- table[k] = small.mul(table[k - 1], squared)
            let v92 := k - (UInt256.ofNat 1)
            checkIndex v92 64
            r14 := table[v92.toNat]!
            let v97 ← Small_mul_2 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
            let v100 ← Range_next_1 r13 v4
            loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v100
          else
            join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v84
      else
        join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v16

def Small_double_3_k2 (self : Small_3) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (v45 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Wide := do
  -- let (d3, borrow) = sbb(s3, self.m.l3, borrow)
  let v51 ← sbb s3 self.m.l3 v45.snd
  let d3 := v51.fst
  -- let (_, borrow) = sbb(x.l3 >> 63, 0, borrow)
  let v60 ← sbb (x.l3 >>> (UInt256.ofNat 63)) (UInt256.ofNat 0) v51.snd
  let join : Wide → FeM Wide := fun v2 => do
    return v2
  -- if borrow == 1 { Wide { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v60.snd = (UInt256.ofNat 1)) then
    join (Wide.mk s0 s1 s2 s3)
  else
    join (Wide.mk d0 d1 d2 d3)

def Small_double_3_k1 (self : Small_3) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) : FeM Wide := do
  -- let s3 = (x.l3 << 1) | (x.l2 >> 63)
  let s3 := (IntTy.shl (.u 64) x.l3 (UInt256.ofNat 1)) ||| (x.l2 >>> (UInt256.ofNat 63))
  -- let (d0, borrow) = sbb(s0, self.m.l0, 0)
  let v33 ← sbb s0 self.m.l0 (UInt256.ofNat 0)
  let d0 := v33.fst
  -- let (d1, borrow) = sbb(s1, self.m.l1, borrow)
  let v39 ← sbb s1 self.m.l1 v33.snd
  let d1 := v39.fst
  -- let (d2, borrow) = sbb(s2, self.m.l2, borrow)
  let v45 ← sbb s2 self.m.l2 v39.snd
  let d2 := v45.fst
  Small_double_3_k2 self x s0 s1 s2 s3 d0 d1 v45 d2

/-- `Small<3>::double` -/
def Small_double_3 (self : Small_3) (x : Wide) : FeM Wide := do
  -- let s0 = x.l0 << 1
  let s0 := IntTy.shl (.u 64) x.l0 (UInt256.ofNat 1)
  -- let s1 = (x.l1 << 1) | (x.l0 >> 63)
  let s1 := (IntTy.shl (.u 64) x.l1 (UInt256.ofNat 1)) ||| (x.l0 >>> (UInt256.ofNat 63))
  -- let s2 = (x.l2 << 1) | (x.l1 >> 63)
  let s2 := (IntTy.shl (.u 64) x.l2 (UInt256.ofNat 1)) ||| (x.l1 >>> (UInt256.ofNat 63))
  Small_double_3_k1 self x s0 s1 s2

def Small_row2_3_k1 (self : Small_3) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (m : UInt256) : FeM (Wide × UInt256) := do
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v31 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v35 ← mac m self.m.l1 v15.fst v31.snd
  -- let (r1, c) = adc(t2, 0, c)
  let v41 ← adc v21.fst (UInt256.ofNat 0) v35.snd
  -- (Wide { l0: r0, l1: r1, l2: 0, l3: 0 }, t3 + c)
  let v48 := (Wide.mk v35.fst v41.fst (UInt256.ofNat 0) (UInt256.ofNat 0))
  return (v48, (IntTy.addW (.u 64) v21.snd v41.snd))

/-- `Small<3>::row2` -/
def Small_row2_3 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, t3) = adc(top, c, 0)
  let v21 ← adc top v15.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  Small_row2_3_k1 self t0 v15 v21 m

def Small_row3_3_k2 (v28 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r2, c) = adc(t3, 0, c)
  let v54 ← adc v28.fst (UInt256.ofNat 0) v48.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: 0 }, t4 + c)
  let v60 := (Wide.mk v42.fst v48.fst v54.fst (UInt256.ofNat 0))
  return (v60, (IntTy.addW (.u 64) v28.snd v54.snd))

def Small_row3_3_k1 (self : Small_3) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v38 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v42 ← mac m self.m.l1 v15.fst v38.snd
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v48 ← mac m self.m.l2 v22.fst v42.snd
  Small_row3_3_k2 v28 v42 v48

/-- `Small<3>::row3` -/
def Small_row3_3 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  -- let (t3, t4) = adc(top, c, 0)
  let v28 ← adc top v22.snd (UInt256.ofNat 0)
  Small_row3_3_k1 self t0 v15 v22 v28

def Small_row4_3_k2 (self : Small_3) (v22 : (UInt256 × UInt256)) (v29 : (UInt256 × UInt256)) (v35 : (UInt256 × UInt256)) (m : UInt256) (v49 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v55 ← mac m self.m.l2 v22.fst v49.snd
  -- let (r2, c) = mac(m, self.m.l3, t3, c)
  let v61 ← mac m self.m.l3 v29.fst v55.snd
  -- let (r3, c) = adc(t4, 0, c)
  let v67 ← adc v35.fst (UInt256.ofNat 0) v61.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: r3 }, t5 + c)
  let v74 := ((Wide.mk v49.fst v55.fst v61.fst v67.fst), (IntTy.addW (.u 64) v35.snd v67.snd))
  return v74

def Small_row4_3_k1 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (t3, c) = mac(a.l3, b, t.l3, c)
  let v29 ← mac a.l3 b t.l3 v22.snd
  -- let (t4, t5) = adc(top, c, 0)
  let v35 ← adc top v29.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v45 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v49 ← mac m self.m.l1 v15.fst v45.snd
  Small_row4_3_k2 self v22 v29 v35 m v49

/-- `Small<3>::row4` -/
def Small_row4_3 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  Small_row4_3_k1 self a b t top t0 v15 v22

/-- `Small<3>::mul` -/
def Small_mul_3 (self : Small_3) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  -- let zero = Wide { l0: 0, l1: 0, l2: 0, l3: 0 }
  let zero_12 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  zero := zero_12
  -- if L == 1 {
  if decide ((UInt256.ofNat 3) = (UInt256.ofNat 1)) then
    -- let t = a.l0 as u128 * b.l0 as u128
    let t_21 := IntTy.mulW (.u 128) a.l0 b.l0
    -- let low: u64 = t.downcast_truncate()
    let low ← downcast_truncate_u128_u64 t_21
    -- let product = q as u128 * self.m.l0 as u128
    let product := IntTy.mulW (.u 128) (IntTy.mulW (.u 64) low self.inverse) self.m.l0
    -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
    let v37 := IntTy.addW (.u 128) (t_21 >>> (UInt256.ofNat 64)) (product >>> (UInt256.ofNat 64))
    let join : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
      let sum := IntTy.addW (.u 128) v37 v3
      let join_1 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 reduced => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        -- return Wide { l0: reduced.downcast_truncate(), l1: 0, l2: 0, l3: 0 }
        let v55 ← downcast_truncate_u128_u64 reduced
        let v59 := (Wide.mk v55 (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
        return v59
      -- let reduced = if sum >= self.m.l0 as u128 { sum - self.m.l0 as u128 } else { sum }
      if decide (self.m.l0.toNat ≤ sum.toNat) then
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (IntTy.subW (.u 128) sum self.m.l0)
      else
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum
    if decide (low ≠ (UInt256.ofNat 0)) then
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 1)
    else
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 0)
  else
    -- if L == 2 {
    if decide ((UInt256.ofNat 3) = (UInt256.ofNat 2)) then
      -- let (t, top) = self.row2(a, b.l0, zero, 0)
      let v68 ← Small_row2_3 self a b.l0 zero (UInt256.ofNat 0)
      t_r1 := v68.fst
      -- let (t, top) = self.row2(a, b.l1, t, top)
      let v75 ← Small_row2_3 self a b.l1 t_r1 v68.snd
      let t_76 := v75.fst
      -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
      let v83 ← sbb t_76.l0 self.m.l0 (UInt256.ofNat 0)
      let d0_84 := v83.fst
      -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
      let v90 ← sbb t_76.l1 self.m.l1 v83.snd
      let d1_91 := v90.fst
      -- let (_, borrow) = sbb(top, 0, borrow)
      let v96 ← sbb v75.snd (UInt256.ofNat 0) v90.snd
      let join_2 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: 0, l3: 0 } }
      if decide (v96.snd = (UInt256.ofNat 1)) then
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_76
      else
        -- Wide { l0: d0, l1: d1, l2: 0, l3: 0 }
        let v104 := (Wide.mk d0_84 d1_91 (UInt256.ofNat 0) (UInt256.ofNat 0))
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v104
    else
      -- if L == 3 {
      if decide ((UInt256.ofNat 3) = (UInt256.ofNat 3)) then
        -- let (t, top) = self.row3(a, b.l0, zero, 0)
        let v114 ← Small_row3_3 self a b.l0 zero (UInt256.ofNat 0)
        t_r2 := v114.fst
        -- let (t, top) = self.row3(a, b.l1, t, top)
        let v121 ← Small_row3_3 self a b.l1 t_r2 v114.snd
        t_r3 := v121.fst
        -- let (t, top) = self.row3(a, b.l2, t, top)
        let v128 ← Small_row3_3 self a b.l2 t_r3 v121.snd
        let t_129 := v128.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v136 ← sbb t_129.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_137 := v136.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v143 ← sbb t_129.l1 self.m.l1 v136.snd
        let d1_144 := v143.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v150 ← sbb t_129.l2 self.m.l2 v143.snd
        let d2_151 := v150.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v156 ← sbb v128.snd (UInt256.ofNat 0) v150.snd
        let join_3 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: 0 } }
        if decide (v156.snd = (UInt256.ofNat 1)) then
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_129
        else
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_137 d1_144 d2_151 (UInt256.ofNat 0))
      else
        -- let (t, top) = self.row4(a, b.l0, zero, 0)
        let v170 ← Small_row4_3 self a b.l0 zero (UInt256.ofNat 0)
        t_r4 := v170.fst
        -- let (t, top) = self.row4(a, b.l1, t, top)
        let v177 ← Small_row4_3 self a b.l1 t_r4 v170.snd
        t_r5 := v177.fst
        -- let (t, top) = self.row4(a, b.l2, t, top)
        let v184 ← Small_row4_3 self a b.l2 t_r5 v177.snd
        t_r6 := v184.fst
        -- let (t, top) = self.row4(a, b.l3, t, top)
        let v191 ← Small_row4_3 self a b.l3 t_r6 v184.snd
        let t_192 := v191.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v199 ← sbb t_192.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_200 := v199.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v206 ← sbb t_192.l1 self.m.l1 v199.snd
        let d1_207 := v206.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v213 ← sbb t_192.l2 self.m.l2 v206.snd
        let d2_214 := v213.fst
        -- let (d3, borrow) = sbb(t.l3, self.m.l3, borrow)
        let v220 ← sbb t_192.l3 self.m.l3 v213.snd
        let d3 := v220.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v226 ← sbb v191.snd (UInt256.ofNat 0) v220.snd
        let join_4 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        -- if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
        if decide (v226.snd = (UInt256.ofNat 1)) then
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_192
        else
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_200 d1_207 d2_214 d3)

/-- `Small<3>::r2` -/
def Small_r2_3 (self : Small_3) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let join : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 top => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    -- let shift: u64 = (255 - leading_zeros(top as u256)).downcast_truncate()
    let v33 := (UInt256.ofNat 255) - (IntTy.leadingZeros (.u 256) top)
    let shift ← downcast_truncate_u256_u64 v33
    -- let bit: u64 = 1 << shift
    let bit := IntTy.shl (.u 64) (UInt256.ofNat 1) shift
    let join_1 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let join_2 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let join_3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let join_4 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            -- var x = Wide {
            x := (Wide.mk v4 v5 v6 v7)
            -- let doublings: usize = (64 - shift).downcast_truncate()
            let doublings ← downcast_truncate_u64_usize (IntTy.subW (.u 64) (UInt256.ofNat 64) shift)
            -- for _ in 0..doublings + L { x = self.double(x) }
            let v73 := (Range_Known_0_Unknown.mk () (doublings + (UInt256.ofNat 3)))
            r1 := v73
            let v75 ← Range_start_0 r1
            let rec loop (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              -- for _ in 0..doublings + L { x = self.double(x) }
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let _ ← Range_at_0 r1 v8
                -- x = self.double(x)
                r2 := x
                let x_84 ← Small_double_3 self r2
                x := x_84
                -- for _ in 0..doublings + L { x = self.double(x) }
                let v88 ← Range_next_0 r1 v8
                loop x r1 r2 r3 r4 r5 v88
              else
                -- for _ in 0..6 { x = self.mul(x, x) }
                r3 := (Range_Known_0_Known_6.mk () ())
                let v95 ← Range_start_0_6 r3
                let rec loop_1 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  -- for _ in 0..6 { x = self.mul(x, x) }
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let _ ← Range_at_0_6 r3 v9
                    -- x = self.mul(x, x)
                    r4 := x
                    r5 := x
                    let x_107 ← Small_mul_3 self r4 r5
                    x := x_107
                    -- for _ in 0..6 { x = self.mul(x, x) }
                    let v111 ← Range_next_0_6 r3 v9
                    loop_1 x r1 r2 r3 r4 r5 v111
                  else
                    return x
                partial_fixpoint
                loop_1 x r1 r2 r3 r4 r5 v95
            partial_fixpoint
            loop x r1 r2 r3 r4 r5 v75
          -- var x = Wide {
          if decide ((UInt256.ofNat 3) = (UInt256.ofNat 4)) then
            join_4 x r1 r2 r3 r4 r5 bit
          else
            join_4 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
        -- var x = Wide {
        if decide ((UInt256.ofNat 3) = (UInt256.ofNat 3)) then
          join_3 x r1 r2 r3 r4 r5 bit
        else
          join_3 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
      -- var x = Wide {
      if decide ((UInt256.ofNat 3) = (UInt256.ofNat 2)) then
        join_2 x r1 r2 r3 r4 r5 bit
      else
        join_2 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
    -- var x = Wide {
    if decide ((UInt256.ofNat 3) = (UInt256.ofNat 1)) then
      join_1 x r1 r2 r3 r4 r5 bit
    else
      join_1 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
  -- let top = if L == 1 { self.m.l0 } else if L == 2 { self.m.l1 } else if L == 3 { self.m.l2 } else { self.m.l3 }
  if decide ((UInt256.ofNat 3) = (UInt256.ofNat 1)) then
    join x r1 r2 r3 r4 r5 self.m.l0
  else
    let join_5 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      join x r1 r2 r3 r4 r5 v2
    if decide ((UInt256.ofNat 3) = (UInt256.ofNat 2)) then
      join_5 x r1 r2 r3 r4 r5 self.m.l1
    else
      let join_6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        join_5 x r1 r2 r3 r4 r5 v3
      if decide ((UInt256.ofNat 3) = (UInt256.ofNat 3)) then
        join_6 x r1 r2 r3 r4 r5 self.m.l2
      else
        join_6 x r1 r2 r3 r4 r5 self.m.l3

/-- `lib::modexp::small_power<3>` -/
def small_power_3 (exponent : Exponent) (base : MemArray_u64) (value : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut inverse : UInt256 := default
  let mut small : Small_3 := default
  let mut table : (Vector Wide 64) := default
  let mut squared : Wide := default
  let mut power : Wide := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut result : MemArray_u64 := default
  let mut r8 : Range_Known_0_Known_5 := default
  let mut r9 : Wide := default
  let mut r10 : Wide := default
  let mut r11 : Wide := default
  let mut r12 : Wide := default
  let mut r13 : Range_Known_1_Unknown := default
  let mut r14 : Wide := default
  let mut r15 : Range_Unknown_Unknown := default
  let mut r16 : Wide := default
  let mut r17 : Wide := default
  let mut r18 : Wide := default
  let mut r19 : Wide := default
  let mut r20 : Wide := default
  let mut r21 : Wide := default
  -- let low = value[0]
  let v8 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
  let low ← mloadInt (.u 64) v8
  -- var inverse = low
  inverse := low
  -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
  r8 := (Range_Known_0_Known_5.mk () ())
  let v16 ← Range_start_0_5 r8
  let rec loop (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
    let mut inverse := inverse
    let mut small := small
    let mut table := table
    let mut squared := squared
    let mut power := power
    let mut index := index
    let mut started := started
    let mut result := result
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
    -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let _ ← Range_at_0_5 r8 v3
      -- inverse = inverse * (2 - low * inverse)
      let v27 := IntTy.subW (.u 64) (UInt256.ofNat 2) (IntTy.mulW (.u 64) low inverse)
      inverse := (IntTy.mulW (.u 64) inverse v27)
      -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
      let v32 ← Range_next_0_5 r8 v3
      loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v32
    else
      -- let small = Small<L> { m: Wide::of(value, L), inverse: 0 - inverse }
      let v35 ← Wide_of value (UInt256.ofNat 3)
      let small_39 := (Small_3.mk v35 (IntTy.subW (.u 64) (UInt256.ofNat 0) inverse))
      small := small_39
      -- let width = window_width(exponent.total - exponent.top)
      let width ← window_width (exponent.total - exponent.top)
      -- let entries: usize = (1 << (width - 1)).downcast_truncate()
      let entries ← downcast_truncate_usize_usize ((UInt256.ofNat 1) <<< (width - (UInt256.ofNat 1)))
      -- var table: [Wide; 64] = [Wide { l0: 0, l1: 0, l2: 0, l3: 0 }; 64]
      let v56 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
      table := (Vector.replicate 64 v56)
      -- table[0] = small.mul(Wide::of(base, L), small.r2())
      let v60 ← Wide_of base (UInt256.ofNat 3)
      r9 := v60
      let v62 ← Small_r2_3 small
      r10 := v62
      let v66 ← Small_mul_3 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let join : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
        let mut inverse := inverse
        let mut small := small
        let mut table := table
        let mut squared := squared
        let mut power := power
        let mut index := index
        let mut started := started
        let mut result := result
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
        -- var power = table[0]
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        power := table[v106.toNat]!
        -- var index = exponent.top
        index := exponent.top
        -- var started = false
        started := false
        let rec loop_1 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- while index < exponent.total {
          if decide (index.toNat < exponent.total.toNat) then
            -- let window = exponent.window(index, width)
            let window ← Exponent_window exponent index width
            let rec loop_2 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
              let mut inverse := inverse
              let mut small := small
              let mut table := table
              let mut squared := squared
              let mut power := power
              let mut index := index
              let mut started := started
              let mut result := result
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
              let join_1 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- index = window.end
                index := window.«end»
                loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              -- if window.odd {
              if window.odd then
                let join_2 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 power_6 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                  power := power_6
                  -- started = true
                  started := true
                  join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                if started then
                  -- small.mul(power, table[window.entry])
                  r18 := power
                  let v155 := window.entry
                  checkIndex v155 64
                  r19 := table[v155.toNat]!
                  let v160 ← Small_mul_3 small r18 r19
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v160
                else
                  -- table[window.entry]
                  let v162 := window.entry
                  checkIndex v162 64
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 table[v162.toNat]!
              else
                join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            -- if started {
            if started then
              -- for _ in index..window.end { power = small.mul(power, power) }
              r15 := (Range_Unknown_Unknown.mk index window.«end»)
              let v128 ← Range_Unknown_Unknown_start r15
              let rec loop_3 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- for _ in index..window.end { power = small.mul(power, power) }
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let _ ← Range_Unknown_Unknown_at r15 v5
                  -- power = small.mul(power, power)
                  r16 := power
                  r17 := power
                  let power_141 ← Small_mul_3 small r16 r17
                  power := power_141
                  -- for _ in index..window.end { power = small.mul(power, power) }
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v145
                else
                  loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v128
            else
              loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            -- let out = small.mul(power, Wide { l0: 1, l1: 0, l2: 0, l3: 0 })
            r20 := power
            let v182 := (Wide.mk (UInt256.ofNat 1) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
            r21 := v182
            let out ← Small_mul_3 small r20 r21
            -- var result = MemArray<u64>::try_new_zeroed(L)?
            let v189 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 3)
            if match v189 with | .Some .. => true | _ => false then
              let result_191 := v189.Some_0
              result := result_191
              -- result[0] = out.l0
              let (v196, v196_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 0)
              mstoreInt (.u 64) v196 out.l0
              let v196_0 ← v196_k v196
              result := v196_0
              -- if L > 1 { result[1] = out.l1 }
              let v200 := decide ((UInt256.ofNat 1).toNat < (UInt256.ofNat 3).toNat)
              let join_3 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- if L > 2 { result[2] = out.l2 }
                let v211 := decide ((UInt256.ofNat 2).toNat < (UInt256.ofNat 3).toNat)
                let join_4 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- if L > 3 { result[3] = out.l3 }
                  let v222 := decide ((UInt256.ofNat 3).toNat < (UInt256.ofNat 3).toNat)
                  let join_5 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                    let mut inverse := inverse
                    let mut small := small
                    let mut table := table
                    let mut squared := squared
                    let mut power := power
                    let mut index := index
                    let mut started := started
                    let mut result := result
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
                    -- Option::Some(result)
                    let result_231 := result
                    return Option_MemArray_u64.Some result_231
                  if v222 then
                    -- result[3] = out.l3
                    let (v226, v226_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 3)
                    mstoreInt (.u 64) v226 out.l3
                    let v226_0 ← v226_k v226
                    result := v226_0
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  -- result[2] = out.l2
                  let (v215, v215_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 2)
                  mstoreInt (.u 64) v215 out.l2
                  let v215_0 ← v215_k v215
                  result := v215_0
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                -- result[1] = out.l1
                let (v204, v204_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 1)
                mstoreInt (.u 64) v204 out.l1
                let v204_0 ← v204_k v204
                result := v204_0
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return Option_MemArray_u64.None
        partial_fixpoint
        loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      -- if entries > 1 {
      if decide ((UInt256.ofNat 1).toNat < entries.toNat) then
        -- let squared = small.mul(table[0], table[0])
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        r11 := table[v71.toNat]!
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        r12 := table[v73.toNat]!
        let squared_78 ← Small_mul_3 small r11 r12
        squared := squared_78
        -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
        r13 := (Range_Known_1_Unknown.mk () entries)
        let v84 ← Range_start_1 r13
        let rec loop_4 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            -- table[k] = small.mul(table[k - 1], squared)
            let v92 := k - (UInt256.ofNat 1)
            checkIndex v92 64
            r14 := table[v92.toNat]!
            let v97 ← Small_mul_3 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
            let v100 ← Range_next_1 r13 v4
            loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v100
          else
            join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v84
      else
        join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v16

def Small_double_4_k2 (self : Small_4) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (v45 : (UInt256 × UInt256)) (d2 : UInt256) : FeM Wide := do
  -- let (d3, borrow) = sbb(s3, self.m.l3, borrow)
  let v51 ← sbb s3 self.m.l3 v45.snd
  let d3 := v51.fst
  -- let (_, borrow) = sbb(x.l3 >> 63, 0, borrow)
  let v60 ← sbb (x.l3 >>> (UInt256.ofNat 63)) (UInt256.ofNat 0) v51.snd
  let join : Wide → FeM Wide := fun v2 => do
    return v2
  -- if borrow == 1 { Wide { l0: s0, l1: s1, l2: s2, l3: s3 } } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
  if decide (v60.snd = (UInt256.ofNat 1)) then
    join (Wide.mk s0 s1 s2 s3)
  else
    join (Wide.mk d0 d1 d2 d3)

def Small_double_4_k1 (self : Small_4) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) : FeM Wide := do
  -- let s3 = (x.l3 << 1) | (x.l2 >> 63)
  let s3 := (IntTy.shl (.u 64) x.l3 (UInt256.ofNat 1)) ||| (x.l2 >>> (UInt256.ofNat 63))
  -- let (d0, borrow) = sbb(s0, self.m.l0, 0)
  let v33 ← sbb s0 self.m.l0 (UInt256.ofNat 0)
  let d0 := v33.fst
  -- let (d1, borrow) = sbb(s1, self.m.l1, borrow)
  let v39 ← sbb s1 self.m.l1 v33.snd
  let d1 := v39.fst
  -- let (d2, borrow) = sbb(s2, self.m.l2, borrow)
  let v45 ← sbb s2 self.m.l2 v39.snd
  let d2 := v45.fst
  Small_double_4_k2 self x s0 s1 s2 s3 d0 d1 v45 d2

/-- `Small<4>::double` -/
def Small_double_4 (self : Small_4) (x : Wide) : FeM Wide := do
  -- let s0 = x.l0 << 1
  let s0 := IntTy.shl (.u 64) x.l0 (UInt256.ofNat 1)
  -- let s1 = (x.l1 << 1) | (x.l0 >> 63)
  let s1 := (IntTy.shl (.u 64) x.l1 (UInt256.ofNat 1)) ||| (x.l0 >>> (UInt256.ofNat 63))
  -- let s2 = (x.l2 << 1) | (x.l1 >> 63)
  let s2 := (IntTy.shl (.u 64) x.l2 (UInt256.ofNat 1)) ||| (x.l1 >>> (UInt256.ofNat 63))
  Small_double_4_k1 self x s0 s1 s2

def Small_row2_4_k1 (self : Small_4) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v21 : (UInt256 × UInt256)) (m : UInt256) : FeM (Wide × UInt256) := do
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v31 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v35 ← mac m self.m.l1 v15.fst v31.snd
  -- let (r1, c) = adc(t2, 0, c)
  let v41 ← adc v21.fst (UInt256.ofNat 0) v35.snd
  -- (Wide { l0: r0, l1: r1, l2: 0, l3: 0 }, t3 + c)
  let v48 := (Wide.mk v35.fst v41.fst (UInt256.ofNat 0) (UInt256.ofNat 0))
  return (v48, (IntTy.addW (.u 64) v21.snd v41.snd))

/-- `Small<4>::row2` -/
def Small_row2_4 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, t3) = adc(top, c, 0)
  let v21 ← adc top v15.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  Small_row2_4_k1 self t0 v15 v21 m

def Small_row3_4_k2 (v28 : (UInt256 × UInt256)) (v42 : (UInt256 × UInt256)) (v48 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r2, c) = adc(t3, 0, c)
  let v54 ← adc v28.fst (UInt256.ofNat 0) v48.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: 0 }, t4 + c)
  let v60 := (Wide.mk v42.fst v48.fst v54.fst (UInt256.ofNat 0))
  return (v60, (IntTy.addW (.u 64) v28.snd v54.snd))

def Small_row3_4_k1 (self : Small_4) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v38 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v42 ← mac m self.m.l1 v15.fst v38.snd
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v48 ← mac m self.m.l2 v22.fst v42.snd
  Small_row3_4_k2 v28 v42 v48

/-- `Small<4>::row3` -/
def Small_row3_4 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  -- let (t3, t4) = adc(top, c, 0)
  let v28 ← adc top v22.snd (UInt256.ofNat 0)
  Small_row3_4_k1 self t0 v15 v22 v28

def Small_row4_4_k2 (self : Small_4) (v22 : (UInt256 × UInt256)) (v29 : (UInt256 × UInt256)) (v35 : (UInt256 × UInt256)) (m : UInt256) (v49 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (r1, c) = mac(m, self.m.l2, t2, c)
  let v55 ← mac m self.m.l2 v22.fst v49.snd
  -- let (r2, c) = mac(m, self.m.l3, t3, c)
  let v61 ← mac m self.m.l3 v29.fst v55.snd
  -- let (r3, c) = adc(t4, 0, c)
  let v67 ← adc v35.fst (UInt256.ofNat 0) v61.snd
  -- (Wide { l0: r0, l1: r1, l2: r2, l3: r3 }, t5 + c)
  let v74 := ((Wide.mk v49.fst v55.fst v61.fst v67.fst), (IntTy.addW (.u 64) v35.snd v67.snd))
  return v74

def Small_row4_4_k1 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) (t0 : UInt256) (v15 : (UInt256 × UInt256)) (v22 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  -- let (t3, c) = mac(a.l3, b, t.l3, c)
  let v29 ← mac a.l3 b t.l3 v22.snd
  -- let (t4, t5) = adc(top, c, 0)
  let v35 ← adc top v29.snd (UInt256.ofNat 0)
  -- let m = t0 * self.inverse
  let m := IntTy.mulW (.u 64) t0 self.inverse
  -- let (_, c) = mac(m, self.m.l0, t0, 0)
  let v45 ← mac m self.m.l0 t0 (UInt256.ofNat 0)
  -- let (r0, c) = mac(m, self.m.l1, t1, c)
  let v49 ← mac m self.m.l1 v15.fst v45.snd
  Small_row4_4_k2 self v22 v29 v35 m v49

/-- `Small<4>::row4` -/
def Small_row4_4 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  -- let (t0, c) = mac(a.l0, b, t.l0, 0)
  let v8 ← mac a.l0 b t.l0 (UInt256.ofNat 0)
  let t0 := v8.fst
  -- let (t1, c) = mac(a.l1, b, t.l1, c)
  let v15 ← mac a.l1 b t.l1 v8.snd
  -- let (t2, c) = mac(a.l2, b, t.l2, c)
  let v22 ← mac a.l2 b t.l2 v15.snd
  Small_row4_4_k1 self a b t top t0 v15 v22

/-- `Small<4>::mul` -/
def Small_mul_4 (self : Small_4) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  -- let zero = Wide { l0: 0, l1: 0, l2: 0, l3: 0 }
  let zero_12 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
  zero := zero_12
  -- if L == 1 {
  if decide ((UInt256.ofNat 4) = (UInt256.ofNat 1)) then
    -- let t = a.l0 as u128 * b.l0 as u128
    let t_21 := IntTy.mulW (.u 128) a.l0 b.l0
    -- let low: u64 = t.downcast_truncate()
    let low ← downcast_truncate_u128_u64 t_21
    -- let product = q as u128 * self.m.l0 as u128
    let product := IntTy.mulW (.u 128) (IntTy.mulW (.u 64) low self.inverse) self.m.l0
    -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
    let v37 := IntTy.addW (.u 128) (t_21 >>> (UInt256.ofNat 64)) (product >>> (UInt256.ofNat 64))
    let join : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      -- let sum = (t >> 64) + (product >> 64) + if low != 0 { 1 } else { 0 }
      let sum := IntTy.addW (.u 128) v37 v3
      let join_1 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 reduced => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        -- return Wide { l0: reduced.downcast_truncate(), l1: 0, l2: 0, l3: 0 }
        let v55 ← downcast_truncate_u128_u64 reduced
        let v59 := (Wide.mk v55 (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
        return v59
      -- let reduced = if sum >= self.m.l0 as u128 { sum - self.m.l0 as u128 } else { sum }
      if decide (self.m.l0.toNat ≤ sum.toNat) then
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (IntTy.subW (.u 128) sum self.m.l0)
      else
        join_1 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum
    if decide (low ≠ (UInt256.ofNat 0)) then
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 1)
    else
      join zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (UInt256.ofNat 0)
  else
    -- if L == 2 {
    if decide ((UInt256.ofNat 4) = (UInt256.ofNat 2)) then
      -- let (t, top) = self.row2(a, b.l0, zero, 0)
      let v68 ← Small_row2_4 self a b.l0 zero (UInt256.ofNat 0)
      t_r1 := v68.fst
      -- let (t, top) = self.row2(a, b.l1, t, top)
      let v75 ← Small_row2_4 self a b.l1 t_r1 v68.snd
      let t_76 := v75.fst
      -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
      let v83 ← sbb t_76.l0 self.m.l0 (UInt256.ofNat 0)
      let d0_84 := v83.fst
      -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
      let v90 ← sbb t_76.l1 self.m.l1 v83.snd
      let d1_91 := v90.fst
      -- let (_, borrow) = sbb(top, 0, borrow)
      let v96 ← sbb v75.snd (UInt256.ofNat 0) v90.snd
      let join_2 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: 0, l3: 0 } }
      if decide (v96.snd = (UInt256.ofNat 1)) then
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_76
      else
        -- Wide { l0: d0, l1: d1, l2: 0, l3: 0 }
        let v104 := (Wide.mk d0_84 d1_91 (UInt256.ofNat 0) (UInt256.ofNat 0))
        join_2 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v104
    else
      -- if L == 3 {
      if decide ((UInt256.ofNat 4) = (UInt256.ofNat 3)) then
        -- let (t, top) = self.row3(a, b.l0, zero, 0)
        let v114 ← Small_row3_4 self a b.l0 zero (UInt256.ofNat 0)
        t_r2 := v114.fst
        -- let (t, top) = self.row3(a, b.l1, t, top)
        let v121 ← Small_row3_4 self a b.l1 t_r2 v114.snd
        t_r3 := v121.fst
        -- let (t, top) = self.row3(a, b.l2, t, top)
        let v128 ← Small_row3_4 self a b.l2 t_r3 v121.snd
        let t_129 := v128.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v136 ← sbb t_129.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_137 := v136.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v143 ← sbb t_129.l1 self.m.l1 v136.snd
        let d1_144 := v143.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v150 ← sbb t_129.l2 self.m.l2 v143.snd
        let d2_151 := v150.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v156 ← sbb v128.snd (UInt256.ofNat 0) v150.snd
        let join_3 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        -- return if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: 0 } }
        if decide (v156.snd = (UInt256.ofNat 1)) then
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_129
        else
          join_3 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_137 d1_144 d2_151 (UInt256.ofNat 0))
      else
        -- let (t, top) = self.row4(a, b.l0, zero, 0)
        let v170 ← Small_row4_4 self a b.l0 zero (UInt256.ofNat 0)
        t_r4 := v170.fst
        -- let (t, top) = self.row4(a, b.l1, t, top)
        let v177 ← Small_row4_4 self a b.l1 t_r4 v170.snd
        t_r5 := v177.fst
        -- let (t, top) = self.row4(a, b.l2, t, top)
        let v184 ← Small_row4_4 self a b.l2 t_r5 v177.snd
        t_r6 := v184.fst
        -- let (t, top) = self.row4(a, b.l3, t, top)
        let v191 ← Small_row4_4 self a b.l3 t_r6 v184.snd
        let t_192 := v191.fst
        -- let (d0, borrow) = sbb(t.l0, self.m.l0, 0)
        let v199 ← sbb t_192.l0 self.m.l0 (UInt256.ofNat 0)
        let d0_200 := v199.fst
        -- let (d1, borrow) = sbb(t.l1, self.m.l1, borrow)
        let v206 ← sbb t_192.l1 self.m.l1 v199.snd
        let d1_207 := v206.fst
        -- let (d2, borrow) = sbb(t.l2, self.m.l2, borrow)
        let v213 ← sbb t_192.l2 self.m.l2 v206.snd
        let d2_214 := v213.fst
        -- let (d3, borrow) = sbb(t.l3, self.m.l3, borrow)
        let v220 ← sbb t_192.l3 self.m.l3 v213.snd
        let d3 := v220.fst
        -- let (_, borrow) = sbb(top, 0, borrow)
        let v226 ← sbb v191.snd (UInt256.ofNat 0) v220.snd
        let join_4 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        -- if borrow == 1 { t } else { Wide { l0: d0, l1: d1, l2: d2, l3: d3 } }
        if decide (v226.snd = (UInt256.ofNat 1)) then
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_192
        else
          join_4 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 (Wide.mk d0_200 d1_207 d2_214 d3)

/-- `Small<4>::r2` -/
def Small_r2_4 (self : Small_4) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let join : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 top => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    -- let shift: u64 = (255 - leading_zeros(top as u256)).downcast_truncate()
    let v33 := (UInt256.ofNat 255) - (IntTy.leadingZeros (.u 256) top)
    let shift ← downcast_truncate_u256_u64 v33
    -- let bit: u64 = 1 << shift
    let bit := IntTy.shl (.u 64) (UInt256.ofNat 1) shift
    let join_1 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let join_2 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let join_3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let join_4 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            -- var x = Wide {
            x := (Wide.mk v4 v5 v6 v7)
            -- let doublings: usize = (64 - shift).downcast_truncate()
            let doublings ← downcast_truncate_u64_usize (IntTy.subW (.u 64) (UInt256.ofNat 64) shift)
            -- for _ in 0..doublings + L { x = self.double(x) }
            let v73 := (Range_Known_0_Unknown.mk () (doublings + (UInt256.ofNat 4)))
            r1 := v73
            let v75 ← Range_start_0 r1
            let rec loop (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              -- for _ in 0..doublings + L { x = self.double(x) }
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let _ ← Range_at_0 r1 v8
                -- x = self.double(x)
                r2 := x
                let x_84 ← Small_double_4 self r2
                x := x_84
                -- for _ in 0..doublings + L { x = self.double(x) }
                let v88 ← Range_next_0 r1 v8
                loop x r1 r2 r3 r4 r5 v88
              else
                -- for _ in 0..6 { x = self.mul(x, x) }
                r3 := (Range_Known_0_Known_6.mk () ())
                let v95 ← Range_start_0_6 r3
                let rec loop_1 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  -- for _ in 0..6 { x = self.mul(x, x) }
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let _ ← Range_at_0_6 r3 v9
                    -- x = self.mul(x, x)
                    r4 := x
                    r5 := x
                    let x_107 ← Small_mul_4 self r4 r5
                    x := x_107
                    -- for _ in 0..6 { x = self.mul(x, x) }
                    let v111 ← Range_next_0_6 r3 v9
                    loop_1 x r1 r2 r3 r4 r5 v111
                  else
                    return x
                partial_fixpoint
                loop_1 x r1 r2 r3 r4 r5 v95
            partial_fixpoint
            loop x r1 r2 r3 r4 r5 v75
          -- var x = Wide {
          if decide ((UInt256.ofNat 4) = (UInt256.ofNat 4)) then
            join_4 x r1 r2 r3 r4 r5 bit
          else
            join_4 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
        -- var x = Wide {
        if decide ((UInt256.ofNat 4) = (UInt256.ofNat 3)) then
          join_3 x r1 r2 r3 r4 r5 bit
        else
          join_3 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
      -- var x = Wide {
      if decide ((UInt256.ofNat 4) = (UInt256.ofNat 2)) then
        join_2 x r1 r2 r3 r4 r5 bit
      else
        join_2 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
    -- var x = Wide {
    if decide ((UInt256.ofNat 4) = (UInt256.ofNat 1)) then
      join_1 x r1 r2 r3 r4 r5 bit
    else
      join_1 x r1 r2 r3 r4 r5 (UInt256.ofNat 0)
  -- let top = if L == 1 { self.m.l0 } else if L == 2 { self.m.l1 } else if L == 3 { self.m.l2 } else { self.m.l3 }
  if decide ((UInt256.ofNat 4) = (UInt256.ofNat 1)) then
    join x r1 r2 r3 r4 r5 self.m.l0
  else
    let join_5 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      join x r1 r2 r3 r4 r5 v2
    if decide ((UInt256.ofNat 4) = (UInt256.ofNat 2)) then
      join_5 x r1 r2 r3 r4 r5 self.m.l1
    else
      let join_6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        join_5 x r1 r2 r3 r4 r5 v3
      if decide ((UInt256.ofNat 4) = (UInt256.ofNat 3)) then
        join_6 x r1 r2 r3 r4 r5 self.m.l2
      else
        join_6 x r1 r2 r3 r4 r5 self.m.l3

/-- `lib::modexp::small_power<4>` -/
def small_power_4 (exponent : Exponent) (base : MemArray_u64) (value : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut inverse : UInt256 := default
  let mut small : Small_4 := default
  let mut table : (Vector Wide 64) := default
  let mut squared : Wide := default
  let mut power : Wide := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut result : MemArray_u64 := default
  let mut r8 : Range_Known_0_Known_5 := default
  let mut r9 : Wide := default
  let mut r10 : Wide := default
  let mut r11 : Wide := default
  let mut r12 : Wide := default
  let mut r13 : Range_Known_1_Unknown := default
  let mut r14 : Wide := default
  let mut r15 : Range_Unknown_Unknown := default
  let mut r16 : Wide := default
  let mut r17 : Wide := default
  let mut r18 : Wide := default
  let mut r19 : Wide := default
  let mut r20 : Wide := default
  let mut r21 : Wide := default
  -- let low = value[0]
  let v8 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
  let low ← mloadInt (.u 64) v8
  -- var inverse = low
  inverse := low
  -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
  r8 := (Range_Known_0_Known_5.mk () ())
  let v16 ← Range_start_0_5 r8
  let rec loop (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
    let mut inverse := inverse
    let mut small := small
    let mut table := table
    let mut squared := squared
    let mut power := power
    let mut index := index
    let mut started := started
    let mut result := result
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
    -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let _ ← Range_at_0_5 r8 v3
      -- inverse = inverse * (2 - low * inverse)
      let v27 := IntTy.subW (.u 64) (UInt256.ofNat 2) (IntTy.mulW (.u 64) low inverse)
      inverse := (IntTy.mulW (.u 64) inverse v27)
      -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
      let v32 ← Range_next_0_5 r8 v3
      loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v32
    else
      -- let small = Small<L> { m: Wide::of(value, L), inverse: 0 - inverse }
      let v35 ← Wide_of value (UInt256.ofNat 4)
      let small_39 := (Small_4.mk v35 (IntTy.subW (.u 64) (UInt256.ofNat 0) inverse))
      small := small_39
      -- let width = window_width(exponent.total - exponent.top)
      let width ← window_width (exponent.total - exponent.top)
      -- let entries: usize = (1 << (width - 1)).downcast_truncate()
      let entries ← downcast_truncate_usize_usize ((UInt256.ofNat 1) <<< (width - (UInt256.ofNat 1)))
      -- var table: [Wide; 64] = [Wide { l0: 0, l1: 0, l2: 0, l3: 0 }; 64]
      let v56 := (Wide.mk (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
      table := (Vector.replicate 64 v56)
      -- table[0] = small.mul(Wide::of(base, L), small.r2())
      let v60 ← Wide_of base (UInt256.ofNat 4)
      r9 := v60
      let v62 ← Small_r2_4 small
      r10 := v62
      let v66 ← Small_mul_4 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let join : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
        let mut inverse := inverse
        let mut small := small
        let mut table := table
        let mut squared := squared
        let mut power := power
        let mut index := index
        let mut started := started
        let mut result := result
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
        -- var power = table[0]
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        power := table[v106.toNat]!
        -- var index = exponent.top
        index := exponent.top
        -- var started = false
        started := false
        let rec loop_1 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- while index < exponent.total {
          if decide (index.toNat < exponent.total.toNat) then
            -- let window = exponent.window(index, width)
            let window ← Exponent_window exponent index width
            let rec loop_2 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
              let mut inverse := inverse
              let mut small := small
              let mut table := table
              let mut squared := squared
              let mut power := power
              let mut index := index
              let mut started := started
              let mut result := result
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
              let join_1 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- index = window.end
                index := window.«end»
                loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              -- if window.odd {
              if window.odd then
                let join_2 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 power_6 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                  power := power_6
                  -- started = true
                  started := true
                  join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                -- power = if started { small.mul(power, table[window.entry]) } else { table[window.entry] }
                if started then
                  -- small.mul(power, table[window.entry])
                  r18 := power
                  let v155 := window.entry
                  checkIndex v155 64
                  r19 := table[v155.toNat]!
                  let v160 ← Small_mul_4 small r18 r19
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v160
                else
                  -- table[window.entry]
                  let v162 := window.entry
                  checkIndex v162 64
                  join_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 table[v162.toNat]!
              else
                join_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            -- if started {
            if started then
              -- for _ in index..window.end { power = small.mul(power, power) }
              r15 := (Range_Unknown_Unknown.mk index window.«end»)
              let v128 ← Range_Unknown_Unknown_start r15
              let rec loop_3 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- for _ in index..window.end { power = small.mul(power, power) }
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let _ ← Range_Unknown_Unknown_at r15 v5
                  -- power = small.mul(power, power)
                  r16 := power
                  r17 := power
                  let power_141 ← Small_mul_4 small r16 r17
                  power := power_141
                  -- for _ in index..window.end { power = small.mul(power, power) }
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v145
                else
                  loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              loop_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v128
            else
              loop_2 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            -- let out = small.mul(power, Wide { l0: 1, l1: 0, l2: 0, l3: 0 })
            r20 := power
            let v182 := (Wide.mk (UInt256.ofNat 1) (UInt256.ofNat 0) (UInt256.ofNat 0) (UInt256.ofNat 0))
            r21 := v182
            let out ← Small_mul_4 small r20 r21
            -- var result = MemArray<u64>::try_new_zeroed(L)?
            let v189 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 4)
            if match v189 with | .Some .. => true | _ => false then
              let result_191 := v189.Some_0
              result := result_191
              -- result[0] = out.l0
              let (v196, v196_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 0)
              mstoreInt (.u 64) v196 out.l0
              let v196_0 ← v196_k v196
              result := v196_0
              -- if L > 1 { result[1] = out.l1 }
              let v200 := decide ((UInt256.ofNat 1).toNat < (UInt256.ofNat 4).toNat)
              let join_3 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                let mut inverse := inverse
                let mut small := small
                let mut table := table
                let mut squared := squared
                let mut power := power
                let mut index := index
                let mut started := started
                let mut result := result
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
                -- if L > 2 { result[2] = out.l2 }
                let v211 := decide ((UInt256.ofNat 2).toNat < (UInt256.ofNat 4).toNat)
                let join_4 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                  let mut inverse := inverse
                  let mut small := small
                  let mut table := table
                  let mut squared := squared
                  let mut power := power
                  let mut index := index
                  let mut started := started
                  let mut result := result
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
                  -- if L > 3 { result[3] = out.l3 }
                  let v222 := decide ((UInt256.ofNat 3).toNat < (UInt256.ofNat 4).toNat)
                  let join_5 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
                    let mut inverse := inverse
                    let mut small := small
                    let mut table := table
                    let mut squared := squared
                    let mut power := power
                    let mut index := index
                    let mut started := started
                    let mut result := result
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
                    -- Option::Some(result)
                    let result_231 := result
                    return Option_MemArray_u64.Some result_231
                  if v222 then
                    -- result[3] = out.l3
                    let (v226, v226_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 3)
                    mstoreInt (.u 64) v226 out.l3
                    let v226_0 ← v226_k v226
                    result := v226_0
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    join_5 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  -- result[2] = out.l2
                  let (v215, v215_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 2)
                  mstoreInt (.u 64) v215 out.l2
                  let v215_0 ← v215_k v215
                  result := v215_0
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  join_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                -- result[1] = out.l1
                let (v204, v204_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 1)
                mstoreInt (.u 64) v204 out.l1
                let v204_0 ← v204_k v204
                result := v204_0
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                join_3 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return Option_MemArray_u64.None
        partial_fixpoint
        loop_1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      -- if entries > 1 {
      if decide ((UInt256.ofNat 1).toNat < entries.toNat) then
        -- let squared = small.mul(table[0], table[0])
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        r11 := table[v71.toNat]!
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        r12 := table[v73.toNat]!
        let squared_78 ← Small_mul_4 small r11 r12
        squared := squared_78
        -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
        r13 := (Range_Known_1_Unknown.mk () entries)
        let v84 ← Range_start_1 r13
        let rec loop_4 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
          let mut inverse := inverse
          let mut small := small
          let mut table := table
          let mut squared := squared
          let mut power := power
          let mut index := index
          let mut started := started
          let mut result := result
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
          -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            -- table[k] = small.mul(table[k - 1], squared)
            let v92 := k - (UInt256.ofNat 1)
            checkIndex v92 64
            r14 := table[v92.toNat]!
            let v97 ← Small_mul_4 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            -- for k in 1..entries { table[k] = small.mul(table[k - 1], squared) }
            let v100 ← Range_next_1 r13 v4
            loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v100
          else
            join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        loop_4 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v84
      else
        join inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  loop inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v16

/-- `lib::modexp::small_power_of` -/
def small_power_of (exponent : Exponent) (base : MemArray_u64) (value : MemArray_u64) (n : UInt256) : FeM Option_MemArray_u64 := do
  let join : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v4 => do
    return v4
  -- if n == 1 { small_power<1>(exponent, base, value) } else if n == 2 { small_power<2>(exponent, base, value) }
  if decide (n = (UInt256.ofNat 1)) then
    -- small_power<1>(exponent, base, value)
    let v10 ← small_power_1 exponent base value
    join v10
  else
    let join_1 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v5 => do
      join v5
    if decide (n = (UInt256.ofNat 2)) then
      -- small_power<2>(exponent, base, value)
      let v15 ← small_power_2 exponent base value
      join_1 v15
    else
      let join_2 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v6 => do
        join_1 v6
      if decide (n = (UInt256.ofNat 3)) then
        -- small_power<3>(exponent, base, value)
        let v20 ← small_power_3 exponent base value
        join_2 v20
      else
        -- small_power<4>(exponent, base, value)
        let v22 ← small_power_4 exponent base value
        join_2 v22

/-- `Montgomery::new` -/
def Montgomery_new (value : MemArray_u64) (len : UInt256) : FeM Montgomery := do
  let mut value := value
  let mut inverse : UInt256 := default
  let mut r2 : Range_Known_0_Known_5 := default
  -- let low = value[0]
  let v4 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
  let low ← mloadInt (.u 64) v4
  -- var inverse = low
  inverse := low
  -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
  r2 := (Range_Known_0_Known_5.mk () ())
  let v12 ← Range_start_0_5 r2
  let rec loop (value : MemArray_u64) (inverse : UInt256) (r2 : Range_Known_0_Known_5) (v1 : UInt256) : FeM Montgomery := do
    let mut value := value
    let mut inverse := inverse
    let mut r2 := r2
    -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
    let v15 ← Range_has_0_5 r2 v1
    if v15 then
      let _ ← Range_at_0_5 r2 v1
      -- inverse = inverse * (2 - low * inverse)
      let v23 := IntTy.subW (.u 64) (UInt256.ofNat 2) (IntTy.mulW (.u 64) low inverse)
      inverse := (IntTy.mulW (.u 64) inverse v23)
      -- for _ in 0..5 { inverse = inverse * (2 - low * inverse) }
      let v28 ← Range_next_0_5 r2 v1
      loop value inverse r2 v28
    else
      -- Self { limbs: value, len, inverse: 0 - inverse }
      let value_30 := value
      let v36 := (Montgomery.mk value_30 len (IntTy.subW (.u 64) (UInt256.ofNat 0) inverse))
      return v36
  partial_fixpoint
  loop value inverse r2 v12

/-- `lib::bignum::rem_2by1` -/
def rem_2by1 (u1 : UInt256) (u0 : UInt256) (d : UInt256) (v : UInt256) : FeM UInt256 := do
  let mut r : UInt256 := default
  -- let q = v as u128 * u1 as u128 + ((u1 as u128) << 64 | u0 as u128)
  let q := IntTy.addW (.u 128) (IntTy.mulW (.u 128) v u1) ((IntTy.shl (.u 128) u1 (UInt256.ofNat 64)) ||| u0)
  -- let high: u64 = (q >> 64).downcast_truncate()
  let high ← downcast_truncate_u128_u64 (q >>> (UInt256.ofNat 64))
  -- let q0: u64 = q.downcast_truncate()
  let q0 ← downcast_truncate_u128_u64 q
  -- var r = u0 - q1 * d
  let v29 := IntTy.mulW (.u 64) (IntTy.addW (.u 64) high (UInt256.ofNat 1)) d
  r := (IntTy.subW (.u 64) u0 v29)
  let join : UInt256 → FeM UInt256 := fun r => do
    let mut r := r
    let join_1 : UInt256 → FeM UInt256 := fun r => do
      let mut r := r
      return r
    -- if r >= d { r -= d }
    if decide (d.toNat ≤ r.toNat) then
      -- r -= d
      r := (IntTy.subW (.u 64) r d)
      join_1 r
    else
      join_1 r
  -- if r > q0 { r += d }
  if decide (q0.toNat < r.toNat) then
    -- r += d
    r := (IntTy.addW (.u 64) r d)
    join r
  else
    join r

def div_3by2_k2 (q0 : UInt256) (d : UInt256) (v52 : UInt256) (q1 : UInt256) (r : UInt256) : FeM UInt256 := do
  let mut q1 := q1
  let mut r := r
  r := (IntTy.subW (.u 128) v52 d)
  -- q1 += 1
  q1 := (IntTy.addW (.u 64) q1 (UInt256.ofNat 1))
  -- let high: u64 = (r >> 64).downcast_truncate()
  let high ← downcast_truncate_u128_u64 (r >>> (UInt256.ofNat 64))
  let join : UInt256 → UInt256 → FeM UInt256 := fun q1 r => do
    let mut q1 := q1
    let mut r := r
    let join_1 : UInt256 → UInt256 → FeM UInt256 := fun q1 r => do
      let mut q1 := q1
      let mut r := r
      return q1
    -- if r >= d { q1 += 1 }
    if decide (d.toNat ≤ r.toNat) then
      -- q1 += 1
      q1 := (IntTy.addW (.u 64) q1 (UInt256.ofNat 1))
      join_1 q1 r
    else
      join_1 q1 r
  -- if high >= q0 {
  if decide (q0.toNat ≤ high.toNat) then
    -- q1 -= 1
    q1 := (IntTy.subW (.u 64) q1 (UInt256.ofNat 1))
    -- r += d
    r := (IntTy.addW (.u 128) r d)
    join q1 r
  else
    join q1 r

def div_3by2_k1 (u1 : UInt256) (u0 : UInt256) (d1 : UInt256) (d0 : UInt256) (q0 : UInt256) (q1 : UInt256) (r : UInt256) : FeM UInt256 := do
  let mut q1 := q1
  let mut r := r
  -- let d = (d1 as u128) << 64 | d0 as u128
  let d := (IntTy.shl (.u 128) d1 (UInt256.ofNat 64)) ||| d0
  -- var r = ((r1 as u128) << 64 | u0 as u128) - d0 as u128 * q1 as u128 - d
  let v43 := IntTy.shl (.u 128) (IntTy.subW (.u 64) u1 (IntTy.mulW (.u 64) q1 d1)) (UInt256.ofNat 64)
  let v52 := IntTy.subW (.u 128) (v43 ||| u0) (IntTy.mulW (.u 128) d0 q1)
  div_3by2_k2 q0 d v52 q1 r

/-- `lib::bignum::div_3by2` -/
def div_3by2 (u2 : UInt256) (u1 : UInt256) (u0 : UInt256) (d1 : UInt256) (d0 : UInt256) (v : UInt256) : FeM UInt256 := do
  let mut q1 : UInt256 := default
  let mut r : UInt256 := default
  -- let q = v as u128 * u2 as u128 + ((u2 as u128) << 64 | u1 as u128)
  let q := IntTy.addW (.u 128) (IntTy.mulW (.u 128) v u2) ((IntTy.shl (.u 128) u2 (UInt256.ofNat 64)) ||| u1)
  -- var q1: u64 = (q >> 64).downcast_truncate()
  let q1_22 ← downcast_truncate_u128_u64 (q >>> (UInt256.ofNat 64))
  q1 := q1_22
  -- let q0: u64 = q.downcast_truncate()
  let q0 ← downcast_truncate_u128_u64 q
  div_3by2_k1 u1 u0 d1 d0 q0 q1 r

/-- `Modulus::reduce` -/
def Modulus_reduce (self : Modulus) (work_1 : MemArray_u64) (len : UInt256) : FeM (Unit × MemArray_u64) := do
  let mut i_r0 : UInt256 := default
  let mut r : UInt256 := default
  let mut i_r2 : UInt256 := default
  let mut j : UInt256 := default
  let mut carry : UInt256 := default
  let mut borrow : UInt256 := default
  let mut add : UInt256 := default
  let mut r7 : Range_Unknown_Unknown := default
  let mut r8 : Range_Known_0_Unknown := default
  let mut r9 : Range_Known_0_Unknown := default
  let mut r10 : Range_Known_0_Unknown := default
  let mut r11 : Range_Unknown_Unknown := default
  let mut work_1 := work_1
  -- let count: usize = len.downcast_truncate()
  let count ← downcast_truncate_u64_usize len
  -- let n: usize = self.len.downcast_truncate()
  let n ← downcast_truncate_u64_usize self.len
  -- core::assert(count < work.len() && n < work.len() && n <= self.limbs.len())
  let v17 ← MemArray_len_u64 work_1
  let join : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v3 => do
    let mut i_r0 := i_r0
    let mut r := r
    let mut i_r2 := i_r2
    let mut j := j
    let mut carry := carry
    let mut borrow := borrow
    let mut add := add
    let mut r7 := r7
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut r11 := r11
    let mut work_1 := work_1
    -- core::assert(count < work.len() && n < work.len() && n <= self.limbs.len())
    let _ ← «assert» v3
    -- let wp = work.ptr()
    let wp ← MemArray_ptr_u64 work_1
    -- let mp = self.limbs.ptr()
    let mp ← MemArray_ptr_u64 self.limbs
    -- if count < n {
    if decide (count.toNat < n.toNat) then
      -- for i in count..n + 1 { unsafe { *limb(wp, i) = 0 } }
      let v42 := (Range_Unknown_Unknown.mk count (n + (UInt256.ofNat 1)))
      r7 := v42
      let v44 ← Range_Unknown_Unknown_start r7
      let rec loop (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v4 : UInt256) : FeM (Unit × MemArray_u64) := do
        let mut i_r0 := i_r0
        let mut r := r
        let mut i_r2 := i_r2
        let mut j := j
        let mut carry := carry
        let mut borrow := borrow
        let mut add := add
        let mut r7 := r7
        let mut r8 := r8
        let mut r9 := r9
        let mut r10 := r10
        let mut r11 := r11
        let mut work_1 := work_1
        -- for i in count..n + 1 { unsafe { *limb(wp, i) = 0 } }
        let v47 ← Range_Unknown_Unknown_has r7 v4
        if v47 then
          let v49 ← Range_Unknown_Unknown_at r7 v4
          -- *limb(wp, i) = 0
          let v52 ← limb wp v49
          mstoreInt (.u 64) v52 (UInt256.ofNat 0)
          -- for i in count..n + 1 { unsafe { *limb(wp, i) = 0 } }
          let v55 ← Range_Unknown_Unknown_next r7 v4
          loop i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v55
        else
          return ((), work_1)
      partial_fixpoint
      loop i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v44
    else
      -- let shift = self.shift
      let shift := self.shift
      -- *limb(wp, count) = 0
      let v62 ← limb wp count
      mstoreInt (.u 64) v62 (UInt256.ofNat 0)
      let join_1 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 => do
        let mut i_r0 := i_r0
        let mut r := r
        let mut i_r2 := i_r2
        let mut j := j
        let mut carry := carry
        let mut borrow := borrow
        let mut add := add
        let mut r7 := r7
        let mut r8 := r8
        let mut r9 := r9
        let mut r10 := r10
        let mut r11 := r11
        let mut work_1 := work_1
        -- if n == 1 {
        if decide (n = (UInt256.ofNat 1)) then
          -- *mp
          let d ← mloadInt (.u 64) mp
          -- var r: u64 = 0
          r := (UInt256.ofNat 0)
          -- var i = count + 1
          i_r2 := (count + (UInt256.ofNat 1))
          let rec loop_1 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
            let mut i_r0 := i_r0
            let mut r := r
            let mut i_r2 := i_r2
            let mut j := j
            let mut carry := carry
            let mut borrow := borrow
            let mut add := add
            let mut r7 := r7
            let mut r8 := r8
            let mut r9 := r9
            let mut r10 := r10
            let mut r11 := r11
            let mut work_1 := work_1
            -- while i > 0 {
            if decide ((UInt256.ofNat 0).toNat < i_r2.toNat) then
              -- i -= 1
              i_r2 := (i_r2 - (UInt256.ofNat 1))
              -- r = rem_2by1(r, unsafe { *limb(wp, i) }, d, self.reciprocal)
              let r_129 := r
              -- *limb(wp, i)
              let v133 ← limb wp i_r2
              let v134 ← mloadInt (.u 64) v133
              -- r = rem_2by1(r, unsafe { *limb(wp, i) }, d, self.reciprocal)
              let r_136 ← rem_2by1 r_129 v134 d self.reciprocal
              r := r_136
              -- *limb(wp, i) = 0
              let v142 ← limb wp i_r2
              mstoreInt (.u 64) v142 (UInt256.ofNat 0)
              loop_1 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
            else
              -- *wp = r >> shift
              mstoreInt (.u 64) wp (r >>> shift)
              return ((), work_1)
          partial_fixpoint
          loop_1 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
        else
          -- *limb(mp, n - 1)
          let v152 ← limb mp (n - (UInt256.ofNat 1))
          let d1 ← mloadInt (.u 64) v152
          -- *limb(mp, n - 2)
          let v157 ← limb mp (n - (UInt256.ofNat 2))
          let d0 ← mloadInt (.u 64) v157
          -- var j = count + 1 - n
          j := ((count + (UInt256.ofNat 1)) - n)
          let rec loop_2 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
            let mut i_r0 := i_r0
            let mut r := r
            let mut i_r2 := i_r2
            let mut j := j
            let mut carry := carry
            let mut borrow := borrow
            let mut add := add
            let mut r7 := r7
            let mut r8 := r8
            let mut r9 := r9
            let mut r10 := r10
            let mut r11 := r11
            let mut work_1 := work_1
            -- while j > 0 {
            if decide ((UInt256.ofNat 0).toNat < j.toNat) then
              -- j -= 1
              j := (j - (UInt256.ofNat 1))
              -- *limb(wp, j + n)
              let v176 ← limb wp (j + n)
              let high_177 ← mloadInt (.u 64) v176
              -- *limb(wp, j + n - 1)
              let v184 ← limb wp ((j + n) - (UInt256.ofNat 1))
              let next ← mloadInt (.u 64) v184
              let join_2 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → UInt256 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 qhat => do
                let mut i_r0 := i_r0
                let mut r := r
                let mut i_r2 := i_r2
                let mut j := j
                let mut carry := carry
                let mut borrow := borrow
                let mut add := add
                let mut r7 := r7
                let mut r8 := r8
                let mut r9 := r9
                let mut r10 := r10
                let mut r11 := r11
                let mut work_1 := work_1
                -- var carry: u64 = 0
                carry := (UInt256.ofNat 0)
                -- var borrow: u64 = 0
                borrow := (UInt256.ofNat 0)
                -- for i in 0..n {
                r8 := (Range_Known_0_Unknown.mk () n)
                let v210 ← Range_start_0 r8
                let rec loop_3 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
                  let mut i_r0 := i_r0
                  let mut r := r
                  let mut i_r2 := i_r2
                  let mut j := j
                  let mut carry := carry
                  let mut borrow := borrow
                  let mut add := add
                  let mut r7 := r7
                  let mut r8 := r8
                  let mut r9 := r9
                  let mut r10 := r10
                  let mut r11 := r11
                  let mut work_1 := work_1
                  -- for i in 0..n {
                  let v213 ← Range_has_0 r8 v6
                  if v213 then
                    let v215 ← Range_at_0 r8 v6
                    let i_216 := v215
                    -- *limb(mp, i)
                    let v218 ← limb mp i_216
                    let v219 ← mloadInt (.u 64) v218
                    -- let product = qhat as u128 * unsafe { *limb(mp, i) } as u128 + carry as u128
                    let product := IntTy.addW (.u 128) (IntTy.mulW (.u 128) qhat v219) carry
                    -- carry = (product >> 64).downcast_truncate()
                    let carry_228 ← downcast_truncate_u128_u64 (product >>> (UInt256.ofNat 64))
                    carry := carry_228
                    -- let low: u64 = product.downcast_truncate()
                    let low_231 ← downcast_truncate_u128_u64 product
                    -- *limb(wp, i + j)
                    let v235 ← limb wp (i_216 + j)
                    let v236 ← mloadInt (.u 64) v235
                    -- let difference = unsafe { *limb(wp, i + j) } as u128 - low as u128 - borrow as u128
                    let difference_242 := IntTy.subW (.u 128) (IntTy.subW (.u 128) v236 low_231) borrow
                    -- *limb(wp, i + j) = difference.downcast_truncate()
                    let v244 ← downcast_truncate_u128_u64 difference_242
                    let v247 ← limb wp (i_216 + j)
                    mstoreInt (.u 64) v247 v244
                    -- borrow = (difference >> 127).downcast_truncate()
                    let borrow_251 ← downcast_truncate_u128_u64 (difference_242 >>> (UInt256.ofNat 127))
                    borrow := borrow_251
                    -- for i in 0..n {
                    let v255 ← Range_next_0 r8 v6
                    loop_3 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v255
                  else
                    -- let difference = high as u128 - carry as u128 - borrow as u128
                    let difference_263 := IntTy.subW (.u 128) (IntTy.subW (.u 128) high_177 carry) borrow
                    -- *limb(wp, j + n) = difference.downcast_truncate()
                    let v265 ← downcast_truncate_u128_u64 difference_263
                    let v269 ← limb wp (j + n)
                    mstoreInt (.u 64) v269 v265
                    -- if difference >> 127 != 0 {
                    let v274 := decide ((difference_263 >>> (UInt256.ofNat 127)) ≠ (UInt256.ofNat 0))
                    let rec loop_4 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
                      let mut i_r0 := i_r0
                      let mut r := r
                      let mut i_r2 := i_r2
                      let mut j := j
                      let mut carry := carry
                      let mut borrow := borrow
                      let mut add := add
                      let mut r7 := r7
                      let mut r8 := r8
                      let mut r9 := r9
                      let mut r10 := r10
                      let mut r11 := r11
                      let mut work_1 := work_1
                      loop_2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
                    partial_fixpoint
                    if v274 then
                      -- var add: u64 = 0
                      add := (UInt256.ofNat 0)
                      -- for i in 0..n {
                      r9 := (Range_Known_0_Unknown.mk () n)
                      let v281 ← Range_start_0 r9
                      let rec loop_5 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
                        let mut i_r0 := i_r0
                        let mut r := r
                        let mut i_r2 := i_r2
                        let mut j := j
                        let mut carry := carry
                        let mut borrow := borrow
                        let mut add := add
                        let mut r7 := r7
                        let mut r8 := r8
                        let mut r9 := r9
                        let mut r10 := r10
                        let mut r11 := r11
                        let mut work_1 := work_1
                        -- for i in 0..n {
                        let v284 ← Range_has_0 r9 v7
                        if v284 then
                          let v286 ← Range_at_0 r9 v7
                          let i_287 := v286
                          -- *limb(wp, i + j)
                          let v290 ← limb wp (i_287 + j)
                          let v291 ← mloadInt (.u 64) v290
                          -- *limb(mp, i)
                          let v293 ← limb mp i_287
                          let v294 ← mloadInt (.u 64) v293
                          -- let sum = unsafe { *limb(wp, i + j) } as u128 + unsafe { *limb(mp, i) } as u128 + add as u128
                          let sum := IntTy.addW (.u 128) (IntTy.addW (.u 128) v291 v294) add
                          -- *limb(wp, i + j) = sum.downcast_truncate()
                          let v301 ← downcast_truncate_u128_u64 sum
                          let v304 ← limb wp (i_287 + j)
                          mstoreInt (.u 64) v304 v301
                          -- add = (sum >> 64).downcast_truncate()
                          let add_308 ← downcast_truncate_u128_u64 (sum >>> (UInt256.ofNat 64))
                          add := add_308
                          -- for i in 0..n {
                          let v312 ← Range_next_0 r9 v7
                          loop_5 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v312
                        else
                          -- *limb(wp, j + n) = *limb(wp, j + n) + add
                          let v317 ← limb wp (j + n)
                          let v318 ← mloadInt (.u 64) v317
                          let v320 := IntTy.addW (.u 64) v318 add
                          let v324 ← limb wp (j + n)
                          mstoreInt (.u 64) v324 v320
                          loop_4 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
                      partial_fixpoint
                      loop_5 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v281
                    else
                      loop_4 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
                partial_fixpoint
                loop_3 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v210
              let join_3 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 => do
                let mut i_r0 := i_r0
                let mut r := r
                let mut i_r2 := i_r2
                let mut j := j
                let mut carry := carry
                let mut borrow := borrow
                let mut add := add
                let mut r7 := r7
                let mut r8 := r8
                let mut r9 := r9
                let mut r10 := r10
                let mut r11 := r11
                let mut work_1 := work_1
                -- *limb(wp, j + n - 2)
                let v196 ← limb wp ((j + n) - (UInt256.ofNat 2))
                let v197 ← mloadInt (.u 64) v196
                -- div_3by2(high, next, unsafe { *limb(wp, j + n - 2) }, d1, d0, self.reciprocal)
                let v199 ← div_3by2 high_177 next v197 d1 d0 self.reciprocal
                join_2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v199
              -- let qhat: u64 = if high == d1 && next == d0 { 0xffffffffffffffff } else {
              if decide (high_177 = d1) then
                if decide (next = d0) then
                  join_2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 (UInt256.ofNat 18446744073709551615)
                else
                  join_3 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
              else
                join_3 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
            else
              -- for i in 0..n {
              r10 := (Range_Known_0_Unknown.mk () n)
              let v333 ← Range_start_0 r10
              let rec loop_6 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v8 : UInt256) : FeM (Unit × MemArray_u64) := do
                let mut i_r0 := i_r0
                let mut r := r
                let mut i_r2 := i_r2
                let mut j := j
                let mut carry := carry
                let mut borrow := borrow
                let mut add := add
                let mut r7 := r7
                let mut r8 := r8
                let mut r9 := r9
                let mut r10 := r10
                let mut r11 := r11
                let mut work_1 := work_1
                -- for i in 0..n {
                let v336 ← Range_has_0 r10 v8
                if v336 then
                  let v338 ← Range_at_0 r10 v8
                  let i_339 := v338
                  -- *limb(wp, i)
                  let v340 ← limb wp i_339
                  let v341 ← mloadInt (.u 64) v340
                  -- let low = unsafe { *limb(wp, i) } >> shift
                  let low_342 := v341 >>> shift
                  let join_4 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → UInt256 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 high_9 => do
                    let mut i_r0 := i_r0
                    let mut r := r
                    let mut i_r2 := i_r2
                    let mut j := j
                    let mut carry := carry
                    let mut borrow := borrow
                    let mut add := add
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut r9 := r9
                    let mut r10 := r10
                    let mut r11 := r11
                    let mut work_1 := work_1
                    -- *limb(wp, i) = low | high
                    let v358 ← limb wp i_339
                    mstoreInt (.u 64) v358 (low_342 ||| high_9)
                    -- for i in 0..n {
                    let v361 ← Range_next_0 r10 v8
                    loop_6 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v361
                  -- let high = if shift == 0 { 0 } else { unsafe { *limb(wp, i + 1) } << (64 - shift) }
                  if decide (shift = (UInt256.ofNat 0)) then
                    join_4 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 (UInt256.ofNat 0)
                  else
                    -- *limb(wp, i + 1)
                    let v350 ← limb wp (i_339 + (UInt256.ofNat 1))
                    let v351 ← mloadInt (.u 64) v350
                    -- unsafe { *limb(wp, i + 1) } << (64 - shift)
                    let v354 := IntTy.shl (.u 64) v351 (IntTy.subW (.u 64) (UInt256.ofNat 64) shift)
                    join_4 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v354
                else
                  -- for i in n..count + 1 { unsafe { *limb(wp, i) = 0 } }
                  let v366 := (Range_Unknown_Unknown.mk n (count + (UInt256.ofNat 1)))
                  r11 := v366
                  let v368 ← Range_Unknown_Unknown_start r11
                  let rec loop_7 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v10 : UInt256) : FeM (Unit × MemArray_u64) := do
                    let mut i_r0 := i_r0
                    let mut r := r
                    let mut i_r2 := i_r2
                    let mut j := j
                    let mut carry := carry
                    let mut borrow := borrow
                    let mut add := add
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut r9 := r9
                    let mut r10 := r10
                    let mut r11 := r11
                    let mut work_1 := work_1
                    -- for i in n..count + 1 { unsafe { *limb(wp, i) = 0 } }
                    let v371 ← Range_Unknown_Unknown_has r11 v10
                    if v371 then
                      let v373 ← Range_Unknown_Unknown_at r11 v10
                      -- *limb(wp, i) = 0
                      let v376 ← limb wp v373
                      mstoreInt (.u 64) v376 (UInt256.ofNat 0)
                      -- for i in n..count + 1 { unsafe { *limb(wp, i) = 0 } }
                      let v379 ← Range_Unknown_Unknown_next r11 v10
                      loop_7 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v379
                    else
                      return ((), work_1)
                  partial_fixpoint
                  loop_7 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v368
              partial_fixpoint
              loop_6 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v333
          partial_fixpoint
          loop_2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
      -- if shift != 0 {
      if decide (shift ≠ (UInt256.ofNat 0)) then
        -- var i = count
        i_r0 := count
        let rec loop_8 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
          let mut i_r0 := i_r0
          let mut r := r
          let mut i_r2 := i_r2
          let mut j := j
          let mut carry := carry
          let mut borrow := borrow
          let mut add := add
          let mut r7 := r7
          let mut r8 := r8
          let mut r9 := r9
          let mut r10 := r10
          let mut r11 := r11
          let mut work_1 := work_1
          -- while i > 0 {
          if decide ((UInt256.ofNat 0).toNat < i_r0.toNat) then
            -- *limb(wp, i) = *limb(wp, i) | (*limb(wp, i - 1) >> (64 - shift))
            let v73 ← limb wp i_r0
            let v74 ← mloadInt (.u 64) v73
            let v79 ← limb wp (i_r0 - (UInt256.ofNat 1))
            let v80 ← mloadInt (.u 64) v79
            let v83 := v80 >>> (IntTy.subW (.u 64) (UInt256.ofNat 64) shift)
            let v87 ← limb wp i_r0
            mstoreInt (.u 64) v87 (v74 ||| v83)
            -- *limb(wp, i - 1) = *limb(wp, i - 1) << shift
            let v93 ← limb wp (i_r0 - (UInt256.ofNat 1))
            let v94 ← mloadInt (.u 64) v93
            let v100 ← limb wp (i_r0 - (UInt256.ofNat 1))
            mstoreInt (.u 64) v100 (IntTy.shl (.u 64) v94 shift)
            -- i -= 1
            i_r0 := (i_r0 - (UInt256.ofNat 1))
            loop_8 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
          else
            join_1 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
        partial_fixpoint
        loop_8 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
      else
        join_1 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
  let join_5 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 => do
    let mut i_r0 := i_r0
    let mut r := r
    let mut i_r2 := i_r2
    let mut j := j
    let mut carry := carry
    let mut borrow := borrow
    let mut add := add
    let mut r7 := r7
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut r11 := r11
    let mut work_1 := work_1
    join i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 false
  if decide (count.toNat < v17.toNat) then
    let v20 ← MemArray_len_u64 work_1
    if decide (n.toNat < v20.toNat) then
      let v24 ← MemArray_len_u64 self.limbs
      if decide (n.toNat ≤ v24.toNat) then
        join i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 true
      else
        join_5 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
    else
      join_5 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
  else
    join_5 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1

/-- `Montgomery::mul` -/
def Montgomery_mul (self : Montgomery) (a : MemSlice_u64) (b : MemSlice_u64) (t_3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut sum : UInt256 := default
  let mut carry : UInt256 := default
  let mut reduced : UInt256 := default
  let mut reduced_carry : UInt256 := default
  let mut at_least : Bool := default
  let mut j : UInt256 := default
  let mut borrow : UInt256 := default
  let mut r7 : Range_Known_0_Unknown := default
  let mut r8 : Range_Known_0_Unknown := default
  let mut r9 : Range_Known_1_Unknown := default
  let mut r10 : Range_Known_0_Unknown := default
  let mut t_3 := t_3
  -- let n: usize = self.len.downcast_truncate()
  let n ← downcast_truncate_u64_usize self.len
  -- core::assert(a.len() >= n && b.len() >= n && self.limbs.len() >= n && t.len() >= n + 2)
  let v12 ← MemSlice_len_u64 a
  let join : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v4 => do
    let mut sum := sum
    let mut carry := carry
    let mut reduced := reduced
    let mut reduced_carry := reduced_carry
    let mut at_least := at_least
    let mut j := j
    let mut borrow := borrow
    let mut r7 := r7
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut t_3 := t_3
    -- core::assert(a.len() >= n && b.len() >= n && self.limbs.len() >= n && t.len() >= n + 2)
    let _ ← «assert» v4
    -- let ap = a.ptr()
    let ap ← MemSlice_ptr_u64 a
    -- let bp = b.ptr()
    let bp ← MemSlice_ptr_u64 b
    -- let mp = self.limbs.ptr()
    let mp ← MemArray_ptr_u64 self.limbs
    -- let tp = t.ptr()
    let tp ← MemArray_ptr_u64 t_3
    -- for j in 0..n + 1 { unsafe { *limb(tp, j) = 0 } }
    let v45 := (Range_Known_0_Unknown.mk () (n + (UInt256.ofNat 1)))
    r7 := v45
    let v47 ← Range_start_0 r7
    let rec loop (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v5 : UInt256) : FeM (Unit × MemArray_u64) := do
      let mut sum := sum
      let mut carry := carry
      let mut reduced := reduced
      let mut reduced_carry := reduced_carry
      let mut at_least := at_least
      let mut j := j
      let mut borrow := borrow
      let mut r7 := r7
      let mut r8 := r8
      let mut r9 := r9
      let mut r10 := r10
      let mut t_3 := t_3
      -- for j in 0..n + 1 { unsafe { *limb(tp, j) = 0 } }
      let v50 ← Range_has_0 r7 v5
      if v50 then
        let v52 ← Range_at_0 r7 v5
        -- *limb(tp, j) = 0
        let v55 ← limb tp v52
        mstoreInt (.u 64) v55 (UInt256.ofNat 0)
        -- for j in 0..n + 1 { unsafe { *limb(tp, j) = 0 } }
        let v58 ← Range_next_0 r7 v5
        loop sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v58
      else
        -- for i in 0..n {
        r8 := (Range_Known_0_Unknown.mk () n)
        let v64 ← Range_start_0 r8
        let rec loop_1 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
          let mut sum := sum
          let mut carry := carry
          let mut reduced := reduced
          let mut reduced_carry := reduced_carry
          let mut at_least := at_least
          let mut j := j
          let mut borrow := borrow
          let mut r7 := r7
          let mut r8 := r8
          let mut r9 := r9
          let mut r10 := r10
          let mut t_3 := t_3
          -- for i in 0..n {
          let v67 ← Range_has_0 r8 v6
          if v67 then
            let v69 ← Range_at_0 r8 v6
            -- *limb(bp, i)
            let v71 ← limb bp v69
            let factor ← mloadInt (.u 64) v71
            -- *tp
            let v75 ← mloadInt (.u 64) tp
            -- *ap
            let v77 ← mloadInt (.u 64) ap
            -- var sum = unsafe { *tp } as u128 + unsafe { *ap } as u128 * factor
            let sum_80 := IntTy.addW (.u 128) v75 (IntTy.mulW (.u 128) v77 factor)
            sum := sum_80
            -- var carry: u64 = (sum >> 64).downcast_truncate()
            let carry_86 ← downcast_truncate_u128_u64 (sum >>> (UInt256.ofNat 64))
            carry := carry_86
            -- let low: u64 = sum.downcast_truncate()
            let low_90 ← downcast_truncate_u128_u64 sum
            -- let m = (low * self.inverse) as u128
            let m := IntTy.mulW (.u 64) low_90 self.inverse
            -- *mp
            let v97 ← mloadInt (.u 64) mp
            -- var reduced = low as u128 + m * unsafe { *mp } as u128
            let reduced_100 := IntTy.addW (.u 128) low_90 (IntTy.mulW (.u 128) m v97)
            reduced := reduced_100
            -- var reduced_carry: u64 = (reduced >> 64).downcast_truncate()
            let reduced_carry_106 ← downcast_truncate_u128_u64 (reduced >>> (UInt256.ofNat 64))
            reduced_carry := reduced_carry_106
            -- for j in 1..n {
            r9 := (Range_Known_1_Unknown.mk () n)
            let v112 ← Range_start_1 r9
            let rec loop_2 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
              let mut sum := sum
              let mut carry := carry
              let mut reduced := reduced
              let mut reduced_carry := reduced_carry
              let mut at_least := at_least
              let mut j := j
              let mut borrow := borrow
              let mut r7 := r7
              let mut r8 := r8
              let mut r9 := r9
              let mut r10 := r10
              let mut t_3 := t_3
              -- for j in 1..n {
              let v115 ← Range_has_1 r9 v7
              if v115 then
                let v117 ← Range_at_1 r9 v7
                let j_118 := v117
                -- *limb(tp, j)
                let v119 ← limb tp j_118
                let v120 ← mloadInt (.u 64) v119
                -- *limb(ap, j)
                let v122 ← limb ap j_118
                let v123 ← mloadInt (.u 64) v122
                -- sum = unsafe { *limb(tp, j) } as u128 + unsafe { *limb(ap, j) } as u128 * factor + carry as u128
                let v126 := IntTy.addW (.u 128) v120 (IntTy.mulW (.u 128) v123 factor)
                sum := (IntTy.addW (.u 128) v126 carry)
                -- carry = (sum >> 64).downcast_truncate()
                let carry_136 ← downcast_truncate_u128_u64 (sum >>> (UInt256.ofNat 64))
                carry := carry_136
                -- let low: u64 = sum.downcast_truncate()
                let low_141 ← downcast_truncate_u128_u64 sum
                -- *limb(mp, j)
                let v144 ← limb mp j_118
                let v145 ← mloadInt (.u 64) v144
                -- reduced = low as u128 + m * unsafe { *limb(mp, j) } as u128 + reduced_carry as u128
                let v148 := IntTy.addW (.u 128) low_141 (IntTy.mulW (.u 128) m v145)
                reduced := (IntTy.addW (.u 128) v148 reduced_carry)
                -- reduced_carry = (reduced >> 64).downcast_truncate()
                let reduced_carry_158 ← downcast_truncate_u128_u64 (reduced >>> (UInt256.ofNat 64))
                reduced_carry := reduced_carry_158
                -- *limb(tp, j - 1) = reduced.downcast_truncate()
                let v163 ← downcast_truncate_u128_u64 reduced
                let v166 ← limb tp (j_118 - (UInt256.ofNat 1))
                mstoreInt (.u 64) v166 v163
                -- for j in 1..n {
                let v169 ← Range_next_1 r9 v7
                loop_2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v169
              else
                -- *limb(tp, n)
                let v171 ← limb tp n
                let v172 ← mloadInt (.u 64) v171
                -- let top = unsafe { *limb(tp, n) } as u128 + carry as u128 + reduced_carry as u128
                let top := IntTy.addW (.u 128) (IntTy.addW (.u 128) v172 carry) reduced_carry
                -- *limb(tp, n - 1) = top.downcast_truncate()
                let v181 ← downcast_truncate_u128_u64 top
                let v184 ← limb tp (n - (UInt256.ofNat 1))
                mstoreInt (.u 64) v184 v181
                -- *limb(tp, n) = (top >> 64).downcast_truncate()
                let v188 ← downcast_truncate_u128_u64 (top >>> (UInt256.ofNat 64))
                let v189 ← limb tp n
                mstoreInt (.u 64) v189 v188
                -- for i in 0..n {
                let v192 ← Range_next_0 r8 v6
                loop_1 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v192
            partial_fixpoint
            loop_2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v112
          else
            -- var at_least = t[n] != 0
            let v195 ← MemArray_index_ref_u64 t_3 n
            let v196 ← mloadInt (.u 64) v195
            at_least := (decide (v196 ≠ (UInt256.ofNat 0)))
            let join_1 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
              let mut sum := sum
              let mut carry := carry
              let mut reduced := reduced
              let mut reduced_carry := reduced_carry
              let mut at_least := at_least
              let mut j := j
              let mut borrow := borrow
              let mut r7 := r7
              let mut r8 := r8
              let mut r9 := r9
              let mut r10 := r10
              let mut t_3 := t_3
              let join_2 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
                let mut sum := sum
                let mut carry := carry
                let mut reduced := reduced
                let mut reduced_carry := reduced_carry
                let mut at_least := at_least
                let mut j := j
                let mut borrow := borrow
                let mut r7 := r7
                let mut r8 := r8
                let mut r9 := r9
                let mut r10 := r10
                let mut t_3 := t_3
                return ((), t_3)
              -- if at_least {
              if at_least then
                -- var borrow: u64 = 0
                borrow := (UInt256.ofNat 0)
                -- for j in 0..n {
                r10 := (Range_Known_0_Unknown.mk () n)
                let v254 ← Range_start_0 r10
                let rec loop_3 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v8 : UInt256) : FeM (Unit × MemArray_u64) := do
                  let mut sum := sum
                  let mut carry := carry
                  let mut reduced := reduced
                  let mut reduced_carry := reduced_carry
                  let mut at_least := at_least
                  let mut j := j
                  let mut borrow := borrow
                  let mut r7 := r7
                  let mut r8 := r8
                  let mut r9 := r9
                  let mut r10 := r10
                  let mut t_3 := t_3
                  -- for j in 0..n {
                  let v257 ← Range_has_0 r10 v8
                  if v257 then
                    let v259 ← Range_at_0 r10 v8
                    let j_260 := v259
                    -- let difference = t[j] as u128 - self.limbs[j] as u128 - borrow as u128
                    let v262 ← MemArray_index_ref_u64 t_3 j_260
                    let v263 ← mloadInt (.u 64) v262
                    let v267 ← MemArray_index_ref_u64 self.limbs j_260
                    let v268 ← mloadInt (.u 64) v267
                    let difference := IntTy.subW (.u 128) (IntTy.subW (.u 128) v263 v268) borrow
                    -- t[j] = difference.downcast_truncate()
                    let v275 ← downcast_truncate_u128_u64 difference
                    let (v276, v276_k) ← MemArray_index_mut_u64 t_3 j_260
                    mstoreInt (.u 64) v276 v275
                    let v276_0 ← v276_k v276
                    t_3 := v276_0
                    -- borrow = (difference >> 127).downcast_truncate()
                    let borrow_280 ← downcast_truncate_u128_u64 (difference >>> (UInt256.ofNat 127))
                    borrow := borrow_280
                    -- for j in 0..n {
                    let v284 ← Range_next_0 r10 v8
                    loop_3 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v284
                  else
                    join_2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                partial_fixpoint
                loop_3 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v254
              else
                join_2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
            -- if !at_least {
            if !at_least then
              -- at_least = true
              at_least := true
              -- var j = n
              j := n
              let rec loop_4 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
                let mut sum := sum
                let mut carry := carry
                let mut reduced := reduced
                let mut reduced_carry := reduced_carry
                let mut at_least := at_least
                let mut j := j
                let mut borrow := borrow
                let mut r7 := r7
                let mut r8 := r8
                let mut r9 := r9
                let mut r10 := r10
                let mut t_3 := t_3
                let join_3 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
                  let mut sum := sum
                  let mut carry := carry
                  let mut reduced := reduced
                  let mut reduced_carry := reduced_carry
                  let mut at_least := at_least
                  let mut j := j
                  let mut borrow := borrow
                  let mut r7 := r7
                  let mut r8 := r8
                  let mut r9 := r9
                  let mut r10 := r10
                  let mut t_3 := t_3
                  join_1 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                -- while j > 0 {
                if decide ((UInt256.ofNat 0).toNat < j.toNat) then
                  -- j -= 1
                  j := (j - (UInt256.ofNat 1))
                  -- if t[j] != self.limbs[j] {
                  let j_215 := j
                  let v218 ← MemArray_index_ref_u64 t_3 j_215
                  let v219 ← mloadInt (.u 64) v218
                  let j_220 := j
                  let v224 ← MemArray_index_ref_u64 self.limbs j_220
                  let v225 ← mloadInt (.u 64) v224
                  if decide (v219 ≠ v225) then
                    -- at_least = t[j] > self.limbs[j]
                    let j_227 := j
                    let v230 ← MemArray_index_ref_u64 t_3 j_227
                    let v231 ← mloadInt (.u 64) v230
                    let j_232 := j
                    let v236 ← MemArray_index_ref_u64 self.limbs j_232
                    let v237 ← mloadInt (.u 64) v236
                    at_least := (decide (v237.toNat < v231.toNat))
                    join_3 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                  else
                    loop_4 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                else
                  join_3 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
              partial_fixpoint
              loop_4 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
            else
              join_1 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
        partial_fixpoint
        loop_1 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v64
    partial_fixpoint
    loop sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v47
  let join_4 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
    let mut sum := sum
    let mut carry := carry
    let mut reduced := reduced
    let mut reduced_carry := reduced_carry
    let mut at_least := at_least
    let mut j := j
    let mut borrow := borrow
    let mut r7 := r7
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut t_3 := t_3
    join sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 false
  if decide (n.toNat ≤ v12.toNat) then
    let v14 ← MemSlice_len_u64 b
    if decide (n.toNat ≤ v14.toNat) then
      let v18 ← MemArray_len_u64 self.limbs
      if decide (n.toNat ≤ v18.toNat) then
        let v21 ← MemArray_len_u64 t_3
        if decide ((n + (UInt256.ofNat 2)).toNat ≤ v21.toNat) then
          join sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 true
        else
          join_4 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
      else
        join_4 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
    else
      join_4 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
  else
    join_4 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3

/-- `Modulus::mul_mod` -/
def Modulus_mul_mod (self : Modulus) (a : MemArray_u64) (b : MemArray_u64) (work_3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut carry : UInt256 := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  let mut work_3 := work_3
  -- let n: usize = self.len.downcast_truncate()
  let n ← downcast_truncate_u64_usize self.len
  -- core::assert(a.len() >= n && b.len() >= n && work.len() > 2 * n)
  let v11 ← MemArray_len_u64 a
  let join : UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun carry r1 r2 r3 work_3 v4 => do
    let mut carry := carry
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut work_3 := work_3
    -- core::assert(a.len() >= n && b.len() >= n && work.len() > 2 * n)
    let _ ← «assert» v4
    -- let ap = a.ptr()
    let ap ← MemArray_ptr_u64 a
    -- let bp = b.ptr()
    let bp ← MemArray_ptr_u64 b
    -- let wp = work.ptr()
    let wp ← MemArray_ptr_u64 work_3
    -- for i in 0..2 * n { unsafe { *limb(wp, i) = 0 } }
    let v36 := (Range_Known_0_Unknown.mk () ((UInt256.ofNat 2) * n))
    r1 := v36
    let v38 ← Range_start_0 r1
    let rec loop (carry : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (work_3 : MemArray_u64) (v5 : UInt256) : FeM (Unit × MemArray_u64) := do
      let mut carry := carry
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut work_3 := work_3
      -- for i in 0..2 * n { unsafe { *limb(wp, i) = 0 } }
      let v41 ← Range_has_0 r1 v5
      if v41 then
        let v43 ← Range_at_0 r1 v5
        -- *limb(wp, i) = 0
        let v46 ← limb wp v43
        mstoreInt (.u 64) v46 (UInt256.ofNat 0)
        -- for i in 0..2 * n { unsafe { *limb(wp, i) = 0 } }
        let v49 ← Range_next_0 r1 v5
        loop carry r1 r2 r3 work_3 v49
      else
        -- for i in 0..n {
        r2 := (Range_Known_0_Unknown.mk () n)
        let v55 ← Range_start_0 r2
        let rec loop_1 (carry : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (work_3 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
          let mut carry := carry
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut work_3 := work_3
          -- for i in 0..n {
          let v58 ← Range_has_0 r2 v6
          if v58 then
            let v60 ← Range_at_0 r2 v6
            let i_61 := v60
            -- *limb(ap, i)
            let v62 ← limb ap i_61
            let left ← mloadInt (.u 64) v62
            -- var carry: u64 = 0
            carry := (UInt256.ofNat 0)
            -- for j in 0..n {
            r3 := (Range_Known_0_Unknown.mk () n)
            let v72 ← Range_start_0 r3
            let rec loop_2 (carry : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (work_3 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
              let mut carry := carry
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut work_3 := work_3
              -- for j in 0..n {
              let v75 ← Range_has_0 r3 v7
              if v75 then
                let v77 ← Range_at_0 r3 v7
                let j := v77
                -- *limb(bp, j)
                let v79 ← limb bp j
                let v80 ← mloadInt (.u 64) v79
                -- *limb(wp, i + j)
                let v84 ← limb wp (i_61 + j)
                let v85 ← mloadInt (.u 64) v84
                -- let t = left * unsafe { *limb(bp, j) } as u128 + unsafe { *limb(wp, i + j) } as u128 + carry as u128
                let v87 := IntTy.addW (.u 128) (IntTy.mulW (.u 128) left v80) v85
                let t := IntTy.addW (.u 128) v87 carry
                -- *limb(wp, i + j) = t.downcast_truncate()
                let v92 ← downcast_truncate_u128_u64 t
                let v94 ← limb wp (i_61 + j)
                mstoreInt (.u 64) v94 v92
                -- carry = (t >> 64).downcast_truncate()
                let carry_98 ← downcast_truncate_u128_u64 (t >>> (UInt256.ofNat 64))
                carry := carry_98
                -- for j in 0..n {
                let v102 ← Range_next_0 r3 v7
                loop_2 carry r1 r2 r3 work_3 v102
              else
                -- *limb(wp, i + n) = carry
                let carry_104 := carry
                let v107 ← limb wp (i_61 + n)
                mstoreInt (.u 64) v107 carry_104
                -- for i in 0..n {
                let v110 ← Range_next_0 r2 v6
                loop_1 carry r1 r2 r3 work_3 v110
            partial_fixpoint
            loop_2 carry r1 r2 r3 work_3 v72
          else
            -- self.reduce(mut work, len: 2 * self.len)
            let (v116, v116_0) ← Modulus_reduce self work_3 (IntTy.mulW (.u 64) (UInt256.ofNat 2) self.len)
            work_3 := v116_0
            return ((), work_3)
        partial_fixpoint
        loop_1 carry r1 r2 r3 work_3 v55
    partial_fixpoint
    loop carry r1 r2 r3 work_3 v38
  let join_1 : UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun carry r1 r2 r3 work_3 => do
    let mut carry := carry
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut work_3 := work_3
    join carry r1 r2 r3 work_3 false
  if decide (n.toNat ≤ v11.toNat) then
    let v13 ← MemArray_len_u64 b
    if decide (n.toNat ≤ v13.toNat) then
      let v16 ← MemArray_len_u64 work_3
      if decide (((UInt256.ofNat 2) * n).toNat < v16.toNat) then
        join carry r1 r2 r3 work_3 true
      else
        join_1 carry r1 r2 r3 work_3
    else
      join_1 carry r1 r2 r3 work_3
  else
    join_1 carry r1 r2 r3 work_3

/-- `lib::modexp::large_power` -/
def large_power (exponent : Exponent) (base : MemArray_u64) (modulus : Modulus) (value : MemArray_u64) (size : UInt256) : FeM Option_MemArray_u64 := do
  let mut result : MemArray_u64 := default
  let mut work : MemArray_u64 := default
  let mut montgomery : Montgomery := default
  let mut table : MemArray_u64 := default
  let mut squared : MemArray_u64 := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut r7 : Range_Known_0_Unknown := default
  let mut r8 : Range_Known_1_Unknown := default
  let mut r9 : Range_Unknown_Unknown := default
  let mut v40 : MemSlice_u64 := default
  let mut v105 : MemSlice_u64 := default
  let mut v106 : MemSlice_u64 := default
  let mut v133 : MemSlice_u64 := default
  let mut v172 : MemSlice_u64 := default
  let mut v173 : MemSlice_u64 := default
  let mut v288 : MemSlice_u64 := default
  let mut v217 : MemSlice_u64 := default
  let mut v249 : MemSlice_u64 := default
  let mut v254 : MemSlice_u64 := default
  -- let total = exponent.total
  let total := exponent.total
  -- var result = MemArray<u64>::try_new_zeroed(size)?
  let v11 ← MemArray_try_new_zeroed_u64 size
  if match v11 with | .Some .. => true | _ => false then
    let result_13 := v11.Some_0
    result := result_13
    -- var work = MemArray<u64>::try_new_zeroed(2 * size + 1)?
    let v17 ← checkedMul (UInt256.ofNat 2) size
    let v19 ← checkedAdd v17 (UInt256.ofNat 1)
    let v20 ← MemArray_try_new_zeroed_u64 v19
    if match v20 with | .Some .. => true | _ => false then
      let work_22 := v20.Some_0
      work := work_22
      -- if total - exponent.top == 2 && exponent.bit(total - 1) == 0 {
      let v25 ← checkedSub total exponent.top
      let join : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
        let mut result := result
        let mut work := work
        let mut montgomery := montgomery
        let mut table := table
        let mut squared := squared
        let mut index := index
        let mut started := started
        let mut r7 := r7
        let mut r8 := r8
        let mut r9 := r9
        let mut v40 := v40
        let mut v105 := v105
        let mut v106 := v106
        let mut v133 := v133
        let mut v172 := v172
        let mut v173 := v173
        let mut v288 := v288
        let mut v217 := v217
        let mut v249 := v249
        let mut v254 := v254
        -- let montgomery = Montgomery::new(value, len: size.downcast_truncate())
        let v49 ← downcast_truncate_usize_u64 size
        let montgomery_50 ← Montgomery_new value v49
        montgomery := montgomery_50
        -- let n: u64 = size.downcast_truncate()
        let n ← downcast_truncate_usize_u64 size
        -- let width = window_width(total - exponent.top)
        let v55 ← checkedSub total exponent.top
        let width ← window_width v55
        -- let entries: usize = (1 << (width - 1)).downcast_truncate()
        let v60 ← checkedSub width (UInt256.ofNat 1)
        let entries ← downcast_truncate_usize_usize ((UInt256.ofNat 1) <<< v60)
        -- var table = MemArray<u64>::try_new_zeroed(entries * size)?
        let v65 ← checkedMul entries size
        let v66 ← MemArray_try_new_zeroed_u64 v65
        if match v66 with | .Some .. => true | _ => false then
          let table_68 := v66.Some_0
          table := table_68
          -- for i in 0..size {
          r7 := (Range_Known_0_Unknown.mk () size)
          let v75 ← Range_start_0 r7
          let rec loop (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v5 : UInt256) : FeM Option_MemArray_u64 := do
            let mut result := result
            let mut work := work
            let mut montgomery := montgomery
            let mut table := table
            let mut squared := squared
            let mut index := index
            let mut started := started
            let mut r7 := r7
            let mut r8 := r8
            let mut r9 := r9
            let mut v40 := v40
            let mut v105 := v105
            let mut v106 := v106
            let mut v133 := v133
            let mut v172 := v172
            let mut v173 := v173
            let mut v288 := v288
            let mut v217 := v217
            let mut v249 := v249
            let mut v254 := v254
            -- for i in 0..size {
            let v78 ← Range_has_0 r7 v5
            if v78 then
              let v80 ← Range_at_0 r7 v5
              let i := v80
              -- work[i] = 0
              let (v84, v84_k) ← MemArray_index_mut_u64 work i
              mstoreInt (.u 64) v84 (UInt256.ofNat 0)
              let v84_0 ← v84_k v84
              work := v84_0
              -- work[size + i] = base[i]
              let v86 ← MemArray_index_ref_u64 base i
              let v87 ← mloadInt (.u 64) v86
              let v89 ← checkedAdd size i
              let (v91, v91_k) ← MemArray_index_mut_u64 work v89
              mstoreInt (.u 64) v91 v87
              let v91_0 ← v91_k v91
              work := v91_0
              -- for i in 0..size {
              let v94 ← Range_next_0 r7 v5
              loop result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v94
            else
              -- modulus.reduce(mut work, len: 2 * n)
              let v98 ← IntTy.mul (.u 64) (UInt256.ofNat 2) n
              let (v99, v99_0) ← Modulus_reduce modulus work v98
              work := v99_0
              -- table.as_slice().head(size).copy_from(work.as_slice().head(size))
              let v101 ← MemArray_as_slice_ref_u64 work
              let v103 ← MemSlice_head_ref_u64 v101 size
              let (v105_g, v105_k) ← MemArray_as_slice_mut_u64 table
              v105 := v105_g
              let (v106_g, v106_k) ← MemSlice_head_mut_u64 v105 size
              v106 := v106_g
              let (v108, v108_0) ← MemSlice_copy_from_u64 v106 v103
              v106 := v108_0
              let v106_0 ← v106_k v106
              v105 := v106_0
              let v105_0 ← v105_k v105
              table := v105_0
              -- var squared = MemArray<u64>::try_new_zeroed(size)?
              let v109 ← MemArray_try_new_zeroed_u64 size
              if match v109 with | .Some .. => true | _ => false then
                let squared_111 := v109.Some_0
                squared := squared_111
                let join_1 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
                  let mut result := result
                  let mut work := work
                  let mut montgomery := montgomery
                  let mut table := table
                  let mut squared := squared
                  let mut index := index
                  let mut started := started
                  let mut r7 := r7
                  let mut r8 := r8
                  let mut r9 := r9
                  let mut v40 := v40
                  let mut v105 := v105
                  let mut v106 := v106
                  let mut v133 := v133
                  let mut v172 := v172
                  let mut v173 := v173
                  let mut v288 := v288
                  let mut v217 := v217
                  let mut v249 := v249
                  let mut v254 := v254
                  -- for k in 1..entries {
                  r8 := (Range_Known_1_Unknown.mk () entries)
                  let v143 ← Range_start_1 r8
                  let rec loop_1 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v6 : UInt256) : FeM Option_MemArray_u64 := do
                    let mut result := result
                    let mut work := work
                    let mut montgomery := montgomery
                    let mut table := table
                    let mut squared := squared
                    let mut index := index
                    let mut started := started
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut r9 := r9
                    let mut v40 := v40
                    let mut v105 := v105
                    let mut v106 := v106
                    let mut v133 := v133
                    let mut v172 := v172
                    let mut v173 := v173
                    let mut v288 := v288
                    let mut v217 := v217
                    let mut v249 := v249
                    let mut v254 := v254
                    -- for k in 1..entries {
                    let v146 ← Range_has_1 r8 v6
                    if v146 then
                      let v148 ← Range_at_1 r8 v6
                      let k := v148
                      -- montgomery.mul(table.as_slice().slice(offset: (k - 1) * size, len: size), squared.as_slice(), mut work)
                      let v151 ← checkedSub k (UInt256.ofNat 1)
                      let v153 ← checkedMul v151 size
                      let v155 ← MemArray_as_slice_ref_u64 table
                      let v157 ← MemSlice_slice_ref_u64 v155 v153 size
                      let v159 ← MemArray_as_slice_ref_u64 squared
                      let (v164, v164_0) ← Montgomery_mul montgomery v157 v159 work
                      work := v164_0
                      -- table.as_slice().slice(offset: k * size, len: size).copy_from(work.as_slice().head(size))
                      let v166 ← MemArray_as_slice_ref_u64 work
                      let v168 ← MemSlice_head_ref_u64 v166 size
                      let v170 ← checkedMul k size
                      let (v172_g, v172_k) ← MemArray_as_slice_mut_u64 table
                      v172 := v172_g
                      let (v173_g, v173_k) ← MemSlice_slice_mut_u64 v172 v170 size
                      v173 := v173_g
                      let (v175, v175_0) ← MemSlice_copy_from_u64 v173 v168
                      v173 := v175_0
                      let v173_0 ← v173_k v173
                      v172 := v173_0
                      let v172_0 ← v172_k v172
                      table := v172_0
                      -- for k in 1..entries {
                      let v177 ← Range_next_1 r8 v6
                      loop_1 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v177
                    else
                      -- var index = exponent.top
                      index := exponent.top
                      -- var started = false
                      started := false
                      let rec loop_2 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) : FeM Option_MemArray_u64 := do
                        let mut result := result
                        let mut work := work
                        let mut montgomery := montgomery
                        let mut table := table
                        let mut squared := squared
                        let mut index := index
                        let mut started := started
                        let mut r7 := r7
                        let mut r8 := r8
                        let mut r9 := r9
                        let mut v40 := v40
                        let mut v105 := v105
                        let mut v106 := v106
                        let mut v133 := v133
                        let mut v172 := v172
                        let mut v173 := v173
                        let mut v288 := v288
                        let mut v217 := v217
                        let mut v249 := v249
                        let mut v254 := v254
                        -- while index < total {
                        if decide (index.toNat < total.toNat) then
                          -- let window = exponent.window(index, width)
                          let window ← Exponent_window exponent index width
                          let rec loop_3 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) : FeM Option_MemArray_u64 := do
                            let mut result := result
                            let mut work := work
                            let mut montgomery := montgomery
                            let mut table := table
                            let mut squared := squared
                            let mut index := index
                            let mut started := started
                            let mut r7 := r7
                            let mut r8 := r8
                            let mut r9 := r9
                            let mut v40 := v40
                            let mut v105 := v105
                            let mut v106 := v106
                            let mut v133 := v133
                            let mut v172 := v172
                            let mut v173 := v173
                            let mut v288 := v288
                            let mut v217 := v217
                            let mut v249 := v249
                            let mut v254 := v254
                            let join_2 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
                              let mut result := result
                              let mut work := work
                              let mut montgomery := montgomery
                              let mut table := table
                              let mut squared := squared
                              let mut index := index
                              let mut started := started
                              let mut r7 := r7
                              let mut r8 := r8
                              let mut r9 := r9
                              let mut v40 := v40
                              let mut v105 := v105
                              let mut v106 := v106
                              let mut v133 := v133
                              let mut v172 := v172
                              let mut v173 := v173
                              let mut v288 := v288
                              let mut v217 := v217
                              let mut v249 := v249
                              let mut v254 := v254
                              -- index = window.end
                              index := window.«end»
                              loop_2 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                            -- if window.odd {
                            if window.odd then
                              -- let entry = table.as_slice().slice(offset: window.entry * size, len: size)
                              let v230 ← checkedMul window.entry size
                              let v232 ← MemArray_as_slice_ref_u64 table
                              let v234 ← MemSlice_slice_ref_u64 v232 v230 size
                              let join_3 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → Unit → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v8 => do
                                let mut result := result
                                let mut work := work
                                let mut montgomery := montgomery
                                let mut table := table
                                let mut squared := squared
                                let mut index := index
                                let mut started := started
                                let mut r7 := r7
                                let mut r8 := r8
                                let mut r9 := r9
                                let mut v40 := v40
                                let mut v105 := v105
                                let mut v106 := v106
                                let mut v133 := v133
                                let mut v172 := v172
                                let mut v173 := v173
                                let mut v288 := v288
                                let mut v217 := v217
                                let mut v249 := v249
                                let mut v254 := v254
                                join_2 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                              -- if started {
                              if started then
                                -- montgomery.mul(result.as_slice(), entry, mut work)
                                let v238 ← MemArray_as_slice_ref_u64 result
                                let (v243, v243_0) ← Montgomery_mul montgomery v238 v234 work
                                work := v243_0
                                -- result.as_slice().copy_from(work.as_slice().head(size))
                                let v245 ← MemArray_as_slice_ref_u64 work
                                let v247 ← MemSlice_head_ref_u64 v245 size
                                let (v249_g, v249_k) ← MemArray_as_slice_mut_u64 result
                                v249 := v249_g
                                let (v251, v251_0) ← MemSlice_copy_from_u64 v249 v247
                                v249 := v251_0
                                let v249_0 ← v249_k v249
                                result := v249_0
                                join_3 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v251
                              else
                                -- result.as_slice().copy_from(entry)
                                let (v254_g, v254_k) ← MemArray_as_slice_mut_u64 result
                                v254 := v254_g
                                let (v256, v256_0) ← MemSlice_copy_from_u64 v254 v234
                                v254 := v256_0
                                let v254_0 ← v254_k v254
                                result := v254_0
                                -- started = true
                                started := true
                                join_3 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 ()
                            else
                              join_2 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                          partial_fixpoint
                          -- if started {
                          if started then
                            -- for _ in index..window.end {
                            r9 := (Range_Unknown_Unknown.mk index window.«end»)
                            let v197 ← Range_Unknown_Unknown_start r9
                            let rec loop_4 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v7 : UInt256) : FeM Option_MemArray_u64 := do
                              let mut result := result
                              let mut work := work
                              let mut montgomery := montgomery
                              let mut table := table
                              let mut squared := squared
                              let mut index := index
                              let mut started := started
                              let mut r7 := r7
                              let mut r8 := r8
                              let mut r9 := r9
                              let mut v40 := v40
                              let mut v105 := v105
                              let mut v106 := v106
                              let mut v133 := v133
                              let mut v172 := v172
                              let mut v173 := v173
                              let mut v288 := v288
                              let mut v217 := v217
                              let mut v249 := v249
                              let mut v254 := v254
                              -- for _ in index..window.end {
                              let v200 ← Range_Unknown_Unknown_has r9 v7
                              if v200 then
                                let _ ← Range_Unknown_Unknown_at r9 v7
                                -- montgomery.mul(result.as_slice(), result.as_slice(), mut work)
                                let v204 ← MemArray_as_slice_ref_u64 result
                                let v206 ← MemArray_as_slice_ref_u64 result
                                let (v211, v211_0) ← Montgomery_mul montgomery v204 v206 work
                                work := v211_0
                                -- result.as_slice().copy_from(work.as_slice().head(size))
                                let v213 ← MemArray_as_slice_ref_u64 work
                                let v215 ← MemSlice_head_ref_u64 v213 size
                                let (v217_g, v217_k) ← MemArray_as_slice_mut_u64 result
                                v217 := v217_g
                                let (v219, v219_0) ← MemSlice_copy_from_u64 v217 v215
                                v217 := v219_0
                                let v217_0 ← v217_k v217
                                result := v217_0
                                -- for _ in index..window.end {
                                let v221 ← Range_Unknown_Unknown_next r9 v7
                                loop_4 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v221
                              else
                                loop_3 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                            partial_fixpoint
                            loop_4 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v197
                          else
                            loop_3 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                        else
                          -- squared.zero()
                          let (v268, v268_0) ← MemArray_zero_u64 squared
                          squared := v268_0
                          -- squared[0] = 1
                          let (v272, v272_k) ← MemArray_index_mut_u64 squared (UInt256.ofNat 0)
                          mstoreInt (.u 64) v272 (UInt256.ofNat 1)
                          let v272_0 ← v272_k v272
                          squared := v272_0
                          -- montgomery.mul(result.as_slice(), squared.as_slice(), mut work)
                          let v275 ← MemArray_as_slice_ref_u64 result
                          let v277 ← MemArray_as_slice_ref_u64 squared
                          let (v282, v282_0) ← Montgomery_mul montgomery v275 v277 work
                          work := v282_0
                          -- result.as_slice().copy_from(work.as_slice().head(size))
                          let v284 ← MemArray_as_slice_ref_u64 work
                          let v286 ← MemSlice_head_ref_u64 v284 size
                          let (v288_g, v288_k) ← MemArray_as_slice_mut_u64 result
                          v288 := v288_g
                          let (v290, v290_0) ← MemSlice_copy_from_u64 v288 v286
                          v288 := v290_0
                          let v288_0 ← v288_k v288
                          result := v288_0
                          -- Option::Some(result)
                          let result_291 := result
                          let _ ← MemArray_drop_u64 squared
                          let _ ← MemArray_drop_u64 table
                          let _ ← MemArray_drop_u64 montgomery.limbs
                          let _ ← MemArray_drop_u64 work
                          return Option_MemArray_u64.Some result_291
                      partial_fixpoint
                      loop_2 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                  partial_fixpoint
                  loop_1 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v143
                -- if entries > 1 {
                if decide ((UInt256.ofNat 1).toNat < entries.toNat) then
                  -- montgomery.mul(table.as_slice().head(size), table.as_slice().head(size), mut work)
                  let v116 ← MemArray_as_slice_ref_u64 table
                  let v118 ← MemSlice_head_ref_u64 v116 size
                  let v120 ← MemArray_as_slice_ref_u64 table
                  let v122 ← MemSlice_head_ref_u64 v120 size
                  let (v127, v127_0) ← Montgomery_mul montgomery v118 v122 work
                  work := v127_0
                  -- squared.as_slice().copy_from(work.as_slice().head(size))
                  let v129 ← MemArray_as_slice_ref_u64 work
                  let v131 ← MemSlice_head_ref_u64 v129 size
                  let (v133_g, v133_k) ← MemArray_as_slice_mut_u64 squared
                  v133 := v133_g
                  let (v135, v135_0) ← MemSlice_copy_from_u64 v133 v131
                  v133 := v135_0
                  let v133_0 ← v133_k v133
                  squared := v133_0
                  join_1 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                else
                  join_1 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
              else
                match (← pure v109) with
                | .Some .. =>
                  let _ ← MemArray_drop_u64 v109.Some_0
                  pure ()
                | _ => pure ()
                let _ ← MemArray_drop_u64 table
                let _ ← MemArray_drop_u64 montgomery.limbs
                let _ ← MemArray_drop_u64 work
                let _ ← MemArray_drop_u64 result
                return Option_MemArray_u64.None
          partial_fixpoint
          loop result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v75
        else
          match (← pure v66) with
          | .Some .. =>
            let _ ← MemArray_drop_u64 v66.Some_0
            pure ()
          | _ => pure ()
          let _ ← MemArray_drop_u64 montgomery.limbs
          let _ ← MemArray_drop_u64 work
          let _ ← MemArray_drop_u64 result
          return Option_MemArray_u64.None
      if decide (v25 = (UInt256.ofNat 2)) then
        let v29 ← checkedSub total (UInt256.ofNat 1)
        let v30 ← Exponent_bit exponent v29
        if decide (v30 = (UInt256.ofNat 0)) then
          -- modulus.mul_mod(base, base, mut work)
          let (v34, v34_0) ← Modulus_mul_mod modulus base base work
          work := v34_0
          -- result.as_slice().copy_from(work.as_slice().head(size))
          let v36 ← MemArray_as_slice_ref_u64 work
          let v38 ← MemSlice_head_ref_u64 v36 size
          let (v40_g, v40_k) ← MemArray_as_slice_mut_u64 result
          v40 := v40_g
          let (v42, v42_0) ← MemSlice_copy_from_u64 v40 v38
          v40 := v42_0
          let v40_0 ← v40_k v40
          result := v40_0
          -- return Option::Some(result)
          let result_43 := result
          let _ ← MemArray_drop_u64 work
          let _ ← MemArray_drop_u64 value
          return Option_MemArray_u64.Some result_43
        else
          join result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
      else
        join result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
    else
      match (← pure v20) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v20.Some_0
        pure ()
      | _ => pure ()
      let _ ← MemArray_drop_u64 result
      let _ ← MemArray_drop_u64 value
      return Option_MemArray_u64.None
  else
    match (← pure v11) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v11.Some_0
      pure ()
    | _ => pure ()
    let _ ← MemArray_drop_u64 value
    return Option_MemArray_u64.None

/-- `lib::modexp::odd_power` -/
def odd_power (exponent : Exponent) (base : MemArray_u64) (base_size : UInt256) (value : MemArray_u64) (n : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
  let mut modulus : Modulus := default
  let mut limbs : MemArray_u64 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut base := base
  let join : Modulus → MemArray_u64 → Range_Known_0_Unknown → MemArray_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun modulus limbs r2 base => do
    let mut modulus := modulus
    let mut limbs := limbs
    let mut r2 := r2
    let mut base := base
    -- let modulus = Modulus::try_new(value, len: n.downcast_truncate())?
    let v17 ← downcast_truncate_usize_u64 n
    let v18 ← Modulus_try_new value v17
    if match v18 with | .Some .. => true | _ => false then
      let modulus_20 := v18.Some_0
      modulus := modulus_20
      let join_1 : Modulus → MemArray_u64 → Range_Known_0_Unknown → MemArray_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun modulus limbs r2 base => do
        let mut modulus := modulus
        let mut limbs := limbs
        let mut r2 := r2
        let mut base := base
        -- if n <= 4 { return small_power_of(exponent, base, value, n) }
        if decide (n.toNat ≤ (UInt256.ofNat 4).toNat) then
          -- return small_power_of(exponent, base, value, n)
          let v37 ← small_power_of exponent base value n
          let _ ← MemArray_drop_u64 modulus.limbs
          return (v37, base)
        else
          -- var limbs = MemArray<u64>::try_new_zeroed(n)?
          let v40 ← MemArray_try_new_zeroed_u64 n
          if match v40 with | .Some .. => true | _ => false then
            let limbs_42 := v40.Some_0
            limbs := limbs_42
            -- for i in 0..n { limbs[i] = value[i] }
            r2 := (Range_Known_0_Unknown.mk () n)
            let v49 ← Range_start_0 r2
            let rec loop (modulus : Modulus) (limbs : MemArray_u64) (r2 : Range_Known_0_Unknown) (base : MemArray_u64) (v5 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
              let mut modulus := modulus
              let mut limbs := limbs
              let mut r2 := r2
              let mut base := base
              -- for i in 0..n { limbs[i] = value[i] }
              let v52 ← Range_has_0 r2 v5
              if v52 then
                let v54 ← Range_at_0 r2 v5
                let i := v54
                -- limbs[i] = value[i]
                let v56 ← MemArray_index_ref_u64 value i
                let v57 ← mloadInt (.u 64) v56
                let (v59, v59_k) ← MemArray_index_mut_u64 limbs i
                mstoreInt (.u 64) v59 v57
                let v59_0 ← v59_k v59
                limbs := v59_0
                -- for i in 0..n { limbs[i] = value[i] }
                let v62 ← Range_next_0 r2 v5
                loop modulus limbs r2 base v62
              else
                -- large_power(exponent, base, modulus, limbs, n)
                let limbs_65 := limbs
                let v69 ← large_power exponent base modulus limbs_65 n
                let _ ← MemArray_drop_u64 modulus.limbs
                return (v69, base)
            partial_fixpoint
            loop modulus limbs r2 base v49
          else
            match (← pure v40) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v40.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 modulus.limbs
            return (Option_MemArray_u64.None, base)
      -- if base_size > n { modulus.reduce(mut base, len: base_size.downcast_truncate()) }
      if decide (n.toNat < base_size.toNat) then
        -- modulus.reduce(mut base, len: base_size.downcast_truncate())
        let v26 ← downcast_truncate_usize_u64 base_size
        let (v28, v28_0) ← Modulus_reduce modulus base v26
        base := v28_0
        join_1 modulus limbs r2 base
      else
        join_1 modulus limbs r2 base
    else
      match (← pure v18) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v18.Some_0.limbs
        pure ()
      | _ => pure ()
      return (Option_MemArray_u64.None, base)
  -- if n <= 4 && base_size <= n { return small_power_of(exponent, base, value, n) }
  if decide (n.toNat ≤ (UInt256.ofNat 4).toNat) then
    if decide (base_size.toNat ≤ n.toNat) then
      -- return small_power_of(exponent, base, value, n)
      let v14 ← small_power_of exponent base value n
      return (v14, base)
    else
      join modulus limbs r2 base
  else
    join modulus limbs r2 base

/-- `lib::modexp::low_product` -/
def low_product (a : MemSlice_u64) (b : MemSlice_u64) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut a_len : UInt256 := default
  let mut b_len : UInt256 := default
  let mut carry : UInt256 := default
  let mut «at» : UInt256 := default
  let mut r4 : Range_Known_0_Unknown := default
  let mut r5 : Range_Known_0_Unknown := default
  let mut r6 : Range_Known_0_Unknown := default
  let mut out_2 := out_2
  -- let len = out.len()
  let len ← MemSlice_len_u64 out_2
  -- core::assert(a.len() >= len && b.len() >= len)
  let v11 ← MemSlice_len_u64 a
  let join : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → Bool → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 v3 => do
    let mut a_len := a_len
    let mut b_len := b_len
    let mut carry := carry
    let mut «at» := «at»
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut out_2 := out_2
    -- core::assert(a.len() >= len && b.len() >= len)
    let _ ← «assert» v3
    -- let ap = a.ptr()
    let ap ← MemSlice_ptr_u64 a
    -- let bp = b.ptr()
    let bp ← MemSlice_ptr_u64 b
    -- let op = out.ptr()
    let op ← MemSlice_ptr_u64 out_2
    -- for j in 0..len { unsafe { *limb(op, j) = 0 } }
    r4 := (Range_Known_0_Unknown.mk () len)
    let v31 ← Range_start_0 r4
    let rec loop (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) (v4 : UInt256) : FeM (Unit × MemSlice_u64) := do
      let mut a_len := a_len
      let mut b_len := b_len
      let mut carry := carry
      let mut «at» := «at»
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let mut out_2 := out_2
      -- for j in 0..len { unsafe { *limb(op, j) = 0 } }
      let v34 ← Range_has_0 r4 v4
      if v34 then
        let v36 ← Range_at_0 r4 v4
        -- *limb(op, j) = 0
        let v39 ← limb op v36
        mstoreInt (.u 64) v39 (UInt256.ofNat 0)
        -- for j in 0..len { unsafe { *limb(op, j) = 0 } }
        let v42 ← Range_next_0 r4 v4
        loop a_len b_len carry «at» r4 r5 r6 out_2 v42
      else
        -- var a_len = len
        a_len := len
        let rec loop_1 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
          let mut a_len := a_len
          let mut b_len := b_len
          let mut carry := carry
          let mut «at» := «at»
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut out_2 := out_2
          let join_1 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
            let mut a_len := a_len
            let mut b_len := b_len
            let mut carry := carry
            let mut «at» := «at»
            let mut r4 := r4
            let mut r5 := r5
            let mut r6 := r6
            let mut out_2 := out_2
            -- var b_len = len
            b_len := len
            let rec loop_2 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
              let mut a_len := a_len
              let mut b_len := b_len
              let mut carry := carry
              let mut «at» := «at»
              let mut r4 := r4
              let mut r5 := r5
              let mut r6 := r6
              let mut out_2 := out_2
              let join_2 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
                let mut a_len := a_len
                let mut b_len := b_len
                let mut carry := carry
                let mut «at» := «at»
                let mut r4 := r4
                let mut r5 := r5
                let mut r6 := r6
                let mut out_2 := out_2
                -- for i in 0..a_len {
                r5 := (Range_Known_0_Unknown.mk () a_len)
                let v85 ← Range_start_0 r5
                let rec loop_3 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) (v5 : UInt256) : FeM (Unit × MemSlice_u64) := do
                  let mut a_len := a_len
                  let mut b_len := b_len
                  let mut carry := carry
                  let mut «at» := «at»
                  let mut r4 := r4
                  let mut r5 := r5
                  let mut r6 := r6
                  let mut out_2 := out_2
                  -- for i in 0..a_len {
                  let v88 ← Range_has_0 r5 v5
                  if v88 then
                    let v90 ← Range_at_0 r5 v5
                    let i := v90
                    -- *limb(ap, i)
                    let v92 ← limb ap i
                    let factor ← mloadInt (.u 64) v92
                    -- var carry: u64 = 0
                    carry := (UInt256.ofNat 0)
                    let join_3 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → UInt256 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 «end» => do
                      let mut a_len := a_len
                      let mut b_len := b_len
                      let mut carry := carry
                      let mut «at» := «at»
                      let mut r4 := r4
                      let mut r5 := r5
                      let mut r6 := r6
                      let mut out_2 := out_2
                      -- for j in 0..end {
                      r6 := (Range_Known_0_Unknown.mk () «end»)
                      let v111 ← Range_start_0 r6
                      let rec loop_4 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) (v7 : UInt256) : FeM (Unit × MemSlice_u64) := do
                        let mut a_len := a_len
                        let mut b_len := b_len
                        let mut carry := carry
                        let mut «at» := «at»
                        let mut r4 := r4
                        let mut r5 := r5
                        let mut r6 := r6
                        let mut out_2 := out_2
                        -- for j in 0..end {
                        let v114 ← Range_has_0 r6 v7
                        if v114 then
                          let v116 ← Range_at_0 r6 v7
                          let j_117 := v116
                          -- *limb(op, i + j)
                          let v119 ← limb op (i + j_117)
                          let v120 ← mloadInt (.u 64) v119
                          -- *limb(bp, j)
                          let v122 ← limb bp j_117
                          let v123 ← mloadInt (.u 64) v122
                          -- let sum = unsafe { *limb(op, i + j) } as u128 + factor * unsafe { *limb(bp, j) } as u128 + carry as u128
                          let v126 := IntTy.addW (.u 128) v120 (IntTy.mulW (.u 128) factor v123)
                          let sum_129 := IntTy.addW (.u 128) v126 carry
                          -- *limb(op, i + j) = sum.downcast_truncate()
                          let v131 ← downcast_truncate_u128_u64 sum_129
                          let v133 ← limb op (i + j_117)
                          mstoreInt (.u 64) v133 v131
                          -- carry = (sum >> 64).downcast_truncate()
                          let carry_137 ← downcast_truncate_u128_u64 (sum_129 >>> (UInt256.ofNat 64))
                          carry := carry_137
                          -- for j in 0..end {
                          let v141 ← Range_next_0 r6 v7
                          loop_4 a_len b_len carry «at» r4 r5 r6 out_2 v141
                        else
                          -- var at = i + end
                          «at» := (i + «end»)
                          let rec loop_5 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
                            let mut a_len := a_len
                            let mut b_len := b_len
                            let mut carry := carry
                            let mut «at» := «at»
                            let mut r4 := r4
                            let mut r5 := r5
                            let mut r6 := r6
                            let mut out_2 := out_2
                            let join_4 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
                              let mut a_len := a_len
                              let mut b_len := b_len
                              let mut carry := carry
                              let mut «at» := «at»
                              let mut r4 := r4
                              let mut r5 := r5
                              let mut r6 := r6
                              let mut out_2 := out_2
                              -- for i in 0..a_len {
                              let v178 ← Range_next_0 r5 v5
                              loop_3 a_len b_len carry «at» r4 r5 r6 out_2 v178
                            -- while carry != 0 && at < len {
                            if decide (carry ≠ (UInt256.ofNat 0)) then
                              if decide («at».toNat < len.toNat) then
                                -- *limb(op, at)
                                let v154 ← limb op «at»
                                let v155 ← mloadInt (.u 64) v154
                                -- let sum = unsafe { *limb(op, at) } as u128 + carry as u128
                                let sum_159 := IntTy.addW (.u 128) v155 carry
                                -- *limb(op, at) = sum.downcast_truncate()
                                let v161 ← downcast_truncate_u128_u64 sum_159
                                let v164 ← limb op «at»
                                mstoreInt (.u 64) v164 v161
                                -- carry = (sum >> 64).downcast_truncate()
                                let carry_168 ← downcast_truncate_u128_u64 (sum_159 >>> (UInt256.ofNat 64))
                                carry := carry_168
                                -- at += 1
                                «at» := («at» + (UInt256.ofNat 1))
                                loop_5 a_len b_len carry «at» r4 r5 r6 out_2
                              else
                                join_4 a_len b_len carry «at» r4 r5 r6 out_2
                            else
                              join_4 a_len b_len carry «at» r4 r5 r6 out_2
                          partial_fixpoint
                          loop_5 a_len b_len carry «at» r4 r5 r6 out_2
                      partial_fixpoint
                      loop_4 a_len b_len carry «at» r4 r5 r6 out_2 v111
                    -- let end = if len - i < b_len { len - i } else { b_len }
                    if decide ((len - i).toNat < b_len.toNat) then
                      join_3 a_len b_len carry «at» r4 r5 r6 out_2 (len - i)
                    else
                      join_3 a_len b_len carry «at» r4 r5 r6 out_2 b_len
                  else
                    return ((), out_2)
                partial_fixpoint
                loop_3 a_len b_len carry «at» r4 r5 r6 out_2 v85
              -- while b_len > 0 && unsafe { *limb(bp, b_len - 1) } == 0 { b_len -= 1 }
              if decide ((UInt256.ofNat 0).toNat < b_len.toNat) then
                -- *limb(bp, b_len - 1)
                let v71 ← limb bp (b_len - (UInt256.ofNat 1))
                let v72 ← mloadInt (.u 64) v71
                -- while b_len > 0 && unsafe { *limb(bp, b_len - 1) } == 0 { b_len -= 1 }
                if decide (v72 = (UInt256.ofNat 0)) then
                  -- b_len -= 1
                  b_len := (b_len - (UInt256.ofNat 1))
                  loop_2 a_len b_len carry «at» r4 r5 r6 out_2
                else
                  join_2 a_len b_len carry «at» r4 r5 r6 out_2
              else
                join_2 a_len b_len carry «at» r4 r5 r6 out_2
            partial_fixpoint
            loop_2 a_len b_len carry «at» r4 r5 r6 out_2
          -- while a_len > 0 && unsafe { *limb(ap, a_len - 1) } == 0 { a_len -= 1 }
          if decide ((UInt256.ofNat 0).toNat < a_len.toNat) then
            -- *limb(ap, a_len - 1)
            let v53 ← limb ap (a_len - (UInt256.ofNat 1))
            let v54 ← mloadInt (.u 64) v53
            -- while a_len > 0 && unsafe { *limb(ap, a_len - 1) } == 0 { a_len -= 1 }
            if decide (v54 = (UInt256.ofNat 0)) then
              -- a_len -= 1
              a_len := (a_len - (UInt256.ofNat 1))
              loop_1 a_len b_len carry «at» r4 r5 r6 out_2
            else
              join_1 a_len b_len carry «at» r4 r5 r6 out_2
          else
            join_1 a_len b_len carry «at» r4 r5 r6 out_2
        partial_fixpoint
        loop_1 a_len b_len carry «at» r4 r5 r6 out_2
    partial_fixpoint
    loop a_len b_len carry «at» r4 r5 r6 out_2 v31
  let join_5 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
    let mut a_len := a_len
    let mut b_len := b_len
    let mut carry := carry
    let mut «at» := «at»
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut out_2 := out_2
    join a_len b_len carry «at» r4 r5 r6 out_2 false
  if decide (len.toNat ≤ v11.toNat) then
    let v13 ← MemSlice_len_u64 b
    if decide (len.toNat ≤ v13.toNat) then
      join a_len b_len carry «at» r4 r5 r6 out_2 true
    else
      join_5 a_len b_len carry «at» r4 r5 r6 out_2
  else
    join_5 a_len b_len carry «at» r4 r5 r6 out_2

/-- `lib::modexp::low_power` -/
def low_power (exponent : Exponent) (base : MemArray_u64) (k : UInt256) : FeM Option_MemArray_u64 := do
  let mut result : MemArray_u64 := default
  let mut power : UInt256 := default
  let mut index_r2 : UInt256 := default
  let mut factor : MemArray_u64 := default
  let mut work : MemArray_u64 := default
  let mut index_r5 : UInt256 := default
  let mut zero : Bool := default
  let mut r7 : Range_Known_0_Unknown := default
  let mut r8 : Range_Known_0_Unknown := default
  let mut v193 : MemSlice_u64 := default
  let mut v208 : MemSlice_u64 := default
  let mut v217 : MemSlice_u64 := default
  -- let low = (k + 63) / 64
  let low ← checkedDiv (k + (UInt256.ofNat 63)) (UInt256.ofNat 64)
  -- let bits: u64 = (k % 64).downcast_truncate()
  let v17 ← checkedRem k (UInt256.ofNat 64)
  let bits ← downcast_truncate_usize_u64 v17
  let join : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 mask => do
    let mut result := result
    let mut power := power
    let mut index_r2 := index_r2
    let mut factor := factor
    let mut work := work
    let mut index_r5 := index_r5
    let mut zero := zero
    let mut r7 := r7
    let mut r8 := r8
    let mut v193 := v193
    let mut v208 := v208
    let mut v217 := v217
    -- var result = MemArray<u64>::try_new_zeroed(low)?
    let v30 ← MemArray_try_new_zeroed_u64 low
    if match v30 with | .Some .. => true | _ => false then
      let result_32 := v30.Some_0
      result := result_32
      -- let start = if base[0] & 1 == 1 && exponent.total - exponent.top > k { exponent.total - k } else { exponent.top }
      let v35 ← MemArray_index_ref_u64 base (UInt256.ofNat 0)
      let v36 ← mloadInt (.u 64) v35
      let v40 := decide ((v36 &&& (UInt256.ofNat 1)) = (UInt256.ofNat 1))
      let join_1 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 start => do
        let mut result := result
        let mut power := power
        let mut index_r2 := index_r2
        let mut factor := factor
        let mut work := work
        let mut index_r5 := index_r5
        let mut zero := zero
        let mut r7 := r7
        let mut r8 := r8
        let mut v193 := v193
        let mut v208 := v208
        let mut v217 := v217
        -- if low <= 2 {
        if decide (low.toNat ≤ (UInt256.ofNat 2).toNat) then
          let join_2 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 wide_mask => do
            let mut result := result
            let mut power := power
            let mut index_r2 := index_r2
            let mut factor := factor
            let mut work := work
            let mut index_r5 := index_r5
            let mut zero := zero
            let mut r7 := r7
            let mut r8 := r8
            let mut v193 := v193
            let mut v208 := v208
            let mut v217 := v217
            -- let factor = (base[0] as u128 | ((base[1] as u128) << 64)) & wide_mask
            let v67 ← MemArray_index_ref_u64 base (UInt256.ofNat 0)
            let v68 ← mloadInt (.u 64) v67
            let v71 ← MemArray_index_ref_u64 base (UInt256.ofNat 1)
            let v72 ← mloadInt (.u 64) v71
            let v76 := v68 ||| (IntTy.shl (.u 128) v72 (UInt256.ofNat 64))
            let factor_77 := v76 &&& wide_mask
            -- var power: u128 = 1
            power := (UInt256.ofNat 1)
            -- var index = start
            index_r2 := start
            let rec loop (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) : FeM Option_MemArray_u64 := do
              let mut result := result
              let mut power := power
              let mut index_r2 := index_r2
              let mut factor := factor
              let mut work := work
              let mut index_r5 := index_r5
              let mut zero := zero
              let mut r7 := r7
              let mut r8 := r8
              let mut v193 := v193
              let mut v208 := v208
              let mut v217 := v217
              let join_3 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
                let mut result := result
                let mut power := power
                let mut index_r2 := index_r2
                let mut factor := factor
                let mut work := work
                let mut index_r5 := index_r5
                let mut zero := zero
                let mut r7 := r7
                let mut r8 := r8
                let mut v193 := v193
                let mut v208 := v208
                let mut v217 := v217
                -- result[0] = power.downcast_truncate()
                let v120 ← downcast_truncate_u128_u64 power
                let (v123, v123_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 0)
                mstoreInt (.u 64) v123 v120
                let v123_0 ← v123_k v123
                result := v123_0
                let join_4 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
                  let mut result := result
                  let mut power := power
                  let mut index_r2 := index_r2
                  let mut factor := factor
                  let mut work := work
                  let mut index_r5 := index_r5
                  let mut zero := zero
                  let mut r7 := r7
                  let mut r8 := r8
                  let mut v193 := v193
                  let mut v208 := v208
                  let mut v217 := v217
                  -- return Option::Some(result)
                  let result_139 := result
                  return Option_MemArray_u64.Some result_139
                -- if low == 2 { result[1] = (power >> 64).downcast_truncate() }
                if decide (low = (UInt256.ofNat 2)) then
                  -- result[1] = (power >> 64).downcast_truncate()
                  let v131 ← downcast_truncate_u128_u64 (power >>> (UInt256.ofNat 64))
                  let (v134, v134_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 1)
                  mstoreInt (.u 64) v134 v131
                  let v134_0 ← v134_k v134
                  result := v134_0
                  join_4 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                else
                  join_4 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
              -- while index < exponent.total && power != 0 {
              if decide (index_r2.toNat < exponent.total.toNat) then
                if decide (power ≠ (UInt256.ofNat 0)) then
                  -- power = power * power
                  power := (IntTy.mulW (.u 128) power power)
                  -- if exponent.bit(index) == 1 { power = power * factor }
                  let v98 ← Exponent_bit exponent index_r2
                  let join_5 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
                    let mut result := result
                    let mut power := power
                    let mut index_r2 := index_r2
                    let mut factor := factor
                    let mut work := work
                    let mut index_r5 := index_r5
                    let mut zero := zero
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut v193 := v193
                    let mut v208 := v208
                    let mut v217 := v217
                    -- power &= wide_mask
                    power := (power &&& wide_mask)
                    -- index += 1
                    index_r2 := (index_r2 + (UInt256.ofNat 1))
                    loop result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                  if decide (v98 = (UInt256.ofNat 1)) then
                    -- power = power * factor
                    power := (IntTy.mulW (.u 128) power factor_77)
                    join_5 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                  else
                    join_5 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                else
                  join_3 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
              else
                join_3 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
            partial_fixpoint
            loop result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
          -- let wide_mask: u128 = if low == 1 { mask as u128 } else { ((mask as u128) << 64) | 0xffffffffffffffff }
          if decide (low = (UInt256.ofNat 1)) then
            join_2 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 mask
          else
            -- ((mask as u128) << 64) | 0xffffffffffffffff
            let v63 := (IntTy.shl (.u 128) mask (UInt256.ofNat 64)) ||| (UInt256.ofNat 18446744073709551615)
            join_2 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v63
        else
          -- var factor = MemArray<u64>::try_new_zeroed(low)?
          let v144 ← MemArray_try_new_zeroed_u64 low
          if match v144 with | .Some .. => true | _ => false then
            let factor_146 := v144.Some_0
            factor := factor_146
            -- var work = MemArray<u64>::try_new_zeroed(low)?
            let v148 ← MemArray_try_new_zeroed_u64 low
            if match v148 with | .Some .. => true | _ => false then
              let work_150 := v148.Some_0
              work := work_150
              -- for i in 0..low { factor[i] = base[i] }
              r7 := (Range_Known_0_Unknown.mk () low)
              let v156 ← Range_start_0 r7
              let rec loop_1 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) (v6 : UInt256) : FeM Option_MemArray_u64 := do
                let mut result := result
                let mut power := power
                let mut index_r2 := index_r2
                let mut factor := factor
                let mut work := work
                let mut index_r5 := index_r5
                let mut zero := zero
                let mut r7 := r7
                let mut r8 := r8
                let mut v193 := v193
                let mut v208 := v208
                let mut v217 := v217
                -- for i in 0..low { factor[i] = base[i] }
                let v159 ← Range_has_0 r7 v6
                if v159 then
                  let v161 ← Range_at_0 r7 v6
                  let i_162 := v161
                  -- factor[i] = base[i]
                  let v163 ← MemArray_index_ref_u64 base i_162
                  let v164 ← mloadInt (.u 64) v163
                  let (v166, v166_k) ← MemArray_index_mut_u64 factor i_162
                  mstoreInt (.u 64) v166 v164
                  let v166_0 ← v166_k v166
                  factor := v166_0
                  -- for i in 0..low { factor[i] = base[i] }
                  let v169 ← Range_next_0 r7 v6
                  loop_1 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v169
                else
                  -- factor[low - 1] &= mask
                  let (v174, v174_k) ← MemArray_index_mut_u64 factor (low - (UInt256.ofNat 1))
                  let v175 ← mloadInt (.u 64) v174
                  mstoreInt (.u 64) v174 (v175 &&& mask)
                  let v174_0 ← v174_k v174
                  factor := v174_0
                  -- result[0] = 1
                  let (v181, v181_k) ← MemArray_index_mut_u64 result (UInt256.ofNat 0)
                  mstoreInt (.u 64) v181 (UInt256.ofNat 1)
                  let v181_0 ← v181_k v181
                  result := v181_0
                  -- var index = start
                  index_r5 := start
                  let rec loop_2 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) : FeM Option_MemArray_u64 := do
                    let mut result := result
                    let mut power := power
                    let mut index_r2 := index_r2
                    let mut factor := factor
                    let mut work := work
                    let mut index_r5 := index_r5
                    let mut zero := zero
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut v193 := v193
                    let mut v208 := v208
                    let mut v217 := v217
                    let join_6 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
                      let mut result := result
                      let mut power := power
                      let mut index_r2 := index_r2
                      let mut factor := factor
                      let mut work := work
                      let mut index_r5 := index_r5
                      let mut zero := zero
                      let mut r7 := r7
                      let mut r8 := r8
                      let mut v193 := v193
                      let mut v208 := v208
                      let mut v217 := v217
                      -- Option::Some(result)
                      let result_264 := result
                      let _ ← MemArray_drop_u64 work
                      let _ ← MemArray_drop_u64 factor
                      return Option_MemArray_u64.Some result_264
                    -- while index < exponent.total {
                    if decide (index_r5.toNat < exponent.total.toNat) then
                      -- low_product(result.as_slice(), result.as_slice(), mut work.as_slice())
                      let v189 ← MemArray_as_slice_ref_u64 result
                      let v191 ← MemArray_as_slice_ref_u64 result
                      let (v193_g, v193_k) ← MemArray_as_slice_mut_u64 work
                      v193 := v193_g
                      let (v197, v197_0) ← low_product v189 v191 v193
                      v193 := v197_0
                      let v193_0 ← v193_k v193
                      work := v193_0
                      -- if exponent.bit(index) == 1 {
                      let v200 ← Exponent_bit exponent index_r5
                      let join_7 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
                        let mut result := result
                        let mut power := power
                        let mut index_r2 := index_r2
                        let mut factor := factor
                        let mut work := work
                        let mut index_r5 := index_r5
                        let mut zero := zero
                        let mut r7 := r7
                        let mut r8 := r8
                        let mut v193 := v193
                        let mut v208 := v208
                        let mut v217 := v217
                        -- result[low - 1] &= mask
                        let (v224, v224_k) ← MemArray_index_mut_u64 result (low - (UInt256.ofNat 1))
                        let v225 ← mloadInt (.u 64) v224
                        mstoreInt (.u 64) v224 (v225 &&& mask)
                        let v224_0 ← v224_k v224
                        result := v224_0
                        -- var zero = true
                        zero := true
                        -- for i in 0..low { zero = zero && result[i] == 0 }
                        r8 := (Range_Known_0_Unknown.mk () low)
                        let v234 ← Range_start_0 r8
                        let rec loop_3 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) (v7 : UInt256) : FeM Option_MemArray_u64 := do
                          let mut result := result
                          let mut power := power
                          let mut index_r2 := index_r2
                          let mut factor := factor
                          let mut work := work
                          let mut index_r5 := index_r5
                          let mut zero := zero
                          let mut r7 := r7
                          let mut r8 := r8
                          let mut v193 := v193
                          let mut v208 := v208
                          let mut v217 := v217
                          -- for i in 0..low { zero = zero && result[i] == 0 }
                          let v237 ← Range_has_0 r8 v7
                          if v237 then
                            let v239 ← Range_at_0 r8 v7
                            let i_240 := v239
                            let join_8 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → Bool → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 zero_8 => do
                              let mut result := result
                              let mut power := power
                              let mut index_r2 := index_r2
                              let mut factor := factor
                              let mut work := work
                              let mut index_r5 := index_r5
                              let mut zero := zero
                              let mut r7 := r7
                              let mut r8 := r8
                              let mut v193 := v193
                              let mut v208 := v208
                              let mut v217 := v217
                              -- zero = zero && result[i] == 0
                              zero := zero_8
                              -- for i in 0..low { zero = zero && result[i] == 0 }
                              let v254 ← Range_next_0 r8 v7
                              loop_3 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v254
                            let join_9 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
                              let mut result := result
                              let mut power := power
                              let mut index_r2 := index_r2
                              let mut factor := factor
                              let mut work := work
                              let mut index_r5 := index_r5
                              let mut zero := zero
                              let mut r7 := r7
                              let mut r8 := r8
                              let mut v193 := v193
                              let mut v208 := v208
                              let mut v217 := v217
                              join_8 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 false
                            -- zero = zero && result[i] == 0
                            if zero then
                              let v243 ← MemArray_index_ref_u64 result i_240
                              let v244 ← mloadInt (.u 64) v243
                              if decide (v244 = (UInt256.ofNat 0)) then
                                join_8 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 true
                              else
                                join_9 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                            else
                              join_9 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                          else
                            -- if zero { break }
                            if zero then
                              join_6 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                            else
                              -- index += 1
                              index_r5 := (index_r5 + (UInt256.ofNat 1))
                              loop_2 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                        partial_fixpoint
                        loop_3 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v234
                      if decide (v200 = (UInt256.ofNat 1)) then
                        -- low_product(work.as_slice(), factor.as_slice(), mut result.as_slice())
                        let v204 ← MemArray_as_slice_ref_u64 work
                        let v206 ← MemArray_as_slice_ref_u64 factor
                        let (v208_g, v208_k) ← MemArray_as_slice_mut_u64 result
                        v208 := v208_g
                        let (v212, v212_0) ← low_product v204 v206 v208
                        v208 := v212_0
                        let v208_0 ← v208_k v208
                        result := v208_0
                        join_7 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                      else
                        -- result.as_slice().copy_from(work.as_slice())
                        let v215 ← MemArray_as_slice_ref_u64 work
                        let (v217_g, v217_k) ← MemArray_as_slice_mut_u64 result
                        v217 := v217_g
                        let (v219, v219_0) ← MemSlice_copy_from_u64 v217 v215
                        v217 := v219_0
                        let v217_0 ← v217_k v217
                        result := v217_0
                        join_7 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                    else
                      join_6 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                  partial_fixpoint
                  loop_2 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
              partial_fixpoint
              loop_1 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v156
            else
              match (← pure v148) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v148.Some_0
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 factor
              let _ ← MemArray_drop_u64 result
              return Option_MemArray_u64.None
          else
            match (← pure v144) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v144.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 result
            return Option_MemArray_u64.None
      let join_10 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
        let mut result := result
        let mut power := power
        let mut index_r2 := index_r2
        let mut factor := factor
        let mut work := work
        let mut index_r5 := index_r5
        let mut zero := zero
        let mut r7 := r7
        let mut r8 := r8
        let mut v193 := v193
        let mut v208 := v208
        let mut v217 := v217
        join_1 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 exponent.top
      if v40 then
        let v45 := decide (k.toNat < (exponent.total - exponent.top).toNat)
        if v45 then
          join_1 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 (exponent.total - k)
        else
          join_10 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
      else
        join_10 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
    else
      match (← pure v30) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v30.Some_0
        pure ()
      | _ => pure ()
      return Option_MemArray_u64.None
  -- let mask: u64 = if bits == 0 { 0xffffffffffffffff } else { (1 << bits) - 1 }
  if decide (bits = (UInt256.ofNat 0)) then
    join result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 (UInt256.ofNat 18446744073709551615)
  else
    -- (1 << bits) - 1
    let v27 := IntTy.subW (.u 64) (IntTy.shl (.u 64) (UInt256.ofNat 1) bits) (UInt256.ofNat 1)
    join result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v27

/-- `lib::modexp::even_power` -/
def even_power (exponent : Exponent) (base : MemArray_u64) (base_size : UInt256) (value : MemArray_u64) (size : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
  let mut zero_limbs : UInt256 := default
  let mut result : MemArray_u64 := default
  let mut y : MemArray_u64 := default
  let mut q : MemArray_u64 := default
  let mut q_len : UInt256 := default
  let mut x : MemArray_u64 := default
  let mut t : MemArray_u64 := default
  let mut borrow : UInt256 := default
  let mut inverse : MemArray_u64 := default
  let mut guess : UInt256 := default
  let mut scratch : MemArray_u64 := default
  let mut correction : MemArray_u64 := default
  let mut correct : UInt256 := default
  let mut carry_r13 : UInt256 := default
  let mut add : UInt256 := default
  let mut h : MemArray_u64 := default
  let mut work : MemArray_u64 := default
  let mut carry_r17 : UInt256 := default
  let mut «at» : UInt256 := default
  let mut r19 : Range_Known_0_Unknown := default
  let mut r20 : Range_Known_0_Unknown := default
  let mut r21 : Range_Known_0_Unknown := default
  let mut r22 : Range_Known_0_Known_5 := default
  let mut r23 : Range_Known_0_Unknown := default
  let mut r24 : Range_Known_0_Unknown := default
  let mut r25 : Range_Known_0_Unknown := default
  let mut r26 : Range_Known_0_Unknown := default
  let mut r27 : Range_Known_0_Unknown := default
  let mut r28 : Range_Known_0_Unknown := default
  let mut base := base
  let mut v320 : MemSlice_u64 := default
  let mut v321 : MemSlice_u64 := default
  let mut v398 : MemSlice_u64 := default
  let mut v399 : MemSlice_u64 := default
  let mut v409 : MemSlice_u64 := default
  let mut v410 : MemSlice_u64 := default
  let mut v424 : MemSlice_u64 := default
  -- var zero_limbs: usize = 0
  zero_limbs := (UInt256.ofNat 0)
  let rec loop (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
    let mut zero_limbs := zero_limbs
    let mut result := result
    let mut y := y
    let mut q := q
    let mut q_len := q_len
    let mut x := x
    let mut t := t
    let mut borrow := borrow
    let mut inverse := inverse
    let mut guess := guess
    let mut scratch := scratch
    let mut correction := correction
    let mut correct := correct
    let mut carry_r13 := carry_r13
    let mut add := add
    let mut h := h
    let mut work := work
    let mut carry_r17 := carry_r17
    let mut «at» := «at»
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
    let mut base := base
    let mut v320 := v320
    let mut v321 := v321
    let mut v398 := v398
    let mut v399 := v399
    let mut v409 := v409
    let mut v410 := v410
    let mut v424 := v424
    -- while value[zero_limbs] == 0 { zero_limbs += 1 }
    let v24 ← MemArray_index_ref_u64 value zero_limbs
    let v25 ← mloadInt (.u 64) v24
    if decide (v25 = (UInt256.ofNat 0)) then
      -- zero_limbs += 1
      zero_limbs := (zero_limbs + (UInt256.ofNat 1))
      loop zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
    else
      -- let k = zero_limbs * 64 + trailing_zeros(value[zero_limbs]).downcast_truncate()
      let v36 := zero_limbs * (UInt256.ofNat 64)
      let v39 ← MemArray_index_ref_u64 value zero_limbs
      let v40 ← mloadInt (.u 64) v39
      let v42 ← downcast_truncate_u64_usize (IntTy.trailingZeros (.u 64) v40)
      let k := v36 + v42
      -- let low = (k + 63) / 64
      let low ← checkedDiv (k + (UInt256.ofNat 63)) (UInt256.ofNat 64)
      -- let bits: u64 = (k % 64).downcast_truncate()
      let v51 ← checkedRem k (UInt256.ofNat 64)
      let bits ← downcast_truncate_usize_u64 v51
      let join : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 mask => do
        let mut zero_limbs := zero_limbs
        let mut result := result
        let mut y := y
        let mut q := q
        let mut q_len := q_len
        let mut x := x
        let mut t := t
        let mut borrow := borrow
        let mut inverse := inverse
        let mut guess := guess
        let mut scratch := scratch
        let mut correction := correction
        let mut correct := correct
        let mut carry_r13 := carry_r13
        let mut add := add
        let mut h := h
        let mut work := work
        let mut carry_r17 := carry_r17
        let mut «at» := «at»
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
        let mut base := base
        let mut v320 := v320
        let mut v321 := v321
        let mut v398 := v398
        let mut v399 := v399
        let mut v409 := v409
        let mut v410 := v410
        let mut v424 := v424
        -- var result = MemArray<u64>::try_new_zeroed(size)?
        let v64 ← MemArray_try_new_zeroed_u64 size
        if match v64 with | .Some .. => true | _ => false then
          let result_66 := v64.Some_0
          result := result_66
          -- let y = low_power(exponent, base, k)?
          let v70 ← low_power exponent base k
          if match v70 with | .Some .. => true | _ => false then
            let y_72 := v70.Some_0
            y := y_72
            -- var q = MemArray<u64>::try_new_zeroed(size)?
            let v74 ← MemArray_try_new_zeroed_u64 size
            if match v74 with | .Some .. => true | _ => false then
              let q_76 := v74.Some_0
              q := q_76
              -- for i in 0..size - k / 64 {
              let v81 ← checkedDiv k (UInt256.ofNat 64)
              r19 := (Range_Known_0_Unknown.mk () (size - v81))
              let v86 ← Range_start_0 r19
              let rec loop_1 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v6 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                let mut zero_limbs := zero_limbs
                let mut result := result
                let mut y := y
                let mut q := q
                let mut q_len := q_len
                let mut x := x
                let mut t := t
                let mut borrow := borrow
                let mut inverse := inverse
                let mut guess := guess
                let mut scratch := scratch
                let mut correction := correction
                let mut correct := correct
                let mut carry_r13 := carry_r13
                let mut add := add
                let mut h := h
                let mut work := work
                let mut carry_r17 := carry_r17
                let mut «at» := «at»
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
                let mut base := base
                let mut v320 := v320
                let mut v321 := v321
                let mut v398 := v398
                let mut v399 := v399
                let mut v409 := v409
                let mut v410 := v410
                let mut v424 := v424
                -- for i in 0..size - k / 64 {
                let v89 ← Range_has_0 r19 v6
                if v89 then
                  let v91 ← Range_at_0 r19 v6
                  let i_92 := v91
                  -- let next = if i + k / 64 + 1 < size { value[i + k / 64 + 1] } else { 0 }
                  let v94 ← checkedDiv k (UInt256.ofNat 64)
                  let v99 := decide (((i_92 + v94) + (UInt256.ofNat 1)).toNat < size.toNat)
                  let join_1 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 next_7 => do
                    let mut zero_limbs := zero_limbs
                    let mut result := result
                    let mut y := y
                    let mut q := q
                    let mut q_len := q_len
                    let mut x := x
                    let mut t := t
                    let mut borrow := borrow
                    let mut inverse := inverse
                    let mut guess := guess
                    let mut scratch := scratch
                    let mut correction := correction
                    let mut correct := correct
                    let mut carry_r13 := carry_r13
                    let mut add := add
                    let mut h := h
                    let mut work := work
                    let mut carry_r17 := carry_r17
                    let mut «at» := «at»
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
                    let mut base := base
                    let mut v320 := v320
                    let mut v321 := v321
                    let mut v398 := v398
                    let mut v399 := v399
                    let mut v409 := v409
                    let mut v410 := v410
                    let mut v424 := v424
                    let join_2 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v8 => do
                      let mut zero_limbs := zero_limbs
                      let mut result := result
                      let mut y := y
                      let mut q := q
                      let mut q_len := q_len
                      let mut x := x
                      let mut t := t
                      let mut borrow := borrow
                      let mut inverse := inverse
                      let mut guess := guess
                      let mut scratch := scratch
                      let mut correction := correction
                      let mut correct := correct
                      let mut carry_r13 := carry_r13
                      let mut add := add
                      let mut h := h
                      let mut work := work
                      let mut carry_r17 := carry_r17
                      let mut «at» := «at»
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
                      let mut base := base
                      let mut v320 := v320
                      let mut v321 := v321
                      let mut v398 := v398
                      let mut v399 := v399
                      let mut v409 := v409
                      let mut v410 := v410
                      let mut v424 := v424
                      -- q[i] = if bits == 0 { value[i + k / 64] } else { (value[i + k / 64] >> bits) | (next << (64 - bits)) }
                      let (v131, v131_k) ← MemArray_index_mut_u64 q i_92
                      mstoreInt (.u 64) v131 v8
                      let v131_0 ← v131_k v131
                      q := v131_0
                      -- for i in 0..size - k / 64 {
                      let v134 ← Range_next_0 r19 v6
                      loop_1 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v134
                    -- q[i] = if bits == 0 { value[i + k / 64] } else { (value[i + k / 64] >> bits) | (next << (64 - bits)) }
                    if decide (bits = (UInt256.ofNat 0)) then
                      -- value[i + k / 64]
                      let v114 ← checkedDiv k (UInt256.ofNat 64)
                      let v116 ← MemArray_index_ref_u64 value (i_92 + v114)
                      -- q[i] = if bits == 0 { value[i + k / 64] } else { (value[i + k / 64] >> bits) | (next << (64 - bits)) }
                      let v117 ← mloadInt (.u 64) v116
                      join_2 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v117
                    else
                      -- (value[i + k / 64] >> bits) | (next << (64 - bits))
                      let v120 ← checkedDiv k (UInt256.ofNat 64)
                      let v122 ← MemArray_index_ref_u64 value (i_92 + v120)
                      let v123 ← mloadInt (.u 64) v122
                      let v127 := IntTy.shl (.u 64) next_7 (IntTy.subW (.u 64) (UInt256.ofNat 64) bits)
                      join_2 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 ((v123 >>> bits) ||| v127)
                  if v99 then
                    -- value[i + k / 64 + 1]
                    let v101 ← checkedDiv k (UInt256.ofNat 64)
                    let v105 ← MemArray_index_ref_u64 value ((i_92 + v101) + (UInt256.ofNat 1))
                    -- let next = if i + k / 64 + 1 < size { value[i + k / 64 + 1] } else { 0 }
                    let v106 ← mloadInt (.u 64) v105
                    join_1 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v106
                  else
                    join_1 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 (UInt256.ofNat 0)
                else
                  -- var q_len = size
                  q_len := size
                  let rec loop_2 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                    let mut zero_limbs := zero_limbs
                    let mut result := result
                    let mut y := y
                    let mut q := q
                    let mut q_len := q_len
                    let mut x := x
                    let mut t := t
                    let mut borrow := borrow
                    let mut inverse := inverse
                    let mut guess := guess
                    let mut scratch := scratch
                    let mut correction := correction
                    let mut correct := correct
                    let mut carry_r13 := carry_r13
                    let mut add := add
                    let mut h := h
                    let mut work := work
                    let mut carry_r17 := carry_r17
                    let mut «at» := «at»
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
                    let mut base := base
                    let mut v320 := v320
                    let mut v321 := v321
                    let mut v398 := v398
                    let mut v399 := v399
                    let mut v409 := v409
                    let mut v410 := v410
                    let mut v424 := v424
                    -- while q[q_len - 1] == 0 { q_len -= 1 }
                    let v141 := q_len - (UInt256.ofNat 1)
                    let v143 ← MemArray_index_ref_u64 q v141
                    let v144 ← mloadInt (.u 64) v143
                    if decide (v144 = (UInt256.ofNat 0)) then
                      -- q_len -= 1
                      q_len := (q_len - (UInt256.ofNat 1))
                      loop_2 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                    else
                      let join_3 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 => do
                        let mut zero_limbs := zero_limbs
                        let mut result := result
                        let mut y := y
                        let mut q := q
                        let mut q_len := q_len
                        let mut x := x
                        let mut t := t
                        let mut borrow := borrow
                        let mut inverse := inverse
                        let mut guess := guess
                        let mut scratch := scratch
                        let mut correction := correction
                        let mut correct := correct
                        let mut carry_r13 := carry_r13
                        let mut add := add
                        let mut h := h
                        let mut work := work
                        let mut carry_r17 := carry_r17
                        let mut «at» := «at»
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
                        let mut base := base
                        let mut v320 := v320
                        let mut v321 := v321
                        let mut v398 := v398
                        let mut v399 := v399
                        let mut v409 := v409
                        let mut v410 := v410
                        let mut v424 := v424
                        -- let x = odd_power(exponent, mut base, base_size, q, q_len)?
                        let q_len_189 := q_len
                        let (v192, v192_0) ← odd_power exponent base base_size q q_len_189
                        base := v192_0
                        if match v192 with | .Some .. => true | _ => false then
                          let x_194 := v192.Some_0
                          x := x_194
                          -- var t = MemArray<u64>::try_new_zeroed(low)?
                          let v196 ← MemArray_try_new_zeroed_u64 low
                          if match v196 with | .Some .. => true | _ => false then
                            let t_198 := v196.Some_0
                            t := t_198
                            -- var borrow: u64 = 0
                            borrow := (UInt256.ofNat 0)
                            -- for i in 0..low {
                            r21 := (Range_Known_0_Unknown.mk () low)
                            let v206 ← Range_start_0 r21
                            let rec loop_3 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v10 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                              let mut zero_limbs := zero_limbs
                              let mut result := result
                              let mut y := y
                              let mut q := q
                              let mut q_len := q_len
                              let mut x := x
                              let mut t := t
                              let mut borrow := borrow
                              let mut inverse := inverse
                              let mut guess := guess
                              let mut scratch := scratch
                              let mut correction := correction
                              let mut correct := correct
                              let mut carry_r13 := carry_r13
                              let mut add := add
                              let mut h := h
                              let mut work := work
                              let mut carry_r17 := carry_r17
                              let mut «at» := «at»
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
                              let mut base := base
                              let mut v320 := v320
                              let mut v321 := v321
                              let mut v398 := v398
                              let mut v399 := v399
                              let mut v409 := v409
                              let mut v410 := v410
                              let mut v424 := v424
                              -- for i in 0..low {
                              let v209 ← Range_has_0 r21 v10
                              if v209 then
                                let v211 ← Range_at_0 r21 v10
                                let i_212 := v211
                                let join_4 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 x_limb => do
                                  let mut zero_limbs := zero_limbs
                                  let mut result := result
                                  let mut y := y
                                  let mut q := q
                                  let mut q_len := q_len
                                  let mut x := x
                                  let mut t := t
                                  let mut borrow := borrow
                                  let mut inverse := inverse
                                  let mut guess := guess
                                  let mut scratch := scratch
                                  let mut correction := correction
                                  let mut correct := correct
                                  let mut carry_r13 := carry_r13
                                  let mut add := add
                                  let mut h := h
                                  let mut work := work
                                  let mut carry_r17 := carry_r17
                                  let mut «at» := «at»
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
                                  let mut base := base
                                  let mut v320 := v320
                                  let mut v321 := v321
                                  let mut v398 := v398
                                  let mut v399 := v399
                                  let mut v409 := v409
                                  let mut v410 := v410
                                  let mut v424 := v424
                                  -- let difference = y[i] as u128 - x_limb as u128 - borrow as u128
                                  let v223 ← MemArray_index_ref_u64 y i_212
                                  let v224 ← mloadInt (.u 64) v223
                                  let difference := IntTy.subW (.u 128) (IntTy.subW (.u 128) v224 x_limb) borrow
                                  -- t[i] = difference.downcast_truncate()
                                  let v232 ← downcast_truncate_u128_u64 difference
                                  let (v234, v234_k) ← MemArray_index_mut_u64 t i_212
                                  mstoreInt (.u 64) v234 v232
                                  let v234_0 ← v234_k v234
                                  t := v234_0
                                  -- borrow = (difference >> 127).downcast_truncate()
                                  let borrow_238 ← downcast_truncate_u128_u64 (difference >>> (UInt256.ofNat 127))
                                  borrow := borrow_238
                                  -- for i in 0..low {
                                  let v242 ← Range_next_0 r21 v10
                                  loop_3 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v242
                                -- let x_limb = if i < q_len { x[i] } else { 0 }
                                if decide (i_212.toNat < q_len.toNat) then
                                  -- x[i]
                                  let v216 ← MemArray_index_ref_u64 x i_212
                                  -- let x_limb = if i < q_len { x[i] } else { 0 }
                                  let v217 ← mloadInt (.u 64) v216
                                  join_4 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v217
                                else
                                  join_4 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 (UInt256.ofNat 0)
                              else
                                -- t[low - 1] &= mask
                                let (v247, v247_k) ← MemArray_index_mut_u64 t (low - (UInt256.ofNat 1))
                                let v248 ← mloadInt (.u 64) v247
                                mstoreInt (.u 64) v247 (v248 &&& mask)
                                let v247_0 ← v247_k v247
                                t := v247_0
                                -- var inverse = MemArray<u64>::try_new_zeroed(low)?
                                let v251 ← MemArray_try_new_zeroed_u64 low
                                if match v251 with | .Some .. => true | _ => false then
                                  let inverse_253 := v251.Some_0
                                  inverse := inverse_253
                                  -- var guess = q[0]
                                  let v257 ← MemArray_index_ref_u64 q (UInt256.ofNat 0)
                                  let guess_258 ← mloadInt (.u 64) v257
                                  guess := guess_258
                                  -- for _ in 0..5 { guess = guess * (2 - q[0] * guess) }
                                  r22 := (Range_Known_0_Known_5.mk () ())
                                  let v264 ← Range_start_0_5 r22
                                  let rec loop_4 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v12 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                    let mut zero_limbs := zero_limbs
                                    let mut result := result
                                    let mut y := y
                                    let mut q := q
                                    let mut q_len := q_len
                                    let mut x := x
                                    let mut t := t
                                    let mut borrow := borrow
                                    let mut inverse := inverse
                                    let mut guess := guess
                                    let mut scratch := scratch
                                    let mut correction := correction
                                    let mut correct := correct
                                    let mut carry_r13 := carry_r13
                                    let mut add := add
                                    let mut h := h
                                    let mut work := work
                                    let mut carry_r17 := carry_r17
                                    let mut «at» := «at»
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
                                    let mut base := base
                                    let mut v320 := v320
                                    let mut v321 := v321
                                    let mut v398 := v398
                                    let mut v399 := v399
                                    let mut v409 := v409
                                    let mut v410 := v410
                                    let mut v424 := v424
                                    -- for _ in 0..5 { guess = guess * (2 - q[0] * guess) }
                                    let v267 ← Range_has_0_5 r22 v12
                                    if v267 then
                                      let _ ← Range_at_0_5 r22 v12
                                      -- guess = guess * (2 - q[0] * guess)
                                      let guess_270 := guess
                                      let v275 ← MemArray_index_ref_u64 q (UInt256.ofNat 0)
                                      let v276 ← mloadInt (.u 64) v275
                                      let v279 := IntTy.subW (.u 64) (UInt256.ofNat 2) (IntTy.mulW (.u 64) v276 guess)
                                      guess := (IntTy.mulW (.u 64) guess_270 v279)
                                      -- for _ in 0..5 { guess = guess * (2 - q[0] * guess) }
                                      let v284 ← Range_next_0_5 r22 v12
                                      loop_4 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v284
                                    else
                                      -- inverse[0] = guess
                                      let guess_286 := guess
                                      let (v290, v290_k) ← MemArray_index_mut_u64 inverse (UInt256.ofNat 0)
                                      mstoreInt (.u 64) v290 guess_286
                                      let v290_0 ← v290_k v290
                                      inverse := v290_0
                                      -- var scratch = MemArray<u64>::try_new_zeroed(low)?
                                      let v292 ← MemArray_try_new_zeroed_u64 low
                                      if match v292 with | .Some .. => true | _ => false then
                                        let scratch_294 := v292.Some_0
                                        scratch := scratch_294
                                        -- var correction = MemArray<u64>::try_new_zeroed(low)?
                                        let v296 ← MemArray_try_new_zeroed_u64 low
                                        if match v296 with | .Some .. => true | _ => false then
                                          let correction_298 := v296.Some_0
                                          correction := correction_298
                                          -- var correct: usize = 1
                                          correct := (UInt256.ofNat 1)
                                          let rec loop_5 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                            let mut zero_limbs := zero_limbs
                                            let mut result := result
                                            let mut y := y
                                            let mut q := q
                                            let mut q_len := q_len
                                            let mut x := x
                                            let mut t := t
                                            let mut borrow := borrow
                                            let mut inverse := inverse
                                            let mut guess := guess
                                            let mut scratch := scratch
                                            let mut correction := correction
                                            let mut correct := correct
                                            let mut carry_r13 := carry_r13
                                            let mut add := add
                                            let mut h := h
                                            let mut work := work
                                            let mut carry_r17 := carry_r17
                                            let mut «at» := «at»
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
                                            let mut base := base
                                            let mut v320 := v320
                                            let mut v321 := v321
                                            let mut v398 := v398
                                            let mut v399 := v399
                                            let mut v409 := v409
                                            let mut v410 := v410
                                            let mut v424 := v424
                                            -- while correct < low {
                                            if decide (correct.toNat < low.toNat) then
                                              -- let next = if 2 * correct < low { 2 * correct } else { low }
                                              let v308 := decide (((UInt256.ofNat 2) * correct).toNat < low.toNat)
                                              let join_5 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 next_13 => do
                                                let mut zero_limbs := zero_limbs
                                                let mut result := result
                                                let mut y := y
                                                let mut q := q
                                                let mut q_len := q_len
                                                let mut x := x
                                                let mut t := t
                                                let mut borrow := borrow
                                                let mut inverse := inverse
                                                let mut guess := guess
                                                let mut scratch := scratch
                                                let mut correction := correction
                                                let mut correct := correct
                                                let mut carry_r13 := carry_r13
                                                let mut add := add
                                                let mut h := h
                                                let mut work := work
                                                let mut carry_r17 := carry_r17
                                                let mut «at» := «at»
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
                                                let mut base := base
                                                let mut v320 := v320
                                                let mut v321 := v321
                                                let mut v398 := v398
                                                let mut v399 := v399
                                                let mut v409 := v409
                                                let mut v410 := v410
                                                let mut v424 := v424
                                                -- low_product(q.as_slice(), inverse.as_slice(), mut scratch.as_slice().head(next))
                                                let v316 ← MemArray_as_slice_ref_u64 q
                                                let v318 ← MemArray_as_slice_ref_u64 inverse
                                                let (v320_g, v320_k) ← MemArray_as_slice_mut_u64 scratch
                                                v320 := v320_g
                                                let (v321_g, v321_k) ← MemSlice_head_mut_u64 v320 next_13
                                                v321 := v321_g
                                                let (v325, v325_0) ← low_product v316 v318 v321
                                                v321 := v325_0
                                                let v321_0 ← v321_k v321
                                                v320 := v321_0
                                                let v320_0 ← v320_k v320
                                                scratch := v320_0
                                                -- var carry: u64 = 1
                                                carry_r13 := (UInt256.ofNat 1)
                                                -- for i in 0..next {
                                                r23 := (Range_Known_0_Unknown.mk () next_13)
                                                let v332 ← Range_start_0 r23
                                                let rec loop_6 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v14 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                  let mut zero_limbs := zero_limbs
                                                  let mut result := result
                                                  let mut y := y
                                                  let mut q := q
                                                  let mut q_len := q_len
                                                  let mut x := x
                                                  let mut t := t
                                                  let mut borrow := borrow
                                                  let mut inverse := inverse
                                                  let mut guess := guess
                                                  let mut scratch := scratch
                                                  let mut correction := correction
                                                  let mut correct := correct
                                                  let mut carry_r13 := carry_r13
                                                  let mut add := add
                                                  let mut h := h
                                                  let mut work := work
                                                  let mut carry_r17 := carry_r17
                                                  let mut «at» := «at»
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
                                                  let mut base := base
                                                  let mut v320 := v320
                                                  let mut v321 := v321
                                                  let mut v398 := v398
                                                  let mut v399 := v399
                                                  let mut v409 := v409
                                                  let mut v410 := v410
                                                  let mut v424 := v424
                                                  -- for i in 0..next {
                                                  let v335 ← Range_has_0 r23 v14
                                                  if v335 then
                                                    let v337 ← Range_at_0 r23 v14
                                                    let i_338 := v337
                                                    -- let negated = (~scratch[i]) as u128 + carry as u128
                                                    let v340 ← MemArray_index_ref_u64 scratch i_338
                                                    let v341 ← mloadInt (.u 64) v340
                                                    let negated := IntTy.addW (.u 128) (IntTy.not (.u 64) v341) carry_r13
                                                    -- correction[i] = negated.downcast_truncate()
                                                    let v348 ← downcast_truncate_u128_u64 negated
                                                    let (v350, v350_k) ← MemArray_index_mut_u64 correction i_338
                                                    mstoreInt (.u 64) v350 v348
                                                    let v350_0 ← v350_k v350
                                                    correction := v350_0
                                                    -- carry = (negated >> 64).downcast_truncate()
                                                    let carry_354 ← downcast_truncate_u128_u64 (negated >>> (UInt256.ofNat 64))
                                                    carry_r13 := carry_354
                                                    -- for i in 0..next {
                                                    let v358 ← Range_next_0 r23 v14
                                                    loop_6 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v358
                                                  else
                                                    -- var add: u64 = 2
                                                    add := (UInt256.ofNat 2)
                                                    -- for i in 0..next {
                                                    r24 := (Range_Known_0_Unknown.mk () next_13)
                                                    let v366 ← Range_start_0 r24
                                                    let rec loop_7 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v15 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                      let mut zero_limbs := zero_limbs
                                                      let mut result := result
                                                      let mut y := y
                                                      let mut q := q
                                                      let mut q_len := q_len
                                                      let mut x := x
                                                      let mut t := t
                                                      let mut borrow := borrow
                                                      let mut inverse := inverse
                                                      let mut guess := guess
                                                      let mut scratch := scratch
                                                      let mut correction := correction
                                                      let mut correct := correct
                                                      let mut carry_r13 := carry_r13
                                                      let mut add := add
                                                      let mut h := h
                                                      let mut work := work
                                                      let mut carry_r17 := carry_r17
                                                      let mut «at» := «at»
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
                                                      let mut base := base
                                                      let mut v320 := v320
                                                      let mut v321 := v321
                                                      let mut v398 := v398
                                                      let mut v399 := v399
                                                      let mut v409 := v409
                                                      let mut v410 := v410
                                                      let mut v424 := v424
                                                      -- for i in 0..next {
                                                      let v369 ← Range_has_0 r24 v15
                                                      if v369 then
                                                        let v371 ← Range_at_0 r24 v15
                                                        let i_372 := v371
                                                        -- let sum = correction[i] as u128 + add as u128
                                                        let v374 ← MemArray_index_ref_u64 correction i_372
                                                        let v375 ← mloadInt (.u 64) v374
                                                        let sum_379 := IntTy.addW (.u 128) v375 add
                                                        -- correction[i] = sum.downcast_truncate()
                                                        let v381 ← downcast_truncate_u128_u64 sum_379
                                                        let (v383, v383_k) ← MemArray_index_mut_u64 correction i_372
                                                        mstoreInt (.u 64) v383 v381
                                                        let v383_0 ← v383_k v383
                                                        correction := v383_0
                                                        -- add = (sum >> 64).downcast_truncate()
                                                        let add_387 ← downcast_truncate_u128_u64 (sum_379 >>> (UInt256.ofNat 64))
                                                        add := add_387
                                                        -- for i in 0..next {
                                                        let v391 ← Range_next_0 r24 v15
                                                        loop_7 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v391
                                                      else
                                                        -- low_product(inverse.as_slice(), correction.as_slice(), mut scratch.as_slice().head(next))
                                                        let v394 ← MemArray_as_slice_ref_u64 inverse
                                                        let v396 ← MemArray_as_slice_ref_u64 correction
                                                        let (v398_g, v398_k) ← MemArray_as_slice_mut_u64 scratch
                                                        v398 := v398_g
                                                        let (v399_g, v399_k) ← MemSlice_head_mut_u64 v398 next_13
                                                        v399 := v399_g
                                                        let (v403, v403_0) ← low_product v394 v396 v399
                                                        v399 := v403_0
                                                        let v399_0 ← v399_k v399
                                                        v398 := v399_0
                                                        let v398_0 ← v398_k v398
                                                        scratch := v398_0
                                                        -- inverse.as_slice().head(next).copy_from(scratch.as_slice().head(next))
                                                        let v405 ← MemArray_as_slice_ref_u64 scratch
                                                        let v407 ← MemSlice_head_ref_u64 v405 next_13
                                                        let (v409_g, v409_k) ← MemArray_as_slice_mut_u64 inverse
                                                        v409 := v409_g
                                                        let (v410_g, v410_k) ← MemSlice_head_mut_u64 v409 next_13
                                                        v410 := v410_g
                                                        let (v412, v412_0) ← MemSlice_copy_from_u64 v410 v407
                                                        v410 := v412_0
                                                        let v410_0 ← v410_k v410
                                                        v409 := v410_0
                                                        let v409_0 ← v409_k v409
                                                        inverse := v409_0
                                                        -- correct = next
                                                        correct := next_13
                                                        loop_5 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                                    partial_fixpoint
                                                    loop_7 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v366
                                                partial_fixpoint
                                                loop_6 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v332
                                              if v308 then
                                                join_5 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 ((UInt256.ofNat 2) * correct)
                                              else
                                                join_5 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 low
                                            else
                                              -- var h = MemArray<u64>::try_new_zeroed(low)?
                                              let v415 ← MemArray_try_new_zeroed_u64 low
                                              if match v415 with | .Some .. => true | _ => false then
                                                let h_417 := v415.Some_0
                                                h := h_417
                                                -- low_product(t.as_slice(), inverse.as_slice(), mut h.as_slice())
                                                let v420 ← MemArray_as_slice_ref_u64 t
                                                let v422 ← MemArray_as_slice_ref_u64 inverse
                                                let (v424_g, v424_k) ← MemArray_as_slice_mut_u64 h
                                                v424 := v424_g
                                                let (v428, v428_0) ← low_product v420 v422 v424
                                                v424 := v428_0
                                                let v424_0 ← v424_k v424
                                                h := v424_0
                                                -- h[low - 1] &= mask
                                                let (v432, v432_k) ← MemArray_index_mut_u64 h (low - (UInt256.ofNat 1))
                                                let v433 ← mloadInt (.u 64) v432
                                                mstoreInt (.u 64) v432 (v433 &&& mask)
                                                let v432_0 ← v432_k v432
                                                h := v432_0
                                                -- var work = MemArray<u64>::try_new_zeroed(q_len + low + 1)?
                                                let v441 ← MemArray_try_new_zeroed_u64 ((q_len + low) + (UInt256.ofNat 1))
                                                if match v441 with | .Some .. => true | _ => false then
                                                  let work_443 := v441.Some_0
                                                  work := work_443
                                                  -- for i in 0..q_len { work[i] = x[i] }
                                                  r25 := (Range_Known_0_Unknown.mk () q_len)
                                                  let v450 ← Range_start_0 r25
                                                  let rec loop_8 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v16 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                    let mut zero_limbs := zero_limbs
                                                    let mut result := result
                                                    let mut y := y
                                                    let mut q := q
                                                    let mut q_len := q_len
                                                    let mut x := x
                                                    let mut t := t
                                                    let mut borrow := borrow
                                                    let mut inverse := inverse
                                                    let mut guess := guess
                                                    let mut scratch := scratch
                                                    let mut correction := correction
                                                    let mut correct := correct
                                                    let mut carry_r13 := carry_r13
                                                    let mut add := add
                                                    let mut h := h
                                                    let mut work := work
                                                    let mut carry_r17 := carry_r17
                                                    let mut «at» := «at»
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
                                                    let mut base := base
                                                    let mut v320 := v320
                                                    let mut v321 := v321
                                                    let mut v398 := v398
                                                    let mut v399 := v399
                                                    let mut v409 := v409
                                                    let mut v410 := v410
                                                    let mut v424 := v424
                                                    -- for i in 0..q_len { work[i] = x[i] }
                                                    let v453 ← Range_has_0 r25 v16
                                                    if v453 then
                                                      let v455 ← Range_at_0 r25 v16
                                                      let i_456 := v455
                                                      -- work[i] = x[i]
                                                      let v458 ← MemArray_index_ref_u64 x i_456
                                                      let v459 ← mloadInt (.u 64) v458
                                                      let (v461, v461_k) ← MemArray_index_mut_u64 work i_456
                                                      mstoreInt (.u 64) v461 v459
                                                      let v461_0 ← v461_k v461
                                                      work := v461_0
                                                      -- for i in 0..q_len { work[i] = x[i] }
                                                      let v464 ← Range_next_0 r25 v16
                                                      loop_8 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v464
                                                    else
                                                      -- for i in 0..q_len {
                                                      r26 := (Range_Known_0_Unknown.mk () q_len)
                                                      let v471 ← Range_start_0 r26
                                                      let rec loop_9 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v17 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                        let mut zero_limbs := zero_limbs
                                                        let mut result := result
                                                        let mut y := y
                                                        let mut q := q
                                                        let mut q_len := q_len
                                                        let mut x := x
                                                        let mut t := t
                                                        let mut borrow := borrow
                                                        let mut inverse := inverse
                                                        let mut guess := guess
                                                        let mut scratch := scratch
                                                        let mut correction := correction
                                                        let mut correct := correct
                                                        let mut carry_r13 := carry_r13
                                                        let mut add := add
                                                        let mut h := h
                                                        let mut work := work
                                                        let mut carry_r17 := carry_r17
                                                        let mut «at» := «at»
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
                                                        let mut base := base
                                                        let mut v320 := v320
                                                        let mut v321 := v321
                                                        let mut v398 := v398
                                                        let mut v399 := v399
                                                        let mut v409 := v409
                                                        let mut v410 := v410
                                                        let mut v424 := v424
                                                        -- for i in 0..q_len {
                                                        let v474 ← Range_has_0 r26 v17
                                                        if v474 then
                                                          let v476 ← Range_at_0 r26 v17
                                                          let i_477 := v476
                                                          -- var carry: u64 = 0
                                                          carry_r17 := (UInt256.ofNat 0)
                                                          -- for j in 0..low {
                                                          r27 := (Range_Known_0_Unknown.mk () low)
                                                          let v484 ← Range_start_0 r27
                                                          let rec loop_10 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v18 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                            let mut zero_limbs := zero_limbs
                                                            let mut result := result
                                                            let mut y := y
                                                            let mut q := q
                                                            let mut q_len := q_len
                                                            let mut x := x
                                                            let mut t := t
                                                            let mut borrow := borrow
                                                            let mut inverse := inverse
                                                            let mut guess := guess
                                                            let mut scratch := scratch
                                                            let mut correction := correction
                                                            let mut correct := correct
                                                            let mut carry_r13 := carry_r13
                                                            let mut add := add
                                                            let mut h := h
                                                            let mut work := work
                                                            let mut carry_r17 := carry_r17
                                                            let mut «at» := «at»
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
                                                            let mut base := base
                                                            let mut v320 := v320
                                                            let mut v321 := v321
                                                            let mut v398 := v398
                                                            let mut v399 := v399
                                                            let mut v409 := v409
                                                            let mut v410 := v410
                                                            let mut v424 := v424
                                                            -- for j in 0..low {
                                                            let v487 ← Range_has_0 r27 v18
                                                            if v487 then
                                                              let v489 ← Range_at_0 r27 v18
                                                              let j := v489
                                                              -- let sum = work[i + j] as u128 + q[i] as u128 * h[j] as u128 + carry as u128
                                                              let v493 ← MemArray_index_ref_u64 work (i_477 + j)
                                                              let v494 ← mloadInt (.u 64) v493
                                                              let v497 ← MemArray_index_ref_u64 q i_477
                                                              let v498 ← mloadInt (.u 64) v497
                                                              let v501 ← MemArray_index_ref_u64 h j
                                                              let v502 ← mloadInt (.u 64) v501
                                                              let v505 := IntTy.addW (.u 128) v494 (IntTy.mulW (.u 128) v498 v502)
                                                              let sum_508 := IntTy.addW (.u 128) v505 carry_r17
                                                              -- work[i + j] = sum.downcast_truncate()
                                                              let v510 ← downcast_truncate_u128_u64 sum_508
                                                              let (v513, v513_k) ← MemArray_index_mut_u64 work (i_477 + j)
                                                              mstoreInt (.u 64) v513 v510
                                                              let v513_0 ← v513_k v513
                                                              work := v513_0
                                                              -- carry = (sum >> 64).downcast_truncate()
                                                              let carry_517 ← downcast_truncate_u128_u64 (sum_508 >>> (UInt256.ofNat 64))
                                                              carry_r17 := carry_517
                                                              -- for j in 0..low {
                                                              let v521 ← Range_next_0 r27 v18
                                                              loop_10 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v521
                                                            else
                                                              -- var at = i + low
                                                              «at» := (i_477 + low)
                                                              let rec loop_11 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                                let mut zero_limbs := zero_limbs
                                                                let mut result := result
                                                                let mut y := y
                                                                let mut q := q
                                                                let mut q_len := q_len
                                                                let mut x := x
                                                                let mut t := t
                                                                let mut borrow := borrow
                                                                let mut inverse := inverse
                                                                let mut guess := guess
                                                                let mut scratch := scratch
                                                                let mut correction := correction
                                                                let mut correct := correct
                                                                let mut carry_r13 := carry_r13
                                                                let mut add := add
                                                                let mut h := h
                                                                let mut work := work
                                                                let mut carry_r17 := carry_r17
                                                                let mut «at» := «at»
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
                                                                let mut base := base
                                                                let mut v320 := v320
                                                                let mut v321 := v321
                                                                let mut v398 := v398
                                                                let mut v399 := v399
                                                                let mut v409 := v409
                                                                let mut v410 := v410
                                                                let mut v424 := v424
                                                                -- while carry != 0 {
                                                                if decide (carry_r17 ≠ (UInt256.ofNat 0)) then
                                                                  -- let sum = work[at] as u128 + carry as u128
                                                                  let at_529 := «at»
                                                                  let v532 ← MemArray_index_ref_u64 work at_529
                                                                  let v533 ← mloadInt (.u 64) v532
                                                                  let sum_537 := IntTy.addW (.u 128) v533 carry_r17
                                                                  -- work[at] = sum.downcast_truncate()
                                                                  let v539 ← downcast_truncate_u128_u64 sum_537
                                                                  let at_540 := «at»
                                                                  let (v543, v543_k) ← MemArray_index_mut_u64 work at_540
                                                                  mstoreInt (.u 64) v543 v539
                                                                  let v543_0 ← v543_k v543
                                                                  work := v543_0
                                                                  -- carry = (sum >> 64).downcast_truncate()
                                                                  let carry_547 ← downcast_truncate_u128_u64 (sum_537 >>> (UInt256.ofNat 64))
                                                                  carry_r17 := carry_547
                                                                  -- at += 1
                                                                  «at» := («at» + (UInt256.ofNat 1))
                                                                  loop_11 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                                                else
                                                                  -- for i in 0..q_len {
                                                                  let v557 ← Range_next_0 r26 v17
                                                                  loop_9 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v557
                                                              partial_fixpoint
                                                              loop_11 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                                          partial_fixpoint
                                                          loop_10 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v484
                                                        else
                                                          -- for i in 0..size { result[i] = work[i] }
                                                          r28 := (Range_Known_0_Unknown.mk () size)
                                                          let v564 ← Range_start_0 r28
                                                          let rec loop_12 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v19 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                                                            let mut zero_limbs := zero_limbs
                                                            let mut result := result
                                                            let mut y := y
                                                            let mut q := q
                                                            let mut q_len := q_len
                                                            let mut x := x
                                                            let mut t := t
                                                            let mut borrow := borrow
                                                            let mut inverse := inverse
                                                            let mut guess := guess
                                                            let mut scratch := scratch
                                                            let mut correction := correction
                                                            let mut correct := correct
                                                            let mut carry_r13 := carry_r13
                                                            let mut add := add
                                                            let mut h := h
                                                            let mut work := work
                                                            let mut carry_r17 := carry_r17
                                                            let mut «at» := «at»
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
                                                            let mut base := base
                                                            let mut v320 := v320
                                                            let mut v321 := v321
                                                            let mut v398 := v398
                                                            let mut v399 := v399
                                                            let mut v409 := v409
                                                            let mut v410 := v410
                                                            let mut v424 := v424
                                                            -- for i in 0..size { result[i] = work[i] }
                                                            let v567 ← Range_has_0 r28 v19
                                                            if v567 then
                                                              let v569 ← Range_at_0 r28 v19
                                                              let i_570 := v569
                                                              -- result[i] = work[i]
                                                              let v572 ← MemArray_index_ref_u64 work i_570
                                                              let v573 ← mloadInt (.u 64) v572
                                                              let (v575, v575_k) ← MemArray_index_mut_u64 result i_570
                                                              mstoreInt (.u 64) v575 v573
                                                              let v575_0 ← v575_k v575
                                                              result := v575_0
                                                              -- for i in 0..size { result[i] = work[i] }
                                                              let v578 ← Range_next_0 r28 v19
                                                              loop_12 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v578
                                                            else
                                                              -- Option::Some(result)
                                                              let result_580 := result
                                                              let _ ← MemArray_drop_u64 work
                                                              let _ ← MemArray_drop_u64 h
                                                              let _ ← MemArray_drop_u64 correction
                                                              let _ ← MemArray_drop_u64 scratch
                                                              let _ ← MemArray_drop_u64 inverse
                                                              let _ ← MemArray_drop_u64 t
                                                              let _ ← MemArray_drop_u64 x
                                                              let _ ← MemArray_drop_u64 q
                                                              let _ ← MemArray_drop_u64 y
                                                              return ((Option_MemArray_u64.Some result_580), base)
                                                          partial_fixpoint
                                                          loop_12 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v564
                                                      partial_fixpoint
                                                      loop_9 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v471
                                                  partial_fixpoint
                                                  loop_8 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v450
                                                else
                                                  match (← pure v441) with
                                                  | .Some .. =>
                                                    let _ ← MemArray_drop_u64 v441.Some_0
                                                    pure ()
                                                  | _ => pure ()
                                                  let _ ← MemArray_drop_u64 h
                                                  let _ ← MemArray_drop_u64 correction
                                                  let _ ← MemArray_drop_u64 scratch
                                                  let _ ← MemArray_drop_u64 inverse
                                                  let _ ← MemArray_drop_u64 t
                                                  let _ ← MemArray_drop_u64 x
                                                  let _ ← MemArray_drop_u64 q
                                                  let _ ← MemArray_drop_u64 y
                                                  let _ ← MemArray_drop_u64 result
                                                  return (Option_MemArray_u64.None, base)
                                              else
                                                match (← pure v415) with
                                                | .Some .. =>
                                                  let _ ← MemArray_drop_u64 v415.Some_0
                                                  pure ()
                                                | _ => pure ()
                                                let _ ← MemArray_drop_u64 correction
                                                let _ ← MemArray_drop_u64 scratch
                                                let _ ← MemArray_drop_u64 inverse
                                                let _ ← MemArray_drop_u64 t
                                                let _ ← MemArray_drop_u64 x
                                                let _ ← MemArray_drop_u64 q
                                                let _ ← MemArray_drop_u64 y
                                                let _ ← MemArray_drop_u64 result
                                                return (Option_MemArray_u64.None, base)
                                          partial_fixpoint
                                          loop_5 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                        else
                                          match (← pure v296) with
                                          | .Some .. =>
                                            let _ ← MemArray_drop_u64 v296.Some_0
                                            pure ()
                                          | _ => pure ()
                                          let _ ← MemArray_drop_u64 scratch
                                          let _ ← MemArray_drop_u64 inverse
                                          let _ ← MemArray_drop_u64 t
                                          let _ ← MemArray_drop_u64 x
                                          let _ ← MemArray_drop_u64 q
                                          let _ ← MemArray_drop_u64 y
                                          let _ ← MemArray_drop_u64 result
                                          return (Option_MemArray_u64.None, base)
                                      else
                                        match (← pure v292) with
                                        | .Some .. =>
                                          let _ ← MemArray_drop_u64 v292.Some_0
                                          pure ()
                                        | _ => pure ()
                                        let _ ← MemArray_drop_u64 inverse
                                        let _ ← MemArray_drop_u64 t
                                        let _ ← MemArray_drop_u64 x
                                        let _ ← MemArray_drop_u64 q
                                        let _ ← MemArray_drop_u64 y
                                        let _ ← MemArray_drop_u64 result
                                        return (Option_MemArray_u64.None, base)
                                  partial_fixpoint
                                  loop_4 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v264
                                else
                                  match (← pure v251) with
                                  | .Some .. =>
                                    let _ ← MemArray_drop_u64 v251.Some_0
                                    pure ()
                                  | _ => pure ()
                                  let _ ← MemArray_drop_u64 t
                                  let _ ← MemArray_drop_u64 x
                                  let _ ← MemArray_drop_u64 q
                                  let _ ← MemArray_drop_u64 y
                                  let _ ← MemArray_drop_u64 result
                                  return (Option_MemArray_u64.None, base)
                            partial_fixpoint
                            loop_3 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v206
                          else
                            match (← pure v196) with
                            | .Some .. =>
                              let _ ← MemArray_drop_u64 v196.Some_0
                              pure ()
                            | _ => pure ()
                            let _ ← MemArray_drop_u64 x
                            let _ ← MemArray_drop_u64 q
                            let _ ← MemArray_drop_u64 y
                            let _ ← MemArray_drop_u64 result
                            return (Option_MemArray_u64.None, base)
                        else
                          match (← pure v192) with
                          | .Some .. =>
                            let _ ← MemArray_drop_u64 v192.Some_0
                            pure ()
                          | _ => pure ()
                          let _ ← MemArray_drop_u64 q
                          let _ ← MemArray_drop_u64 y
                          let _ ← MemArray_drop_u64 result
                          return (Option_MemArray_u64.None, base)
                      -- if q_len == 1 && q[0] == 1 {
                      if decide (q_len = (UInt256.ofNat 1)) then
                        let v158 ← MemArray_index_ref_u64 q (UInt256.ofNat 0)
                        let v159 ← mloadInt (.u 64) v158
                        if decide (v159 = (UInt256.ofNat 1)) then
                          -- for i in 0..low { result[i] = y[i] }
                          r20 := (Range_Known_0_Unknown.mk () low)
                          let v166 ← Range_start_0 r20
                          let rec loop_13 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v9 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
                            let mut zero_limbs := zero_limbs
                            let mut result := result
                            let mut y := y
                            let mut q := q
                            let mut q_len := q_len
                            let mut x := x
                            let mut t := t
                            let mut borrow := borrow
                            let mut inverse := inverse
                            let mut guess := guess
                            let mut scratch := scratch
                            let mut correction := correction
                            let mut correct := correct
                            let mut carry_r13 := carry_r13
                            let mut add := add
                            let mut h := h
                            let mut work := work
                            let mut carry_r17 := carry_r17
                            let mut «at» := «at»
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
                            let mut base := base
                            let mut v320 := v320
                            let mut v321 := v321
                            let mut v398 := v398
                            let mut v399 := v399
                            let mut v409 := v409
                            let mut v410 := v410
                            let mut v424 := v424
                            -- for i in 0..low { result[i] = y[i] }
                            let v169 ← Range_has_0 r20 v9
                            if v169 then
                              let v171 ← Range_at_0 r20 v9
                              let i_172 := v171
                              -- result[i] = y[i]
                              let v174 ← MemArray_index_ref_u64 y i_172
                              let v175 ← mloadInt (.u 64) v174
                              let (v177, v177_k) ← MemArray_index_mut_u64 result i_172
                              mstoreInt (.u 64) v177 v175
                              let v177_0 ← v177_k v177
                              result := v177_0
                              -- for i in 0..low { result[i] = y[i] }
                              let v180 ← Range_next_0 r20 v9
                              loop_13 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v180
                            else
                              -- return Option::Some(result)
                              let result_182 := result
                              let _ ← MemArray_drop_u64 q
                              let _ ← MemArray_drop_u64 y
                              return ((Option_MemArray_u64.Some result_182), base)
                          partial_fixpoint
                          loop_13 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v166
                        else
                          join_3 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                      else
                        join_3 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                  partial_fixpoint
                  loop_2 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
              partial_fixpoint
              loop_1 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v86
            else
              match (← pure v74) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v74.Some_0
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 y
              let _ ← MemArray_drop_u64 result
              return (Option_MemArray_u64.None, base)
          else
            match (← pure v70) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v70.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 result
            return (Option_MemArray_u64.None, base)
        else
          match (← pure v64) with
          | .Some .. =>
            let _ ← MemArray_drop_u64 v64.Some_0
            pure ()
          | _ => pure ()
          return (Option_MemArray_u64.None, base)
      -- let mask: u64 = if bits == 0 { 0xffffffffffffffff } else { (1 << bits) - 1 }
      if decide (bits = (UInt256.ofNat 0)) then
        join zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 (UInt256.ofNat 18446744073709551615)
      else
        -- (1 << bits) - 1
        let v61 := IntTy.subW (.u 64) (IntTy.shl (.u 64) (UInt256.ofNat 1) bits) (UInt256.ofNat 1)
        join zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v61
  partial_fixpoint
  loop zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424

/-- `lib::modexp::power<MemBuffer>` -/
def power_MemBuffer (input : MemBuffer) (base_len : UInt256) (exponent_len : UInt256) (modulus_len : UInt256) : FeM Option_MemArray_u64 := do
  let mut value : MemArray_u64 := default
  let mut n : UInt256 := default
  let mut exponent : Exponent := default
  let mut one : MemArray_u64 := default
  let mut base : MemArray_u64 := default
  let mut base_size : UInt256 := default
  -- let exponent_start = 96 + base_len
  let exponent_start ← checkedAdd (UInt256.ofNat 96) base_len
  -- let modulus_start = exponent_start + exponent_len
  let modulus_start ← checkedAdd exponent_start exponent_len
  -- let value = input_limbs(input, offset: modulus_start, len: modulus_len, count: (modulus_len + 7) / 8)?
  let v14 ← checkedAdd modulus_len (UInt256.ofNat 7)
  let v16 ← checkedDiv v14 (UInt256.ofNat 8)
  let v17 ← input_limbs_MemBuffer input modulus_start modulus_len v16
  if match v17 with | .Some .. => true | _ => false then
    let value_19 := v17.Some_0
    value := value_19
    -- var n = value.len()
    let n_22 ← MemArray_len_u64 value
    n := n_22
    let rec loop (value : MemArray_u64) (n : UInt256) (exponent : Exponent) (one : MemArray_u64) (base : MemArray_u64) (base_size : UInt256) : FeM Option_MemArray_u64 := do
      let mut value := value
      let mut n := n
      let mut exponent := exponent
      let mut one := one
      let mut base := base
      let mut base_size := base_size
      let join : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
        let mut value := value
        let mut n := n
        let mut exponent := exponent
        let mut one := one
        let mut base := base
        let mut base_size := base_size
        let join_1 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
          let mut value := value
          let mut n := n
          let mut exponent := exponent
          let mut one := one
          let mut base := base
          let mut base_size := base_size
          -- return MemArray<u64>::try_new_zeroed(1)
          let v156 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 1)
          let _ ← MemArray_drop_u64 value
          return v156
        -- if n == 0 || (n == 1 && value[0] == 1) { return MemArray<u64>::try_new_zeroed(1) }
        if decide (n = (UInt256.ofNat 0)) then
          join_1 value n exponent one base base_size
        else
          let join_2 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
            let mut value := value
            let mut n := n
            let mut exponent := exponent
            let mut one := one
            let mut base := base
            let mut base_size := base_size
            -- let exponent = Exponent::try_read(input, start: exponent_start, len: exponent_len)?
            let v58 ← Exponent_try_read_MemBuffer input exponent_start exponent_len
            if match v58 with | .Some .. => true | _ => false then
              let exponent_60 := v58.Some_0
              exponent := exponent_60
              -- if exponent.top == exponent.total {
              if decide (exponent.top = exponent.total) then
                -- var one = MemArray<u64>::try_new_zeroed(1)?
                let v66 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 1)
                if match v66 with | .Some .. => true | _ => false then
                  let one_68 := v66.Some_0
                  one := one_68
                  -- one[0] = 1
                  let (v73, v73_k) ← MemArray_index_mut_u64 one (UInt256.ofNat 0)
                  mstoreInt (.u 64) v73 (UInt256.ofNat 1)
                  let v73_0 ← v73_k v73
                  one := v73_0
                  -- return Option::Some(one)
                  let one_75 := one
                  let _ ← MemArray_drop_u64 exponent.limbs
                  let _ ← MemArray_drop_u64 value
                  return Option_MemArray_u64.Some one_75
                else
                  match (← pure v66) with
                  | .Some .. =>
                    let _ ← MemArray_drop_u64 v66.Some_0
                    pure ()
                  | _ => pure ()
                  let _ ← MemArray_drop_u64 exponent.limbs
                  let _ ← MemArray_drop_u64 value
                  return Option_MemArray_u64.None
              else
                -- let base_limbs = (base_len + 7) / 8
                let v83 ← checkedAdd base_len (UInt256.ofNat 7)
                let base_limbs ← checkedDiv v83 (UInt256.ofNat 8)
                let join_3 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size v4 => do
                  let mut value := value
                  let mut n := n
                  let mut exponent := exponent
                  let mut one := one
                  let mut base := base
                  let mut base_size := base_size
                  -- var base = input_limbs(input, offset: 96, len: base_len, count: (if base_limbs > n { base_limbs } else { n }) + 1)?
                  let v95 ← checkedAdd v4 (UInt256.ofNat 1)
                  let v96 ← input_limbs_MemBuffer input (UInt256.ofNat 96) base_len v95
                  if match v96 with | .Some .. => true | _ => false then
                    let base_98 := v96.Some_0
                    base := base_98
                    -- var base_size = base_limbs
                    base_size := base_limbs
                    let rec loop_1 (value : MemArray_u64) (n : UInt256) (exponent : Exponent) (one : MemArray_u64) (base : MemArray_u64) (base_size : UInt256) : FeM Option_MemArray_u64 := do
                      let mut value := value
                      let mut n := n
                      let mut exponent := exponent
                      let mut one := one
                      let mut base := base
                      let mut base_size := base_size
                      let join_4 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
                        let mut value := value
                        let mut n := n
                        let mut exponent := exponent
                        let mut one := one
                        let mut base := base
                        let mut base_size := base_size
                        -- if base_size == 0 { return MemArray<u64>::try_new_zeroed(1) }
                        if decide (base_size = (UInt256.ofNat 0)) then
                          -- return MemArray<u64>::try_new_zeroed(1)
                          let v124 ← MemArray_try_new_zeroed_u64 (UInt256.ofNat 1)
                          let _ ← MemArray_drop_u64 base
                          let _ ← MemArray_drop_u64 exponent.limbs
                          let _ ← MemArray_drop_u64 value
                          return v124
                        else
                          -- if value[0] & 1 == 1 { return odd_power(exponent, mut base, base_size, value, n) }
                          let v129 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
                          let v130 ← mloadInt (.u 64) v129
                          let v134 := decide ((v130 &&& (UInt256.ofNat 1)) = (UInt256.ofNat 1))
                          if v134 then
                            -- return odd_power(exponent, mut base, base_size, value, n)
                            let base_size_136 := base_size
                            let n_138 := n
                            let (v142, v142_0) ← odd_power exponent base base_size_136 value n_138
                            base := v142_0
                            let _ ← MemArray_drop_u64 base
                            let _ ← MemArray_drop_u64 exponent.limbs
                            let _ ← MemArray_drop_u64 value
                            return v142
                          else
                            -- even_power(exponent, mut base, base_size, value, n)
                            let base_size_146 := base_size
                            let n_148 := n
                            let (v152, v152_0) ← even_power exponent base base_size_146 value n_148
                            base := v152_0
                            let _ ← MemArray_drop_u64 base
                            let _ ← MemArray_drop_u64 exponent.limbs
                            let _ ← MemArray_drop_u64 value
                            return v152
                      -- while base_size > 0 && base[base_size - 1] == 0 { base_size -= 1 }
                      if decide ((UInt256.ofNat 0).toNat < base_size.toNat) then
                        let v108 ← checkedSub base_size (UInt256.ofNat 1)
                        let v110 ← MemArray_index_ref_u64 base v108
                        let v111 ← mloadInt (.u 64) v110
                        if decide (v111 = (UInt256.ofNat 0)) then
                          -- base_size -= 1
                          let base_size_116 ← checkedSub base_size (UInt256.ofNat 1)
                          base_size := base_size_116
                          loop_1 value n exponent one base base_size
                        else
                          join_4 value n exponent one base base_size
                      else
                        join_4 value n exponent one base base_size
                    partial_fixpoint
                    loop_1 value n exponent one base base_size
                  else
                    match (← pure v96) with
                    | .Some .. =>
                      let _ ← MemArray_drop_u64 v96.Some_0
                      pure ()
                    | _ => pure ()
                    let _ ← MemArray_drop_u64 exponent.limbs
                    let _ ← MemArray_drop_u64 value
                    return Option_MemArray_u64.None
                -- var base = input_limbs(input, offset: 96, len: base_len, count: (if base_limbs > n { base_limbs } else { n }) + 1)?
                if decide (n.toNat < base_limbs.toNat) then
                  join_3 value n exponent one base base_size base_limbs
                else
                  join_3 value n exponent one base base_size n
            else
              match (← pure v58) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v58.Some_0.limbs
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 value
              return Option_MemArray_u64.None
          if decide (n = (UInt256.ofNat 1)) then
            let v52 ← MemArray_index_ref_u64 value (UInt256.ofNat 0)
            let v53 ← mloadInt (.u 64) v52
            if decide (v53 = (UInt256.ofNat 1)) then
              join_1 value n exponent one base base_size
            else
              join_2 value n exponent one base base_size
          else
            join_2 value n exponent one base base_size
      -- while n > 0 && value[n - 1] == 0 { n -= 1 }
      if decide ((UInt256.ofNat 0).toNat < n.toNat) then
        let v31 ← checkedSub n (UInt256.ofNat 1)
        let v33 ← MemArray_index_ref_u64 value v31
        let v34 ← mloadInt (.u 64) v33
        if decide (v34 = (UInt256.ofNat 0)) then
          -- n -= 1
          let n_39 ← checkedSub n (UInt256.ofNat 1)
          n := n_39
          loop value n exponent one base base_size
        else
          join value n exponent one base base_size
      else
        join value n exponent one base base_size
    partial_fixpoint
    loop value n exponent one base base_size
  else
    match (← pure v17) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v17.Some_0
      pure ()
    | _ => pure ()
    return Option_MemArray_u64.None

/-- `lib::modexp::encoded` -/
def encoded (result : MemArray_u64) (output : MemBuffer) : FeM MemBuffer := do
  let mut output := output
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Range_Known_0_Known_8 := default
  -- let len = output.len()
  let len ← MemBuffer_len output
  -- let bytes = output.ptr()
  let bytes ← MemBuffer_ptr output
  -- for i in 0..result.len() {
  let v10 ← MemArray_len_u64 result
  r1 := (Range_Known_0_Unknown.mk () v10)
  let v14 ← Range_start_0 r1
  let rec loop (output : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) (v1 : UInt256) : FeM MemBuffer := do
    let mut output := output
    let mut r1 := r1
    let mut r2 := r2
    -- for i in 0..result.len() {
    let v17 ← Range_has_0 r1 v1
    if v17 then
      let v19 ← Range_at_0 r1 v1
      let i := v19
      -- let value = result[i]
      let v21 ← MemArray_index_ref_u64 result i
      let value ← mloadInt (.u 64) v21
      -- if 8 * i + 8 <= len {
      let v27 := decide ((((UInt256.ofNat 8) * i) + (UInt256.ofNat 8)).toNat ≤ len.toNat)
      let rec loop_1 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) : FeM MemBuffer := do
        let mut output := output
        let mut r1 := r1
        let mut r2 := r2
        -- for i in 0..result.len() {
        let v75 ← Range_next_0 r1 v1
        loop output r1 r2 v75
      partial_fixpoint
      if v27 then
        -- *cast<u8, u64>(offset_bytes(bytes, len - 8 * i - 8)) = to_be(value)
        let v33 := (len - ((UInt256.ofNat 8) * i)) - (UInt256.ofNat 8)
        let v35 ← cast_u8_u64 (bytes + v33)
        mstoreInt (.u 64) v35 value
        loop_1 output r1 r2
      else
        -- for byte in 0..8 {
        r2 := (Range_Known_0_Known_8.mk () ())
        let v43 ← Range_start_0_8 r2
        let rec loop_2 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) (v2 : UInt256) : FeM MemBuffer := do
          let mut output := output
          let mut r1 := r1
          let mut r2 := r2
          -- for byte in 0..8 {
          let v46 ← Range_has_0_8 r2 v2
          if v46 then
            let v48 ← Range_at_0_8 r2 v2
            let byte := v48
            -- let position = 8 * i + byte
            let position := ((UInt256.ofNat 8) * i) + byte
            let join : MemBuffer → Range_Known_0_Unknown → Range_Known_0_Known_8 → FeM MemBuffer := fun output r1 r2 => do
              let mut output := output
              let mut r1 := r1
              let mut r2 := r2
              -- for byte in 0..8 {
              let v70 ← Range_next_0_8 r2 v2
              loop_2 output r1 r2 v70
            -- if position < len {
            if decide (position.toNat < len.toNat) then
              -- let shift: u64 = (8 * byte).downcast_truncate()
              let shift ← downcast_truncate_usize_u64 ((UInt256.ofNat 8) * byte)
              -- *offset_bytes(bytes, len - 1 - position) = (value >> shift).downcast_truncate()
              let v60 ← downcast_truncate_u64_u8 (value >>> shift)
              mstoreInt (.u 8) (bytes + ((len - (UInt256.ofNat 1)) - position)) v60
              join output r1 r2
            else
              join output r1 r2
          else
            loop_1 output r1 r2
        partial_fixpoint
        loop_2 output r1 r2 v43
    else
      return output
  partial_fixpoint
  loop output r1 r2 v14

/-- `lib::modexp::run<MemBuffer>` -/
def run_MemBuffer (input : MemBuffer) (gas_1 : UInt256) : FeM Output := do
  let mut result : MemArray_u64 := default
  -- let base_word = input_word(input, offset: 0, len: 32)
  let base_word ← input_word_MemBuffer input (UInt256.ofNat 0) (UInt256.ofNat 32)
  -- let exponent_word = input_word(input, offset: 32, len: 32)
  let exponent_word ← input_word_MemBuffer input (UInt256.ofNat 32) (UInt256.ofNat 32)
  -- let modulus_word = input_word(input, offset: 64, len: 32)
  let modulus_word ← input_word_MemBuffer input (UInt256.ofNat 64) (UInt256.ofNat 32)
  -- if base_word > MAX_LENGTH || exponent_word > MAX_LENGTH || modulus_word > MAX_LENGTH {
  let v14 ← MAX_LENGTH
  let join : MemArray_u64 → FeM Output := fun result => do
    let mut result := result
    -- return Output { status: FAILURE, gas_used: gas, output: MemBuffer::empty() }
    let v60 ← FAILURE
    let v62 ← MemBuffer_empty
    return Output.mk v60 gas_1 v62
  if decide (v14.toNat < base_word.toNat) then
    join result
  else
    let v16 ← MAX_LENGTH
    if decide (v16.toNat < exponent_word.toNat) then
      join result
    else
      let v18 ← MAX_LENGTH
      if decide (v18.toNat < modulus_word.toNat) then
        join result
      else
        -- let base_len: usize = base_word.downcast_unchecked()
        let base_len ← downcast_unchecked_u256_usize base_word
        -- let exponent_len: usize = exponent_word.downcast_unchecked()
        let exponent_len ← downcast_unchecked_u256_usize exponent_word
        -- let modulus_len: usize = modulus_word.downcast_unchecked()
        let modulus_len ← downcast_unchecked_u256_usize modulus_word
        -- let gas_used = gas_cost(input, base: base_len, exponent: exponent_len, modulus: modulus_len)
        let gas_used ← gas_cost_MemBuffer input base_len exponent_len modulus_len
        -- if gas_used > gas { return Output { status: FAILURE, gas_used: gas, output: MemBuffer::empty() } }
        if decide (gas_1.toNat < gas_used.toNat) then
          -- return Output { status: FAILURE, gas_used: gas, output: MemBuffer::empty() }
          let v32 ← FAILURE
          let v34 ← MemBuffer_empty
          return Output.mk v32 gas_1 v34
        else
          -- if modulus_len == 0 { return Output { status: SUCCESS, gas_used, output: MemBuffer::empty() } }
          if decide (modulus_len = (UInt256.ofNat 0)) then
            -- return Output { status: SUCCESS, gas_used, output: MemBuffer::empty() }
            let v40 ← SUCCESS
            let v41 ← MemBuffer_empty
            return Output.mk v40 gas_used v41
          else
            -- let Option::Some(output) = MemBuffer::try_new(modulus_len) else { return host_allocation(gas_used) }
            let v45 ← MemBuffer_try_new modulus_len
            match v45 with
            | .Some .. =>
              let output := v45.Some_0
              -- let Option::Some(result) = power(input, base_len, exponent_len, modulus_len) else {
              let v49 ← power_MemBuffer input base_len exponent_len modulus_len
              match v49 with
              | .Some .. =>
                result := v49.Some_0
                -- Output { status: SUCCESS, gas_used, output: encoded(result, output) }
                let v53 ← SUCCESS
                let v56 ← encoded result output
                let _ ← MemArray_drop_u64 result
                return Output.mk v53 gas_used v56
              | _ =>
                -- return host_allocation(gas_used)
                let v58 ← host_allocation gas_used
                match v49 with
                | .Some .. =>
                  let _ ← MemArray_drop_u64 v49.Some_0
                  pure ()
                | _ => pure ()
                let _ ← MemBuffer_drop output
                return v58
            | _ =>
              -- return host_allocation(gas_used)
              let v59 ← host_allocation gas_used
              match v45 with
              | .Some .. =>
                let _ ← MemBuffer_drop v45.Some_0
                pure ()
              | _ => pure ()
              return v59

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Empty := do
  let mut result : Output := default
  let mut r2 : MemBuffer := default
  -- let result = modexp::run(input(), gas: 0xffffffffffffffff)
  let v0 ← input_Evm_6
  r2 := v0
  let result_3 ← run_MemBuffer r2 (UInt256.ofNat 18446744073709551615)
  result := result_3
  let _ ← MemBuffer_drop r2
  -- if result.status != SUCCESS { raw.revert(data: MemSpan::empty()) }
  let v6 ← SUCCESS
  if decide (result.status ≠ v6) then
    -- raw.revert(data: MemSpan::empty())
    let v8 ← MemSlice_empty_u8
    let v11 ← Evm_revert v8
    nomatch v11
  else
    -- raw.return_data(data: result.output.span())
    let v16 ← MemBuffer_span_ref result.output
    let v19 ← Evm_return_data v16
    nomatch v19

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  nonPayable
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `Modexp`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
  else
    let selector ← calldataSelector
    recv_fallback_entry

/-- The deployment entry of contract `Modexp`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


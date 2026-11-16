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

def init : FeM Unit := do
  return ()

def input_Evm_6 : FeM MemBuffer := do
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

def input_word_MemBuffer (input : MemBuffer) (offset_1 : UInt256) (len_2 : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut i : UInt256 := default
  let v5 ← MemBuffer_len_1 input
  let «end» := v5
  let offset_7 := offset_1
  let v8 := decide («end».toNat ≤ offset_7.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun value i v3 => do
    let mut value := value
    let mut i := i
    let present := v3
    let v22 := (UInt256.ofNat 0)
    let v23 := decide (present = v22)
    if v23 then
      let v24 := (UInt256.ofNat 0)
      return v24
    else
      let v25 := ()
      let v26 := v25
      let v27 := (UInt256.ofNat 0)
      let value_28 := v27
      value := value_28
      let v29 := (UInt256.ofNat 0)
      let i_30 := v29
      i := i_30
      let rec b10 (value : UInt256) (i : UInt256) : FeM UInt256 := do
        let mut value := value
        let mut i := i
        let i_31 := i
        let i_32 := i_31
        let v33 := (UInt256.ofNat 8)
        let v34 := i_32 + v33
        let v35 := decide (v34.toNat ≤ present.toNat)
        if v35 then
          let value_36 := value
          let value_37 := value_36
          let v38 := (UInt256.ofNat 64)
          let v39 := value_37 <<< v38
          let offset_40 := offset_1
          let i_41 := i
          let v42 := offset_40 + i_41
          let v43 ← MemBuffer_lane_at input v42
          let v44 := IntTy.swapBytes (.u 64) v43
          let v45 := v44
          let v46 := v39 ||| v45
          let value_47 := v46
          value := value_47
          let v48 := ()
          let v49 := (UInt256.ofNat 8)
          let i_50 := i
          let v51 := i_50 + v49
          let i_52 := v51
          i := i_52
          let v53 := ()
          b10 value i
        else
          let rec b13 (value : UInt256) (i : UInt256) : FeM UInt256 := do
            let mut value := value
            let mut i := i
            let i_54 := i
            let i_55 := i_54
            let v56 := decide (i_55.toNat < present.toNat)
            if v56 then
              let value_57 := value
              let value_58 := value_57
              let v59 := (UInt256.ofNat 8)
              let v60 := value_58 <<< v59
              let offset_61 := offset_1
              let i_62 := i
              let v63 := offset_61 + i_62
              let v64 ← MemBuffer_byte_at input v63
              let v65 := v64
              let v66 := v60 ||| v65
              let value_67 := v66
              value := value_67
              let v68 := ()
              let v69 := (UInt256.ofNat 1)
              let i_70 := i
              let v71 := i_70 + v69
              let i_72 := v71
              i := i_72
              let v73 := ()
              b13 value i
            else
              let value_74 := value
              let value_75 := value_74
              let len_76 := len_2
              let v77 := len_76 - present
              let v78 := (UInt256.ofNat 8)
              let v79 := v77 * v78
              let v80 := v79
              let v81 := value_75 <<< v80
              return v81
          partial_fixpoint
          b13 value i
      partial_fixpoint
      b10 value i
  if v8 then
    let v9 := (UInt256.ofNat 0)
    let v10 := v9
    b3 value i v10
  else
    let offset_11 := offset_1
    let v12 := «end» - offset_11
    let len_13 := len_2
    let v14 := decide (v12.toNat < len_13.toNat)
    let b6 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun value i v4 => do
      let mut value := value
      let mut i := i
      let v20 := v4
      b3 value i v20
    if v14 then
      let offset_15 := offset_1
      let v16 := «end» - offset_15
      let v17 := v16
      b6 value i v17
    else
      let len_18 := len_2
      let len_19 := len_18
      b6 value i len_19

def MAX_LENGTH : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1024)
  return v0

def FAILURE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  return v0

def bit_length (value : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 256)
  let v2 := IntTy.leadingZeros (.u 256) value
  let v3 ← checkedSub v1 v2
  let v4 ← downcast_truncate_u256_u64 v3
  return v4

def gas_cost_MemBuffer (input : MemBuffer) (base_1 : UInt256) (exponent_2 : UInt256) (modulus_3 : UInt256) : FeM UInt256 := do
  let base_11 := base_1
  let modulus_12 := modulus_3
  let v13 := decide (modulus_12.toNat < base_11.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v4 => do
    let v18 ← downcast_truncate_usize_u64 v4
    let longest := v18
    let v20 := (UInt256.ofNat 7)
    let v21 ← IntTy.add (.u 64) longest v20
    let v22 := (UInt256.ofNat 8)
    let v23 ← checkedDiv v21 v22
    let words := v23
    let v25 := (UInt256.ofNat 32)
    let v26 := decide (v25.toNat < longest.toNat)
    let b6 : UInt256 → FeM UInt256 := fun v5 => do
      let complexity := v5
      let v34 := (UInt256.ofNat 96)
      let base_35 := base_1
      let v36 ← checkedAdd v34 base_35
      let exponent_37 := exponent_2
      let v38 := (UInt256.ofNat 32)
      let v39 := decide (exponent_37.toNat < v38.toNat)
      let b9 : UInt256 → FeM UInt256 := fun v6 => do
        let v44 ← input_word_MemBuffer input v36 v6
        let head := v44
        let v46 ← bit_length head
        let head_bits := v46
        let v48 := (UInt256.ofNat 0)
        let v49 := decide (v48.toNat < head_bits.toNat)
        let b12 : UInt256 → FeM UInt256 := fun v7 => do
          let top_bits := v7
          let exponent_56 := exponent_2
          let v57 := (UInt256.ofNat 32)
          let v58 := decide (v57.toNat < exponent_56.toNat)
          let b15 : UInt256 → FeM UInt256 := fun v8 => do
            let extra := v8
            let v67 := (UInt256.ofNat 16)
            let v68 ← IntTy.mul (.u 64) v67 extra
            let v69 ← IntTy.add (.u 64) v68 top_bits
            let count_70 := v69
            let v71 := (UInt256.ofNat 0)
            let v72 := decide (count_70 = v71)
            let b18 : UInt256 → FeM UInt256 := fun v9 => do
              let v76 ← IntTy.mul (.u 64) complexity v9
              let cost_77 := v76
              let v78 := (UInt256.ofNat 500)
              let v79 := decide (cost_77.toNat < v78.toNat)
              let b21 : UInt256 → FeM UInt256 := fun v10 => do
                return v10
              if v79 then
                let v80 := (UInt256.ofNat 500)
                let v81 := v80
                b21 v81
              else
                let cost_82 := cost_77
                b21 cost_82
            if v72 then
              let v73 := (UInt256.ofNat 1)
              let v74 := v73
              b18 v74
            else
              let count_75 := count_70
              b18 count_75
          if v58 then
            let exponent_59 := exponent_2
            let v60 := (UInt256.ofNat 32)
            let v61 ← checkedSub exponent_59 v60
            let v62 ← downcast_truncate_usize_u64 v61
            let v63 := v62
            b15 v63
          else
            let v64 := (UInt256.ofNat 0)
            let v65 := v64
            b15 v65
        if v49 then
          let v50 := (UInt256.ofNat 1)
          let v51 ← IntTy.sub (.u 64) head_bits v50
          let v52 := v51
          b12 v52
        else
          let v53 := (UInt256.ofNat 0)
          let v54 := v53
          b12 v54
      if v39 then
        let exponent_40 := exponent_2
        let exponent_41 := exponent_40
        b9 exponent_41
      else
        let v42 := (UInt256.ofNat 32)
        let v43 := v42
        b9 v43
    if v26 then
      let v27 := (UInt256.ofNat 2)
      let v28 ← IntTy.mul (.u 64) v27 words
      let v29 ← IntTy.mul (.u 64) v28 words
      let v30 := v29
      b6 v30
    else
      let v31 := (UInt256.ofNat 16)
      let v32 := v31
      b6 v32
  if v13 then
    let base_14 := base_1
    let base_15 := base_14
    b3 base_15
  else
    let modulus_16 := modulus_3
    let modulus_17 := modulus_16
    b3 modulus_17

def input_byte_MemBuffer (input : MemBuffer) (index_1 : UInt256) : FeM UInt256 := do
  let index_3 := index_1
  let v4 ← MemBuffer_len_1 input
  let v5 := decide (index_3.toNat < v4.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  if v5 then
    let v6 ← MemBuffer_byte_at input index_1
    let v7 := v6
    b3 v7
  else
    let v8 := (UInt256.ofNat 0)
    let v9 := v8
    b3 v9

def input_limb_MemBuffer (input : MemBuffer) (offset_1 : UInt256) (len_2 : UInt256) (index_3 : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut byte : UInt256 := default
  let offset_4 := offset_1
  let len_5 := len_2
  let v6 := offset_4 + len_5
  let index_7 := index_3
  let v8 := (UInt256.ofNat 8)
  let v9 := index_7 * v8
  let v10 := v6 - v9
  let «end» := v10
  let index_12 := index_3
  let v13 := (UInt256.ofNat 8)
  let v14 := index_12 * v13
  let v15 := (UInt256.ofNat 8)
  let v16 := v14 + v15
  let len_17 := len_2
  let v18 := decide (v16.toNat ≤ len_17.toNat)
  let b2 : UInt256 → UInt256 → FeM UInt256 := fun value byte => do
    let mut value := value
    let mut byte := byte
    let v25 := ()
    let v26 := v25
    let v27 := (UInt256.ofNat 0)
    let value_28 := v27
    value := value_28
    let v29 := (UInt256.ofNat 8)
    let byte_30 := v29
    byte := byte_30
    let rec b5 (value : UInt256) (byte : UInt256) : FeM UInt256 := do
      let mut value := value
      let mut byte := byte
      let byte_31 := byte
      let byte_32 := byte_31
      let v33 := (UInt256.ofNat 0)
      let v34 := decide (v33.toNat < byte_32.toNat)
      if v34 then
        let v35 := (UInt256.ofNat 1)
        let byte_36 := byte
        let v37 := byte_36 - v35
        let byte_38 := v37
        byte := byte_38
        let v39 := ()
        let index_40 := index_3
        let v41 := (UInt256.ofNat 8)
        let v42 := index_40 * v41
        let byte_43 := byte
        let v44 := v42 + byte_43
        let position := v44
        let value_46 := value
        let value_47 := value_46
        let v48 := (UInt256.ofNat 8)
        let v49 := IntTy.shl (.u 64) value_47 v48
        let value_50 := v49
        value := value_50
        let v51 := ()
        let len_52 := len_2
        let v53 := decide (position.toNat < len_52.toNat)
        let b10 : UInt256 → UInt256 → FeM UInt256 := fun value byte => do
          let mut value := value
          let mut byte := byte
          b5 value byte
        if v53 then
          let offset_54 := offset_1
          let len_55 := len_2
          let v56 := offset_54 + len_55
          let v57 := (UInt256.ofNat 1)
          let v58 := v56 - v57
          let v59 := v58 - position
          let v60 ← input_byte_MemBuffer input v59
          let v61 := v60
          let value_62 := value
          let v63 := value_62 ||| v61
          let value_64 := v63
          value := value_64
          let v65 := ()
          let v66 := v65
          b10 value byte
        else
          let v67 := ()
          let v68 := v67
          b10 value byte
      else
        let value_69 := value
        return value_69
    partial_fixpoint
    b5 value byte
  if v18 then
    let v19 ← MemBuffer_len_1 input
    let v20 := decide («end».toNat ≤ v19.toNat)
    if v20 then
      let v21 := (UInt256.ofNat 8)
      let v22 := «end» - v21
      let v23 ← MemBuffer_lane_at input v22
      let v24 := IntTy.swapBytes (.u 64) v23
      return v24
    else
      b2 value byte
  else
    b2 value byte

def input_limbs_MemBuffer (input : MemBuffer) (offset : UInt256) (len_2 : UInt256) (count_3 : UInt256) : FeM Option_MemArray_u64 := do
  let mut limbs : MemArray_u64 := default
  let mut r1 : Range_Known_0_Unknown := default
  let v6 ← MemArray_try_new_zeroed_u64 count_3
  let v7 := match v6 with | .Some .. => true | _ => false
  if v7 then
    let v8 := v6.Some_0
    let limbs_9 := v8
    limbs := limbs_9
    let len_10 := len_2
    let v11 := (UInt256.ofNat 7)
    let v12 ← checkedAdd len_10 v11
    let v13 := (UInt256.ofNat 8)
    let v14 ← checkedDiv v12 v13
    let present_15 := v14
    let count_16 := count_3
    let v17 := decide (present_15.toNat < count_16.toNat)
    let b5 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_MemArray_u64 := fun limbs r1 v4 => do
      let mut limbs := limbs
      let mut r1 := r1
      let read := v4
      let v22 := (UInt256.ofNat 0)
      let v23 := ()
      let v24 := (Range_Known_0_Unknown.mk v23 read)
      r1 := v24
      let v26 ← Range_start_0 r1
      let v27 := v26
      let rec b6 (limbs : MemArray_u64) (r1 : Range_Known_0_Unknown) (v5 : UInt256) : FeM Option_MemArray_u64 := do
        let mut limbs := limbs
        let mut r1 := r1
        let v29 ← Range_has_0 r1 v5
        if v29 then
          let v31 ← Range_at_0 r1 v5
          let i := v31
          let v33 ← input_limb_MemBuffer input offset len_2 i
          let (v35, v35_k) ← MemArray_index_mut_u64 limbs i
          mstoreInt (.u 64) v35 v33
          let v35_0 ← v35_k v35
          limbs := v35_0
          let v36 := ()
          let v38 ← Range_next_0 r1 v5
          let v39 := v38
          b6 limbs r1 v39
        else
          let limbs_40 := limbs
          let limbs_41 := limbs_40
          let v42 := Option_MemArray_u64.Some limbs_41
          return v42
      partial_fixpoint
      b6 limbs r1 v27
    if v17 then
      let present_18 := present_15
      b5 limbs r1 present_18
    else
      let count_19 := count_3
      let count_20 := count_19
      b5 limbs r1 count_20
  else
    let v43 := Option_MemArray_u64.None
    match (← pure v6) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v6.Some_0
      pure ()
    | _ => pure ()
    return v43

def Exponent_try_read_MemBuffer (input : MemBuffer) (start : UInt256) (len_2 : UInt256) : FeM Option_Exponent := do
  let mut limbs : MemArray_u64 := default
  let mut count : UInt256 := default
  let len_4 := len_2
  let v5 := (UInt256.ofNat 8)
  let v6 ← checkedMul len_4 v5
  let total_7 := v6
  let len_8 := len_2
  let v9 := (UInt256.ofNat 7)
  let v10 ← checkedAdd len_8 v9
  let v11 := (UInt256.ofNat 8)
  let v12 ← checkedDiv v10 v11
  let v13 ← input_limbs_MemBuffer input start len_2 v12
  let v14 := match v13 with | .Some .. => true | _ => false
  if v14 then
    let v15 := v13.Some_0
    let limbs_16 := v15
    limbs := limbs_16
    let v18 ← MemArray_len_u64 limbs
    let count_19 := v18
    count := count_19
    let rec b3 (limbs : MemArray_u64) (count : UInt256) : FeM Option_Exponent := do
      let mut limbs := limbs
      let mut count := count
      let count_20 := count
      let count_21 := count_20
      let v22 := (UInt256.ofNat 0)
      let v23 := decide (v22.toNat < count_21.toNat)
      let b5 : MemArray_u64 → UInt256 → FeM Option_Exponent := fun limbs count => do
        let mut limbs := limbs
        let mut count := count
        let count_38 := count
        let count_39 := count_38
        let v40 := (UInt256.ofNat 0)
        let v41 := decide (count_39 = v40)
        let b9 : MemArray_u64 → UInt256 → UInt256 → FeM Option_Exponent := fun limbs count v3 => do
          let mut limbs := limbs
          let mut count := count
          let top := v3
          let limbs_68 := limbs
          let limbs_69 := limbs_68
          let v70 := (Exponent.mk limbs_69 top total_7)
          let v71 := Option_Exponent.Some v70
          return v71
        if v41 then
          let total_42 := total_7
          b9 limbs count total_42
        else
          let v43 := (UInt256.ofNat 255)
          let count_44 := count
          let count_45 := count_44
          let v46 := (UInt256.ofNat 1)
          let v47 ← checkedSub count_45 v46
          let v49 ← MemArray_index_ref_u64 limbs v47
          let v50 ← mloadInt (.u 64) v49
          let v51 := v50
          let v52 := IntTy.leadingZeros (.u 256) v51
          let v53 ← checkedSub v43 v52
          let v54 ← downcast_truncate_u256_usize v53
          let highest := v54
          let v56 := (UInt256.ofNat 1)
          let v57 ← checkedSub total_7 v56
          let count_58 := count
          let count_59 := count_58
          let v60 := (UInt256.ofNat 1)
          let v61 ← checkedSub count_59 v60
          let v62 := (UInt256.ofNat 64)
          let v63 ← checkedMul v61 v62
          let v64 ← checkedAdd v63 highest
          let v65 ← checkedSub v57 v64
          let v66 := v65
          b9 limbs count v66
      if v23 then
        let count_24 := count
        let count_25 := count_24
        let v26 := (UInt256.ofNat 1)
        let v27 ← checkedSub count_25 v26
        let v29 ← MemArray_index_ref_u64 limbs v27
        let v30 ← mloadInt (.u 64) v29
        let v31 := (UInt256.ofNat 0)
        let v32 := decide (v30 = v31)
        if v32 then
          let v33 := (UInt256.ofNat 1)
          let count_34 := count
          let v35 ← checkedSub count_34 v33
          let count_36 := v35
          count := count_36
          let v37 := ()
          b3 limbs count
        else
          b5 limbs count
      else
        b5 limbs count
    partial_fixpoint
    b3 limbs count
  else
    let v72 := Option_Exponent.None
    match (← pure v13) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v13.Some_0
      pure ()
    | _ => pure ()
    return v72

def LIMB_BASE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 18446744073709551616)
  return v0

def reciprocal_2by1 (d_0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 340282366920938463463374607431768211455)
  let all := v1
  let d_3 := d_0
  let v4 := d_3
  let v5 ← checkedDiv all v4
  let v6 ← LIMB_BASE
  let v7 ← IntTy.sub (.u 128) v5 v6
  let v8 ← downcast_truncate_u128_u64 v7
  return v8

def reciprocal_3by2 (d1_0 : UInt256) (d0_1 : UInt256) : FeM UInt256 := do
  let mut v : UInt256 := default
  let mut p : UInt256 := default
  let v3 ← reciprocal_2by1 d1_0
  let v_4 := v3
  v := v_4
  let d1_5 := d1_0
  let v_6 := v
  let v7 := IntTy.mulW (.u 64) d1_5 v_6
  let d0_8 := d0_1
  let v9 := IntTy.addW (.u 64) v7 d0_8
  let p_10 := v9
  p := p_10
  let p_11 := p
  let p_12 := p_11
  let d0_13 := d0_1
  let v14 := decide (p_12.toNat < d0_13.toNat)
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
    let mut v := v
    let mut p := p
    let v_45 := v
    let v46 := v_45
    let d0_47 := d0_1
    let v48 := d0_47
    let v49 := IntTy.mulW (.u 128) v46 v48
    let t := v49
    let v51 := (UInt256.ofNat 64)
    let v52 := t >>> v51
    let v53 ← downcast_truncate_u128_u64 v52
    let t1 := v53
    let v55 ← downcast_truncate_u128_u64 t
    let t0 := v55
    let p_57 := p
    let v58 := IntTy.addW (.u 64) p_57 t1
    let p_59 := v58
    p := p_59
    let v60 := ()
    let p_61 := p
    let p_62 := p_61
    let v63 := decide (p_62.toNat < t1.toNat)
    let b9 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
      let mut v := v
      let mut p := p
      let v_90 := v
      return v_90
    if v63 then
      let v64 := (UInt256.ofNat 1)
      let v_65 := v
      let v66 := IntTy.subW (.u 64) v_65 v64
      let v_67 := v66
      v := v_67
      let v68 := ()
      let p_69 := p
      let p_70 := p_69
      let d1_71 := d1_0
      let v72 := decide (d1_71.toNat < p_70.toNat)
      let b12 : UInt256 → UInt256 → Unit → FeM UInt256 := fun v p v2 => do
        let mut v := v
        let mut p := p
        let v87 := v2
        b9 v p
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
        let mut v := v
        let mut p := p
        let v81 := (UInt256.ofNat 1)
        let v_82 := v
        let v83 := IntTy.subW (.u 64) v_82 v81
        let v_84 := v83
        v := v_84
        let v85 := ()
        let v86 := v85
        b12 v p v86
      if v72 then
        b10 v p
      else
        let p_73 := p
        let p_74 := p_73
        let d1_75 := d1_0
        let v76 := decide (p_74 = d1_75)
        let b11 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
          let mut v := v
          let mut p := p
          let v79 := ()
          let v80 := v79
          b12 v p v80
        if v76 then
          let d0_77 := d0_1
          let v78 := decide (d0_77.toNat ≤ t0.toNat)
          if v78 then
            b10 v p
          else
            b11 v p
        else
          b11 v p
    else
      let v88 := ()
      let v89 := v88
      b9 v p
  if v14 then
    let v15 := (UInt256.ofNat 1)
    let v_16 := v
    let v17 := IntTy.subW (.u 64) v_16 v15
    let v_18 := v17
    v := v_18
    let v19 := ()
    let p_20 := p
    let p_21 := p_20
    let d1_22 := d1_0
    let v23 := decide (d1_22.toNat ≤ p_21.toNat)
    let b6 : UInt256 → UInt256 → FeM UInt256 := fun v p => do
      let mut v := v
      let mut p := p
      let d1_37 := d1_0
      let p_38 := p
      let v39 := IntTy.subW (.u 64) p_38 d1_37
      let p_40 := v39
      p := p_40
      let v41 := ()
      let v42 := v41
      b3 v p
    if v23 then
      let v24 := (UInt256.ofNat 1)
      let v_25 := v
      let v26 := IntTy.subW (.u 64) v_25 v24
      let v_27 := v26
      v := v_27
      let v28 := ()
      let d1_29 := d1_0
      let p_30 := p
      let v31 := IntTy.subW (.u 64) p_30 d1_29
      let p_32 := v31
      p := p_32
      let v33 := ()
      let v34 := v33
      b6 v p
    else
      let v35 := ()
      let v36 := v35
      b6 v p
  else
    let v43 := ()
    let v44 := v43
    b3 v p

def Modulus_try_new (value : MemArray_u64) (len_1 : UInt256) : FeM Option_Modulus := do
  let mut limbs : MemArray_u64 := default
  let mut r1 : Range_Known_0_Unknown := default
  let v6 ← downcast_truncate_u64_usize len_1
  let count := v6
  let len_8 := len_1
  let v9 := (UInt256.ofNat 0)
  let v10 := decide (v9.toNat < len_8.toNat)
  let b3 : MemArray_u64 → Range_Known_0_Unknown → Bool → FeM Option_Modulus := fun limbs r1 v2 => do
    let mut limbs := limbs
    let mut r1 := r1
    let v25 ← «assert» v2
    let len_26 := len_1
    let v27 := (UInt256.ofNat 1)
    let v28 ← IntTy.sub (.u 64) len_26 v27
    let v29 ← downcast_truncate_u64_usize v28
    let v30 ← MemArray_index_ref_u64 value v29
    let v31 ← mloadInt (.u 64) v30
    let v32 := v31
    let v33 := IntTy.leadingZeros (.u 256) v32
    let v34 := (UInt256.ofNat 192)
    let v35 ← checkedSub v33 v34
    let v36 ← downcast_truncate_u256_u64 v35
    let shift := v36
    let v38 ← downcast_truncate_u64_usize len_1
    let v39 ← MemArray_try_new_zeroed_u64 v38
    let v40 := match v39 with | .Some .. => true | _ => false
    if v40 then
      let v41 := v39.Some_0
      let limbs_42 := v41
      limbs := limbs_42
      let v43 := (UInt256.ofNat 0)
      let v44 ← downcast_truncate_u64_usize len_1
      let v45 := ()
      let v46 := (Range_Known_0_Unknown.mk v45 v44)
      r1 := v46
      let v48 ← Range_start_0 r1
      let v49 := v48
      let rec b8 (limbs : MemArray_u64) (r1 : Range_Known_0_Unknown) (v3 : UInt256) : FeM Option_Modulus := do
        let mut limbs := limbs
        let mut r1 := r1
        let v51 ← Range_has_0 r1 v3
        if v51 then
          let v53 ← Range_at_0 r1 v3
          let i := v53
          let v55 ← MemArray_index_ref_u64 value i
          let v56 ← mloadInt (.u 64) v55
          let v57 := IntTy.shl (.u 64) v56 shift
          let high := v57
          let v59 := (UInt256.ofNat 0)
          let v60 := decide (shift = v59)
          let b14 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_Modulus := fun limbs r1 v4 => do
            let mut limbs := limbs
            let mut r1 := r1
            let low := v4
            let v74 := high ||| low
            let (v76, v76_k) ← MemArray_index_mut_u64 limbs i
            mstoreInt (.u 64) v76 v74
            let v76_0 ← v76_k v76
            limbs := v76_0
            let v77 := ()
            let v79 ← Range_next_0 r1 v3
            let v80 := v79
            b8 limbs r1 v80
          let b12 : MemArray_u64 → Range_Known_0_Unknown → FeM Option_Modulus := fun limbs r1 => do
            let mut limbs := limbs
            let mut r1 := r1
            let v71 := (UInt256.ofNat 0)
            let v72 := v71
            b14 limbs r1 v72
          if v60 then
            b12 limbs r1
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := decide (i = v61)
            if v62 then
              b12 limbs r1
            else
              let v63 := (UInt256.ofNat 1)
              let v64 ← checkedSub i v63
              let v65 ← MemArray_index_ref_u64 value v64
              let v66 ← mloadInt (.u 64) v65
              let v67 := (UInt256.ofNat 64)
              let v68 ← IntTy.sub (.u 64) v67 shift
              let v69 := v66 >>> v68
              let v70 := v69
              b14 limbs r1 v70
        else
          let v81 := (UInt256.ofNat 1)
          let v82 := decide (count = v81)
          let b18 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_Modulus := fun limbs r1 v5 => do
            let mut limbs := limbs
            let mut r1 := r1
            let reciprocal := v5
            let limbs_102 := limbs
            let limbs_103 := limbs_102
            let len_104 := len_1
            let v105 := (Modulus.mk limbs_103 len_104 shift reciprocal)
            let v106 := Option_Modulus.Some v105
            return v106
          if v82 then
            let v83 := (UInt256.ofNat 0)
            let v85 ← MemArray_index_ref_u64 limbs v83
            let v86 ← mloadInt (.u 64) v85
            let v87 ← reciprocal_2by1 v86
            let v88 := v87
            b18 limbs r1 v88
          else
            let v89 := (UInt256.ofNat 1)
            let v90 ← checkedSub count v89
            let v92 ← MemArray_index_ref_u64 limbs v90
            let v93 := (UInt256.ofNat 2)
            let v94 ← checkedSub count v93
            let v96 ← MemArray_index_ref_u64 limbs v94
            let v97 ← mloadInt (.u 64) v92
            let v98 ← mloadInt (.u 64) v96
            let v99 ← reciprocal_3by2 v97 v98
            let v100 := v99
            b18 limbs r1 v100
      partial_fixpoint
      b8 limbs r1 v49
    else
      let v107 := Option_Modulus.None
      match (← pure v39) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v39.Some_0
        pure ()
      | _ => pure ()
      return v107
  let b2 : MemArray_u64 → Range_Known_0_Unknown → FeM Option_Modulus := fun limbs r1 => do
    let mut limbs := limbs
    let mut r1 := r1
    let v23 := false
    let v24 := v23
    b3 limbs r1 v24
  if v10 then
    let v11 ← MemArray_len_u64 value
    let v12 := decide (count.toNat ≤ v11.toNat)
    if v12 then
      let len_13 := len_1
      let v14 := (UInt256.ofNat 1)
      let v15 ← IntTy.sub (.u 64) len_13 v14
      let v16 ← downcast_truncate_u64_usize v15
      let v17 ← MemArray_index_ref_u64 value v16
      let v18 ← mloadInt (.u 64) v17
      let v19 := (UInt256.ofNat 0)
      let v20 := decide (v18 ≠ v19)
      if v20 then
        let v21 := true
        let v22 := v21
        b3 limbs r1 v22
      else
        b2 limbs r1
    else
      b2 limbs r1
  else
    b2 limbs r1

def Wide_of (limbs : MemArray_u64) (n_1 : UInt256) : FeM Wide := do
  let v5 := (UInt256.ofNat 0)
  let v6 ← MemArray_index_ref_u64 limbs v5
  let v7 ← mloadInt (.u 64) v6
  let n_8 := n_1
  let v9 := (UInt256.ofNat 1)
  let v10 := decide (v9.toNat < n_8.toNat)
  let b3 : UInt256 → FeM Wide := fun v2 => do
    let n_17 := n_1
    let v18 := (UInt256.ofNat 2)
    let v19 := decide (v18.toNat < n_17.toNat)
    let b6 : UInt256 → FeM Wide := fun v3 => do
      let n_26 := n_1
      let v27 := (UInt256.ofNat 3)
      let v28 := decide (v27.toNat < n_26.toNat)
      let b9 : UInt256 → FeM Wide := fun v4 => do
        let v35 := (Wide.mk v7 v2 v3 v4)
        return v35
      if v28 then
        let v29 := (UInt256.ofNat 3)
        let v30 ← MemArray_index_ref_u64 limbs v29
        let v31 ← mloadInt (.u 64) v30
        let v32 := v31
        b9 v32
      else
        let v33 := (UInt256.ofNat 0)
        let v34 := v33
        b9 v34
    if v19 then
      let v20 := (UInt256.ofNat 2)
      let v21 ← MemArray_index_ref_u64 limbs v20
      let v22 ← mloadInt (.u 64) v21
      let v23 := v22
      b6 v23
    else
      let v24 := (UInt256.ofNat 0)
      let v25 := v24
      b6 v25
  if v10 then
    let v11 := (UInt256.ofNat 1)
    let v12 ← MemArray_index_ref_u64 limbs v11
    let v13 ← mloadInt (.u 64) v12
    let v14 := v13
    b3 v14
  else
    let v15 := (UInt256.ofNat 0)
    let v16 := v15
    b3 v16

def window_width (bits_0 : UInt256) : FeM UInt256 := do
  let mut best : UInt256 := default
  let mut best_cost : UInt256 := default
  let mut w : UInt256 := default
  let v1 := (UInt256.ofNat 1)
  let best_2 := v1
  best := best_2
  let bits_3 := bits_0
  let v4 := (UInt256.ofNat 2)
  let v5 ← checkedDiv bits_3 v4
  let best_cost_6 := v5
  best_cost := best_cost_6
  let v7 := (UInt256.ofNat 2)
  let w_8 := v7
  w := w_8
  let rec b1 (best : UInt256) (best_cost : UInt256) (w : UInt256) : FeM UInt256 := do
    let mut best := best
    let mut best_cost := best_cost
    let mut w := w
    let w_9 := w
    let w_10 := w_9
    let v11 := (UInt256.ofNat 7)
    let v12 := decide (w_10.toNat ≤ v11.toNat)
    if v12 then
      let v13 := (UInt256.ofNat 1)
      let w_14 := w
      let w_15 := w_14
      let v16 := (UInt256.ofNat 1)
      let v17 ← checkedSub w_15 v16
      let v18 := v13 <<< v17
      let bits_19 := bits_0
      let w_20 := w
      let w_21 := w_20
      let v22 := (UInt256.ofNat 1)
      let v23 ← checkedAdd w_21 v22
      let v24 ← checkedDiv bits_19 v23
      let v25 ← checkedAdd v18 v24
      let cost_26 := v25
      let best_cost_27 := best_cost
      let v28 := decide (cost_26.toNat < best_cost_27.toNat)
      let b6 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun best best_cost w => do
        let mut best := best
        let mut best_cost := best_cost
        let mut w := w
        let v38 := (UInt256.ofNat 1)
        let w_39 := w
        let v40 ← checkedAdd w_39 v38
        let w_41 := v40
        w := w_41
        let v42 := ()
        b1 best best_cost w
      if v28 then
        let w_29 := w
        let w_30 := w_29
        let w_31 := w_30
        best := w_31
        let v32 := ()
        let cost_33 := cost_26
        best_cost := cost_33
        let v34 := ()
        let v35 := v34
        b6 best best_cost w
      else
        let v36 := ()
        let v37 := v36
        b6 best best_cost w
    else
      let best_43 := best
      return best_43
  partial_fixpoint
  b1 best best_cost w

def Small_double_1_k2 (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (borrow_49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb s3 v50 borrow_49
  let v52 := v51.fst
  let d3 := v52
  let v54 := v51.snd
  let borrow_55 := v54
  let v56 := x.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 borrow_55
  let v61 := v60.snd
  let borrow_62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (borrow_62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk s0 s1 s2 s3)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk d0 d1 d2 d3)
    let v68 := v67
    b3 v68

def Small_double_1_k1 (self : Small_1) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let s3 := v29
  let v31 := self.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb s0 v31 v32
  let v34 := v33.fst
  let d0 := v34
  let v36 := v33.snd
  let borrow_37 := v36
  let v38 := self.m.l1
  let v39 ← sbb s1 v38 borrow_37
  let v40 := v39.fst
  let d1 := v40
  let v42 := v39.snd
  let borrow_43 := v42
  let v44 := self.m.l2
  let v45 ← sbb s2 v44 borrow_43
  let v46 := v45.fst
  let d2 := v46
  let v48 := v45.snd
  let borrow_49 := v48
  let v50 := self.m.l3
  Small_double_1_k2 x s0 s1 s2 s3 d0 d1 d2 borrow_49 v50

def Small_double_1 (self : Small_1) (x : Wide) : FeM Wide := do
  let v3 := x.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let s0 := v5
  let v7 := x.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := x.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let s1 := v13
  let v15 := x.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := x.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let s2 := v21
  let v23 := x.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := x.l2
  Small_double_1_k1 self x s0 s1 s2 v25 v26

def mac (a_0 : UInt256) (b_1 : UInt256) (c_2 : UInt256) (carry_3 : UInt256) : FeM (UInt256 × UInt256) := do
  let a_4 := a_0
  let v5 := a_4
  let b_6 := b_1
  let v7 := b_6
  let v8 ← IntTy.mul (.u 128) v5 v7
  let c_9 := c_2
  let v10 := c_9
  let v11 ← IntTy.add (.u 128) v8 v10
  let carry_12 := carry_3
  let v13 := carry_12
  let v14 ← IntTy.add (.u 128) v11 v13
  let t := v14
  let v16 ← downcast_truncate_u128_u64 t
  let v17 := (UInt256.ofNat 64)
  let v18 := t >>> v17
  let v19 ← downcast_truncate_u128_u64 v18
  let v20 := (v16, v19)
  return v20

def Small_row2_1_k1 (self : Small_1) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (t3 : UInt256) (m : UInt256) : FeM (Wide × UInt256) := do
  let v29 := self.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac m v29 t0 v30
  let v32 := v31.snd
  let c_33 := v32
  let v34 := self.m.l1
  let v35 ← mac m v34 t1 c_33
  let v36 := v35.fst
  let r0 := v36
  let v38 := v35.snd
  let c_39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc t2 v40 c_39
  let v42 := v41.fst
  let r1 := v42
  let v44 := v41.snd
  let c_45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk r0 r1 v46 v47)
  let v49 := IntTy.addW (.u 64) t3 c_45
  let v50 := (v48, v49)
  return v50

def Small_row2_1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc top c_19 v20
  let v22 := v21.fst
  let t2 := v22
  let v24 := v21.snd
  let t3 := v24
  let v26 := self.inverse
  let v27 := IntTy.mulW (.u 64) t0 v26
  let m := v27
  Small_row2_1_k1 self t0 t1 t2 t3 m

def Small_row3_1_k2 (t3 : UInt256) (t4 : UInt256) (r0 : UInt256) (r1 : UInt256) (c_52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc t3 v53 c_52
  let v55 := v54.fst
  let r2 := v55
  let v57 := v54.snd
  let c_58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk r0 r1 r2 v59)
  let v61 := IntTy.addW (.u 64) t4 c_58
  let v62 := (v60, v61)
  return v62

def Small_row3_1_k1 (self : Small_1) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let t3 := v29
  let v31 := v28.snd
  let t4 := v31
  let v33 := self.inverse
  let v34 := IntTy.mulW (.u 64) t0 v33
  let m := v34
  let v36 := self.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac m v36 t0 v37
  let v39 := v38.snd
  let c_40 := v39
  let v41 := self.m.l1
  let v42 ← mac m v41 t1 c_40
  let v43 := v42.fst
  let r0 := v43
  let v45 := v42.snd
  let c_46 := v45
  let v47 := self.m.l2
  let v48 ← mac m v47 t2 c_46
  let v49 := v48.fst
  let r1 := v49
  let v51 := v48.snd
  let c_52 := v51
  Small_row3_1_k2 t3 t4 r0 r1 c_52

def Small_row3_1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc top c_26 v27
  Small_row3_1_k1 self t0 t1 t2 v28

def Small_row4_1_k2 (self : Small_1) (t2 : UInt256) (t3 : UInt256) (t4 : UInt256) (t5 : UInt256) (m : UInt256) (r0 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let c_53 := v52
  let v54 := self.m.l2
  let v55 ← mac m v54 t2 c_53
  let v56 := v55.fst
  let r1 := v56
  let v58 := v55.snd
  let c_59 := v58
  let v60 := self.m.l3
  let v61 ← mac m v60 t3 c_59
  let v62 := v61.fst
  let r2 := v62
  let v64 := v61.snd
  let c_65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc t4 v66 c_65
  let v68 := v67.fst
  let r3 := v68
  let v70 := v67.snd
  let c_71 := v70
  let v72 := (Wide.mk r0 r1 r2 r3)
  let v73 := IntTy.addW (.u 64) t5 c_71
  let v74 := (v72, v73)
  return v74

def Small_row4_1_k1 (self : Small_1) (b : UInt256) (top : UInt256) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (c_26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 b v28 c_26
  let v30 := v29.fst
  let t3 := v30
  let v32 := v29.snd
  let c_33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc top c_33 v34
  let v36 := v35.fst
  let t4 := v36
  let v38 := v35.snd
  let t5 := v38
  let v40 := self.inverse
  let v41 := IntTy.mulW (.u 64) t0 v40
  let m := v41
  let v43 := self.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac m v43 t0 v44
  let v46 := v45.snd
  let c_47 := v46
  let v48 := self.m.l1
  let v49 ← mac m v48 t1 c_47
  let v50 := v49.fst
  let r0 := v50
  let v52 := v49.snd
  Small_row4_1_k2 self t2 t3 t4 t5 m r0 v52

def Small_row4_1 (self : Small_1) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := a.l3
  let v28 := t.l3
  Small_row4_1_k1 self b top t0 t1 t2 c_26 v27 v28

def Small_mul_1 (self : Small_1) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let zero_13 := v12
  zero := zero_13
  let v14 := (UInt256.ofNat 1)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := a.l0
    let v18 := v17
    let v19 := b.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let t_22 := v21
    let v23 ← downcast_truncate_u128_u64 t_22
    let low := v23
    let v25 := self.inverse
    let v26 := IntTy.mulW (.u 64) low v25
    let q := v26
    let v28 := q
    let v29 := self.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let product := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := t_22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := product >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (low ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let sum_45 := v44
      let v46 := self.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ sum_45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v4 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        let reduced := v4
        let v55 ← downcast_truncate_u128_u64 reduced
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := self.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) sum_45 v50
        let v52 := v51
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v52
      else
        let sum_53 := sum_45
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum_53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 1)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := b.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_1 self a v65 zero v66
      let v69 := v68.fst
      let t_70 := v69
      t_r1 := t_70
      let v71 := v68.snd
      let top_72 := v71
      let v73 := b.l1
      let v75 ← Small_row2_1 self a v73 t_r1 top_72
      let v76 := v75.fst
      let t_77 := v76
      let v78 := v75.snd
      let top_79 := v78
      let v80 := t_77.l0
      let v81 := self.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let d0_85 := v84
      let v86 := v83.snd
      let borrow_87 := v86
      let v88 := t_77.l1
      let v89 := self.m.l1
      let v90 ← sbb v88 v89 borrow_87
      let v91 := v90.fst
      let d1_92 := v91
      let v93 := v90.snd
      let borrow_94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb top_79 v95 borrow_94
      let v97 := v96.snd
      let borrow_98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (borrow_98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      if v100 then
        let t_101 := t_77
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk d0_85 d1_92 v102 v103)
        let v105 := v104
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 1)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := b.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_1 self a v111 zero v112
        let v115 := v114.fst
        let t_116 := v115
        t_r2 := t_116
        let v117 := v114.snd
        let top_118 := v117
        let v119 := b.l1
        let v121 ← Small_row3_1 self a v119 t_r2 top_118
        let v122 := v121.fst
        let t_123 := v122
        t_r3 := t_123
        let v124 := v121.snd
        let top_125 := v124
        let v126 := b.l2
        let v128 ← Small_row3_1 self a v126 t_r3 top_125
        let v129 := v128.fst
        let t_130 := v129
        let v131 := v128.snd
        let top_132 := v131
        let v133 := t_130.l0
        let v134 := self.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let d0_138 := v137
        let v139 := v136.snd
        let borrow_140 := v139
        let v141 := t_130.l1
        let v142 := self.m.l1
        let v143 ← sbb v141 v142 borrow_140
        let v144 := v143.fst
        let d1_145 := v144
        let v146 := v143.snd
        let borrow_147 := v146
        let v148 := t_130.l2
        let v149 := self.m.l2
        let v150 ← sbb v148 v149 borrow_147
        let v151 := v150.fst
        let d2_152 := v151
        let v153 := v150.snd
        let borrow_154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb top_132 v155 borrow_154
        let v157 := v156.snd
        let borrow_158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (borrow_158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        if v160 then
          let t_161 := t_130
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk d0_138 d1_145 d2_152 v162)
          let v164 := v163
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := b.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_1 self a v167 zero v168
        let v171 := v170.fst
        let t_172 := v171
        t_r4 := t_172
        let v173 := v170.snd
        let top_174 := v173
        let v175 := b.l1
        let v177 ← Small_row4_1 self a v175 t_r4 top_174
        let v178 := v177.fst
        let t_179 := v178
        t_r5 := t_179
        let v180 := v177.snd
        let top_181 := v180
        let v182 := b.l2
        let v184 ← Small_row4_1 self a v182 t_r5 top_181
        let v185 := v184.fst
        let t_186 := v185
        t_r6 := t_186
        let v187 := v184.snd
        let top_188 := v187
        let v189 := b.l3
        let v191 ← Small_row4_1 self a v189 t_r6 top_188
        let v192 := v191.fst
        let t_193 := v192
        let v194 := v191.snd
        let top_195 := v194
        let v196 := t_193.l0
        let v197 := self.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let d0_201 := v200
        let v202 := v199.snd
        let borrow_203 := v202
        let v204 := t_193.l1
        let v205 := self.m.l1
        let v206 ← sbb v204 v205 borrow_203
        let v207 := v206.fst
        let d1_208 := v207
        let v209 := v206.snd
        let borrow_210 := v209
        let v211 := t_193.l2
        let v212 := self.m.l2
        let v213 ← sbb v211 v212 borrow_210
        let v214 := v213.fst
        let d2_215 := v214
        let v216 := v213.snd
        let borrow_217 := v216
        let v218 := t_193.l3
        let v219 := self.m.l3
        let v220 ← sbb v218 v219 borrow_217
        let v221 := v220.fst
        let d3 := v221
        let v223 := v220.snd
        let borrow_224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb top_195 v225 borrow_224
        let v227 := v226.snd
        let borrow_228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (borrow_228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        if v230 then
          let t_231 := t_193
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_231
        else
          let v232 := (Wide.mk d0_201 d1_208 d2_215 d3)
          let v233 := v232
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v233

def Small_r2_1 (self : Small_1) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 1)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v1 => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let top := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := top
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let shift := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 shift
    let bit_38 := v37
    let v39 := (UInt256.ofNat 1)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 1)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 1)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 1)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let x_64 := v63
            x := x_64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 shift
            let v67 ← downcast_truncate_u64_usize v66
            let doublings := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 1)
            let v71 := doublings + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let x_81 := x
                let x_82 := x_81
                r2 := x_82
                let v84 ← Small_double_1 self r2
                let x_85 := v84
                x := x_85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 x r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let x_101 := x
                    let x_102 := x_101
                    r4 := x_102
                    let x_103 := x
                    let x_104 := x_103
                    r5 := x_104
                    let v107 ← Small_mul_1 self r4 r5
                    let x_108 := v107
                    x := x_108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 x r1 r2 r3 r4 r5 v112
                  else
                    let x_113 := x
                    return x_113
                partial_fixpoint
                b26 x r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 x r1 r2 r3 r4 r5 v76
          if v59 then
            let bit_60 := bit_38
            b21 x r1 r2 r3 r4 r5 bit_60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 x r1 r2 r3 r4 r5 v62
        if v53 then
          let bit_54 := bit_38
          b18 x r1 r2 r3 r4 r5 bit_54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 x r1 r2 r3 r4 r5 v56
      if v47 then
        let bit_48 := bit_38
        b15 x r1 r2 r3 r4 r5 bit_48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 x r1 r2 r3 r4 r5 v50
    if v41 then
      let bit_42 := bit_38
      b12 x r1 r2 r3 r4 r5 bit_42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 x r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := self.m.l0
    let v14 := v13
    b3 x r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 1)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 x r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := self.m.l1
      let v19 := v18
      b6 x r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 1)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 x r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := self.m.l2
        let v24 := v23
        b9 x r1 r2 r3 r4 r5 v24
      else
        let v25 := self.m.l3
        let v26 := v25
        b9 x r1 r2 r3 r4 r5 v26

def limb (limbs : UInt256) (index_1 : UInt256) : FeM UInt256 := do
  let index_2 := index_1
  let v3 := (UInt256.ofNat 8)
  let v4 := index_2 * v3
  let v5 := limbs + v4
  return v5

def Exponent_bit (self : Exponent) (index_1 : UInt256) : FeM UInt256 := do
  let v2 := self.total
  let v3 := (UInt256.ofNat 1)
  let v4 := v2 - v3
  let index_5 := index_1
  let v6 := v4 - index_5
  let position := v6
  let v8 := (UInt256.ofNat 63)
  let v9 := position &&& v8
  let v10 ← downcast_truncate_usize_u64 v9
  let shift := v10
  let v14 ← MemArray_ptr_u64 self.limbs
  let v15 := (UInt256.ofNat 6)
  let v16 := position >>> v15
  let v17 ← limb v14 v16
  let v18 ← mloadInt (.u 64) v17
  let v19 := v18 >>> shift
  let v20 := (UInt256.ofNat 1)
  let v21 := v19 &&& v20
  let v22 ← downcast_truncate_u64_usize v21
  return v22

def Exponent_window (self : Exponent) (index_1 : UInt256) (width_2 : UInt256) : FeM Window := do
  let mut «end» : UInt256 := default
  let mut digit : UInt256 := default
  let mut r2 : Range_Unknown_Unknown := default
  let v5 ← Exponent_bit self index_1
  let v6 := (UInt256.ofNat 0)
  let v7 := decide (v5 = v6)
  if v7 then
    let index_8 := index_1
    let v9 := (UInt256.ofNat 1)
    let v10 := index_8 + v9
    let v11 := (UInt256.ofNat 0)
    let v12 := false
    let v13 := (Window.mk v10 v11 v12)
    return v13
  else
    let v14 := ()
    let v15 := v14
    let v16 := self.total
    let index_17 := index_1
    let v18 := v16 - index_17
    let width_19 := width_2
    let v20 := decide (v18.toNat < width_19.toNat)
    let b6 : UInt256 → UInt256 → Range_Unknown_Unknown → UInt256 → FeM Window := fun «end» digit r2 v3 => do
      let mut «end» := «end»
      let mut digit := digit
      let mut r2 := r2
      let end_27 := v3
      «end» := end_27
      let rec b7 («end» : UInt256) (digit : UInt256) (r2 : Range_Unknown_Unknown) : FeM Window := do
        let mut «end» := «end»
        let mut digit := digit
        let mut r2 := r2
        let end_28 := «end»
        let end_29 := end_28
        let v30 := (UInt256.ofNat 1)
        let v31 := end_29 - v30
        let v32 ← Exponent_bit self v31
        let v33 := (UInt256.ofNat 0)
        let v34 := decide (v32 = v33)
        if v34 then
          let v35 := (UInt256.ofNat 1)
          let end_36 := «end»
          let v37 := end_36 - v35
          let end_38 := v37
          «end» := end_38
          let v39 := ()
          b7 «end» digit r2
        else
          let v40 := (UInt256.ofNat 0)
          let digit_41 := v40
          digit := digit_41
          let index_42 := index_1
          let v43 := ()
          let end_44 := «end»
          let v45 := (Range_Unknown_Unknown.mk index_42 end_44)
          r2 := v45
          let v47 ← Range_Unknown_Unknown_start r2
          let v48 := v47
          let rec b10 («end» : UInt256) (digit : UInt256) (r2 : Range_Unknown_Unknown) (v4 : UInt256) : FeM Window := do
            let mut «end» := «end»
            let mut digit := digit
            let mut r2 := r2
            let v50 ← Range_Unknown_Unknown_has r2 v4
            if v50 then
              let v52 ← Range_Unknown_Unknown_at r2 v4
              let bit := v52
              let digit_54 := digit
              let digit_55 := digit_54
              let v56 := (UInt256.ofNat 1)
              let v57 := digit_55 <<< v56
              let v58 ← Exponent_bit self bit
              let v59 := v57 ||| v58
              let digit_60 := v59
              digit := digit_60
              let v61 := ()
              let v63 ← Range_Unknown_Unknown_next r2 v4
              let v64 := v63
              b10 «end» digit r2 v64
            else
              let end_65 := «end»
              let end_66 := end_65
              let digit_67 := digit
              let digit_68 := digit_67
              let v69 := (UInt256.ofNat 1)
              let v70 := digit_68 - v69
              let v71 := (UInt256.ofNat 2)
              let v72 ← checkedDiv v70 v71
              let v73 := true
              let v74 := (Window.mk end_66 v72 v73)
              return v74
          partial_fixpoint
          b10 «end» digit r2 v48
      partial_fixpoint
      b7 «end» digit r2
    if v20 then
      let v21 := self.total
      let v22 := v21
      b6 «end» digit r2 v22
    else
      let index_23 := index_1
      let width_24 := width_2
      let v25 := index_23 + width_24
      let v26 := v25
      b6 «end» digit r2 v26

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
  let v7 := (UInt256.ofNat 0)
  let v8 ← MemArray_index_ref_u64 value v7
  let low_9 ← mloadInt (.u 64) v8
  let low_10 := low_9
  inverse := low_10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let inverse_22 := inverse
      let inverse_23 := inverse_22
      let v24 := (UInt256.ofNat 2)
      let inverse_25 := inverse
      let v26 := IntTy.mulW (.u 64) low_9 inverse_25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) inverse_23 v27
      let inverse_29 := v28
      inverse := inverse_29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 1)
      let v35 ← Wide_of value v34
      let v36 := (UInt256.ofNat 0)
      let inverse_37 := inverse
      let v38 := IntTy.subW (.u 64) v36 inverse_37
      let v39 := (Small_1.mk v35 v38)
      let small_40 := v39
      small := small_40
      let v41 := exponent.total
      let v42 := exponent.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let width := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := width - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let entries := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let table_58 := v57
      table := table_58
      let v59 := (UInt256.ofNat 1)
      let v60 ← Wide_of base v59
      r9 := v60
      let v62 ← Small_r2_1 small
      r10 := v62
      let v66 ← Small_mul_1 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < entries.toNat)
      let b7 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := table[v106.toNat]!
        let power_108 := v107
        power := power_108
        let v109 := exponent.top
        let index_110 := v109
        index := index_110
        let v111 := false
        let started_112 := v111
        started := started_112
        let rec b12 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let index_113 := index
          let index_114 := index_113
          let v115 := exponent.total
          let v116 := decide (index_114.toNat < v115.toNat)
          if v116 then
            let index_117 := index
            let index_118 := index_117
            let v119 ← Exponent_window exponent index_118 width
            let window := v119
            let started_121 := started
            let rec b17 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := window.odd
              let b24 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := window.«end»
                let index_174 := v173
                index := index_174
                let v175 := ()
                b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let started_152 := started
                let b27 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let power_165 := v6
                  power := power_165
                  let v166 := ()
                  let v167 := true
                  let started_168 := v167
                  started := started_168
                  let v169 := ()
                  let v170 := v169
                  b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if started_152 then
                  let power_153 := power
                  let power_154 := power_153
                  r18 := power_154
                  let v155 := window.entry
                  checkIndex v155 64
                  let v156 := table[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_1 small r18 r19
                  let v161 := v160
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := window.entry
                  checkIndex v162 64
                  let v163 := table[v162.toNat]!
                  let v164 := v163
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if started_121 then
              let index_122 := index
              let index_123 := index_122
              let v124 := window.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk index_123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let power_134 := power
                  let power_135 := power_134
                  r16 := power_135
                  let power_136 := power
                  let power_137 := power_136
                  r17 := power_137
                  let v141 ← Small_mul_1 small r16 r17
                  let power_142 := v141
                  power := power_142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let power_176 := power
            let power_177 := power_176
            r20 := power_177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_1 small r20 r21
            let out := v186
            let v188 := (UInt256.ofNat 1)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let result_192 := v191
              result := result_192
              let v193 := out.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 result v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              result := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 1)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 1)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 1)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let result_231 := result
                    let result_232 := result_231
                    let v233 := Option_MemArray_u64.Some result_232
                    return v233
                  if v222 then
                    let v223 := out.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 result v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    result := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := out.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 result v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  result := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := out.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 result v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                result := v204_0
                let v205 := ()
                let v206 := v205
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := table[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := table[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_1 small r11 r12
        let squared_79 := v78
        squared := squared_79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 entries)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (inverse : UInt256) (small : Small_1) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := k - v91
            checkIndex v92 64
            let v93 := table[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_1 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def Small_double_2_k2 (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (borrow_49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb s3 v50 borrow_49
  let v52 := v51.fst
  let d3 := v52
  let v54 := v51.snd
  let borrow_55 := v54
  let v56 := x.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 borrow_55
  let v61 := v60.snd
  let borrow_62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (borrow_62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk s0 s1 s2 s3)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk d0 d1 d2 d3)
    let v68 := v67
    b3 v68

def Small_double_2_k1 (self : Small_2) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let s3 := v29
  let v31 := self.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb s0 v31 v32
  let v34 := v33.fst
  let d0 := v34
  let v36 := v33.snd
  let borrow_37 := v36
  let v38 := self.m.l1
  let v39 ← sbb s1 v38 borrow_37
  let v40 := v39.fst
  let d1 := v40
  let v42 := v39.snd
  let borrow_43 := v42
  let v44 := self.m.l2
  let v45 ← sbb s2 v44 borrow_43
  let v46 := v45.fst
  let d2 := v46
  let v48 := v45.snd
  let borrow_49 := v48
  let v50 := self.m.l3
  Small_double_2_k2 x s0 s1 s2 s3 d0 d1 d2 borrow_49 v50

def Small_double_2 (self : Small_2) (x : Wide) : FeM Wide := do
  let v3 := x.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let s0 := v5
  let v7 := x.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := x.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let s1 := v13
  let v15 := x.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := x.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let s2 := v21
  let v23 := x.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := x.l2
  Small_double_2_k1 self x s0 s1 s2 v25 v26

def Small_row2_2_k1 (self : Small_2) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (t3 : UInt256) (m : UInt256) : FeM (Wide × UInt256) := do
  let v29 := self.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac m v29 t0 v30
  let v32 := v31.snd
  let c_33 := v32
  let v34 := self.m.l1
  let v35 ← mac m v34 t1 c_33
  let v36 := v35.fst
  let r0 := v36
  let v38 := v35.snd
  let c_39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc t2 v40 c_39
  let v42 := v41.fst
  let r1 := v42
  let v44 := v41.snd
  let c_45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk r0 r1 v46 v47)
  let v49 := IntTy.addW (.u 64) t3 c_45
  let v50 := (v48, v49)
  return v50

def Small_row2_2 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc top c_19 v20
  let v22 := v21.fst
  let t2 := v22
  let v24 := v21.snd
  let t3 := v24
  let v26 := self.inverse
  let v27 := IntTy.mulW (.u 64) t0 v26
  let m := v27
  Small_row2_2_k1 self t0 t1 t2 t3 m

def Small_row3_2_k2 (t3 : UInt256) (t4 : UInt256) (r0 : UInt256) (r1 : UInt256) (c_52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc t3 v53 c_52
  let v55 := v54.fst
  let r2 := v55
  let v57 := v54.snd
  let c_58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk r0 r1 r2 v59)
  let v61 := IntTy.addW (.u 64) t4 c_58
  let v62 := (v60, v61)
  return v62

def Small_row3_2_k1 (self : Small_2) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let t3 := v29
  let v31 := v28.snd
  let t4 := v31
  let v33 := self.inverse
  let v34 := IntTy.mulW (.u 64) t0 v33
  let m := v34
  let v36 := self.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac m v36 t0 v37
  let v39 := v38.snd
  let c_40 := v39
  let v41 := self.m.l1
  let v42 ← mac m v41 t1 c_40
  let v43 := v42.fst
  let r0 := v43
  let v45 := v42.snd
  let c_46 := v45
  let v47 := self.m.l2
  let v48 ← mac m v47 t2 c_46
  let v49 := v48.fst
  let r1 := v49
  let v51 := v48.snd
  let c_52 := v51
  Small_row3_2_k2 t3 t4 r0 r1 c_52

def Small_row3_2 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc top c_26 v27
  Small_row3_2_k1 self t0 t1 t2 v28

def Small_row4_2_k2 (self : Small_2) (t2 : UInt256) (t3 : UInt256) (t4 : UInt256) (t5 : UInt256) (m : UInt256) (r0 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let c_53 := v52
  let v54 := self.m.l2
  let v55 ← mac m v54 t2 c_53
  let v56 := v55.fst
  let r1 := v56
  let v58 := v55.snd
  let c_59 := v58
  let v60 := self.m.l3
  let v61 ← mac m v60 t3 c_59
  let v62 := v61.fst
  let r2 := v62
  let v64 := v61.snd
  let c_65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc t4 v66 c_65
  let v68 := v67.fst
  let r3 := v68
  let v70 := v67.snd
  let c_71 := v70
  let v72 := (Wide.mk r0 r1 r2 r3)
  let v73 := IntTy.addW (.u 64) t5 c_71
  let v74 := (v72, v73)
  return v74

def Small_row4_2_k1 (self : Small_2) (b : UInt256) (top : UInt256) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (c_26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 b v28 c_26
  let v30 := v29.fst
  let t3 := v30
  let v32 := v29.snd
  let c_33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc top c_33 v34
  let v36 := v35.fst
  let t4 := v36
  let v38 := v35.snd
  let t5 := v38
  let v40 := self.inverse
  let v41 := IntTy.mulW (.u 64) t0 v40
  let m := v41
  let v43 := self.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac m v43 t0 v44
  let v46 := v45.snd
  let c_47 := v46
  let v48 := self.m.l1
  let v49 ← mac m v48 t1 c_47
  let v50 := v49.fst
  let r0 := v50
  let v52 := v49.snd
  Small_row4_2_k2 self t2 t3 t4 t5 m r0 v52

def Small_row4_2 (self : Small_2) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := a.l3
  let v28 := t.l3
  Small_row4_2_k1 self b top t0 t1 t2 c_26 v27 v28

def Small_mul_2 (self : Small_2) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let zero_13 := v12
  zero := zero_13
  let v14 := (UInt256.ofNat 2)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := a.l0
    let v18 := v17
    let v19 := b.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let t_22 := v21
    let v23 ← downcast_truncate_u128_u64 t_22
    let low := v23
    let v25 := self.inverse
    let v26 := IntTy.mulW (.u 64) low v25
    let q := v26
    let v28 := q
    let v29 := self.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let product := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := t_22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := product >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (low ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let sum_45 := v44
      let v46 := self.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ sum_45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v4 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        let reduced := v4
        let v55 ← downcast_truncate_u128_u64 reduced
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := self.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) sum_45 v50
        let v52 := v51
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v52
      else
        let sum_53 := sum_45
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum_53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 2)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := b.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_2 self a v65 zero v66
      let v69 := v68.fst
      let t_70 := v69
      t_r1 := t_70
      let v71 := v68.snd
      let top_72 := v71
      let v73 := b.l1
      let v75 ← Small_row2_2 self a v73 t_r1 top_72
      let v76 := v75.fst
      let t_77 := v76
      let v78 := v75.snd
      let top_79 := v78
      let v80 := t_77.l0
      let v81 := self.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let d0_85 := v84
      let v86 := v83.snd
      let borrow_87 := v86
      let v88 := t_77.l1
      let v89 := self.m.l1
      let v90 ← sbb v88 v89 borrow_87
      let v91 := v90.fst
      let d1_92 := v91
      let v93 := v90.snd
      let borrow_94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb top_79 v95 borrow_94
      let v97 := v96.snd
      let borrow_98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (borrow_98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      if v100 then
        let t_101 := t_77
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk d0_85 d1_92 v102 v103)
        let v105 := v104
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 2)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := b.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_2 self a v111 zero v112
        let v115 := v114.fst
        let t_116 := v115
        t_r2 := t_116
        let v117 := v114.snd
        let top_118 := v117
        let v119 := b.l1
        let v121 ← Small_row3_2 self a v119 t_r2 top_118
        let v122 := v121.fst
        let t_123 := v122
        t_r3 := t_123
        let v124 := v121.snd
        let top_125 := v124
        let v126 := b.l2
        let v128 ← Small_row3_2 self a v126 t_r3 top_125
        let v129 := v128.fst
        let t_130 := v129
        let v131 := v128.snd
        let top_132 := v131
        let v133 := t_130.l0
        let v134 := self.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let d0_138 := v137
        let v139 := v136.snd
        let borrow_140 := v139
        let v141 := t_130.l1
        let v142 := self.m.l1
        let v143 ← sbb v141 v142 borrow_140
        let v144 := v143.fst
        let d1_145 := v144
        let v146 := v143.snd
        let borrow_147 := v146
        let v148 := t_130.l2
        let v149 := self.m.l2
        let v150 ← sbb v148 v149 borrow_147
        let v151 := v150.fst
        let d2_152 := v151
        let v153 := v150.snd
        let borrow_154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb top_132 v155 borrow_154
        let v157 := v156.snd
        let borrow_158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (borrow_158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        if v160 then
          let t_161 := t_130
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk d0_138 d1_145 d2_152 v162)
          let v164 := v163
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := b.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_2 self a v167 zero v168
        let v171 := v170.fst
        let t_172 := v171
        t_r4 := t_172
        let v173 := v170.snd
        let top_174 := v173
        let v175 := b.l1
        let v177 ← Small_row4_2 self a v175 t_r4 top_174
        let v178 := v177.fst
        let t_179 := v178
        t_r5 := t_179
        let v180 := v177.snd
        let top_181 := v180
        let v182 := b.l2
        let v184 ← Small_row4_2 self a v182 t_r5 top_181
        let v185 := v184.fst
        let t_186 := v185
        t_r6 := t_186
        let v187 := v184.snd
        let top_188 := v187
        let v189 := b.l3
        let v191 ← Small_row4_2 self a v189 t_r6 top_188
        let v192 := v191.fst
        let t_193 := v192
        let v194 := v191.snd
        let top_195 := v194
        let v196 := t_193.l0
        let v197 := self.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let d0_201 := v200
        let v202 := v199.snd
        let borrow_203 := v202
        let v204 := t_193.l1
        let v205 := self.m.l1
        let v206 ← sbb v204 v205 borrow_203
        let v207 := v206.fst
        let d1_208 := v207
        let v209 := v206.snd
        let borrow_210 := v209
        let v211 := t_193.l2
        let v212 := self.m.l2
        let v213 ← sbb v211 v212 borrow_210
        let v214 := v213.fst
        let d2_215 := v214
        let v216 := v213.snd
        let borrow_217 := v216
        let v218 := t_193.l3
        let v219 := self.m.l3
        let v220 ← sbb v218 v219 borrow_217
        let v221 := v220.fst
        let d3 := v221
        let v223 := v220.snd
        let borrow_224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb top_195 v225 borrow_224
        let v227 := v226.snd
        let borrow_228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (borrow_228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        if v230 then
          let t_231 := t_193
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_231
        else
          let v232 := (Wide.mk d0_201 d1_208 d2_215 d3)
          let v233 := v232
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v233

def Small_r2_2 (self : Small_2) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 2)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v1 => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let top := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := top
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let shift := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 shift
    let bit_38 := v37
    let v39 := (UInt256.ofNat 2)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 2)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 2)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 2)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let x_64 := v63
            x := x_64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 shift
            let v67 ← downcast_truncate_u64_usize v66
            let doublings := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 2)
            let v71 := doublings + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let x_81 := x
                let x_82 := x_81
                r2 := x_82
                let v84 ← Small_double_2 self r2
                let x_85 := v84
                x := x_85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 x r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let x_101 := x
                    let x_102 := x_101
                    r4 := x_102
                    let x_103 := x
                    let x_104 := x_103
                    r5 := x_104
                    let v107 ← Small_mul_2 self r4 r5
                    let x_108 := v107
                    x := x_108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 x r1 r2 r3 r4 r5 v112
                  else
                    let x_113 := x
                    return x_113
                partial_fixpoint
                b26 x r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 x r1 r2 r3 r4 r5 v76
          if v59 then
            let bit_60 := bit_38
            b21 x r1 r2 r3 r4 r5 bit_60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 x r1 r2 r3 r4 r5 v62
        if v53 then
          let bit_54 := bit_38
          b18 x r1 r2 r3 r4 r5 bit_54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 x r1 r2 r3 r4 r5 v56
      if v47 then
        let bit_48 := bit_38
        b15 x r1 r2 r3 r4 r5 bit_48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 x r1 r2 r3 r4 r5 v50
    if v41 then
      let bit_42 := bit_38
      b12 x r1 r2 r3 r4 r5 bit_42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 x r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := self.m.l0
    let v14 := v13
    b3 x r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 2)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 x r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := self.m.l1
      let v19 := v18
      b6 x r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 2)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 x r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := self.m.l2
        let v24 := v23
        b9 x r1 r2 r3 r4 r5 v24
      else
        let v25 := self.m.l3
        let v26 := v25
        b9 x r1 r2 r3 r4 r5 v26

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
  let v7 := (UInt256.ofNat 0)
  let v8 ← MemArray_index_ref_u64 value v7
  let low_9 ← mloadInt (.u 64) v8
  let low_10 := low_9
  inverse := low_10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let inverse_22 := inverse
      let inverse_23 := inverse_22
      let v24 := (UInt256.ofNat 2)
      let inverse_25 := inverse
      let v26 := IntTy.mulW (.u 64) low_9 inverse_25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) inverse_23 v27
      let inverse_29 := v28
      inverse := inverse_29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 2)
      let v35 ← Wide_of value v34
      let v36 := (UInt256.ofNat 0)
      let inverse_37 := inverse
      let v38 := IntTy.subW (.u 64) v36 inverse_37
      let v39 := (Small_2.mk v35 v38)
      let small_40 := v39
      small := small_40
      let v41 := exponent.total
      let v42 := exponent.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let width := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := width - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let entries := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let table_58 := v57
      table := table_58
      let v59 := (UInt256.ofNat 2)
      let v60 ← Wide_of base v59
      r9 := v60
      let v62 ← Small_r2_2 small
      r10 := v62
      let v66 ← Small_mul_2 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < entries.toNat)
      let b7 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := table[v106.toNat]!
        let power_108 := v107
        power := power_108
        let v109 := exponent.top
        let index_110 := v109
        index := index_110
        let v111 := false
        let started_112 := v111
        started := started_112
        let rec b12 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let index_113 := index
          let index_114 := index_113
          let v115 := exponent.total
          let v116 := decide (index_114.toNat < v115.toNat)
          if v116 then
            let index_117 := index
            let index_118 := index_117
            let v119 ← Exponent_window exponent index_118 width
            let window := v119
            let started_121 := started
            let rec b17 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := window.odd
              let b24 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := window.«end»
                let index_174 := v173
                index := index_174
                let v175 := ()
                b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let started_152 := started
                let b27 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let power_165 := v6
                  power := power_165
                  let v166 := ()
                  let v167 := true
                  let started_168 := v167
                  started := started_168
                  let v169 := ()
                  let v170 := v169
                  b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if started_152 then
                  let power_153 := power
                  let power_154 := power_153
                  r18 := power_154
                  let v155 := window.entry
                  checkIndex v155 64
                  let v156 := table[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_2 small r18 r19
                  let v161 := v160
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := window.entry
                  checkIndex v162 64
                  let v163 := table[v162.toNat]!
                  let v164 := v163
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if started_121 then
              let index_122 := index
              let index_123 := index_122
              let v124 := window.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk index_123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let power_134 := power
                  let power_135 := power_134
                  r16 := power_135
                  let power_136 := power
                  let power_137 := power_136
                  r17 := power_137
                  let v141 ← Small_mul_2 small r16 r17
                  let power_142 := v141
                  power := power_142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let power_176 := power
            let power_177 := power_176
            r20 := power_177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_2 small r20 r21
            let out := v186
            let v188 := (UInt256.ofNat 2)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let result_192 := v191
              result := result_192
              let v193 := out.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 result v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              result := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 2)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 2)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 2)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let result_231 := result
                    let result_232 := result_231
                    let v233 := Option_MemArray_u64.Some result_232
                    return v233
                  if v222 then
                    let v223 := out.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 result v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    result := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := out.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 result v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  result := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := out.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 result v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                result := v204_0
                let v205 := ()
                let v206 := v205
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := table[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := table[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_2 small r11 r12
        let squared_79 := v78
        squared := squared_79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 entries)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (inverse : UInt256) (small : Small_2) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := k - v91
            checkIndex v92 64
            let v93 := table[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_2 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def Small_double_3_k2 (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (borrow_49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb s3 v50 borrow_49
  let v52 := v51.fst
  let d3 := v52
  let v54 := v51.snd
  let borrow_55 := v54
  let v56 := x.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 borrow_55
  let v61 := v60.snd
  let borrow_62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (borrow_62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk s0 s1 s2 s3)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk d0 d1 d2 d3)
    let v68 := v67
    b3 v68

def Small_double_3_k1 (self : Small_3) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let s3 := v29
  let v31 := self.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb s0 v31 v32
  let v34 := v33.fst
  let d0 := v34
  let v36 := v33.snd
  let borrow_37 := v36
  let v38 := self.m.l1
  let v39 ← sbb s1 v38 borrow_37
  let v40 := v39.fst
  let d1 := v40
  let v42 := v39.snd
  let borrow_43 := v42
  let v44 := self.m.l2
  let v45 ← sbb s2 v44 borrow_43
  let v46 := v45.fst
  let d2 := v46
  let v48 := v45.snd
  let borrow_49 := v48
  let v50 := self.m.l3
  Small_double_3_k2 x s0 s1 s2 s3 d0 d1 d2 borrow_49 v50

def Small_double_3 (self : Small_3) (x : Wide) : FeM Wide := do
  let v3 := x.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let s0 := v5
  let v7 := x.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := x.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let s1 := v13
  let v15 := x.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := x.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let s2 := v21
  let v23 := x.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := x.l2
  Small_double_3_k1 self x s0 s1 s2 v25 v26

def Small_row2_3_k1 (self : Small_3) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (t3 : UInt256) (m : UInt256) : FeM (Wide × UInt256) := do
  let v29 := self.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac m v29 t0 v30
  let v32 := v31.snd
  let c_33 := v32
  let v34 := self.m.l1
  let v35 ← mac m v34 t1 c_33
  let v36 := v35.fst
  let r0 := v36
  let v38 := v35.snd
  let c_39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc t2 v40 c_39
  let v42 := v41.fst
  let r1 := v42
  let v44 := v41.snd
  let c_45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk r0 r1 v46 v47)
  let v49 := IntTy.addW (.u 64) t3 c_45
  let v50 := (v48, v49)
  return v50

def Small_row2_3 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc top c_19 v20
  let v22 := v21.fst
  let t2 := v22
  let v24 := v21.snd
  let t3 := v24
  let v26 := self.inverse
  let v27 := IntTy.mulW (.u 64) t0 v26
  let m := v27
  Small_row2_3_k1 self t0 t1 t2 t3 m

def Small_row3_3_k2 (t3 : UInt256) (t4 : UInt256) (r0 : UInt256) (r1 : UInt256) (c_52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc t3 v53 c_52
  let v55 := v54.fst
  let r2 := v55
  let v57 := v54.snd
  let c_58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk r0 r1 r2 v59)
  let v61 := IntTy.addW (.u 64) t4 c_58
  let v62 := (v60, v61)
  return v62

def Small_row3_3_k1 (self : Small_3) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let t3 := v29
  let v31 := v28.snd
  let t4 := v31
  let v33 := self.inverse
  let v34 := IntTy.mulW (.u 64) t0 v33
  let m := v34
  let v36 := self.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac m v36 t0 v37
  let v39 := v38.snd
  let c_40 := v39
  let v41 := self.m.l1
  let v42 ← mac m v41 t1 c_40
  let v43 := v42.fst
  let r0 := v43
  let v45 := v42.snd
  let c_46 := v45
  let v47 := self.m.l2
  let v48 ← mac m v47 t2 c_46
  let v49 := v48.fst
  let r1 := v49
  let v51 := v48.snd
  let c_52 := v51
  Small_row3_3_k2 t3 t4 r0 r1 c_52

def Small_row3_3 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc top c_26 v27
  Small_row3_3_k1 self t0 t1 t2 v28

def Small_row4_3_k2 (self : Small_3) (t2 : UInt256) (t3 : UInt256) (t4 : UInt256) (t5 : UInt256) (m : UInt256) (r0 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let c_53 := v52
  let v54 := self.m.l2
  let v55 ← mac m v54 t2 c_53
  let v56 := v55.fst
  let r1 := v56
  let v58 := v55.snd
  let c_59 := v58
  let v60 := self.m.l3
  let v61 ← mac m v60 t3 c_59
  let v62 := v61.fst
  let r2 := v62
  let v64 := v61.snd
  let c_65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc t4 v66 c_65
  let v68 := v67.fst
  let r3 := v68
  let v70 := v67.snd
  let c_71 := v70
  let v72 := (Wide.mk r0 r1 r2 r3)
  let v73 := IntTy.addW (.u 64) t5 c_71
  let v74 := (v72, v73)
  return v74

def Small_row4_3_k1 (self : Small_3) (b : UInt256) (top : UInt256) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (c_26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 b v28 c_26
  let v30 := v29.fst
  let t3 := v30
  let v32 := v29.snd
  let c_33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc top c_33 v34
  let v36 := v35.fst
  let t4 := v36
  let v38 := v35.snd
  let t5 := v38
  let v40 := self.inverse
  let v41 := IntTy.mulW (.u 64) t0 v40
  let m := v41
  let v43 := self.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac m v43 t0 v44
  let v46 := v45.snd
  let c_47 := v46
  let v48 := self.m.l1
  let v49 ← mac m v48 t1 c_47
  let v50 := v49.fst
  let r0 := v50
  let v52 := v49.snd
  Small_row4_3_k2 self t2 t3 t4 t5 m r0 v52

def Small_row4_3 (self : Small_3) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := a.l3
  let v28 := t.l3
  Small_row4_3_k1 self b top t0 t1 t2 c_26 v27 v28

def Small_mul_3 (self : Small_3) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let zero_13 := v12
  zero := zero_13
  let v14 := (UInt256.ofNat 3)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := a.l0
    let v18 := v17
    let v19 := b.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let t_22 := v21
    let v23 ← downcast_truncate_u128_u64 t_22
    let low := v23
    let v25 := self.inverse
    let v26 := IntTy.mulW (.u 64) low v25
    let q := v26
    let v28 := q
    let v29 := self.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let product := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := t_22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := product >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (low ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let sum_45 := v44
      let v46 := self.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ sum_45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v4 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        let reduced := v4
        let v55 ← downcast_truncate_u128_u64 reduced
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := self.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) sum_45 v50
        let v52 := v51
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v52
      else
        let sum_53 := sum_45
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum_53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 3)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := b.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_3 self a v65 zero v66
      let v69 := v68.fst
      let t_70 := v69
      t_r1 := t_70
      let v71 := v68.snd
      let top_72 := v71
      let v73 := b.l1
      let v75 ← Small_row2_3 self a v73 t_r1 top_72
      let v76 := v75.fst
      let t_77 := v76
      let v78 := v75.snd
      let top_79 := v78
      let v80 := t_77.l0
      let v81 := self.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let d0_85 := v84
      let v86 := v83.snd
      let borrow_87 := v86
      let v88 := t_77.l1
      let v89 := self.m.l1
      let v90 ← sbb v88 v89 borrow_87
      let v91 := v90.fst
      let d1_92 := v91
      let v93 := v90.snd
      let borrow_94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb top_79 v95 borrow_94
      let v97 := v96.snd
      let borrow_98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (borrow_98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      if v100 then
        let t_101 := t_77
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk d0_85 d1_92 v102 v103)
        let v105 := v104
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 3)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := b.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_3 self a v111 zero v112
        let v115 := v114.fst
        let t_116 := v115
        t_r2 := t_116
        let v117 := v114.snd
        let top_118 := v117
        let v119 := b.l1
        let v121 ← Small_row3_3 self a v119 t_r2 top_118
        let v122 := v121.fst
        let t_123 := v122
        t_r3 := t_123
        let v124 := v121.snd
        let top_125 := v124
        let v126 := b.l2
        let v128 ← Small_row3_3 self a v126 t_r3 top_125
        let v129 := v128.fst
        let t_130 := v129
        let v131 := v128.snd
        let top_132 := v131
        let v133 := t_130.l0
        let v134 := self.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let d0_138 := v137
        let v139 := v136.snd
        let borrow_140 := v139
        let v141 := t_130.l1
        let v142 := self.m.l1
        let v143 ← sbb v141 v142 borrow_140
        let v144 := v143.fst
        let d1_145 := v144
        let v146 := v143.snd
        let borrow_147 := v146
        let v148 := t_130.l2
        let v149 := self.m.l2
        let v150 ← sbb v148 v149 borrow_147
        let v151 := v150.fst
        let d2_152 := v151
        let v153 := v150.snd
        let borrow_154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb top_132 v155 borrow_154
        let v157 := v156.snd
        let borrow_158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (borrow_158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        if v160 then
          let t_161 := t_130
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk d0_138 d1_145 d2_152 v162)
          let v164 := v163
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := b.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_3 self a v167 zero v168
        let v171 := v170.fst
        let t_172 := v171
        t_r4 := t_172
        let v173 := v170.snd
        let top_174 := v173
        let v175 := b.l1
        let v177 ← Small_row4_3 self a v175 t_r4 top_174
        let v178 := v177.fst
        let t_179 := v178
        t_r5 := t_179
        let v180 := v177.snd
        let top_181 := v180
        let v182 := b.l2
        let v184 ← Small_row4_3 self a v182 t_r5 top_181
        let v185 := v184.fst
        let t_186 := v185
        t_r6 := t_186
        let v187 := v184.snd
        let top_188 := v187
        let v189 := b.l3
        let v191 ← Small_row4_3 self a v189 t_r6 top_188
        let v192 := v191.fst
        let t_193 := v192
        let v194 := v191.snd
        let top_195 := v194
        let v196 := t_193.l0
        let v197 := self.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let d0_201 := v200
        let v202 := v199.snd
        let borrow_203 := v202
        let v204 := t_193.l1
        let v205 := self.m.l1
        let v206 ← sbb v204 v205 borrow_203
        let v207 := v206.fst
        let d1_208 := v207
        let v209 := v206.snd
        let borrow_210 := v209
        let v211 := t_193.l2
        let v212 := self.m.l2
        let v213 ← sbb v211 v212 borrow_210
        let v214 := v213.fst
        let d2_215 := v214
        let v216 := v213.snd
        let borrow_217 := v216
        let v218 := t_193.l3
        let v219 := self.m.l3
        let v220 ← sbb v218 v219 borrow_217
        let v221 := v220.fst
        let d3 := v221
        let v223 := v220.snd
        let borrow_224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb top_195 v225 borrow_224
        let v227 := v226.snd
        let borrow_228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (borrow_228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        if v230 then
          let t_231 := t_193
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_231
        else
          let v232 := (Wide.mk d0_201 d1_208 d2_215 d3)
          let v233 := v232
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v233

def Small_r2_3 (self : Small_3) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 3)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v1 => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let top := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := top
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let shift := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 shift
    let bit_38 := v37
    let v39 := (UInt256.ofNat 3)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 3)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 3)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 3)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let x_64 := v63
            x := x_64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 shift
            let v67 ← downcast_truncate_u64_usize v66
            let doublings := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 3)
            let v71 := doublings + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let x_81 := x
                let x_82 := x_81
                r2 := x_82
                let v84 ← Small_double_3 self r2
                let x_85 := v84
                x := x_85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 x r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let x_101 := x
                    let x_102 := x_101
                    r4 := x_102
                    let x_103 := x
                    let x_104 := x_103
                    r5 := x_104
                    let v107 ← Small_mul_3 self r4 r5
                    let x_108 := v107
                    x := x_108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 x r1 r2 r3 r4 r5 v112
                  else
                    let x_113 := x
                    return x_113
                partial_fixpoint
                b26 x r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 x r1 r2 r3 r4 r5 v76
          if v59 then
            let bit_60 := bit_38
            b21 x r1 r2 r3 r4 r5 bit_60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 x r1 r2 r3 r4 r5 v62
        if v53 then
          let bit_54 := bit_38
          b18 x r1 r2 r3 r4 r5 bit_54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 x r1 r2 r3 r4 r5 v56
      if v47 then
        let bit_48 := bit_38
        b15 x r1 r2 r3 r4 r5 bit_48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 x r1 r2 r3 r4 r5 v50
    if v41 then
      let bit_42 := bit_38
      b12 x r1 r2 r3 r4 r5 bit_42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 x r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := self.m.l0
    let v14 := v13
    b3 x r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 3)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 x r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := self.m.l1
      let v19 := v18
      b6 x r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 3)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 x r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := self.m.l2
        let v24 := v23
        b9 x r1 r2 r3 r4 r5 v24
      else
        let v25 := self.m.l3
        let v26 := v25
        b9 x r1 r2 r3 r4 r5 v26

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
  let v7 := (UInt256.ofNat 0)
  let v8 ← MemArray_index_ref_u64 value v7
  let low_9 ← mloadInt (.u 64) v8
  let low_10 := low_9
  inverse := low_10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let inverse_22 := inverse
      let inverse_23 := inverse_22
      let v24 := (UInt256.ofNat 2)
      let inverse_25 := inverse
      let v26 := IntTy.mulW (.u 64) low_9 inverse_25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) inverse_23 v27
      let inverse_29 := v28
      inverse := inverse_29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 3)
      let v35 ← Wide_of value v34
      let v36 := (UInt256.ofNat 0)
      let inverse_37 := inverse
      let v38 := IntTy.subW (.u 64) v36 inverse_37
      let v39 := (Small_3.mk v35 v38)
      let small_40 := v39
      small := small_40
      let v41 := exponent.total
      let v42 := exponent.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let width := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := width - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let entries := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let table_58 := v57
      table := table_58
      let v59 := (UInt256.ofNat 3)
      let v60 ← Wide_of base v59
      r9 := v60
      let v62 ← Small_r2_3 small
      r10 := v62
      let v66 ← Small_mul_3 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < entries.toNat)
      let b7 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := table[v106.toNat]!
        let power_108 := v107
        power := power_108
        let v109 := exponent.top
        let index_110 := v109
        index := index_110
        let v111 := false
        let started_112 := v111
        started := started_112
        let rec b12 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let index_113 := index
          let index_114 := index_113
          let v115 := exponent.total
          let v116 := decide (index_114.toNat < v115.toNat)
          if v116 then
            let index_117 := index
            let index_118 := index_117
            let v119 ← Exponent_window exponent index_118 width
            let window := v119
            let started_121 := started
            let rec b17 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := window.odd
              let b24 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := window.«end»
                let index_174 := v173
                index := index_174
                let v175 := ()
                b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let started_152 := started
                let b27 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let power_165 := v6
                  power := power_165
                  let v166 := ()
                  let v167 := true
                  let started_168 := v167
                  started := started_168
                  let v169 := ()
                  let v170 := v169
                  b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if started_152 then
                  let power_153 := power
                  let power_154 := power_153
                  r18 := power_154
                  let v155 := window.entry
                  checkIndex v155 64
                  let v156 := table[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_3 small r18 r19
                  let v161 := v160
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := window.entry
                  checkIndex v162 64
                  let v163 := table[v162.toNat]!
                  let v164 := v163
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if started_121 then
              let index_122 := index
              let index_123 := index_122
              let v124 := window.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk index_123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let power_134 := power
                  let power_135 := power_134
                  r16 := power_135
                  let power_136 := power
                  let power_137 := power_136
                  r17 := power_137
                  let v141 ← Small_mul_3 small r16 r17
                  let power_142 := v141
                  power := power_142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let power_176 := power
            let power_177 := power_176
            r20 := power_177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_3 small r20 r21
            let out := v186
            let v188 := (UInt256.ofNat 3)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let result_192 := v191
              result := result_192
              let v193 := out.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 result v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              result := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 3)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 3)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 3)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let result_231 := result
                    let result_232 := result_231
                    let v233 := Option_MemArray_u64.Some result_232
                    return v233
                  if v222 then
                    let v223 := out.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 result v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    result := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := out.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 result v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  result := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := out.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 result v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                result := v204_0
                let v205 := ()
                let v206 := v205
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := table[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := table[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_3 small r11 r12
        let squared_79 := v78
        squared := squared_79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 entries)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (inverse : UInt256) (small : Small_3) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := k - v91
            checkIndex v92 64
            let v93 := table[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_3 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def Small_double_4_k2 (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (s3 : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (borrow_49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb s3 v50 borrow_49
  let v52 := v51.fst
  let d3 := v52
  let v54 := v51.snd
  let borrow_55 := v54
  let v56 := x.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 borrow_55
  let v61 := v60.snd
  let borrow_62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (borrow_62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk s0 s1 s2 s3)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk d0 d1 d2 d3)
    let v68 := v67
    b3 v68

def Small_double_4_k1 (self : Small_4) (x : Wide) (s0 : UInt256) (s1 : UInt256) (s2 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let s3 := v29
  let v31 := self.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb s0 v31 v32
  let v34 := v33.fst
  let d0 := v34
  let v36 := v33.snd
  let borrow_37 := v36
  let v38 := self.m.l1
  let v39 ← sbb s1 v38 borrow_37
  let v40 := v39.fst
  let d1 := v40
  let v42 := v39.snd
  let borrow_43 := v42
  let v44 := self.m.l2
  let v45 ← sbb s2 v44 borrow_43
  let v46 := v45.fst
  let d2 := v46
  let v48 := v45.snd
  let borrow_49 := v48
  let v50 := self.m.l3
  Small_double_4_k2 x s0 s1 s2 s3 d0 d1 d2 borrow_49 v50

def Small_double_4 (self : Small_4) (x : Wide) : FeM Wide := do
  let v3 := x.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let s0 := v5
  let v7 := x.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := x.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let s1 := v13
  let v15 := x.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := x.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let s2 := v21
  let v23 := x.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := x.l2
  Small_double_4_k1 self x s0 s1 s2 v25 v26

def Small_row2_4_k1 (self : Small_4) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (t3 : UInt256) (m : UInt256) : FeM (Wide × UInt256) := do
  let v29 := self.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac m v29 t0 v30
  let v32 := v31.snd
  let c_33 := v32
  let v34 := self.m.l1
  let v35 ← mac m v34 t1 c_33
  let v36 := v35.fst
  let r0 := v36
  let v38 := v35.snd
  let c_39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc t2 v40 c_39
  let v42 := v41.fst
  let r1 := v42
  let v44 := v41.snd
  let c_45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk r0 r1 v46 v47)
  let v49 := IntTy.addW (.u 64) t3 c_45
  let v50 := (v48, v49)
  return v50

def Small_row2_4 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc top c_19 v20
  let v22 := v21.fst
  let t2 := v22
  let v24 := v21.snd
  let t3 := v24
  let v26 := self.inverse
  let v27 := IntTy.mulW (.u 64) t0 v26
  let m := v27
  Small_row2_4_k1 self t0 t1 t2 t3 m

def Small_row3_4_k2 (t3 : UInt256) (t4 : UInt256) (r0 : UInt256) (r1 : UInt256) (c_52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc t3 v53 c_52
  let v55 := v54.fst
  let r2 := v55
  let v57 := v54.snd
  let c_58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk r0 r1 r2 v59)
  let v61 := IntTy.addW (.u 64) t4 c_58
  let v62 := (v60, v61)
  return v62

def Small_row3_4_k1 (self : Small_4) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let t3 := v29
  let v31 := v28.snd
  let t4 := v31
  let v33 := self.inverse
  let v34 := IntTy.mulW (.u 64) t0 v33
  let m := v34
  let v36 := self.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac m v36 t0 v37
  let v39 := v38.snd
  let c_40 := v39
  let v41 := self.m.l1
  let v42 ← mac m v41 t1 c_40
  let v43 := v42.fst
  let r0 := v43
  let v45 := v42.snd
  let c_46 := v45
  let v47 := self.m.l2
  let v48 ← mac m v47 t2 c_46
  let v49 := v48.fst
  let r1 := v49
  let v51 := v48.snd
  let c_52 := v51
  Small_row3_4_k2 t3 t4 r0 r1 c_52

def Small_row3_4 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc top c_26 v27
  Small_row3_4_k1 self t0 t1 t2 v28

def Small_row4_4_k2 (self : Small_4) (t2 : UInt256) (t3 : UInt256) (t4 : UInt256) (t5 : UInt256) (m : UInt256) (r0 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let c_53 := v52
  let v54 := self.m.l2
  let v55 ← mac m v54 t2 c_53
  let v56 := v55.fst
  let r1 := v56
  let v58 := v55.snd
  let c_59 := v58
  let v60 := self.m.l3
  let v61 ← mac m v60 t3 c_59
  let v62 := v61.fst
  let r2 := v62
  let v64 := v61.snd
  let c_65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc t4 v66 c_65
  let v68 := v67.fst
  let r3 := v68
  let v70 := v67.snd
  let c_71 := v70
  let v72 := (Wide.mk r0 r1 r2 r3)
  let v73 := IntTy.addW (.u 64) t5 c_71
  let v74 := (v72, v73)
  return v74

def Small_row4_4_k1 (self : Small_4) (b : UInt256) (top : UInt256) (t0 : UInt256) (t1 : UInt256) (t2 : UInt256) (c_26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 b v28 c_26
  let v30 := v29.fst
  let t3 := v30
  let v32 := v29.snd
  let c_33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc top c_33 v34
  let v36 := v35.fst
  let t4 := v36
  let v38 := v35.snd
  let t5 := v38
  let v40 := self.inverse
  let v41 := IntTy.mulW (.u 64) t0 v40
  let m := v41
  let v43 := self.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac m v43 t0 v44
  let v46 := v45.snd
  let c_47 := v46
  let v48 := self.m.l1
  let v49 ← mac m v48 t1 c_47
  let v50 := v49.fst
  let r0 := v50
  let v52 := v49.snd
  Small_row4_4_k2 self t2 t3 t4 t5 m r0 v52

def Small_row4_4 (self : Small_4) (a : Wide) (b : UInt256) (t : Wide) (top : UInt256) : FeM (Wide × UInt256) := do
  let v5 := a.l0
  let v6 := t.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 b v6 v7
  let v9 := v8.fst
  let t0 := v9
  let v11 := v8.snd
  let c_12 := v11
  let v13 := a.l1
  let v14 := t.l1
  let v15 ← mac v13 b v14 c_12
  let v16 := v15.fst
  let t1 := v16
  let v18 := v15.snd
  let c_19 := v18
  let v20 := a.l2
  let v21 := t.l2
  let v22 ← mac v20 b v21 c_19
  let v23 := v22.fst
  let t2 := v23
  let v25 := v22.snd
  let c_26 := v25
  let v27 := a.l3
  let v28 := t.l3
  Small_row4_4_k1 self b top t0 t1 t2 c_26 v27 v28

def Small_mul_4 (self : Small_4) (a : Wide) (b : Wide) : FeM Wide := do
  let mut zero : Wide := default
  let mut t_r1 : Wide := default
  let mut t_r2 : Wide := default
  let mut t_r3 : Wide := default
  let mut t_r4 : Wide := default
  let mut t_r5 : Wide := default
  let mut t_r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let zero_13 := v12
  zero := zero_13
  let v14 := (UInt256.ofNat 4)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := a.l0
    let v18 := v17
    let v19 := b.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let t_22 := v21
    let v23 ← downcast_truncate_u128_u64 t_22
    let low := v23
    let v25 := self.inverse
    let v26 := IntTy.mulW (.u 64) low v25
    let q := v26
    let v28 := q
    let v29 := self.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let product := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := t_22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := product >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (low ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v3 => do
      let mut zero := zero
      let mut t_r1 := t_r1
      let mut t_r2 := t_r2
      let mut t_r3 := t_r3
      let mut t_r4 := t_r4
      let mut t_r5 := t_r5
      let mut t_r6 := t_r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let sum_45 := v44
      let v46 := self.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ sum_45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v4 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        let reduced := v4
        let v55 ← downcast_truncate_u128_u64 reduced
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := self.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) sum_45 v50
        let v52 := v51
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v52
      else
        let sum_53 := sum_45
        b9 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 sum_53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 4)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := b.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_4 self a v65 zero v66
      let v69 := v68.fst
      let t_70 := v69
      t_r1 := t_70
      let v71 := v68.snd
      let top_72 := v71
      let v73 := b.l1
      let v75 ← Small_row2_4 self a v73 t_r1 top_72
      let v76 := v75.fst
      let t_77 := v76
      let v78 := v75.snd
      let top_79 := v78
      let v80 := t_77.l0
      let v81 := self.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let d0_85 := v84
      let v86 := v83.snd
      let borrow_87 := v86
      let v88 := t_77.l1
      let v89 := self.m.l1
      let v90 ← sbb v88 v89 borrow_87
      let v91 := v90.fst
      let d1_92 := v91
      let v93 := v90.snd
      let borrow_94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb top_79 v95 borrow_94
      let v97 := v96.snd
      let borrow_98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (borrow_98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v5 => do
        let mut zero := zero
        let mut t_r1 := t_r1
        let mut t_r2 := t_r2
        let mut t_r3 := t_r3
        let mut t_r4 := t_r4
        let mut t_r5 := t_r5
        let mut t_r6 := t_r6
        return v5
      if v100 then
        let t_101 := t_77
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk d0_85 d1_92 v102 v103)
        let v105 := v104
        b15 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 4)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := b.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_4 self a v111 zero v112
        let v115 := v114.fst
        let t_116 := v115
        t_r2 := t_116
        let v117 := v114.snd
        let top_118 := v117
        let v119 := b.l1
        let v121 ← Small_row3_4 self a v119 t_r2 top_118
        let v122 := v121.fst
        let t_123 := v122
        t_r3 := t_123
        let v124 := v121.snd
        let top_125 := v124
        let v126 := b.l2
        let v128 ← Small_row3_4 self a v126 t_r3 top_125
        let v129 := v128.fst
        let t_130 := v129
        let v131 := v128.snd
        let top_132 := v131
        let v133 := t_130.l0
        let v134 := self.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let d0_138 := v137
        let v139 := v136.snd
        let borrow_140 := v139
        let v141 := t_130.l1
        let v142 := self.m.l1
        let v143 ← sbb v141 v142 borrow_140
        let v144 := v143.fst
        let d1_145 := v144
        let v146 := v143.snd
        let borrow_147 := v146
        let v148 := t_130.l2
        let v149 := self.m.l2
        let v150 ← sbb v148 v149 borrow_147
        let v151 := v150.fst
        let d2_152 := v151
        let v153 := v150.snd
        let borrow_154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb top_132 v155 borrow_154
        let v157 := v156.snd
        let borrow_158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (borrow_158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v6 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v6
        if v160 then
          let t_161 := t_130
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk d0_138 d1_145 d2_152 v162)
          let v164 := v163
          b21 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := b.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_4 self a v167 zero v168
        let v171 := v170.fst
        let t_172 := v171
        t_r4 := t_172
        let v173 := v170.snd
        let top_174 := v173
        let v175 := b.l1
        let v177 ← Small_row4_4 self a v175 t_r4 top_174
        let v178 := v177.fst
        let t_179 := v178
        t_r5 := t_179
        let v180 := v177.snd
        let top_181 := v180
        let v182 := b.l2
        let v184 ← Small_row4_4 self a v182 t_r5 top_181
        let v185 := v184.fst
        let t_186 := v185
        t_r6 := t_186
        let v187 := v184.snd
        let top_188 := v187
        let v189 := b.l3
        let v191 ← Small_row4_4 self a v189 t_r6 top_188
        let v192 := v191.fst
        let t_193 := v192
        let v194 := v191.snd
        let top_195 := v194
        let v196 := t_193.l0
        let v197 := self.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let d0_201 := v200
        let v202 := v199.snd
        let borrow_203 := v202
        let v204 := t_193.l1
        let v205 := self.m.l1
        let v206 ← sbb v204 v205 borrow_203
        let v207 := v206.fst
        let d1_208 := v207
        let v209 := v206.snd
        let borrow_210 := v209
        let v211 := t_193.l2
        let v212 := self.m.l2
        let v213 ← sbb v211 v212 borrow_210
        let v214 := v213.fst
        let d2_215 := v214
        let v216 := v213.snd
        let borrow_217 := v216
        let v218 := t_193.l3
        let v219 := self.m.l3
        let v220 ← sbb v218 v219 borrow_217
        let v221 := v220.fst
        let d3 := v221
        let v223 := v220.snd
        let borrow_224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb top_195 v225 borrow_224
        let v227 := v226.snd
        let borrow_228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (borrow_228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v7 => do
          let mut zero := zero
          let mut t_r1 := t_r1
          let mut t_r2 := t_r2
          let mut t_r3 := t_r3
          let mut t_r4 := t_r4
          let mut t_r5 := t_r5
          let mut t_r6 := t_r6
          return v7
        if v230 then
          let t_231 := t_193
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 t_231
        else
          let v232 := (Wide.mk d0_201 d1_208 d2_215 d3)
          let v233 := v232
          b24 zero t_r1 t_r2 t_r3 t_r4 t_r5 t_r6 v233

def Small_r2_4 (self : Small_4) : FeM Wide := do
  let mut x : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 4)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v1 => do
    let mut x := x
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let top := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := top
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let shift := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 shift
    let bit_38 := v37
    let v39 := (UInt256.ofNat 4)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v4 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 4)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v5 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 4)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v6 => do
          let mut x := x
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 4)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v7 => do
            let mut x := x
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let x_64 := v63
            x := x_64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 shift
            let v67 ← downcast_truncate_u64_usize v66
            let doublings := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 4)
            let v71 := doublings + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut x := x
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let x_81 := x
                let x_82 := x_81
                r2 := x_82
                let v84 ← Small_double_4 self r2
                let x_85 := v84
                x := x_85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 x r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (x : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut x := x
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let x_101 := x
                    let x_102 := x_101
                    r4 := x_102
                    let x_103 := x
                    let x_104 := x_103
                    r5 := x_104
                    let v107 ← Small_mul_4 self r4 r5
                    let x_108 := v107
                    x := x_108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 x r1 r2 r3 r4 r5 v112
                  else
                    let x_113 := x
                    return x_113
                partial_fixpoint
                b26 x r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 x r1 r2 r3 r4 r5 v76
          if v59 then
            let bit_60 := bit_38
            b21 x r1 r2 r3 r4 r5 bit_60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 x r1 r2 r3 r4 r5 v62
        if v53 then
          let bit_54 := bit_38
          b18 x r1 r2 r3 r4 r5 bit_54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 x r1 r2 r3 r4 r5 v56
      if v47 then
        let bit_48 := bit_38
        b15 x r1 r2 r3 r4 r5 bit_48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 x r1 r2 r3 r4 r5 v50
    if v41 then
      let bit_42 := bit_38
      b12 x r1 r2 r3 r4 r5 bit_42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 x r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := self.m.l0
    let v14 := v13
    b3 x r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 4)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v2 => do
      let mut x := x
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 x r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := self.m.l1
      let v19 := v18
      b6 x r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 4)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun x r1 r2 r3 r4 r5 v3 => do
        let mut x := x
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 x r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := self.m.l2
        let v24 := v23
        b9 x r1 r2 r3 r4 r5 v24
      else
        let v25 := self.m.l3
        let v26 := v25
        b9 x r1 r2 r3 r4 r5 v26

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
  let v7 := (UInt256.ofNat 0)
  let v8 ← MemArray_index_ref_u64 value v7
  let low_9 ← mloadInt (.u 64) v8
  let low_10 := low_9
  inverse := low_10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let inverse_22 := inverse
      let inverse_23 := inverse_22
      let v24 := (UInt256.ofNat 2)
      let inverse_25 := inverse
      let v26 := IntTy.mulW (.u 64) low_9 inverse_25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) inverse_23 v27
      let inverse_29 := v28
      inverse := inverse_29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 4)
      let v35 ← Wide_of value v34
      let v36 := (UInt256.ofNat 0)
      let inverse_37 := inverse
      let v38 := IntTy.subW (.u 64) v36 inverse_37
      let v39 := (Small_4.mk v35 v38)
      let small_40 := v39
      small := small_40
      let v41 := exponent.total
      let v42 := exponent.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let width := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := width - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let entries := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let table_58 := v57
      table := table_58
      let v59 := (UInt256.ofNat 4)
      let v60 ← Wide_of base v59
      r9 := v60
      let v62 ← Small_r2_4 small
      r10 := v62
      let v66 ← Small_mul_4 small r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      table := (table.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < entries.toNat)
      let b7 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := table[v106.toNat]!
        let power_108 := v107
        power := power_108
        let v109 := exponent.top
        let index_110 := v109
        index := index_110
        let v111 := false
        let started_112 := v111
        started := started_112
        let rec b12 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let index_113 := index
          let index_114 := index_113
          let v115 := exponent.total
          let v116 := decide (index_114.toNat < v115.toNat)
          if v116 then
            let index_117 := index
            let index_118 := index_117
            let v119 ← Exponent_window exponent index_118 width
            let window := v119
            let started_121 := started
            let rec b17 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := window.odd
              let b24 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := window.«end»
                let index_174 := v173
                index := index_174
                let v175 := ()
                b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let started_152 := started
                let b27 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let power_165 := v6
                  power := power_165
                  let v166 := ()
                  let v167 := true
                  let started_168 := v167
                  started := started_168
                  let v169 := ()
                  let v170 := v169
                  b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if started_152 then
                  let power_153 := power
                  let power_154 := power_153
                  r18 := power_154
                  let v155 := window.entry
                  checkIndex v155 64
                  let v156 := table[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_4 small r18 r19
                  let v161 := v160
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := window.entry
                  checkIndex v162 64
                  let v163 := table[v162.toNat]!
                  let v164 := v163
                  b27 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if started_121 then
              let index_122 := index
              let index_123 := index_122
              let v124 := window.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk index_123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let power_134 := power
                  let power_135 := power_134
                  r16 := power_135
                  let power_136 := power
                  let power_137 := power_136
                  r17 := power_137
                  let v141 ← Small_mul_4 small r16 r17
                  let power_142 := v141
                  power := power_142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let power_176 := power
            let power_177 := power_176
            r20 := power_177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_4 small r20 r21
            let out := v186
            let v188 := (UInt256.ofNat 4)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let result_192 := v191
              result := result_192
              let v193 := out.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 result v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              result := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 4)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 4)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 4)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let result_231 := result
                    let result_232 := result_231
                    let v233 := Option_MemArray_u64.Some result_232
                    return v233
                  if v222 then
                    let v223 := out.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 result v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    result := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := out.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 result v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  result := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := out.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 result v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                result := v204_0
                let v205 := ()
                let v206 := v205
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := table[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := table[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_4 small r11 r12
        let squared_79 := v78
        squared := squared_79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 entries)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (inverse : UInt256) (small : Small_4) (table : (Vector Wide 64)) (squared : Wide) (power : Wide) (index : UInt256) (started : Bool) (result : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let k := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := k - v91
            checkIndex v92 64
            let v93 := table[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_4 small r14 squared
            checkIndex k 64
            table := (table.set! k.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 inverse small table squared power index started result r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def small_power_of (exponent : Exponent) (base : MemArray_u64) (value : MemArray_u64) (n_3 : UInt256) : FeM Option_MemArray_u64 := do
  let n_7 := n_3
  let v8 := (UInt256.ofNat 1)
  let v9 := decide (n_7 = v8)
  let b3 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v4 => do
    return v4
  if v9 then
    let v10 ← small_power_1 exponent base value
    let v11 := v10
    b3 v11
  else
    let n_12 := n_3
    let v13 := (UInt256.ofNat 2)
    let v14 := decide (n_12 = v13)
    let b6 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v5 => do
      let v25 := v5
      b3 v25
    if v14 then
      let v15 ← small_power_2 exponent base value
      let v16 := v15
      b6 v16
    else
      let n_17 := n_3
      let v18 := (UInt256.ofNat 3)
      let v19 := decide (n_17 = v18)
      let b9 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v6 => do
        let v24 := v6
        b6 v24
      if v19 then
        let v20 ← small_power_3 exponent base value
        let v21 := v20
        b9 v21
      else
        let v22 ← small_power_4 exponent base value
        let v23 := v22
        b9 v23

def Montgomery_new (value : MemArray_u64) (len_0 : UInt256) : FeM Montgomery := do
  let mut value := value
  let mut inverse : UInt256 := default
  let mut r2 : Range_Known_0_Known_5 := default
  let v2 := (UInt256.ofNat 0)
  let v4 ← MemArray_index_ref_u64 value v2
  let low_5 ← mloadInt (.u 64) v4
  let low_6 := low_5
  inverse := low_6
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 5)
  let v9 := ()
  let v10 := (Range_Known_0_Known_5.mk v9 v9)
  r2 := v10
  let v12 ← Range_start_0_5 r2
  let v13 := v12
  let rec b1 (value : MemArray_u64) (inverse : UInt256) (r2 : Range_Known_0_Known_5) (v1 : UInt256) : FeM Montgomery := do
    let mut value := value
    let mut inverse := inverse
    let mut r2 := r2
    let v15 ← Range_has_0_5 r2 v1
    if v15 then
      let v17 ← Range_at_0_5 r2 v1
      let inverse_18 := inverse
      let inverse_19 := inverse_18
      let v20 := (UInt256.ofNat 2)
      let inverse_21 := inverse
      let v22 := IntTy.mulW (.u 64) low_5 inverse_21
      let v23 := IntTy.subW (.u 64) v20 v22
      let v24 := IntTy.mulW (.u 64) inverse_19 v23
      let inverse_25 := v24
      inverse := inverse_25
      let v26 := ()
      let v28 ← Range_next_0_5 r2 v1
      let v29 := v28
      b1 value inverse r2 v29
    else
      let value_30 := value
      let value_31 := value_30
      let len_32 := len_0
      let v33 := (UInt256.ofNat 0)
      let inverse_34 := inverse
      let v35 := IntTy.subW (.u 64) v33 inverse_34
      let v36 := (Montgomery.mk value_31 len_32 v35)
      return v36
  partial_fixpoint
  b1 value inverse r2 v13

def rem_2by1 (u1_0 : UInt256) (u0_1 : UInt256) (d_2 : UInt256) (v_3 : UInt256) : FeM UInt256 := do
  let mut r : UInt256 := default
  let v_4 := v_3
  let v5 := v_4
  let u1_6 := u1_0
  let v7 := u1_6
  let v8 := IntTy.mulW (.u 128) v5 v7
  let u1_9 := u1_0
  let v10 := u1_9
  let v11 := (UInt256.ofNat 64)
  let v12 := IntTy.shl (.u 128) v10 v11
  let u0_13 := u0_1
  let v14 := u0_13
  let v15 := v12 ||| v14
  let v16 := IntTy.addW (.u 128) v8 v15
  let q := v16
  let v18 := (UInt256.ofNat 64)
  let v19 := q >>> v18
  let v20 ← downcast_truncate_u128_u64 v19
  let high := v20
  let v22 := (UInt256.ofNat 1)
  let v23 := IntTy.addW (.u 64) high v22
  let q1 := v23
  let v25 ← downcast_truncate_u128_u64 q
  let q0 := v25
  let u0_27 := u0_1
  let d_28 := d_2
  let v29 := IntTy.mulW (.u 64) q1 d_28
  let v30 := IntTy.subW (.u 64) u0_27 v29
  let r_31 := v30
  r := r_31
  let r_32 := r
  let r_33 := r_32
  let v34 := decide (q0.toNat < r_33.toNat)
  let b3 : UInt256 → FeM UInt256 := fun r => do
    let mut r := r
    let r_43 := r
    let r_44 := r_43
    let d_45 := d_2
    let v46 := decide (d_45.toNat ≤ r_44.toNat)
    let b6 : UInt256 → FeM UInt256 := fun r => do
      let mut r := r
      let r_55 := r
      return r_55
    if v46 then
      let d_47 := d_2
      let r_48 := r
      let v49 := IntTy.subW (.u 64) r_48 d_47
      let r_50 := v49
      r := r_50
      let v51 := ()
      let v52 := v51
      b6 r
    else
      let v53 := ()
      let v54 := v53
      b6 r
  if v34 then
    let d_35 := d_2
    let r_36 := r
    let v37 := IntTy.addW (.u 64) r_36 d_35
    let r_38 := v37
    r := r_38
    let v39 := ()
    let v40 := v39
    b3 r
  else
    let v41 := ()
    let v42 := v41
    b3 r

def div_3by2_k2 (q0 : UInt256) (d : UInt256) (v52 : UInt256) (q1 : UInt256) (r : UInt256) : FeM UInt256 := do
  let mut q1 := q1
  let mut r := r
  let v53 := IntTy.subW (.u 128) v52 d
  let r_54 := v53
  r := r_54
  let v55 := (UInt256.ofNat 1)
  let q1_56 := q1
  let v57 := IntTy.addW (.u 64) q1_56 v55
  let q1_58 := v57
  q1 := q1_58
  let v59 := ()
  let r_60 := r
  let r_61 := r_60
  let v62 := (UInt256.ofNat 64)
  let v63 := r_61 >>> v62
  let v64 ← downcast_truncate_u128_u64 v63
  let high := v64
  let v66 := decide (q0.toNat ≤ high.toNat)
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun q1 r => do
    let mut q1 := q1
    let mut r := r
    let r_79 := r
    let r_80 := r_79
    let v81 := decide (d.toNat ≤ r_80.toNat)
    let b6 : UInt256 → UInt256 → FeM UInt256 := fun q1 r => do
      let mut q1 := q1
      let mut r := r
      let q1_90 := q1
      return q1_90
    if v81 then
      let v82 := (UInt256.ofNat 1)
      let q1_83 := q1
      let v84 := IntTy.addW (.u 64) q1_83 v82
      let q1_85 := v84
      q1 := q1_85
      let v86 := ()
      let v87 := v86
      b6 q1 r
    else
      let v88 := ()
      let v89 := v88
      b6 q1 r
  if v66 then
    let v67 := (UInt256.ofNat 1)
    let q1_68 := q1
    let v69 := IntTy.subW (.u 64) q1_68 v67
    let q1_70 := v69
    q1 := q1_70
    let v71 := ()
    let r_72 := r
    let v73 := IntTy.addW (.u 128) r_72 d
    let r_74 := v73
    r := r_74
    let v75 := ()
    let v76 := v75
    b3 q1 r
  else
    let v77 := ()
    let v78 := v77
    b3 q1 r

def div_3by2_k1 (u0_2 : UInt256) (d1_3 : UInt256) (d0_4 : UInt256) (q0 : UInt256) (u1_26 : UInt256) (q1_28 : UInt256) (q1 : UInt256) (r : UInt256) : FeM UInt256 := do
  let mut q1 := q1
  let mut r := r
  let d1_29 := d1_3
  let v30 := IntTy.mulW (.u 64) q1_28 d1_29
  let v31 := IntTy.subW (.u 64) u1_26 v30
  let r1 := v31
  let d1_33 := d1_3
  let v34 := d1_33
  let v35 := (UInt256.ofNat 64)
  let v36 := IntTy.shl (.u 128) v34 v35
  let d0_37 := d0_4
  let v38 := d0_37
  let v39 := v36 ||| v38
  let d := v39
  let v41 := r1
  let v42 := (UInt256.ofNat 64)
  let v43 := IntTy.shl (.u 128) v41 v42
  let u0_44 := u0_2
  let v45 := u0_44
  let v46 := v43 ||| v45
  let d0_47 := d0_4
  let v48 := d0_47
  let q1_49 := q1
  let v50 := q1_49
  let v51 := IntTy.mulW (.u 128) v48 v50
  let v52 := IntTy.subW (.u 128) v46 v51
  div_3by2_k2 q0 d v52 q1 r

def div_3by2 (u2_0 : UInt256) (u1_1 : UInt256) (u0_2 : UInt256) (d1_3 : UInt256) (d0_4 : UInt256) (v_5 : UInt256) : FeM UInt256 := do
  let mut q1 : UInt256 := default
  let mut r : UInt256 := default
  let v_6 := v_5
  let v7 := v_6
  let u2_8 := u2_0
  let v9 := u2_8
  let v10 := IntTy.mulW (.u 128) v7 v9
  let u2_11 := u2_0
  let v12 := u2_11
  let v13 := (UInt256.ofNat 64)
  let v14 := IntTy.shl (.u 128) v12 v13
  let u1_15 := u1_1
  let v16 := u1_15
  let v17 := v14 ||| v16
  let v18 := IntTy.addW (.u 128) v10 v17
  let q := v18
  let v20 := (UInt256.ofNat 64)
  let v21 := q >>> v20
  let v22 ← downcast_truncate_u128_u64 v21
  let q1_23 := v22
  q1 := q1_23
  let v24 ← downcast_truncate_u128_u64 q
  let q0 := v24
  let u1_26 := u1_1
  let q1_27 := q1
  let q1_28 := q1_27
  div_3by2_k1 u0_2 d1_3 d0_4 q0 u1_26 q1_28 q1 r

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
  let v11 ← downcast_truncate_u64_usize len
  let count_12 := v11
  let v13 := self.len
  let v14 ← downcast_truncate_u64_usize v13
  let n := v14
  let v17 ← MemArray_len_u64 work_1
  let v18 := decide (count_12.toNat < v17.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v3 => do
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
    let v30 ← «assert» v3
    let v32 ← MemArray_ptr_u64 work_1
    let wp := v32
    let v36 ← MemArray_ptr_u64 self.limbs
    let mp := v36
    let v38 := decide (count_12.toNat < n.toNat)
    if v38 then
      let v39 := (UInt256.ofNat 1)
      let v40 := n + v39
      let v41 := ()
      let v42 := (Range_Unknown_Unknown.mk count_12 v40)
      r7 := v42
      let v44 ← Range_Unknown_Unknown_start r7
      let v45 := v44
      let rec b9 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v4 : UInt256) : FeM (Unit × MemArray_u64) := do
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
        let v47 ← Range_Unknown_Unknown_has r7 v4
        if v47 then
          let v49 ← Range_Unknown_Unknown_at r7 v4
          let i_50 := v49
          let v51 := (UInt256.ofNat 0)
          let v52 ← limb wp i_50
          mstoreInt (.u 64) v52 v51
          let v53 := ()
          let v55 ← Range_Unknown_Unknown_next r7 v4
          let v56 := v55
          b9 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v56
        else
          return ((), work_1)
      partial_fixpoint
      b9 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v45
    else
      let v57 := ()
      let v58 := v57
      let v59 := self.shift
      let shift := v59
      let v61 := (UInt256.ofNat 0)
      let v62 ← limb wp count_12
      mstoreInt (.u 64) v62 v61
      let v63 := ()
      let v64 := (UInt256.ofNat 0)
      let v65 := decide (shift ≠ v64)
      let b15 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 => do
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
        let v111 := (UInt256.ofNat 1)
        let v112 := decide (n = v111)
        if v112 then
          let v113 ← mloadInt (.u 64) mp
          let d := v113
          let v115 := (UInt256.ofNat 0)
          let r_116 := v115
          r := r_116
          let v117 := (UInt256.ofNat 1)
          let v118 := count_12 + v117
          let i_119 := v118
          i_r2 := i_119
          let rec b22 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
            let i_120 := i_r2
            let i_121 := i_120
            let v122 := (UInt256.ofNat 0)
            let v123 := decide (v122.toNat < i_121.toNat)
            if v123 then
              let v124 := (UInt256.ofNat 1)
              let i_125 := i_r2
              let v126 := i_125 - v124
              let i_127 := v126
              i_r2 := i_127
              let v128 := ()
              let r_129 := r
              let r_130 := r_129
              let i_131 := i_r2
              let i_132 := i_131
              let v133 ← limb wp i_132
              let v134 ← mloadInt (.u 64) v133
              let v135 := self.reciprocal
              let v136 ← rem_2by1 r_130 v134 d v135
              let r_137 := v136
              r := r_137
              let v138 := ()
              let v139 := (UInt256.ofNat 0)
              let i_140 := i_r2
              let i_141 := i_140
              let v142 ← limb wp i_141
              mstoreInt (.u 64) v142 v139
              let v143 := ()
              b22 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
            else
              let r_144 := r
              let r_145 := r_144
              let v146 := r_145 >>> shift
              mstoreInt (.u 64) wp v146
              let v147 := ()
              return ((), work_1)
          partial_fixpoint
          b22 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
        else
          let v148 := ()
          let v149 := v148
          let v150 := (UInt256.ofNat 1)
          let v151 := n - v150
          let v152 ← limb mp v151
          let v153 ← mloadInt (.u 64) v152
          let d1 := v153
          let v155 := (UInt256.ofNat 2)
          let v156 := n - v155
          let v157 ← limb mp v156
          let v158 ← mloadInt (.u 64) v157
          let d0 := v158
          let v160 := (UInt256.ofNat 1)
          let v161 := count_12 + v160
          let v162 := v161 - n
          let j_163 := v162
          j := j_163
          let rec b25 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
            let j_164 := j
            let j_165 := j_164
            let v166 := (UInt256.ofNat 0)
            let v167 := decide (v166.toNat < j_165.toNat)
            if v167 then
              let v168 := (UInt256.ofNat 1)
              let j_169 := j
              let v170 := j_169 - v168
              let j_171 := v170
              j := j_171
              let v172 := ()
              let j_173 := j
              let j_174 := j_173
              let v175 := j_174 + n
              let v176 ← limb wp v175
              let v177 ← mloadInt (.u 64) v176
              let high_178 := v177
              let j_179 := j
              let j_180 := j_179
              let v181 := j_180 + n
              let v182 := (UInt256.ofNat 1)
              let v183 := v181 - v182
              let v184 ← limb wp v183
              let v185 ← mloadInt (.u 64) v184
              let next := v185
              let v187 := decide (high_178 = d1)
              let b30 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → UInt256 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v5 => do
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
                let qhat := v5
                let v202 := (UInt256.ofNat 0)
                let carry_203 := v202
                carry := carry_203
                let v204 := (UInt256.ofNat 0)
                let borrow_205 := v204
                borrow := borrow_205
                let v206 := (UInt256.ofNat 0)
                let v207 := ()
                let v208 := (Range_Known_0_Unknown.mk v207 n)
                r8 := v208
                let v210 ← Range_start_0 r8
                let v211 := v210
                let rec b32 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                  let v213 ← Range_has_0 r8 v6
                  if v213 then
                    let v215 ← Range_at_0 r8 v6
                    let i_216 := v215
                    let v217 := qhat
                    let v218 ← limb mp i_216
                    let v219 ← mloadInt (.u 64) v218
                    let v220 := v219
                    let v221 := IntTy.mulW (.u 128) v217 v220
                    let carry_222 := carry
                    let v223 := carry_222
                    let v224 := IntTy.addW (.u 128) v221 v223
                    let product := v224
                    let v226 := (UInt256.ofNat 64)
                    let v227 := product >>> v226
                    let v228 ← downcast_truncate_u128_u64 v227
                    let carry_229 := v228
                    carry := carry_229
                    let v230 := ()
                    let v231 ← downcast_truncate_u128_u64 product
                    let low_232 := v231
                    let j_233 := j
                    let v234 := i_216 + j_233
                    let v235 ← limb wp v234
                    let v236 ← mloadInt (.u 64) v235
                    let v237 := v236
                    let v238 := low_232
                    let v239 := IntTy.subW (.u 128) v237 v238
                    let borrow_240 := borrow
                    let v241 := borrow_240
                    let v242 := IntTy.subW (.u 128) v239 v241
                    let difference_243 := v242
                    let v244 ← downcast_truncate_u128_u64 difference_243
                    let j_245 := j
                    let v246 := i_216 + j_245
                    let v247 ← limb wp v246
                    mstoreInt (.u 64) v247 v244
                    let v248 := ()
                    let v249 := (UInt256.ofNat 127)
                    let v250 := difference_243 >>> v249
                    let v251 ← downcast_truncate_u128_u64 v250
                    let borrow_252 := v251
                    borrow := borrow_252
                    let v253 := ()
                    let v255 ← Range_next_0 r8 v6
                    let v256 := v255
                    b32 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v256
                  else
                    let v257 := high_178
                    let carry_258 := carry
                    let v259 := carry_258
                    let v260 := IntTy.subW (.u 128) v257 v259
                    let borrow_261 := borrow
                    let v262 := borrow_261
                    let v263 := IntTy.subW (.u 128) v260 v262
                    let difference_264 := v263
                    let v265 ← downcast_truncate_u128_u64 difference_264
                    let j_266 := j
                    let j_267 := j_266
                    let v268 := j_267 + n
                    let v269 ← limb wp v268
                    mstoreInt (.u 64) v269 v265
                    let v270 := ()
                    let v271 := (UInt256.ofNat 127)
                    let v272 := difference_264 >>> v271
                    let v273 := (UInt256.ofNat 0)
                    let v274 := decide (v272 ≠ v273)
                    let rec b38 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
                      b25 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
                    partial_fixpoint
                    if v274 then
                      let v275 := (UInt256.ofNat 0)
                      let add_276 := v275
                      add := add_276
                      let v277 := (UInt256.ofNat 0)
                      let v278 := ()
                      let v279 := (Range_Known_0_Unknown.mk v278 n)
                      r9 := v279
                      let v281 ← Range_start_0 r9
                      let v282 := v281
                      let rec b39 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                        let v284 ← Range_has_0 r9 v7
                        if v284 then
                          let v286 ← Range_at_0 r9 v7
                          let i_287 := v286
                          let j_288 := j
                          let v289 := i_287 + j_288
                          let v290 ← limb wp v289
                          let v291 ← mloadInt (.u 64) v290
                          let v292 := v291
                          let v293 ← limb mp i_287
                          let v294 ← mloadInt (.u 64) v293
                          let v295 := v294
                          let v296 := IntTy.addW (.u 128) v292 v295
                          let add_297 := add
                          let v298 := add_297
                          let v299 := IntTy.addW (.u 128) v296 v298
                          let sum := v299
                          let v301 ← downcast_truncate_u128_u64 sum
                          let j_302 := j
                          let v303 := i_287 + j_302
                          let v304 ← limb wp v303
                          mstoreInt (.u 64) v304 v301
                          let v305 := ()
                          let v306 := (UInt256.ofNat 64)
                          let v307 := sum >>> v306
                          let v308 ← downcast_truncate_u128_u64 v307
                          let add_309 := v308
                          add := add_309
                          let v310 := ()
                          let v312 ← Range_next_0 r9 v7
                          let v313 := v312
                          b39 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v313
                        else
                          let j_314 := j
                          let j_315 := j_314
                          let v316 := j_315 + n
                          let v317 ← limb wp v316
                          let v318 ← mloadInt (.u 64) v317
                          let add_319 := add
                          let v320 := IntTy.addW (.u 64) v318 add_319
                          let j_321 := j
                          let j_322 := j_321
                          let v323 := j_322 + n
                          let v324 ← limb wp v323
                          mstoreInt (.u 64) v324 v320
                          let v325 := ()
                          let v326 := v325
                          b38 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
                      partial_fixpoint
                      b39 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v282
                    else
                      let v327 := ()
                      let v328 := v327
                      b38 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
                partial_fixpoint
                b32 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v211
              let b29 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 => do
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
                let j_191 := j
                let j_192 := j_191
                let v193 := j_192 + n
                let v194 := (UInt256.ofNat 2)
                let v195 := v193 - v194
                let v196 ← limb wp v195
                let v197 ← mloadInt (.u 64) v196
                let v198 := self.reciprocal
                let v199 ← div_3by2 high_178 next v197 d1 d0 v198
                let v200 := v199
                b30 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v200
              if v187 then
                let v188 := decide (next = d0)
                if v188 then
                  let v189 := (UInt256.ofNat 18446744073709551615)
                  let v190 := v189
                  b30 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v190
                else
                  b29 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
              else
                b29 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
            else
              let v329 := (UInt256.ofNat 0)
              let v330 := ()
              let v331 := (Range_Known_0_Unknown.mk v330 n)
              r10 := v331
              let v333 ← Range_start_0 r10
              let v334 := v333
              let rec b43 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v8 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                let v336 ← Range_has_0 r10 v8
                if v336 then
                  let v338 ← Range_at_0 r10 v8
                  let i_339 := v338
                  let v340 ← limb wp i_339
                  let v341 ← mloadInt (.u 64) v340
                  let v342 := v341 >>> shift
                  let low_343 := v342
                  let v344 := (UInt256.ofNat 0)
                  let v345 := decide (shift = v344)
                  let b49 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → UInt256 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v9 => do
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
                    let high_356 := v9
                    let v357 := low_343 ||| high_356
                    let v358 ← limb wp i_339
                    mstoreInt (.u 64) v358 v357
                    let v359 := ()
                    let v361 ← Range_next_0 r10 v8
                    let v362 := v361
                    b43 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v362
                  if v345 then
                    let v346 := (UInt256.ofNat 0)
                    let v347 := v346
                    b49 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v347
                  else
                    let v348 := (UInt256.ofNat 1)
                    let v349 := i_339 + v348
                    let v350 ← limb wp v349
                    let v351 ← mloadInt (.u 64) v350
                    let v352 := (UInt256.ofNat 64)
                    let v353 := IntTy.subW (.u 64) v352 shift
                    let v354 := IntTy.shl (.u 64) v351 v353
                    let v355 := v354
                    b49 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v355
                else
                  let v363 := (UInt256.ofNat 1)
                  let v364 := count_12 + v363
                  let v365 := ()
                  let v366 := (Range_Unknown_Unknown.mk n v364)
                  r11 := v366
                  let v368 ← Range_Unknown_Unknown_start r11
                  let v369 := v368
                  let rec b50 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) (v10 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                    let v371 ← Range_Unknown_Unknown_has r11 v10
                    if v371 then
                      let v373 ← Range_Unknown_Unknown_at r11 v10
                      let i_374 := v373
                      let v375 := (UInt256.ofNat 0)
                      let v376 ← limb wp i_374
                      mstoreInt (.u 64) v376 v375
                      let v377 := ()
                      let v379 ← Range_Unknown_Unknown_next r11 v10
                      let v380 := v379
                      b50 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v380
                    else
                      let v381 := ()
                      return ((), work_1)
                  partial_fixpoint
                  b50 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v369
              partial_fixpoint
              b43 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v334
          partial_fixpoint
          b25 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
      if v65 then
        let count_66 := count_12
        i_r0 := count_66
        let rec b16 (i_r0 : UInt256) (r : UInt256) (i_r2 : UInt256) (j : UInt256) (carry : UInt256) (borrow : UInt256) (add : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (work_1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
          let i_67 := i_r0
          let i_68 := i_67
          let v69 := (UInt256.ofNat 0)
          let v70 := decide (v69.toNat < i_68.toNat)
          if v70 then
            let i_71 := i_r0
            let i_72 := i_71
            let v73 ← limb wp i_72
            let v74 ← mloadInt (.u 64) v73
            let i_75 := i_r0
            let i_76 := i_75
            let v77 := (UInt256.ofNat 1)
            let v78 := i_76 - v77
            let v79 ← limb wp v78
            let v80 ← mloadInt (.u 64) v79
            let v81 := (UInt256.ofNat 64)
            let v82 := IntTy.subW (.u 64) v81 shift
            let v83 := v80 >>> v82
            let v84 := v74 ||| v83
            let i_85 := i_r0
            let i_86 := i_85
            let v87 ← limb wp i_86
            mstoreInt (.u 64) v87 v84
            let v88 := ()
            let i_89 := i_r0
            let i_90 := i_89
            let v91 := (UInt256.ofNat 1)
            let v92 := i_90 - v91
            let v93 ← limb wp v92
            let v94 ← mloadInt (.u 64) v93
            let v95 := IntTy.shl (.u 64) v94 shift
            let i_96 := i_r0
            let i_97 := i_96
            let v98 := (UInt256.ofNat 1)
            let v99 := i_97 - v98
            let v100 ← limb wp v99
            mstoreInt (.u 64) v100 v95
            let v101 := ()
            let v102 := (UInt256.ofNat 1)
            let i_103 := i_r0
            let v104 := i_103 - v102
            let i_105 := v104
            i_r0 := i_105
            let v106 := ()
            b16 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
          else
            let v107 := ()
            let v108 := v107
            b15 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
        partial_fixpoint
        b16 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
      else
        let v109 := ()
        let v110 := v109
        b15 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
  let b2 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 => do
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
    let v28 := false
    let v29 := v28
    b3 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v29
  if v18 then
    let v20 ← MemArray_len_u64 work_1
    let v21 := decide (n.toNat < v20.toNat)
    if v21 then
      let v24 ← MemArray_len_u64 self.limbs
      let v25 := decide (n.toNat ≤ v24.toNat)
      if v25 then
        let v26 := true
        let v27 := v26
        b3 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1 v27
      else
        b2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
    else
      b2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1
  else
    b2 i_r0 r i_r2 j carry borrow add r7 r8 r9 r10 r11 work_1

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
  let v9 := self.len
  let v10 ← downcast_truncate_u64_usize v9
  let n_11 := v10
  let v12 ← MemSlice_len_u64 a
  let v13 := decide (n_11.toNat ≤ v12.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v4 => do
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
    let v29 ← «assert» v4
    let v30 ← MemSlice_ptr_u64 a
    let ap := v30
    let v32 ← MemSlice_ptr_u64 b
    let bp := v32
    let v36 ← MemArray_ptr_u64 self.limbs
    let mp := v36
    let v39 ← MemArray_ptr_u64 t_3
    let tp := v39
    let v41 := (UInt256.ofNat 0)
    let v42 := (UInt256.ofNat 1)
    let v43 := n_11 + v42
    let v44 := ()
    let v45 := (Range_Known_0_Unknown.mk v44 v43)
    r7 := v45
    let v47 ← Range_start_0 r7
    let v48 := v47
    let rec b7 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v5 : UInt256) : FeM (Unit × MemArray_u64) := do
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
      let v50 ← Range_has_0 r7 v5
      if v50 then
        let v52 ← Range_at_0 r7 v5
        let j_53 := v52
        let v54 := (UInt256.ofNat 0)
        let v55 ← limb tp j_53
        mstoreInt (.u 64) v55 v54
        let v56 := ()
        let v58 ← Range_next_0 r7 v5
        let v59 := v58
        b7 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v59
      else
        let v60 := (UInt256.ofNat 0)
        let v61 := ()
        let v62 := (Range_Known_0_Unknown.mk v61 n_11)
        r8 := v62
        let v64 ← Range_start_0 r8
        let v65 := v64
        let rec b11 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
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
          let v67 ← Range_has_0 r8 v6
          if v67 then
            let v69 ← Range_at_0 r8 v6
            let i := v69
            let v71 ← limb bp i
            let v72 ← mloadInt (.u 64) v71
            let v73 := v72
            let factor := v73
            let v75 ← mloadInt (.u 64) tp
            let v76 := v75
            let v77 ← mloadInt (.u 64) ap
            let v78 := v77
            let v79 := IntTy.mulW (.u 128) v78 factor
            let v80 := IntTy.addW (.u 128) v76 v79
            let sum_81 := v80
            sum := sum_81
            let sum_82 := sum
            let sum_83 := sum_82
            let v84 := (UInt256.ofNat 64)
            let v85 := sum_83 >>> v84
            let v86 ← downcast_truncate_u128_u64 v85
            let carry_87 := v86
            carry := carry_87
            let sum_88 := sum
            let sum_89 := sum_88
            let v90 ← downcast_truncate_u128_u64 sum_89
            let low_91 := v90
            let v92 := self.inverse
            let v93 := IntTy.mulW (.u 64) low_91 v92
            let v94 := v93
            let m := v94
            let v96 := low_91
            let v97 ← mloadInt (.u 64) mp
            let v98 := v97
            let v99 := IntTy.mulW (.u 128) m v98
            let v100 := IntTy.addW (.u 128) v96 v99
            let reduced_101 := v100
            reduced := reduced_101
            let reduced_102 := reduced
            let reduced_103 := reduced_102
            let v104 := (UInt256.ofNat 64)
            let v105 := reduced_103 >>> v104
            let v106 ← downcast_truncate_u128_u64 v105
            let reduced_carry_107 := v106
            reduced_carry := reduced_carry_107
            let v108 := (UInt256.ofNat 1)
            let v109 := ()
            let v110 := (Range_Known_1_Unknown.mk v109 n_11)
            r9 := v110
            let v112 ← Range_start_1 r9
            let v113 := v112
            let rec b15 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
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
              let v115 ← Range_has_1 r9 v7
              if v115 then
                let v117 ← Range_at_1 r9 v7
                let j_118 := v117
                let v119 ← limb tp j_118
                let v120 ← mloadInt (.u 64) v119
                let v121 := v120
                let v122 ← limb ap j_118
                let v123 ← mloadInt (.u 64) v122
                let v124 := v123
                let v125 := IntTy.mulW (.u 128) v124 factor
                let v126 := IntTy.addW (.u 128) v121 v125
                let carry_127 := carry
                let v128 := carry_127
                let v129 := IntTy.addW (.u 128) v126 v128
                let sum_130 := v129
                sum := sum_130
                let v131 := ()
                let sum_132 := sum
                let sum_133 := sum_132
                let v134 := (UInt256.ofNat 64)
                let v135 := sum_133 >>> v134
                let v136 ← downcast_truncate_u128_u64 v135
                let carry_137 := v136
                carry := carry_137
                let v138 := ()
                let sum_139 := sum
                let sum_140 := sum_139
                let v141 ← downcast_truncate_u128_u64 sum_140
                let low_142 := v141
                let v143 := low_142
                let v144 ← limb mp j_118
                let v145 ← mloadInt (.u 64) v144
                let v146 := v145
                let v147 := IntTy.mulW (.u 128) m v146
                let v148 := IntTy.addW (.u 128) v143 v147
                let reduced_carry_149 := reduced_carry
                let v150 := reduced_carry_149
                let v151 := IntTy.addW (.u 128) v148 v150
                let reduced_152 := v151
                reduced := reduced_152
                let v153 := ()
                let reduced_154 := reduced
                let reduced_155 := reduced_154
                let v156 := (UInt256.ofNat 64)
                let v157 := reduced_155 >>> v156
                let v158 ← downcast_truncate_u128_u64 v157
                let reduced_carry_159 := v158
                reduced_carry := reduced_carry_159
                let v160 := ()
                let reduced_161 := reduced
                let reduced_162 := reduced_161
                let v163 ← downcast_truncate_u128_u64 reduced_162
                let v164 := (UInt256.ofNat 1)
                let v165 := j_118 - v164
                let v166 ← limb tp v165
                mstoreInt (.u 64) v166 v163
                let v167 := ()
                let v169 ← Range_next_1 r9 v7
                let v170 := v169
                b15 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v170
              else
                let v171 ← limb tp n_11
                let v172 ← mloadInt (.u 64) v171
                let v173 := v172
                let carry_174 := carry
                let v175 := carry_174
                let v176 := IntTy.addW (.u 128) v173 v175
                let reduced_carry_177 := reduced_carry
                let v178 := reduced_carry_177
                let v179 := IntTy.addW (.u 128) v176 v178
                let top := v179
                let v181 ← downcast_truncate_u128_u64 top
                let v182 := (UInt256.ofNat 1)
                let v183 := n_11 - v182
                let v184 ← limb tp v183
                mstoreInt (.u 64) v184 v181
                let v185 := ()
                let v186 := (UInt256.ofNat 64)
                let v187 := top >>> v186
                let v188 ← downcast_truncate_u128_u64 v187
                let v189 ← limb tp n_11
                mstoreInt (.u 64) v189 v188
                let v190 := ()
                let v192 ← Range_next_0 r8 v6
                let v193 := v192
                b11 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v193
            partial_fixpoint
            b15 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v113
          else
            let v195 ← MemArray_index_ref_u64 t_3 n_11
            let v196 ← mloadInt (.u 64) v195
            let v197 := (UInt256.ofNat 0)
            let v198 := decide (v196 ≠ v197)
            let at_least_199 := v198
            at_least := at_least_199
            let at_least_200 := at_least
            let v201 := !at_least_200
            let b21 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
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
              let at_least_247 := at_least
              let b30 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
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
              if at_least_247 then
                let v248 := (UInt256.ofNat 0)
                let borrow_249 := v248
                borrow := borrow_249
                let v250 := (UInt256.ofNat 0)
                let v251 := ()
                let v252 := (Range_Known_0_Unknown.mk v251 n_11)
                r10 := v252
                let v254 ← Range_start_0 r10
                let v255 := v254
                let rec b31 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) (v8 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                  let v257 ← Range_has_0 r10 v8
                  if v257 then
                    let v259 ← Range_at_0 r10 v8
                    let j_260 := v259
                    let v262 ← MemArray_index_ref_u64 t_3 j_260
                    let v263 ← mloadInt (.u 64) v262
                    let v264 := v263
                    let v267 ← MemArray_index_ref_u64 self.limbs j_260
                    let v268 ← mloadInt (.u 64) v267
                    let v269 := v268
                    let v270 := IntTy.subW (.u 128) v264 v269
                    let borrow_271 := borrow
                    let v272 := borrow_271
                    let v273 := IntTy.subW (.u 128) v270 v272
                    let difference := v273
                    let v275 ← downcast_truncate_u128_u64 difference
                    let (v276, v276_k) ← MemArray_index_mut_u64 t_3 j_260
                    mstoreInt (.u 64) v276 v275
                    let v276_0 ← v276_k v276
                    t_3 := v276_0
                    let v277 := ()
                    let v278 := (UInt256.ofNat 127)
                    let v279 := difference >>> v278
                    let v280 ← downcast_truncate_u128_u64 v279
                    let borrow_281 := v280
                    borrow := borrow_281
                    let v282 := ()
                    let v284 ← Range_next_0 r10 v8
                    let v285 := v284
                    b31 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v285
                  else
                    let v286 := ()
                    let v287 := v286
                    b30 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                partial_fixpoint
                b31 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v255
              else
                let v288 := ()
                let v289 := v288
                b30 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
            if v201 then
              let v202 := true
              let at_least_203 := v202
              at_least := at_least_203
              let v204 := ()
              let n_205 := n_11
              j := n_205
              let rec b22 (sum : UInt256) (carry : UInt256) (reduced : UInt256) (reduced_carry : UInt256) (at_least : Bool) (j : UInt256) (borrow : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (t_3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
                let j_206 := j
                let j_207 := j_206
                let v208 := (UInt256.ofNat 0)
                let v209 := decide (v208.toNat < j_207.toNat)
                let b24 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
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
                  let v243 := ()
                  let v244 := v243
                  b21 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                if v209 then
                  let v210 := (UInt256.ofNat 1)
                  let j_211 := j
                  let v212 := j_211 - v210
                  let j_213 := v212
                  j := j_213
                  let v214 := ()
                  let j_215 := j
                  let j_216 := j_215
                  let v218 ← MemArray_index_ref_u64 t_3 j_216
                  let v219 ← mloadInt (.u 64) v218
                  let j_220 := j
                  let j_221 := j_220
                  let v224 ← MemArray_index_ref_u64 self.limbs j_221
                  let v225 ← mloadInt (.u 64) v224
                  let v226 := decide (v219 ≠ v225)
                  if v226 then
                    let j_227 := j
                    let j_228 := j_227
                    let v230 ← MemArray_index_ref_u64 t_3 j_228
                    let v231 ← mloadInt (.u 64) v230
                    let j_232 := j
                    let j_233 := j_232
                    let v236 ← MemArray_index_ref_u64 self.limbs j_233
                    let v237 ← mloadInt (.u 64) v236
                    let v238 := decide (v237.toNat < v231.toNat)
                    let at_least_239 := v238
                    at_least := at_least_239
                    let v240 := ()
                    b24 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                  else
                    let v241 := ()
                    let v242 := v241
                    b22 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
                else
                  b24 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
              partial_fixpoint
              b22 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
            else
              let v245 := ()
              let v246 := v245
              b21 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
        partial_fixpoint
        b11 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v65
    partial_fixpoint
    b7 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v48
  let b2 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 => do
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
    let v27 := false
    let v28 := v27
    b3 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v28
  if v13 then
    let v14 ← MemSlice_len_u64 b
    let v15 := decide (n_11.toNat ≤ v14.toNat)
    if v15 then
      let v18 ← MemArray_len_u64 self.limbs
      let v19 := decide (n_11.toNat ≤ v18.toNat)
      if v19 then
        let v21 ← MemArray_len_u64 t_3
        let v22 := (UInt256.ofNat 2)
        let v23 := n_11 + v22
        let v24 := decide (v23.toNat ≤ v21.toNat)
        if v24 then
          let v25 := true
          let v26 := v25
          b3 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3 v26
        else
          b2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
      else
        b2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
    else
      b2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3
  else
    b2 sum carry reduced reduced_carry at_least j borrow r7 r8 r9 r10 t_3

def Modulus_mul_mod (self : Modulus) (a : MemArray_u64) (b : MemArray_u64) (work_3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut carry : UInt256 := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  let mut work_3 := work_3
  let v8 := self.len
  let v9 ← downcast_truncate_u64_usize v8
  let n := v9
  let v11 ← MemArray_len_u64 a
  let v12 := decide (n.toNat ≤ v11.toNat)
  let b3 : UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun carry r1 r2 r3 work_3 v4 => do
    let mut carry := carry
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut work_3 := work_3
    let v24 ← «assert» v4
    let v25 ← MemArray_ptr_u64 a
    let ap := v25
    let v27 ← MemArray_ptr_u64 b
    let bp := v27
    let v30 ← MemArray_ptr_u64 work_3
    let wp := v30
    let v32 := (UInt256.ofNat 0)
    let v33 := (UInt256.ofNat 2)
    let v34 := v33 * n
    let v35 := ()
    let v36 := (Range_Known_0_Unknown.mk v35 v34)
    r1 := v36
    let v38 ← Range_start_0 r1
    let v39 := v38
    let rec b6 (carry : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (work_3 : MemArray_u64) (v5 : UInt256) : FeM (Unit × MemArray_u64) := do
      let mut carry := carry
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut work_3 := work_3
      let v41 ← Range_has_0 r1 v5
      if v41 then
        let v43 ← Range_at_0 r1 v5
        let i_44 := v43
        let v45 := (UInt256.ofNat 0)
        let v46 ← limb wp i_44
        mstoreInt (.u 64) v46 v45
        let v47 := ()
        let v49 ← Range_next_0 r1 v5
        let v50 := v49
        b6 carry r1 r2 r3 work_3 v50
      else
        let v51 := (UInt256.ofNat 0)
        let v52 := ()
        let v53 := (Range_Known_0_Unknown.mk v52 n)
        r2 := v53
        let v55 ← Range_start_0 r2
        let v56 := v55
        let rec b10 (carry : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (work_3 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
          let mut carry := carry
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut work_3 := work_3
          let v58 ← Range_has_0 r2 v6
          if v58 then
            let v60 ← Range_at_0 r2 v6
            let i_61 := v60
            let v62 ← limb ap i_61
            let v63 ← mloadInt (.u 64) v62
            let v64 := v63
            let left := v64
            let v66 := (UInt256.ofNat 0)
            let carry_67 := v66
            carry := carry_67
            let v68 := (UInt256.ofNat 0)
            let v69 := ()
            let v70 := (Range_Known_0_Unknown.mk v69 n)
            r3 := v70
            let v72 ← Range_start_0 r3
            let v73 := v72
            let rec b14 (carry : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (work_3 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
              let mut carry := carry
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut work_3 := work_3
              let v75 ← Range_has_0 r3 v7
              if v75 then
                let v77 ← Range_at_0 r3 v7
                let j := v77
                let v79 ← limb bp j
                let v80 ← mloadInt (.u 64) v79
                let v81 := v80
                let v82 := IntTy.mulW (.u 128) left v81
                let v83 := i_61 + j
                let v84 ← limb wp v83
                let v85 ← mloadInt (.u 64) v84
                let v86 := v85
                let v87 := IntTy.addW (.u 128) v82 v86
                let carry_88 := carry
                let v89 := carry_88
                let v90 := IntTy.addW (.u 128) v87 v89
                let t := v90
                let v92 ← downcast_truncate_u128_u64 t
                let v93 := i_61 + j
                let v94 ← limb wp v93
                mstoreInt (.u 64) v94 v92
                let v95 := ()
                let v96 := (UInt256.ofNat 64)
                let v97 := t >>> v96
                let v98 ← downcast_truncate_u128_u64 v97
                let carry_99 := v98
                carry := carry_99
                let v100 := ()
                let v102 ← Range_next_0 r3 v7
                let v103 := v102
                b14 carry r1 r2 r3 work_3 v103
              else
                let carry_104 := carry
                let carry_105 := carry_104
                let v106 := i_61 + n
                let v107 ← limb wp v106
                mstoreInt (.u 64) v107 carry_105
                let v108 := ()
                let v110 ← Range_next_0 r2 v6
                let v111 := v110
                b10 carry r1 r2 r3 work_3 v111
            partial_fixpoint
            b14 carry r1 r2 r3 work_3 v73
          else
            let v113 := (UInt256.ofNat 2)
            let v114 := self.len
            let v115 := IntTy.mulW (.u 64) v113 v114
            let (v116, v116_0) ← Modulus_reduce self work_3 v115
            work_3 := v116_0
            return ((), work_3)
        partial_fixpoint
        b10 carry r1 r2 r3 work_3 v56
    partial_fixpoint
    b6 carry r1 r2 r3 work_3 v39
  let b2 : UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun carry r1 r2 r3 work_3 => do
    let mut carry := carry
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut work_3 := work_3
    let v22 := false
    let v23 := v22
    b3 carry r1 r2 r3 work_3 v23
  if v12 then
    let v13 ← MemArray_len_u64 b
    let v14 := decide (n.toNat ≤ v13.toNat)
    if v14 then
      let v16 ← MemArray_len_u64 work_3
      let v17 := (UInt256.ofNat 2)
      let v18 := v17 * n
      let v19 := decide (v18.toNat < v16.toNat)
      if v19 then
        let v20 := true
        let v21 := v20
        b3 carry r1 r2 r3 work_3 v21
      else
        b2 carry r1 r2 r3 work_3
    else
      b2 carry r1 r2 r3 work_3
  else
    b2 carry r1 r2 r3 work_3

def large_power (exponent : Exponent) (base : MemArray_u64) (modulus : Modulus) (value_3 : MemArray_u64) (size_4 : UInt256) : FeM Option_MemArray_u64 := do
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
  let v9 := exponent.total
  let total := v9
  let v11 ← MemArray_try_new_zeroed_u64 size_4
  let v12 := match v11 with | .Some .. => true | _ => false
  if v12 then
    let v13 := v11.Some_0
    let result_14 := v13
    result := result_14
    let v15 := (UInt256.ofNat 2)
    let size_16 := size_4
    let v17 ← checkedMul v15 size_16
    let v18 := (UInt256.ofNat 1)
    let v19 ← checkedAdd v17 v18
    let v20 ← MemArray_try_new_zeroed_u64 v19
    let v21 := match v20 with | .Some .. => true | _ => false
    if v21 then
      let v22 := v20.Some_0
      let work_23 := v22
      work := work_23
      let v24 := exponent.top
      let v25 ← checkedSub total v24
      let v26 := (UInt256.ofNat 2)
      let v27 := decide (v25 = v26)
      let b6 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
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
        let v46 := ()
        let v47 := v46
        let value_48 := value_3
        let v49 ← downcast_truncate_usize_u64 size_4
        let v50 ← Montgomery_new value_48 v49
        let montgomery_51 := v50
        montgomery := montgomery_51
        let v52 ← downcast_truncate_usize_u64 size_4
        let n := v52
        let v54 := exponent.top
        let v55 ← checkedSub total v54
        let v56 ← window_width v55
        let width := v56
        let v58 := (UInt256.ofNat 1)
        let v59 := (UInt256.ofNat 1)
        let v60 ← checkedSub width v59
        let v61 := v58 <<< v60
        let v62 ← downcast_truncate_usize_usize v61
        let entries := v62
        let size_64 := size_4
        let v65 ← checkedMul entries size_64
        let v66 ← MemArray_try_new_zeroed_u64 v65
        let v67 := match v66 with | .Some .. => true | _ => false
        if v67 then
          let v68 := v66.Some_0
          let table_69 := v68
          table := table_69
          let v70 := (UInt256.ofNat 0)
          let size_71 := size_4
          let v72 := ()
          let v73 := (Range_Known_0_Unknown.mk v72 size_71)
          r7 := v73
          let v75 ← Range_start_0 r7
          let v76 := v75
          let rec b11 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
            let v78 ← Range_has_0 r7 v5
            if v78 then
              let v80 ← Range_at_0 r7 v5
              let i := v80
              let v82 := (UInt256.ofNat 0)
              let (v84, v84_k) ← MemArray_index_mut_u64 work i
              mstoreInt (.u 64) v84 v82
              let v84_0 ← v84_k v84
              work := v84_0
              let v85 := ()
              let v86 ← MemArray_index_ref_u64 base i
              let v87 ← mloadInt (.u 64) v86
              let size_88 := size_4
              let v89 ← checkedAdd size_88 i
              let (v91, v91_k) ← MemArray_index_mut_u64 work v89
              mstoreInt (.u 64) v91 v87
              let v91_0 ← v91_k v91
              work := v91_0
              let v92 := ()
              let v94 ← Range_next_0 r7 v5
              let v95 := v94
              b11 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v95
            else
              let v97 := (UInt256.ofNat 2)
              let v98 ← IntTy.mul (.u 64) v97 n
              let (v99, v99_0) ← Modulus_reduce modulus work v98
              work := v99_0
              let v101 ← MemArray_as_slice_ref_u64 work
              let v103 ← MemSlice_head_ref_u64 v101 size_4
              let (v105_g, v105_k) ← MemArray_as_slice_mut_u64 table
              v105 := v105_g
              let (v106_g, v106_k) ← MemSlice_head_mut_u64 v105 size_4
              v106 := v106_g
              let (v108, v108_0) ← MemSlice_copy_from_u64 v106 v103
              v106 := v108_0
              let v106_0 ← v106_k v106
              v105 := v106_0
              let v105_0 ← v105_k v105
              table := v105_0
              let v109 ← MemArray_try_new_zeroed_u64 size_4
              let v110 := match v109 with | .Some .. => true | _ => false
              if v110 then
                let v111 := v109.Some_0
                let squared_112 := v111
                squared := squared_112
                let v113 := (UInt256.ofNat 1)
                let v114 := decide (v113.toNat < entries.toNat)
                let b19 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
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
                  let v139 := (UInt256.ofNat 1)
                  let v140 := ()
                  let v141 := (Range_Known_1_Unknown.mk v140 entries)
                  r8 := v141
                  let v143 ← Range_start_1 r8
                  let v144 := v143
                  let rec b20 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v6 : UInt256) : FeM Option_MemArray_u64 := do
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
                    let v146 ← Range_has_1 r8 v6
                    if v146 then
                      let v148 ← Range_at_1 r8 v6
                      let k := v148
                      let v150 := (UInt256.ofNat 1)
                      let v151 ← checkedSub k v150
                      let size_152 := size_4
                      let v153 ← checkedMul v151 size_152
                      let v155 ← MemArray_as_slice_ref_u64 table
                      let v157 ← MemSlice_slice_ref_u64 v155 v153 size_4
                      let v159 ← MemArray_as_slice_ref_u64 squared
                      let (v164, v164_0) ← Montgomery_mul montgomery v157 v159 work
                      work := v164_0
                      let v166 ← MemArray_as_slice_ref_u64 work
                      let v168 ← MemSlice_head_ref_u64 v166 size_4
                      let size_169 := size_4
                      let v170 ← checkedMul k size_169
                      let (v172_g, v172_k) ← MemArray_as_slice_mut_u64 table
                      v172 := v172_g
                      let (v173_g, v173_k) ← MemSlice_slice_mut_u64 v172 v170 size_4
                      v173 := v173_g
                      let (v175, v175_0) ← MemSlice_copy_from_u64 v173 v168
                      v173 := v175_0
                      let v173_0 ← v173_k v173
                      v172 := v173_0
                      let v172_0 ← v172_k v172
                      table := v172_0
                      let v177 ← Range_next_1 r8 v6
                      let v178 := v177
                      b20 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v178
                    else
                      let v179 := exponent.top
                      let index_180 := v179
                      index := index_180
                      let v181 := false
                      let started_182 := v181
                      started := started_182
                      let rec b24 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) : FeM Option_MemArray_u64 := do
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
                        let index_183 := index
                        let index_184 := index_183
                        let v185 := decide (index_184.toNat < total.toNat)
                        if v185 then
                          let index_186 := index
                          let index_187 := index_186
                          let v188 ← Exponent_window exponent index_187 width
                          let window := v188
                          let started_190 := started
                          let rec b29 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) : FeM Option_MemArray_u64 := do
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
                            let v227 := window.odd
                            let b36 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
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
                              let v264 := window.«end»
                              let index_265 := v264
                              index := index_265
                              let v266 := ()
                              b24 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                            if v227 then
                              let v228 := window.entry
                              let size_229 := size_4
                              let v230 ← checkedMul v228 size_229
                              let v232 ← MemArray_as_slice_ref_u64 table
                              let v234 ← MemSlice_slice_ref_u64 v232 v230 size_4
                              let started_236 := started
                              let b39 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → Unit → FeM Option_MemArray_u64 := fun result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v8 => do
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
                                let v261 := v8
                                b36 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                              if started_236 then
                                let v238 ← MemArray_as_slice_ref_u64 result
                                let (v243, v243_0) ← Montgomery_mul montgomery v238 v234 work
                                work := v243_0
                                let v245 ← MemArray_as_slice_ref_u64 work
                                let v247 ← MemSlice_head_ref_u64 v245 size_4
                                let (v249_g, v249_k) ← MemArray_as_slice_mut_u64 result
                                v249 := v249_g
                                let (v251, v251_0) ← MemSlice_copy_from_u64 v249 v247
                                v249 := v251_0
                                let v249_0 ← v249_k v249
                                result := v249_0
                                let v252 := v251
                                b39 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v252
                              else
                                let (v254_g, v254_k) ← MemArray_as_slice_mut_u64 result
                                v254 := v254_g
                                let (v256, v256_0) ← MemSlice_copy_from_u64 v254 v234
                                v254 := v256_0
                                let v254_0 ← v254_k v254
                                result := v254_0
                                let v257 := true
                                let started_258 := v257
                                started := started_258
                                let v259 := ()
                                let v260 := v259
                                b39 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v260
                            else
                              let v262 := ()
                              let v263 := v262
                              b36 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                          partial_fixpoint
                          if started_190 then
                            let index_191 := index
                            let index_192 := index_191
                            let v193 := window.«end»
                            let v194 := ()
                            let v195 := (Range_Unknown_Unknown.mk index_192 v193)
                            r9 := v195
                            let v197 ← Range_Unknown_Unknown_start r9
                            let v198 := v197
                            let rec b30 (result : MemArray_u64) (work : MemArray_u64) (montgomery : Montgomery) (table : MemArray_u64) (squared : MemArray_u64) (index : UInt256) (started : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v7 : UInt256) : FeM Option_MemArray_u64 := do
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
                              let v200 ← Range_Unknown_Unknown_has r9 v7
                              if v200 then
                                let v202 ← Range_Unknown_Unknown_at r9 v7
                                let v204 ← MemArray_as_slice_ref_u64 result
                                let v206 ← MemArray_as_slice_ref_u64 result
                                let (v211, v211_0) ← Montgomery_mul montgomery v204 v206 work
                                work := v211_0
                                let v213 ← MemArray_as_slice_ref_u64 work
                                let v215 ← MemSlice_head_ref_u64 v213 size_4
                                let (v217_g, v217_k) ← MemArray_as_slice_mut_u64 result
                                v217 := v217_g
                                let (v219, v219_0) ← MemSlice_copy_from_u64 v217 v215
                                v217 := v219_0
                                let v217_0 ← v217_k v217
                                result := v217_0
                                let v221 ← Range_Unknown_Unknown_next r9 v7
                                let v222 := v221
                                b30 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v222
                              else
                                let v223 := ()
                                let v224 := v223
                                b29 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                            partial_fixpoint
                            b30 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v198
                          else
                            let v225 := ()
                            let v226 := v225
                            b29 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                        else
                          let (v268, v268_0) ← MemArray_zero_u64 squared
                          squared := v268_0
                          let v269 := (UInt256.ofNat 1)
                          let v270 := (UInt256.ofNat 0)
                          let (v272, v272_k) ← MemArray_index_mut_u64 squared v270
                          mstoreInt (.u 64) v272 v269
                          let v272_0 ← v272_k v272
                          squared := v272_0
                          let v273 := ()
                          let v275 ← MemArray_as_slice_ref_u64 result
                          let v277 ← MemArray_as_slice_ref_u64 squared
                          let (v282, v282_0) ← Montgomery_mul montgomery v275 v277 work
                          work := v282_0
                          let v284 ← MemArray_as_slice_ref_u64 work
                          let v286 ← MemSlice_head_ref_u64 v284 size_4
                          let (v288_g, v288_k) ← MemArray_as_slice_mut_u64 result
                          v288 := v288_g
                          let (v290, v290_0) ← MemSlice_copy_from_u64 v288 v286
                          v288 := v290_0
                          let v288_0 ← v288_k v288
                          result := v288_0
                          let result_291 := result
                          let result_292 := result_291
                          let v293 := Option_MemArray_u64.Some result_292
                          let _ ← MemArray_drop_u64 squared
                          let _ ← MemArray_drop_u64 table
                          let _ ← MemArray_drop_u64 montgomery.limbs
                          let _ ← MemArray_drop_u64 work
                          return v293
                      partial_fixpoint
                      b24 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                  partial_fixpoint
                  b20 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v144
                if v114 then
                  let v116 ← MemArray_as_slice_ref_u64 table
                  let v118 ← MemSlice_head_ref_u64 v116 size_4
                  let v120 ← MemArray_as_slice_ref_u64 table
                  let v122 ← MemSlice_head_ref_u64 v120 size_4
                  let (v127, v127_0) ← Montgomery_mul montgomery v118 v122 work
                  work := v127_0
                  let v129 ← MemArray_as_slice_ref_u64 work
                  let v131 ← MemSlice_head_ref_u64 v129 size_4
                  let (v133_g, v133_k) ← MemArray_as_slice_mut_u64 squared
                  v133 := v133_g
                  let (v135, v135_0) ← MemSlice_copy_from_u64 v133 v131
                  v133 := v135_0
                  let v133_0 ← v133_k v133
                  squared := v133_0
                  let v136 := v135
                  b19 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                else
                  let v137 := ()
                  let v138 := v137
                  b19 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
              else
                let v294 := Option_MemArray_u64.None
                match (← pure v109) with
                | .Some .. =>
                  let _ ← MemArray_drop_u64 v109.Some_0
                  pure ()
                | _ => pure ()
                let _ ← MemArray_drop_u64 table
                let _ ← MemArray_drop_u64 montgomery.limbs
                let _ ← MemArray_drop_u64 work
                let _ ← MemArray_drop_u64 result
                return v294
          partial_fixpoint
          b11 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v76
        else
          let v295 := Option_MemArray_u64.None
          match (← pure v66) with
          | .Some .. =>
            let _ ← MemArray_drop_u64 v66.Some_0
            pure ()
          | _ => pure ()
          let _ ← MemArray_drop_u64 montgomery.limbs
          let _ ← MemArray_drop_u64 work
          let _ ← MemArray_drop_u64 result
          return v295
      if v27 then
        let v28 := (UInt256.ofNat 1)
        let v29 ← checkedSub total v28
        let v30 ← Exponent_bit exponent v29
        let v31 := (UInt256.ofNat 0)
        let v32 := decide (v30 = v31)
        if v32 then
          let (v34, v34_0) ← Modulus_mul_mod modulus base base work
          work := v34_0
          let v36 ← MemArray_as_slice_ref_u64 work
          let v38 ← MemSlice_head_ref_u64 v36 size_4
          let (v40_g, v40_k) ← MemArray_as_slice_mut_u64 result
          v40 := v40_g
          let (v42, v42_0) ← MemSlice_copy_from_u64 v40 v38
          v40 := v42_0
          let v40_0 ← v40_k v40
          result := v40_0
          let result_43 := result
          let result_44 := result_43
          let v45 := Option_MemArray_u64.Some result_44
          let _ ← MemArray_drop_u64 work
          let _ ← MemArray_drop_u64 value_3
          return v45
        else
          b6 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
      else
        b6 result work montgomery table squared index started r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
    else
      let v296 := Option_MemArray_u64.None
      match (← pure v20) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v20.Some_0
        pure ()
      | _ => pure ()
      let _ ← MemArray_drop_u64 result
      let _ ← MemArray_drop_u64 value_3
      return v296
  else
    let v297 := Option_MemArray_u64.None
    match (← pure v11) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v11.Some_0
      pure ()
    | _ => pure ()
    let _ ← MemArray_drop_u64 value_3
    return v297

def odd_power (exponent : Exponent) (base : MemArray_u64) (base_size_2 : UInt256) (value : MemArray_u64) (n_4 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
  let mut modulus : Modulus := default
  let mut limbs : MemArray_u64 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut base := base
  let n_6 := n_4
  let v7 := (UInt256.ofNat 4)
  let v8 := decide (n_6.toNat ≤ v7.toNat)
  let b2 : Modulus → MemArray_u64 → Range_Known_0_Unknown → MemArray_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun modulus limbs r2 base => do
    let mut modulus := modulus
    let mut limbs := limbs
    let mut r2 := r2
    let mut base := base
    let v15 := ()
    let v16 := v15
    let v17 ← downcast_truncate_usize_u64 n_4
    let v18 ← Modulus_try_new value v17
    let v19 := match v18 with | .Some .. => true | _ => false
    if v19 then
      let v20 := v18.Some_0
      let modulus_21 := v20
      modulus := modulus_21
      let base_size_22 := base_size_2
      let n_23 := n_4
      let v24 := decide (n_23.toNat < base_size_22.toNat)
      let b9 : Modulus → MemArray_u64 → Range_Known_0_Unknown → MemArray_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun modulus limbs r2 base => do
        let mut modulus := modulus
        let mut limbs := limbs
        let mut r2 := r2
        let mut base := base
        let n_32 := n_4
        let v33 := (UInt256.ofNat 4)
        let v34 := decide (n_32.toNat ≤ v33.toNat)
        if v34 then
          let v37 ← small_power_of exponent base value n_4
          let _ ← MemArray_drop_u64 modulus.limbs
          return (v37, base)
        else
          let v38 := ()
          let v39 := v38
          let v40 ← MemArray_try_new_zeroed_u64 n_4
          let v41 := match v40 with | .Some .. => true | _ => false
          if v41 then
            let v42 := v40.Some_0
            let limbs_43 := v42
            limbs := limbs_43
            let v44 := (UInt256.ofNat 0)
            let n_45 := n_4
            let v46 := ()
            let v47 := (Range_Known_0_Unknown.mk v46 n_45)
            r2 := v47
            let v49 ← Range_start_0 r2
            let v50 := v49
            let rec b15 (modulus : Modulus) (limbs : MemArray_u64) (r2 : Range_Known_0_Unknown) (base : MemArray_u64) (v5 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
              let mut modulus := modulus
              let mut limbs := limbs
              let mut r2 := r2
              let mut base := base
              let v52 ← Range_has_0 r2 v5
              if v52 then
                let v54 ← Range_at_0 r2 v5
                let i := v54
                let v56 ← MemArray_index_ref_u64 value i
                let v57 ← mloadInt (.u 64) v56
                let (v59, v59_k) ← MemArray_index_mut_u64 limbs i
                mstoreInt (.u 64) v59 v57
                let v59_0 ← v59_k v59
                limbs := v59_0
                let v60 := ()
                let v62 ← Range_next_0 r2 v5
                let v63 := v62
                b15 modulus limbs r2 base v63
              else
                let limbs_65 := limbs
                let limbs_66 := limbs_65
                let v69 ← large_power exponent base modulus limbs_66 n_4
                let _ ← MemArray_drop_u64 modulus.limbs
                return (v69, base)
            partial_fixpoint
            b15 modulus limbs r2 base v50
          else
            let v70 := Option_MemArray_u64.None
            match (← pure v40) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v40.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 modulus.limbs
            return (v70, base)
      if v24 then
        let v26 ← downcast_truncate_usize_u64 base_size_2
        let (v28, v28_0) ← Modulus_reduce modulus base v26
        base := v28_0
        let v29 := v28
        b9 modulus limbs r2 base
      else
        let v30 := ()
        let v31 := v30
        b9 modulus limbs r2 base
    else
      let v71 := Option_MemArray_u64.None
      match (← pure v18) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v18.Some_0.limbs
        pure ()
      | _ => pure ()
      return (v71, base)
  if v8 then
    let base_size_9 := base_size_2
    let n_10 := n_4
    let v11 := decide (base_size_9.toNat ≤ n_10.toNat)
    if v11 then
      let v14 ← small_power_of exponent base value n_4
      return (v14, base)
    else
      b2 modulus limbs r2 base
  else
    b2 modulus limbs r2 base

def low_product (a : MemSlice_u64) (b : MemSlice_u64) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut a_len : UInt256 := default
  let mut b_len : UInt256 := default
  let mut carry : UInt256 := default
  let mut «at» : UInt256 := default
  let mut r4 : Range_Known_0_Unknown := default
  let mut r5 : Range_Known_0_Unknown := default
  let mut r6 : Range_Known_0_Unknown := default
  let mut out_2 := out_2
  let v9 ← MemSlice_len_u64 out_2
  let len_10 := v9
  let v11 ← MemSlice_len_u64 a
  let v12 := decide (len_10.toNat ≤ v11.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → Bool → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 v3 => do
    let mut a_len := a_len
    let mut b_len := b_len
    let mut carry := carry
    let mut «at» := «at»
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut out_2 := out_2
    let v19 ← «assert» v3
    let v20 ← MemSlice_ptr_u64 a
    let ap := v20
    let v22 ← MemSlice_ptr_u64 b
    let bp := v22
    let v25 ← MemSlice_ptr_u64 out_2
    let op := v25
    let v27 := (UInt256.ofNat 0)
    let v28 := ()
    let v29 := (Range_Known_0_Unknown.mk v28 len_10)
    r4 := v29
    let v31 ← Range_start_0 r4
    let v32 := v31
    let rec b5 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) (v4 : UInt256) : FeM (Unit × MemSlice_u64) := do
      let mut a_len := a_len
      let mut b_len := b_len
      let mut carry := carry
      let mut «at» := «at»
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let mut out_2 := out_2
      let v34 ← Range_has_0 r4 v4
      if v34 then
        let v36 ← Range_at_0 r4 v4
        let j_37 := v36
        let v38 := (UInt256.ofNat 0)
        let v39 ← limb op j_37
        mstoreInt (.u 64) v39 v38
        let v40 := ()
        let v42 ← Range_next_0 r4 v4
        let v43 := v42
        b5 a_len b_len carry «at» r4 r5 r6 out_2 v43
      else
        let len_44 := len_10
        a_len := len_44
        let rec b9 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
          let mut a_len := a_len
          let mut b_len := b_len
          let mut carry := carry
          let mut «at» := «at»
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut out_2 := out_2
          let a_len_45 := a_len
          let a_len_46 := a_len_45
          let v47 := (UInt256.ofNat 0)
          let v48 := decide (v47.toNat < a_len_46.toNat)
          let b11 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
            let mut a_len := a_len
            let mut b_len := b_len
            let mut carry := carry
            let mut «at» := «at»
            let mut r4 := r4
            let mut r5 := r5
            let mut r6 := r6
            let mut out_2 := out_2
            let len_62 := len_10
            b_len := len_62
            let rec b13 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
              let mut a_len := a_len
              let mut b_len := b_len
              let mut carry := carry
              let mut «at» := «at»
              let mut r4 := r4
              let mut r5 := r5
              let mut r6 := r6
              let mut out_2 := out_2
              let b_len_63 := b_len
              let b_len_64 := b_len_63
              let v65 := (UInt256.ofNat 0)
              let v66 := decide (v65.toNat < b_len_64.toNat)
              let b15 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
                let mut a_len := a_len
                let mut b_len := b_len
                let mut carry := carry
                let mut «at» := «at»
                let mut r4 := r4
                let mut r5 := r5
                let mut r6 := r6
                let mut out_2 := out_2
                let v80 := (UInt256.ofNat 0)
                let v81 := ()
                let a_len_82 := a_len
                let v83 := (Range_Known_0_Unknown.mk v81 a_len_82)
                r5 := v83
                let v85 ← Range_start_0 r5
                let v86 := v85
                let rec b17 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) (v5 : UInt256) : FeM (Unit × MemSlice_u64) := do
                  let mut a_len := a_len
                  let mut b_len := b_len
                  let mut carry := carry
                  let mut «at» := «at»
                  let mut r4 := r4
                  let mut r5 := r5
                  let mut r6 := r6
                  let mut out_2 := out_2
                  let v88 ← Range_has_0 r5 v5
                  if v88 then
                    let v90 ← Range_at_0 r5 v5
                    let i := v90
                    let v92 ← limb ap i
                    let v93 ← mloadInt (.u 64) v92
                    let v94 := v93
                    let factor := v94
                    let v96 := (UInt256.ofNat 0)
                    let carry_97 := v96
                    carry := carry_97
                    let v98 := len_10 - i
                    let b_len_99 := b_len
                    let v100 := decide (v98.toNat < b_len_99.toNat)
                    let b23 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → UInt256 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 v6 => do
                      let mut a_len := a_len
                      let mut b_len := b_len
                      let mut carry := carry
                      let mut «at» := «at»
                      let mut r4 := r4
                      let mut r5 := r5
                      let mut r6 := r6
                      let mut out_2 := out_2
                      let «end» := v6
                      let v107 := (UInt256.ofNat 0)
                      let v108 := ()
                      let v109 := (Range_Known_0_Unknown.mk v108 «end»)
                      r6 := v109
                      let v111 ← Range_start_0 r6
                      let v112 := v111
                      let rec b24 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) (v7 : UInt256) : FeM (Unit × MemSlice_u64) := do
                        let mut a_len := a_len
                        let mut b_len := b_len
                        let mut carry := carry
                        let mut «at» := «at»
                        let mut r4 := r4
                        let mut r5 := r5
                        let mut r6 := r6
                        let mut out_2 := out_2
                        let v114 ← Range_has_0 r6 v7
                        if v114 then
                          let v116 ← Range_at_0 r6 v7
                          let j_117 := v116
                          let v118 := i + j_117
                          let v119 ← limb op v118
                          let v120 ← mloadInt (.u 64) v119
                          let v121 := v120
                          let v122 ← limb bp j_117
                          let v123 ← mloadInt (.u 64) v122
                          let v124 := v123
                          let v125 := IntTy.mulW (.u 128) factor v124
                          let v126 := IntTy.addW (.u 128) v121 v125
                          let carry_127 := carry
                          let v128 := carry_127
                          let v129 := IntTy.addW (.u 128) v126 v128
                          let sum_130 := v129
                          let v131 ← downcast_truncate_u128_u64 sum_130
                          let v132 := i + j_117
                          let v133 ← limb op v132
                          mstoreInt (.u 64) v133 v131
                          let v134 := ()
                          let v135 := (UInt256.ofNat 64)
                          let v136 := sum_130 >>> v135
                          let v137 ← downcast_truncate_u128_u64 v136
                          let carry_138 := v137
                          carry := carry_138
                          let v139 := ()
                          let v141 ← Range_next_0 r6 v7
                          let v142 := v141
                          b24 a_len b_len carry «at» r4 r5 r6 out_2 v142
                        else
                          let v143 := i + «end»
                          let at_144 := v143
                          «at» := at_144
                          let rec b28 (a_len : UInt256) (b_len : UInt256) (carry : UInt256) («at» : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (out_2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
                            let mut a_len := a_len
                            let mut b_len := b_len
                            let mut carry := carry
                            let mut «at» := «at»
                            let mut r4 := r4
                            let mut r5 := r5
                            let mut r6 := r6
                            let mut out_2 := out_2
                            let carry_145 := carry
                            let carry_146 := carry_145
                            let v147 := (UInt256.ofNat 0)
                            let v148 := decide (carry_146 ≠ v147)
                            let b30 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
                              let mut a_len := a_len
                              let mut b_len := b_len
                              let mut carry := carry
                              let mut «at» := «at»
                              let mut r4 := r4
                              let mut r5 := r5
                              let mut r6 := r6
                              let mut out_2 := out_2
                              let v176 := ()
                              let v178 ← Range_next_0 r5 v5
                              let v179 := v178
                              b17 a_len b_len carry «at» r4 r5 r6 out_2 v179
                            if v148 then
                              let at_149 := «at»
                              let at_150 := at_149
                              let v151 := decide (at_150.toNat < len_10.toNat)
                              if v151 then
                                let at_152 := «at»
                                let at_153 := at_152
                                let v154 ← limb op at_153
                                let v155 ← mloadInt (.u 64) v154
                                let v156 := v155
                                let carry_157 := carry
                                let v158 := carry_157
                                let v159 := IntTy.addW (.u 128) v156 v158
                                let sum_160 := v159
                                let v161 ← downcast_truncate_u128_u64 sum_160
                                let at_162 := «at»
                                let at_163 := at_162
                                let v164 ← limb op at_163
                                mstoreInt (.u 64) v164 v161
                                let v165 := ()
                                let v166 := (UInt256.ofNat 64)
                                let v167 := sum_160 >>> v166
                                let v168 ← downcast_truncate_u128_u64 v167
                                let carry_169 := v168
                                carry := carry_169
                                let v170 := ()
                                let v171 := (UInt256.ofNat 1)
                                let at_172 := «at»
                                let v173 := at_172 + v171
                                let at_174 := v173
                                «at» := at_174
                                let v175 := ()
                                b28 a_len b_len carry «at» r4 r5 r6 out_2
                              else
                                b30 a_len b_len carry «at» r4 r5 r6 out_2
                            else
                              b30 a_len b_len carry «at» r4 r5 r6 out_2
                          partial_fixpoint
                          b28 a_len b_len carry «at» r4 r5 r6 out_2
                      partial_fixpoint
                      b24 a_len b_len carry «at» r4 r5 r6 out_2 v112
                    if v100 then
                      let v101 := len_10 - i
                      let v102 := v101
                      b23 a_len b_len carry «at» r4 r5 r6 out_2 v102
                    else
                      let b_len_103 := b_len
                      let b_len_104 := b_len_103
                      let b_len_105 := b_len_104
                      b23 a_len b_len carry «at» r4 r5 r6 out_2 b_len_105
                  else
                    let v180 := ()
                    return ((), out_2)
                partial_fixpoint
                b17 a_len b_len carry «at» r4 r5 r6 out_2 v86
              if v66 then
                let b_len_67 := b_len
                let b_len_68 := b_len_67
                let v69 := (UInt256.ofNat 1)
                let v70 := b_len_68 - v69
                let v71 ← limb bp v70
                let v72 ← mloadInt (.u 64) v71
                let v73 := (UInt256.ofNat 0)
                let v74 := decide (v72 = v73)
                if v74 then
                  let v75 := (UInt256.ofNat 1)
                  let b_len_76 := b_len
                  let v77 := b_len_76 - v75
                  let b_len_78 := v77
                  b_len := b_len_78
                  let v79 := ()
                  b13 a_len b_len carry «at» r4 r5 r6 out_2
                else
                  b15 a_len b_len carry «at» r4 r5 r6 out_2
              else
                b15 a_len b_len carry «at» r4 r5 r6 out_2
            partial_fixpoint
            b13 a_len b_len carry «at» r4 r5 r6 out_2
          if v48 then
            let a_len_49 := a_len
            let a_len_50 := a_len_49
            let v51 := (UInt256.ofNat 1)
            let v52 := a_len_50 - v51
            let v53 ← limb ap v52
            let v54 ← mloadInt (.u 64) v53
            let v55 := (UInt256.ofNat 0)
            let v56 := decide (v54 = v55)
            if v56 then
              let v57 := (UInt256.ofNat 1)
              let a_len_58 := a_len
              let v59 := a_len_58 - v57
              let a_len_60 := v59
              a_len := a_len_60
              let v61 := ()
              b9 a_len b_len carry «at» r4 r5 r6 out_2
            else
              b11 a_len b_len carry «at» r4 r5 r6 out_2
          else
            b11 a_len b_len carry «at» r4 r5 r6 out_2
        partial_fixpoint
        b9 a_len b_len carry «at» r4 r5 r6 out_2
    partial_fixpoint
    b5 a_len b_len carry «at» r4 r5 r6 out_2 v32
  let b2 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun a_len b_len carry «at» r4 r5 r6 out_2 => do
    let mut a_len := a_len
    let mut b_len := b_len
    let mut carry := carry
    let mut «at» := «at»
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut out_2 := out_2
    let v17 := false
    let v18 := v17
    b3 a_len b_len carry «at» r4 r5 r6 out_2 v18
  if v12 then
    let v13 ← MemSlice_len_u64 b
    let v14 := decide (len_10.toNat ≤ v13.toNat)
    if v14 then
      let v15 := true
      let v16 := v15
      b3 a_len b_len carry «at» r4 r5 r6 out_2 v16
    else
      b2 a_len b_len carry «at» r4 r5 r6 out_2
  else
    b2 a_len b_len carry «at» r4 r5 r6 out_2

def low_power (exponent : Exponent) (base : MemArray_u64) (k_2 : UInt256) : FeM Option_MemArray_u64 := do
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
  let k_9 := k_2
  let v10 := (UInt256.ofNat 63)
  let v11 := k_9 + v10
  let v12 := (UInt256.ofNat 64)
  let v13 ← checkedDiv v11 v12
  let low := v13
  let k_15 := k_2
  let v16 := (UInt256.ofNat 64)
  let v17 ← checkedRem k_15 v16
  let v18 ← downcast_truncate_usize_u64 v17
  let bits := v18
  let v20 := (UInt256.ofNat 0)
  let v21 := decide (bits = v20)
  let b3 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v3 => do
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
    let mask := v3
    let v30 ← MemArray_try_new_zeroed_u64 low
    let v31 := match v30 with | .Some .. => true | _ => false
    if v31 then
      let v32 := v30.Some_0
      let result_33 := v32
      result := result_33
      let v34 := (UInt256.ofNat 0)
      let v35 ← MemArray_index_ref_u64 base v34
      let v36 ← mloadInt (.u 64) v35
      let v37 := (UInt256.ofNat 1)
      let v38 := v36 &&& v37
      let v39 := (UInt256.ofNat 1)
      let v40 := decide (v38 = v39)
      let b8 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v4 => do
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
        let start_52 := v4
        let v53 := (UInt256.ofNat 2)
        let v54 := decide (low.toNat ≤ v53.toNat)
        if v54 then
          let v55 := (UInt256.ofNat 1)
          let v56 := decide (low = v55)
          let b15 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v5 => do
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
            let wide_mask := v5
            let v66 := (UInt256.ofNat 0)
            let v67 ← MemArray_index_ref_u64 base v66
            let v68 ← mloadInt (.u 64) v67
            let v69 := v68
            let v70 := (UInt256.ofNat 1)
            let v71 ← MemArray_index_ref_u64 base v70
            let v72 ← mloadInt (.u 64) v71
            let v73 := v72
            let v74 := (UInt256.ofNat 64)
            let v75 := IntTy.shl (.u 128) v73 v74
            let v76 := v69 ||| v75
            let v77 := v76 &&& wide_mask
            let factor_78 := v77
            let v79 := (UInt256.ofNat 1)
            let power_80 := v79
            power := power_80
            let start_81 := start_52
            index_r2 := start_81
            let rec b16 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) : FeM Option_MemArray_u64 := do
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
              let index_82 := index_r2
              let index_83 := index_82
              let v84 := exponent.total
              let v85 := decide (index_83.toNat < v84.toNat)
              let b18 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
                let power_118 := power
                let power_119 := power_118
                let v120 ← downcast_truncate_u128_u64 power_119
                let v121 := (UInt256.ofNat 0)
                let (v123, v123_k) ← MemArray_index_mut_u64 result v121
                mstoreInt (.u 64) v123 v120
                let v123_0 ← v123_k v123
                result := v123_0
                let v124 := ()
                let v125 := (UInt256.ofNat 2)
                let v126 := decide (low = v125)
                let b25 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
                  let result_139 := result
                  let result_140 := result_139
                  let v141 := Option_MemArray_u64.Some result_140
                  return v141
                if v126 then
                  let power_127 := power
                  let power_128 := power_127
                  let v129 := (UInt256.ofNat 64)
                  let v130 := power_128 >>> v129
                  let v131 ← downcast_truncate_u128_u64 v130
                  let v132 := (UInt256.ofNat 1)
                  let (v134, v134_k) ← MemArray_index_mut_u64 result v132
                  mstoreInt (.u 64) v134 v131
                  let v134_0 ← v134_k v134
                  result := v134_0
                  let v135 := ()
                  let v136 := v135
                  b25 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                else
                  let v137 := ()
                  let v138 := v137
                  b25 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
              if v85 then
                let power_86 := power
                let power_87 := power_86
                let v88 := (UInt256.ofNat 0)
                let v89 := decide (power_87 ≠ v88)
                if v89 then
                  let power_90 := power
                  let power_91 := power_90
                  let power_92 := power
                  let v93 := IntTy.mulW (.u 128) power_91 power_92
                  let power_94 := v93
                  power := power_94
                  let v95 := ()
                  let index_96 := index_r2
                  let index_97 := index_96
                  let v98 ← Exponent_bit exponent index_97
                  let v99 := (UInt256.ofNat 1)
                  let v100 := decide (v98 = v99)
                  let b22 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
                    let power_109 := power
                    let v110 := power_109 &&& wide_mask
                    let power_111 := v110
                    power := power_111
                    let v112 := ()
                    let v113 := (UInt256.ofNat 1)
                    let index_114 := index_r2
                    let v115 := index_114 + v113
                    let index_116 := v115
                    index_r2 := index_116
                    let v117 := ()
                    b16 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                  if v100 then
                    let power_101 := power
                    let power_102 := power_101
                    let v103 := IntTy.mulW (.u 128) power_102 factor_78
                    let power_104 := v103
                    power := power_104
                    let v105 := ()
                    let v106 := v105
                    b22 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                  else
                    let v107 := ()
                    let v108 := v107
                    b22 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                else
                  b18 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
              else
                b18 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
            partial_fixpoint
            b16 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
          if v56 then
            let v57 := mask
            let v58 := v57
            b15 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v58
          else
            let v59 := mask
            let v60 := (UInt256.ofNat 64)
            let v61 := IntTy.shl (.u 128) v59 v60
            let v62 := (UInt256.ofNat 18446744073709551615)
            let v63 := v61 ||| v62
            let v64 := v63
            b15 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v64
        else
          let v142 := ()
          let v143 := v142
          let v144 ← MemArray_try_new_zeroed_u64 low
          let v145 := match v144 with | .Some .. => true | _ => false
          if v145 then
            let v146 := v144.Some_0
            let factor_147 := v146
            factor := factor_147
            let v148 ← MemArray_try_new_zeroed_u64 low
            let v149 := match v148 with | .Some .. => true | _ => false
            if v149 then
              let v150 := v148.Some_0
              let work_151 := v150
              work := work_151
              let v152 := (UInt256.ofNat 0)
              let v153 := ()
              let v154 := (Range_Known_0_Unknown.mk v153 low)
              r7 := v154
              let v156 ← Range_start_0 r7
              let v157 := v156
              let rec b30 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) (v6 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v159 ← Range_has_0 r7 v6
                if v159 then
                  let v161 ← Range_at_0 r7 v6
                  let i_162 := v161
                  let v163 ← MemArray_index_ref_u64 base i_162
                  let v164 ← mloadInt (.u 64) v163
                  let (v166, v166_k) ← MemArray_index_mut_u64 factor i_162
                  mstoreInt (.u 64) v166 v164
                  let v166_0 ← v166_k v166
                  factor := v166_0
                  let v167 := ()
                  let v169 ← Range_next_0 r7 v6
                  let v170 := v169
                  b30 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v170
                else
                  let v171 := (UInt256.ofNat 1)
                  let v172 := low - v171
                  let (v174, v174_k) ← MemArray_index_mut_u64 factor v172
                  let v175 ← mloadInt (.u 64) v174
                  let v176 := v175 &&& mask
                  mstoreInt (.u 64) v174 v176
                  let v174_0 ← v174_k v174
                  factor := v174_0
                  let v177 := ()
                  let v178 := (UInt256.ofNat 1)
                  let v179 := (UInt256.ofNat 0)
                  let (v181, v181_k) ← MemArray_index_mut_u64 result v179
                  mstoreInt (.u 64) v181 v178
                  let v181_0 ← v181_k v181
                  result := v181_0
                  let v182 := ()
                  let start_183 := start_52
                  index_r5 := start_183
                  let rec b34 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) : FeM Option_MemArray_u64 := do
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
                    let index_184 := index_r5
                    let index_185 := index_184
                    let v186 := exponent.total
                    let v187 := decide (index_185.toNat < v186.toNat)
                    let b36 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
                      let result_264 := result
                      let result_265 := result_264
                      let v266 := Option_MemArray_u64.Some result_265
                      let _ ← MemArray_drop_u64 work
                      let _ ← MemArray_drop_u64 factor
                      return v266
                    if v187 then
                      let v189 ← MemArray_as_slice_ref_u64 result
                      let v191 ← MemArray_as_slice_ref_u64 result
                      let (v193_g, v193_k) ← MemArray_as_slice_mut_u64 work
                      v193 := v193_g
                      let (v197, v197_0) ← low_product v189 v191 v193
                      v193 := v197_0
                      let v193_0 ← v193_k v193
                      work := v193_0
                      let index_198 := index_r5
                      let index_199 := index_198
                      let v200 ← Exponent_bit exponent index_199
                      let v201 := (UInt256.ofNat 1)
                      let v202 := decide (v200 = v201)
                      let b39 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
                        let v221 := (UInt256.ofNat 1)
                        let v222 := low - v221
                        let (v224, v224_k) ← MemArray_index_mut_u64 result v222
                        let v225 ← mloadInt (.u 64) v224
                        let v226 := v225 &&& mask
                        mstoreInt (.u 64) v224 v226
                        let v224_0 ← v224_k v224
                        result := v224_0
                        let v227 := ()
                        let v228 := true
                        let zero_229 := v228
                        zero := zero_229
                        let v230 := (UInt256.ofNat 0)
                        let v231 := ()
                        let v232 := (Range_Known_0_Unknown.mk v231 low)
                        r8 := v232
                        let v234 ← Range_start_0 r8
                        let v235 := v234
                        let rec b40 (result : MemArray_u64) (power : UInt256) (index_r2 : UInt256) (factor : MemArray_u64) (work : MemArray_u64) (index_r5 : UInt256) (zero : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) (v7 : UInt256) : FeM Option_MemArray_u64 := do
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
                          let v237 ← Range_has_0 r8 v7
                          if v237 then
                            let v239 ← Range_at_0 r8 v7
                            let i_240 := v239
                            let zero_241 := zero
                            let b46 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → Bool → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v8 => do
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
                              let zero_251 := v8
                              zero := zero_251
                              let v252 := ()
                              let v254 ← Range_next_0 r8 v7
                              let v255 := v254
                              b40 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v255
                            let b45 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
                              let v249 := false
                              let v250 := v249
                              b46 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v250
                            if zero_241 then
                              let v243 ← MemArray_index_ref_u64 result i_240
                              let v244 ← mloadInt (.u 64) v243
                              let v245 := (UInt256.ofNat 0)
                              let v246 := decide (v244 = v245)
                              if v246 then
                                let v247 := true
                                let v248 := v247
                                b46 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v248
                              else
                                b45 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                            else
                              b45 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                          else
                            let zero_256 := zero
                            if zero_256 then
                              b36 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                            else
                              let v257 := ()
                              let v258 := v257
                              let v259 := (UInt256.ofNat 1)
                              let index_260 := index_r5
                              let v261 := index_260 + v259
                              let index_262 := v261
                              index_r5 := index_262
                              let v263 := ()
                              b34 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                        partial_fixpoint
                        b40 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v235
                      if v202 then
                        let v204 ← MemArray_as_slice_ref_u64 work
                        let v206 ← MemArray_as_slice_ref_u64 factor
                        let (v208_g, v208_k) ← MemArray_as_slice_mut_u64 result
                        v208 := v208_g
                        let (v212, v212_0) ← low_product v204 v206 v208
                        v208 := v212_0
                        let v208_0 ← v208_k v208
                        result := v208_0
                        let v213 := v212
                        b39 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                      else
                        let v215 ← MemArray_as_slice_ref_u64 work
                        let (v217_g, v217_k) ← MemArray_as_slice_mut_u64 result
                        v217 := v217_g
                        let (v219, v219_0) ← MemSlice_copy_from_u64 v217 v215
                        v217 := v219_0
                        let v217_0 ← v217_k v217
                        result := v217_0
                        let v220 := v219
                        b39 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                    else
                      b36 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
                  partial_fixpoint
                  b34 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
              partial_fixpoint
              b30 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v157
            else
              let v267 := Option_MemArray_u64.None
              match (← pure v148) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v148.Some_0
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 factor
              let _ ← MemArray_drop_u64 result
              return v267
          else
            let v268 := Option_MemArray_u64.None
            match (← pure v144) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v144.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 result
            return v268
      let b7 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 => do
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
        let v50 := exponent.top
        let v51 := v50
        b8 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v51
      if v40 then
        let v41 := exponent.total
        let v42 := exponent.top
        let v43 := v41 - v42
        let k_44 := k_2
        let v45 := decide (k_44.toNat < v43.toNat)
        if v45 then
          let v46 := exponent.total
          let k_47 := k_2
          let v48 := v46 - k_47
          let v49 := v48
          b8 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v49
        else
          b7 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
      else
        b7 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217
    else
      let v269 := Option_MemArray_u64.None
      match (← pure v30) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v30.Some_0
        pure ()
      | _ => pure ()
      return v269
  if v21 then
    let v22 := (UInt256.ofNat 18446744073709551615)
    let v23 := v22
    b3 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v23
  else
    let v24 := (UInt256.ofNat 1)
    let v25 := IntTy.shl (.u 64) v24 bits
    let v26 := (UInt256.ofNat 1)
    let v27 := IntTy.subW (.u 64) v25 v26
    let v28 := v27
    b3 result power index_r2 factor work index_r5 zero r7 r8 v193 v208 v217 v28

def even_power (exponent : Exponent) (base : MemArray_u64) (base_size : UInt256) (value : MemArray_u64) (size_4 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
  let v20 := (UInt256.ofNat 0)
  let zero_limbs_21 := v20
  zero_limbs := zero_limbs_21
  let rec b1 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
    let zero_limbs_22 := zero_limbs
    let zero_limbs_23 := zero_limbs_22
    let v24 ← MemArray_index_ref_u64 value zero_limbs_23
    let v25 ← mloadInt (.u 64) v24
    let v26 := (UInt256.ofNat 0)
    let v27 := decide (v25 = v26)
    if v27 then
      let v28 := (UInt256.ofNat 1)
      let zero_limbs_29 := zero_limbs
      let v30 := zero_limbs_29 + v28
      let zero_limbs_31 := v30
      zero_limbs := zero_limbs_31
      let v32 := ()
      b1 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
    else
      let zero_limbs_33 := zero_limbs
      let zero_limbs_34 := zero_limbs_33
      let v35 := (UInt256.ofNat 64)
      let v36 := zero_limbs_34 * v35
      let zero_limbs_37 := zero_limbs
      let zero_limbs_38 := zero_limbs_37
      let v39 ← MemArray_index_ref_u64 value zero_limbs_38
      let v40 ← mloadInt (.u 64) v39
      let v41 := IntTy.trailingZeros (.u 64) v40
      let v42 ← downcast_truncate_u64_usize v41
      let v43 := v36 + v42
      let k := v43
      let v45 := (UInt256.ofNat 63)
      let v46 := k + v45
      let v47 := (UInt256.ofNat 64)
      let v48 ← checkedDiv v46 v47
      let low_49 := v48
      let v50 := (UInt256.ofNat 64)
      let v51 ← checkedRem k v50
      let v52 ← downcast_truncate_usize_u64 v51
      let bits := v52
      let v54 := (UInt256.ofNat 0)
      let v55 := decide (bits = v54)
      let b6 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v5 => do
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
        let mask := v5
        let v64 ← MemArray_try_new_zeroed_u64 size_4
        let v65 := match v64 with | .Some .. => true | _ => false
        if v65 then
          let v66 := v64.Some_0
          let result_67 := v66
          result := result_67
          let v70 ← low_power exponent base k
          let v71 := match v70 with | .Some .. => true | _ => false
          if v71 then
            let v72 := v70.Some_0
            let y_73 := v72
            y := y_73
            let v74 ← MemArray_try_new_zeroed_u64 size_4
            let v75 := match v74 with | .Some .. => true | _ => false
            if v75 then
              let v76 := v74.Some_0
              let q_77 := v76
              q := q_77
              let v78 := (UInt256.ofNat 0)
              let size_79 := size_4
              let v80 := (UInt256.ofNat 64)
              let v81 ← checkedDiv k v80
              let v82 := size_79 - v81
              let v83 := ()
              let v84 := (Range_Known_0_Unknown.mk v83 v82)
              r19 := v84
              let v86 ← Range_start_0 r19
              let v87 := v86
              let rec b13 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v6 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                let v89 ← Range_has_0 r19 v6
                if v89 then
                  let v91 ← Range_at_0 r19 v6
                  let i_92 := v91
                  let v93 := (UInt256.ofNat 64)
                  let v94 ← checkedDiv k v93
                  let v95 := i_92 + v94
                  let v96 := (UInt256.ofNat 1)
                  let v97 := v95 + v96
                  let size_98 := size_4
                  let v99 := decide (v97.toNat < size_98.toNat)
                  let b19 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v7 => do
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
                    let next_110 := v7
                    let v111 := (UInt256.ofNat 0)
                    let v112 := decide (bits = v111)
                    let b22 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v8 => do
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
                      let (v131, v131_k) ← MemArray_index_mut_u64 q i_92
                      mstoreInt (.u 64) v131 v8
                      let v131_0 ← v131_k v131
                      q := v131_0
                      let v132 := ()
                      let v134 ← Range_next_0 r19 v6
                      let v135 := v134
                      b13 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v135
                    if v112 then
                      let v113 := (UInt256.ofNat 64)
                      let v114 ← checkedDiv k v113
                      let v115 := i_92 + v114
                      let v116 ← MemArray_index_ref_u64 value v115
                      let v117 ← mloadInt (.u 64) v116
                      let v118 := v117
                      b22 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v118
                    else
                      let v119 := (UInt256.ofNat 64)
                      let v120 ← checkedDiv k v119
                      let v121 := i_92 + v120
                      let v122 ← MemArray_index_ref_u64 value v121
                      let v123 ← mloadInt (.u 64) v122
                      let v124 := v123 >>> bits
                      let v125 := (UInt256.ofNat 64)
                      let v126 := IntTy.subW (.u 64) v125 bits
                      let v127 := IntTy.shl (.u 64) next_110 v126
                      let v128 := v124 ||| v127
                      let v129 := v128
                      b22 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v129
                  if v99 then
                    let v100 := (UInt256.ofNat 64)
                    let v101 ← checkedDiv k v100
                    let v102 := i_92 + v101
                    let v103 := (UInt256.ofNat 1)
                    let v104 := v102 + v103
                    let v105 ← MemArray_index_ref_u64 value v104
                    let v106 ← mloadInt (.u 64) v105
                    let v107 := v106
                    b19 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v107
                  else
                    let v108 := (UInt256.ofNat 0)
                    let v109 := v108
                    b19 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v109
                else
                  let size_136 := size_4
                  let size_137 := size_136
                  q_len := size_137
                  let rec b23 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                    let q_len_138 := q_len
                    let q_len_139 := q_len_138
                    let v140 := (UInt256.ofNat 1)
                    let v141 := q_len_139 - v140
                    let v143 ← MemArray_index_ref_u64 q v141
                    let v144 ← mloadInt (.u 64) v143
                    let v145 := (UInt256.ofNat 0)
                    let v146 := decide (v144 = v145)
                    if v146 then
                      let v147 := (UInt256.ofNat 1)
                      let q_len_148 := q_len
                      let v149 := q_len_148 - v147
                      let q_len_150 := v149
                      q_len := q_len_150
                      let v151 := ()
                      b23 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                    else
                      let q_len_152 := q_len
                      let q_len_153 := q_len_152
                      let v154 := (UInt256.ofNat 1)
                      let v155 := decide (q_len_153 = v154)
                      let b27 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 => do
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
                        let v185 := ()
                        let v186 := v185
                        let q_len_189 := q_len
                        let q_len_190 := q_len_189
                        let (v192, v192_0) ← odd_power exponent base base_size q q_len_190
                        base := v192_0
                        let v193 := match v192 with | .Some .. => true | _ => false
                        if v193 then
                          let v194 := v192.Some_0
                          let x_195 := v194
                          x := x_195
                          let v196 ← MemArray_try_new_zeroed_u64 low_49
                          let v197 := match v196 with | .Some .. => true | _ => false
                          if v197 then
                            let v198 := v196.Some_0
                            let t_199 := v198
                            t := t_199
                            let v200 := (UInt256.ofNat 0)
                            let borrow_201 := v200
                            borrow := borrow_201
                            let v202 := (UInt256.ofNat 0)
                            let v203 := ()
                            let v204 := (Range_Known_0_Unknown.mk v203 low_49)
                            r21 := v204
                            let v206 ← Range_start_0 r21
                            let v207 := v206
                            let rec b38 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v10 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                              let v209 ← Range_has_0 r21 v10
                              if v209 then
                                let v211 ← Range_at_0 r21 v10
                                let i_212 := v211
                                let q_len_213 := q_len
                                let v214 := decide (i_212.toNat < q_len_213.toNat)
                                let b44 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v11 => do
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
                                  let x_limb := v11
                                  let v223 ← MemArray_index_ref_u64 y i_212
                                  let v224 ← mloadInt (.u 64) v223
                                  let v225 := v224
                                  let v226 := x_limb
                                  let v227 := IntTy.subW (.u 128) v225 v226
                                  let borrow_228 := borrow
                                  let v229 := borrow_228
                                  let v230 := IntTy.subW (.u 128) v227 v229
                                  let difference := v230
                                  let v232 ← downcast_truncate_u128_u64 difference
                                  let (v234, v234_k) ← MemArray_index_mut_u64 t i_212
                                  mstoreInt (.u 64) v234 v232
                                  let v234_0 ← v234_k v234
                                  t := v234_0
                                  let v235 := ()
                                  let v236 := (UInt256.ofNat 127)
                                  let v237 := difference >>> v236
                                  let v238 ← downcast_truncate_u128_u64 v237
                                  let borrow_239 := v238
                                  borrow := borrow_239
                                  let v240 := ()
                                  let v242 ← Range_next_0 r21 v10
                                  let v243 := v242
                                  b38 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v243
                                if v214 then
                                  let v216 ← MemArray_index_ref_u64 x i_212
                                  let v217 ← mloadInt (.u 64) v216
                                  let v218 := v217
                                  b44 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v218
                                else
                                  let v219 := (UInt256.ofNat 0)
                                  let v220 := v219
                                  b44 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v220
                              else
                                let v244 := (UInt256.ofNat 1)
                                let v245 := low_49 - v244
                                let (v247, v247_k) ← MemArray_index_mut_u64 t v245
                                let v248 ← mloadInt (.u 64) v247
                                let v249 := v248 &&& mask
                                mstoreInt (.u 64) v247 v249
                                let v247_0 ← v247_k v247
                                t := v247_0
                                let v250 := ()
                                let v251 ← MemArray_try_new_zeroed_u64 low_49
                                let v252 := match v251 with | .Some .. => true | _ => false
                                if v252 then
                                  let v253 := v251.Some_0
                                  let inverse_254 := v253
                                  inverse := inverse_254
                                  let v255 := (UInt256.ofNat 0)
                                  let v257 ← MemArray_index_ref_u64 q v255
                                  let guess_258 ← mloadInt (.u 64) v257
                                  guess := guess_258
                                  let v259 := (UInt256.ofNat 0)
                                  let v260 := (UInt256.ofNat 5)
                                  let v261 := ()
                                  let v262 := (Range_Known_0_Known_5.mk v261 v261)
                                  r22 := v262
                                  let v264 ← Range_start_0_5 r22
                                  let v265 := v264
                                  let rec b47 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v12 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                    let v267 ← Range_has_0_5 r22 v12
                                    if v267 then
                                      let v269 ← Range_at_0_5 r22 v12
                                      let guess_270 := guess
                                      let guess_271 := guess_270
                                      let v272 := (UInt256.ofNat 2)
                                      let v273 := (UInt256.ofNat 0)
                                      let v275 ← MemArray_index_ref_u64 q v273
                                      let v276 ← mloadInt (.u 64) v275
                                      let guess_277 := guess
                                      let v278 := IntTy.mulW (.u 64) v276 guess_277
                                      let v279 := IntTy.subW (.u 64) v272 v278
                                      let v280 := IntTy.mulW (.u 64) guess_271 v279
                                      let guess_281 := v280
                                      guess := guess_281
                                      let v282 := ()
                                      let v284 ← Range_next_0_5 r22 v12
                                      let v285 := v284
                                      b47 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v285
                                    else
                                      let guess_286 := guess
                                      let guess_287 := guess_286
                                      let v288 := (UInt256.ofNat 0)
                                      let (v290, v290_k) ← MemArray_index_mut_u64 inverse v288
                                      mstoreInt (.u 64) v290 guess_287
                                      let v290_0 ← v290_k v290
                                      inverse := v290_0
                                      let v291 := ()
                                      let v292 ← MemArray_try_new_zeroed_u64 low_49
                                      let v293 := match v292 with | .Some .. => true | _ => false
                                      if v293 then
                                        let v294 := v292.Some_0
                                        let scratch_295 := v294
                                        scratch := scratch_295
                                        let v296 ← MemArray_try_new_zeroed_u64 low_49
                                        let v297 := match v296 with | .Some .. => true | _ => false
                                        if v297 then
                                          let v298 := v296.Some_0
                                          let correction_299 := v298
                                          correction := correction_299
                                          let v300 := (UInt256.ofNat 1)
                                          let correct_301 := v300
                                          correct := correct_301
                                          let rec b55 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                            let correct_302 := correct
                                            let correct_303 := correct_302
                                            let v304 := decide (correct_303.toNat < low_49.toNat)
                                            if v304 then
                                              let v305 := (UInt256.ofNat 2)
                                              let correct_306 := correct
                                              let v307 := v305 * correct_306
                                              let v308 := decide (v307.toNat < low_49.toNat)
                                              let b60 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v13 => do
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
                                                let next_314 := v13
                                                let v316 ← MemArray_as_slice_ref_u64 q
                                                let v318 ← MemArray_as_slice_ref_u64 inverse
                                                let (v320_g, v320_k) ← MemArray_as_slice_mut_u64 scratch
                                                v320 := v320_g
                                                let (v321_g, v321_k) ← MemSlice_head_mut_u64 v320 next_314
                                                v321 := v321_g
                                                let (v325, v325_0) ← low_product v316 v318 v321
                                                v321 := v325_0
                                                let v321_0 ← v321_k v321
                                                v320 := v321_0
                                                let v320_0 ← v320_k v320
                                                scratch := v320_0
                                                let v326 := (UInt256.ofNat 1)
                                                let carry_327 := v326
                                                carry_r13 := carry_327
                                                let v328 := (UInt256.ofNat 0)
                                                let v329 := ()
                                                let v330 := (Range_Known_0_Unknown.mk v329 next_314)
                                                r23 := v330
                                                let v332 ← Range_start_0 r23
                                                let v333 := v332
                                                let rec b61 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v14 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                  let v335 ← Range_has_0 r23 v14
                                                  if v335 then
                                                    let v337 ← Range_at_0 r23 v14
                                                    let i_338 := v337
                                                    let v340 ← MemArray_index_ref_u64 scratch i_338
                                                    let v341 ← mloadInt (.u 64) v340
                                                    let v342 := IntTy.not (.u 64) v341
                                                    let v343 := v342
                                                    let carry_344 := carry_r13
                                                    let v345 := carry_344
                                                    let v346 := IntTy.addW (.u 128) v343 v345
                                                    let negated := v346
                                                    let v348 ← downcast_truncate_u128_u64 negated
                                                    let (v350, v350_k) ← MemArray_index_mut_u64 correction i_338
                                                    mstoreInt (.u 64) v350 v348
                                                    let v350_0 ← v350_k v350
                                                    correction := v350_0
                                                    let v351 := ()
                                                    let v352 := (UInt256.ofNat 64)
                                                    let v353 := negated >>> v352
                                                    let v354 ← downcast_truncate_u128_u64 v353
                                                    let carry_355 := v354
                                                    carry_r13 := carry_355
                                                    let v356 := ()
                                                    let v358 ← Range_next_0 r23 v14
                                                    let v359 := v358
                                                    b61 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v359
                                                  else
                                                    let v360 := (UInt256.ofNat 2)
                                                    let add_361 := v360
                                                    add := add_361
                                                    let v362 := (UInt256.ofNat 0)
                                                    let v363 := ()
                                                    let v364 := (Range_Known_0_Unknown.mk v363 next_314)
                                                    r24 := v364
                                                    let v366 ← Range_start_0 r24
                                                    let v367 := v366
                                                    let rec b65 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v15 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                      let v369 ← Range_has_0 r24 v15
                                                      if v369 then
                                                        let v371 ← Range_at_0 r24 v15
                                                        let i_372 := v371
                                                        let v374 ← MemArray_index_ref_u64 correction i_372
                                                        let v375 ← mloadInt (.u 64) v374
                                                        let v376 := v375
                                                        let add_377 := add
                                                        let v378 := add_377
                                                        let v379 := IntTy.addW (.u 128) v376 v378
                                                        let sum_380 := v379
                                                        let v381 ← downcast_truncate_u128_u64 sum_380
                                                        let (v383, v383_k) ← MemArray_index_mut_u64 correction i_372
                                                        mstoreInt (.u 64) v383 v381
                                                        let v383_0 ← v383_k v383
                                                        correction := v383_0
                                                        let v384 := ()
                                                        let v385 := (UInt256.ofNat 64)
                                                        let v386 := sum_380 >>> v385
                                                        let v387 ← downcast_truncate_u128_u64 v386
                                                        let add_388 := v387
                                                        add := add_388
                                                        let v389 := ()
                                                        let v391 ← Range_next_0 r24 v15
                                                        let v392 := v391
                                                        b65 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v392
                                                      else
                                                        let v394 ← MemArray_as_slice_ref_u64 inverse
                                                        let v396 ← MemArray_as_slice_ref_u64 correction
                                                        let (v398_g, v398_k) ← MemArray_as_slice_mut_u64 scratch
                                                        v398 := v398_g
                                                        let (v399_g, v399_k) ← MemSlice_head_mut_u64 v398 next_314
                                                        v399 := v399_g
                                                        let (v403, v403_0) ← low_product v394 v396 v399
                                                        v399 := v403_0
                                                        let v399_0 ← v399_k v399
                                                        v398 := v399_0
                                                        let v398_0 ← v398_k v398
                                                        scratch := v398_0
                                                        let v405 ← MemArray_as_slice_ref_u64 scratch
                                                        let v407 ← MemSlice_head_ref_u64 v405 next_314
                                                        let (v409_g, v409_k) ← MemArray_as_slice_mut_u64 inverse
                                                        v409 := v409_g
                                                        let (v410_g, v410_k) ← MemSlice_head_mut_u64 v409 next_314
                                                        v410 := v410_g
                                                        let (v412, v412_0) ← MemSlice_copy_from_u64 v410 v407
                                                        v410 := v412_0
                                                        let v410_0 ← v410_k v410
                                                        v409 := v410_0
                                                        let v409_0 ← v409_k v409
                                                        inverse := v409_0
                                                        let next_413 := next_314
                                                        correct := next_413
                                                        let v414 := ()
                                                        b55 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                                    partial_fixpoint
                                                    b65 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v367
                                                partial_fixpoint
                                                b61 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v333
                                              if v308 then
                                                let v309 := (UInt256.ofNat 2)
                                                let correct_310 := correct
                                                let v311 := v309 * correct_310
                                                let v312 := v311
                                                b60 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v312
                                              else
                                                let low_313 := low_49
                                                b60 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 low_313
                                            else
                                              let v415 ← MemArray_try_new_zeroed_u64 low_49
                                              let v416 := match v415 with | .Some .. => true | _ => false
                                              if v416 then
                                                let v417 := v415.Some_0
                                                let h_418 := v417
                                                h := h_418
                                                let v420 ← MemArray_as_slice_ref_u64 t
                                                let v422 ← MemArray_as_slice_ref_u64 inverse
                                                let (v424_g, v424_k) ← MemArray_as_slice_mut_u64 h
                                                v424 := v424_g
                                                let (v428, v428_0) ← low_product v420 v422 v424
                                                v424 := v428_0
                                                let v424_0 ← v424_k v424
                                                h := v424_0
                                                let v429 := (UInt256.ofNat 1)
                                                let v430 := low_49 - v429
                                                let (v432, v432_k) ← MemArray_index_mut_u64 h v430
                                                let v433 ← mloadInt (.u 64) v432
                                                let v434 := v433 &&& mask
                                                mstoreInt (.u 64) v432 v434
                                                let v432_0 ← v432_k v432
                                                h := v432_0
                                                let v435 := ()
                                                let q_len_436 := q_len
                                                let q_len_437 := q_len_436
                                                let v438 := q_len_437 + low_49
                                                let v439 := (UInt256.ofNat 1)
                                                let v440 := v438 + v439
                                                let v441 ← MemArray_try_new_zeroed_u64 v440
                                                let v442 := match v441 with | .Some .. => true | _ => false
                                                if v442 then
                                                  let v443 := v441.Some_0
                                                  let work_444 := v443
                                                  work := work_444
                                                  let v445 := (UInt256.ofNat 0)
                                                  let v446 := ()
                                                  let q_len_447 := q_len
                                                  let v448 := (Range_Known_0_Unknown.mk v446 q_len_447)
                                                  r25 := v448
                                                  let v450 ← Range_start_0 r25
                                                  let v451 := v450
                                                  let rec b73 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v16 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                    let v453 ← Range_has_0 r25 v16
                                                    if v453 then
                                                      let v455 ← Range_at_0 r25 v16
                                                      let i_456 := v455
                                                      let v458 ← MemArray_index_ref_u64 x i_456
                                                      let v459 ← mloadInt (.u 64) v458
                                                      let (v461, v461_k) ← MemArray_index_mut_u64 work i_456
                                                      mstoreInt (.u 64) v461 v459
                                                      let v461_0 ← v461_k v461
                                                      work := v461_0
                                                      let v462 := ()
                                                      let v464 ← Range_next_0 r25 v16
                                                      let v465 := v464
                                                      b73 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v465
                                                    else
                                                      let v466 := (UInt256.ofNat 0)
                                                      let v467 := ()
                                                      let q_len_468 := q_len
                                                      let v469 := (Range_Known_0_Unknown.mk v467 q_len_468)
                                                      r26 := v469
                                                      let v471 ← Range_start_0 r26
                                                      let v472 := v471
                                                      let rec b77 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v17 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                        let v474 ← Range_has_0 r26 v17
                                                        if v474 then
                                                          let v476 ← Range_at_0 r26 v17
                                                          let i_477 := v476
                                                          let v478 := (UInt256.ofNat 0)
                                                          let carry_479 := v478
                                                          carry_r17 := carry_479
                                                          let v480 := (UInt256.ofNat 0)
                                                          let v481 := ()
                                                          let v482 := (Range_Known_0_Unknown.mk v481 low_49)
                                                          r27 := v482
                                                          let v484 ← Range_start_0 r27
                                                          let v485 := v484
                                                          let rec b81 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v18 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                            let v487 ← Range_has_0 r27 v18
                                                            if v487 then
                                                              let v489 ← Range_at_0 r27 v18
                                                              let j := v489
                                                              let v491 := i_477 + j
                                                              let v493 ← MemArray_index_ref_u64 work v491
                                                              let v494 ← mloadInt (.u 64) v493
                                                              let v495 := v494
                                                              let v497 ← MemArray_index_ref_u64 q i_477
                                                              let v498 ← mloadInt (.u 64) v497
                                                              let v499 := v498
                                                              let v501 ← MemArray_index_ref_u64 h j
                                                              let v502 ← mloadInt (.u 64) v501
                                                              let v503 := v502
                                                              let v504 := IntTy.mulW (.u 128) v499 v503
                                                              let v505 := IntTy.addW (.u 128) v495 v504
                                                              let carry_506 := carry_r17
                                                              let v507 := carry_506
                                                              let v508 := IntTy.addW (.u 128) v505 v507
                                                              let sum_509 := v508
                                                              let v510 ← downcast_truncate_u128_u64 sum_509
                                                              let v511 := i_477 + j
                                                              let (v513, v513_k) ← MemArray_index_mut_u64 work v511
                                                              mstoreInt (.u 64) v513 v510
                                                              let v513_0 ← v513_k v513
                                                              work := v513_0
                                                              let v514 := ()
                                                              let v515 := (UInt256.ofNat 64)
                                                              let v516 := sum_509 >>> v515
                                                              let v517 ← downcast_truncate_u128_u64 v516
                                                              let carry_518 := v517
                                                              carry_r17 := carry_518
                                                              let v519 := ()
                                                              let v521 ← Range_next_0 r27 v18
                                                              let v522 := v521
                                                              b81 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v522
                                                            else
                                                              let v523 := i_477 + low_49
                                                              let at_524 := v523
                                                              «at» := at_524
                                                              let rec b85 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                                let carry_525 := carry_r17
                                                                let carry_526 := carry_525
                                                                let v527 := (UInt256.ofNat 0)
                                                                let v528 := decide (carry_526 ≠ v527)
                                                                if v528 then
                                                                  let at_529 := «at»
                                                                  let at_530 := at_529
                                                                  let v532 ← MemArray_index_ref_u64 work at_530
                                                                  let v533 ← mloadInt (.u 64) v532
                                                                  let v534 := v533
                                                                  let carry_535 := carry_r17
                                                                  let v536 := carry_535
                                                                  let v537 := IntTy.addW (.u 128) v534 v536
                                                                  let sum_538 := v537
                                                                  let v539 ← downcast_truncate_u128_u64 sum_538
                                                                  let at_540 := «at»
                                                                  let at_541 := at_540
                                                                  let (v543, v543_k) ← MemArray_index_mut_u64 work at_541
                                                                  mstoreInt (.u 64) v543 v539
                                                                  let v543_0 ← v543_k v543
                                                                  work := v543_0
                                                                  let v544 := ()
                                                                  let v545 := (UInt256.ofNat 64)
                                                                  let v546 := sum_538 >>> v545
                                                                  let v547 ← downcast_truncate_u128_u64 v546
                                                                  let carry_548 := v547
                                                                  carry_r17 := carry_548
                                                                  let v549 := ()
                                                                  let v550 := (UInt256.ofNat 1)
                                                                  let at_551 := «at»
                                                                  let v552 := at_551 + v550
                                                                  let at_553 := v552
                                                                  «at» := at_553
                                                                  let v554 := ()
                                                                  b85 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                                                else
                                                                  let v555 := ()
                                                                  let v557 ← Range_next_0 r26 v17
                                                                  let v558 := v557
                                                                  b77 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v558
                                                              partial_fixpoint
                                                              b85 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                                          partial_fixpoint
                                                          b81 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v485
                                                        else
                                                          let v559 := (UInt256.ofNat 0)
                                                          let size_560 := size_4
                                                          let v561 := ()
                                                          let v562 := (Range_Known_0_Unknown.mk v561 size_560)
                                                          r28 := v562
                                                          let v564 ← Range_start_0 r28
                                                          let v565 := v564
                                                          let rec b88 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v19 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                            let v567 ← Range_has_0 r28 v19
                                                            if v567 then
                                                              let v569 ← Range_at_0 r28 v19
                                                              let i_570 := v569
                                                              let v572 ← MemArray_index_ref_u64 work i_570
                                                              let v573 ← mloadInt (.u 64) v572
                                                              let (v575, v575_k) ← MemArray_index_mut_u64 result i_570
                                                              mstoreInt (.u 64) v575 v573
                                                              let v575_0 ← v575_k v575
                                                              result := v575_0
                                                              let v576 := ()
                                                              let v578 ← Range_next_0 r28 v19
                                                              let v579 := v578
                                                              b88 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v579
                                                            else
                                                              let result_580 := result
                                                              let result_581 := result_580
                                                              let v582 := Option_MemArray_u64.Some result_581
                                                              let _ ← MemArray_drop_u64 work
                                                              let _ ← MemArray_drop_u64 h
                                                              let _ ← MemArray_drop_u64 correction
                                                              let _ ← MemArray_drop_u64 scratch
                                                              let _ ← MemArray_drop_u64 inverse
                                                              let _ ← MemArray_drop_u64 t
                                                              let _ ← MemArray_drop_u64 x
                                                              let _ ← MemArray_drop_u64 q
                                                              let _ ← MemArray_drop_u64 y
                                                              return (v582, base)
                                                          partial_fixpoint
                                                          b88 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v565
                                                      partial_fixpoint
                                                      b77 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v472
                                                  partial_fixpoint
                                                  b73 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v451
                                                else
                                                  let v583 := Option_MemArray_u64.None
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
                                                  return (v583, base)
                                              else
                                                let v584 := Option_MemArray_u64.None
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
                                                return (v584, base)
                                          partial_fixpoint
                                          b55 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                                        else
                                          let v585 := Option_MemArray_u64.None
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
                                          return (v585, base)
                                      else
                                        let v586 := Option_MemArray_u64.None
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
                                        return (v586, base)
                                  partial_fixpoint
                                  b47 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v265
                                else
                                  let v587 := Option_MemArray_u64.None
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
                                  return (v587, base)
                            partial_fixpoint
                            b38 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v207
                          else
                            let v588 := Option_MemArray_u64.None
                            match (← pure v196) with
                            | .Some .. =>
                              let _ ← MemArray_drop_u64 v196.Some_0
                              pure ()
                            | _ => pure ()
                            let _ ← MemArray_drop_u64 x
                            let _ ← MemArray_drop_u64 q
                            let _ ← MemArray_drop_u64 y
                            let _ ← MemArray_drop_u64 result
                            return (v588, base)
                        else
                          let v589 := Option_MemArray_u64.None
                          match (← pure v192) with
                          | .Some .. =>
                            let _ ← MemArray_drop_u64 v192.Some_0
                            pure ()
                          | _ => pure ()
                          let _ ← MemArray_drop_u64 q
                          let _ ← MemArray_drop_u64 y
                          let _ ← MemArray_drop_u64 result
                          return (v589, base)
                      if v155 then
                        let v156 := (UInt256.ofNat 0)
                        let v158 ← MemArray_index_ref_u64 q v156
                        let v159 ← mloadInt (.u 64) v158
                        let v160 := (UInt256.ofNat 1)
                        let v161 := decide (v159 = v160)
                        if v161 then
                          let v162 := (UInt256.ofNat 0)
                          let v163 := ()
                          let v164 := (Range_Known_0_Unknown.mk v163 low_49)
                          r20 := v164
                          let v166 ← Range_start_0 r20
                          let v167 := v166
                          let rec b30 (zero_limbs : UInt256) (result : MemArray_u64) (y : MemArray_u64) (q : MemArray_u64) (q_len : UInt256) (x : MemArray_u64) (t : MemArray_u64) (borrow : UInt256) (inverse : MemArray_u64) (guess : UInt256) (scratch : MemArray_u64) (correction : MemArray_u64) (correct : UInt256) (carry_r13 : UInt256) (add : UInt256) (h : MemArray_u64) (work : MemArray_u64) (carry_r17 : UInt256) («at» : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (base : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v9 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                            let v169 ← Range_has_0 r20 v9
                            if v169 then
                              let v171 ← Range_at_0 r20 v9
                              let i_172 := v171
                              let v174 ← MemArray_index_ref_u64 y i_172
                              let v175 ← mloadInt (.u 64) v174
                              let (v177, v177_k) ← MemArray_index_mut_u64 result i_172
                              mstoreInt (.u 64) v177 v175
                              let v177_0 ← v177_k v177
                              result := v177_0
                              let v178 := ()
                              let v180 ← Range_next_0 r20 v9
                              let v181 := v180
                              b30 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v181
                            else
                              let result_182 := result
                              let result_183 := result_182
                              let v184 := Option_MemArray_u64.Some result_183
                              let _ ← MemArray_drop_u64 q
                              let _ ← MemArray_drop_u64 y
                              return (v184, base)
                          partial_fixpoint
                          b30 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v167
                        else
                          b27 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                      else
                        b27 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
                  partial_fixpoint
                  b23 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424
              partial_fixpoint
              b13 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v87
            else
              let v590 := Option_MemArray_u64.None
              match (← pure v74) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v74.Some_0
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 y
              let _ ← MemArray_drop_u64 result
              return (v590, base)
          else
            let v591 := Option_MemArray_u64.None
            match (← pure v70) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v70.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 result
            return (v591, base)
        else
          let v592 := Option_MemArray_u64.None
          match (← pure v64) with
          | .Some .. =>
            let _ ← MemArray_drop_u64 v64.Some_0
            pure ()
          | _ => pure ()
          return (v592, base)
      if v55 then
        let v56 := (UInt256.ofNat 18446744073709551615)
        let v57 := v56
        b6 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v57
      else
        let v58 := (UInt256.ofNat 1)
        let v59 := IntTy.shl (.u 64) v58 bits
        let v60 := (UInt256.ofNat 1)
        let v61 := IntTy.subW (.u 64) v59 v60
        let v62 := v61
        b6 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424 v62
  partial_fixpoint
  b1 zero_limbs result y q q_len x t borrow inverse guess scratch correction correct carry_r13 add h work carry_r17 «at» r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 base v320 v321 v398 v399 v409 v410 v424

def power_MemBuffer (input : MemBuffer) (base_len_1 : UInt256) (exponent_len_2 : UInt256) (modulus_len_3 : UInt256) : FeM Option_MemArray_u64 := do
  let mut value : MemArray_u64 := default
  let mut n : UInt256 := default
  let mut exponent : Exponent := default
  let mut one : MemArray_u64 := default
  let mut base : MemArray_u64 := default
  let mut base_size : UInt256 := default
  let v5 := (UInt256.ofNat 96)
  let base_len_6 := base_len_1
  let v7 ← checkedAdd v5 base_len_6
  let exponent_start := v7
  let exponent_len_9 := exponent_len_2
  let v10 ← checkedAdd exponent_start exponent_len_9
  let modulus_start := v10
  let modulus_len_12 := modulus_len_3
  let v13 := (UInt256.ofNat 7)
  let v14 ← checkedAdd modulus_len_12 v13
  let v15 := (UInt256.ofNat 8)
  let v16 ← checkedDiv v14 v15
  let v17 ← input_limbs_MemBuffer input modulus_start modulus_len_3 v16
  let v18 := match v17 with | .Some .. => true | _ => false
  if v18 then
    let v19 := v17.Some_0
    let value_20 := v19
    value := value_20
    let v22 ← MemArray_len_u64 value
    let n_23 := v22
    n := n_23
    let rec b3 (value : MemArray_u64) (n : UInt256) (exponent : Exponent) (one : MemArray_u64) (base : MemArray_u64) (base_size : UInt256) : FeM Option_MemArray_u64 := do
      let mut value := value
      let mut n := n
      let mut exponent := exponent
      let mut one := one
      let mut base := base
      let mut base_size := base_size
      let n_24 := n
      let n_25 := n_24
      let v26 := (UInt256.ofNat 0)
      let v27 := decide (v26.toNat < n_25.toNat)
      let b5 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
        let mut value := value
        let mut n := n
        let mut exponent := exponent
        let mut one := one
        let mut base := base
        let mut base_size := base_size
        let n_42 := n
        let n_43 := n_42
        let v44 := (UInt256.ofNat 0)
        let v45 := decide (n_43 = v44)
        let b7 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
          let mut value := value
          let mut n := n
          let mut exponent := exponent
          let mut one := one
          let mut base := base
          let mut base_size := base_size
          let v155 := (UInt256.ofNat 1)
          let v156 ← MemArray_try_new_zeroed_u64 v155
          let _ ← MemArray_drop_u64 value
          return v156
        if v45 then
          b7 value n exponent one base base_size
        else
          let n_46 := n
          let n_47 := n_46
          let v48 := (UInt256.ofNat 1)
          let v49 := decide (n_47 = v48)
          let b8 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
            let mut value := value
            let mut n := n
            let mut exponent := exponent
            let mut one := one
            let mut base := base
            let mut base_size := base_size
            let v56 := ()
            let v57 := v56
            let v58 ← Exponent_try_read_MemBuffer input exponent_start exponent_len_2
            let v59 := match v58 with | .Some .. => true | _ => false
            if v59 then
              let v60 := v58.Some_0
              let exponent_61 := v60
              exponent := exponent_61
              let v62 := exponent.top
              let v63 := exponent.total
              let v64 := decide (v62 = v63)
              if v64 then
                let v65 := (UInt256.ofNat 1)
                let v66 ← MemArray_try_new_zeroed_u64 v65
                let v67 := match v66 with | .Some .. => true | _ => false
                if v67 then
                  let v68 := v66.Some_0
                  let one_69 := v68
                  one := one_69
                  let v70 := (UInt256.ofNat 1)
                  let v71 := (UInt256.ofNat 0)
                  let (v73, v73_k) ← MemArray_index_mut_u64 one v71
                  mstoreInt (.u 64) v73 v70
                  let v73_0 ← v73_k v73
                  one := v73_0
                  let v74 := ()
                  let one_75 := one
                  let one_76 := one_75
                  let v77 := Option_MemArray_u64.Some one_76
                  let _ ← MemArray_drop_u64 exponent.limbs
                  let _ ← MemArray_drop_u64 value
                  return v77
                else
                  let v78 := Option_MemArray_u64.None
                  match (← pure v66) with
                  | .Some .. =>
                    let _ ← MemArray_drop_u64 v66.Some_0
                    pure ()
                  | _ => pure ()
                  let _ ← MemArray_drop_u64 exponent.limbs
                  let _ ← MemArray_drop_u64 value
                  return v78
              else
                let v79 := ()
                let v80 := v79
                let base_len_81 := base_len_1
                let v82 := (UInt256.ofNat 7)
                let v83 ← checkedAdd base_len_81 v82
                let v84 := (UInt256.ofNat 8)
                let v85 ← checkedDiv v83 v84
                let base_limbs_86 := v85
                let v87 := (UInt256.ofNat 96)
                let n_88 := n
                let v89 := decide (n_88.toNat < base_limbs_86.toNat)
                let b21 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size v4 => do
                  let mut value := value
                  let mut n := n
                  let mut exponent := exponent
                  let mut one := one
                  let mut base := base
                  let mut base_size := base_size
                  let v94 := (UInt256.ofNat 1)
                  let v95 ← checkedAdd v4 v94
                  let v96 ← input_limbs_MemBuffer input v87 base_len_1 v95
                  let v97 := match v96 with | .Some .. => true | _ => false
                  if v97 then
                    let v98 := v96.Some_0
                    let base_99 := v98
                    base := base_99
                    let base_limbs_100 := base_limbs_86
                    base_size := base_limbs_100
                    let rec b24 (value : MemArray_u64) (n : UInt256) (exponent : Exponent) (one : MemArray_u64) (base : MemArray_u64) (base_size : UInt256) : FeM Option_MemArray_u64 := do
                      let mut value := value
                      let mut n := n
                      let mut exponent := exponent
                      let mut one := one
                      let mut base := base
                      let mut base_size := base_size
                      let base_size_101 := base_size
                      let base_size_102 := base_size_101
                      let v103 := (UInt256.ofNat 0)
                      let v104 := decide (v103.toNat < base_size_102.toNat)
                      let b26 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun value n exponent one base base_size => do
                        let mut value := value
                        let mut n := n
                        let mut exponent := exponent
                        let mut one := one
                        let mut base := base
                        let mut base_size := base_size
                        let base_size_119 := base_size
                        let base_size_120 := base_size_119
                        let v121 := (UInt256.ofNat 0)
                        let v122 := decide (base_size_120 = v121)
                        if v122 then
                          let v123 := (UInt256.ofNat 1)
                          let v124 ← MemArray_try_new_zeroed_u64 v123
                          let _ ← MemArray_drop_u64 base
                          let _ ← MemArray_drop_u64 exponent.limbs
                          let _ ← MemArray_drop_u64 value
                          return v124
                        else
                          let v125 := ()
                          let v126 := v125
                          let v127 := (UInt256.ofNat 0)
                          let v129 ← MemArray_index_ref_u64 value v127
                          let v130 ← mloadInt (.u 64) v129
                          let v131 := (UInt256.ofNat 1)
                          let v132 := v130 &&& v131
                          let v133 := (UInt256.ofNat 1)
                          let v134 := decide (v132 = v133)
                          if v134 then
                            let base_size_136 := base_size
                            let base_size_137 := base_size_136
                            let n_138 := n
                            let n_139 := n_138
                            let (v142, v142_0) ← odd_power exponent base base_size_137 value n_139
                            base := v142_0
                            let _ ← MemArray_drop_u64 base
                            let _ ← MemArray_drop_u64 exponent.limbs
                            let _ ← MemArray_drop_u64 value
                            return v142
                          else
                            let v143 := ()
                            let v144 := v143
                            let base_size_146 := base_size
                            let base_size_147 := base_size_146
                            let n_148 := n
                            let n_149 := n_148
                            let (v152, v152_0) ← even_power exponent base base_size_147 value n_149
                            base := v152_0
                            let _ ← MemArray_drop_u64 base
                            let _ ← MemArray_drop_u64 exponent.limbs
                            let _ ← MemArray_drop_u64 value
                            return v152
                      if v104 then
                        let base_size_105 := base_size
                        let base_size_106 := base_size_105
                        let v107 := (UInt256.ofNat 1)
                        let v108 ← checkedSub base_size_106 v107
                        let v110 ← MemArray_index_ref_u64 base v108
                        let v111 ← mloadInt (.u 64) v110
                        let v112 := (UInt256.ofNat 0)
                        let v113 := decide (v111 = v112)
                        if v113 then
                          let v114 := (UInt256.ofNat 1)
                          let base_size_115 := base_size
                          let v116 ← checkedSub base_size_115 v114
                          let base_size_117 := v116
                          base_size := base_size_117
                          let v118 := ()
                          b24 value n exponent one base base_size
                        else
                          b26 value n exponent one base base_size
                      else
                        b26 value n exponent one base base_size
                    partial_fixpoint
                    b24 value n exponent one base base_size
                  else
                    let v153 := Option_MemArray_u64.None
                    match (← pure v96) with
                    | .Some .. =>
                      let _ ← MemArray_drop_u64 v96.Some_0
                      pure ()
                    | _ => pure ()
                    let _ ← MemArray_drop_u64 exponent.limbs
                    let _ ← MemArray_drop_u64 value
                    return v153
                if v89 then
                  let base_limbs_90 := base_limbs_86
                  b21 value n exponent one base base_size base_limbs_90
                else
                  let n_91 := n
                  let n_92 := n_91
                  let n_93 := n_92
                  b21 value n exponent one base base_size n_93
            else
              let v154 := Option_MemArray_u64.None
              match (← pure v58) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v58.Some_0.limbs
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 value
              return v154
          if v49 then
            let v50 := (UInt256.ofNat 0)
            let v52 ← MemArray_index_ref_u64 value v50
            let v53 ← mloadInt (.u 64) v52
            let v54 := (UInt256.ofNat 1)
            let v55 := decide (v53 = v54)
            if v55 then
              b7 value n exponent one base base_size
            else
              b8 value n exponent one base base_size
          else
            b8 value n exponent one base base_size
      if v27 then
        let n_28 := n
        let n_29 := n_28
        let v30 := (UInt256.ofNat 1)
        let v31 ← checkedSub n_29 v30
        let v33 ← MemArray_index_ref_u64 value v31
        let v34 ← mloadInt (.u 64) v33
        let v35 := (UInt256.ofNat 0)
        let v36 := decide (v34 = v35)
        if v36 then
          let v37 := (UInt256.ofNat 1)
          let n_38 := n
          let v39 ← checkedSub n_38 v37
          let n_40 := v39
          n := n_40
          let v41 := ()
          b3 value n exponent one base base_size
        else
          b5 value n exponent one base base_size
      else
        b5 value n exponent one base base_size
    partial_fixpoint
    b3 value n exponent one base base_size
  else
    let v157 := Option_MemArray_u64.None
    match (← pure v17) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v17.Some_0
      pure ()
    | _ => pure ()
    return v157

def encoded (result : MemArray_u64) (output : MemBuffer) : FeM MemBuffer := do
  let mut output := output
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Range_Known_0_Known_8 := default
  let v4 ← MemBuffer_len output
  let len := v4
  let v7 ← MemBuffer_ptr output
  let bytes := v7
  let v9 := (UInt256.ofNat 0)
  let v10 ← MemArray_len_u64 result
  let v11 := ()
  let v12 := (Range_Known_0_Unknown.mk v11 v10)
  r1 := v12
  let v14 ← Range_start_0 r1
  let v15 := v14
  let rec b1 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) (v1 : UInt256) : FeM MemBuffer := do
    let mut output := output
    let mut r1 := r1
    let mut r2 := r2
    let v17 ← Range_has_0 r1 v1
    if v17 then
      let v19 ← Range_at_0 r1 v1
      let i := v19
      let v21 ← MemArray_index_ref_u64 result i
      let value ← mloadInt (.u 64) v21
      let v23 := (UInt256.ofNat 8)
      let v24 := v23 * i
      let v25 := (UInt256.ofNat 8)
      let v26 := v24 + v25
      let v27 := decide (v26.toNat ≤ len.toNat)
      let rec b7 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) : FeM MemBuffer := do
        let mut output := output
        let mut r1 := r1
        let mut r2 := r2
        let v75 ← Range_next_0 r1 v1
        let v76 := v75
        b1 output r1 r2 v76
      partial_fixpoint
      if v27 then
        let v28 := value
        let v29 := (UInt256.ofNat 8)
        let v30 := v29 * i
        let v31 := len - v30
        let v32 := (UInt256.ofNat 8)
        let v33 := v31 - v32
        let v34 := bytes + v33
        let v35 ← cast_u8_u64 v34
        mstoreInt (.u 64) v35 v28
        let v36 := ()
        let v37 := v36
        b7 output r1 r2
      else
        let v38 := (UInt256.ofNat 0)
        let v39 := (UInt256.ofNat 8)
        let v40 := ()
        let v41 := (Range_Known_0_Known_8.mk v40 v40)
        r2 := v41
        let v43 ← Range_start_0_8 r2
        let v44 := v43
        let rec b8 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) (v2 : UInt256) : FeM MemBuffer := do
          let mut output := output
          let mut r1 := r1
          let mut r2 := r2
          let v46 ← Range_has_0_8 r2 v2
          if v46 then
            let v48 ← Range_at_0_8 r2 v2
            let byte := v48
            let v50 := (UInt256.ofNat 8)
            let v51 := v50 * i
            let v52 := v51 + byte
            let position := v52
            let v54 := decide (position.toNat < len.toNat)
            let b14 : MemBuffer → Range_Known_0_Unknown → Range_Known_0_Known_8 → FeM MemBuffer := fun output r1 r2 => do
              let mut output := output
              let mut r1 := r1
              let mut r2 := r2
              let v70 ← Range_next_0_8 r2 v2
              let v71 := v70
              b8 output r1 r2 v71
            if v54 then
              let v55 := (UInt256.ofNat 8)
              let v56 := v55 * byte
              let v57 ← downcast_truncate_usize_u64 v56
              let shift := v57
              let v59 := value >>> shift
              let v60 ← downcast_truncate_u64_u8 v59
              let v61 := (UInt256.ofNat 1)
              let v62 := len - v61
              let v63 := v62 - position
              let v64 := bytes + v63
              mstoreInt (.u 8) v64 v60
              let v65 := ()
              let v66 := v65
              b14 output r1 r2
            else
              let v67 := ()
              let v68 := v67
              b14 output r1 r2
          else
            let v72 := ()
            let v73 := v72
            b7 output r1 r2
        partial_fixpoint
        b8 output r1 r2 v44
    else
      let output_77 := output
      return output_77
  partial_fixpoint
  b1 output r1 r2 v15

def run_MemBuffer (input : MemBuffer) (gas_1 : UInt256) : FeM Output := do
  let mut result : MemArray_u64 := default
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 32)
  let v4 ← input_word_MemBuffer input v2 v3
  let base_word := v4
  let v6 := (UInt256.ofNat 32)
  let v7 := (UInt256.ofNat 32)
  let v8 ← input_word_MemBuffer input v6 v7
  let exponent_word := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := (UInt256.ofNat 32)
  let v12 ← input_word_MemBuffer input v10 v11
  let modulus_word := v12
  let v14 ← MAX_LENGTH
  let v15 := decide (v14.toNat < base_word.toNat)
  let b1 : MemArray_u64 → FeM Output := fun result => do
    let mut result := result
    let v60 ← FAILURE
    let gas_61 := gas_1
    let v62 ← MemBuffer_empty
    let v63 := (Output.mk v60 gas_61 v62)
    return v63
  if v15 then
    b1 result
  else
    let v16 ← MAX_LENGTH
    let v17 := decide (v16.toNat < exponent_word.toNat)
    if v17 then
      b1 result
    else
      let v18 ← MAX_LENGTH
      let v19 := decide (v18.toNat < modulus_word.toNat)
      if v19 then
        b1 result
      else
        let v20 := ()
        let v21 := v20
        let v22 ← downcast_unchecked_u256_usize base_word
        let base_len := v22
        let v24 ← downcast_unchecked_u256_usize exponent_word
        let exponent_len := v24
        let v26 ← downcast_unchecked_u256_usize modulus_word
        let modulus_len := v26
        let v28 ← gas_cost_MemBuffer input base_len exponent_len modulus_len
        let gas_used := v28
        let gas_30 := gas_1
        let v31 := decide (gas_30.toNat < gas_used.toNat)
        if v31 then
          let v32 ← FAILURE
          let gas_33 := gas_1
          let v34 ← MemBuffer_empty
          let v35 := (Output.mk v32 gas_33 v34)
          return v35
        else
          let v36 := ()
          let v37 := v36
          let v38 := (UInt256.ofNat 0)
          let v39 := decide (modulus_len = v38)
          if v39 then
            let v40 ← SUCCESS
            let v41 ← MemBuffer_empty
            let v42 := (Output.mk v40 gas_used v41)
            return v42
          else
            let v43 := ()
            let v44 := v43
            let v45 ← MemBuffer_try_new modulus_len
            let v46 := v45
            let v47 := v46
            match v47 with
            | .Some .. =>
              let output_48 := v46.Some_0
              let v49 ← power_MemBuffer input base_len exponent_len modulus_len
              let v50 := v49
              let v51 := v50
              match v51 with
              | .Some .. =>
                let result_52 := v50.Some_0
                result := result_52
                let v53 ← SUCCESS
                let output_54 := output_48
                let v56 ← encoded result output_54
                let v57 := (Output.mk v53 gas_used v56)
                let _ ← MemArray_drop_u64 result
                return v57
              | _ =>
                let v58 ← host_allocation gas_used
                match v50 with
                | .Some .. =>
                  let _ ← MemArray_drop_u64 v50.Some_0
                  pure ()
                | _ => pure ()
                let _ ← MemBuffer_drop output_48
                return v58
            | _ =>
              let v59 ← host_allocation gas_used
              match v46 with
              | .Some .. =>
                let _ ← MemBuffer_drop v46.Some_0
                pure ()
              | _ => pure ()
              return v59

def recv0_arm0 : FeM Empty := do
  let mut result : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_6
  r2 := v0
  let v1 := (UInt256.ofNat 18446744073709551615)
  let v3 ← run_MemBuffer r2 v1
  let result_4 := v3
  result := result_4
  let _ ← MemBuffer_drop r2
  let v5 := result.status
  let v6 ← SUCCESS
  let v7 := decide (v5 ≠ v6)
  if v7 then
    let v8 ← MemSlice_empty_u8
    let v11 ← Evm_revert v8
    nomatch v11
  else
    let v12 := ()
    let v13 := v12
    let v16 ← MemBuffer_span_ref result.output
    let v19 ← Evm_return_data v16
    nomatch v19

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


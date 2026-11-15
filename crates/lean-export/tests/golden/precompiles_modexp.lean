export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemArray_as_slice_mut_u64 MemArray_as_slice_ref_u64 MemArray_drop_u64 MemArray_index_mut_u64 MemArray_index_ref_u64 MemArray_len_u64 MemArray_ptr_u64 MemArray_try_new_zeroed_u64 MemArray_u64 MemArray_zero_u64 MemBuffer MemBuffer_drop MemBuffer_empty MemBuffer_len MemBuffer_new MemBuffer_ptr MemBuffer_span_ref MemBuffer_try_new MemSlice_copy_from_u64 MemSlice_empty_u8 MemSlice_head_mut_u64 MemSlice_head_ref_u64 MemSlice_len_u64 MemSlice_ptr_u64 MemSlice_slice_mut_u64 MemSlice_slice_ref_u64 MemSlice_u64 Option_MemArray_u64 Option_MemArray_u64.None Option_MemArray_u64.Some Range_Known_0_Known_5 Range_Known_0_Known_5.mk Range_Known_0_Known_6 Range_Known_0_Known_6.mk Range_Known_0_Known_8 Range_Known_0_Known_8.mk Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_Known_1_Unknown Range_Known_1_Unknown.mk Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start Range_at_0 Range_at_0_5 Range_at_0_6 Range_at_0_8 Range_at_1 Range_has_0 Range_has_0_5 Range_has_0_6 Range_has_0_8 Range_has_1 Range_next_0 Range_next_0_5 Range_next_0_6 Range_next_0_8 Range_next_1 Range_start_0 Range_start_0_5 Range_start_0_6 Range_start_0_8 Range_start_1 assert cast_u8_u64 downcast_truncate_u128_u64 downcast_truncate_u256_u64 downcast_truncate_u256_usize downcast_truncate_u64_u8 downcast_truncate_u64_usize downcast_truncate_usize_u64 downcast_truncate_usize_usize downcast_unchecked_u256_usize)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1 Output Output.mk SUCCESS adc host_allocation sbb)

structure Exponent where
  limbs : MemArray_u64
  top : UInt256
  total : UInt256
deriving Inhabited

inductive Option_Exponent where
  | None
  | Some (f0 : Exponent)
deriving Inhabited

def Option_Exponent.tag : Option_Exponent → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Exponent.Some_0 : Option_Exponent → Exponent
  | .Some f0 => f0
  | _ => default

def Option_Exponent.set_Some_0 (x : Option_Exponent) (v : Exponent) : Option_Exponent :=
  match x with
  | .Some f0 => .Some v
  | x => x

structure Modulus where
  limbs : MemArray_u64
  len : UInt256
  shift : UInt256
  reciprocal : UInt256
deriving Inhabited

inductive Option_Modulus where
  | None
  | Some (f0 : Modulus)
deriving Inhabited

def Option_Modulus.tag : Option_Modulus → UInt256
  | .None => UInt256.ofNat 0
  | .Some .. => UInt256.ofNat 1

def Option_Modulus.Some_0 : Option_Modulus → Modulus
  | .Some f0 => f0
  | _ => default

def Option_Modulus.set_Some_0 (x : Option_Modulus) (v : Modulus) : Option_Modulus :=
  match x with
  | .Some f0 => .Some v
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

def input_word_MemBuffer (v0 : MemBuffer) (v1 : UInt256) (v2 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v5 ← MemBuffer_len_1 v0
  let v6 := v5
  let v7 := v1
  let v8 := decide (v6.toNat ≤ v7.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun r0 r1 v3 => do
    let mut r0 := r0
    let mut r1 := r1
    let v21 := v3
    let v22 := (UInt256.ofNat 0)
    let v23 := decide (v21 = v22)
    if v23 then
      let v24 := (UInt256.ofNat 0)
      return v24
    else
      let v25 := ()
      let v26 := v25
      let v27 := (UInt256.ofNat 0)
      let v28 := v27
      r0 := v28
      let v29 := (UInt256.ofNat 0)
      let v30 := v29
      r1 := v30
      let rec b10 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
        let mut r0 := r0
        let mut r1 := r1
        let v31 := r1
        let v32 := v31
        let v33 := (UInt256.ofNat 8)
        let v34 := v32 + v33
        let v35 := decide (v34.toNat ≤ v21.toNat)
        if v35 then
          let v36 := r0
          let v37 := v36
          let v38 := (UInt256.ofNat 64)
          let v39 := v37 <<< v38
          let v40 := v1
          let v41 := r1
          let v42 := v40 + v41
          let v43 ← MemBuffer_lane_at v0 v42
          let v44 := IntTy.swapBytes (.u 64) v43
          let v45 := v44
          let v46 := v39 ||| v45
          let v47 := v46
          r0 := v47
          let v48 := ()
          let v49 := (UInt256.ofNat 8)
          let v50 := r1
          let v51 := v50 + v49
          let v52 := v51
          r1 := v52
          let v53 := ()
          b10 r0 r1
        else
          let rec b13 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
            let mut r0 := r0
            let mut r1 := r1
            let v54 := r1
            let v55 := v54
            let v56 := decide (v55.toNat < v21.toNat)
            if v56 then
              let v57 := r0
              let v58 := v57
              let v59 := (UInt256.ofNat 8)
              let v60 := v58 <<< v59
              let v61 := v1
              let v62 := r1
              let v63 := v61 + v62
              let v64 ← MemBuffer_byte_at v0 v63
              let v65 := v64
              let v66 := v60 ||| v65
              let v67 := v66
              r0 := v67
              let v68 := ()
              let v69 := (UInt256.ofNat 1)
              let v70 := r1
              let v71 := v70 + v69
              let v72 := v71
              r1 := v72
              let v73 := ()
              b13 r0 r1
            else
              let v74 := r0
              let v75 := v74
              let v76 := v2
              let v77 := v76 - v21
              let v78 := (UInt256.ofNat 8)
              let v79 := v77 * v78
              let v80 := v79
              let v81 := v75 <<< v80
              return v81
          partial_fixpoint
          b13 r0 r1
      partial_fixpoint
      b10 r0 r1
  if v8 then
    let v9 := (UInt256.ofNat 0)
    let v10 := v9
    b3 r0 r1 v10
  else
    let v11 := v1
    let v12 := v6 - v11
    let v13 := v2
    let v14 := decide (v12.toNat < v13.toNat)
    let b6 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun r0 r1 v4 => do
      let mut r0 := r0
      let mut r1 := r1
      let v20 := v4
      b3 r0 r1 v20
    if v14 then
      let v15 := v1
      let v16 := v6 - v15
      let v17 := v16
      b6 r0 r1 v17
    else
      let v18 := v2
      let v19 := v18
      b6 r0 r1 v19

def MAX_LENGTH : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1024)
  return v0

def FAILURE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 1)
  return v0

def bit_length (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 256)
  let v2 := IntTy.leadingZeros (.u 256) v0
  let v3 ← checkedSub v1 v2
  let v4 ← downcast_truncate_u256_u64 v3
  return v4

def gas_cost_MemBuffer (v0 : MemBuffer) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let v11 := v1
  let v12 := v3
  let v13 := decide (v12.toNat < v11.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v4 => do
    let v18 ← downcast_truncate_usize_u64 v4
    let v19 := v18
    let v20 := (UInt256.ofNat 7)
    let v21 ← IntTy.add (.u 64) v19 v20
    let v22 := (UInt256.ofNat 8)
    let v23 ← checkedDiv v21 v22
    let v24 := v23
    let v25 := (UInt256.ofNat 32)
    let v26 := decide (v25.toNat < v19.toNat)
    let b6 : UInt256 → FeM UInt256 := fun v5 => do
      let v33 := v5
      let v34 := (UInt256.ofNat 96)
      let v35 := v1
      let v36 ← checkedAdd v34 v35
      let v37 := v2
      let v38 := (UInt256.ofNat 32)
      let v39 := decide (v37.toNat < v38.toNat)
      let b9 : UInt256 → FeM UInt256 := fun v6 => do
        let v44 ← input_word_MemBuffer v0 v36 v6
        let v45 := v44
        let v46 ← bit_length v45
        let v47 := v46
        let v48 := (UInt256.ofNat 0)
        let v49 := decide (v48.toNat < v47.toNat)
        let b12 : UInt256 → FeM UInt256 := fun v7 => do
          let v55 := v7
          let v56 := v2
          let v57 := (UInt256.ofNat 32)
          let v58 := decide (v57.toNat < v56.toNat)
          let b15 : UInt256 → FeM UInt256 := fun v8 => do
            let v66 := v8
            let v67 := (UInt256.ofNat 16)
            let v68 ← IntTy.mul (.u 64) v67 v66
            let v69 ← IntTy.add (.u 64) v68 v55
            let v70 := v69
            let v71 := (UInt256.ofNat 0)
            let v72 := decide (v70 = v71)
            let b18 : UInt256 → FeM UInt256 := fun v9 => do
              let v76 ← IntTy.mul (.u 64) v33 v9
              let v77 := v76
              let v78 := (UInt256.ofNat 500)
              let v79 := decide (v77.toNat < v78.toNat)
              let b21 : UInt256 → FeM UInt256 := fun v10 => do
                return v10
              if v79 then
                let v80 := (UInt256.ofNat 500)
                let v81 := v80
                b21 v81
              else
                let v82 := v77
                b21 v82
            if v72 then
              let v73 := (UInt256.ofNat 1)
              let v74 := v73
              b18 v74
            else
              let v75 := v70
              b18 v75
          if v58 then
            let v59 := v2
            let v60 := (UInt256.ofNat 32)
            let v61 ← checkedSub v59 v60
            let v62 ← downcast_truncate_usize_u64 v61
            let v63 := v62
            b15 v63
          else
            let v64 := (UInt256.ofNat 0)
            let v65 := v64
            b15 v65
        if v49 then
          let v50 := (UInt256.ofNat 1)
          let v51 ← IntTy.sub (.u 64) v47 v50
          let v52 := v51
          b12 v52
        else
          let v53 := (UInt256.ofNat 0)
          let v54 := v53
          b12 v54
      if v39 then
        let v40 := v2
        let v41 := v40
        b9 v41
      else
        let v42 := (UInt256.ofNat 32)
        let v43 := v42
        b9 v43
    if v26 then
      let v27 := (UInt256.ofNat 2)
      let v28 ← IntTy.mul (.u 64) v27 v24
      let v29 ← IntTy.mul (.u 64) v28 v24
      let v30 := v29
      b6 v30
    else
      let v31 := (UInt256.ofNat 16)
      let v32 := v31
      b6 v32
  if v13 then
    let v14 := v1
    let v15 := v14
    b3 v15
  else
    let v16 := v3
    let v17 := v16
    b3 v17

def input_byte_MemBuffer (v0 : MemBuffer) (v1 : UInt256) : FeM UInt256 := do
  let v3 := v1
  let v4 ← MemBuffer_len_1 v0
  let v5 := decide (v3.toNat < v4.toNat)
  let b3 : UInt256 → FeM UInt256 := fun v2 => do
    return v2
  if v5 then
    let v6 ← MemBuffer_byte_at v0 v1
    let v7 := v6
    b3 v7
  else
    let v8 := (UInt256.ofNat 0)
    let v9 := v8
    b3 v9

def input_limb_MemBuffer (v0 : MemBuffer) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v4 := v1
  let v5 := v2
  let v6 := v4 + v5
  let v7 := v3
  let v8 := (UInt256.ofNat 8)
  let v9 := v7 * v8
  let v10 := v6 - v9
  let v11 := v10
  let v12 := v3
  let v13 := (UInt256.ofNat 8)
  let v14 := v12 * v13
  let v15 := (UInt256.ofNat 8)
  let v16 := v14 + v15
  let v17 := v2
  let v18 := decide (v16.toNat ≤ v17.toNat)
  let b2 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
    let mut r0 := r0
    let mut r1 := r1
    let v25 := ()
    let v26 := v25
    let v27 := (UInt256.ofNat 0)
    let v28 := v27
    r0 := v28
    let v29 := (UInt256.ofNat 8)
    let v30 := v29
    r1 := v30
    let rec b5 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
      let mut r0 := r0
      let mut r1 := r1
      let v31 := r1
      let v32 := v31
      let v33 := (UInt256.ofNat 0)
      let v34 := decide (v33.toNat < v32.toNat)
      if v34 then
        let v35 := (UInt256.ofNat 1)
        let v36 := r1
        let v37 := v36 - v35
        let v38 := v37
        r1 := v38
        let v39 := ()
        let v40 := v3
        let v41 := (UInt256.ofNat 8)
        let v42 := v40 * v41
        let v43 := r1
        let v44 := v42 + v43
        let v45 := v44
        let v46 := r0
        let v47 := v46
        let v48 := (UInt256.ofNat 8)
        let v49 := IntTy.shl (.u 64) v47 v48
        let v50 := v49
        r0 := v50
        let v51 := ()
        let v52 := v2
        let v53 := decide (v45.toNat < v52.toNat)
        let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
          let mut r0 := r0
          let mut r1 := r1
          b5 r0 r1
        if v53 then
          let v54 := v1
          let v55 := v2
          let v56 := v54 + v55
          let v57 := (UInt256.ofNat 1)
          let v58 := v56 - v57
          let v59 := v58 - v45
          let v60 ← input_byte_MemBuffer v0 v59
          let v61 := v60
          let v62 := r0
          let v63 := v62 ||| v61
          let v64 := v63
          r0 := v64
          let v65 := ()
          let v66 := v65
          b10 r0 r1
        else
          let v67 := ()
          let v68 := v67
          b10 r0 r1
      else
        let v69 := r0
        return v69
    partial_fixpoint
    b5 r0 r1
  if v18 then
    let v19 ← MemBuffer_len_1 v0
    let v20 := decide (v11.toNat ≤ v19.toNat)
    if v20 then
      let v21 := (UInt256.ofNat 8)
      let v22 := v11 - v21
      let v23 ← MemBuffer_lane_at v0 v22
      let v24 := IntTy.swapBytes (.u 64) v23
      return v24
    else
      b2 r0 r1
  else
    b2 r0 r1

def input_limbs_MemBuffer (v0 : MemBuffer) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM Option_MemArray_u64 := do
  let mut r0 : MemArray_u64 := default
  let mut r1 : Range_Known_0_Unknown := default
  let v6 ← MemArray_try_new_zeroed_u64 v3
  let v7 := match v6 with | .Some .. => true | _ => false
  if v7 then
    let v8 := v6.Some_0
    let v9 := v8
    r0 := v9
    let v10 := v2
    let v11 := (UInt256.ofNat 7)
    let v12 ← checkedAdd v10 v11
    let v13 := (UInt256.ofNat 8)
    let v14 ← checkedDiv v12 v13
    let v15 := v14
    let v16 := v3
    let v17 := decide (v15.toNat < v16.toNat)
    let b5 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 v4 => do
      let mut r0 := r0
      let mut r1 := r1
      let v21 := v4
      let v22 := (UInt256.ofNat 0)
      let v23 := ()
      let v24 := (Range_Known_0_Unknown.mk v23 v21)
      r1 := v24
      let v26 ← Range_start_0 r1
      let v27 := v26
      let rec b6 (r0 : MemArray_u64) (r1 : Range_Known_0_Unknown) (v5 : UInt256) : FeM Option_MemArray_u64 := do
        let mut r0 := r0
        let mut r1 := r1
        let v29 ← Range_has_0 r1 v5
        if v29 then
          let v31 ← Range_at_0 r1 v5
          let v32 := v31
          let v33 ← input_limb_MemBuffer v0 v1 v2 v32
          let (v35, v35_k) ← MemArray_index_mut_u64 r0 v32
          mstoreInt (.u 64) v35 v33
          let v35_0 ← v35_k v35
          r0 := v35_0
          let v36 := ()
          let v38 ← Range_next_0 r1 v5
          let v39 := v38
          b6 r0 r1 v39
        else
          let v40 := r0
          let v41 := v40
          let v42 := Option_MemArray_u64.Some v41
          return v42
      partial_fixpoint
      b6 r0 r1 v27
    if v17 then
      let v18 := v15
      b5 r0 r1 v18
    else
      let v19 := v3
      let v20 := v19
      b5 r0 r1 v20
  else
    let v43 := Option_MemArray_u64.None
    match (← pure v6) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v6.Some_0
      pure ()
    | _ => pure ()
    return v43

def Exponent_try_read_MemBuffer (v0 : MemBuffer) (v1 : UInt256) (v2 : UInt256) : FeM Option_Exponent := do
  let mut r0 : MemArray_u64 := default
  let mut r1 : UInt256 := default
  let v4 := v2
  let v5 := (UInt256.ofNat 8)
  let v6 ← checkedMul v4 v5
  let v7 := v6
  let v8 := v2
  let v9 := (UInt256.ofNat 7)
  let v10 ← checkedAdd v8 v9
  let v11 := (UInt256.ofNat 8)
  let v12 ← checkedDiv v10 v11
  let v13 ← input_limbs_MemBuffer v0 v1 v2 v12
  let v14 := match v13 with | .Some .. => true | _ => false
  if v14 then
    let v15 := v13.Some_0
    let v16 := v15
    r0 := v16
    let v18 ← MemArray_len_u64 r0
    let v19 := v18
    r1 := v19
    let rec b3 (r0 : MemArray_u64) (r1 : UInt256) : FeM Option_Exponent := do
      let mut r0 := r0
      let mut r1 := r1
      let v20 := r1
      let v21 := v20
      let v22 := (UInt256.ofNat 0)
      let v23 := decide (v22.toNat < v21.toNat)
      let b5 : MemArray_u64 → UInt256 → FeM Option_Exponent := fun r0 r1 => do
        let mut r0 := r0
        let mut r1 := r1
        let v38 := r1
        let v39 := v38
        let v40 := (UInt256.ofNat 0)
        let v41 := decide (v39 = v40)
        let b9 : MemArray_u64 → UInt256 → UInt256 → FeM Option_Exponent := fun r0 r1 v3 => do
          let mut r0 := r0
          let mut r1 := r1
          let v67 := v3
          let v68 := r0
          let v69 := v68
          let v70 := (Exponent.mk v69 v67 v7)
          let v71 := Option_Exponent.Some v70
          return v71
        if v41 then
          let v42 := v7
          b9 r0 r1 v42
        else
          let v43 := (UInt256.ofNat 255)
          let v44 := r1
          let v45 := v44
          let v46 := (UInt256.ofNat 1)
          let v47 ← checkedSub v45 v46
          let v49 ← MemArray_index_ref_u64 r0 v47
          let v50 ← mloadInt (.u 64) v49
          let v51 := v50
          let v52 := IntTy.leadingZeros (.u 256) v51
          let v53 ← checkedSub v43 v52
          let v54 ← downcast_truncate_u256_usize v53
          let v55 := v54
          let v56 := (UInt256.ofNat 1)
          let v57 ← checkedSub v7 v56
          let v58 := r1
          let v59 := v58
          let v60 := (UInt256.ofNat 1)
          let v61 ← checkedSub v59 v60
          let v62 := (UInt256.ofNat 64)
          let v63 ← checkedMul v61 v62
          let v64 ← checkedAdd v63 v55
          let v65 ← checkedSub v57 v64
          let v66 := v65
          b9 r0 r1 v66
      if v23 then
        let v24 := r1
        let v25 := v24
        let v26 := (UInt256.ofNat 1)
        let v27 ← checkedSub v25 v26
        let v29 ← MemArray_index_ref_u64 r0 v27
        let v30 ← mloadInt (.u 64) v29
        let v31 := (UInt256.ofNat 0)
        let v32 := decide (v30 = v31)
        if v32 then
          let v33 := (UInt256.ofNat 1)
          let v34 := r1
          let v35 ← checkedSub v34 v33
          let v36 := v35
          r1 := v36
          let v37 := ()
          b3 r0 r1
        else
          b5 r0 r1
      else
        b5 r0 r1
    partial_fixpoint
    b3 r0 r1
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

def reciprocal_2by1 (v0 : UInt256) : FeM UInt256 := do
  let v1 := (UInt256.ofNat 340282366920938463463374607431768211455)
  let v2 := v1
  let v3 := v0
  let v4 := v3
  let v5 ← checkedDiv v2 v4
  let v6 ← LIMB_BASE
  let v7 ← IntTy.sub (.u 128) v5 v6
  let v8 ← downcast_truncate_u128_u64 v7
  return v8

def reciprocal_3by2 (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v3 ← reciprocal_2by1 v0
  let v4 := v3
  r0 := v4
  let v5 := v0
  let v6 := r0
  let v7 := IntTy.mulW (.u 64) v5 v6
  let v8 := v1
  let v9 := IntTy.addW (.u 64) v7 v8
  let v10 := v9
  r1 := v10
  let v11 := r1
  let v12 := v11
  let v13 := v1
  let v14 := decide (v12.toNat < v13.toNat)
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
    let mut r0 := r0
    let mut r1 := r1
    let v45 := r0
    let v46 := v45
    let v47 := v1
    let v48 := v47
    let v49 := IntTy.mulW (.u 128) v46 v48
    let v50 := v49
    let v51 := (UInt256.ofNat 64)
    let v52 := v50 >>> v51
    let v53 ← downcast_truncate_u128_u64 v52
    let v54 := v53
    let v55 ← downcast_truncate_u128_u64 v50
    let v56 := v55
    let v57 := r1
    let v58 := IntTy.addW (.u 64) v57 v54
    let v59 := v58
    r1 := v59
    let v60 := ()
    let v61 := r1
    let v62 := v61
    let v63 := decide (v62.toNat < v54.toNat)
    let b9 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
      let mut r0 := r0
      let mut r1 := r1
      let v90 := r0
      return v90
    if v63 then
      let v64 := (UInt256.ofNat 1)
      let v65 := r0
      let v66 := IntTy.subW (.u 64) v65 v64
      let v67 := v66
      r0 := v67
      let v68 := ()
      let v69 := r1
      let v70 := v69
      let v71 := v0
      let v72 := decide (v71.toNat < v70.toNat)
      let b12 : UInt256 → UInt256 → Unit → FeM UInt256 := fun r0 r1 v2 => do
        let mut r0 := r0
        let mut r1 := r1
        let v87 := v2
        b9 r0 r1
      let b10 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
        let mut r0 := r0
        let mut r1 := r1
        let v81 := (UInt256.ofNat 1)
        let v82 := r0
        let v83 := IntTy.subW (.u 64) v82 v81
        let v84 := v83
        r0 := v84
        let v85 := ()
        let v86 := v85
        b12 r0 r1 v86
      if v72 then
        b10 r0 r1
      else
        let v73 := r1
        let v74 := v73
        let v75 := v0
        let v76 := decide (v74 = v75)
        let b11 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
          let mut r0 := r0
          let mut r1 := r1
          let v79 := ()
          let v80 := v79
          b12 r0 r1 v80
        if v76 then
          let v77 := v1
          let v78 := decide (v77.toNat ≤ v56.toNat)
          if v78 then
            b10 r0 r1
          else
            b11 r0 r1
        else
          b11 r0 r1
    else
      let v88 := ()
      let v89 := v88
      b9 r0 r1
  if v14 then
    let v15 := (UInt256.ofNat 1)
    let v16 := r0
    let v17 := IntTy.subW (.u 64) v16 v15
    let v18 := v17
    r0 := v18
    let v19 := ()
    let v20 := r1
    let v21 := v20
    let v22 := v0
    let v23 := decide (v22.toNat ≤ v21.toNat)
    let b6 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
      let mut r0 := r0
      let mut r1 := r1
      let v37 := v0
      let v38 := r1
      let v39 := IntTy.subW (.u 64) v38 v37
      let v40 := v39
      r1 := v40
      let v41 := ()
      let v42 := v41
      b3 r0 r1
    if v23 then
      let v24 := (UInt256.ofNat 1)
      let v25 := r0
      let v26 := IntTy.subW (.u 64) v25 v24
      let v27 := v26
      r0 := v27
      let v28 := ()
      let v29 := v0
      let v30 := r1
      let v31 := IntTy.subW (.u 64) v30 v29
      let v32 := v31
      r1 := v32
      let v33 := ()
      let v34 := v33
      b6 r0 r1
    else
      let v35 := ()
      let v36 := v35
      b6 r0 r1
  else
    let v43 := ()
    let v44 := v43
    b3 r0 r1

def Modulus_try_new (v0 : MemArray_u64) (v1 : UInt256) : FeM Option_Modulus := do
  let mut r0 : MemArray_u64 := default
  let mut r1 : Range_Known_0_Unknown := default
  let v6 ← downcast_truncate_u64_usize v1
  let v7 := v6
  let v8 := v1
  let v9 := (UInt256.ofNat 0)
  let v10 := decide (v9.toNat < v8.toNat)
  let b3 : MemArray_u64 → Range_Known_0_Unknown → Bool → FeM Option_Modulus := fun r0 r1 v2 => do
    let mut r0 := r0
    let mut r1 := r1
    let v25 ← assert v2
    let v26 := v1
    let v27 := (UInt256.ofNat 1)
    let v28 ← IntTy.sub (.u 64) v26 v27
    let v29 ← downcast_truncate_u64_usize v28
    let v30 ← MemArray_index_ref_u64 v0 v29
    let v31 ← mloadInt (.u 64) v30
    let v32 := v31
    let v33 := IntTy.leadingZeros (.u 256) v32
    let v34 := (UInt256.ofNat 192)
    let v35 ← checkedSub v33 v34
    let v36 ← downcast_truncate_u256_u64 v35
    let v37 := v36
    let v38 ← downcast_truncate_u64_usize v1
    let v39 ← MemArray_try_new_zeroed_u64 v38
    let v40 := match v39 with | .Some .. => true | _ => false
    if v40 then
      let v41 := v39.Some_0
      let v42 := v41
      r0 := v42
      let v43 := (UInt256.ofNat 0)
      let v44 ← downcast_truncate_u64_usize v1
      let v45 := ()
      let v46 := (Range_Known_0_Unknown.mk v45 v44)
      r1 := v46
      let v48 ← Range_start_0 r1
      let v49 := v48
      let rec b8 (r0 : MemArray_u64) (r1 : Range_Known_0_Unknown) (v3 : UInt256) : FeM Option_Modulus := do
        let mut r0 := r0
        let mut r1 := r1
        let v51 ← Range_has_0 r1 v3
        if v51 then
          let v53 ← Range_at_0 r1 v3
          let v54 := v53
          let v55 ← MemArray_index_ref_u64 v0 v54
          let v56 ← mloadInt (.u 64) v55
          let v57 := IntTy.shl (.u 64) v56 v37
          let v58 := v57
          let v59 := (UInt256.ofNat 0)
          let v60 := decide (v37 = v59)
          let b14 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_Modulus := fun r0 r1 v4 => do
            let mut r0 := r0
            let mut r1 := r1
            let v73 := v4
            let v74 := v58 ||| v73
            let (v76, v76_k) ← MemArray_index_mut_u64 r0 v54
            mstoreInt (.u 64) v76 v74
            let v76_0 ← v76_k v76
            r0 := v76_0
            let v77 := ()
            let v79 ← Range_next_0 r1 v3
            let v80 := v79
            b8 r0 r1 v80
          let b12 : MemArray_u64 → Range_Known_0_Unknown → FeM Option_Modulus := fun r0 r1 => do
            let mut r0 := r0
            let mut r1 := r1
            let v71 := (UInt256.ofNat 0)
            let v72 := v71
            b14 r0 r1 v72
          if v60 then
            b12 r0 r1
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := decide (v54 = v61)
            if v62 then
              b12 r0 r1
            else
              let v63 := (UInt256.ofNat 1)
              let v64 ← checkedSub v54 v63
              let v65 ← MemArray_index_ref_u64 v0 v64
              let v66 ← mloadInt (.u 64) v65
              let v67 := (UInt256.ofNat 64)
              let v68 ← IntTy.sub (.u 64) v67 v37
              let v69 := v66 >>> v68
              let v70 := v69
              b14 r0 r1 v70
        else
          let v81 := (UInt256.ofNat 1)
          let v82 := decide (v7 = v81)
          let b18 : MemArray_u64 → Range_Known_0_Unknown → UInt256 → FeM Option_Modulus := fun r0 r1 v5 => do
            let mut r0 := r0
            let mut r1 := r1
            let v101 := v5
            let v102 := r0
            let v103 := v102
            let v104 := v1
            let v105 := (Modulus.mk v103 v104 v37 v101)
            let v106 := Option_Modulus.Some v105
            return v106
          if v82 then
            let v83 := (UInt256.ofNat 0)
            let v85 ← MemArray_index_ref_u64 r0 v83
            let v86 ← mloadInt (.u 64) v85
            let v87 ← reciprocal_2by1 v86
            let v88 := v87
            b18 r0 r1 v88
          else
            let v89 := (UInt256.ofNat 1)
            let v90 ← checkedSub v7 v89
            let v92 ← MemArray_index_ref_u64 r0 v90
            let v93 := (UInt256.ofNat 2)
            let v94 ← checkedSub v7 v93
            let v96 ← MemArray_index_ref_u64 r0 v94
            let v97 ← mloadInt (.u 64) v92
            let v98 ← mloadInt (.u 64) v96
            let v99 ← reciprocal_3by2 v97 v98
            let v100 := v99
            b18 r0 r1 v100
      partial_fixpoint
      b8 r0 r1 v49
    else
      let v107 := Option_Modulus.None
      match (← pure v39) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v39.Some_0
        pure ()
      | _ => pure ()
      return v107
  let b2 : MemArray_u64 → Range_Known_0_Unknown → FeM Option_Modulus := fun r0 r1 => do
    let mut r0 := r0
    let mut r1 := r1
    let v23 := false
    let v24 := v23
    b3 r0 r1 v24
  if v10 then
    let v11 ← MemArray_len_u64 v0
    let v12 := decide (v7.toNat ≤ v11.toNat)
    if v12 then
      let v13 := v1
      let v14 := (UInt256.ofNat 1)
      let v15 ← IntTy.sub (.u 64) v13 v14
      let v16 ← downcast_truncate_u64_usize v15
      let v17 ← MemArray_index_ref_u64 v0 v16
      let v18 ← mloadInt (.u 64) v17
      let v19 := (UInt256.ofNat 0)
      let v20 := decide (v18 ≠ v19)
      if v20 then
        let v21 := true
        let v22 := v21
        b3 r0 r1 v22
      else
        b2 r0 r1
    else
      b2 r0 r1
  else
    b2 r0 r1

def Wide_of (v0 : MemArray_u64) (v1 : UInt256) : FeM Wide := do
  let v5 := (UInt256.ofNat 0)
  let v6 ← MemArray_index_ref_u64 v0 v5
  let v7 ← mloadInt (.u 64) v6
  let v8 := v1
  let v9 := (UInt256.ofNat 1)
  let v10 := decide (v9.toNat < v8.toNat)
  let b3 : UInt256 → FeM Wide := fun v2 => do
    let v17 := v1
    let v18 := (UInt256.ofNat 2)
    let v19 := decide (v18.toNat < v17.toNat)
    let b6 : UInt256 → FeM Wide := fun v3 => do
      let v26 := v1
      let v27 := (UInt256.ofNat 3)
      let v28 := decide (v27.toNat < v26.toNat)
      let b9 : UInt256 → FeM Wide := fun v4 => do
        let v35 := (Wide.mk v7 v2 v3 v4)
        return v35
      if v28 then
        let v29 := (UInt256.ofNat 3)
        let v30 ← MemArray_index_ref_u64 v0 v29
        let v31 ← mloadInt (.u 64) v30
        let v32 := v31
        b9 v32
      else
        let v33 := (UInt256.ofNat 0)
        let v34 := v33
        b9 v34
    if v19 then
      let v20 := (UInt256.ofNat 2)
      let v21 ← MemArray_index_ref_u64 v0 v20
      let v22 ← mloadInt (.u 64) v21
      let v23 := v22
      b6 v23
    else
      let v24 := (UInt256.ofNat 0)
      let v25 := v24
      b6 v25
  if v10 then
    let v11 := (UInt256.ofNat 1)
    let v12 ← MemArray_index_ref_u64 v0 v11
    let v13 ← mloadInt (.u 64) v12
    let v14 := v13
    b3 v14
  else
    let v15 := (UInt256.ofNat 0)
    let v16 := v15
    b3 v16

def window_width (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let v1 := (UInt256.ofNat 1)
  let v2 := v1
  r0 := v2
  let v3 := v0
  let v4 := (UInt256.ofNat 2)
  let v5 ← checkedDiv v3 v4
  let v6 := v5
  r1 := v6
  let v7 := (UInt256.ofNat 2)
  let v8 := v7
  r2 := v8
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v9 := r2
    let v10 := v9
    let v11 := (UInt256.ofNat 7)
    let v12 := decide (v10.toNat ≤ v11.toNat)
    if v12 then
      let v13 := (UInt256.ofNat 1)
      let v14 := r2
      let v15 := v14
      let v16 := (UInt256.ofNat 1)
      let v17 ← checkedSub v15 v16
      let v18 := v13 <<< v17
      let v19 := v0
      let v20 := r2
      let v21 := v20
      let v22 := (UInt256.ofNat 1)
      let v23 ← checkedAdd v21 v22
      let v24 ← checkedDiv v19 v23
      let v25 ← checkedAdd v18 v24
      let v26 := v25
      let v27 := r1
      let v28 := decide (v26.toNat < v27.toNat)
      let b6 : UInt256 → UInt256 → UInt256 → FeM UInt256 := fun r0 r1 r2 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let v38 := (UInt256.ofNat 1)
        let v39 := r2
        let v40 ← checkedAdd v39 v38
        let v41 := v40
        r2 := v41
        let v42 := ()
        b1 r0 r1 r2
      if v28 then
        let v29 := r2
        let v30 := v29
        let v31 := v30
        r0 := v31
        let v32 := ()
        let v33 := v26
        r1 := v33
        let v34 := ()
        let v35 := v34
        b6 r0 r1 r2
      else
        let v36 := ()
        let v37 := v36
        b6 r0 r1 r2
    else
      let v43 := r0
      return v43
  partial_fixpoint
  b1 r0 r1 r2

def Small_double_1_k2 (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v30 : UInt256) (v35 : UInt256) (v41 : UInt256) (v47 : UInt256) (v49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb v30 v50 v49
  let v52 := v51.fst
  let v53 := v52
  let v54 := v51.snd
  let v55 := v54
  let v56 := v1.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 v55
  let v61 := v60.snd
  let v62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (v62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk v6 v14 v22 v30)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk v35 v41 v47 v53)
    let v68 := v67
    b3 v68

def Small_double_1_k1 (v0 : Small_1) (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let v30 := v29
  let v31 := v0.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb v6 v31 v32
  let v34 := v33.fst
  let v35 := v34
  let v36 := v33.snd
  let v37 := v36
  let v38 := v0.m.l1
  let v39 ← sbb v14 v38 v37
  let v40 := v39.fst
  let v41 := v40
  let v42 := v39.snd
  let v43 := v42
  let v44 := v0.m.l2
  let v45 ← sbb v22 v44 v43
  let v46 := v45.fst
  let v47 := v46
  let v48 := v45.snd
  let v49 := v48
  let v50 := v0.m.l3
  Small_double_1_k2 v1 v6 v14 v22 v30 v35 v41 v47 v49 v50

def Small_double_1 (v0 : Small_1) (v1 : Wide) : FeM Wide := do
  let v3 := v1.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let v6 := v5
  let v7 := v1.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := v1.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let v14 := v13
  let v15 := v1.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := v1.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let v22 := v21
  let v23 := v1.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := v1.l2
  Small_double_1_k1 v0 v1 v6 v14 v22 v25 v26

def mac (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM (UInt256 × UInt256) := do
  let v4 := v0
  let v5 := v4
  let v6 := v1
  let v7 := v6
  let v8 ← IntTy.mul (.u 128) v5 v7
  let v9 := v2
  let v10 := v9
  let v11 ← IntTy.add (.u 128) v8 v10
  let v12 := v3
  let v13 := v12
  let v14 ← IntTy.add (.u 128) v11 v13
  let v15 := v14
  let v16 ← downcast_truncate_u128_u64 v15
  let v17 := (UInt256.ofNat 64)
  let v18 := v15 >>> v17
  let v19 ← downcast_truncate_u128_u64 v18
  let v20 := (v16, v19)
  return v20

def Small_row2_1_k1 (v0 : Small_1) (v10 : UInt256) (v17 : UInt256) (v23 : UInt256) (v25 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 := v0.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac v28 v29 v10 v30
  let v32 := v31.snd
  let v33 := v32
  let v34 := v0.m.l1
  let v35 ← mac v28 v34 v17 v33
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc v23 v40 v39
  let v42 := v41.fst
  let v43 := v42
  let v44 := v41.snd
  let v45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk v37 v43 v46 v47)
  let v49 := IntTy.addW (.u 64) v25 v45
  let v50 := (v48, v49)
  return v50

def Small_row2_1 (v0 : Small_1) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc v4 v19 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  let v26 := v0.inverse
  let v27 := IntTy.mulW (.u 64) v10 v26
  let v28 := v27
  Small_row2_1_k1 v0 v10 v17 v23 v25 v28

def Small_row3_1_k2 (v30 : UInt256) (v32 : UInt256) (v44 : UInt256) (v50 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc v30 v53 v52
  let v55 := v54.fst
  let v56 := v55
  let v57 := v54.snd
  let v58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk v44 v50 v56 v59)
  let v61 := IntTy.addW (.u 64) v32 v58
  let v62 := (v60, v61)
  return v62

def Small_row3_1_k1 (v0 : Small_1) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let v30 := v29
  let v31 := v28.snd
  let v32 := v31
  let v33 := v0.inverse
  let v34 := IntTy.mulW (.u 64) v10 v33
  let v35 := v34
  let v36 := v0.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac v35 v36 v10 v37
  let v39 := v38.snd
  let v40 := v39
  let v41 := v0.m.l1
  let v42 ← mac v35 v41 v17 v40
  let v43 := v42.fst
  let v44 := v43
  let v45 := v42.snd
  let v46 := v45
  let v47 := v0.m.l2
  let v48 ← mac v35 v47 v24 v46
  let v49 := v48.fst
  let v50 := v49
  let v51 := v48.snd
  let v52 := v51
  Small_row3_1_k2 v30 v32 v44 v50 v52

def Small_row3_1 (v0 : Small_1) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc v4 v26 v27
  Small_row3_1_k1 v0 v10 v17 v24 v28

def Small_row4_1_k2 (v0 : Small_1) (v24 : UInt256) (v31 : UInt256) (v37 : UInt256) (v39 : UInt256) (v42 : UInt256) (v51 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := v52
  let v54 := v0.m.l2
  let v55 ← mac v42 v54 v24 v53
  let v56 := v55.fst
  let v57 := v56
  let v58 := v55.snd
  let v59 := v58
  let v60 := v0.m.l3
  let v61 ← mac v42 v60 v31 v59
  let v62 := v61.fst
  let v63 := v62
  let v64 := v61.snd
  let v65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc v37 v66 v65
  let v68 := v67.fst
  let v69 := v68
  let v70 := v67.snd
  let v71 := v70
  let v72 := (Wide.mk v51 v57 v63 v69)
  let v73 := IntTy.addW (.u 64) v39 v71
  let v74 := (v72, v73)
  return v74

def Small_row4_1_k1 (v0 : Small_1) (v2 : UInt256) (v4 : UInt256) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 v2 v28 v26
  let v30 := v29.fst
  let v31 := v30
  let v32 := v29.snd
  let v33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc v4 v33 v34
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := v0.inverse
  let v41 := IntTy.mulW (.u 64) v10 v40
  let v42 := v41
  let v43 := v0.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac v42 v43 v10 v44
  let v46 := v45.snd
  let v47 := v46
  let v48 := v0.m.l1
  let v49 ← mac v42 v48 v17 v47
  let v50 := v49.fst
  let v51 := v50
  let v52 := v49.snd
  Small_row4_1_k2 v0 v24 v31 v37 v39 v42 v51 v52

def Small_row4_1 (v0 : Small_1) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := v1.l3
  let v28 := v3.l3
  Small_row4_1_k1 v0 v2 v4 v10 v17 v24 v26 v27 v28

def Small_mul_1 (v0 : Small_1) (v1 : Wide) (v2 : Wide) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Wide := default
  let mut r2 : Wide := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let mut r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let v13 := v12
  r0 := v13
  let v14 := (UInt256.ofNat 1)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := v1.l0
    let v18 := v17
    let v19 := v2.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let v22 := v21
    let v23 ← downcast_truncate_u128_u64 v22
    let v24 := v23
    let v25 := v0.inverse
    let v26 := IntTy.mulW (.u 64) v24 v25
    let v27 := v26
    let v28 := v27
    let v29 := v0.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let v32 := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := v22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := v32 >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (v24 ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v3 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let v45 := v44
      let v46 := v0.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ v45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let v54 := v4
        let v55 ← downcast_truncate_u128_u64 v54
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := v0.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) v45 v50
        let v52 := v51
        b9 r0 r1 r2 r3 r4 r5 r6 v52
      else
        let v53 := v45
        b9 r0 r1 r2 r3 r4 r5 r6 v53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 r0 r1 r2 r3 r4 r5 r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 r0 r1 r2 r3 r4 r5 r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 1)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := v2.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_1 v0 v1 v65 r0 v66
      let v69 := v68.fst
      let v70 := v69
      r1 := v70
      let v71 := v68.snd
      let v72 := v71
      let v73 := v2.l1
      let v75 ← Small_row2_1 v0 v1 v73 r1 v72
      let v76 := v75.fst
      let v77 := v76
      let v78 := v75.snd
      let v79 := v78
      let v80 := v77.l0
      let v81 := v0.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let v85 := v84
      let v86 := v83.snd
      let v87 := v86
      let v88 := v77.l1
      let v89 := v0.m.l1
      let v90 ← sbb v88 v89 v87
      let v91 := v90.fst
      let v92 := v91
      let v93 := v90.snd
      let v94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb v79 v95 v94
      let v97 := v96.snd
      let v98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (v98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        return v5
      if v100 then
        let v101 := v77
        b15 r0 r1 r2 r3 r4 r5 r6 v101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk v85 v92 v102 v103)
        let v105 := v104
        b15 r0 r1 r2 r3 r4 r5 r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 1)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := v2.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_1 v0 v1 v111 r0 v112
        let v115 := v114.fst
        let v116 := v115
        r2 := v116
        let v117 := v114.snd
        let v118 := v117
        let v119 := v2.l1
        let v121 ← Small_row3_1 v0 v1 v119 r2 v118
        let v122 := v121.fst
        let v123 := v122
        r3 := v123
        let v124 := v121.snd
        let v125 := v124
        let v126 := v2.l2
        let v128 ← Small_row3_1 v0 v1 v126 r3 v125
        let v129 := v128.fst
        let v130 := v129
        let v131 := v128.snd
        let v132 := v131
        let v133 := v130.l0
        let v134 := v0.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let v138 := v137
        let v139 := v136.snd
        let v140 := v139
        let v141 := v130.l1
        let v142 := v0.m.l1
        let v143 ← sbb v141 v142 v140
        let v144 := v143.fst
        let v145 := v144
        let v146 := v143.snd
        let v147 := v146
        let v148 := v130.l2
        let v149 := v0.m.l2
        let v150 ← sbb v148 v149 v147
        let v151 := v150.fst
        let v152 := v151
        let v153 := v150.snd
        let v154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb v132 v155 v154
        let v157 := v156.snd
        let v158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (v158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v6
        if v160 then
          let v161 := v130
          b21 r0 r1 r2 r3 r4 r5 r6 v161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk v138 v145 v152 v162)
          let v164 := v163
          b21 r0 r1 r2 r3 r4 r5 r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := v2.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_1 v0 v1 v167 r0 v168
        let v171 := v170.fst
        let v172 := v171
        r4 := v172
        let v173 := v170.snd
        let v174 := v173
        let v175 := v2.l1
        let v177 ← Small_row4_1 v0 v1 v175 r4 v174
        let v178 := v177.fst
        let v179 := v178
        r5 := v179
        let v180 := v177.snd
        let v181 := v180
        let v182 := v2.l2
        let v184 ← Small_row4_1 v0 v1 v182 r5 v181
        let v185 := v184.fst
        let v186 := v185
        r6 := v186
        let v187 := v184.snd
        let v188 := v187
        let v189 := v2.l3
        let v191 ← Small_row4_1 v0 v1 v189 r6 v188
        let v192 := v191.fst
        let v193 := v192
        let v194 := v191.snd
        let v195 := v194
        let v196 := v193.l0
        let v197 := v0.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let v201 := v200
        let v202 := v199.snd
        let v203 := v202
        let v204 := v193.l1
        let v205 := v0.m.l1
        let v206 ← sbb v204 v205 v203
        let v207 := v206.fst
        let v208 := v207
        let v209 := v206.snd
        let v210 := v209
        let v211 := v193.l2
        let v212 := v0.m.l2
        let v213 ← sbb v211 v212 v210
        let v214 := v213.fst
        let v215 := v214
        let v216 := v213.snd
        let v217 := v216
        let v218 := v193.l3
        let v219 := v0.m.l3
        let v220 ← sbb v218 v219 v217
        let v221 := v220.fst
        let v222 := v221
        let v223 := v220.snd
        let v224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb v195 v225 v224
        let v227 := v226.snd
        let v228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (v228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v7 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v7
        if v230 then
          let v231 := v193
          b24 r0 r1 r2 r3 r4 r5 r6 v231
        else
          let v232 := (Wide.mk v201 v208 v215 v222)
          let v233 := v232
          b24 r0 r1 r2 r3 r4 r5 r6 v233

def Small_r2_1 (v0 : Small_1) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 1)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let v29 := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := v29
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let v35 := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 v35
    let v38 := v37
    let v39 := (UInt256.ofNat 1)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v4 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 1)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 1)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 1)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v7 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let v64 := v63
            r0 := v64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 v35
            let v67 ← downcast_truncate_u64_usize v66
            let v68 := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 1)
            let v71 := v68 + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let v81 := r0
                let v82 := v81
                r2 := v82
                let v84 ← Small_double_1 v0 r2
                let v85 := v84
                r0 := v85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 r0 r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let v101 := r0
                    let v102 := v101
                    r4 := v102
                    let v103 := r0
                    let v104 := v103
                    r5 := v104
                    let v107 ← Small_mul_1 v0 r4 r5
                    let v108 := v107
                    r0 := v108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 r0 r1 r2 r3 r4 r5 v112
                  else
                    let v113 := r0
                    return v113
                partial_fixpoint
                b26 r0 r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 r0 r1 r2 r3 r4 r5 v76
          if v59 then
            let v60 := v38
            b21 r0 r1 r2 r3 r4 r5 v60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 r0 r1 r2 r3 r4 r5 v62
        if v53 then
          let v54 := v38
          b18 r0 r1 r2 r3 r4 r5 v54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 r0 r1 r2 r3 r4 r5 v56
      if v47 then
        let v48 := v38
        b15 r0 r1 r2 r3 r4 r5 v48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 r0 r1 r2 r3 r4 r5 v50
    if v41 then
      let v42 := v38
      b12 r0 r1 r2 r3 r4 r5 v42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 r0 r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := v0.m.l0
    let v14 := v13
    b3 r0 r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 1)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v2 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 r0 r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := v0.m.l1
      let v19 := v18
      b6 r0 r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 1)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v3 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 r0 r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := v0.m.l2
        let v24 := v23
        b9 r0 r1 r2 r3 r4 r5 v24
      else
        let v25 := v0.m.l3
        let v26 := v25
        b9 r0 r1 r2 r3 r4 r5 v26

def limb (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v1
  let v3 := (UInt256.ofNat 8)
  let v4 := v2 * v3
  let v5 := v0 + v4
  return v5

def Exponent_bit (v0 : Exponent) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0.total
  let v3 := (UInt256.ofNat 1)
  let v4 := v2 - v3
  let v5 := v1
  let v6 := v4 - v5
  let v7 := v6
  let v8 := (UInt256.ofNat 63)
  let v9 := v7 &&& v8
  let v10 ← downcast_truncate_usize_u64 v9
  let v11 := v10
  let v14 ← MemArray_ptr_u64 v0.limbs
  let v15 := (UInt256.ofNat 6)
  let v16 := v7 >>> v15
  let v17 ← limb v14 v16
  let v18 ← mloadInt (.u 64) v17
  let v19 := v18 >>> v11
  let v20 := (UInt256.ofNat 1)
  let v21 := v19 &&& v20
  let v22 ← downcast_truncate_u64_usize v21
  return v22

def Exponent_window (v0 : Exponent) (v1 : UInt256) (v2 : UInt256) : FeM Window := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : Range_Unknown_Unknown := default
  let v5 ← Exponent_bit v0 v1
  let v6 := (UInt256.ofNat 0)
  let v7 := decide (v5 = v6)
  if v7 then
    let v8 := v1
    let v9 := (UInt256.ofNat 1)
    let v10 := v8 + v9
    let v11 := (UInt256.ofNat 0)
    let v12 := false
    let v13 := (Window.mk v10 v11 v12)
    return v13
  else
    let v14 := ()
    let v15 := v14
    let v16 := v0.total
    let v17 := v1
    let v18 := v16 - v17
    let v19 := v2
    let v20 := decide (v18.toNat < v19.toNat)
    let b6 : UInt256 → UInt256 → Range_Unknown_Unknown → UInt256 → FeM Window := fun r0 r1 r2 v3 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let v27 := v3
      r0 := v27
      let rec b7 (r0 : UInt256) (r1 : UInt256) (r2 : Range_Unknown_Unknown) : FeM Window := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let v28 := r0
        let v29 := v28
        let v30 := (UInt256.ofNat 1)
        let v31 := v29 - v30
        let v32 ← Exponent_bit v0 v31
        let v33 := (UInt256.ofNat 0)
        let v34 := decide (v32 = v33)
        if v34 then
          let v35 := (UInt256.ofNat 1)
          let v36 := r0
          let v37 := v36 - v35
          let v38 := v37
          r0 := v38
          let v39 := ()
          b7 r0 r1 r2
        else
          let v40 := (UInt256.ofNat 0)
          let v41 := v40
          r1 := v41
          let v42 := v1
          let v43 := ()
          let v44 := r0
          let v45 := (Range_Unknown_Unknown.mk v42 v44)
          r2 := v45
          let v47 ← Range_Unknown_Unknown_start r2
          let v48 := v47
          let rec b10 (r0 : UInt256) (r1 : UInt256) (r2 : Range_Unknown_Unknown) (v4 : UInt256) : FeM Window := do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let v50 ← Range_Unknown_Unknown_has r2 v4
            if v50 then
              let v52 ← Range_Unknown_Unknown_at r2 v4
              let v53 := v52
              let v54 := r1
              let v55 := v54
              let v56 := (UInt256.ofNat 1)
              let v57 := v55 <<< v56
              let v58 ← Exponent_bit v0 v53
              let v59 := v57 ||| v58
              let v60 := v59
              r1 := v60
              let v61 := ()
              let v63 ← Range_Unknown_Unknown_next r2 v4
              let v64 := v63
              b10 r0 r1 r2 v64
            else
              let v65 := r0
              let v66 := v65
              let v67 := r1
              let v68 := v67
              let v69 := (UInt256.ofNat 1)
              let v70 := v68 - v69
              let v71 := (UInt256.ofNat 2)
              let v72 ← checkedDiv v70 v71
              let v73 := true
              let v74 := (Window.mk v66 v72 v73)
              return v74
          partial_fixpoint
          b10 r0 r1 r2 v48
      partial_fixpoint
      b7 r0 r1 r2
    if v20 then
      let v21 := v0.total
      let v22 := v21
      b6 r0 r1 r2 v22
    else
      let v23 := v1
      let v24 := v2
      let v25 := v23 + v24
      let v26 := v25
      b6 r0 r1 r2 v26

def small_power_1 (v0 : Exponent) (v1 : MemArray_u64) (v2 : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut r0 : UInt256 := default
  let mut r1 : Small_1 := default
  let mut r2 : (Vector Wide 64) := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : UInt256 := default
  let mut r6 : Bool := default
  let mut r7 : MemArray_u64 := default
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
  let v8 ← MemArray_index_ref_u64 v2 v7
  let v9 ← mloadInt (.u 64) v8
  let v10 := v9
  r0 := v10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (r0 : UInt256) (r1 : Small_1) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let v22 := r0
      let v23 := v22
      let v24 := (UInt256.ofNat 2)
      let v25 := r0
      let v26 := IntTy.mulW (.u 64) v9 v25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) v23 v27
      let v29 := v28
      r0 := v29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 1)
      let v35 ← Wide_of v2 v34
      let v36 := (UInt256.ofNat 0)
      let v37 := r0
      let v38 := IntTy.subW (.u 64) v36 v37
      let v39 := (Small_1.mk v35 v38)
      let v40 := v39
      r1 := v40
      let v41 := v0.total
      let v42 := v0.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let v45 := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := v45 - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let v51 := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let v58 := v57
      r2 := v58
      let v59 := (UInt256.ofNat 1)
      let v60 ← Wide_of v1 v59
      r9 := v60
      let v62 ← Small_r2_1 r1
      r10 := v62
      let v66 ← Small_mul_1 r1 r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      r2 := (r2.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < v51.toNat)
      let b7 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := r2[v106.toNat]!
        let v108 := v107
        r4 := v108
        let v109 := v0.top
        let v110 := v109
        r5 := v110
        let v111 := false
        let v112 := v111
        r6 := v112
        let rec b12 (r0 : UInt256) (r1 : Small_1) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let v113 := r5
          let v114 := v113
          let v115 := v0.total
          let v116 := decide (v114.toNat < v115.toNat)
          if v116 then
            let v117 := r5
            let v118 := v117
            let v119 ← Exponent_window v0 v118 v45
            let v120 := v119
            let v121 := r6
            let rec b17 (r0 : UInt256) (r1 : Small_1) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := v120.odd
              let b24 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := v120.«end»
                let v174 := v173
                r5 := v174
                let v175 := ()
                b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let v152 := r6
                let b27 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let v165 := v6
                  r4 := v165
                  let v166 := ()
                  let v167 := true
                  let v168 := v167
                  r6 := v168
                  let v169 := ()
                  let v170 := v169
                  b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v152 then
                  let v153 := r4
                  let v154 := v153
                  r18 := v154
                  let v155 := v120.entry
                  checkIndex v155 64
                  let v156 := r2[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_1 r1 r18 r19
                  let v161 := v160
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := v120.entry
                  checkIndex v162 64
                  let v163 := r2[v162.toNat]!
                  let v164 := v163
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if v121 then
              let v122 := r5
              let v123 := v122
              let v124 := v120.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk v123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (r0 : UInt256) (r1 : Small_1) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let v134 := r4
                  let v135 := v134
                  r16 := v135
                  let v136 := r4
                  let v137 := v136
                  r17 := v137
                  let v141 ← Small_mul_1 r1 r16 r17
                  let v142 := v141
                  r4 := v142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let v176 := r4
            let v177 := v176
            r20 := v177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_1 r1 r20 r21
            let v187 := v186
            let v188 := (UInt256.ofNat 1)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let v192 := v191
              r7 := v192
              let v193 := v187.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 r7 v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              r7 := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 1)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 1)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 1)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_1 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let v231 := r7
                    let v232 := v231
                    let v233 := Option_MemArray_u64.Some v232
                    return v233
                  if v222 then
                    let v223 := v187.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 r7 v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    r7 := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := v187.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 r7 v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  r7 := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := v187.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 r7 v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                r7 := v204_0
                let v205 := ()
                let v206 := v205
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := r2[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := r2[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_1 r1 r11 r12
        let v79 := v78
        r3 := v79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 v51)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (r0 : UInt256) (r1 : Small_1) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let v90 := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := v90 - v91
            checkIndex v92 64
            let v93 := r2[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_1 r1 r14 r3
            checkIndex v90 64
            r2 := (r2.set! v90.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def Small_double_2_k2 (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v30 : UInt256) (v35 : UInt256) (v41 : UInt256) (v47 : UInt256) (v49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb v30 v50 v49
  let v52 := v51.fst
  let v53 := v52
  let v54 := v51.snd
  let v55 := v54
  let v56 := v1.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 v55
  let v61 := v60.snd
  let v62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (v62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk v6 v14 v22 v30)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk v35 v41 v47 v53)
    let v68 := v67
    b3 v68

def Small_double_2_k1 (v0 : Small_2) (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let v30 := v29
  let v31 := v0.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb v6 v31 v32
  let v34 := v33.fst
  let v35 := v34
  let v36 := v33.snd
  let v37 := v36
  let v38 := v0.m.l1
  let v39 ← sbb v14 v38 v37
  let v40 := v39.fst
  let v41 := v40
  let v42 := v39.snd
  let v43 := v42
  let v44 := v0.m.l2
  let v45 ← sbb v22 v44 v43
  let v46 := v45.fst
  let v47 := v46
  let v48 := v45.snd
  let v49 := v48
  let v50 := v0.m.l3
  Small_double_2_k2 v1 v6 v14 v22 v30 v35 v41 v47 v49 v50

def Small_double_2 (v0 : Small_2) (v1 : Wide) : FeM Wide := do
  let v3 := v1.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let v6 := v5
  let v7 := v1.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := v1.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let v14 := v13
  let v15 := v1.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := v1.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let v22 := v21
  let v23 := v1.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := v1.l2
  Small_double_2_k1 v0 v1 v6 v14 v22 v25 v26

def Small_row2_2_k1 (v0 : Small_2) (v10 : UInt256) (v17 : UInt256) (v23 : UInt256) (v25 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 := v0.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac v28 v29 v10 v30
  let v32 := v31.snd
  let v33 := v32
  let v34 := v0.m.l1
  let v35 ← mac v28 v34 v17 v33
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc v23 v40 v39
  let v42 := v41.fst
  let v43 := v42
  let v44 := v41.snd
  let v45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk v37 v43 v46 v47)
  let v49 := IntTy.addW (.u 64) v25 v45
  let v50 := (v48, v49)
  return v50

def Small_row2_2 (v0 : Small_2) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc v4 v19 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  let v26 := v0.inverse
  let v27 := IntTy.mulW (.u 64) v10 v26
  let v28 := v27
  Small_row2_2_k1 v0 v10 v17 v23 v25 v28

def Small_row3_2_k2 (v30 : UInt256) (v32 : UInt256) (v44 : UInt256) (v50 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc v30 v53 v52
  let v55 := v54.fst
  let v56 := v55
  let v57 := v54.snd
  let v58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk v44 v50 v56 v59)
  let v61 := IntTy.addW (.u 64) v32 v58
  let v62 := (v60, v61)
  return v62

def Small_row3_2_k1 (v0 : Small_2) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let v30 := v29
  let v31 := v28.snd
  let v32 := v31
  let v33 := v0.inverse
  let v34 := IntTy.mulW (.u 64) v10 v33
  let v35 := v34
  let v36 := v0.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac v35 v36 v10 v37
  let v39 := v38.snd
  let v40 := v39
  let v41 := v0.m.l1
  let v42 ← mac v35 v41 v17 v40
  let v43 := v42.fst
  let v44 := v43
  let v45 := v42.snd
  let v46 := v45
  let v47 := v0.m.l2
  let v48 ← mac v35 v47 v24 v46
  let v49 := v48.fst
  let v50 := v49
  let v51 := v48.snd
  let v52 := v51
  Small_row3_2_k2 v30 v32 v44 v50 v52

def Small_row3_2 (v0 : Small_2) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc v4 v26 v27
  Small_row3_2_k1 v0 v10 v17 v24 v28

def Small_row4_2_k2 (v0 : Small_2) (v24 : UInt256) (v31 : UInt256) (v37 : UInt256) (v39 : UInt256) (v42 : UInt256) (v51 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := v52
  let v54 := v0.m.l2
  let v55 ← mac v42 v54 v24 v53
  let v56 := v55.fst
  let v57 := v56
  let v58 := v55.snd
  let v59 := v58
  let v60 := v0.m.l3
  let v61 ← mac v42 v60 v31 v59
  let v62 := v61.fst
  let v63 := v62
  let v64 := v61.snd
  let v65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc v37 v66 v65
  let v68 := v67.fst
  let v69 := v68
  let v70 := v67.snd
  let v71 := v70
  let v72 := (Wide.mk v51 v57 v63 v69)
  let v73 := IntTy.addW (.u 64) v39 v71
  let v74 := (v72, v73)
  return v74

def Small_row4_2_k1 (v0 : Small_2) (v2 : UInt256) (v4 : UInt256) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 v2 v28 v26
  let v30 := v29.fst
  let v31 := v30
  let v32 := v29.snd
  let v33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc v4 v33 v34
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := v0.inverse
  let v41 := IntTy.mulW (.u 64) v10 v40
  let v42 := v41
  let v43 := v0.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac v42 v43 v10 v44
  let v46 := v45.snd
  let v47 := v46
  let v48 := v0.m.l1
  let v49 ← mac v42 v48 v17 v47
  let v50 := v49.fst
  let v51 := v50
  let v52 := v49.snd
  Small_row4_2_k2 v0 v24 v31 v37 v39 v42 v51 v52

def Small_row4_2 (v0 : Small_2) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := v1.l3
  let v28 := v3.l3
  Small_row4_2_k1 v0 v2 v4 v10 v17 v24 v26 v27 v28

def Small_mul_2 (v0 : Small_2) (v1 : Wide) (v2 : Wide) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Wide := default
  let mut r2 : Wide := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let mut r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let v13 := v12
  r0 := v13
  let v14 := (UInt256.ofNat 2)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := v1.l0
    let v18 := v17
    let v19 := v2.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let v22 := v21
    let v23 ← downcast_truncate_u128_u64 v22
    let v24 := v23
    let v25 := v0.inverse
    let v26 := IntTy.mulW (.u 64) v24 v25
    let v27 := v26
    let v28 := v27
    let v29 := v0.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let v32 := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := v22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := v32 >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (v24 ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v3 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let v45 := v44
      let v46 := v0.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ v45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let v54 := v4
        let v55 ← downcast_truncate_u128_u64 v54
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := v0.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) v45 v50
        let v52 := v51
        b9 r0 r1 r2 r3 r4 r5 r6 v52
      else
        let v53 := v45
        b9 r0 r1 r2 r3 r4 r5 r6 v53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 r0 r1 r2 r3 r4 r5 r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 r0 r1 r2 r3 r4 r5 r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 2)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := v2.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_2 v0 v1 v65 r0 v66
      let v69 := v68.fst
      let v70 := v69
      r1 := v70
      let v71 := v68.snd
      let v72 := v71
      let v73 := v2.l1
      let v75 ← Small_row2_2 v0 v1 v73 r1 v72
      let v76 := v75.fst
      let v77 := v76
      let v78 := v75.snd
      let v79 := v78
      let v80 := v77.l0
      let v81 := v0.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let v85 := v84
      let v86 := v83.snd
      let v87 := v86
      let v88 := v77.l1
      let v89 := v0.m.l1
      let v90 ← sbb v88 v89 v87
      let v91 := v90.fst
      let v92 := v91
      let v93 := v90.snd
      let v94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb v79 v95 v94
      let v97 := v96.snd
      let v98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (v98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        return v5
      if v100 then
        let v101 := v77
        b15 r0 r1 r2 r3 r4 r5 r6 v101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk v85 v92 v102 v103)
        let v105 := v104
        b15 r0 r1 r2 r3 r4 r5 r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 2)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := v2.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_2 v0 v1 v111 r0 v112
        let v115 := v114.fst
        let v116 := v115
        r2 := v116
        let v117 := v114.snd
        let v118 := v117
        let v119 := v2.l1
        let v121 ← Small_row3_2 v0 v1 v119 r2 v118
        let v122 := v121.fst
        let v123 := v122
        r3 := v123
        let v124 := v121.snd
        let v125 := v124
        let v126 := v2.l2
        let v128 ← Small_row3_2 v0 v1 v126 r3 v125
        let v129 := v128.fst
        let v130 := v129
        let v131 := v128.snd
        let v132 := v131
        let v133 := v130.l0
        let v134 := v0.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let v138 := v137
        let v139 := v136.snd
        let v140 := v139
        let v141 := v130.l1
        let v142 := v0.m.l1
        let v143 ← sbb v141 v142 v140
        let v144 := v143.fst
        let v145 := v144
        let v146 := v143.snd
        let v147 := v146
        let v148 := v130.l2
        let v149 := v0.m.l2
        let v150 ← sbb v148 v149 v147
        let v151 := v150.fst
        let v152 := v151
        let v153 := v150.snd
        let v154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb v132 v155 v154
        let v157 := v156.snd
        let v158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (v158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v6
        if v160 then
          let v161 := v130
          b21 r0 r1 r2 r3 r4 r5 r6 v161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk v138 v145 v152 v162)
          let v164 := v163
          b21 r0 r1 r2 r3 r4 r5 r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := v2.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_2 v0 v1 v167 r0 v168
        let v171 := v170.fst
        let v172 := v171
        r4 := v172
        let v173 := v170.snd
        let v174 := v173
        let v175 := v2.l1
        let v177 ← Small_row4_2 v0 v1 v175 r4 v174
        let v178 := v177.fst
        let v179 := v178
        r5 := v179
        let v180 := v177.snd
        let v181 := v180
        let v182 := v2.l2
        let v184 ← Small_row4_2 v0 v1 v182 r5 v181
        let v185 := v184.fst
        let v186 := v185
        r6 := v186
        let v187 := v184.snd
        let v188 := v187
        let v189 := v2.l3
        let v191 ← Small_row4_2 v0 v1 v189 r6 v188
        let v192 := v191.fst
        let v193 := v192
        let v194 := v191.snd
        let v195 := v194
        let v196 := v193.l0
        let v197 := v0.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let v201 := v200
        let v202 := v199.snd
        let v203 := v202
        let v204 := v193.l1
        let v205 := v0.m.l1
        let v206 ← sbb v204 v205 v203
        let v207 := v206.fst
        let v208 := v207
        let v209 := v206.snd
        let v210 := v209
        let v211 := v193.l2
        let v212 := v0.m.l2
        let v213 ← sbb v211 v212 v210
        let v214 := v213.fst
        let v215 := v214
        let v216 := v213.snd
        let v217 := v216
        let v218 := v193.l3
        let v219 := v0.m.l3
        let v220 ← sbb v218 v219 v217
        let v221 := v220.fst
        let v222 := v221
        let v223 := v220.snd
        let v224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb v195 v225 v224
        let v227 := v226.snd
        let v228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (v228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v7 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v7
        if v230 then
          let v231 := v193
          b24 r0 r1 r2 r3 r4 r5 r6 v231
        else
          let v232 := (Wide.mk v201 v208 v215 v222)
          let v233 := v232
          b24 r0 r1 r2 r3 r4 r5 r6 v233

def Small_r2_2 (v0 : Small_2) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 2)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let v29 := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := v29
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let v35 := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 v35
    let v38 := v37
    let v39 := (UInt256.ofNat 2)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v4 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 2)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 2)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 2)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v7 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let v64 := v63
            r0 := v64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 v35
            let v67 ← downcast_truncate_u64_usize v66
            let v68 := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 2)
            let v71 := v68 + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let v81 := r0
                let v82 := v81
                r2 := v82
                let v84 ← Small_double_2 v0 r2
                let v85 := v84
                r0 := v85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 r0 r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let v101 := r0
                    let v102 := v101
                    r4 := v102
                    let v103 := r0
                    let v104 := v103
                    r5 := v104
                    let v107 ← Small_mul_2 v0 r4 r5
                    let v108 := v107
                    r0 := v108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 r0 r1 r2 r3 r4 r5 v112
                  else
                    let v113 := r0
                    return v113
                partial_fixpoint
                b26 r0 r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 r0 r1 r2 r3 r4 r5 v76
          if v59 then
            let v60 := v38
            b21 r0 r1 r2 r3 r4 r5 v60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 r0 r1 r2 r3 r4 r5 v62
        if v53 then
          let v54 := v38
          b18 r0 r1 r2 r3 r4 r5 v54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 r0 r1 r2 r3 r4 r5 v56
      if v47 then
        let v48 := v38
        b15 r0 r1 r2 r3 r4 r5 v48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 r0 r1 r2 r3 r4 r5 v50
    if v41 then
      let v42 := v38
      b12 r0 r1 r2 r3 r4 r5 v42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 r0 r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := v0.m.l0
    let v14 := v13
    b3 r0 r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 2)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v2 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 r0 r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := v0.m.l1
      let v19 := v18
      b6 r0 r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 2)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v3 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 r0 r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := v0.m.l2
        let v24 := v23
        b9 r0 r1 r2 r3 r4 r5 v24
      else
        let v25 := v0.m.l3
        let v26 := v25
        b9 r0 r1 r2 r3 r4 r5 v26

def small_power_2 (v0 : Exponent) (v1 : MemArray_u64) (v2 : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut r0 : UInt256 := default
  let mut r1 : Small_2 := default
  let mut r2 : (Vector Wide 64) := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : UInt256 := default
  let mut r6 : Bool := default
  let mut r7 : MemArray_u64 := default
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
  let v8 ← MemArray_index_ref_u64 v2 v7
  let v9 ← mloadInt (.u 64) v8
  let v10 := v9
  r0 := v10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (r0 : UInt256) (r1 : Small_2) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let v22 := r0
      let v23 := v22
      let v24 := (UInt256.ofNat 2)
      let v25 := r0
      let v26 := IntTy.mulW (.u 64) v9 v25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) v23 v27
      let v29 := v28
      r0 := v29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 2)
      let v35 ← Wide_of v2 v34
      let v36 := (UInt256.ofNat 0)
      let v37 := r0
      let v38 := IntTy.subW (.u 64) v36 v37
      let v39 := (Small_2.mk v35 v38)
      let v40 := v39
      r1 := v40
      let v41 := v0.total
      let v42 := v0.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let v45 := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := v45 - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let v51 := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let v58 := v57
      r2 := v58
      let v59 := (UInt256.ofNat 2)
      let v60 ← Wide_of v1 v59
      r9 := v60
      let v62 ← Small_r2_2 r1
      r10 := v62
      let v66 ← Small_mul_2 r1 r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      r2 := (r2.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < v51.toNat)
      let b7 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := r2[v106.toNat]!
        let v108 := v107
        r4 := v108
        let v109 := v0.top
        let v110 := v109
        r5 := v110
        let v111 := false
        let v112 := v111
        r6 := v112
        let rec b12 (r0 : UInt256) (r1 : Small_2) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let v113 := r5
          let v114 := v113
          let v115 := v0.total
          let v116 := decide (v114.toNat < v115.toNat)
          if v116 then
            let v117 := r5
            let v118 := v117
            let v119 ← Exponent_window v0 v118 v45
            let v120 := v119
            let v121 := r6
            let rec b17 (r0 : UInt256) (r1 : Small_2) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := v120.odd
              let b24 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := v120.«end»
                let v174 := v173
                r5 := v174
                let v175 := ()
                b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let v152 := r6
                let b27 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let v165 := v6
                  r4 := v165
                  let v166 := ()
                  let v167 := true
                  let v168 := v167
                  r6 := v168
                  let v169 := ()
                  let v170 := v169
                  b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v152 then
                  let v153 := r4
                  let v154 := v153
                  r18 := v154
                  let v155 := v120.entry
                  checkIndex v155 64
                  let v156 := r2[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_2 r1 r18 r19
                  let v161 := v160
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := v120.entry
                  checkIndex v162 64
                  let v163 := r2[v162.toNat]!
                  let v164 := v163
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if v121 then
              let v122 := r5
              let v123 := v122
              let v124 := v120.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk v123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (r0 : UInt256) (r1 : Small_2) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let v134 := r4
                  let v135 := v134
                  r16 := v135
                  let v136 := r4
                  let v137 := v136
                  r17 := v137
                  let v141 ← Small_mul_2 r1 r16 r17
                  let v142 := v141
                  r4 := v142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let v176 := r4
            let v177 := v176
            r20 := v177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_2 r1 r20 r21
            let v187 := v186
            let v188 := (UInt256.ofNat 2)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let v192 := v191
              r7 := v192
              let v193 := v187.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 r7 v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              r7 := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 2)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 2)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 2)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_2 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let v231 := r7
                    let v232 := v231
                    let v233 := Option_MemArray_u64.Some v232
                    return v233
                  if v222 then
                    let v223 := v187.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 r7 v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    r7 := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := v187.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 r7 v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  r7 := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := v187.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 r7 v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                r7 := v204_0
                let v205 := ()
                let v206 := v205
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := r2[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := r2[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_2 r1 r11 r12
        let v79 := v78
        r3 := v79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 v51)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (r0 : UInt256) (r1 : Small_2) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let v90 := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := v90 - v91
            checkIndex v92 64
            let v93 := r2[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_2 r1 r14 r3
            checkIndex v90 64
            r2 := (r2.set! v90.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def Small_double_3_k2 (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v30 : UInt256) (v35 : UInt256) (v41 : UInt256) (v47 : UInt256) (v49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb v30 v50 v49
  let v52 := v51.fst
  let v53 := v52
  let v54 := v51.snd
  let v55 := v54
  let v56 := v1.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 v55
  let v61 := v60.snd
  let v62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (v62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk v6 v14 v22 v30)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk v35 v41 v47 v53)
    let v68 := v67
    b3 v68

def Small_double_3_k1 (v0 : Small_3) (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let v30 := v29
  let v31 := v0.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb v6 v31 v32
  let v34 := v33.fst
  let v35 := v34
  let v36 := v33.snd
  let v37 := v36
  let v38 := v0.m.l1
  let v39 ← sbb v14 v38 v37
  let v40 := v39.fst
  let v41 := v40
  let v42 := v39.snd
  let v43 := v42
  let v44 := v0.m.l2
  let v45 ← sbb v22 v44 v43
  let v46 := v45.fst
  let v47 := v46
  let v48 := v45.snd
  let v49 := v48
  let v50 := v0.m.l3
  Small_double_3_k2 v1 v6 v14 v22 v30 v35 v41 v47 v49 v50

def Small_double_3 (v0 : Small_3) (v1 : Wide) : FeM Wide := do
  let v3 := v1.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let v6 := v5
  let v7 := v1.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := v1.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let v14 := v13
  let v15 := v1.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := v1.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let v22 := v21
  let v23 := v1.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := v1.l2
  Small_double_3_k1 v0 v1 v6 v14 v22 v25 v26

def Small_row2_3_k1 (v0 : Small_3) (v10 : UInt256) (v17 : UInt256) (v23 : UInt256) (v25 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 := v0.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac v28 v29 v10 v30
  let v32 := v31.snd
  let v33 := v32
  let v34 := v0.m.l1
  let v35 ← mac v28 v34 v17 v33
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc v23 v40 v39
  let v42 := v41.fst
  let v43 := v42
  let v44 := v41.snd
  let v45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk v37 v43 v46 v47)
  let v49 := IntTy.addW (.u 64) v25 v45
  let v50 := (v48, v49)
  return v50

def Small_row2_3 (v0 : Small_3) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc v4 v19 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  let v26 := v0.inverse
  let v27 := IntTy.mulW (.u 64) v10 v26
  let v28 := v27
  Small_row2_3_k1 v0 v10 v17 v23 v25 v28

def Small_row3_3_k2 (v30 : UInt256) (v32 : UInt256) (v44 : UInt256) (v50 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc v30 v53 v52
  let v55 := v54.fst
  let v56 := v55
  let v57 := v54.snd
  let v58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk v44 v50 v56 v59)
  let v61 := IntTy.addW (.u 64) v32 v58
  let v62 := (v60, v61)
  return v62

def Small_row3_3_k1 (v0 : Small_3) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let v30 := v29
  let v31 := v28.snd
  let v32 := v31
  let v33 := v0.inverse
  let v34 := IntTy.mulW (.u 64) v10 v33
  let v35 := v34
  let v36 := v0.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac v35 v36 v10 v37
  let v39 := v38.snd
  let v40 := v39
  let v41 := v0.m.l1
  let v42 ← mac v35 v41 v17 v40
  let v43 := v42.fst
  let v44 := v43
  let v45 := v42.snd
  let v46 := v45
  let v47 := v0.m.l2
  let v48 ← mac v35 v47 v24 v46
  let v49 := v48.fst
  let v50 := v49
  let v51 := v48.snd
  let v52 := v51
  Small_row3_3_k2 v30 v32 v44 v50 v52

def Small_row3_3 (v0 : Small_3) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc v4 v26 v27
  Small_row3_3_k1 v0 v10 v17 v24 v28

def Small_row4_3_k2 (v0 : Small_3) (v24 : UInt256) (v31 : UInt256) (v37 : UInt256) (v39 : UInt256) (v42 : UInt256) (v51 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := v52
  let v54 := v0.m.l2
  let v55 ← mac v42 v54 v24 v53
  let v56 := v55.fst
  let v57 := v56
  let v58 := v55.snd
  let v59 := v58
  let v60 := v0.m.l3
  let v61 ← mac v42 v60 v31 v59
  let v62 := v61.fst
  let v63 := v62
  let v64 := v61.snd
  let v65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc v37 v66 v65
  let v68 := v67.fst
  let v69 := v68
  let v70 := v67.snd
  let v71 := v70
  let v72 := (Wide.mk v51 v57 v63 v69)
  let v73 := IntTy.addW (.u 64) v39 v71
  let v74 := (v72, v73)
  return v74

def Small_row4_3_k1 (v0 : Small_3) (v2 : UInt256) (v4 : UInt256) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 v2 v28 v26
  let v30 := v29.fst
  let v31 := v30
  let v32 := v29.snd
  let v33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc v4 v33 v34
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := v0.inverse
  let v41 := IntTy.mulW (.u 64) v10 v40
  let v42 := v41
  let v43 := v0.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac v42 v43 v10 v44
  let v46 := v45.snd
  let v47 := v46
  let v48 := v0.m.l1
  let v49 ← mac v42 v48 v17 v47
  let v50 := v49.fst
  let v51 := v50
  let v52 := v49.snd
  Small_row4_3_k2 v0 v24 v31 v37 v39 v42 v51 v52

def Small_row4_3 (v0 : Small_3) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := v1.l3
  let v28 := v3.l3
  Small_row4_3_k1 v0 v2 v4 v10 v17 v24 v26 v27 v28

def Small_mul_3 (v0 : Small_3) (v1 : Wide) (v2 : Wide) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Wide := default
  let mut r2 : Wide := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let mut r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let v13 := v12
  r0 := v13
  let v14 := (UInt256.ofNat 3)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := v1.l0
    let v18 := v17
    let v19 := v2.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let v22 := v21
    let v23 ← downcast_truncate_u128_u64 v22
    let v24 := v23
    let v25 := v0.inverse
    let v26 := IntTy.mulW (.u 64) v24 v25
    let v27 := v26
    let v28 := v27
    let v29 := v0.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let v32 := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := v22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := v32 >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (v24 ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v3 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let v45 := v44
      let v46 := v0.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ v45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let v54 := v4
        let v55 ← downcast_truncate_u128_u64 v54
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := v0.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) v45 v50
        let v52 := v51
        b9 r0 r1 r2 r3 r4 r5 r6 v52
      else
        let v53 := v45
        b9 r0 r1 r2 r3 r4 r5 r6 v53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 r0 r1 r2 r3 r4 r5 r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 r0 r1 r2 r3 r4 r5 r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 3)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := v2.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_3 v0 v1 v65 r0 v66
      let v69 := v68.fst
      let v70 := v69
      r1 := v70
      let v71 := v68.snd
      let v72 := v71
      let v73 := v2.l1
      let v75 ← Small_row2_3 v0 v1 v73 r1 v72
      let v76 := v75.fst
      let v77 := v76
      let v78 := v75.snd
      let v79 := v78
      let v80 := v77.l0
      let v81 := v0.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let v85 := v84
      let v86 := v83.snd
      let v87 := v86
      let v88 := v77.l1
      let v89 := v0.m.l1
      let v90 ← sbb v88 v89 v87
      let v91 := v90.fst
      let v92 := v91
      let v93 := v90.snd
      let v94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb v79 v95 v94
      let v97 := v96.snd
      let v98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (v98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        return v5
      if v100 then
        let v101 := v77
        b15 r0 r1 r2 r3 r4 r5 r6 v101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk v85 v92 v102 v103)
        let v105 := v104
        b15 r0 r1 r2 r3 r4 r5 r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 3)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := v2.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_3 v0 v1 v111 r0 v112
        let v115 := v114.fst
        let v116 := v115
        r2 := v116
        let v117 := v114.snd
        let v118 := v117
        let v119 := v2.l1
        let v121 ← Small_row3_3 v0 v1 v119 r2 v118
        let v122 := v121.fst
        let v123 := v122
        r3 := v123
        let v124 := v121.snd
        let v125 := v124
        let v126 := v2.l2
        let v128 ← Small_row3_3 v0 v1 v126 r3 v125
        let v129 := v128.fst
        let v130 := v129
        let v131 := v128.snd
        let v132 := v131
        let v133 := v130.l0
        let v134 := v0.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let v138 := v137
        let v139 := v136.snd
        let v140 := v139
        let v141 := v130.l1
        let v142 := v0.m.l1
        let v143 ← sbb v141 v142 v140
        let v144 := v143.fst
        let v145 := v144
        let v146 := v143.snd
        let v147 := v146
        let v148 := v130.l2
        let v149 := v0.m.l2
        let v150 ← sbb v148 v149 v147
        let v151 := v150.fst
        let v152 := v151
        let v153 := v150.snd
        let v154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb v132 v155 v154
        let v157 := v156.snd
        let v158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (v158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v6
        if v160 then
          let v161 := v130
          b21 r0 r1 r2 r3 r4 r5 r6 v161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk v138 v145 v152 v162)
          let v164 := v163
          b21 r0 r1 r2 r3 r4 r5 r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := v2.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_3 v0 v1 v167 r0 v168
        let v171 := v170.fst
        let v172 := v171
        r4 := v172
        let v173 := v170.snd
        let v174 := v173
        let v175 := v2.l1
        let v177 ← Small_row4_3 v0 v1 v175 r4 v174
        let v178 := v177.fst
        let v179 := v178
        r5 := v179
        let v180 := v177.snd
        let v181 := v180
        let v182 := v2.l2
        let v184 ← Small_row4_3 v0 v1 v182 r5 v181
        let v185 := v184.fst
        let v186 := v185
        r6 := v186
        let v187 := v184.snd
        let v188 := v187
        let v189 := v2.l3
        let v191 ← Small_row4_3 v0 v1 v189 r6 v188
        let v192 := v191.fst
        let v193 := v192
        let v194 := v191.snd
        let v195 := v194
        let v196 := v193.l0
        let v197 := v0.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let v201 := v200
        let v202 := v199.snd
        let v203 := v202
        let v204 := v193.l1
        let v205 := v0.m.l1
        let v206 ← sbb v204 v205 v203
        let v207 := v206.fst
        let v208 := v207
        let v209 := v206.snd
        let v210 := v209
        let v211 := v193.l2
        let v212 := v0.m.l2
        let v213 ← sbb v211 v212 v210
        let v214 := v213.fst
        let v215 := v214
        let v216 := v213.snd
        let v217 := v216
        let v218 := v193.l3
        let v219 := v0.m.l3
        let v220 ← sbb v218 v219 v217
        let v221 := v220.fst
        let v222 := v221
        let v223 := v220.snd
        let v224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb v195 v225 v224
        let v227 := v226.snd
        let v228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (v228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v7 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v7
        if v230 then
          let v231 := v193
          b24 r0 r1 r2 r3 r4 r5 r6 v231
        else
          let v232 := (Wide.mk v201 v208 v215 v222)
          let v233 := v232
          b24 r0 r1 r2 r3 r4 r5 r6 v233

def Small_r2_3 (v0 : Small_3) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 3)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let v29 := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := v29
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let v35 := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 v35
    let v38 := v37
    let v39 := (UInt256.ofNat 3)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v4 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 3)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 3)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 3)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v7 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let v64 := v63
            r0 := v64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 v35
            let v67 ← downcast_truncate_u64_usize v66
            let v68 := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 3)
            let v71 := v68 + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let v81 := r0
                let v82 := v81
                r2 := v82
                let v84 ← Small_double_3 v0 r2
                let v85 := v84
                r0 := v85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 r0 r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let v101 := r0
                    let v102 := v101
                    r4 := v102
                    let v103 := r0
                    let v104 := v103
                    r5 := v104
                    let v107 ← Small_mul_3 v0 r4 r5
                    let v108 := v107
                    r0 := v108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 r0 r1 r2 r3 r4 r5 v112
                  else
                    let v113 := r0
                    return v113
                partial_fixpoint
                b26 r0 r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 r0 r1 r2 r3 r4 r5 v76
          if v59 then
            let v60 := v38
            b21 r0 r1 r2 r3 r4 r5 v60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 r0 r1 r2 r3 r4 r5 v62
        if v53 then
          let v54 := v38
          b18 r0 r1 r2 r3 r4 r5 v54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 r0 r1 r2 r3 r4 r5 v56
      if v47 then
        let v48 := v38
        b15 r0 r1 r2 r3 r4 r5 v48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 r0 r1 r2 r3 r4 r5 v50
    if v41 then
      let v42 := v38
      b12 r0 r1 r2 r3 r4 r5 v42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 r0 r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := v0.m.l0
    let v14 := v13
    b3 r0 r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 3)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v2 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 r0 r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := v0.m.l1
      let v19 := v18
      b6 r0 r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 3)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v3 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 r0 r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := v0.m.l2
        let v24 := v23
        b9 r0 r1 r2 r3 r4 r5 v24
      else
        let v25 := v0.m.l3
        let v26 := v25
        b9 r0 r1 r2 r3 r4 r5 v26

def small_power_3 (v0 : Exponent) (v1 : MemArray_u64) (v2 : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut r0 : UInt256 := default
  let mut r1 : Small_3 := default
  let mut r2 : (Vector Wide 64) := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : UInt256 := default
  let mut r6 : Bool := default
  let mut r7 : MemArray_u64 := default
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
  let v8 ← MemArray_index_ref_u64 v2 v7
  let v9 ← mloadInt (.u 64) v8
  let v10 := v9
  r0 := v10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (r0 : UInt256) (r1 : Small_3) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let v22 := r0
      let v23 := v22
      let v24 := (UInt256.ofNat 2)
      let v25 := r0
      let v26 := IntTy.mulW (.u 64) v9 v25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) v23 v27
      let v29 := v28
      r0 := v29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 3)
      let v35 ← Wide_of v2 v34
      let v36 := (UInt256.ofNat 0)
      let v37 := r0
      let v38 := IntTy.subW (.u 64) v36 v37
      let v39 := (Small_3.mk v35 v38)
      let v40 := v39
      r1 := v40
      let v41 := v0.total
      let v42 := v0.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let v45 := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := v45 - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let v51 := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let v58 := v57
      r2 := v58
      let v59 := (UInt256.ofNat 3)
      let v60 ← Wide_of v1 v59
      r9 := v60
      let v62 ← Small_r2_3 r1
      r10 := v62
      let v66 ← Small_mul_3 r1 r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      r2 := (r2.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < v51.toNat)
      let b7 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := r2[v106.toNat]!
        let v108 := v107
        r4 := v108
        let v109 := v0.top
        let v110 := v109
        r5 := v110
        let v111 := false
        let v112 := v111
        r6 := v112
        let rec b12 (r0 : UInt256) (r1 : Small_3) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let v113 := r5
          let v114 := v113
          let v115 := v0.total
          let v116 := decide (v114.toNat < v115.toNat)
          if v116 then
            let v117 := r5
            let v118 := v117
            let v119 ← Exponent_window v0 v118 v45
            let v120 := v119
            let v121 := r6
            let rec b17 (r0 : UInt256) (r1 : Small_3) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := v120.odd
              let b24 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := v120.«end»
                let v174 := v173
                r5 := v174
                let v175 := ()
                b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let v152 := r6
                let b27 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let v165 := v6
                  r4 := v165
                  let v166 := ()
                  let v167 := true
                  let v168 := v167
                  r6 := v168
                  let v169 := ()
                  let v170 := v169
                  b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v152 then
                  let v153 := r4
                  let v154 := v153
                  r18 := v154
                  let v155 := v120.entry
                  checkIndex v155 64
                  let v156 := r2[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_3 r1 r18 r19
                  let v161 := v160
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := v120.entry
                  checkIndex v162 64
                  let v163 := r2[v162.toNat]!
                  let v164 := v163
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if v121 then
              let v122 := r5
              let v123 := v122
              let v124 := v120.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk v123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (r0 : UInt256) (r1 : Small_3) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let v134 := r4
                  let v135 := v134
                  r16 := v135
                  let v136 := r4
                  let v137 := v136
                  r17 := v137
                  let v141 ← Small_mul_3 r1 r16 r17
                  let v142 := v141
                  r4 := v142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let v176 := r4
            let v177 := v176
            r20 := v177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_3 r1 r20 r21
            let v187 := v186
            let v188 := (UInt256.ofNat 3)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let v192 := v191
              r7 := v192
              let v193 := v187.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 r7 v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              r7 := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 3)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 3)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 3)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_3 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let v231 := r7
                    let v232 := v231
                    let v233 := Option_MemArray_u64.Some v232
                    return v233
                  if v222 then
                    let v223 := v187.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 r7 v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    r7 := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := v187.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 r7 v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  r7 := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := v187.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 r7 v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                r7 := v204_0
                let v205 := ()
                let v206 := v205
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := r2[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := r2[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_3 r1 r11 r12
        let v79 := v78
        r3 := v79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 v51)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (r0 : UInt256) (r1 : Small_3) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let v90 := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := v90 - v91
            checkIndex v92 64
            let v93 := r2[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_3 r1 r14 r3
            checkIndex v90 64
            r2 := (r2.set! v90.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def Small_double_4_k2 (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v30 : UInt256) (v35 : UInt256) (v41 : UInt256) (v47 : UInt256) (v49 : UInt256) (v50 : UInt256) : FeM Wide := do
  let v51 ← sbb v30 v50 v49
  let v52 := v51.fst
  let v53 := v52
  let v54 := v51.snd
  let v55 := v54
  let v56 := v1.l3
  let v57 := (UInt256.ofNat 63)
  let v58 := v56 >>> v57
  let v59 := (UInt256.ofNat 0)
  let v60 ← sbb v58 v59 v55
  let v61 := v60.snd
  let v62 := v61
  let v63 := (UInt256.ofNat 1)
  let v64 := decide (v62 = v63)
  let b3 : Wide → FeM Wide := fun v2 => do
    return v2
  if v64 then
    let v65 := (Wide.mk v6 v14 v22 v30)
    let v66 := v65
    b3 v66
  else
    let v67 := (Wide.mk v35 v41 v47 v53)
    let v68 := v67
    b3 v68

def Small_double_4_k1 (v0 : Small_4) (v1 : Wide) (v6 : UInt256) (v14 : UInt256) (v22 : UInt256) (v25 : UInt256) (v26 : UInt256) : FeM Wide := do
  let v27 := (UInt256.ofNat 63)
  let v28 := v26 >>> v27
  let v29 := v25 ||| v28
  let v30 := v29
  let v31 := v0.m.l0
  let v32 := (UInt256.ofNat 0)
  let v33 ← sbb v6 v31 v32
  let v34 := v33.fst
  let v35 := v34
  let v36 := v33.snd
  let v37 := v36
  let v38 := v0.m.l1
  let v39 ← sbb v14 v38 v37
  let v40 := v39.fst
  let v41 := v40
  let v42 := v39.snd
  let v43 := v42
  let v44 := v0.m.l2
  let v45 ← sbb v22 v44 v43
  let v46 := v45.fst
  let v47 := v46
  let v48 := v45.snd
  let v49 := v48
  let v50 := v0.m.l3
  Small_double_4_k2 v1 v6 v14 v22 v30 v35 v41 v47 v49 v50

def Small_double_4 (v0 : Small_4) (v1 : Wide) : FeM Wide := do
  let v3 := v1.l0
  let v4 := (UInt256.ofNat 1)
  let v5 := IntTy.shl (.u 64) v3 v4
  let v6 := v5
  let v7 := v1.l1
  let v8 := (UInt256.ofNat 1)
  let v9 := IntTy.shl (.u 64) v7 v8
  let v10 := v1.l0
  let v11 := (UInt256.ofNat 63)
  let v12 := v10 >>> v11
  let v13 := v9 ||| v12
  let v14 := v13
  let v15 := v1.l2
  let v16 := (UInt256.ofNat 1)
  let v17 := IntTy.shl (.u 64) v15 v16
  let v18 := v1.l1
  let v19 := (UInt256.ofNat 63)
  let v20 := v18 >>> v19
  let v21 := v17 ||| v20
  let v22 := v21
  let v23 := v1.l3
  let v24 := (UInt256.ofNat 1)
  let v25 := IntTy.shl (.u 64) v23 v24
  let v26 := v1.l2
  Small_double_4_k1 v0 v1 v6 v14 v22 v25 v26

def Small_row2_4_k1 (v0 : Small_4) (v10 : UInt256) (v17 : UInt256) (v23 : UInt256) (v25 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 := v0.m.l0
  let v30 := (UInt256.ofNat 0)
  let v31 ← mac v28 v29 v10 v30
  let v32 := v31.snd
  let v33 := v32
  let v34 := v0.m.l1
  let v35 ← mac v28 v34 v17 v33
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := (UInt256.ofNat 0)
  let v41 ← adc v23 v40 v39
  let v42 := v41.fst
  let v43 := v42
  let v44 := v41.snd
  let v45 := v44
  let v46 := (UInt256.ofNat 0)
  let v47 := (UInt256.ofNat 0)
  let v48 := (Wide.mk v37 v43 v46 v47)
  let v49 := IntTy.addW (.u 64) v25 v45
  let v50 := (v48, v49)
  return v50

def Small_row2_4 (v0 : Small_4) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 ← adc v4 v19 v20
  let v22 := v21.fst
  let v23 := v22
  let v24 := v21.snd
  let v25 := v24
  let v26 := v0.inverse
  let v27 := IntTy.mulW (.u 64) v10 v26
  let v28 := v27
  Small_row2_4_k1 v0 v10 v17 v23 v25 v28

def Small_row3_4_k2 (v30 : UInt256) (v32 : UInt256) (v44 : UInt256) (v50 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := (UInt256.ofNat 0)
  let v54 ← adc v30 v53 v52
  let v55 := v54.fst
  let v56 := v55
  let v57 := v54.snd
  let v58 := v57
  let v59 := (UInt256.ofNat 0)
  let v60 := (Wide.mk v44 v50 v56 v59)
  let v61 := IntTy.addW (.u 64) v32 v58
  let v62 := (v60, v61)
  return v62

def Small_row3_4_k1 (v0 : Small_4) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v28 : (UInt256 × UInt256)) : FeM (Wide × UInt256) := do
  let v29 := v28.fst
  let v30 := v29
  let v31 := v28.snd
  let v32 := v31
  let v33 := v0.inverse
  let v34 := IntTy.mulW (.u 64) v10 v33
  let v35 := v34
  let v36 := v0.m.l0
  let v37 := (UInt256.ofNat 0)
  let v38 ← mac v35 v36 v10 v37
  let v39 := v38.snd
  let v40 := v39
  let v41 := v0.m.l1
  let v42 ← mac v35 v41 v17 v40
  let v43 := v42.fst
  let v44 := v43
  let v45 := v42.snd
  let v46 := v45
  let v47 := v0.m.l2
  let v48 ← mac v35 v47 v24 v46
  let v49 := v48.fst
  let v50 := v49
  let v51 := v48.snd
  let v52 := v51
  Small_row3_4_k2 v30 v32 v44 v50 v52

def Small_row3_4 (v0 : Small_4) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := (UInt256.ofNat 0)
  let v28 ← adc v4 v26 v27
  Small_row3_4_k1 v0 v10 v17 v24 v28

def Small_row4_4_k2 (v0 : Small_4) (v24 : UInt256) (v31 : UInt256) (v37 : UInt256) (v39 : UInt256) (v42 : UInt256) (v51 : UInt256) (v52 : UInt256) : FeM (Wide × UInt256) := do
  let v53 := v52
  let v54 := v0.m.l2
  let v55 ← mac v42 v54 v24 v53
  let v56 := v55.fst
  let v57 := v56
  let v58 := v55.snd
  let v59 := v58
  let v60 := v0.m.l3
  let v61 ← mac v42 v60 v31 v59
  let v62 := v61.fst
  let v63 := v62
  let v64 := v61.snd
  let v65 := v64
  let v66 := (UInt256.ofNat 0)
  let v67 ← adc v37 v66 v65
  let v68 := v67.fst
  let v69 := v68
  let v70 := v67.snd
  let v71 := v70
  let v72 := (Wide.mk v51 v57 v63 v69)
  let v73 := IntTy.addW (.u 64) v39 v71
  let v74 := (v72, v73)
  return v74

def Small_row4_4_k1 (v0 : Small_4) (v2 : UInt256) (v4 : UInt256) (v10 : UInt256) (v17 : UInt256) (v24 : UInt256) (v26 : UInt256) (v27 : UInt256) (v28 : UInt256) : FeM (Wide × UInt256) := do
  let v29 ← mac v27 v2 v28 v26
  let v30 := v29.fst
  let v31 := v30
  let v32 := v29.snd
  let v33 := v32
  let v34 := (UInt256.ofNat 0)
  let v35 ← adc v4 v33 v34
  let v36 := v35.fst
  let v37 := v36
  let v38 := v35.snd
  let v39 := v38
  let v40 := v0.inverse
  let v41 := IntTy.mulW (.u 64) v10 v40
  let v42 := v41
  let v43 := v0.m.l0
  let v44 := (UInt256.ofNat 0)
  let v45 ← mac v42 v43 v10 v44
  let v46 := v45.snd
  let v47 := v46
  let v48 := v0.m.l1
  let v49 ← mac v42 v48 v17 v47
  let v50 := v49.fst
  let v51 := v50
  let v52 := v49.snd
  Small_row4_4_k2 v0 v24 v31 v37 v39 v42 v51 v52

def Small_row4_4 (v0 : Small_4) (v1 : Wide) (v2 : UInt256) (v3 : Wide) (v4 : UInt256) : FeM (Wide × UInt256) := do
  let v5 := v1.l0
  let v6 := v3.l0
  let v7 := (UInt256.ofNat 0)
  let v8 ← mac v5 v2 v6 v7
  let v9 := v8.fst
  let v10 := v9
  let v11 := v8.snd
  let v12 := v11
  let v13 := v1.l1
  let v14 := v3.l1
  let v15 ← mac v13 v2 v14 v12
  let v16 := v15.fst
  let v17 := v16
  let v18 := v15.snd
  let v19 := v18
  let v20 := v1.l2
  let v21 := v3.l2
  let v22 ← mac v20 v2 v21 v19
  let v23 := v22.fst
  let v24 := v23
  let v25 := v22.snd
  let v26 := v25
  let v27 := v1.l3
  let v28 := v3.l3
  Small_row4_4_k1 v0 v2 v4 v10 v17 v24 v26 v27 v28

def Small_mul_4 (v0 : Small_4) (v1 : Wide) (v2 : Wide) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Wide := default
  let mut r2 : Wide := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let mut r6 : Wide := default
  let v8 := (UInt256.ofNat 0)
  let v9 := (UInt256.ofNat 0)
  let v10 := (UInt256.ofNat 0)
  let v11 := (UInt256.ofNat 0)
  let v12 := (Wide.mk v8 v9 v10 v11)
  let v13 := v12
  r0 := v13
  let v14 := (UInt256.ofNat 4)
  let v15 := (UInt256.ofNat 1)
  let v16 := decide (v14 = v15)
  if v16 then
    let v17 := v1.l0
    let v18 := v17
    let v19 := v2.l0
    let v20 := v19
    let v21 := IntTy.mulW (.u 128) v18 v20
    let v22 := v21
    let v23 ← downcast_truncate_u128_u64 v22
    let v24 := v23
    let v25 := v0.inverse
    let v26 := IntTy.mulW (.u 64) v24 v25
    let v27 := v26
    let v28 := v27
    let v29 := v0.m.l0
    let v30 := v29
    let v31 := IntTy.mulW (.u 128) v28 v30
    let v32 := v31
    let v33 := (UInt256.ofNat 64)
    let v34 := v22 >>> v33
    let v35 := (UInt256.ofNat 64)
    let v36 := v32 >>> v35
    let v37 := IntTy.addW (.u 128) v34 v36
    let v38 := (UInt256.ofNat 0)
    let v39 := decide (v24 ≠ v38)
    let b6 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v3 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let v44 := IntTy.addW (.u 128) v37 v3
      let v45 := v44
      let v46 := v0.m.l0
      let v47 := v46
      let v48 := decide (v47.toNat ≤ v45.toNat)
      let b9 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let v54 := v4
        let v55 ← downcast_truncate_u128_u64 v54
        let v56 := (UInt256.ofNat 0)
        let v57 := (UInt256.ofNat 0)
        let v58 := (UInt256.ofNat 0)
        let v59 := (Wide.mk v55 v56 v57 v58)
        return v59
      if v48 then
        let v49 := v0.m.l0
        let v50 := v49
        let v51 := IntTy.subW (.u 128) v45 v50
        let v52 := v51
        b9 r0 r1 r2 r3 r4 r5 r6 v52
      else
        let v53 := v45
        b9 r0 r1 r2 r3 r4 r5 r6 v53
    if v39 then
      let v40 := (UInt256.ofNat 1)
      let v41 := v40
      b6 r0 r1 r2 r3 r4 r5 r6 v41
    else
      let v42 := (UInt256.ofNat 0)
      let v43 := v42
      b6 r0 r1 r2 r3 r4 r5 r6 v43
  else
    let v60 := ()
    let v61 := v60
    let v62 := (UInt256.ofNat 4)
    let v63 := (UInt256.ofNat 2)
    let v64 := decide (v62 = v63)
    if v64 then
      let v65 := v2.l0
      let v66 := (UInt256.ofNat 0)
      let v68 ← Small_row2_4 v0 v1 v65 r0 v66
      let v69 := v68.fst
      let v70 := v69
      r1 := v70
      let v71 := v68.snd
      let v72 := v71
      let v73 := v2.l1
      let v75 ← Small_row2_4 v0 v1 v73 r1 v72
      let v76 := v75.fst
      let v77 := v76
      let v78 := v75.snd
      let v79 := v78
      let v80 := v77.l0
      let v81 := v0.m.l0
      let v82 := (UInt256.ofNat 0)
      let v83 ← sbb v80 v81 v82
      let v84 := v83.fst
      let v85 := v84
      let v86 := v83.snd
      let v87 := v86
      let v88 := v77.l1
      let v89 := v0.m.l1
      let v90 ← sbb v88 v89 v87
      let v91 := v90.fst
      let v92 := v91
      let v93 := v90.snd
      let v94 := v93
      let v95 := (UInt256.ofNat 0)
      let v96 ← sbb v79 v95 v94
      let v97 := v96.snd
      let v98 := v97
      let v99 := (UInt256.ofNat 1)
      let v100 := decide (v98 = v99)
      let b15 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        return v5
      if v100 then
        let v101 := v77
        b15 r0 r1 r2 r3 r4 r5 r6 v101
      else
        let v102 := (UInt256.ofNat 0)
        let v103 := (UInt256.ofNat 0)
        let v104 := (Wide.mk v85 v92 v102 v103)
        let v105 := v104
        b15 r0 r1 r2 r3 r4 r5 r6 v105
    else
      let v106 := ()
      let v107 := v106
      let v108 := (UInt256.ofNat 4)
      let v109 := (UInt256.ofNat 3)
      let v110 := decide (v108 = v109)
      if v110 then
        let v111 := v2.l0
        let v112 := (UInt256.ofNat 0)
        let v114 ← Small_row3_4 v0 v1 v111 r0 v112
        let v115 := v114.fst
        let v116 := v115
        r2 := v116
        let v117 := v114.snd
        let v118 := v117
        let v119 := v2.l1
        let v121 ← Small_row3_4 v0 v1 v119 r2 v118
        let v122 := v121.fst
        let v123 := v122
        r3 := v123
        let v124 := v121.snd
        let v125 := v124
        let v126 := v2.l2
        let v128 ← Small_row3_4 v0 v1 v126 r3 v125
        let v129 := v128.fst
        let v130 := v129
        let v131 := v128.snd
        let v132 := v131
        let v133 := v130.l0
        let v134 := v0.m.l0
        let v135 := (UInt256.ofNat 0)
        let v136 ← sbb v133 v134 v135
        let v137 := v136.fst
        let v138 := v137
        let v139 := v136.snd
        let v140 := v139
        let v141 := v130.l1
        let v142 := v0.m.l1
        let v143 ← sbb v141 v142 v140
        let v144 := v143.fst
        let v145 := v144
        let v146 := v143.snd
        let v147 := v146
        let v148 := v130.l2
        let v149 := v0.m.l2
        let v150 ← sbb v148 v149 v147
        let v151 := v150.fst
        let v152 := v151
        let v153 := v150.snd
        let v154 := v153
        let v155 := (UInt256.ofNat 0)
        let v156 ← sbb v132 v155 v154
        let v157 := v156.snd
        let v158 := v157
        let v159 := (UInt256.ofNat 1)
        let v160 := decide (v158 = v159)
        let b21 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v6
        if v160 then
          let v161 := v130
          b21 r0 r1 r2 r3 r4 r5 r6 v161
        else
          let v162 := (UInt256.ofNat 0)
          let v163 := (Wide.mk v138 v145 v152 v162)
          let v164 := v163
          b21 r0 r1 r2 r3 r4 r5 r6 v164
      else
        let v165 := ()
        let v166 := v165
        let v167 := v2.l0
        let v168 := (UInt256.ofNat 0)
        let v170 ← Small_row4_4 v0 v1 v167 r0 v168
        let v171 := v170.fst
        let v172 := v171
        r4 := v172
        let v173 := v170.snd
        let v174 := v173
        let v175 := v2.l1
        let v177 ← Small_row4_4 v0 v1 v175 r4 v174
        let v178 := v177.fst
        let v179 := v178
        r5 := v179
        let v180 := v177.snd
        let v181 := v180
        let v182 := v2.l2
        let v184 ← Small_row4_4 v0 v1 v182 r5 v181
        let v185 := v184.fst
        let v186 := v185
        r6 := v186
        let v187 := v184.snd
        let v188 := v187
        let v189 := v2.l3
        let v191 ← Small_row4_4 v0 v1 v189 r6 v188
        let v192 := v191.fst
        let v193 := v192
        let v194 := v191.snd
        let v195 := v194
        let v196 := v193.l0
        let v197 := v0.m.l0
        let v198 := (UInt256.ofNat 0)
        let v199 ← sbb v196 v197 v198
        let v200 := v199.fst
        let v201 := v200
        let v202 := v199.snd
        let v203 := v202
        let v204 := v193.l1
        let v205 := v0.m.l1
        let v206 ← sbb v204 v205 v203
        let v207 := v206.fst
        let v208 := v207
        let v209 := v206.snd
        let v210 := v209
        let v211 := v193.l2
        let v212 := v0.m.l2
        let v213 ← sbb v211 v212 v210
        let v214 := v213.fst
        let v215 := v214
        let v216 := v213.snd
        let v217 := v216
        let v218 := v193.l3
        let v219 := v0.m.l3
        let v220 ← sbb v218 v219 v217
        let v221 := v220.fst
        let v222 := v221
        let v223 := v220.snd
        let v224 := v223
        let v225 := (UInt256.ofNat 0)
        let v226 ← sbb v195 v225 v224
        let v227 := v226.snd
        let v228 := v227
        let v229 := (UInt256.ofNat 1)
        let v230 := decide (v228 = v229)
        let b24 : Wide → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Wide := fun r0 r1 r2 r3 r4 r5 r6 v7 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          return v7
        if v230 then
          let v231 := v193
          b24 r0 r1 r2 r3 r4 r5 r6 v231
        else
          let v232 := (Wide.mk v201 v208 v215 v222)
          let v233 := v232
          b24 r0 r1 r2 r3 r4 r5 r6 v233

def Small_r2_4 (v0 : Small_4) : FeM Wide := do
  let mut r0 : Wide := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Wide := default
  let mut r3 : Range_Known_0_Known_6 := default
  let mut r4 : Wide := default
  let mut r5 : Wide := default
  let v10 := (UInt256.ofNat 4)
  let v11 := (UInt256.ofNat 1)
  let v12 := decide (v10 = v11)
  let b3 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let v29 := v1
    let v30 := (UInt256.ofNat 255)
    let v31 := v29
    let v32 := IntTy.leadingZeros (.u 256) v31
    let v33 := v30 - v32
    let v34 ← downcast_truncate_u256_u64 v33
    let v35 := v34
    let v36 := (UInt256.ofNat 1)
    let v37 := IntTy.shl (.u 64) v36 v35
    let v38 := v37
    let v39 := (UInt256.ofNat 4)
    let v40 := (UInt256.ofNat 1)
    let v41 := decide (v39 = v40)
    let b12 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v4 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v45 := (UInt256.ofNat 4)
      let v46 := (UInt256.ofNat 2)
      let v47 := decide (v45 = v46)
      let b15 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v51 := (UInt256.ofNat 4)
        let v52 := (UInt256.ofNat 3)
        let v53 := decide (v51 = v52)
        let b18 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v6 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v57 := (UInt256.ofNat 4)
          let v58 := (UInt256.ofNat 4)
          let v59 := decide (v57 = v58)
          let b21 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v7 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v63 := (Wide.mk v4 v5 v6 v7)
            let v64 := v63
            r0 := v64
            let v65 := (UInt256.ofNat 64)
            let v66 := IntTy.subW (.u 64) v65 v35
            let v67 ← downcast_truncate_u64_usize v66
            let v68 := v67
            let v69 := (UInt256.ofNat 0)
            let v70 := (UInt256.ofNat 4)
            let v71 := v68 + v70
            let v72 := ()
            let v73 := (Range_Known_0_Unknown.mk v72 v71)
            r1 := v73
            let v75 ← Range_start_0 r1
            let v76 := v75
            let rec b22 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v8 : UInt256) : FeM Wide := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let v78 ← Range_has_0 r1 v8
              if v78 then
                let v80 ← Range_at_0 r1 v8
                let v81 := r0
                let v82 := v81
                r2 := v82
                let v84 ← Small_double_4 v0 r2
                let v85 := v84
                r0 := v85
                let v86 := ()
                let v88 ← Range_next_0 r1 v8
                let v89 := v88
                b22 r0 r1 r2 r3 r4 r5 v89
              else
                let v90 := (UInt256.ofNat 0)
                let v91 := (UInt256.ofNat 6)
                let v92 := ()
                let v93 := (Range_Known_0_Known_6.mk v92 v92)
                r3 := v93
                let v95 ← Range_start_0_6 r3
                let v96 := v95
                let rec b26 (r0 : Wide) (r1 : Range_Known_0_Unknown) (r2 : Wide) (r3 : Range_Known_0_Known_6) (r4 : Wide) (r5 : Wide) (v9 : UInt256) : FeM Wide := do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v98 ← Range_has_0_6 r3 v9
                  if v98 then
                    let v100 ← Range_at_0_6 r3 v9
                    let v101 := r0
                    let v102 := v101
                    r4 := v102
                    let v103 := r0
                    let v104 := v103
                    r5 := v104
                    let v107 ← Small_mul_4 v0 r4 r5
                    let v108 := v107
                    r0 := v108
                    let v109 := ()
                    let v111 ← Range_next_0_6 r3 v9
                    let v112 := v111
                    b26 r0 r1 r2 r3 r4 r5 v112
                  else
                    let v113 := r0
                    return v113
                partial_fixpoint
                b26 r0 r1 r2 r3 r4 r5 v96
            partial_fixpoint
            b22 r0 r1 r2 r3 r4 r5 v76
          if v59 then
            let v60 := v38
            b21 r0 r1 r2 r3 r4 r5 v60
          else
            let v61 := (UInt256.ofNat 0)
            let v62 := v61
            b21 r0 r1 r2 r3 r4 r5 v62
        if v53 then
          let v54 := v38
          b18 r0 r1 r2 r3 r4 r5 v54
        else
          let v55 := (UInt256.ofNat 0)
          let v56 := v55
          b18 r0 r1 r2 r3 r4 r5 v56
      if v47 then
        let v48 := v38
        b15 r0 r1 r2 r3 r4 r5 v48
      else
        let v49 := (UInt256.ofNat 0)
        let v50 := v49
        b15 r0 r1 r2 r3 r4 r5 v50
    if v41 then
      let v42 := v38
      b12 r0 r1 r2 r3 r4 r5 v42
    else
      let v43 := (UInt256.ofNat 0)
      let v44 := v43
      b12 r0 r1 r2 r3 r4 r5 v44
  if v12 then
    let v13 := v0.m.l0
    let v14 := v13
    b3 r0 r1 r2 r3 r4 r5 v14
  else
    let v15 := (UInt256.ofNat 4)
    let v16 := (UInt256.ofNat 2)
    let v17 := decide (v15 = v16)
    let b6 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v2 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v28 := v2
      b3 r0 r1 r2 r3 r4 r5 v28
    if v17 then
      let v18 := v0.m.l1
      let v19 := v18
      b6 r0 r1 r2 r3 r4 r5 v19
    else
      let v20 := (UInt256.ofNat 4)
      let v21 := (UInt256.ofNat 3)
      let v22 := decide (v20 = v21)
      let b9 : Wide → Range_Known_0_Unknown → Wide → Range_Known_0_Known_6 → Wide → Wide → UInt256 → FeM Wide := fun r0 r1 r2 r3 r4 r5 v3 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v27 := v3
        b6 r0 r1 r2 r3 r4 r5 v27
      if v22 then
        let v23 := v0.m.l2
        let v24 := v23
        b9 r0 r1 r2 r3 r4 r5 v24
      else
        let v25 := v0.m.l3
        let v26 := v25
        b9 r0 r1 r2 r3 r4 r5 v26

def small_power_4 (v0 : Exponent) (v1 : MemArray_u64) (v2 : MemArray_u64) : FeM Option_MemArray_u64 := do
  let mut r0 : UInt256 := default
  let mut r1 : Small_4 := default
  let mut r2 : (Vector Wide 64) := default
  let mut r3 : Wide := default
  let mut r4 : Wide := default
  let mut r5 : UInt256 := default
  let mut r6 : Bool := default
  let mut r7 : MemArray_u64 := default
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
  let v8 ← MemArray_index_ref_u64 v2 v7
  let v9 ← mloadInt (.u 64) v8
  let v10 := v9
  r0 := v10
  let v11 := (UInt256.ofNat 0)
  let v12 := (UInt256.ofNat 5)
  let v13 := ()
  let v14 := (Range_Known_0_Known_5.mk v13 v13)
  r8 := v14
  let v16 ← Range_start_0_5 r8
  let v17 := v16
  let rec b1 (r0 : UInt256) (r1 : Small_4) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v3 : UInt256) : FeM Option_MemArray_u64 := do
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
    let v19 ← Range_has_0_5 r8 v3
    if v19 then
      let v21 ← Range_at_0_5 r8 v3
      let v22 := r0
      let v23 := v22
      let v24 := (UInt256.ofNat 2)
      let v25 := r0
      let v26 := IntTy.mulW (.u 64) v9 v25
      let v27 := IntTy.subW (.u 64) v24 v26
      let v28 := IntTy.mulW (.u 64) v23 v27
      let v29 := v28
      r0 := v29
      let v30 := ()
      let v32 ← Range_next_0_5 r8 v3
      let v33 := v32
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v33
    else
      let v34 := (UInt256.ofNat 4)
      let v35 ← Wide_of v2 v34
      let v36 := (UInt256.ofNat 0)
      let v37 := r0
      let v38 := IntTy.subW (.u 64) v36 v37
      let v39 := (Small_4.mk v35 v38)
      let v40 := v39
      r1 := v40
      let v41 := v0.total
      let v42 := v0.top
      let v43 := v41 - v42
      let v44 ← window_width v43
      let v45 := v44
      let v46 := (UInt256.ofNat 1)
      let v47 := (UInt256.ofNat 1)
      let v48 := v45 - v47
      let v49 := v46 <<< v48
      let v50 ← downcast_truncate_usize_usize v49
      let v51 := v50
      let v52 := (UInt256.ofNat 0)
      let v53 := (UInt256.ofNat 0)
      let v54 := (UInt256.ofNat 0)
      let v55 := (UInt256.ofNat 0)
      let v56 := (Wide.mk v52 v53 v54 v55)
      let v57 := Vector.replicate 64 v56
      let v58 := v57
      r2 := v58
      let v59 := (UInt256.ofNat 4)
      let v60 ← Wide_of v1 v59
      r9 := v60
      let v62 ← Small_r2_4 r1
      r10 := v62
      let v66 ← Small_mul_4 r1 r9 r10
      let v67 := (UInt256.ofNat 0)
      checkIndex v67 64
      r2 := (r2.set! v67.toNat v66)
      let v68 := ()
      let v69 := (UInt256.ofNat 1)
      let v70 := decide (v69.toNat < v51.toNat)
      let b7 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
        let v106 := (UInt256.ofNat 0)
        checkIndex v106 64
        let v107 := r2[v106.toNat]!
        let v108 := v107
        r4 := v108
        let v109 := v0.top
        let v110 := v109
        r5 := v110
        let v111 := false
        let v112 := v111
        r6 := v112
        let rec b12 (r0 : UInt256) (r1 : Small_4) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
          let v113 := r5
          let v114 := v113
          let v115 := v0.total
          let v116 := decide (v114.toNat < v115.toNat)
          if v116 then
            let v117 := r5
            let v118 := v117
            let v119 ← Exponent_window v0 v118 v45
            let v120 := v119
            let v121 := r6
            let rec b17 (r0 : UInt256) (r1 : Small_4) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) : FeM Option_MemArray_u64 := do
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
              let v151 := v120.odd
              let b24 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v173 := v120.«end»
                let v174 := v173
                r5 := v174
                let v175 := ()
                b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v151 then
                let v152 := r6
                let b27 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v6 => do
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
                  let v165 := v6
                  r4 := v165
                  let v166 := ()
                  let v167 := true
                  let v168 := v167
                  r6 := v168
                  let v169 := ()
                  let v170 := v169
                  b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v152 then
                  let v153 := r4
                  let v154 := v153
                  r18 := v154
                  let v155 := v120.entry
                  checkIndex v155 64
                  let v156 := r2[v155.toNat]!
                  r19 := v156
                  let v160 ← Small_mul_4 r1 r18 r19
                  let v161 := v160
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v161
                else
                  let v162 := v120.entry
                  checkIndex v162 64
                  let v163 := r2[v162.toNat]!
                  let v164 := v163
                  b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v164
              else
                let v171 := ()
                let v172 := v171
                b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            partial_fixpoint
            if v121 then
              let v122 := r5
              let v123 := v122
              let v124 := v120.«end»
              let v125 := ()
              let v126 := (Range_Unknown_Unknown.mk v123 v124)
              r15 := v126
              let v128 ← Range_Unknown_Unknown_start r15
              let v129 := v128
              let rec b18 (r0 : UInt256) (r1 : Small_4) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
                let v131 ← Range_Unknown_Unknown_has r15 v5
                if v131 then
                  let v133 ← Range_Unknown_Unknown_at r15 v5
                  let v134 := r4
                  let v135 := v134
                  r16 := v135
                  let v136 := r4
                  let v137 := v136
                  r17 := v137
                  let v141 ← Small_mul_4 r1 r16 r17
                  let v142 := v141
                  r4 := v142
                  let v143 := ()
                  let v145 ← Range_Unknown_Unknown_next r15 v5
                  let v146 := v145
                  b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v146
                else
                  let v147 := ()
                  let v148 := v147
                  b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              partial_fixpoint
              b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v129
            else
              let v149 := ()
              let v150 := v149
              b17 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
          else
            let v176 := r4
            let v177 := v176
            r20 := v177
            let v178 := (UInt256.ofNat 1)
            let v179 := (UInt256.ofNat 0)
            let v180 := (UInt256.ofNat 0)
            let v181 := (UInt256.ofNat 0)
            let v182 := (Wide.mk v178 v179 v180 v181)
            r21 := v182
            let v186 ← Small_mul_4 r1 r20 r21
            let v187 := v186
            let v188 := (UInt256.ofNat 4)
            let v189 ← MemArray_try_new_zeroed_u64 v188
            let v190 := match v189 with | .Some .. => true | _ => false
            if v190 then
              let v191 := v189.Some_0
              let v192 := v191
              r7 := v192
              let v193 := v187.l0
              let v194 := (UInt256.ofNat 0)
              let (v196, v196_k) ← MemArray_index_mut_u64 r7 v194
              mstoreInt (.u 64) v196 v193
              let v196_0 ← v196_k v196
              r7 := v196_0
              let v197 := ()
              let v198 := (UInt256.ofNat 4)
              let v199 := (UInt256.ofNat 1)
              let v200 := decide (v199.toNat < v198.toNat)
              let b32 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                let v209 := (UInt256.ofNat 4)
                let v210 := (UInt256.ofNat 2)
                let v211 := decide (v210.toNat < v209.toNat)
                let b35 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                  let v220 := (UInt256.ofNat 4)
                  let v221 := (UInt256.ofNat 3)
                  let v222 := decide (v221.toNat < v220.toNat)
                  let b38 : UInt256 → Small_4 → (Vector Wide 64) → Wide → Wide → UInt256 → Bool → MemArray_u64 → Range_Known_0_Known_5 → Wide → Wide → Wide → Wide → Range_Known_1_Unknown → Wide → Range_Unknown_Unknown → Wide → Wide → Wide → Wide → Wide → Wide → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 => do
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
                    let v231 := r7
                    let v232 := v231
                    let v233 := Option_MemArray_u64.Some v232
                    return v233
                  if v222 then
                    let v223 := v187.l3
                    let v224 := (UInt256.ofNat 3)
                    let (v226, v226_k) ← MemArray_index_mut_u64 r7 v224
                    mstoreInt (.u 64) v226 v223
                    let v226_0 ← v226_k v226
                    r7 := v226_0
                    let v227 := ()
                    let v228 := v227
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                  else
                    let v229 := ()
                    let v230 := v229
                    b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                if v211 then
                  let v212 := v187.l2
                  let v213 := (UInt256.ofNat 2)
                  let (v215, v215_k) ← MemArray_index_mut_u64 r7 v213
                  mstoreInt (.u 64) v215 v212
                  let v215_0 ← v215_k v215
                  r7 := v215_0
                  let v216 := ()
                  let v217 := v216
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
                else
                  let v218 := ()
                  let v219 := v218
                  b35 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              if v200 then
                let v201 := v187.l1
                let v202 := (UInt256.ofNat 1)
                let (v204, v204_k) ← MemArray_index_mut_u64 r7 v202
                mstoreInt (.u 64) v204 v201
                let v204_0 ← v204_k v204
                r7 := v204_0
                let v205 := ()
                let v206 := v205
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
              else
                let v207 := ()
                let v208 := v207
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
            else
              let v234 := Option_MemArray_u64.None
              match (← pure v189) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v189.Some_0
                pure ()
              | _ => pure ()
              return v234
        partial_fixpoint
        b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
      if v70 then
        let v71 := (UInt256.ofNat 0)
        checkIndex v71 64
        let v72 := r2[v71.toNat]!
        r11 := v72
        let v73 := (UInt256.ofNat 0)
        checkIndex v73 64
        let v74 := r2[v73.toNat]!
        r12 := v74
        let v78 ← Small_mul_4 r1 r11 r12
        let v79 := v78
        r3 := v79
        let v80 := (UInt256.ofNat 1)
        let v81 := ()
        let v82 := (Range_Known_1_Unknown.mk v81 v51)
        r13 := v82
        let v84 ← Range_start_1 r13
        let v85 := v84
        let rec b8 (r0 : UInt256) (r1 : Small_4) (r2 : (Vector Wide 64)) (r3 : Wide) (r4 : Wide) (r5 : UInt256) (r6 : Bool) (r7 : MemArray_u64) (r8 : Range_Known_0_Known_5) (r9 : Wide) (r10 : Wide) (r11 : Wide) (r12 : Wide) (r13 : Range_Known_1_Unknown) (r14 : Wide) (r15 : Range_Unknown_Unknown) (r16 : Wide) (r17 : Wide) (r18 : Wide) (r19 : Wide) (r20 : Wide) (r21 : Wide) (v4 : UInt256) : FeM Option_MemArray_u64 := do
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
          let v87 ← Range_has_1 r13 v4
          if v87 then
            let v89 ← Range_at_1 r13 v4
            let v90 := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := v90 - v91
            checkIndex v92 64
            let v93 := r2[v92.toNat]!
            r14 := v93
            let v97 ← Small_mul_4 r1 r14 r3
            checkIndex v90 64
            r2 := (r2.set! v90.toNat v97)
            let v98 := ()
            let v100 ← Range_next_1 r13 v4
            let v101 := v100
            b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v101
          else
            let v102 := ()
            let v103 := v102
            b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
        partial_fixpoint
        b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v85
      else
        let v104 := ()
        let v105 := v104
        b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 v17

def small_power_of (v0 : Exponent) (v1 : MemArray_u64) (v2 : MemArray_u64) (v3 : UInt256) : FeM Option_MemArray_u64 := do
  let v7 := v3
  let v8 := (UInt256.ofNat 1)
  let v9 := decide (v7 = v8)
  let b3 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v4 => do
    return v4
  if v9 then
    let v10 ← small_power_1 v0 v1 v2
    let v11 := v10
    b3 v11
  else
    let v12 := v3
    let v13 := (UInt256.ofNat 2)
    let v14 := decide (v12 = v13)
    let b6 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v5 => do
      let v25 := v5
      b3 v25
    if v14 then
      let v15 ← small_power_2 v0 v1 v2
      let v16 := v15
      b6 v16
    else
      let v17 := v3
      let v18 := (UInt256.ofNat 3)
      let v19 := decide (v17 = v18)
      let b9 : Option_MemArray_u64 → FeM Option_MemArray_u64 := fun v6 => do
        let v24 := v6
        b6 v24
      if v19 then
        let v20 ← small_power_3 v0 v1 v2
        let v21 := v20
        b9 v21
      else
        let v22 ← small_power_4 v0 v1 v2
        let v23 := v22
        b9 v23

def Montgomery_new (r0 : MemArray_u64) (v0 : UInt256) : FeM Montgomery := do
  let mut r0 := r0
  let mut r1 : UInt256 := default
  let mut r2 : Range_Known_0_Known_5 := default
  let v2 := (UInt256.ofNat 0)
  let v4 ← MemArray_index_ref_u64 r0 v2
  let v5 ← mloadInt (.u 64) v4
  let v6 := v5
  r1 := v6
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 5)
  let v9 := ()
  let v10 := (Range_Known_0_Known_5.mk v9 v9)
  r2 := v10
  let v12 ← Range_start_0_5 r2
  let v13 := v12
  let rec b1 (r0 : MemArray_u64) (r1 : UInt256) (r2 : Range_Known_0_Known_5) (v1 : UInt256) : FeM Montgomery := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v15 ← Range_has_0_5 r2 v1
    if v15 then
      let v17 ← Range_at_0_5 r2 v1
      let v18 := r1
      let v19 := v18
      let v20 := (UInt256.ofNat 2)
      let v21 := r1
      let v22 := IntTy.mulW (.u 64) v5 v21
      let v23 := IntTy.subW (.u 64) v20 v22
      let v24 := IntTy.mulW (.u 64) v19 v23
      let v25 := v24
      r1 := v25
      let v26 := ()
      let v28 ← Range_next_0_5 r2 v1
      let v29 := v28
      b1 r0 r1 r2 v29
    else
      let v30 := r0
      let v31 := v30
      let v32 := v0
      let v33 := (UInt256.ofNat 0)
      let v34 := r1
      let v35 := IntTy.subW (.u 64) v33 v34
      let v36 := (Montgomery.mk v31 v32 v35)
      return v36
  partial_fixpoint
  b1 r0 r1 r2 v13

def rem_2by1 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let v4 := v3
  let v5 := v4
  let v6 := v0
  let v7 := v6
  let v8 := IntTy.mulW (.u 128) v5 v7
  let v9 := v0
  let v10 := v9
  let v11 := (UInt256.ofNat 64)
  let v12 := IntTy.shl (.u 128) v10 v11
  let v13 := v1
  let v14 := v13
  let v15 := v12 ||| v14
  let v16 := IntTy.addW (.u 128) v8 v15
  let v17 := v16
  let v18 := (UInt256.ofNat 64)
  let v19 := v17 >>> v18
  let v20 ← downcast_truncate_u128_u64 v19
  let v21 := v20
  let v22 := (UInt256.ofNat 1)
  let v23 := IntTy.addW (.u 64) v21 v22
  let v24 := v23
  let v25 ← downcast_truncate_u128_u64 v17
  let v26 := v25
  let v27 := v1
  let v28 := v2
  let v29 := IntTy.mulW (.u 64) v24 v28
  let v30 := IntTy.subW (.u 64) v27 v29
  let v31 := v30
  r0 := v31
  let v32 := r0
  let v33 := v32
  let v34 := decide (v26.toNat < v33.toNat)
  let b3 : UInt256 → FeM UInt256 := fun r0 => do
    let mut r0 := r0
    let v43 := r0
    let v44 := v43
    let v45 := v2
    let v46 := decide (v45.toNat ≤ v44.toNat)
    let b6 : UInt256 → FeM UInt256 := fun r0 => do
      let mut r0 := r0
      let v55 := r0
      return v55
    if v46 then
      let v47 := v2
      let v48 := r0
      let v49 := IntTy.subW (.u 64) v48 v47
      let v50 := v49
      r0 := v50
      let v51 := ()
      let v52 := v51
      b6 r0
    else
      let v53 := ()
      let v54 := v53
      b6 r0
  if v34 then
    let v35 := v2
    let v36 := r0
    let v37 := IntTy.addW (.u 64) v36 v35
    let v38 := v37
    r0 := v38
    let v39 := ()
    let v40 := v39
    b3 r0
  else
    let v41 := ()
    let v42 := v41
    b3 r0

def div_3by2_k2 (v25 : UInt256) (v40 : UInt256) (v52 : UInt256) (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  let mut r1 := r1
  let v53 := IntTy.subW (.u 128) v52 v40
  let v54 := v53
  r1 := v54
  let v55 := (UInt256.ofNat 1)
  let v56 := r0
  let v57 := IntTy.addW (.u 64) v56 v55
  let v58 := v57
  r0 := v58
  let v59 := ()
  let v60 := r1
  let v61 := v60
  let v62 := (UInt256.ofNat 64)
  let v63 := v61 >>> v62
  let v64 ← downcast_truncate_u128_u64 v63
  let v65 := v64
  let v66 := decide (v25.toNat ≤ v65.toNat)
  let b3 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
    let mut r0 := r0
    let mut r1 := r1
    let v79 := r1
    let v80 := v79
    let v81 := decide (v40.toNat ≤ v80.toNat)
    let b6 : UInt256 → UInt256 → FeM UInt256 := fun r0 r1 => do
      let mut r0 := r0
      let mut r1 := r1
      let v90 := r0
      return v90
    if v81 then
      let v82 := (UInt256.ofNat 1)
      let v83 := r0
      let v84 := IntTy.addW (.u 64) v83 v82
      let v85 := v84
      r0 := v85
      let v86 := ()
      let v87 := v86
      b6 r0 r1
    else
      let v88 := ()
      let v89 := v88
      b6 r0 r1
  if v66 then
    let v67 := (UInt256.ofNat 1)
    let v68 := r0
    let v69 := IntTy.subW (.u 64) v68 v67
    let v70 := v69
    r0 := v70
    let v71 := ()
    let v72 := r1
    let v73 := IntTy.addW (.u 128) v72 v40
    let v74 := v73
    r1 := v74
    let v75 := ()
    let v76 := v75
    b3 r0 r1
  else
    let v77 := ()
    let v78 := v77
    b3 r0 r1

def div_3by2_k1 (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v25 : UInt256) (v26 : UInt256) (v28 : UInt256) (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
  let mut r0 := r0
  let mut r1 := r1
  let v29 := v3
  let v30 := IntTy.mulW (.u 64) v28 v29
  let v31 := IntTy.subW (.u 64) v26 v30
  let v32 := v31
  let v33 := v3
  let v34 := v33
  let v35 := (UInt256.ofNat 64)
  let v36 := IntTy.shl (.u 128) v34 v35
  let v37 := v4
  let v38 := v37
  let v39 := v36 ||| v38
  let v40 := v39
  let v41 := v32
  let v42 := (UInt256.ofNat 64)
  let v43 := IntTy.shl (.u 128) v41 v42
  let v44 := v2
  let v45 := v44
  let v46 := v43 ||| v45
  let v47 := v4
  let v48 := v47
  let v49 := r0
  let v50 := v49
  let v51 := IntTy.mulW (.u 128) v48 v50
  let v52 := IntTy.subW (.u 128) v46 v51
  div_3by2_k2 v25 v40 v52 r0 r1

def div_3by2 (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) (v4 : UInt256) (v5 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v6 := v5
  let v7 := v6
  let v8 := v0
  let v9 := v8
  let v10 := IntTy.mulW (.u 128) v7 v9
  let v11 := v0
  let v12 := v11
  let v13 := (UInt256.ofNat 64)
  let v14 := IntTy.shl (.u 128) v12 v13
  let v15 := v1
  let v16 := v15
  let v17 := v14 ||| v16
  let v18 := IntTy.addW (.u 128) v10 v17
  let v19 := v18
  let v20 := (UInt256.ofNat 64)
  let v21 := v19 >>> v20
  let v22 ← downcast_truncate_u128_u64 v21
  let v23 := v22
  r0 := v23
  let v24 ← downcast_truncate_u128_u64 v19
  let v25 := v24
  let v26 := v1
  let v27 := r0
  let v28 := v27
  div_3by2_k1 v2 v3 v4 v25 v26 v28 r0 r1

def Modulus_reduce (v0 : Modulus) (v1 : MemArray_u64) (v2 : UInt256) : FeM (Unit × MemArray_u64) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : UInt256 := default
  let mut r4 : UInt256 := default
  let mut r5 : UInt256 := default
  let mut r6 : UInt256 := default
  let mut r7 : Range_Unknown_Unknown := default
  let mut r8 : Range_Known_0_Unknown := default
  let mut r9 : Range_Known_0_Unknown := default
  let mut r10 : Range_Known_0_Unknown := default
  let mut r11 : Range_Unknown_Unknown := default
  let mut v1 := v1
  let v11 ← downcast_truncate_u64_usize v2
  let v12 := v11
  let v13 := v0.len
  let v14 ← downcast_truncate_u64_usize v13
  let v15 := v14
  let v17 ← MemArray_len_u64 v1
  let v18 := decide (v12.toNat < v17.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v3 => do
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
    let mut v1 := v1
    let v30 ← assert v3
    let v32 ← MemArray_ptr_u64 v1
    let v33 := v32
    let v36 ← MemArray_ptr_u64 v0.limbs
    let v37 := v36
    let v38 := decide (v12.toNat < v15.toNat)
    if v38 then
      let v39 := (UInt256.ofNat 1)
      let v40 := v15 + v39
      let v41 := ()
      let v42 := (Range_Unknown_Unknown.mk v12 v40)
      r7 := v42
      let v44 ← Range_Unknown_Unknown_start r7
      let v45 := v44
      let rec b9 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) (v4 : UInt256) : FeM (Unit × MemArray_u64) := do
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
        let mut v1 := v1
        let v47 ← Range_Unknown_Unknown_has r7 v4
        if v47 then
          let v49 ← Range_Unknown_Unknown_at r7 v4
          let v50 := v49
          let v51 := (UInt256.ofNat 0)
          let v52 ← limb v33 v50
          mstoreInt (.u 64) v52 v51
          let v53 := ()
          let v55 ← Range_Unknown_Unknown_next r7 v4
          let v56 := v55
          b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v56
        else
          return ((), v1)
      partial_fixpoint
      b9 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v45
    else
      let v57 := ()
      let v58 := v57
      let v59 := v0.shift
      let v60 := v59
      let v61 := (UInt256.ofNat 0)
      let v62 ← limb v33 v12
      mstoreInt (.u 64) v62 v61
      let v63 := ()
      let v64 := (UInt256.ofNat 0)
      let v65 := decide (v60 ≠ v64)
      let b15 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 => do
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
        let mut v1 := v1
        let v111 := (UInt256.ofNat 1)
        let v112 := decide (v15 = v111)
        if v112 then
          let v113 ← mloadInt (.u 64) v37
          let v114 := v113
          let v115 := (UInt256.ofNat 0)
          let v116 := v115
          r1 := v116
          let v117 := (UInt256.ofNat 1)
          let v118 := v12 + v117
          let v119 := v118
          r2 := v119
          let rec b22 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
            let mut v1 := v1
            let v120 := r2
            let v121 := v120
            let v122 := (UInt256.ofNat 0)
            let v123 := decide (v122.toNat < v121.toNat)
            if v123 then
              let v124 := (UInt256.ofNat 1)
              let v125 := r2
              let v126 := v125 - v124
              let v127 := v126
              r2 := v127
              let v128 := ()
              let v129 := r1
              let v130 := v129
              let v131 := r2
              let v132 := v131
              let v133 ← limb v33 v132
              let v134 ← mloadInt (.u 64) v133
              let v135 := v0.reciprocal
              let v136 ← rem_2by1 v130 v134 v114 v135
              let v137 := v136
              r1 := v137
              let v138 := ()
              let v139 := (UInt256.ofNat 0)
              let v140 := r2
              let v141 := v140
              let v142 ← limb v33 v141
              mstoreInt (.u 64) v142 v139
              let v143 := ()
              b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
            else
              let v144 := r1
              let v145 := v144
              let v146 := v145 >>> v60
              mstoreInt (.u 64) v33 v146
              let v147 := ()
              return ((), v1)
          partial_fixpoint
          b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
        else
          let v148 := ()
          let v149 := v148
          let v150 := (UInt256.ofNat 1)
          let v151 := v15 - v150
          let v152 ← limb v37 v151
          let v153 ← mloadInt (.u 64) v152
          let v154 := v153
          let v155 := (UInt256.ofNat 2)
          let v156 := v15 - v155
          let v157 ← limb v37 v156
          let v158 ← mloadInt (.u 64) v157
          let v159 := v158
          let v160 := (UInt256.ofNat 1)
          let v161 := v12 + v160
          let v162 := v161 - v15
          let v163 := v162
          r3 := v163
          let rec b25 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
            let mut v1 := v1
            let v164 := r3
            let v165 := v164
            let v166 := (UInt256.ofNat 0)
            let v167 := decide (v166.toNat < v165.toNat)
            if v167 then
              let v168 := (UInt256.ofNat 1)
              let v169 := r3
              let v170 := v169 - v168
              let v171 := v170
              r3 := v171
              let v172 := ()
              let v173 := r3
              let v174 := v173
              let v175 := v174 + v15
              let v176 ← limb v33 v175
              let v177 ← mloadInt (.u 64) v176
              let v178 := v177
              let v179 := r3
              let v180 := v179
              let v181 := v180 + v15
              let v182 := (UInt256.ofNat 1)
              let v183 := v181 - v182
              let v184 ← limb v33 v183
              let v185 ← mloadInt (.u 64) v184
              let v186 := v185
              let v187 := decide (v178 = v154)
              let b30 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → UInt256 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v5 => do
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
                let mut v1 := v1
                let v201 := v5
                let v202 := (UInt256.ofNat 0)
                let v203 := v202
                r4 := v203
                let v204 := (UInt256.ofNat 0)
                let v205 := v204
                r5 := v205
                let v206 := (UInt256.ofNat 0)
                let v207 := ()
                let v208 := (Range_Known_0_Unknown.mk v207 v15)
                r8 := v208
                let v210 ← Range_start_0 r8
                let v211 := v210
                let rec b32 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                  let mut v1 := v1
                  let v213 ← Range_has_0 r8 v6
                  if v213 then
                    let v215 ← Range_at_0 r8 v6
                    let v216 := v215
                    let v217 := v201
                    let v218 ← limb v37 v216
                    let v219 ← mloadInt (.u 64) v218
                    let v220 := v219
                    let v221 := IntTy.mulW (.u 128) v217 v220
                    let v222 := r4
                    let v223 := v222
                    let v224 := IntTy.addW (.u 128) v221 v223
                    let v225 := v224
                    let v226 := (UInt256.ofNat 64)
                    let v227 := v225 >>> v226
                    let v228 ← downcast_truncate_u128_u64 v227
                    let v229 := v228
                    r4 := v229
                    let v230 := ()
                    let v231 ← downcast_truncate_u128_u64 v225
                    let v232 := v231
                    let v233 := r3
                    let v234 := v216 + v233
                    let v235 ← limb v33 v234
                    let v236 ← mloadInt (.u 64) v235
                    let v237 := v236
                    let v238 := v232
                    let v239 := IntTy.subW (.u 128) v237 v238
                    let v240 := r5
                    let v241 := v240
                    let v242 := IntTy.subW (.u 128) v239 v241
                    let v243 := v242
                    let v244 ← downcast_truncate_u128_u64 v243
                    let v245 := r3
                    let v246 := v216 + v245
                    let v247 ← limb v33 v246
                    mstoreInt (.u 64) v247 v244
                    let v248 := ()
                    let v249 := (UInt256.ofNat 127)
                    let v250 := v243 >>> v249
                    let v251 ← downcast_truncate_u128_u64 v250
                    let v252 := v251
                    r5 := v252
                    let v253 := ()
                    let v255 ← Range_next_0 r8 v6
                    let v256 := v255
                    b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v256
                  else
                    let v257 := v178
                    let v258 := r4
                    let v259 := v258
                    let v260 := IntTy.subW (.u 128) v257 v259
                    let v261 := r5
                    let v262 := v261
                    let v263 := IntTy.subW (.u 128) v260 v262
                    let v264 := v263
                    let v265 ← downcast_truncate_u128_u64 v264
                    let v266 := r3
                    let v267 := v266
                    let v268 := v267 + v15
                    let v269 ← limb v33 v268
                    mstoreInt (.u 64) v269 v265
                    let v270 := ()
                    let v271 := (UInt256.ofNat 127)
                    let v272 := v264 >>> v271
                    let v273 := (UInt256.ofNat 0)
                    let v274 := decide (v272 ≠ v273)
                    let rec b38 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
                      let mut v1 := v1
                      b25 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
                    partial_fixpoint
                    if v274 then
                      let v275 := (UInt256.ofNat 0)
                      let v276 := v275
                      r6 := v276
                      let v277 := (UInt256.ofNat 0)
                      let v278 := ()
                      let v279 := (Range_Known_0_Unknown.mk v278 v15)
                      r9 := v279
                      let v281 ← Range_start_0 r9
                      let v282 := v281
                      let rec b39 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                        let mut v1 := v1
                        let v284 ← Range_has_0 r9 v7
                        if v284 then
                          let v286 ← Range_at_0 r9 v7
                          let v287 := v286
                          let v288 := r3
                          let v289 := v287 + v288
                          let v290 ← limb v33 v289
                          let v291 ← mloadInt (.u 64) v290
                          let v292 := v291
                          let v293 ← limb v37 v287
                          let v294 ← mloadInt (.u 64) v293
                          let v295 := v294
                          let v296 := IntTy.addW (.u 128) v292 v295
                          let v297 := r6
                          let v298 := v297
                          let v299 := IntTy.addW (.u 128) v296 v298
                          let v300 := v299
                          let v301 ← downcast_truncate_u128_u64 v300
                          let v302 := r3
                          let v303 := v287 + v302
                          let v304 ← limb v33 v303
                          mstoreInt (.u 64) v304 v301
                          let v305 := ()
                          let v306 := (UInt256.ofNat 64)
                          let v307 := v300 >>> v306
                          let v308 ← downcast_truncate_u128_u64 v307
                          let v309 := v308
                          r6 := v309
                          let v310 := ()
                          let v312 ← Range_next_0 r9 v7
                          let v313 := v312
                          b39 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v313
                        else
                          let v314 := r3
                          let v315 := v314
                          let v316 := v315 + v15
                          let v317 ← limb v33 v316
                          let v318 ← mloadInt (.u 64) v317
                          let v319 := r6
                          let v320 := IntTy.addW (.u 64) v318 v319
                          let v321 := r3
                          let v322 := v321
                          let v323 := v322 + v15
                          let v324 ← limb v33 v323
                          mstoreInt (.u 64) v324 v320
                          let v325 := ()
                          let v326 := v325
                          b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
                      partial_fixpoint
                      b39 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v282
                    else
                      let v327 := ()
                      let v328 := v327
                      b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
                partial_fixpoint
                b32 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v211
              let b29 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 => do
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
                let mut v1 := v1
                let v191 := r3
                let v192 := v191
                let v193 := v192 + v15
                let v194 := (UInt256.ofNat 2)
                let v195 := v193 - v194
                let v196 ← limb v33 v195
                let v197 ← mloadInt (.u 64) v196
                let v198 := v0.reciprocal
                let v199 ← div_3by2 v178 v186 v197 v154 v159 v198
                let v200 := v199
                b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v200
              if v187 then
                let v188 := decide (v186 = v159)
                if v188 then
                  let v189 := (UInt256.ofNat 18446744073709551615)
                  let v190 := v189
                  b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v190
                else
                  b29 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
              else
                b29 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
            else
              let v329 := (UInt256.ofNat 0)
              let v330 := ()
              let v331 := (Range_Known_0_Unknown.mk v330 v15)
              r10 := v331
              let v333 ← Range_start_0 r10
              let v334 := v333
              let rec b43 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) (v8 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                let mut v1 := v1
                let v336 ← Range_has_0 r10 v8
                if v336 then
                  let v338 ← Range_at_0 r10 v8
                  let v339 := v338
                  let v340 ← limb v33 v339
                  let v341 ← mloadInt (.u 64) v340
                  let v342 := v341 >>> v60
                  let v343 := v342
                  let v344 := (UInt256.ofNat 0)
                  let v345 := decide (v60 = v344)
                  let b49 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → UInt256 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v9 => do
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
                    let mut v1 := v1
                    let v356 := v9
                    let v357 := v343 ||| v356
                    let v358 ← limb v33 v339
                    mstoreInt (.u 64) v358 v357
                    let v359 := ()
                    let v361 ← Range_next_0 r10 v8
                    let v362 := v361
                    b43 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v362
                  if v345 then
                    let v346 := (UInt256.ofNat 0)
                    let v347 := v346
                    b49 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v347
                  else
                    let v348 := (UInt256.ofNat 1)
                    let v349 := v339 + v348
                    let v350 ← limb v33 v349
                    let v351 ← mloadInt (.u 64) v350
                    let v352 := (UInt256.ofNat 64)
                    let v353 := IntTy.subW (.u 64) v352 v60
                    let v354 := IntTy.shl (.u 64) v351 v353
                    let v355 := v354
                    b49 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v355
                else
                  let v363 := (UInt256.ofNat 1)
                  let v364 := v12 + v363
                  let v365 := ()
                  let v366 := (Range_Unknown_Unknown.mk v15 v364)
                  r11 := v366
                  let v368 ← Range_Unknown_Unknown_start r11
                  let v369 := v368
                  let rec b50 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) (v10 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                    let mut v1 := v1
                    let v371 ← Range_Unknown_Unknown_has r11 v10
                    if v371 then
                      let v373 ← Range_Unknown_Unknown_at r11 v10
                      let v374 := v373
                      let v375 := (UInt256.ofNat 0)
                      let v376 ← limb v33 v374
                      mstoreInt (.u 64) v376 v375
                      let v377 := ()
                      let v379 ← Range_Unknown_Unknown_next r11 v10
                      let v380 := v379
                      b50 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v380
                    else
                      let v381 := ()
                      return ((), v1)
                  partial_fixpoint
                  b50 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v369
              partial_fixpoint
              b43 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v334
          partial_fixpoint
          b25 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
      if v65 then
        let v66 := v12
        r0 := v66
        let rec b16 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Unknown_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_0_Unknown) (r10 : Range_Known_0_Unknown) (r11 : Range_Unknown_Unknown) (v1 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
          let mut v1 := v1
          let v67 := r0
          let v68 := v67
          let v69 := (UInt256.ofNat 0)
          let v70 := decide (v69.toNat < v68.toNat)
          if v70 then
            let v71 := r0
            let v72 := v71
            let v73 ← limb v33 v72
            let v74 ← mloadInt (.u 64) v73
            let v75 := r0
            let v76 := v75
            let v77 := (UInt256.ofNat 1)
            let v78 := v76 - v77
            let v79 ← limb v33 v78
            let v80 ← mloadInt (.u 64) v79
            let v81 := (UInt256.ofNat 64)
            let v82 := IntTy.subW (.u 64) v81 v60
            let v83 := v80 >>> v82
            let v84 := v74 ||| v83
            let v85 := r0
            let v86 := v85
            let v87 ← limb v33 v86
            mstoreInt (.u 64) v87 v84
            let v88 := ()
            let v89 := r0
            let v90 := v89
            let v91 := (UInt256.ofNat 1)
            let v92 := v90 - v91
            let v93 ← limb v33 v92
            let v94 ← mloadInt (.u 64) v93
            let v95 := IntTy.shl (.u 64) v94 v60
            let v96 := r0
            let v97 := v96
            let v98 := (UInt256.ofNat 1)
            let v99 := v97 - v98
            let v100 ← limb v33 v99
            mstoreInt (.u 64) v100 v95
            let v101 := ()
            let v102 := (UInt256.ofNat 1)
            let v103 := r0
            let v104 := v103 - v102
            let v105 := v104
            r0 := v105
            let v106 := ()
            b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
          else
            let v107 := ()
            let v108 := v107
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
        partial_fixpoint
        b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
      else
        let v109 := ()
        let v110 := v109
        b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
  let b2 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → Range_Unknown_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Unknown_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 => do
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
    let mut v1 := v1
    let v28 := false
    let v29 := v28
    b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v29
  if v18 then
    let v20 ← MemArray_len_u64 v1
    let v21 := decide (v15.toNat < v20.toNat)
    if v21 then
      let v24 ← MemArray_len_u64 v0.limbs
      let v25 := decide (v15.toNat ≤ v24.toNat)
      if v25 then
        let v26 := true
        let v27 := v26
        b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1 v27
      else
        b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
    else
      b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1
  else
    b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 v1

def Montgomery_mul (v0 : Montgomery) (v1 : MemSlice_u64) (v2 : MemSlice_u64) (v3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : UInt256 := default
  let mut r4 : Bool := default
  let mut r5 : UInt256 := default
  let mut r6 : UInt256 := default
  let mut r7 : Range_Known_0_Unknown := default
  let mut r8 : Range_Known_0_Unknown := default
  let mut r9 : Range_Known_1_Unknown := default
  let mut r10 : Range_Known_0_Unknown := default
  let mut v3 := v3
  let v9 := v0.len
  let v10 ← downcast_truncate_u64_usize v9
  let v11 := v10
  let v12 ← MemSlice_len_u64 v1
  let v13 := decide (v11.toNat ≤ v12.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v4 => do
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
    let mut v3 := v3
    let v29 ← assert v4
    let v30 ← MemSlice_ptr_u64 v1
    let v31 := v30
    let v32 ← MemSlice_ptr_u64 v2
    let v33 := v32
    let v36 ← MemArray_ptr_u64 v0.limbs
    let v37 := v36
    let v39 ← MemArray_ptr_u64 v3
    let v40 := v39
    let v41 := (UInt256.ofNat 0)
    let v42 := (UInt256.ofNat 1)
    let v43 := v11 + v42
    let v44 := ()
    let v45 := (Range_Known_0_Unknown.mk v44 v43)
    r7 := v45
    let v47 ← Range_start_0 r7
    let v48 := v47
    let rec b7 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Bool) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v5 : UInt256) : FeM (Unit × MemArray_u64) := do
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
      let mut v3 := v3
      let v50 ← Range_has_0 r7 v5
      if v50 then
        let v52 ← Range_at_0 r7 v5
        let v53 := v52
        let v54 := (UInt256.ofNat 0)
        let v55 ← limb v40 v53
        mstoreInt (.u 64) v55 v54
        let v56 := ()
        let v58 ← Range_next_0 r7 v5
        let v59 := v58
        b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v59
      else
        let v60 := (UInt256.ofNat 0)
        let v61 := ()
        let v62 := (Range_Known_0_Unknown.mk v61 v11)
        r8 := v62
        let v64 ← Range_start_0 r8
        let v65 := v64
        let rec b11 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Bool) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
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
          let mut v3 := v3
          let v67 ← Range_has_0 r8 v6
          if v67 then
            let v69 ← Range_at_0 r8 v6
            let v70 := v69
            let v71 ← limb v33 v70
            let v72 ← mloadInt (.u 64) v71
            let v73 := v72
            let v74 := v73
            let v75 ← mloadInt (.u 64) v40
            let v76 := v75
            let v77 ← mloadInt (.u 64) v31
            let v78 := v77
            let v79 := IntTy.mulW (.u 128) v78 v74
            let v80 := IntTy.addW (.u 128) v76 v79
            let v81 := v80
            r0 := v81
            let v82 := r0
            let v83 := v82
            let v84 := (UInt256.ofNat 64)
            let v85 := v83 >>> v84
            let v86 ← downcast_truncate_u128_u64 v85
            let v87 := v86
            r1 := v87
            let v88 := r0
            let v89 := v88
            let v90 ← downcast_truncate_u128_u64 v89
            let v91 := v90
            let v92 := v0.inverse
            let v93 := IntTy.mulW (.u 64) v91 v92
            let v94 := v93
            let v95 := v94
            let v96 := v91
            let v97 ← mloadInt (.u 64) v37
            let v98 := v97
            let v99 := IntTy.mulW (.u 128) v95 v98
            let v100 := IntTy.addW (.u 128) v96 v99
            let v101 := v100
            r2 := v101
            let v102 := r2
            let v103 := v102
            let v104 := (UInt256.ofNat 64)
            let v105 := v103 >>> v104
            let v106 ← downcast_truncate_u128_u64 v105
            let v107 := v106
            r3 := v107
            let v108 := (UInt256.ofNat 1)
            let v109 := ()
            let v110 := (Range_Known_1_Unknown.mk v109 v11)
            r9 := v110
            let v112 ← Range_start_1 r9
            let v113 := v112
            let rec b15 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Bool) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
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
              let mut v3 := v3
              let v115 ← Range_has_1 r9 v7
              if v115 then
                let v117 ← Range_at_1 r9 v7
                let v118 := v117
                let v119 ← limb v40 v118
                let v120 ← mloadInt (.u 64) v119
                let v121 := v120
                let v122 ← limb v31 v118
                let v123 ← mloadInt (.u 64) v122
                let v124 := v123
                let v125 := IntTy.mulW (.u 128) v124 v74
                let v126 := IntTy.addW (.u 128) v121 v125
                let v127 := r1
                let v128 := v127
                let v129 := IntTy.addW (.u 128) v126 v128
                let v130 := v129
                r0 := v130
                let v131 := ()
                let v132 := r0
                let v133 := v132
                let v134 := (UInt256.ofNat 64)
                let v135 := v133 >>> v134
                let v136 ← downcast_truncate_u128_u64 v135
                let v137 := v136
                r1 := v137
                let v138 := ()
                let v139 := r0
                let v140 := v139
                let v141 ← downcast_truncate_u128_u64 v140
                let v142 := v141
                let v143 := v142
                let v144 ← limb v37 v118
                let v145 ← mloadInt (.u 64) v144
                let v146 := v145
                let v147 := IntTy.mulW (.u 128) v95 v146
                let v148 := IntTy.addW (.u 128) v143 v147
                let v149 := r3
                let v150 := v149
                let v151 := IntTy.addW (.u 128) v148 v150
                let v152 := v151
                r2 := v152
                let v153 := ()
                let v154 := r2
                let v155 := v154
                let v156 := (UInt256.ofNat 64)
                let v157 := v155 >>> v156
                let v158 ← downcast_truncate_u128_u64 v157
                let v159 := v158
                r3 := v159
                let v160 := ()
                let v161 := r2
                let v162 := v161
                let v163 ← downcast_truncate_u128_u64 v162
                let v164 := (UInt256.ofNat 1)
                let v165 := v118 - v164
                let v166 ← limb v40 v165
                mstoreInt (.u 64) v166 v163
                let v167 := ()
                let v169 ← Range_next_1 r9 v7
                let v170 := v169
                b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v170
              else
                let v171 ← limb v40 v11
                let v172 ← mloadInt (.u 64) v171
                let v173 := v172
                let v174 := r1
                let v175 := v174
                let v176 := IntTy.addW (.u 128) v173 v175
                let v177 := r3
                let v178 := v177
                let v179 := IntTy.addW (.u 128) v176 v178
                let v180 := v179
                let v181 ← downcast_truncate_u128_u64 v180
                let v182 := (UInt256.ofNat 1)
                let v183 := v11 - v182
                let v184 ← limb v40 v183
                mstoreInt (.u 64) v184 v181
                let v185 := ()
                let v186 := (UInt256.ofNat 64)
                let v187 := v180 >>> v186
                let v188 ← downcast_truncate_u128_u64 v187
                let v189 ← limb v40 v11
                mstoreInt (.u 64) v189 v188
                let v190 := ()
                let v192 ← Range_next_0 r8 v6
                let v193 := v192
                b11 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v193
            partial_fixpoint
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v113
          else
            let v195 ← MemArray_index_ref_u64 v3 v11
            let v196 ← mloadInt (.u 64) v195
            let v197 := (UInt256.ofNat 0)
            let v198 := decide (v196 ≠ v197)
            let v199 := v198
            r4 := v199
            let v200 := r4
            let v201 := !v200
            let b21 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 => do
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
              let mut v3 := v3
              let v247 := r4
              let b30 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 => do
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
                let mut v3 := v3
                return ((), v3)
              if v247 then
                let v248 := (UInt256.ofNat 0)
                let v249 := v248
                r6 := v249
                let v250 := (UInt256.ofNat 0)
                let v251 := ()
                let v252 := (Range_Known_0_Unknown.mk v251 v11)
                r10 := v252
                let v254 ← Range_start_0 r10
                let v255 := v254
                let rec b31 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Bool) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v8 : UInt256) : FeM (Unit × MemArray_u64) := do
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
                  let mut v3 := v3
                  let v257 ← Range_has_0 r10 v8
                  if v257 then
                    let v259 ← Range_at_0 r10 v8
                    let v260 := v259
                    let v262 ← MemArray_index_ref_u64 v3 v260
                    let v263 ← mloadInt (.u 64) v262
                    let v264 := v263
                    let v267 ← MemArray_index_ref_u64 v0.limbs v260
                    let v268 ← mloadInt (.u 64) v267
                    let v269 := v268
                    let v270 := IntTy.subW (.u 128) v264 v269
                    let v271 := r6
                    let v272 := v271
                    let v273 := IntTy.subW (.u 128) v270 v272
                    let v274 := v273
                    let v275 ← downcast_truncate_u128_u64 v274
                    let (v276, v276_k) ← MemArray_index_mut_u64 v3 v260
                    mstoreInt (.u 64) v276 v275
                    let v276_0 ← v276_k v276
                    v3 := v276_0
                    let v277 := ()
                    let v278 := (UInt256.ofNat 127)
                    let v279 := v274 >>> v278
                    let v280 ← downcast_truncate_u128_u64 v279
                    let v281 := v280
                    r6 := v281
                    let v282 := ()
                    let v284 ← Range_next_0 r10 v8
                    let v285 := v284
                    b31 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v285
                  else
                    let v286 := ()
                    let v287 := v286
                    b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
                partial_fixpoint
                b31 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v255
              else
                let v288 := ()
                let v289 := v288
                b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
            if v201 then
              let v202 := true
              let v203 := v202
              r4 := v203
              let v204 := ()
              let v205 := v11
              r5 := v205
              let rec b22 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Bool) (r5 : UInt256) (r6 : UInt256) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (r9 : Range_Known_1_Unknown) (r10 : Range_Known_0_Unknown) (v3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
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
                let mut v3 := v3
                let v206 := r5
                let v207 := v206
                let v208 := (UInt256.ofNat 0)
                let v209 := decide (v208.toNat < v207.toNat)
                let b24 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 => do
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
                  let mut v3 := v3
                  let v243 := ()
                  let v244 := v243
                  b21 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
                if v209 then
                  let v210 := (UInt256.ofNat 1)
                  let v211 := r5
                  let v212 := v211 - v210
                  let v213 := v212
                  r5 := v213
                  let v214 := ()
                  let v215 := r5
                  let v216 := v215
                  let v218 ← MemArray_index_ref_u64 v3 v216
                  let v219 ← mloadInt (.u 64) v218
                  let v220 := r5
                  let v221 := v220
                  let v224 ← MemArray_index_ref_u64 v0.limbs v221
                  let v225 ← mloadInt (.u 64) v224
                  let v226 := decide (v219 ≠ v225)
                  if v226 then
                    let v227 := r5
                    let v228 := v227
                    let v230 ← MemArray_index_ref_u64 v3 v228
                    let v231 ← mloadInt (.u 64) v230
                    let v232 := r5
                    let v233 := v232
                    let v236 ← MemArray_index_ref_u64 v0.limbs v233
                    let v237 ← mloadInt (.u 64) v236
                    let v238 := decide (v237.toNat < v231.toNat)
                    let v239 := v238
                    r4 := v239
                    let v240 := ()
                    b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
                  else
                    let v241 := ()
                    let v242 := v241
                    b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
                else
                  b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
              partial_fixpoint
              b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
            else
              let v245 := ()
              let v246 := v245
              b21 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
        partial_fixpoint
        b11 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v65
    partial_fixpoint
    b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v48
  let b2 : UInt256 → UInt256 → UInt256 → UInt256 → Bool → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 => do
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
    let mut v3 := v3
    let v27 := false
    let v28 := v27
    b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v28
  if v13 then
    let v14 ← MemSlice_len_u64 v2
    let v15 := decide (v11.toNat ≤ v14.toNat)
    if v15 then
      let v18 ← MemArray_len_u64 v0.limbs
      let v19 := decide (v11.toNat ≤ v18.toNat)
      if v19 then
        let v21 ← MemArray_len_u64 v3
        let v22 := (UInt256.ofNat 2)
        let v23 := v11 + v22
        let v24 := decide (v23.toNat ≤ v21.toNat)
        if v24 then
          let v25 := true
          let v26 := v25
          b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3 v26
        else
          b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
      else
        b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
    else
      b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3
  else
    b2 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 v3

def Modulus_mul_mod (v0 : Modulus) (v1 : MemArray_u64) (v2 : MemArray_u64) (v3 : MemArray_u64) : FeM (Unit × MemArray_u64) := do
  let mut r0 : UInt256 := default
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  let mut v3 := v3
  let v8 := v0.len
  let v9 ← downcast_truncate_u64_usize v8
  let v10 := v9
  let v11 ← MemArray_len_u64 v1
  let v12 := decide (v10.toNat ≤ v11.toNat)
  let b3 : UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → Bool → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 v3 v4 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut v3 := v3
    let v24 ← assert v4
    let v25 ← MemArray_ptr_u64 v1
    let v26 := v25
    let v27 ← MemArray_ptr_u64 v2
    let v28 := v27
    let v30 ← MemArray_ptr_u64 v3
    let v31 := v30
    let v32 := (UInt256.ofNat 0)
    let v33 := (UInt256.ofNat 2)
    let v34 := v33 * v10
    let v35 := ()
    let v36 := (Range_Known_0_Unknown.mk v35 v34)
    r1 := v36
    let v38 ← Range_start_0 r1
    let v39 := v38
    let rec b6 (r0 : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v5 : UInt256) : FeM (Unit × MemArray_u64) := do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut v3 := v3
      let v41 ← Range_has_0 r1 v5
      if v41 then
        let v43 ← Range_at_0 r1 v5
        let v44 := v43
        let v45 := (UInt256.ofNat 0)
        let v46 ← limb v31 v44
        mstoreInt (.u 64) v46 v45
        let v47 := ()
        let v49 ← Range_next_0 r1 v5
        let v50 := v49
        b6 r0 r1 r2 r3 v3 v50
      else
        let v51 := (UInt256.ofNat 0)
        let v52 := ()
        let v53 := (Range_Known_0_Unknown.mk v52 v10)
        r2 := v53
        let v55 ← Range_start_0 r2
        let v56 := v55
        let rec b10 (r0 : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v6 : UInt256) : FeM (Unit × MemArray_u64) := do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut v3 := v3
          let v58 ← Range_has_0 r2 v6
          if v58 then
            let v60 ← Range_at_0 r2 v6
            let v61 := v60
            let v62 ← limb v26 v61
            let v63 ← mloadInt (.u 64) v62
            let v64 := v63
            let v65 := v64
            let v66 := (UInt256.ofNat 0)
            let v67 := v66
            r0 := v67
            let v68 := (UInt256.ofNat 0)
            let v69 := ()
            let v70 := (Range_Known_0_Unknown.mk v69 v10)
            r3 := v70
            let v72 ← Range_start_0 r3
            let v73 := v72
            let rec b14 (r0 : UInt256) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v3 : MemArray_u64) (v7 : UInt256) : FeM (Unit × MemArray_u64) := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut v3 := v3
              let v75 ← Range_has_0 r3 v7
              if v75 then
                let v77 ← Range_at_0 r3 v7
                let v78 := v77
                let v79 ← limb v28 v78
                let v80 ← mloadInt (.u 64) v79
                let v81 := v80
                let v82 := IntTy.mulW (.u 128) v65 v81
                let v83 := v61 + v78
                let v84 ← limb v31 v83
                let v85 ← mloadInt (.u 64) v84
                let v86 := v85
                let v87 := IntTy.addW (.u 128) v82 v86
                let v88 := r0
                let v89 := v88
                let v90 := IntTy.addW (.u 128) v87 v89
                let v91 := v90
                let v92 ← downcast_truncate_u128_u64 v91
                let v93 := v61 + v78
                let v94 ← limb v31 v93
                mstoreInt (.u 64) v94 v92
                let v95 := ()
                let v96 := (UInt256.ofNat 64)
                let v97 := v91 >>> v96
                let v98 ← downcast_truncate_u128_u64 v97
                let v99 := v98
                r0 := v99
                let v100 := ()
                let v102 ← Range_next_0 r3 v7
                let v103 := v102
                b14 r0 r1 r2 r3 v3 v103
              else
                let v104 := r0
                let v105 := v104
                let v106 := v61 + v10
                let v107 ← limb v31 v106
                mstoreInt (.u 64) v107 v105
                let v108 := ()
                let v110 ← Range_next_0 r2 v6
                let v111 := v110
                b10 r0 r1 r2 r3 v3 v111
            partial_fixpoint
            b14 r0 r1 r2 r3 v3 v73
          else
            let v113 := (UInt256.ofNat 2)
            let v114 := v0.len
            let v115 := IntTy.mulW (.u 64) v113 v114
            let (v116, v116_0) ← Modulus_reduce v0 v3 v115
            v3 := v116_0
            return ((), v3)
        partial_fixpoint
        b10 r0 r1 r2 r3 v3 v56
    partial_fixpoint
    b6 r0 r1 r2 r3 v3 v39
  let b2 : UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → FeM (Unit × MemArray_u64) := fun r0 r1 r2 r3 v3 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut v3 := v3
    let v22 := false
    let v23 := v22
    b3 r0 r1 r2 r3 v3 v23
  if v12 then
    let v13 ← MemArray_len_u64 v2
    let v14 := decide (v10.toNat ≤ v13.toNat)
    if v14 then
      let v16 ← MemArray_len_u64 v3
      let v17 := (UInt256.ofNat 2)
      let v18 := v17 * v10
      let v19 := decide (v18.toNat < v16.toNat)
      if v19 then
        let v20 := true
        let v21 := v20
        b3 r0 r1 r2 r3 v3 v21
      else
        b2 r0 r1 r2 r3 v3
    else
      b2 r0 r1 r2 r3 v3
  else
    b2 r0 r1 r2 r3 v3

def large_power (v0 : Exponent) (v1 : MemArray_u64) (v2 : Modulus) (v3 : MemArray_u64) (v4 : UInt256) : FeM Option_MemArray_u64 := do
  let mut r0 : MemArray_u64 := default
  let mut r1 : MemArray_u64 := default
  let mut r2 : Montgomery := default
  let mut r3 : MemArray_u64 := default
  let mut r4 : MemArray_u64 := default
  let mut r5 : UInt256 := default
  let mut r6 : Bool := default
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
  let v9 := v0.total
  let v10 := v9
  let v11 ← MemArray_try_new_zeroed_u64 v4
  let v12 := match v11 with | .Some .. => true | _ => false
  if v12 then
    let v13 := v11.Some_0
    let v14 := v13
    r0 := v14
    let v15 := (UInt256.ofNat 2)
    let v16 := v4
    let v17 ← checkedMul v15 v16
    let v18 := (UInt256.ofNat 1)
    let v19 ← checkedAdd v17 v18
    let v20 ← MemArray_try_new_zeroed_u64 v19
    let v21 := match v20 with | .Some .. => true | _ => false
    if v21 then
      let v22 := v20.Some_0
      let v23 := v22
      r1 := v23
      let v24 := v0.top
      let v25 ← checkedSub v10 v24
      let v26 := (UInt256.ofNat 2)
      let v27 := decide (v25 = v26)
      let b6 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
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
        let v48 := v3
        let v49 ← downcast_truncate_usize_u64 v4
        let v50 ← Montgomery_new v48 v49
        let v51 := v50
        r2 := v51
        let v52 ← downcast_truncate_usize_u64 v4
        let v53 := v52
        let v54 := v0.top
        let v55 ← checkedSub v10 v54
        let v56 ← window_width v55
        let v57 := v56
        let v58 := (UInt256.ofNat 1)
        let v59 := (UInt256.ofNat 1)
        let v60 ← checkedSub v57 v59
        let v61 := v58 <<< v60
        let v62 ← downcast_truncate_usize_usize v61
        let v63 := v62
        let v64 := v4
        let v65 ← checkedMul v63 v64
        let v66 ← MemArray_try_new_zeroed_u64 v65
        let v67 := match v66 with | .Some .. => true | _ => false
        if v67 then
          let v68 := v66.Some_0
          let v69 := v68
          r3 := v69
          let v70 := (UInt256.ofNat 0)
          let v71 := v4
          let v72 := ()
          let v73 := (Range_Known_0_Unknown.mk v72 v71)
          r7 := v73
          let v75 ← Range_start_0 r7
          let v76 := v75
          let rec b11 (r0 : MemArray_u64) (r1 : MemArray_u64) (r2 : Montgomery) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v5 : UInt256) : FeM Option_MemArray_u64 := do
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
              let v81 := v80
              let v82 := (UInt256.ofNat 0)
              let (v84, v84_k) ← MemArray_index_mut_u64 r1 v81
              mstoreInt (.u 64) v84 v82
              let v84_0 ← v84_k v84
              r1 := v84_0
              let v85 := ()
              let v86 ← MemArray_index_ref_u64 v1 v81
              let v87 ← mloadInt (.u 64) v86
              let v88 := v4
              let v89 ← checkedAdd v88 v81
              let (v91, v91_k) ← MemArray_index_mut_u64 r1 v89
              mstoreInt (.u 64) v91 v87
              let v91_0 ← v91_k v91
              r1 := v91_0
              let v92 := ()
              let v94 ← Range_next_0 r7 v5
              let v95 := v94
              b11 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v95
            else
              let v97 := (UInt256.ofNat 2)
              let v98 ← IntTy.mul (.u 64) v97 v53
              let (v99, v99_0) ← Modulus_reduce v2 r1 v98
              r1 := v99_0
              let v101 ← MemArray_as_slice_ref_u64 r1
              let v103 ← MemSlice_head_ref_u64 v101 v4
              let (v105_g, v105_k) ← MemArray_as_slice_mut_u64 r3
              v105 := v105_g
              let (v106_g, v106_k) ← MemSlice_head_mut_u64 v105 v4
              v106 := v106_g
              let (v108, v108_0) ← MemSlice_copy_from_u64 v106 v103
              v106 := v108_0
              let v106_0 ← v106_k v106
              v105 := v106_0
              let v105_0 ← v105_k v105
              r3 := v105_0
              let v109 ← MemArray_try_new_zeroed_u64 v4
              let v110 := match v109 with | .Some .. => true | _ => false
              if v110 then
                let v111 := v109.Some_0
                let v112 := v111
                r4 := v112
                let v113 := (UInt256.ofNat 1)
                let v114 := decide (v113.toNat < v63.toNat)
                let b19 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
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
                  let v141 := (Range_Known_1_Unknown.mk v140 v63)
                  r8 := v141
                  let v143 ← Range_start_1 r8
                  let v144 := v143
                  let rec b20 (r0 : MemArray_u64) (r1 : MemArray_u64) (r2 : Montgomery) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v6 : UInt256) : FeM Option_MemArray_u64 := do
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
                      let v149 := v148
                      let v150 := (UInt256.ofNat 1)
                      let v151 ← checkedSub v149 v150
                      let v152 := v4
                      let v153 ← checkedMul v151 v152
                      let v155 ← MemArray_as_slice_ref_u64 r3
                      let v157 ← MemSlice_slice_ref_u64 v155 v153 v4
                      let v159 ← MemArray_as_slice_ref_u64 r4
                      let (v164, v164_0) ← Montgomery_mul r2 v157 v159 r1
                      r1 := v164_0
                      let v166 ← MemArray_as_slice_ref_u64 r1
                      let v168 ← MemSlice_head_ref_u64 v166 v4
                      let v169 := v4
                      let v170 ← checkedMul v149 v169
                      let (v172_g, v172_k) ← MemArray_as_slice_mut_u64 r3
                      v172 := v172_g
                      let (v173_g, v173_k) ← MemSlice_slice_mut_u64 v172 v170 v4
                      v173 := v173_g
                      let (v175, v175_0) ← MemSlice_copy_from_u64 v173 v168
                      v173 := v175_0
                      let v173_0 ← v173_k v173
                      v172 := v173_0
                      let v172_0 ← v172_k v172
                      r3 := v172_0
                      let v177 ← Range_next_1 r8 v6
                      let v178 := v177
                      b20 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v178
                    else
                      let v179 := v0.top
                      let v180 := v179
                      r5 := v180
                      let v181 := false
                      let v182 := v181
                      r6 := v182
                      let rec b24 (r0 : MemArray_u64) (r1 : MemArray_u64) (r2 : Montgomery) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) : FeM Option_MemArray_u64 := do
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
                        let v183 := r5
                        let v184 := v183
                        let v185 := decide (v184.toNat < v10.toNat)
                        if v185 then
                          let v186 := r5
                          let v187 := v186
                          let v188 ← Exponent_window v0 v187 v57
                          let v189 := v188
                          let v190 := r6
                          let rec b29 (r0 : MemArray_u64) (r1 : MemArray_u64) (r2 : Montgomery) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) : FeM Option_MemArray_u64 := do
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
                            let v227 := v189.odd
                            let b36 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 => do
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
                              let v264 := v189.«end»
                              let v265 := v264
                              r5 := v265
                              let v266 := ()
                              b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                            if v227 then
                              let v228 := v189.entry
                              let v229 := v4
                              let v230 ← checkedMul v228 v229
                              let v232 ← MemArray_as_slice_ref_u64 r3
                              let v234 ← MemSlice_slice_ref_u64 v232 v230 v4
                              let v236 := r6
                              let b39 : MemArray_u64 → MemArray_u64 → Montgomery → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_1_Unknown → Range_Unknown_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → Unit → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v8 => do
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
                                b36 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                              if v236 then
                                let v238 ← MemArray_as_slice_ref_u64 r0
                                let (v243, v243_0) ← Montgomery_mul r2 v238 v234 r1
                                r1 := v243_0
                                let v245 ← MemArray_as_slice_ref_u64 r1
                                let v247 ← MemSlice_head_ref_u64 v245 v4
                                let (v249_g, v249_k) ← MemArray_as_slice_mut_u64 r0
                                v249 := v249_g
                                let (v251, v251_0) ← MemSlice_copy_from_u64 v249 v247
                                v249 := v251_0
                                let v249_0 ← v249_k v249
                                r0 := v249_0
                                let v252 := v251
                                b39 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v252
                              else
                                let (v254_g, v254_k) ← MemArray_as_slice_mut_u64 r0
                                v254 := v254_g
                                let (v256, v256_0) ← MemSlice_copy_from_u64 v254 v234
                                v254 := v256_0
                                let v254_0 ← v254_k v254
                                r0 := v254_0
                                let v257 := true
                                let v258 := v257
                                r6 := v258
                                let v259 := ()
                                let v260 := v259
                                b39 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v260
                            else
                              let v262 := ()
                              let v263 := v262
                              b36 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                          partial_fixpoint
                          if v190 then
                            let v191 := r5
                            let v192 := v191
                            let v193 := v189.«end»
                            let v194 := ()
                            let v195 := (Range_Unknown_Unknown.mk v192 v193)
                            r9 := v195
                            let v197 ← Range_Unknown_Unknown_start r9
                            let v198 := v197
                            let rec b30 (r0 : MemArray_u64) (r1 : MemArray_u64) (r2 : Montgomery) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_1_Unknown) (r9 : Range_Unknown_Unknown) (v40 : MemSlice_u64) (v105 : MemSlice_u64) (v106 : MemSlice_u64) (v133 : MemSlice_u64) (v172 : MemSlice_u64) (v173 : MemSlice_u64) (v288 : MemSlice_u64) (v217 : MemSlice_u64) (v249 : MemSlice_u64) (v254 : MemSlice_u64) (v7 : UInt256) : FeM Option_MemArray_u64 := do
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
                                let v204 ← MemArray_as_slice_ref_u64 r0
                                let v206 ← MemArray_as_slice_ref_u64 r0
                                let (v211, v211_0) ← Montgomery_mul r2 v204 v206 r1
                                r1 := v211_0
                                let v213 ← MemArray_as_slice_ref_u64 r1
                                let v215 ← MemSlice_head_ref_u64 v213 v4
                                let (v217_g, v217_k) ← MemArray_as_slice_mut_u64 r0
                                v217 := v217_g
                                let (v219, v219_0) ← MemSlice_copy_from_u64 v217 v215
                                v217 := v219_0
                                let v217_0 ← v217_k v217
                                r0 := v217_0
                                let v221 ← Range_Unknown_Unknown_next r9 v7
                                let v222 := v221
                                b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v222
                              else
                                let v223 := ()
                                let v224 := v223
                                b29 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                            partial_fixpoint
                            b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v198
                          else
                            let v225 := ()
                            let v226 := v225
                            b29 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                        else
                          let (v268, v268_0) ← MemArray_zero_u64 r4
                          r4 := v268_0
                          let v269 := (UInt256.ofNat 1)
                          let v270 := (UInt256.ofNat 0)
                          let (v272, v272_k) ← MemArray_index_mut_u64 r4 v270
                          mstoreInt (.u 64) v272 v269
                          let v272_0 ← v272_k v272
                          r4 := v272_0
                          let v273 := ()
                          let v275 ← MemArray_as_slice_ref_u64 r0
                          let v277 ← MemArray_as_slice_ref_u64 r4
                          let (v282, v282_0) ← Montgomery_mul r2 v275 v277 r1
                          r1 := v282_0
                          let v284 ← MemArray_as_slice_ref_u64 r1
                          let v286 ← MemSlice_head_ref_u64 v284 v4
                          let (v288_g, v288_k) ← MemArray_as_slice_mut_u64 r0
                          v288 := v288_g
                          let (v290, v290_0) ← MemSlice_copy_from_u64 v288 v286
                          v288 := v290_0
                          let v288_0 ← v288_k v288
                          r0 := v288_0
                          let v291 := r0
                          let v292 := v291
                          let v293 := Option_MemArray_u64.Some v292
                          let _ ← MemArray_drop_u64 r4
                          let _ ← MemArray_drop_u64 r3
                          let _ ← MemArray_drop_u64 r2.limbs
                          let _ ← MemArray_drop_u64 r1
                          return v293
                      partial_fixpoint
                      b24 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                  partial_fixpoint
                  b20 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v144
                if v114 then
                  let v116 ← MemArray_as_slice_ref_u64 r3
                  let v118 ← MemSlice_head_ref_u64 v116 v4
                  let v120 ← MemArray_as_slice_ref_u64 r3
                  let v122 ← MemSlice_head_ref_u64 v120 v4
                  let (v127, v127_0) ← Montgomery_mul r2 v118 v122 r1
                  r1 := v127_0
                  let v129 ← MemArray_as_slice_ref_u64 r1
                  let v131 ← MemSlice_head_ref_u64 v129 v4
                  let (v133_g, v133_k) ← MemArray_as_slice_mut_u64 r4
                  v133 := v133_g
                  let (v135, v135_0) ← MemSlice_copy_from_u64 v133 v131
                  v133 := v135_0
                  let v133_0 ← v133_k v133
                  r4 := v133_0
                  let v136 := v135
                  b19 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
                else
                  let v137 := ()
                  let v138 := v137
                  b19 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
              else
                let v294 := Option_MemArray_u64.None
                match (← pure v109) with
                | .Some .. =>
                  let _ ← MemArray_drop_u64 v109.Some_0
                  pure ()
                | _ => pure ()
                let _ ← MemArray_drop_u64 r3
                let _ ← MemArray_drop_u64 r2.limbs
                let _ ← MemArray_drop_u64 r1
                let _ ← MemArray_drop_u64 r0
                return v294
          partial_fixpoint
          b11 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254 v76
        else
          let v295 := Option_MemArray_u64.None
          match (← pure v66) with
          | .Some .. =>
            let _ ← MemArray_drop_u64 v66.Some_0
            pure ()
          | _ => pure ()
          let _ ← MemArray_drop_u64 r2.limbs
          let _ ← MemArray_drop_u64 r1
          let _ ← MemArray_drop_u64 r0
          return v295
      if v27 then
        let v28 := (UInt256.ofNat 1)
        let v29 ← checkedSub v10 v28
        let v30 ← Exponent_bit v0 v29
        let v31 := (UInt256.ofNat 0)
        let v32 := decide (v30 = v31)
        if v32 then
          let (v34, v34_0) ← Modulus_mul_mod v2 v1 v1 r1
          r1 := v34_0
          let v36 ← MemArray_as_slice_ref_u64 r1
          let v38 ← MemSlice_head_ref_u64 v36 v4
          let (v40_g, v40_k) ← MemArray_as_slice_mut_u64 r0
          v40 := v40_g
          let (v42, v42_0) ← MemSlice_copy_from_u64 v40 v38
          v40 := v42_0
          let v40_0 ← v40_k v40
          r0 := v40_0
          let v43 := r0
          let v44 := v43
          let v45 := Option_MemArray_u64.Some v44
          let _ ← MemArray_drop_u64 r1
          let _ ← MemArray_drop_u64 v3
          return v45
        else
          b6 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
      else
        b6 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 v40 v105 v106 v133 v172 v173 v288 v217 v249 v254
    else
      let v296 := Option_MemArray_u64.None
      match (← pure v20) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v20.Some_0
        pure ()
      | _ => pure ()
      let _ ← MemArray_drop_u64 r0
      let _ ← MemArray_drop_u64 v3
      return v296
  else
    let v297 := Option_MemArray_u64.None
    match (← pure v11) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v11.Some_0
      pure ()
    | _ => pure ()
    let _ ← MemArray_drop_u64 v3
    return v297

def odd_power (v0 : Exponent) (v1 : MemArray_u64) (v2 : UInt256) (v3 : MemArray_u64) (v4 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
  let mut r0 : Modulus := default
  let mut r1 : MemArray_u64 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut v1 := v1
  let v6 := v4
  let v7 := (UInt256.ofNat 4)
  let v8 := decide (v6.toNat ≤ v7.toNat)
  let b2 : Modulus → MemArray_u64 → Range_Known_0_Unknown → MemArray_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 v1 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut v1 := v1
    let v15 := ()
    let v16 := v15
    let v17 ← downcast_truncate_usize_u64 v4
    let v18 ← Modulus_try_new v3 v17
    let v19 := match v18 with | .Some .. => true | _ => false
    if v19 then
      let v20 := v18.Some_0
      let v21 := v20
      r0 := v21
      let v22 := v2
      let v23 := v4
      let v24 := decide (v23.toNat < v22.toNat)
      let b9 : Modulus → MemArray_u64 → Range_Known_0_Unknown → MemArray_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 v1 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut v1 := v1
        let v32 := v4
        let v33 := (UInt256.ofNat 4)
        let v34 := decide (v32.toNat ≤ v33.toNat)
        if v34 then
          let v37 ← small_power_of v0 v1 v3 v4
          let _ ← MemArray_drop_u64 r0.limbs
          return (v37, v1)
        else
          let v38 := ()
          let v39 := v38
          let v40 ← MemArray_try_new_zeroed_u64 v4
          let v41 := match v40 with | .Some .. => true | _ => false
          if v41 then
            let v42 := v40.Some_0
            let v43 := v42
            r1 := v43
            let v44 := (UInt256.ofNat 0)
            let v45 := v4
            let v46 := ()
            let v47 := (Range_Known_0_Unknown.mk v46 v45)
            r2 := v47
            let v49 ← Range_start_0 r2
            let v50 := v49
            let rec b15 (r0 : Modulus) (r1 : MemArray_u64) (r2 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v5 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut v1 := v1
              let v52 ← Range_has_0 r2 v5
              if v52 then
                let v54 ← Range_at_0 r2 v5
                let v55 := v54
                let v56 ← MemArray_index_ref_u64 v3 v55
                let v57 ← mloadInt (.u 64) v56
                let (v59, v59_k) ← MemArray_index_mut_u64 r1 v55
                mstoreInt (.u 64) v59 v57
                let v59_0 ← v59_k v59
                r1 := v59_0
                let v60 := ()
                let v62 ← Range_next_0 r2 v5
                let v63 := v62
                b15 r0 r1 r2 v1 v63
              else
                let v65 := r1
                let v66 := v65
                let v69 ← large_power v0 v1 r0 v66 v4
                let _ ← MemArray_drop_u64 r0.limbs
                return (v69, v1)
            partial_fixpoint
            b15 r0 r1 r2 v1 v50
          else
            let v70 := Option_MemArray_u64.None
            match (← pure v40) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v40.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 r0.limbs
            return (v70, v1)
      if v24 then
        let v26 ← downcast_truncate_usize_u64 v2
        let (v28, v28_0) ← Modulus_reduce r0 v1 v26
        v1 := v28_0
        let v29 := v28
        b9 r0 r1 r2 v1
      else
        let v30 := ()
        let v31 := v30
        b9 r0 r1 r2 v1
    else
      let v71 := Option_MemArray_u64.None
      match (← pure v18) with
      | .Some .. =>
        let _ ← MemArray_drop_u64 v18.Some_0.limbs
        pure ()
      | _ => pure ()
      return (v71, v1)
  if v8 then
    let v9 := v2
    let v10 := v4
    let v11 := decide (v9.toNat ≤ v10.toNat)
    if v11 then
      let v14 ← small_power_of v0 v1 v3 v4
      return (v14, v1)
    else
      b2 r0 r1 r2 v1
  else
    b2 r0 r1 r2 v1

def low_product (v0 : MemSlice_u64) (v1 : MemSlice_u64) (v2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : UInt256 := default
  let mut r4 : Range_Known_0_Unknown := default
  let mut r5 : Range_Known_0_Unknown := default
  let mut r6 : Range_Known_0_Unknown := default
  let mut v2 := v2
  let v9 ← MemSlice_len_u64 v2
  let v10 := v9
  let v11 ← MemSlice_len_u64 v0
  let v12 := decide (v10.toNat ≤ v11.toNat)
  let b3 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → Bool → FeM (Unit × MemSlice_u64) := fun r0 r1 r2 r3 r4 r5 r6 v2 v3 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut v2 := v2
    let v19 ← assert v3
    let v20 ← MemSlice_ptr_u64 v0
    let v21 := v20
    let v22 ← MemSlice_ptr_u64 v1
    let v23 := v22
    let v25 ← MemSlice_ptr_u64 v2
    let v26 := v25
    let v27 := (UInt256.ofNat 0)
    let v28 := ()
    let v29 := (Range_Known_0_Unknown.mk v28 v10)
    r4 := v29
    let v31 ← Range_start_0 r4
    let v32 := v31
    let rec b5 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (v2 : MemSlice_u64) (v4 : UInt256) : FeM (Unit × MemSlice_u64) := do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let mut r6 := r6
      let mut v2 := v2
      let v34 ← Range_has_0 r4 v4
      if v34 then
        let v36 ← Range_at_0 r4 v4
        let v37 := v36
        let v38 := (UInt256.ofNat 0)
        let v39 ← limb v26 v37
        mstoreInt (.u 64) v39 v38
        let v40 := ()
        let v42 ← Range_next_0 r4 v4
        let v43 := v42
        b5 r0 r1 r2 r3 r4 r5 r6 v2 v43
      else
        let v44 := v10
        r0 := v44
        let rec b9 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (v2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut v2 := v2
          let v45 := r0
          let v46 := v45
          let v47 := (UInt256.ofNat 0)
          let v48 := decide (v47.toNat < v46.toNat)
          let b11 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun r0 r1 r2 r3 r4 r5 r6 v2 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let mut r6 := r6
            let mut v2 := v2
            let v62 := v10
            r1 := v62
            let rec b13 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (v2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let mut r6 := r6
              let mut v2 := v2
              let v63 := r1
              let v64 := v63
              let v65 := (UInt256.ofNat 0)
              let v66 := decide (v65.toNat < v64.toNat)
              let b15 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun r0 r1 r2 r3 r4 r5 r6 v2 => do
                let mut r0 := r0
                let mut r1 := r1
                let mut r2 := r2
                let mut r3 := r3
                let mut r4 := r4
                let mut r5 := r5
                let mut r6 := r6
                let mut v2 := v2
                let v80 := (UInt256.ofNat 0)
                let v81 := ()
                let v82 := r0
                let v83 := (Range_Known_0_Unknown.mk v81 v82)
                r5 := v83
                let v85 ← Range_start_0 r5
                let v86 := v85
                let rec b17 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (v2 : MemSlice_u64) (v5 : UInt256) : FeM (Unit × MemSlice_u64) := do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let mut r6 := r6
                  let mut v2 := v2
                  let v88 ← Range_has_0 r5 v5
                  if v88 then
                    let v90 ← Range_at_0 r5 v5
                    let v91 := v90
                    let v92 ← limb v21 v91
                    let v93 ← mloadInt (.u 64) v92
                    let v94 := v93
                    let v95 := v94
                    let v96 := (UInt256.ofNat 0)
                    let v97 := v96
                    r2 := v97
                    let v98 := v10 - v91
                    let v99 := r1
                    let v100 := decide (v98.toNat < v99.toNat)
                    let b23 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → UInt256 → FeM (Unit × MemSlice_u64) := fun r0 r1 r2 r3 r4 r5 r6 v2 v6 => do
                      let mut r0 := r0
                      let mut r1 := r1
                      let mut r2 := r2
                      let mut r3 := r3
                      let mut r4 := r4
                      let mut r5 := r5
                      let mut r6 := r6
                      let mut v2 := v2
                      let v106 := v6
                      let v107 := (UInt256.ofNat 0)
                      let v108 := ()
                      let v109 := (Range_Known_0_Unknown.mk v108 v106)
                      r6 := v109
                      let v111 ← Range_start_0 r6
                      let v112 := v111
                      let rec b24 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (v2 : MemSlice_u64) (v7 : UInt256) : FeM (Unit × MemSlice_u64) := do
                        let mut r0 := r0
                        let mut r1 := r1
                        let mut r2 := r2
                        let mut r3 := r3
                        let mut r4 := r4
                        let mut r5 := r5
                        let mut r6 := r6
                        let mut v2 := v2
                        let v114 ← Range_has_0 r6 v7
                        if v114 then
                          let v116 ← Range_at_0 r6 v7
                          let v117 := v116
                          let v118 := v91 + v117
                          let v119 ← limb v26 v118
                          let v120 ← mloadInt (.u 64) v119
                          let v121 := v120
                          let v122 ← limb v23 v117
                          let v123 ← mloadInt (.u 64) v122
                          let v124 := v123
                          let v125 := IntTy.mulW (.u 128) v95 v124
                          let v126 := IntTy.addW (.u 128) v121 v125
                          let v127 := r2
                          let v128 := v127
                          let v129 := IntTy.addW (.u 128) v126 v128
                          let v130 := v129
                          let v131 ← downcast_truncate_u128_u64 v130
                          let v132 := v91 + v117
                          let v133 ← limb v26 v132
                          mstoreInt (.u 64) v133 v131
                          let v134 := ()
                          let v135 := (UInt256.ofNat 64)
                          let v136 := v130 >>> v135
                          let v137 ← downcast_truncate_u128_u64 v136
                          let v138 := v137
                          r2 := v138
                          let v139 := ()
                          let v141 ← Range_next_0 r6 v7
                          let v142 := v141
                          b24 r0 r1 r2 r3 r4 r5 r6 v2 v142
                        else
                          let v143 := v91 + v106
                          let v144 := v143
                          r3 := v144
                          let rec b28 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : Range_Known_0_Unknown) (r5 : Range_Known_0_Unknown) (r6 : Range_Known_0_Unknown) (v2 : MemSlice_u64) : FeM (Unit × MemSlice_u64) := do
                            let mut r0 := r0
                            let mut r1 := r1
                            let mut r2 := r2
                            let mut r3 := r3
                            let mut r4 := r4
                            let mut r5 := r5
                            let mut r6 := r6
                            let mut v2 := v2
                            let v145 := r2
                            let v146 := v145
                            let v147 := (UInt256.ofNat 0)
                            let v148 := decide (v146 ≠ v147)
                            let b30 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun r0 r1 r2 r3 r4 r5 r6 v2 => do
                              let mut r0 := r0
                              let mut r1 := r1
                              let mut r2 := r2
                              let mut r3 := r3
                              let mut r4 := r4
                              let mut r5 := r5
                              let mut r6 := r6
                              let mut v2 := v2
                              let v176 := ()
                              let v178 ← Range_next_0 r5 v5
                              let v179 := v178
                              b17 r0 r1 r2 r3 r4 r5 r6 v2 v179
                            if v148 then
                              let v149 := r3
                              let v150 := v149
                              let v151 := decide (v150.toNat < v10.toNat)
                              if v151 then
                                let v152 := r3
                                let v153 := v152
                                let v154 ← limb v26 v153
                                let v155 ← mloadInt (.u 64) v154
                                let v156 := v155
                                let v157 := r2
                                let v158 := v157
                                let v159 := IntTy.addW (.u 128) v156 v158
                                let v160 := v159
                                let v161 ← downcast_truncate_u128_u64 v160
                                let v162 := r3
                                let v163 := v162
                                let v164 ← limb v26 v163
                                mstoreInt (.u 64) v164 v161
                                let v165 := ()
                                let v166 := (UInt256.ofNat 64)
                                let v167 := v160 >>> v166
                                let v168 ← downcast_truncate_u128_u64 v167
                                let v169 := v168
                                r2 := v169
                                let v170 := ()
                                let v171 := (UInt256.ofNat 1)
                                let v172 := r3
                                let v173 := v172 + v171
                                let v174 := v173
                                r3 := v174
                                let v175 := ()
                                b28 r0 r1 r2 r3 r4 r5 r6 v2
                              else
                                b30 r0 r1 r2 r3 r4 r5 r6 v2
                            else
                              b30 r0 r1 r2 r3 r4 r5 r6 v2
                          partial_fixpoint
                          b28 r0 r1 r2 r3 r4 r5 r6 v2
                      partial_fixpoint
                      b24 r0 r1 r2 r3 r4 r5 r6 v2 v112
                    if v100 then
                      let v101 := v10 - v91
                      let v102 := v101
                      b23 r0 r1 r2 r3 r4 r5 r6 v2 v102
                    else
                      let v103 := r1
                      let v104 := v103
                      let v105 := v104
                      b23 r0 r1 r2 r3 r4 r5 r6 v2 v105
                  else
                    let v180 := ()
                    return ((), v2)
                partial_fixpoint
                b17 r0 r1 r2 r3 r4 r5 r6 v2 v86
              if v66 then
                let v67 := r1
                let v68 := v67
                let v69 := (UInt256.ofNat 1)
                let v70 := v68 - v69
                let v71 ← limb v23 v70
                let v72 ← mloadInt (.u 64) v71
                let v73 := (UInt256.ofNat 0)
                let v74 := decide (v72 = v73)
                if v74 then
                  let v75 := (UInt256.ofNat 1)
                  let v76 := r1
                  let v77 := v76 - v75
                  let v78 := v77
                  r1 := v78
                  let v79 := ()
                  b13 r0 r1 r2 r3 r4 r5 r6 v2
                else
                  b15 r0 r1 r2 r3 r4 r5 r6 v2
              else
                b15 r0 r1 r2 r3 r4 r5 r6 v2
            partial_fixpoint
            b13 r0 r1 r2 r3 r4 r5 r6 v2
          if v48 then
            let v49 := r0
            let v50 := v49
            let v51 := (UInt256.ofNat 1)
            let v52 := v50 - v51
            let v53 ← limb v21 v52
            let v54 ← mloadInt (.u 64) v53
            let v55 := (UInt256.ofNat 0)
            let v56 := decide (v54 = v55)
            if v56 then
              let v57 := (UInt256.ofNat 1)
              let v58 := r0
              let v59 := v58 - v57
              let v60 := v59
              r0 := v60
              let v61 := ()
              b9 r0 r1 r2 r3 r4 r5 r6 v2
            else
              b11 r0 r1 r2 r3 r4 r5 r6 v2
          else
            b11 r0 r1 r2 r3 r4 r5 r6 v2
        partial_fixpoint
        b9 r0 r1 r2 r3 r4 r5 r6 v2
    partial_fixpoint
    b5 r0 r1 r2 r3 r4 r5 r6 v2 v32
  let b2 : UInt256 → UInt256 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → FeM (Unit × MemSlice_u64) := fun r0 r1 r2 r3 r4 r5 r6 v2 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut v2 := v2
    let v17 := false
    let v18 := v17
    b3 r0 r1 r2 r3 r4 r5 r6 v2 v18
  if v12 then
    let v13 ← MemSlice_len_u64 v1
    let v14 := decide (v10.toNat ≤ v13.toNat)
    if v14 then
      let v15 := true
      let v16 := v15
      b3 r0 r1 r2 r3 r4 r5 r6 v2 v16
    else
      b2 r0 r1 r2 r3 r4 r5 r6 v2
  else
    b2 r0 r1 r2 r3 r4 r5 r6 v2

def low_power (v0 : Exponent) (v1 : MemArray_u64) (v2 : UInt256) : FeM Option_MemArray_u64 := do
  let mut r0 : MemArray_u64 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : MemArray_u64 := default
  let mut r4 : MemArray_u64 := default
  let mut r5 : UInt256 := default
  let mut r6 : Bool := default
  let mut r7 : Range_Known_0_Unknown := default
  let mut r8 : Range_Known_0_Unknown := default
  let mut v193 : MemSlice_u64 := default
  let mut v208 : MemSlice_u64 := default
  let mut v217 : MemSlice_u64 := default
  let v9 := v2
  let v10 := (UInt256.ofNat 63)
  let v11 := v9 + v10
  let v12 := (UInt256.ofNat 64)
  let v13 ← checkedDiv v11 v12
  let v14 := v13
  let v15 := v2
  let v16 := (UInt256.ofNat 64)
  let v17 ← checkedRem v15 v16
  let v18 ← downcast_truncate_usize_u64 v17
  let v19 := v18
  let v20 := (UInt256.ofNat 0)
  let v21 := decide (v19 = v20)
  let b3 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v3 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    let mut v193 := v193
    let mut v208 := v208
    let mut v217 := v217
    let v29 := v3
    let v30 ← MemArray_try_new_zeroed_u64 v14
    let v31 := match v30 with | .Some .. => true | _ => false
    if v31 then
      let v32 := v30.Some_0
      let v33 := v32
      r0 := v33
      let v34 := (UInt256.ofNat 0)
      let v35 ← MemArray_index_ref_u64 v1 v34
      let v36 ← mloadInt (.u 64) v35
      let v37 := (UInt256.ofNat 1)
      let v38 := v36 &&& v37
      let v39 := (UInt256.ofNat 1)
      let v40 := decide (v38 = v39)
      let b8 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v4 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let mut r7 := r7
        let mut r8 := r8
        let mut v193 := v193
        let mut v208 := v208
        let mut v217 := v217
        let v52 := v4
        let v53 := (UInt256.ofNat 2)
        let v54 := decide (v14.toNat ≤ v53.toNat)
        if v54 then
          let v55 := (UInt256.ofNat 1)
          let v56 := decide (v14 = v55)
          let b15 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v5 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let mut r6 := r6
            let mut r7 := r7
            let mut r8 := r8
            let mut v193 := v193
            let mut v208 := v208
            let mut v217 := v217
            let v65 := v5
            let v66 := (UInt256.ofNat 0)
            let v67 ← MemArray_index_ref_u64 v1 v66
            let v68 ← mloadInt (.u 64) v67
            let v69 := v68
            let v70 := (UInt256.ofNat 1)
            let v71 ← MemArray_index_ref_u64 v1 v70
            let v72 ← mloadInt (.u 64) v71
            let v73 := v72
            let v74 := (UInt256.ofNat 64)
            let v75 := IntTy.shl (.u 128) v73 v74
            let v76 := v69 ||| v75
            let v77 := v76 &&& v65
            let v78 := v77
            let v79 := (UInt256.ofNat 1)
            let v80 := v79
            r1 := v80
            let v81 := v52
            r2 := v81
            let rec b16 (r0 : MemArray_u64) (r1 : UInt256) (r2 : UInt256) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) : FeM Option_MemArray_u64 := do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let mut r3 := r3
              let mut r4 := r4
              let mut r5 := r5
              let mut r6 := r6
              let mut r7 := r7
              let mut r8 := r8
              let mut v193 := v193
              let mut v208 := v208
              let mut v217 := v217
              let v82 := r2
              let v83 := v82
              let v84 := v0.total
              let v85 := decide (v83.toNat < v84.toNat)
              let b18 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
                let mut r0 := r0
                let mut r1 := r1
                let mut r2 := r2
                let mut r3 := r3
                let mut r4 := r4
                let mut r5 := r5
                let mut r6 := r6
                let mut r7 := r7
                let mut r8 := r8
                let mut v193 := v193
                let mut v208 := v208
                let mut v217 := v217
                let v118 := r1
                let v119 := v118
                let v120 ← downcast_truncate_u128_u64 v119
                let v121 := (UInt256.ofNat 0)
                let (v123, v123_k) ← MemArray_index_mut_u64 r0 v121
                mstoreInt (.u 64) v123 v120
                let v123_0 ← v123_k v123
                r0 := v123_0
                let v124 := ()
                let v125 := (UInt256.ofNat 2)
                let v126 := decide (v14 = v125)
                let b25 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let mut r6 := r6
                  let mut r7 := r7
                  let mut r8 := r8
                  let mut v193 := v193
                  let mut v208 := v208
                  let mut v217 := v217
                  let v139 := r0
                  let v140 := v139
                  let v141 := Option_MemArray_u64.Some v140
                  return v141
                if v126 then
                  let v127 := r1
                  let v128 := v127
                  let v129 := (UInt256.ofNat 64)
                  let v130 := v128 >>> v129
                  let v131 ← downcast_truncate_u128_u64 v130
                  let v132 := (UInt256.ofNat 1)
                  let (v134, v134_k) ← MemArray_index_mut_u64 r0 v132
                  mstoreInt (.u 64) v134 v131
                  let v134_0 ← v134_k v134
                  r0 := v134_0
                  let v135 := ()
                  let v136 := v135
                  b25 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                else
                  let v137 := ()
                  let v138 := v137
                  b25 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
              if v85 then
                let v86 := r1
                let v87 := v86
                let v88 := (UInt256.ofNat 0)
                let v89 := decide (v87 ≠ v88)
                if v89 then
                  let v90 := r1
                  let v91 := v90
                  let v92 := r1
                  let v93 := IntTy.mulW (.u 128) v91 v92
                  let v94 := v93
                  r1 := v94
                  let v95 := ()
                  let v96 := r2
                  let v97 := v96
                  let v98 ← Exponent_bit v0 v97
                  let v99 := (UInt256.ofNat 1)
                  let v100 := decide (v98 = v99)
                  let b22 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
                    let mut r0 := r0
                    let mut r1 := r1
                    let mut r2 := r2
                    let mut r3 := r3
                    let mut r4 := r4
                    let mut r5 := r5
                    let mut r6 := r6
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut v193 := v193
                    let mut v208 := v208
                    let mut v217 := v217
                    let v109 := r1
                    let v110 := v109 &&& v65
                    let v111 := v110
                    r1 := v111
                    let v112 := ()
                    let v113 := (UInt256.ofNat 1)
                    let v114 := r2
                    let v115 := v114 + v113
                    let v116 := v115
                    r2 := v116
                    let v117 := ()
                    b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                  if v100 then
                    let v101 := r1
                    let v102 := v101
                    let v103 := IntTy.mulW (.u 128) v102 v78
                    let v104 := v103
                    r1 := v104
                    let v105 := ()
                    let v106 := v105
                    b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                  else
                    let v107 := ()
                    let v108 := v107
                    b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                else
                  b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
              else
                b18 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
            partial_fixpoint
            b16 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
          if v56 then
            let v57 := v29
            let v58 := v57
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v58
          else
            let v59 := v29
            let v60 := (UInt256.ofNat 64)
            let v61 := IntTy.shl (.u 128) v59 v60
            let v62 := (UInt256.ofNat 18446744073709551615)
            let v63 := v61 ||| v62
            let v64 := v63
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v64
        else
          let v142 := ()
          let v143 := v142
          let v144 ← MemArray_try_new_zeroed_u64 v14
          let v145 := match v144 with | .Some .. => true | _ => false
          if v145 then
            let v146 := v144.Some_0
            let v147 := v146
            r3 := v147
            let v148 ← MemArray_try_new_zeroed_u64 v14
            let v149 := match v148 with | .Some .. => true | _ => false
            if v149 then
              let v150 := v148.Some_0
              let v151 := v150
              r4 := v151
              let v152 := (UInt256.ofNat 0)
              let v153 := ()
              let v154 := (Range_Known_0_Unknown.mk v153 v14)
              r7 := v154
              let v156 ← Range_start_0 r7
              let v157 := v156
              let rec b30 (r0 : MemArray_u64) (r1 : UInt256) (r2 : UInt256) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) (v6 : UInt256) : FeM Option_MemArray_u64 := do
                let mut r0 := r0
                let mut r1 := r1
                let mut r2 := r2
                let mut r3 := r3
                let mut r4 := r4
                let mut r5 := r5
                let mut r6 := r6
                let mut r7 := r7
                let mut r8 := r8
                let mut v193 := v193
                let mut v208 := v208
                let mut v217 := v217
                let v159 ← Range_has_0 r7 v6
                if v159 then
                  let v161 ← Range_at_0 r7 v6
                  let v162 := v161
                  let v163 ← MemArray_index_ref_u64 v1 v162
                  let v164 ← mloadInt (.u 64) v163
                  let (v166, v166_k) ← MemArray_index_mut_u64 r3 v162
                  mstoreInt (.u 64) v166 v164
                  let v166_0 ← v166_k v166
                  r3 := v166_0
                  let v167 := ()
                  let v169 ← Range_next_0 r7 v6
                  let v170 := v169
                  b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v170
                else
                  let v171 := (UInt256.ofNat 1)
                  let v172 := v14 - v171
                  let (v174, v174_k) ← MemArray_index_mut_u64 r3 v172
                  let v175 ← mloadInt (.u 64) v174
                  let v176 := v175 &&& v29
                  mstoreInt (.u 64) v174 v176
                  let v174_0 ← v174_k v174
                  r3 := v174_0
                  let v177 := ()
                  let v178 := (UInt256.ofNat 1)
                  let v179 := (UInt256.ofNat 0)
                  let (v181, v181_k) ← MemArray_index_mut_u64 r0 v179
                  mstoreInt (.u 64) v181 v178
                  let v181_0 ← v181_k v181
                  r0 := v181_0
                  let v182 := ()
                  let v183 := v52
                  r5 := v183
                  let rec b34 (r0 : MemArray_u64) (r1 : UInt256) (r2 : UInt256) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) : FeM Option_MemArray_u64 := do
                    let mut r0 := r0
                    let mut r1 := r1
                    let mut r2 := r2
                    let mut r3 := r3
                    let mut r4 := r4
                    let mut r5 := r5
                    let mut r6 := r6
                    let mut r7 := r7
                    let mut r8 := r8
                    let mut v193 := v193
                    let mut v208 := v208
                    let mut v217 := v217
                    let v184 := r5
                    let v185 := v184
                    let v186 := v0.total
                    let v187 := decide (v185.toNat < v186.toNat)
                    let b36 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
                      let mut r0 := r0
                      let mut r1 := r1
                      let mut r2 := r2
                      let mut r3 := r3
                      let mut r4 := r4
                      let mut r5 := r5
                      let mut r6 := r6
                      let mut r7 := r7
                      let mut r8 := r8
                      let mut v193 := v193
                      let mut v208 := v208
                      let mut v217 := v217
                      let v264 := r0
                      let v265 := v264
                      let v266 := Option_MemArray_u64.Some v265
                      let _ ← MemArray_drop_u64 r4
                      let _ ← MemArray_drop_u64 r3
                      return v266
                    if v187 then
                      let v189 ← MemArray_as_slice_ref_u64 r0
                      let v191 ← MemArray_as_slice_ref_u64 r0
                      let (v193_g, v193_k) ← MemArray_as_slice_mut_u64 r4
                      v193 := v193_g
                      let (v197, v197_0) ← low_product v189 v191 v193
                      v193 := v197_0
                      let v193_0 ← v193_k v193
                      r4 := v193_0
                      let v198 := r5
                      let v199 := v198
                      let v200 ← Exponent_bit v0 v199
                      let v201 := (UInt256.ofNat 1)
                      let v202 := decide (v200 = v201)
                      let b39 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
                        let mut r0 := r0
                        let mut r1 := r1
                        let mut r2 := r2
                        let mut r3 := r3
                        let mut r4 := r4
                        let mut r5 := r5
                        let mut r6 := r6
                        let mut r7 := r7
                        let mut r8 := r8
                        let mut v193 := v193
                        let mut v208 := v208
                        let mut v217 := v217
                        let v221 := (UInt256.ofNat 1)
                        let v222 := v14 - v221
                        let (v224, v224_k) ← MemArray_index_mut_u64 r0 v222
                        let v225 ← mloadInt (.u 64) v224
                        let v226 := v225 &&& v29
                        mstoreInt (.u 64) v224 v226
                        let v224_0 ← v224_k v224
                        r0 := v224_0
                        let v227 := ()
                        let v228 := true
                        let v229 := v228
                        r6 := v229
                        let v230 := (UInt256.ofNat 0)
                        let v231 := ()
                        let v232 := (Range_Known_0_Unknown.mk v231 v14)
                        r8 := v232
                        let v234 ← Range_start_0 r8
                        let v235 := v234
                        let rec b40 (r0 : MemArray_u64) (r1 : UInt256) (r2 : UInt256) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) (r6 : Bool) (r7 : Range_Known_0_Unknown) (r8 : Range_Known_0_Unknown) (v193 : MemSlice_u64) (v208 : MemSlice_u64) (v217 : MemSlice_u64) (v7 : UInt256) : FeM Option_MemArray_u64 := do
                          let mut r0 := r0
                          let mut r1 := r1
                          let mut r2 := r2
                          let mut r3 := r3
                          let mut r4 := r4
                          let mut r5 := r5
                          let mut r6 := r6
                          let mut r7 := r7
                          let mut r8 := r8
                          let mut v193 := v193
                          let mut v208 := v208
                          let mut v217 := v217
                          let v237 ← Range_has_0 r8 v7
                          if v237 then
                            let v239 ← Range_at_0 r8 v7
                            let v240 := v239
                            let v241 := r6
                            let b46 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → Bool → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v8 => do
                              let mut r0 := r0
                              let mut r1 := r1
                              let mut r2 := r2
                              let mut r3 := r3
                              let mut r4 := r4
                              let mut r5 := r5
                              let mut r6 := r6
                              let mut r7 := r7
                              let mut r8 := r8
                              let mut v193 := v193
                              let mut v208 := v208
                              let mut v217 := v217
                              let v251 := v8
                              r6 := v251
                              let v252 := ()
                              let v254 ← Range_next_0 r8 v7
                              let v255 := v254
                              b40 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v255
                            let b45 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
                              let mut r0 := r0
                              let mut r1 := r1
                              let mut r2 := r2
                              let mut r3 := r3
                              let mut r4 := r4
                              let mut r5 := r5
                              let mut r6 := r6
                              let mut r7 := r7
                              let mut r8 := r8
                              let mut v193 := v193
                              let mut v208 := v208
                              let mut v217 := v217
                              let v249 := false
                              let v250 := v249
                              b46 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v250
                            if v241 then
                              let v243 ← MemArray_index_ref_u64 r0 v240
                              let v244 ← mloadInt (.u 64) v243
                              let v245 := (UInt256.ofNat 0)
                              let v246 := decide (v244 = v245)
                              if v246 then
                                let v247 := true
                                let v248 := v247
                                b46 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v248
                              else
                                b45 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                            else
                              b45 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                          else
                            let v256 := r6
                            if v256 then
                              b36 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                            else
                              let v257 := ()
                              let v258 := v257
                              let v259 := (UInt256.ofNat 1)
                              let v260 := r5
                              let v261 := v260 + v259
                              let v262 := v261
                              r5 := v262
                              let v263 := ()
                              b34 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                        partial_fixpoint
                        b40 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v235
                      if v202 then
                        let v204 ← MemArray_as_slice_ref_u64 r4
                        let v206 ← MemArray_as_slice_ref_u64 r3
                        let (v208_g, v208_k) ← MemArray_as_slice_mut_u64 r0
                        v208 := v208_g
                        let (v212, v212_0) ← low_product v204 v206 v208
                        v208 := v212_0
                        let v208_0 ← v208_k v208
                        r0 := v208_0
                        let v213 := v212
                        b39 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                      else
                        let v215 ← MemArray_as_slice_ref_u64 r4
                        let (v217_g, v217_k) ← MemArray_as_slice_mut_u64 r0
                        v217 := v217_g
                        let (v219, v219_0) ← MemSlice_copy_from_u64 v217 v215
                        v217 := v219_0
                        let v217_0 ← v217_k v217
                        r0 := v217_0
                        let v220 := v219
                        b39 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                    else
                      b36 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
                  partial_fixpoint
                  b34 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
              partial_fixpoint
              b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v157
            else
              let v267 := Option_MemArray_u64.None
              match (← pure v148) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v148.Some_0
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 r3
              let _ ← MemArray_drop_u64 r0
              return v267
          else
            let v268 := Option_MemArray_u64.None
            match (← pure v144) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v144.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 r0
            return v268
      let b7 : MemArray_u64 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → Bool → Range_Known_0_Unknown → Range_Known_0_Unknown → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let mut r7 := r7
        let mut r8 := r8
        let mut v193 := v193
        let mut v208 := v208
        let mut v217 := v217
        let v50 := v0.top
        let v51 := v50
        b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v51
      if v40 then
        let v41 := v0.total
        let v42 := v0.top
        let v43 := v41 - v42
        let v44 := v2
        let v45 := decide (v44.toNat < v43.toNat)
        if v45 then
          let v46 := v0.total
          let v47 := v2
          let v48 := v46 - v47
          let v49 := v48
          b8 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v49
        else
          b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
      else
        b7 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217
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
    b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v23
  else
    let v24 := (UInt256.ofNat 1)
    let v25 := IntTy.shl (.u 64) v24 v19
    let v26 := (UInt256.ofNat 1)
    let v27 := IntTy.subW (.u 64) v25 v26
    let v28 := v27
    b3 r0 r1 r2 r3 r4 r5 r6 r7 r8 v193 v208 v217 v28

def even_power (v0 : Exponent) (v1 : MemArray_u64) (v2 : UInt256) (v3 : MemArray_u64) (v4 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
  let mut r0 : UInt256 := default
  let mut r1 : MemArray_u64 := default
  let mut r2 : MemArray_u64 := default
  let mut r3 : MemArray_u64 := default
  let mut r4 : UInt256 := default
  let mut r5 : MemArray_u64 := default
  let mut r6 : MemArray_u64 := default
  let mut r7 : UInt256 := default
  let mut r8 : MemArray_u64 := default
  let mut r9 : UInt256 := default
  let mut r10 : MemArray_u64 := default
  let mut r11 : MemArray_u64 := default
  let mut r12 : UInt256 := default
  let mut r13 : UInt256 := default
  let mut r14 : UInt256 := default
  let mut r15 : MemArray_u64 := default
  let mut r16 : MemArray_u64 := default
  let mut r17 : UInt256 := default
  let mut r18 : UInt256 := default
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
  let mut v1 := v1
  let mut v320 : MemSlice_u64 := default
  let mut v321 : MemSlice_u64 := default
  let mut v398 : MemSlice_u64 := default
  let mut v399 : MemSlice_u64 := default
  let mut v409 : MemSlice_u64 := default
  let mut v410 : MemSlice_u64 := default
  let mut v424 : MemSlice_u64 := default
  let v20 := (UInt256.ofNat 0)
  let v21 := v20
  r0 := v21
  let rec b1 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
    let mut v1 := v1
    let mut v320 := v320
    let mut v321 := v321
    let mut v398 := v398
    let mut v399 := v399
    let mut v409 := v409
    let mut v410 := v410
    let mut v424 := v424
    let v22 := r0
    let v23 := v22
    let v24 ← MemArray_index_ref_u64 v3 v23
    let v25 ← mloadInt (.u 64) v24
    let v26 := (UInt256.ofNat 0)
    let v27 := decide (v25 = v26)
    if v27 then
      let v28 := (UInt256.ofNat 1)
      let v29 := r0
      let v30 := v29 + v28
      let v31 := v30
      r0 := v31
      let v32 := ()
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
    else
      let v33 := r0
      let v34 := v33
      let v35 := (UInt256.ofNat 64)
      let v36 := v34 * v35
      let v37 := r0
      let v38 := v37
      let v39 ← MemArray_index_ref_u64 v3 v38
      let v40 ← mloadInt (.u 64) v39
      let v41 := IntTy.trailingZeros (.u 64) v40
      let v42 ← downcast_truncate_u64_usize v41
      let v43 := v36 + v42
      let v44 := v43
      let v45 := (UInt256.ofNat 63)
      let v46 := v44 + v45
      let v47 := (UInt256.ofNat 64)
      let v48 ← checkedDiv v46 v47
      let v49 := v48
      let v50 := (UInt256.ofNat 64)
      let v51 ← checkedRem v44 v50
      let v52 ← downcast_truncate_usize_u64 v51
      let v53 := v52
      let v54 := (UInt256.ofNat 0)
      let v55 := decide (v53 = v54)
      let b6 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v5 => do
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
        let mut v1 := v1
        let mut v320 := v320
        let mut v321 := v321
        let mut v398 := v398
        let mut v399 := v399
        let mut v409 := v409
        let mut v410 := v410
        let mut v424 := v424
        let v63 := v5
        let v64 ← MemArray_try_new_zeroed_u64 v4
        let v65 := match v64 with | .Some .. => true | _ => false
        if v65 then
          let v66 := v64.Some_0
          let v67 := v66
          r1 := v67
          let v70 ← low_power v0 v1 v44
          let v71 := match v70 with | .Some .. => true | _ => false
          if v71 then
            let v72 := v70.Some_0
            let v73 := v72
            r2 := v73
            let v74 ← MemArray_try_new_zeroed_u64 v4
            let v75 := match v74 with | .Some .. => true | _ => false
            if v75 then
              let v76 := v74.Some_0
              let v77 := v76
              r3 := v77
              let v78 := (UInt256.ofNat 0)
              let v79 := v4
              let v80 := (UInt256.ofNat 64)
              let v81 ← checkedDiv v44 v80
              let v82 := v79 - v81
              let v83 := ()
              let v84 := (Range_Known_0_Unknown.mk v83 v82)
              r19 := v84
              let v86 ← Range_start_0 r19
              let v87 := v86
              let rec b13 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v6 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                let mut v1 := v1
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
                  let v92 := v91
                  let v93 := (UInt256.ofNat 64)
                  let v94 ← checkedDiv v44 v93
                  let v95 := v92 + v94
                  let v96 := (UInt256.ofNat 1)
                  let v97 := v95 + v96
                  let v98 := v4
                  let v99 := decide (v97.toNat < v98.toNat)
                  let b19 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v7 => do
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
                    let mut v1 := v1
                    let mut v320 := v320
                    let mut v321 := v321
                    let mut v398 := v398
                    let mut v399 := v399
                    let mut v409 := v409
                    let mut v410 := v410
                    let mut v424 := v424
                    let v110 := v7
                    let v111 := (UInt256.ofNat 0)
                    let v112 := decide (v53 = v111)
                    let b22 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v8 => do
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
                      let mut v1 := v1
                      let mut v320 := v320
                      let mut v321 := v321
                      let mut v398 := v398
                      let mut v399 := v399
                      let mut v409 := v409
                      let mut v410 := v410
                      let mut v424 := v424
                      let (v131, v131_k) ← MemArray_index_mut_u64 r3 v92
                      mstoreInt (.u 64) v131 v8
                      let v131_0 ← v131_k v131
                      r3 := v131_0
                      let v132 := ()
                      let v134 ← Range_next_0 r19 v6
                      let v135 := v134
                      b13 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v135
                    if v112 then
                      let v113 := (UInt256.ofNat 64)
                      let v114 ← checkedDiv v44 v113
                      let v115 := v92 + v114
                      let v116 ← MemArray_index_ref_u64 v3 v115
                      let v117 ← mloadInt (.u 64) v116
                      let v118 := v117
                      b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v118
                    else
                      let v119 := (UInt256.ofNat 64)
                      let v120 ← checkedDiv v44 v119
                      let v121 := v92 + v120
                      let v122 ← MemArray_index_ref_u64 v3 v121
                      let v123 ← mloadInt (.u 64) v122
                      let v124 := v123 >>> v53
                      let v125 := (UInt256.ofNat 64)
                      let v126 := IntTy.subW (.u 64) v125 v53
                      let v127 := IntTy.shl (.u 64) v110 v126
                      let v128 := v124 ||| v127
                      let v129 := v128
                      b22 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v129
                  if v99 then
                    let v100 := (UInt256.ofNat 64)
                    let v101 ← checkedDiv v44 v100
                    let v102 := v92 + v101
                    let v103 := (UInt256.ofNat 1)
                    let v104 := v102 + v103
                    let v105 ← MemArray_index_ref_u64 v3 v104
                    let v106 ← mloadInt (.u 64) v105
                    let v107 := v106
                    b19 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v107
                  else
                    let v108 := (UInt256.ofNat 0)
                    let v109 := v108
                    b19 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v109
                else
                  let v136 := v4
                  let v137 := v136
                  r4 := v137
                  let rec b23 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                    let mut v1 := v1
                    let mut v320 := v320
                    let mut v321 := v321
                    let mut v398 := v398
                    let mut v399 := v399
                    let mut v409 := v409
                    let mut v410 := v410
                    let mut v424 := v424
                    let v138 := r4
                    let v139 := v138
                    let v140 := (UInt256.ofNat 1)
                    let v141 := v139 - v140
                    let v143 ← MemArray_index_ref_u64 r3 v141
                    let v144 ← mloadInt (.u 64) v143
                    let v145 := (UInt256.ofNat 0)
                    let v146 := decide (v144 = v145)
                    if v146 then
                      let v147 := (UInt256.ofNat 1)
                      let v148 := r4
                      let v149 := v148 - v147
                      let v150 := v149
                      r4 := v150
                      let v151 := ()
                      b23 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                    else
                      let v152 := r4
                      let v153 := v152
                      let v154 := (UInt256.ofNat 1)
                      let v155 := decide (v153 = v154)
                      let b27 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 => do
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
                        let mut v1 := v1
                        let mut v320 := v320
                        let mut v321 := v321
                        let mut v398 := v398
                        let mut v399 := v399
                        let mut v409 := v409
                        let mut v410 := v410
                        let mut v424 := v424
                        let v185 := ()
                        let v186 := v185
                        let v189 := r4
                        let v190 := v189
                        let (v192, v192_0) ← odd_power v0 v1 v2 r3 v190
                        v1 := v192_0
                        let v193 := match v192 with | .Some .. => true | _ => false
                        if v193 then
                          let v194 := v192.Some_0
                          let v195 := v194
                          r5 := v195
                          let v196 ← MemArray_try_new_zeroed_u64 v49
                          let v197 := match v196 with | .Some .. => true | _ => false
                          if v197 then
                            let v198 := v196.Some_0
                            let v199 := v198
                            r6 := v199
                            let v200 := (UInt256.ofNat 0)
                            let v201 := v200
                            r7 := v201
                            let v202 := (UInt256.ofNat 0)
                            let v203 := ()
                            let v204 := (Range_Known_0_Unknown.mk v203 v49)
                            r21 := v204
                            let v206 ← Range_start_0 r21
                            let v207 := v206
                            let rec b38 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v10 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                              let mut v1 := v1
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
                                let v212 := v211
                                let v213 := r4
                                let v214 := decide (v212.toNat < v213.toNat)
                                let b44 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v11 => do
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
                                  let mut v1 := v1
                                  let mut v320 := v320
                                  let mut v321 := v321
                                  let mut v398 := v398
                                  let mut v399 := v399
                                  let mut v409 := v409
                                  let mut v410 := v410
                                  let mut v424 := v424
                                  let v221 := v11
                                  let v223 ← MemArray_index_ref_u64 r2 v212
                                  let v224 ← mloadInt (.u 64) v223
                                  let v225 := v224
                                  let v226 := v221
                                  let v227 := IntTy.subW (.u 128) v225 v226
                                  let v228 := r7
                                  let v229 := v228
                                  let v230 := IntTy.subW (.u 128) v227 v229
                                  let v231 := v230
                                  let v232 ← downcast_truncate_u128_u64 v231
                                  let (v234, v234_k) ← MemArray_index_mut_u64 r6 v212
                                  mstoreInt (.u 64) v234 v232
                                  let v234_0 ← v234_k v234
                                  r6 := v234_0
                                  let v235 := ()
                                  let v236 := (UInt256.ofNat 127)
                                  let v237 := v231 >>> v236
                                  let v238 ← downcast_truncate_u128_u64 v237
                                  let v239 := v238
                                  r7 := v239
                                  let v240 := ()
                                  let v242 ← Range_next_0 r21 v10
                                  let v243 := v242
                                  b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v243
                                if v214 then
                                  let v216 ← MemArray_index_ref_u64 r5 v212
                                  let v217 ← mloadInt (.u 64) v216
                                  let v218 := v217
                                  b44 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v218
                                else
                                  let v219 := (UInt256.ofNat 0)
                                  let v220 := v219
                                  b44 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v220
                              else
                                let v244 := (UInt256.ofNat 1)
                                let v245 := v49 - v244
                                let (v247, v247_k) ← MemArray_index_mut_u64 r6 v245
                                let v248 ← mloadInt (.u 64) v247
                                let v249 := v248 &&& v63
                                mstoreInt (.u 64) v247 v249
                                let v247_0 ← v247_k v247
                                r6 := v247_0
                                let v250 := ()
                                let v251 ← MemArray_try_new_zeroed_u64 v49
                                let v252 := match v251 with | .Some .. => true | _ => false
                                if v252 then
                                  let v253 := v251.Some_0
                                  let v254 := v253
                                  r8 := v254
                                  let v255 := (UInt256.ofNat 0)
                                  let v257 ← MemArray_index_ref_u64 r3 v255
                                  let v258 ← mloadInt (.u 64) v257
                                  r9 := v258
                                  let v259 := (UInt256.ofNat 0)
                                  let v260 := (UInt256.ofNat 5)
                                  let v261 := ()
                                  let v262 := (Range_Known_0_Known_5.mk v261 v261)
                                  r22 := v262
                                  let v264 ← Range_start_0_5 r22
                                  let v265 := v264
                                  let rec b47 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v12 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                    let mut v1 := v1
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
                                      let v270 := r9
                                      let v271 := v270
                                      let v272 := (UInt256.ofNat 2)
                                      let v273 := (UInt256.ofNat 0)
                                      let v275 ← MemArray_index_ref_u64 r3 v273
                                      let v276 ← mloadInt (.u 64) v275
                                      let v277 := r9
                                      let v278 := IntTy.mulW (.u 64) v276 v277
                                      let v279 := IntTy.subW (.u 64) v272 v278
                                      let v280 := IntTy.mulW (.u 64) v271 v279
                                      let v281 := v280
                                      r9 := v281
                                      let v282 := ()
                                      let v284 ← Range_next_0_5 r22 v12
                                      let v285 := v284
                                      b47 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v285
                                    else
                                      let v286 := r9
                                      let v287 := v286
                                      let v288 := (UInt256.ofNat 0)
                                      let (v290, v290_k) ← MemArray_index_mut_u64 r8 v288
                                      mstoreInt (.u 64) v290 v287
                                      let v290_0 ← v290_k v290
                                      r8 := v290_0
                                      let v291 := ()
                                      let v292 ← MemArray_try_new_zeroed_u64 v49
                                      let v293 := match v292 with | .Some .. => true | _ => false
                                      if v293 then
                                        let v294 := v292.Some_0
                                        let v295 := v294
                                        r10 := v295
                                        let v296 ← MemArray_try_new_zeroed_u64 v49
                                        let v297 := match v296 with | .Some .. => true | _ => false
                                        if v297 then
                                          let v298 := v296.Some_0
                                          let v299 := v298
                                          r11 := v299
                                          let v300 := (UInt256.ofNat 1)
                                          let v301 := v300
                                          r12 := v301
                                          let rec b55 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                            let mut v1 := v1
                                            let mut v320 := v320
                                            let mut v321 := v321
                                            let mut v398 := v398
                                            let mut v399 := v399
                                            let mut v409 := v409
                                            let mut v410 := v410
                                            let mut v424 := v424
                                            let v302 := r12
                                            let v303 := v302
                                            let v304 := decide (v303.toNat < v49.toNat)
                                            if v304 then
                                              let v305 := (UInt256.ofNat 2)
                                              let v306 := r12
                                              let v307 := v305 * v306
                                              let v308 := decide (v307.toNat < v49.toNat)
                                              let b60 : UInt256 → MemArray_u64 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → MemArray_u64 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → UInt256 → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Known_5 → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → Range_Known_0_Unknown → MemArray_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → MemSlice_u64 → UInt256 → FeM (Option_MemArray_u64 × MemArray_u64) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v13 => do
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
                                                let mut v1 := v1
                                                let mut v320 := v320
                                                let mut v321 := v321
                                                let mut v398 := v398
                                                let mut v399 := v399
                                                let mut v409 := v409
                                                let mut v410 := v410
                                                let mut v424 := v424
                                                let v314 := v13
                                                let v316 ← MemArray_as_slice_ref_u64 r3
                                                let v318 ← MemArray_as_slice_ref_u64 r8
                                                let (v320_g, v320_k) ← MemArray_as_slice_mut_u64 r10
                                                v320 := v320_g
                                                let (v321_g, v321_k) ← MemSlice_head_mut_u64 v320 v314
                                                v321 := v321_g
                                                let (v325, v325_0) ← low_product v316 v318 v321
                                                v321 := v325_0
                                                let v321_0 ← v321_k v321
                                                v320 := v321_0
                                                let v320_0 ← v320_k v320
                                                r10 := v320_0
                                                let v326 := (UInt256.ofNat 1)
                                                let v327 := v326
                                                r13 := v327
                                                let v328 := (UInt256.ofNat 0)
                                                let v329 := ()
                                                let v330 := (Range_Known_0_Unknown.mk v329 v314)
                                                r23 := v330
                                                let v332 ← Range_start_0 r23
                                                let v333 := v332
                                                let rec b61 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v14 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                  let mut v1 := v1
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
                                                    let v338 := v337
                                                    let v340 ← MemArray_index_ref_u64 r10 v338
                                                    let v341 ← mloadInt (.u 64) v340
                                                    let v342 := IntTy.not (.u 64) v341
                                                    let v343 := v342
                                                    let v344 := r13
                                                    let v345 := v344
                                                    let v346 := IntTy.addW (.u 128) v343 v345
                                                    let v347 := v346
                                                    let v348 ← downcast_truncate_u128_u64 v347
                                                    let (v350, v350_k) ← MemArray_index_mut_u64 r11 v338
                                                    mstoreInt (.u 64) v350 v348
                                                    let v350_0 ← v350_k v350
                                                    r11 := v350_0
                                                    let v351 := ()
                                                    let v352 := (UInt256.ofNat 64)
                                                    let v353 := v347 >>> v352
                                                    let v354 ← downcast_truncate_u128_u64 v353
                                                    let v355 := v354
                                                    r13 := v355
                                                    let v356 := ()
                                                    let v358 ← Range_next_0 r23 v14
                                                    let v359 := v358
                                                    b61 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v359
                                                  else
                                                    let v360 := (UInt256.ofNat 2)
                                                    let v361 := v360
                                                    r14 := v361
                                                    let v362 := (UInt256.ofNat 0)
                                                    let v363 := ()
                                                    let v364 := (Range_Known_0_Unknown.mk v363 v314)
                                                    r24 := v364
                                                    let v366 ← Range_start_0 r24
                                                    let v367 := v366
                                                    let rec b65 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v15 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                      let mut v1 := v1
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
                                                        let v372 := v371
                                                        let v374 ← MemArray_index_ref_u64 r11 v372
                                                        let v375 ← mloadInt (.u 64) v374
                                                        let v376 := v375
                                                        let v377 := r14
                                                        let v378 := v377
                                                        let v379 := IntTy.addW (.u 128) v376 v378
                                                        let v380 := v379
                                                        let v381 ← downcast_truncate_u128_u64 v380
                                                        let (v383, v383_k) ← MemArray_index_mut_u64 r11 v372
                                                        mstoreInt (.u 64) v383 v381
                                                        let v383_0 ← v383_k v383
                                                        r11 := v383_0
                                                        let v384 := ()
                                                        let v385 := (UInt256.ofNat 64)
                                                        let v386 := v380 >>> v385
                                                        let v387 ← downcast_truncate_u128_u64 v386
                                                        let v388 := v387
                                                        r14 := v388
                                                        let v389 := ()
                                                        let v391 ← Range_next_0 r24 v15
                                                        let v392 := v391
                                                        b65 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v392
                                                      else
                                                        let v394 ← MemArray_as_slice_ref_u64 r8
                                                        let v396 ← MemArray_as_slice_ref_u64 r11
                                                        let (v398_g, v398_k) ← MemArray_as_slice_mut_u64 r10
                                                        v398 := v398_g
                                                        let (v399_g, v399_k) ← MemSlice_head_mut_u64 v398 v314
                                                        v399 := v399_g
                                                        let (v403, v403_0) ← low_product v394 v396 v399
                                                        v399 := v403_0
                                                        let v399_0 ← v399_k v399
                                                        v398 := v399_0
                                                        let v398_0 ← v398_k v398
                                                        r10 := v398_0
                                                        let v405 ← MemArray_as_slice_ref_u64 r10
                                                        let v407 ← MemSlice_head_ref_u64 v405 v314
                                                        let (v409_g, v409_k) ← MemArray_as_slice_mut_u64 r8
                                                        v409 := v409_g
                                                        let (v410_g, v410_k) ← MemSlice_head_mut_u64 v409 v314
                                                        v410 := v410_g
                                                        let (v412, v412_0) ← MemSlice_copy_from_u64 v410 v407
                                                        v410 := v412_0
                                                        let v410_0 ← v410_k v410
                                                        v409 := v410_0
                                                        let v409_0 ← v409_k v409
                                                        r8 := v409_0
                                                        let v413 := v314
                                                        r12 := v413
                                                        let v414 := ()
                                                        b55 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                                                    partial_fixpoint
                                                    b65 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v367
                                                partial_fixpoint
                                                b61 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v333
                                              if v308 then
                                                let v309 := (UInt256.ofNat 2)
                                                let v310 := r12
                                                let v311 := v309 * v310
                                                let v312 := v311
                                                b60 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v312
                                              else
                                                let v313 := v49
                                                b60 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v313
                                            else
                                              let v415 ← MemArray_try_new_zeroed_u64 v49
                                              let v416 := match v415 with | .Some .. => true | _ => false
                                              if v416 then
                                                let v417 := v415.Some_0
                                                let v418 := v417
                                                r15 := v418
                                                let v420 ← MemArray_as_slice_ref_u64 r6
                                                let v422 ← MemArray_as_slice_ref_u64 r8
                                                let (v424_g, v424_k) ← MemArray_as_slice_mut_u64 r15
                                                v424 := v424_g
                                                let (v428, v428_0) ← low_product v420 v422 v424
                                                v424 := v428_0
                                                let v424_0 ← v424_k v424
                                                r15 := v424_0
                                                let v429 := (UInt256.ofNat 1)
                                                let v430 := v49 - v429
                                                let (v432, v432_k) ← MemArray_index_mut_u64 r15 v430
                                                let v433 ← mloadInt (.u 64) v432
                                                let v434 := v433 &&& v63
                                                mstoreInt (.u 64) v432 v434
                                                let v432_0 ← v432_k v432
                                                r15 := v432_0
                                                let v435 := ()
                                                let v436 := r4
                                                let v437 := v436
                                                let v438 := v437 + v49
                                                let v439 := (UInt256.ofNat 1)
                                                let v440 := v438 + v439
                                                let v441 ← MemArray_try_new_zeroed_u64 v440
                                                let v442 := match v441 with | .Some .. => true | _ => false
                                                if v442 then
                                                  let v443 := v441.Some_0
                                                  let v444 := v443
                                                  r16 := v444
                                                  let v445 := (UInt256.ofNat 0)
                                                  let v446 := ()
                                                  let v447 := r4
                                                  let v448 := (Range_Known_0_Unknown.mk v446 v447)
                                                  r25 := v448
                                                  let v450 ← Range_start_0 r25
                                                  let v451 := v450
                                                  let rec b73 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v16 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                    let mut v1 := v1
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
                                                      let v456 := v455
                                                      let v458 ← MemArray_index_ref_u64 r5 v456
                                                      let v459 ← mloadInt (.u 64) v458
                                                      let (v461, v461_k) ← MemArray_index_mut_u64 r16 v456
                                                      mstoreInt (.u 64) v461 v459
                                                      let v461_0 ← v461_k v461
                                                      r16 := v461_0
                                                      let v462 := ()
                                                      let v464 ← Range_next_0 r25 v16
                                                      let v465 := v464
                                                      b73 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v465
                                                    else
                                                      let v466 := (UInt256.ofNat 0)
                                                      let v467 := ()
                                                      let v468 := r4
                                                      let v469 := (Range_Known_0_Unknown.mk v467 v468)
                                                      r26 := v469
                                                      let v471 ← Range_start_0 r26
                                                      let v472 := v471
                                                      let rec b77 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v17 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                        let mut v1 := v1
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
                                                          let v477 := v476
                                                          let v478 := (UInt256.ofNat 0)
                                                          let v479 := v478
                                                          r17 := v479
                                                          let v480 := (UInt256.ofNat 0)
                                                          let v481 := ()
                                                          let v482 := (Range_Known_0_Unknown.mk v481 v49)
                                                          r27 := v482
                                                          let v484 ← Range_start_0 r27
                                                          let v485 := v484
                                                          let rec b81 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v18 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                            let mut v1 := v1
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
                                                              let v490 := v489
                                                              let v491 := v477 + v490
                                                              let v493 ← MemArray_index_ref_u64 r16 v491
                                                              let v494 ← mloadInt (.u 64) v493
                                                              let v495 := v494
                                                              let v497 ← MemArray_index_ref_u64 r3 v477
                                                              let v498 ← mloadInt (.u 64) v497
                                                              let v499 := v498
                                                              let v501 ← MemArray_index_ref_u64 r15 v490
                                                              let v502 ← mloadInt (.u 64) v501
                                                              let v503 := v502
                                                              let v504 := IntTy.mulW (.u 128) v499 v503
                                                              let v505 := IntTy.addW (.u 128) v495 v504
                                                              let v506 := r17
                                                              let v507 := v506
                                                              let v508 := IntTy.addW (.u 128) v505 v507
                                                              let v509 := v508
                                                              let v510 ← downcast_truncate_u128_u64 v509
                                                              let v511 := v477 + v490
                                                              let (v513, v513_k) ← MemArray_index_mut_u64 r16 v511
                                                              mstoreInt (.u 64) v513 v510
                                                              let v513_0 ← v513_k v513
                                                              r16 := v513_0
                                                              let v514 := ()
                                                              let v515 := (UInt256.ofNat 64)
                                                              let v516 := v509 >>> v515
                                                              let v517 ← downcast_truncate_u128_u64 v516
                                                              let v518 := v517
                                                              r17 := v518
                                                              let v519 := ()
                                                              let v521 ← Range_next_0 r27 v18
                                                              let v522 := v521
                                                              b81 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v522
                                                            else
                                                              let v523 := v477 + v49
                                                              let v524 := v523
                                                              r18 := v524
                                                              let rec b85 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                                let mut v1 := v1
                                                                let mut v320 := v320
                                                                let mut v321 := v321
                                                                let mut v398 := v398
                                                                let mut v399 := v399
                                                                let mut v409 := v409
                                                                let mut v410 := v410
                                                                let mut v424 := v424
                                                                let v525 := r17
                                                                let v526 := v525
                                                                let v527 := (UInt256.ofNat 0)
                                                                let v528 := decide (v526 ≠ v527)
                                                                if v528 then
                                                                  let v529 := r18
                                                                  let v530 := v529
                                                                  let v532 ← MemArray_index_ref_u64 r16 v530
                                                                  let v533 ← mloadInt (.u 64) v532
                                                                  let v534 := v533
                                                                  let v535 := r17
                                                                  let v536 := v535
                                                                  let v537 := IntTy.addW (.u 128) v534 v536
                                                                  let v538 := v537
                                                                  let v539 ← downcast_truncate_u128_u64 v538
                                                                  let v540 := r18
                                                                  let v541 := v540
                                                                  let (v543, v543_k) ← MemArray_index_mut_u64 r16 v541
                                                                  mstoreInt (.u 64) v543 v539
                                                                  let v543_0 ← v543_k v543
                                                                  r16 := v543_0
                                                                  let v544 := ()
                                                                  let v545 := (UInt256.ofNat 64)
                                                                  let v546 := v538 >>> v545
                                                                  let v547 ← downcast_truncate_u128_u64 v546
                                                                  let v548 := v547
                                                                  r17 := v548
                                                                  let v549 := ()
                                                                  let v550 := (UInt256.ofNat 1)
                                                                  let v551 := r18
                                                                  let v552 := v551 + v550
                                                                  let v553 := v552
                                                                  r18 := v553
                                                                  let v554 := ()
                                                                  b85 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                                                                else
                                                                  let v555 := ()
                                                                  let v557 ← Range_next_0 r26 v17
                                                                  let v558 := v557
                                                                  b77 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v558
                                                              partial_fixpoint
                                                              b85 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                                                          partial_fixpoint
                                                          b81 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v485
                                                        else
                                                          let v559 := (UInt256.ofNat 0)
                                                          let v560 := v4
                                                          let v561 := ()
                                                          let v562 := (Range_Known_0_Unknown.mk v561 v560)
                                                          r28 := v562
                                                          let v564 ← Range_start_0 r28
                                                          let v565 := v564
                                                          let rec b88 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v19 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                                                            let mut v1 := v1
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
                                                              let v570 := v569
                                                              let v572 ← MemArray_index_ref_u64 r16 v570
                                                              let v573 ← mloadInt (.u 64) v572
                                                              let (v575, v575_k) ← MemArray_index_mut_u64 r1 v570
                                                              mstoreInt (.u 64) v575 v573
                                                              let v575_0 ← v575_k v575
                                                              r1 := v575_0
                                                              let v576 := ()
                                                              let v578 ← Range_next_0 r28 v19
                                                              let v579 := v578
                                                              b88 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v579
                                                            else
                                                              let v580 := r1
                                                              let v581 := v580
                                                              let v582 := Option_MemArray_u64.Some v581
                                                              let _ ← MemArray_drop_u64 r16
                                                              let _ ← MemArray_drop_u64 r15
                                                              let _ ← MemArray_drop_u64 r11
                                                              let _ ← MemArray_drop_u64 r10
                                                              let _ ← MemArray_drop_u64 r8
                                                              let _ ← MemArray_drop_u64 r6
                                                              let _ ← MemArray_drop_u64 r5
                                                              let _ ← MemArray_drop_u64 r3
                                                              let _ ← MemArray_drop_u64 r2
                                                              return (v582, v1)
                                                          partial_fixpoint
                                                          b88 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v565
                                                      partial_fixpoint
                                                      b77 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v472
                                                  partial_fixpoint
                                                  b73 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v451
                                                else
                                                  let v583 := Option_MemArray_u64.None
                                                  match (← pure v441) with
                                                  | .Some .. =>
                                                    let _ ← MemArray_drop_u64 v441.Some_0
                                                    pure ()
                                                  | _ => pure ()
                                                  let _ ← MemArray_drop_u64 r15
                                                  let _ ← MemArray_drop_u64 r11
                                                  let _ ← MemArray_drop_u64 r10
                                                  let _ ← MemArray_drop_u64 r8
                                                  let _ ← MemArray_drop_u64 r6
                                                  let _ ← MemArray_drop_u64 r5
                                                  let _ ← MemArray_drop_u64 r3
                                                  let _ ← MemArray_drop_u64 r2
                                                  let _ ← MemArray_drop_u64 r1
                                                  return (v583, v1)
                                              else
                                                let v584 := Option_MemArray_u64.None
                                                match (← pure v415) with
                                                | .Some .. =>
                                                  let _ ← MemArray_drop_u64 v415.Some_0
                                                  pure ()
                                                | _ => pure ()
                                                let _ ← MemArray_drop_u64 r11
                                                let _ ← MemArray_drop_u64 r10
                                                let _ ← MemArray_drop_u64 r8
                                                let _ ← MemArray_drop_u64 r6
                                                let _ ← MemArray_drop_u64 r5
                                                let _ ← MemArray_drop_u64 r3
                                                let _ ← MemArray_drop_u64 r2
                                                let _ ← MemArray_drop_u64 r1
                                                return (v584, v1)
                                          partial_fixpoint
                                          b55 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                                        else
                                          let v585 := Option_MemArray_u64.None
                                          match (← pure v296) with
                                          | .Some .. =>
                                            let _ ← MemArray_drop_u64 v296.Some_0
                                            pure ()
                                          | _ => pure ()
                                          let _ ← MemArray_drop_u64 r10
                                          let _ ← MemArray_drop_u64 r8
                                          let _ ← MemArray_drop_u64 r6
                                          let _ ← MemArray_drop_u64 r5
                                          let _ ← MemArray_drop_u64 r3
                                          let _ ← MemArray_drop_u64 r2
                                          let _ ← MemArray_drop_u64 r1
                                          return (v585, v1)
                                      else
                                        let v586 := Option_MemArray_u64.None
                                        match (← pure v292) with
                                        | .Some .. =>
                                          let _ ← MemArray_drop_u64 v292.Some_0
                                          pure ()
                                        | _ => pure ()
                                        let _ ← MemArray_drop_u64 r8
                                        let _ ← MemArray_drop_u64 r6
                                        let _ ← MemArray_drop_u64 r5
                                        let _ ← MemArray_drop_u64 r3
                                        let _ ← MemArray_drop_u64 r2
                                        let _ ← MemArray_drop_u64 r1
                                        return (v586, v1)
                                  partial_fixpoint
                                  b47 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v265
                                else
                                  let v587 := Option_MemArray_u64.None
                                  match (← pure v251) with
                                  | .Some .. =>
                                    let _ ← MemArray_drop_u64 v251.Some_0
                                    pure ()
                                  | _ => pure ()
                                  let _ ← MemArray_drop_u64 r6
                                  let _ ← MemArray_drop_u64 r5
                                  let _ ← MemArray_drop_u64 r3
                                  let _ ← MemArray_drop_u64 r2
                                  let _ ← MemArray_drop_u64 r1
                                  return (v587, v1)
                            partial_fixpoint
                            b38 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v207
                          else
                            let v588 := Option_MemArray_u64.None
                            match (← pure v196) with
                            | .Some .. =>
                              let _ ← MemArray_drop_u64 v196.Some_0
                              pure ()
                            | _ => pure ()
                            let _ ← MemArray_drop_u64 r5
                            let _ ← MemArray_drop_u64 r3
                            let _ ← MemArray_drop_u64 r2
                            let _ ← MemArray_drop_u64 r1
                            return (v588, v1)
                        else
                          let v589 := Option_MemArray_u64.None
                          match (← pure v192) with
                          | .Some .. =>
                            let _ ← MemArray_drop_u64 v192.Some_0
                            pure ()
                          | _ => pure ()
                          let _ ← MemArray_drop_u64 r3
                          let _ ← MemArray_drop_u64 r2
                          let _ ← MemArray_drop_u64 r1
                          return (v589, v1)
                      if v155 then
                        let v156 := (UInt256.ofNat 0)
                        let v158 ← MemArray_index_ref_u64 r3 v156
                        let v159 ← mloadInt (.u 64) v158
                        let v160 := (UInt256.ofNat 1)
                        let v161 := decide (v159 = v160)
                        if v161 then
                          let v162 := (UInt256.ofNat 0)
                          let v163 := ()
                          let v164 := (Range_Known_0_Unknown.mk v163 v49)
                          r20 := v164
                          let v166 ← Range_start_0 r20
                          let v167 := v166
                          let rec b30 (r0 : UInt256) (r1 : MemArray_u64) (r2 : MemArray_u64) (r3 : MemArray_u64) (r4 : UInt256) (r5 : MemArray_u64) (r6 : MemArray_u64) (r7 : UInt256) (r8 : MemArray_u64) (r9 : UInt256) (r10 : MemArray_u64) (r11 : MemArray_u64) (r12 : UInt256) (r13 : UInt256) (r14 : UInt256) (r15 : MemArray_u64) (r16 : MemArray_u64) (r17 : UInt256) (r18 : UInt256) (r19 : Range_Known_0_Unknown) (r20 : Range_Known_0_Unknown) (r21 : Range_Known_0_Unknown) (r22 : Range_Known_0_Known_5) (r23 : Range_Known_0_Unknown) (r24 : Range_Known_0_Unknown) (r25 : Range_Known_0_Unknown) (r26 : Range_Known_0_Unknown) (r27 : Range_Known_0_Unknown) (r28 : Range_Known_0_Unknown) (v1 : MemArray_u64) (v320 : MemSlice_u64) (v321 : MemSlice_u64) (v398 : MemSlice_u64) (v399 : MemSlice_u64) (v409 : MemSlice_u64) (v410 : MemSlice_u64) (v424 : MemSlice_u64) (v9 : UInt256) : FeM (Option_MemArray_u64 × MemArray_u64) := do
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
                            let mut v1 := v1
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
                              let v172 := v171
                              let v174 ← MemArray_index_ref_u64 r2 v172
                              let v175 ← mloadInt (.u 64) v174
                              let (v177, v177_k) ← MemArray_index_mut_u64 r1 v172
                              mstoreInt (.u 64) v177 v175
                              let v177_0 ← v177_k v177
                              r1 := v177_0
                              let v178 := ()
                              let v180 ← Range_next_0 r20 v9
                              let v181 := v180
                              b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v181
                            else
                              let v182 := r1
                              let v183 := v182
                              let v184 := Option_MemArray_u64.Some v183
                              let _ ← MemArray_drop_u64 r3
                              let _ ← MemArray_drop_u64 r2
                              return (v184, v1)
                          partial_fixpoint
                          b30 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v167
                        else
                          b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                      else
                        b27 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
                  partial_fixpoint
                  b23 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424
              partial_fixpoint
              b13 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v87
            else
              let v590 := Option_MemArray_u64.None
              match (← pure v74) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v74.Some_0
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 r2
              let _ ← MemArray_drop_u64 r1
              return (v590, v1)
          else
            let v591 := Option_MemArray_u64.None
            match (← pure v70) with
            | .Some .. =>
              let _ ← MemArray_drop_u64 v70.Some_0
              pure ()
            | _ => pure ()
            let _ ← MemArray_drop_u64 r1
            return (v591, v1)
        else
          let v592 := Option_MemArray_u64.None
          match (← pure v64) with
          | .Some .. =>
            let _ ← MemArray_drop_u64 v64.Some_0
            pure ()
          | _ => pure ()
          return (v592, v1)
      if v55 then
        let v56 := (UInt256.ofNat 18446744073709551615)
        let v57 := v56
        b6 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v57
      else
        let v58 := (UInt256.ofNat 1)
        let v59 := IntTy.shl (.u 64) v58 v53
        let v60 := (UInt256.ofNat 1)
        let v61 := IntTy.subW (.u 64) v59 v60
        let v62 := v61
        b6 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424 v62
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 v1 v320 v321 v398 v399 v409 v410 v424

def power_MemBuffer (v0 : MemBuffer) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM Option_MemArray_u64 := do
  let mut r0 : MemArray_u64 := default
  let mut r1 : UInt256 := default
  let mut r2 : Exponent := default
  let mut r3 : MemArray_u64 := default
  let mut r4 : MemArray_u64 := default
  let mut r5 : UInt256 := default
  let v5 := (UInt256.ofNat 96)
  let v6 := v1
  let v7 ← checkedAdd v5 v6
  let v8 := v7
  let v9 := v2
  let v10 ← checkedAdd v8 v9
  let v11 := v10
  let v12 := v3
  let v13 := (UInt256.ofNat 7)
  let v14 ← checkedAdd v12 v13
  let v15 := (UInt256.ofNat 8)
  let v16 ← checkedDiv v14 v15
  let v17 ← input_limbs_MemBuffer v0 v11 v3 v16
  let v18 := match v17 with | .Some .. => true | _ => false
  if v18 then
    let v19 := v17.Some_0
    let v20 := v19
    r0 := v20
    let v22 ← MemArray_len_u64 r0
    let v23 := v22
    r1 := v23
    let rec b3 (r0 : MemArray_u64) (r1 : UInt256) (r2 : Exponent) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) : FeM Option_MemArray_u64 := do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let mut r4 := r4
      let mut r5 := r5
      let v24 := r1
      let v25 := v24
      let v26 := (UInt256.ofNat 0)
      let v27 := decide (v26.toNat < v25.toNat)
      let b5 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let v42 := r1
        let v43 := v42
        let v44 := (UInt256.ofNat 0)
        let v45 := decide (v43 = v44)
        let b7 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let v155 := (UInt256.ofNat 1)
          let v156 ← MemArray_try_new_zeroed_u64 v155
          let _ ← MemArray_drop_u64 r0
          return v156
        if v45 then
          b7 r0 r1 r2 r3 r4 r5
        else
          let v46 := r1
          let v47 := v46
          let v48 := (UInt256.ofNat 1)
          let v49 := decide (v47 = v48)
          let b8 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let v56 := ()
            let v57 := v56
            let v58 ← Exponent_try_read_MemBuffer v0 v8 v2
            let v59 := match v58 with | .Some .. => true | _ => false
            if v59 then
              let v60 := v58.Some_0
              let v61 := v60
              r2 := v61
              let v62 := r2.top
              let v63 := r2.total
              let v64 := decide (v62 = v63)
              if v64 then
                let v65 := (UInt256.ofNat 1)
                let v66 ← MemArray_try_new_zeroed_u64 v65
                let v67 := match v66 with | .Some .. => true | _ => false
                if v67 then
                  let v68 := v66.Some_0
                  let v69 := v68
                  r3 := v69
                  let v70 := (UInt256.ofNat 1)
                  let v71 := (UInt256.ofNat 0)
                  let (v73, v73_k) ← MemArray_index_mut_u64 r3 v71
                  mstoreInt (.u 64) v73 v70
                  let v73_0 ← v73_k v73
                  r3 := v73_0
                  let v74 := ()
                  let v75 := r3
                  let v76 := v75
                  let v77 := Option_MemArray_u64.Some v76
                  let _ ← MemArray_drop_u64 r2.limbs
                  let _ ← MemArray_drop_u64 r0
                  return v77
                else
                  let v78 := Option_MemArray_u64.None
                  match (← pure v66) with
                  | .Some .. =>
                    let _ ← MemArray_drop_u64 v66.Some_0
                    pure ()
                  | _ => pure ()
                  let _ ← MemArray_drop_u64 r2.limbs
                  let _ ← MemArray_drop_u64 r0
                  return v78
              else
                let v79 := ()
                let v80 := v79
                let v81 := v1
                let v82 := (UInt256.ofNat 7)
                let v83 ← checkedAdd v81 v82
                let v84 := (UInt256.ofNat 8)
                let v85 ← checkedDiv v83 v84
                let v86 := v85
                let v87 := (UInt256.ofNat 96)
                let v88 := r1
                let v89 := decide (v88.toNat < v86.toNat)
                let b21 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 v4 => do
                  let mut r0 := r0
                  let mut r1 := r1
                  let mut r2 := r2
                  let mut r3 := r3
                  let mut r4 := r4
                  let mut r5 := r5
                  let v94 := (UInt256.ofNat 1)
                  let v95 ← checkedAdd v4 v94
                  let v96 ← input_limbs_MemBuffer v0 v87 v1 v95
                  let v97 := match v96 with | .Some .. => true | _ => false
                  if v97 then
                    let v98 := v96.Some_0
                    let v99 := v98
                    r4 := v99
                    let v100 := v86
                    r5 := v100
                    let rec b24 (r0 : MemArray_u64) (r1 : UInt256) (r2 : Exponent) (r3 : MemArray_u64) (r4 : MemArray_u64) (r5 : UInt256) : FeM Option_MemArray_u64 := do
                      let mut r0 := r0
                      let mut r1 := r1
                      let mut r2 := r2
                      let mut r3 := r3
                      let mut r4 := r4
                      let mut r5 := r5
                      let v101 := r5
                      let v102 := v101
                      let v103 := (UInt256.ofNat 0)
                      let v104 := decide (v103.toNat < v102.toNat)
                      let b26 : MemArray_u64 → UInt256 → Exponent → MemArray_u64 → MemArray_u64 → UInt256 → FeM Option_MemArray_u64 := fun r0 r1 r2 r3 r4 r5 => do
                        let mut r0 := r0
                        let mut r1 := r1
                        let mut r2 := r2
                        let mut r3 := r3
                        let mut r4 := r4
                        let mut r5 := r5
                        let v119 := r5
                        let v120 := v119
                        let v121 := (UInt256.ofNat 0)
                        let v122 := decide (v120 = v121)
                        if v122 then
                          let v123 := (UInt256.ofNat 1)
                          let v124 ← MemArray_try_new_zeroed_u64 v123
                          let _ ← MemArray_drop_u64 r4
                          let _ ← MemArray_drop_u64 r2.limbs
                          let _ ← MemArray_drop_u64 r0
                          return v124
                        else
                          let v125 := ()
                          let v126 := v125
                          let v127 := (UInt256.ofNat 0)
                          let v129 ← MemArray_index_ref_u64 r0 v127
                          let v130 ← mloadInt (.u 64) v129
                          let v131 := (UInt256.ofNat 1)
                          let v132 := v130 &&& v131
                          let v133 := (UInt256.ofNat 1)
                          let v134 := decide (v132 = v133)
                          if v134 then
                            let v136 := r5
                            let v137 := v136
                            let v138 := r1
                            let v139 := v138
                            let (v142, v142_0) ← odd_power r2 r4 v137 r0 v139
                            r4 := v142_0
                            let _ ← MemArray_drop_u64 r4
                            let _ ← MemArray_drop_u64 r2.limbs
                            let _ ← MemArray_drop_u64 r0
                            return v142
                          else
                            let v143 := ()
                            let v144 := v143
                            let v146 := r5
                            let v147 := v146
                            let v148 := r1
                            let v149 := v148
                            let (v152, v152_0) ← even_power r2 r4 v147 r0 v149
                            r4 := v152_0
                            let _ ← MemArray_drop_u64 r4
                            let _ ← MemArray_drop_u64 r2.limbs
                            let _ ← MemArray_drop_u64 r0
                            return v152
                      if v104 then
                        let v105 := r5
                        let v106 := v105
                        let v107 := (UInt256.ofNat 1)
                        let v108 ← checkedSub v106 v107
                        let v110 ← MemArray_index_ref_u64 r4 v108
                        let v111 ← mloadInt (.u 64) v110
                        let v112 := (UInt256.ofNat 0)
                        let v113 := decide (v111 = v112)
                        if v113 then
                          let v114 := (UInt256.ofNat 1)
                          let v115 := r5
                          let v116 ← checkedSub v115 v114
                          let v117 := v116
                          r5 := v117
                          let v118 := ()
                          b24 r0 r1 r2 r3 r4 r5
                        else
                          b26 r0 r1 r2 r3 r4 r5
                      else
                        b26 r0 r1 r2 r3 r4 r5
                    partial_fixpoint
                    b24 r0 r1 r2 r3 r4 r5
                  else
                    let v153 := Option_MemArray_u64.None
                    match (← pure v96) with
                    | .Some .. =>
                      let _ ← MemArray_drop_u64 v96.Some_0
                      pure ()
                    | _ => pure ()
                    let _ ← MemArray_drop_u64 r2.limbs
                    let _ ← MemArray_drop_u64 r0
                    return v153
                if v89 then
                  let v90 := v86
                  b21 r0 r1 r2 r3 r4 r5 v90
                else
                  let v91 := r1
                  let v92 := v91
                  let v93 := v92
                  b21 r0 r1 r2 r3 r4 r5 v93
            else
              let v154 := Option_MemArray_u64.None
              match (← pure v58) with
              | .Some .. =>
                let _ ← MemArray_drop_u64 v58.Some_0.limbs
                pure ()
              | _ => pure ()
              let _ ← MemArray_drop_u64 r0
              return v154
          if v49 then
            let v50 := (UInt256.ofNat 0)
            let v52 ← MemArray_index_ref_u64 r0 v50
            let v53 ← mloadInt (.u 64) v52
            let v54 := (UInt256.ofNat 1)
            let v55 := decide (v53 = v54)
            if v55 then
              b7 r0 r1 r2 r3 r4 r5
            else
              b8 r0 r1 r2 r3 r4 r5
          else
            b8 r0 r1 r2 r3 r4 r5
      if v27 then
        let v28 := r1
        let v29 := v28
        let v30 := (UInt256.ofNat 1)
        let v31 ← checkedSub v29 v30
        let v33 ← MemArray_index_ref_u64 r0 v31
        let v34 ← mloadInt (.u 64) v33
        let v35 := (UInt256.ofNat 0)
        let v36 := decide (v34 = v35)
        if v36 then
          let v37 := (UInt256.ofNat 1)
          let v38 := r1
          let v39 ← checkedSub v38 v37
          let v40 := v39
          r1 := v40
          let v41 := ()
          b3 r0 r1 r2 r3 r4 r5
        else
          b5 r0 r1 r2 r3 r4 r5
      else
        b5 r0 r1 r2 r3 r4 r5
    partial_fixpoint
    b3 r0 r1 r2 r3 r4 r5
  else
    let v157 := Option_MemArray_u64.None
    match (← pure v17) with
    | .Some .. =>
      let _ ← MemArray_drop_u64 v17.Some_0
      pure ()
    | _ => pure ()
    return v157

def encoded (v0 : MemArray_u64) (r0 : MemBuffer) : FeM MemBuffer := do
  let mut r0 := r0
  let mut r1 : Range_Known_0_Unknown := default
  let mut r2 : Range_Known_0_Known_8 := default
  let v4 ← MemBuffer_len r0
  let v5 := v4
  let v7 ← MemBuffer_ptr r0
  let v8 := v7
  let v9 := (UInt256.ofNat 0)
  let v10 ← MemArray_len_u64 v0
  let v11 := ()
  let v12 := (Range_Known_0_Unknown.mk v11 v10)
  r1 := v12
  let v14 ← Range_start_0 r1
  let v15 := v14
  let rec b1 (r0 : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) (v1 : UInt256) : FeM MemBuffer := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let v17 ← Range_has_0 r1 v1
    if v17 then
      let v19 ← Range_at_0 r1 v1
      let v20 := v19
      let v21 ← MemArray_index_ref_u64 v0 v20
      let v22 ← mloadInt (.u 64) v21
      let v23 := (UInt256.ofNat 8)
      let v24 := v23 * v20
      let v25 := (UInt256.ofNat 8)
      let v26 := v24 + v25
      let v27 := decide (v26.toNat ≤ v5.toNat)
      let rec b7 (r0 : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) : FeM MemBuffer := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let v75 ← Range_next_0 r1 v1
        let v76 := v75
        b1 r0 r1 r2 v76
      partial_fixpoint
      if v27 then
        let v28 := v22
        let v29 := (UInt256.ofNat 8)
        let v30 := v29 * v20
        let v31 := v5 - v30
        let v32 := (UInt256.ofNat 8)
        let v33 := v31 - v32
        let v34 := v8 + v33
        let v35 ← cast_u8_u64 v34
        mstoreInt (.u 64) v35 v28
        let v36 := ()
        let v37 := v36
        b7 r0 r1 r2
      else
        let v38 := (UInt256.ofNat 0)
        let v39 := (UInt256.ofNat 8)
        let v40 := ()
        let v41 := (Range_Known_0_Known_8.mk v40 v40)
        r2 := v41
        let v43 ← Range_start_0_8 r2
        let v44 := v43
        let rec b8 (r0 : MemBuffer) (r1 : Range_Known_0_Unknown) (r2 : Range_Known_0_Known_8) (v2 : UInt256) : FeM MemBuffer := do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let v46 ← Range_has_0_8 r2 v2
          if v46 then
            let v48 ← Range_at_0_8 r2 v2
            let v49 := v48
            let v50 := (UInt256.ofNat 8)
            let v51 := v50 * v20
            let v52 := v51 + v49
            let v53 := v52
            let v54 := decide (v53.toNat < v5.toNat)
            let b14 : MemBuffer → Range_Known_0_Unknown → Range_Known_0_Known_8 → FeM MemBuffer := fun r0 r1 r2 => do
              let mut r0 := r0
              let mut r1 := r1
              let mut r2 := r2
              let v70 ← Range_next_0_8 r2 v2
              let v71 := v70
              b8 r0 r1 r2 v71
            if v54 then
              let v55 := (UInt256.ofNat 8)
              let v56 := v55 * v49
              let v57 ← downcast_truncate_usize_u64 v56
              let v58 := v57
              let v59 := v22 >>> v58
              let v60 ← downcast_truncate_u64_u8 v59
              let v61 := (UInt256.ofNat 1)
              let v62 := v5 - v61
              let v63 := v62 - v53
              let v64 := v8 + v63
              mstoreInt (.u 8) v64 v60
              let v65 := ()
              let v66 := v65
              b14 r0 r1 r2
            else
              let v67 := ()
              let v68 := v67
              b14 r0 r1 r2
          else
            let v72 := ()
            let v73 := v72
            b7 r0 r1 r2
        partial_fixpoint
        b8 r0 r1 r2 v44
    else
      let v77 := r0
      return v77
  partial_fixpoint
  b1 r0 r1 r2 v15

def run_MemBuffer (v0 : MemBuffer) (v1 : UInt256) : FeM Output := do
  let mut r0 : MemArray_u64 := default
  let v2 := (UInt256.ofNat 0)
  let v3 := (UInt256.ofNat 32)
  let v4 ← input_word_MemBuffer v0 v2 v3
  let v5 := v4
  let v6 := (UInt256.ofNat 32)
  let v7 := (UInt256.ofNat 32)
  let v8 ← input_word_MemBuffer v0 v6 v7
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 := (UInt256.ofNat 32)
  let v12 ← input_word_MemBuffer v0 v10 v11
  let v13 := v12
  let v14 ← MAX_LENGTH
  let v15 := decide (v14.toNat < v5.toNat)
  let b1 : MemArray_u64 → FeM Output := fun r0 => do
    let mut r0 := r0
    let v60 ← FAILURE
    let v61 := v1
    let v62 ← MemBuffer_empty
    let v63 := (Output.mk v60 v61 v62)
    return v63
  if v15 then
    b1 r0
  else
    let v16 ← MAX_LENGTH
    let v17 := decide (v16.toNat < v9.toNat)
    if v17 then
      b1 r0
    else
      let v18 ← MAX_LENGTH
      let v19 := decide (v18.toNat < v13.toNat)
      if v19 then
        b1 r0
      else
        let v20 := ()
        let v21 := v20
        let v22 ← downcast_unchecked_u256_usize v5
        let v23 := v22
        let v24 ← downcast_unchecked_u256_usize v9
        let v25 := v24
        let v26 ← downcast_unchecked_u256_usize v13
        let v27 := v26
        let v28 ← gas_cost_MemBuffer v0 v23 v25 v27
        let v29 := v28
        let v30 := v1
        let v31 := decide (v30.toNat < v29.toNat)
        if v31 then
          let v32 ← FAILURE
          let v33 := v1
          let v34 ← MemBuffer_empty
          let v35 := (Output.mk v32 v33 v34)
          return v35
        else
          let v36 := ()
          let v37 := v36
          let v38 := (UInt256.ofNat 0)
          let v39 := decide (v27 = v38)
          if v39 then
            let v40 ← SUCCESS
            let v41 ← MemBuffer_empty
            let v42 := (Output.mk v40 v29 v41)
            return v42
          else
            let v43 := ()
            let v44 := v43
            let v45 ← MemBuffer_try_new v27
            let v46 := v45
            let v47 := v46
            match v47 with
            | .Some .. =>
              let v48 := v46.Some_0
              let v49 ← power_MemBuffer v0 v23 v25 v27
              let v50 := v49
              let v51 := v50
              match v51 with
              | .Some .. =>
                let v52 := v50.Some_0
                r0 := v52
                let v53 ← SUCCESS
                let v54 := v48
                let v56 ← encoded r0 v54
                let v57 := (Output.mk v53 v29 v56)
                let _ ← MemArray_drop_u64 r0
                return v57
              | _ =>
                let v58 ← host_allocation v29
                match v50 with
                | .Some .. =>
                  let _ ← MemArray_drop_u64 v50.Some_0
                  pure ()
                | _ => pure ()
                let _ ← MemBuffer_drop v48
                return v58
            | _ =>
              let v59 ← host_allocation v29
              match v46 with
              | .Some .. =>
                let _ ← MemBuffer_drop v46.Some_0
                pure ()
              | _ => pure ()
              return v59

def recv0_arm0 : FeM Empty := do
  let mut r1 : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_6
  r2 := v0
  let v1 := (UInt256.ofNat 18446744073709551615)
  let v3 ← run_MemBuffer r2 v1
  let v4 := v3
  r1 := v4
  let _ ← MemBuffer_drop r2
  let v5 := r1.status
  let v6 ← SUCCESS
  let v7 := decide (v5 ≠ v6)
  if v7 then
    let v8 ← MemSlice_empty_u8
    let v11 ← Evm_revert v8
    nomatch v11
  else
    let v12 := ()
    let v13 := v12
    let v16 ← MemBuffer_span_ref r1.output
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


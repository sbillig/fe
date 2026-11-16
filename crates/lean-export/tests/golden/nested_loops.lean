export Lib (Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start Range_at_0 Range_has_0 Range_next_0 Range_start_0 downcast_truncate_u256_usize)

def init : FeM Unit := do
  return ()

def windows (n_0 : UInt256) (width_1 : UInt256) : FeM UInt256 := do
  let mut acc : UInt256 := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut r3 : Range_Unknown_Unknown := default
  let v4 := (UInt256.ofNat 1)
  let acc_5 := v4
  acc := acc_5
  let v6 := (UInt256.ofNat 0)
  let index_7 := v6
  index := index_7
  let v8 := false
  let started_9 := v8
  started := started_9
  let rec b1 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
    let mut acc := acc
    let mut index := index
    let mut started := started
    let mut r3 := r3
    let index_10 := index
    let index_11 := index_10
    let n_12 := n_0
    let v13 := decide (index_11.toNat < n_12.toNat)
    if v13 then
      let index_14 := index
      let index_15 := index_14
      let width_16 := width_1
      let v17 ← checkedAdd index_15 width_16
      let v18 := (UInt256.ofNat 1)
      let v19 ← checkedAdd v17 v18
      let n_20 := n_0
      let v21 := decide (v19.toNat < n_20.toNat)
      let b6 : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun acc index started r3 v2 => do
        let mut acc := acc
        let mut index := index
        let mut started := started
        let mut r3 := r3
        let end_31 := v2
        let started_32 := started
        let rec b9 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
          let mut acc := acc
          let mut index := index
          let mut started := started
          let mut r3 := r3
          let v59 := (UInt256.ofNat 2)
          let v60 ← checkedRem end_31 v59
          let v61 := (UInt256.ofNat 1)
          let v62 := decide (v60 = v61)
          let b16 : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → FeM UInt256 := fun acc index started r3 => do
            let mut acc := acc
            let mut index := index
            let mut started := started
            let mut r3 := r3
            let end_75 := end_31
            index := end_75
            let v76 := ()
            b1 acc index started r3
          if v62 then
            let acc_63 := acc
            let acc_64 := acc_63
            let v65 := end_31
            let v66 ← checkedAdd acc_64 v65
            let acc_67 := v66
            acc := acc_67
            let v68 := ()
            let v69 := true
            let started_70 := v69
            started := started_70
            let v71 := ()
            let v72 := v71
            b16 acc index started r3
          else
            let v73 := ()
            let v74 := v73
            b16 acc index started r3
        partial_fixpoint
        if started_32 then
          let index_33 := index
          let index_34 := index_33
          let v35 := ()
          let v36 := (Range_Unknown_Unknown.mk index_34 end_31)
          r3 := v36
          let v38 ← Range_Unknown_Unknown_start r3
          let v39 := v38
          let rec b10 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) (v3 : UInt256) : FeM UInt256 := do
            let mut acc := acc
            let mut index := index
            let mut started := started
            let mut r3 := r3
            let v41 ← Range_Unknown_Unknown_has r3 v3
            if v41 then
              let v43 ← Range_Unknown_Unknown_at r3 v3
              let acc_44 := acc
              let acc_45 := acc_44
              let v46 := (UInt256.ofNat 2)
              let v47 ← checkedMul acc_45 v46
              let v48 := (UInt256.ofNat 1000003)
              let v49 ← checkedRem v47 v48
              let acc_50 := v49
              acc := acc_50
              let v51 := ()
              let v53 ← Range_Unknown_Unknown_next r3 v3
              let v54 := v53
              b10 acc index started r3 v54
            else
              let v55 := ()
              let v56 := v55
              b9 acc index started r3
          partial_fixpoint
          b10 acc index started r3 v39
        else
          let v57 := ()
          let v58 := v57
          b9 acc index started r3
      if v21 then
        let index_22 := index
        let index_23 := index_22
        let width_24 := width_1
        let v25 ← checkedAdd index_23 width_24
        let v26 := (UInt256.ofNat 1)
        let v27 ← checkedAdd v25 v26
        let v28 := v27
        b6 acc index started r3 v28
      else
        let n_29 := n_0
        let n_30 := n_29
        b6 acc index started r3 n_30
    else
      let acc_77 := acc
      return acc_77
  partial_fixpoint
  b1 acc index started r3

def recv_Windows (n : UInt256) (width : UInt256) : FeM UInt256 := do
  let v2 ← downcast_truncate_u256_usize n
  let v3 ← downcast_truncate_u256_usize width
  let v4 ← windows v2 v3
  return v4

def break_both (n_0 : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut found : UInt256 := default
  let mut done : Bool := default
  let mut j : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let i_2 := v1
  i := i_2
  let v3 := (UInt256.ofNat 0)
  let found_4 := v3
  found := found_4
  let v5 := false
  let done_6 := v5
  done := done_6
  let rec b1 (i : UInt256) (found : UInt256) (done : Bool) (j : UInt256) : FeM UInt256 := do
    let mut i := i
    let mut found := found
    let mut done := done
    let mut j := j
    let i_7 := i
    let i_8 := i_7
    let n_9 := n_0
    let v10 := decide (i_8.toNat < n_9.toNat)
    let b3 : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun i found done j => do
      let mut i := i
      let mut found := found
      let mut done := done
      let mut j := j
      let found_49 := found
      return found_49
    if v10 then
      let v11 := (UInt256.ofNat 0)
      let j_12 := v11
      j := j_12
      let rec b4 (i : UInt256) (found : UInt256) (done : Bool) (j : UInt256) : FeM UInt256 := do
        let mut i := i
        let mut found := found
        let mut done := done
        let mut j := j
        let j_13 := j
        let j_14 := j_13
        let i_15 := i
        let v16 := decide (j_14.toNat < i_15.toNat)
        let b6 : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun i found done j => do
          let mut i := i
          let mut found := found
          let mut done := done
          let mut j := j
          let done_41 := done
          if done_41 then
            b3 i found done j
          else
            let v42 := ()
            let v43 := v42
            let v44 := (UInt256.ofNat 1)
            let i_45 := i
            let v46 ← checkedAdd i_45 v44
            let i_47 := v46
            i := i_47
            let v48 := ()
            b1 i found done j
        if v16 then
          let i_17 := i
          let i_18 := i_17
          let j_19 := j
          let v20 ← checkedMul i_18 j_19
          let v21 := (UInt256.ofNat 12)
          let v22 := decide (v20 = v21)
          if v22 then
            let i_23 := i
            let i_24 := i_23
            let v25 := (UInt256.ofNat 100)
            let v26 ← checkedMul i_24 v25
            let j_27 := j
            let v28 ← checkedAdd v26 j_27
            let found_29 := v28
            found := found_29
            let v30 := ()
            let v31 := true
            let done_32 := v31
            done := done_32
            let v33 := ()
            b6 i found done j
          else
            let v34 := ()
            let v35 := v34
            let v36 := (UInt256.ofNat 1)
            let j_37 := j
            let v38 ← checkedAdd j_37 v36
            let j_39 := v38
            j := j_39
            let v40 := ()
            b4 i found done j
        else
          b6 i found done j
      partial_fixpoint
      b4 i found done j
    else
      b3 i found done j
  partial_fixpoint
  b1 i found done j

def recv_BreakBoth (n : UInt256) : FeM UInt256 := do
  let v1 ← break_both n
  return v1

def continue_outer (n_0 : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut total : UInt256 := default
  let mut j : UInt256 := default
  let mut skip : Bool := default
  let v1 := (UInt256.ofNat 0)
  let i_2 := v1
  i := i_2
  let v3 := (UInt256.ofNat 0)
  let total_4 := v3
  total := total_4
  let rec b1 (i : UInt256) (total : UInt256) (j : UInt256) (skip : Bool) : FeM UInt256 := do
    let mut i := i
    let mut total := total
    let mut j := j
    let mut skip := skip
    let i_5 := i
    let i_6 := i_5
    let n_7 := n_0
    let v8 := decide (i_6.toNat < n_7.toNat)
    if v8 then
      let v9 := (UInt256.ofNat 1)
      let i_10 := i
      let v11 ← checkedAdd i_10 v9
      let i_12 := v11
      i := i_12
      let v13 := ()
      let v14 := (UInt256.ofNat 0)
      let j_15 := v14
      j := j_15
      let v16 := false
      let skip_17 := v16
      skip := skip_17
      let rec b4 (i : UInt256) (total : UInt256) (j : UInt256) (skip : Bool) : FeM UInt256 := do
        let mut i := i
        let mut total := total
        let mut j := j
        let mut skip := skip
        let j_18 := j
        let j_19 := j_18
        let i_20 := i
        let v21 := decide (j_19.toNat < i_20.toNat)
        let b6 : UInt256 → UInt256 → UInt256 → Bool → FeM UInt256 := fun i total j skip => do
          let mut i := i
          let mut total := total
          let mut j := j
          let mut skip := skip
          let skip_36 := skip
          if skip_36 then
            b1 i total j skip
          else
            let v37 := ()
            let v38 := v37
            let i_39 := i
            let i_40 := i_39
            let total_41 := total
            let v42 ← checkedAdd total_41 i_40
            let total_43 := v42
            total := total_43
            let v44 := ()
            b1 i total j skip
        if v21 then
          let j_22 := j
          let j_23 := j_22
          let v24 := (UInt256.ofNat 3)
          let v25 := decide (j_23 = v24)
          if v25 then
            let v26 := true
            let skip_27 := v26
            skip := skip_27
            let v28 := ()
            b6 i total j skip
          else
            let v29 := ()
            let v30 := v29
            let v31 := (UInt256.ofNat 1)
            let j_32 := j
            let v33 ← checkedAdd j_32 v31
            let j_34 := v33
            j := j_34
            let v35 := ()
            b4 i total j skip
        else
          b6 i total j skip
      partial_fixpoint
      b4 i total j skip
    else
      let total_45 := total
      return total_45
  partial_fixpoint
  b1 i total j skip

def recv_ContinueOuter (n : UInt256) : FeM UInt256 := do
  let v1 ← continue_outer n
  return v1

def return_inner (n_0 : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut j : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let i_2 := v1
  i := i_2
  let rec b1 (i : UInt256) (j : UInt256) : FeM UInt256 := do
    let mut i := i
    let mut j := j
    let i_3 := i
    let i_4 := i_3
    let n_5 := n_0
    let v6 := decide (i_4.toNat < n_5.toNat)
    if v6 then
      let v7 := (UInt256.ofNat 0)
      let j_8 := v7
      j := j_8
      let rec b4 (i : UInt256) (j : UInt256) : FeM UInt256 := do
        let mut i := i
        let mut j := j
        let j_9 := j
        let j_10 := j_9
        let n_11 := n_0
        let v12 := decide (j_10.toNat < n_11.toNat)
        if v12 then
          let i_13 := i
          let i_14 := i_13
          let j_15 := j
          let v16 ← checkedAdd i_14 j_15
          let v17 := (UInt256.ofNat 7)
          let v18 := decide (v16 = v17)
          if v18 then
            let i_19 := i
            let i_20 := i_19
            let v21 := (UInt256.ofNat 10)
            let v22 ← checkedMul i_20 v21
            let j_23 := j
            let v24 ← checkedAdd v22 j_23
            return v24
          else
            let v25 := ()
            let v26 := v25
            let v27 := (UInt256.ofNat 1)
            let j_28 := j
            let v29 ← checkedAdd j_28 v27
            let j_30 := v29
            j := j_30
            let v31 := ()
            b4 i j
        else
          let v32 := (UInt256.ofNat 1)
          let i_33 := i
          let v34 ← checkedAdd i_33 v32
          let i_35 := v34
          i := i_35
          let v36 := ()
          b1 i j
      partial_fixpoint
      b4 i j
    else
      let v37 := (UInt256.ofNat 0)
      return v37
  partial_fixpoint
  b1 i j

def recv_ReturnInner (n : UInt256) : FeM UInt256 := do
  let v1 ← return_inner n
  return v1

def three_levels (n_0 : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut k : UInt256 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  let v3 := (UInt256.ofNat 0)
  let total_4 := v3
  total := total_4
  let v5 := (UInt256.ofNat 0)
  let n_6 := n_0
  let v7 := ()
  let v8 := (Range_Known_0_Unknown.mk v7 n_6)
  r2 := v8
  let v10 ← Range_start_0 r2
  let v11 := v10
  let rec b1 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v1 : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut k := k
    let mut r2 := r2
    let mut r3 := r3
    let v13 ← Range_has_0 r2 v1
    if v13 then
      let v15 ← Range_at_0 r2 v1
      let i := v15
      let v17 := (UInt256.ofNat 0)
      let v18 := ()
      let v19 := (Range_Known_0_Unknown.mk v18 i)
      r3 := v19
      let v21 ← Range_start_0 r3
      let v22 := v21
      let rec b5 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v2 : UInt256) : FeM UInt256 := do
        let mut total := total
        let mut k := k
        let mut r2 := r2
        let mut r3 := r3
        let v24 ← Range_has_0 r3 v2
        if v24 then
          let v26 ← Range_at_0 r3 v2
          let j := v26
          let v28 := (UInt256.ofNat 0)
          let k_29 := v28
          k := k_29
          let rec b9 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) : FeM UInt256 := do
            let mut total := total
            let mut k := k
            let mut r2 := r2
            let mut r3 := r3
            let k_30 := k
            let k_31 := k_30
            let v32 := decide (k_31.toNat < j.toNat)
            if v32 then
              let v33 ← checkedAdd i j
              let k_34 := k
              let v35 ← checkedAdd v33 k_34
              let v36 := (UInt256.ofNat 3)
              let v37 ← checkedRem v35 v36
              let v38 := (UInt256.ofNat 0)
              let v39 := decide (v37 = v38)
              let b14 : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun total k r2 r3 => do
                let mut total := total
                let mut k := k
                let mut r2 := r2
                let mut r3 := r3
                let v52 := (UInt256.ofNat 1)
                let k_53 := k
                let v54 ← checkedAdd k_53 v52
                let k_55 := v54
                k := k_55
                let v56 := ()
                b9 total k r2 r3
              if v39 then
                let v40 := (UInt256.ofNat 1)
                let total_41 := total
                let v42 ← checkedAdd total_41 v40
                let total_43 := v42
                total := total_43
                let v44 := ()
                let v45 := v44
                b14 total k r2 r3
              else
                let v46 := (UInt256.ofNat 2)
                let total_47 := total
                let v48 ← checkedAdd total_47 v46
                let total_49 := v48
                total := total_49
                let v50 := ()
                let v51 := v50
                b14 total k r2 r3
            else
              let v57 := (UInt256.ofNat 2)
              let v58 ← checkedRem j v57
              let v59 := (UInt256.ofNat 0)
              let v60 := decide (v58 = v59)
              let b17 : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun total k r2 r3 => do
                let mut total := total
                let mut k := k
                let mut r2 := r2
                let mut r3 := r3
                let v70 ← Range_next_0 r3 v2
                let v71 := v70
                b5 total k r2 r3 v71
              if v60 then
                let v61 := (UInt256.ofNat 10)
                let total_62 := total
                let v63 ← checkedAdd total_62 v61
                let total_64 := v63
                total := total_64
                let v65 := ()
                let v66 := v65
                b17 total k r2 r3
              else
                let v67 := ()
                let v68 := v67
                b17 total k r2 r3
          partial_fixpoint
          b9 total k r2 r3
        else
          let v72 := ()
          let v74 ← Range_next_0 r2 v1
          let v75 := v74
          b1 total k r2 r3 v75
      partial_fixpoint
      b5 total k r2 r3 v22
    else
      let total_76 := total
      return total_76
  partial_fixpoint
  b1 total k r2 r3 v11

def recv_ThreeLevels (n : UInt256) : FeM UInt256 := do
  let v1 ← downcast_truncate_u256_usize n
  let v2 ← three_levels v1
  return v2


export Lib (Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start Range_at_0 Range_has_0 Range_next_0 Range_start_0 downcast_truncate_u256_usize)

/-- The `init` of contract `NestedLoops`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `nested_loops::windows` -/
def windows (n : UInt256) (width : UInt256) : FeM UInt256 := do
  let mut acc : UInt256 := default
  let mut index : UInt256 := default
  let mut started : Bool := default
  let mut r3 : Range_Unknown_Unknown := default
  acc := (UInt256.ofNat 1)
  index := (UInt256.ofNat 0)
  started := false
  let rec loop (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
    let mut acc := acc
    let mut index := index
    let mut started := started
    let mut r3 := r3
    let index_10 := index
    let v13 := decide (index_10.toNat < n.toNat)
    if v13 then
      let index_14 := index
      let v17 ← checkedAdd index_14 width
      let v19 ← checkedAdd v17 (UInt256.ofNat 1)
      let v21 := decide (v19.toNat < n.toNat)
      let join : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun acc index started r3 «end» => do
        let mut acc := acc
        let mut index := index
        let mut started := started
        let mut r3 := r3
        let started_32 := started
        let rec loop_1 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
          let mut acc := acc
          let mut index := index
          let mut started := started
          let mut r3 := r3
          let v60 ← checkedRem «end» (UInt256.ofNat 2)
          let v62 := decide (v60 = (UInt256.ofNat 1))
          let join_1 : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → FeM UInt256 := fun acc index started r3 => do
            let mut acc := acc
            let mut index := index
            let mut started := started
            let mut r3 := r3
            index := «end»
            loop acc index started r3
          if v62 then
            let acc_63 := acc
            let acc_66 ← checkedAdd acc_63 «end»
            acc := acc_66
            started := true
            join_1 acc index started r3
          else
            join_1 acc index started r3
        partial_fixpoint
        if started_32 then
          let index_33 := index
          let v36 := (Range_Unknown_Unknown.mk index_33 «end»)
          r3 := v36
          let v38 ← Range_Unknown_Unknown_start r3
          let rec loop_2 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) (v3 : UInt256) : FeM UInt256 := do
            let mut acc := acc
            let mut index := index
            let mut started := started
            let mut r3 := r3
            let v41 ← Range_Unknown_Unknown_has r3 v3
            if v41 then
              let _ ← Range_Unknown_Unknown_at r3 v3
              let acc_44 := acc
              let v47 ← checkedMul acc_44 (UInt256.ofNat 2)
              let acc_49 ← checkedRem v47 (UInt256.ofNat 1000003)
              acc := acc_49
              let v53 ← Range_Unknown_Unknown_next r3 v3
              loop_2 acc index started r3 v53
            else
              loop_1 acc index started r3
          partial_fixpoint
          loop_2 acc index started r3 v38
        else
          loop_1 acc index started r3
      if v21 then
        let index_22 := index
        let v25 ← checkedAdd index_22 width
        let v27 ← checkedAdd v25 (UInt256.ofNat 1)
        join acc index started r3 v27
      else
        join acc index started r3 n
    else
      let acc_77 := acc
      return acc_77
  partial_fixpoint
  loop acc index started r3

/-- `Windows { n, width } -> u256`, selector `0x6a39ed49` (`windows(uint256,uint256)`) -/
def recv_Windows (n : UInt256) (width : UInt256) : FeM UInt256 := do
  let v2 ← downcast_truncate_u256_usize n
  let v3 ← downcast_truncate_u256_usize width
  let v4 ← windows v2 v3
  return v4

/-- `nested_loops::break_both` -/
def break_both (n : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut found : UInt256 := default
  let mut done : Bool := default
  let mut j : UInt256 := default
  i := (UInt256.ofNat 0)
  found := (UInt256.ofNat 0)
  done := false
  let rec loop (i : UInt256) (found : UInt256) (done : Bool) (j : UInt256) : FeM UInt256 := do
    let mut i := i
    let mut found := found
    let mut done := done
    let mut j := j
    let i_7 := i
    let v10 := decide (i_7.toNat < n.toNat)
    let join : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun i found done j => do
      let mut i := i
      let mut found := found
      let mut done := done
      let mut j := j
      let found_49 := found
      return found_49
    if v10 then
      j := (UInt256.ofNat 0)
      let rec loop_1 (i : UInt256) (found : UInt256) (done : Bool) (j : UInt256) : FeM UInt256 := do
        let mut i := i
        let mut found := found
        let mut done := done
        let mut j := j
        let j_13 := j
        let i_15 := i
        let v16 := decide (j_13.toNat < i_15.toNat)
        let join_1 : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun i found done j => do
          let mut i := i
          let mut found := found
          let mut done := done
          let mut j := j
          let done_41 := done
          if done_41 then
            join i found done j
          else
            let i_45 := i
            let i_46 ← checkedAdd i_45 (UInt256.ofNat 1)
            i := i_46
            loop i found done j
        if v16 then
          let i_17 := i
          let j_19 := j
          let v20 ← checkedMul i_17 j_19
          let v22 := decide (v20 = (UInt256.ofNat 12))
          if v22 then
            let i_23 := i
            let v26 ← checkedMul i_23 (UInt256.ofNat 100)
            let j_27 := j
            let found_28 ← checkedAdd v26 j_27
            found := found_28
            done := true
            join_1 i found done j
          else
            let j_37 := j
            let j_38 ← checkedAdd j_37 (UInt256.ofNat 1)
            j := j_38
            loop_1 i found done j
        else
          join_1 i found done j
      partial_fixpoint
      loop_1 i found done j
    else
      join i found done j
  partial_fixpoint
  loop i found done j

/-- `BreakBoth { n } -> u256`, selector `0x17ea0cd5` (`breakBoth(uint256)`) -/
def recv_BreakBoth (n : UInt256) : FeM UInt256 := do
  let v1 ← break_both n
  return v1

/-- `nested_loops::continue_outer` -/
def continue_outer (n : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut total : UInt256 := default
  let mut j : UInt256 := default
  let mut skip : Bool := default
  i := (UInt256.ofNat 0)
  total := (UInt256.ofNat 0)
  let rec loop (i : UInt256) (total : UInt256) (j : UInt256) (skip : Bool) : FeM UInt256 := do
    let mut i := i
    let mut total := total
    let mut j := j
    let mut skip := skip
    let i_5 := i
    let v8 := decide (i_5.toNat < n.toNat)
    if v8 then
      let i_10 := i
      let i_11 ← checkedAdd i_10 (UInt256.ofNat 1)
      i := i_11
      j := (UInt256.ofNat 0)
      skip := false
      let rec loop_1 (i : UInt256) (total : UInt256) (j : UInt256) (skip : Bool) : FeM UInt256 := do
        let mut i := i
        let mut total := total
        let mut j := j
        let mut skip := skip
        let j_18 := j
        let i_20 := i
        let v21 := decide (j_18.toNat < i_20.toNat)
        let join : UInt256 → UInt256 → UInt256 → Bool → FeM UInt256 := fun i total j skip => do
          let mut i := i
          let mut total := total
          let mut j := j
          let mut skip := skip
          let skip_36 := skip
          if skip_36 then
            loop i total j skip
          else
            let i_39 := i
            let total_41 := total
            let total_42 ← checkedAdd total_41 i_39
            total := total_42
            loop i total j skip
        if v21 then
          let j_22 := j
          let v25 := decide (j_22 = (UInt256.ofNat 3))
          if v25 then
            skip := true
            join i total j skip
          else
            let j_32 := j
            let j_33 ← checkedAdd j_32 (UInt256.ofNat 1)
            j := j_33
            loop_1 i total j skip
        else
          join i total j skip
      partial_fixpoint
      loop_1 i total j skip
    else
      let total_45 := total
      return total_45
  partial_fixpoint
  loop i total j skip

/-- `ContinueOuter { n } -> u256`, selector `0x0dec9eef` (`continueOuter(uint256)`) -/
def recv_ContinueOuter (n : UInt256) : FeM UInt256 := do
  let v1 ← continue_outer n
  return v1

/-- `nested_loops::return_inner` -/
def return_inner (n : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut j : UInt256 := default
  i := (UInt256.ofNat 0)
  let rec loop (i : UInt256) (j : UInt256) : FeM UInt256 := do
    let mut i := i
    let mut j := j
    let i_3 := i
    let v6 := decide (i_3.toNat < n.toNat)
    if v6 then
      j := (UInt256.ofNat 0)
      let rec loop_1 (i : UInt256) (j : UInt256) : FeM UInt256 := do
        let mut i := i
        let mut j := j
        let j_9 := j
        let v12 := decide (j_9.toNat < n.toNat)
        if v12 then
          let i_13 := i
          let j_15 := j
          let v16 ← checkedAdd i_13 j_15
          let v18 := decide (v16 = (UInt256.ofNat 7))
          if v18 then
            let i_19 := i
            let v22 ← checkedMul i_19 (UInt256.ofNat 10)
            let j_23 := j
            let v24 ← checkedAdd v22 j_23
            return v24
          else
            let j_28 := j
            let j_29 ← checkedAdd j_28 (UInt256.ofNat 1)
            j := j_29
            loop_1 i j
        else
          let i_33 := i
          let i_34 ← checkedAdd i_33 (UInt256.ofNat 1)
          i := i_34
          loop i j
      partial_fixpoint
      loop_1 i j
    else
      return (UInt256.ofNat 0)
  partial_fixpoint
  loop i j

/-- `ReturnInner { n } -> u256`, selector `0x92ba46ad` (`returnInner(uint256)`) -/
def recv_ReturnInner (n : UInt256) : FeM UInt256 := do
  let v1 ← return_inner n
  return v1

/-- `nested_loops::three_levels` -/
def three_levels (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut k : UInt256 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  total := (UInt256.ofNat 0)
  let v8 := (Range_Known_0_Unknown.mk () n)
  r2 := v8
  let v10 ← Range_start_0 r2
  let rec loop (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v1 : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut k := k
    let mut r2 := r2
    let mut r3 := r3
    let v13 ← Range_has_0 r2 v1
    if v13 then
      let v15 ← Range_at_0 r2 v1
      let i := v15
      let v19 := (Range_Known_0_Unknown.mk () i)
      r3 := v19
      let v21 ← Range_start_0 r3
      let rec loop_1 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v2 : UInt256) : FeM UInt256 := do
        let mut total := total
        let mut k := k
        let mut r2 := r2
        let mut r3 := r3
        let v24 ← Range_has_0 r3 v2
        if v24 then
          let v26 ← Range_at_0 r3 v2
          let j := v26
          k := (UInt256.ofNat 0)
          let rec loop_2 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) : FeM UInt256 := do
            let mut total := total
            let mut k := k
            let mut r2 := r2
            let mut r3 := r3
            let k_30 := k
            let v32 := decide (k_30.toNat < j.toNat)
            if v32 then
              let v33 ← checkedAdd i j
              let k_34 := k
              let v35 ← checkedAdd v33 k_34
              let v37 ← checkedRem v35 (UInt256.ofNat 3)
              let v39 := decide (v37 = (UInt256.ofNat 0))
              let join : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun total k r2 r3 => do
                let mut total := total
                let mut k := k
                let mut r2 := r2
                let mut r3 := r3
                let k_53 := k
                let k_54 ← checkedAdd k_53 (UInt256.ofNat 1)
                k := k_54
                loop_2 total k r2 r3
              if v39 then
                let total_41 := total
                let total_42 ← checkedAdd total_41 (UInt256.ofNat 1)
                total := total_42
                join total k r2 r3
              else
                let total_47 := total
                let total_48 ← checkedAdd total_47 (UInt256.ofNat 2)
                total := total_48
                join total k r2 r3
            else
              let v58 ← checkedRem j (UInt256.ofNat 2)
              let v60 := decide (v58 = (UInt256.ofNat 0))
              let join_1 : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun total k r2 r3 => do
                let mut total := total
                let mut k := k
                let mut r2 := r2
                let mut r3 := r3
                let v70 ← Range_next_0 r3 v2
                loop_1 total k r2 r3 v70
              if v60 then
                let total_62 := total
                let total_63 ← checkedAdd total_62 (UInt256.ofNat 10)
                total := total_63
                join_1 total k r2 r3
              else
                join_1 total k r2 r3
          partial_fixpoint
          loop_2 total k r2 r3
        else
          let v74 ← Range_next_0 r2 v1
          loop total k r2 r3 v74
      partial_fixpoint
      loop_1 total k r2 r3 v21
    else
      let total_76 := total
      return total_76
  partial_fixpoint
  loop total k r2 r3 v10

/-- `ThreeLevels { n } -> u256`, selector `0x81037424` (`threeLevels(uint256)`) -/
def recv_ThreeLevels (n : UInt256) : FeM UInt256 := do
  let v1 ← downcast_truncate_u256_usize n
  let v2 ← three_levels v1
  return v2


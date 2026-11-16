export Lib (Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_Unknown_Unknown Range_Unknown_Unknown.mk Range_Unknown_Unknown_at Range_Unknown_Unknown_has Range_Unknown_Unknown_next Range_Unknown_Unknown_start Range_at_0 Range_has_0 Range_next_0 Range_start_0 downcast_truncate_u256_usize)

def init : FeM Unit := do
  return ()

def windows (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : Bool := default
  let mut r3 : Range_Unknown_Unknown := default
  let v4 := (UInt256.ofNat 1)
  let v5 := v4
  r0 := v5
  let v6 := (UInt256.ofNat 0)
  let v7 := v6
  r1 := v7
  let v8 := false
  let v9 := v8
  r2 := v9
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let v10 := r1
    let v11 := v10
    let v12 := v0
    let v13 := decide (v11.toNat < v12.toNat)
    if v13 then
      let v14 := r1
      let v15 := v14
      let v16 := v1
      let v17 ← checkedAdd v15 v16
      let v18 := (UInt256.ofNat 1)
      let v19 ← checkedAdd v17 v18
      let v20 := v0
      let v21 := decide (v19.toNat < v20.toNat)
      let b6 : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun r0 r1 r2 r3 v2 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let v31 := v2
        let v32 := r2
        let rec b9 (r0 : UInt256) (r1 : UInt256) (r2 : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let v59 := (UInt256.ofNat 2)
          let v60 ← checkedRem v31 v59
          let v61 := (UInt256.ofNat 1)
          let v62 := decide (v60 = v61)
          let b16 : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → FeM UInt256 := fun r0 r1 r2 r3 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let v75 := v31
            r1 := v75
            let v76 := ()
            b1 r0 r1 r2 r3
          if v62 then
            let v63 := r0
            let v64 := v63
            let v65 := v31
            let v66 ← checkedAdd v64 v65
            let v67 := v66
            r0 := v67
            let v68 := ()
            let v69 := true
            let v70 := v69
            r2 := v70
            let v71 := ()
            let v72 := v71
            b16 r0 r1 r2 r3
          else
            let v73 := ()
            let v74 := v73
            b16 r0 r1 r2 r3
        partial_fixpoint
        if v32 then
          let v33 := r1
          let v34 := v33
          let v35 := ()
          let v36 := (Range_Unknown_Unknown.mk v34 v31)
          r3 := v36
          let v38 ← Range_Unknown_Unknown_start r3
          let v39 := v38
          let rec b10 (r0 : UInt256) (r1 : UInt256) (r2 : Bool) (r3 : Range_Unknown_Unknown) (v3 : UInt256) : FeM UInt256 := do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let v41 ← Range_Unknown_Unknown_has r3 v3
            if v41 then
              let v43 ← Range_Unknown_Unknown_at r3 v3
              let v44 := r0
              let v45 := v44
              let v46 := (UInt256.ofNat 2)
              let v47 ← checkedMul v45 v46
              let v48 := (UInt256.ofNat 1000003)
              let v49 ← checkedRem v47 v48
              let v50 := v49
              r0 := v50
              let v51 := ()
              let v53 ← Range_Unknown_Unknown_next r3 v3
              let v54 := v53
              b10 r0 r1 r2 r3 v54
            else
              let v55 := ()
              let v56 := v55
              b9 r0 r1 r2 r3
          partial_fixpoint
          b10 r0 r1 r2 r3 v39
        else
          let v57 := ()
          let v58 := v57
          b9 r0 r1 r2 r3
      if v21 then
        let v22 := r1
        let v23 := v22
        let v24 := v1
        let v25 ← checkedAdd v23 v24
        let v26 := (UInt256.ofNat 1)
        let v27 ← checkedAdd v25 v26
        let v28 := v27
        b6 r0 r1 r2 r3 v28
      else
        let v29 := v0
        let v30 := v29
        b6 r0 r1 r2 r3 v30
    else
      let v77 := r0
      return v77
  partial_fixpoint
  b1 r0 r1 r2 r3

def recv_Windows (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← downcast_truncate_u256_usize v0
  let v3 ← downcast_truncate_u256_usize v1
  let v4 ← windows v2 v3
  return v4

def break_both (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : Bool := default
  let mut r3 : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let v2 := v1
  r0 := v2
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  r1 := v4
  let v5 := false
  let v6 := v5
  r2 := v6
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : Bool) (r3 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let v7 := r0
    let v8 := v7
    let v9 := v0
    let v10 := decide (v8.toNat < v9.toNat)
    let b3 : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun r0 r1 r2 r3 => do
      let mut r0 := r0
      let mut r1 := r1
      let mut r2 := r2
      let mut r3 := r3
      let v49 := r1
      return v49
    if v10 then
      let v11 := (UInt256.ofNat 0)
      let v12 := v11
      r3 := v12
      let rec b4 (r0 : UInt256) (r1 : UInt256) (r2 : Bool) (r3 : UInt256) : FeM UInt256 := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let v13 := r3
        let v14 := v13
        let v15 := r0
        let v16 := decide (v14.toNat < v15.toNat)
        let b6 : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun r0 r1 r2 r3 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let v41 := r2
          if v41 then
            b3 r0 r1 r2 r3
          else
            let v42 := ()
            let v43 := v42
            let v44 := (UInt256.ofNat 1)
            let v45 := r0
            let v46 ← checkedAdd v45 v44
            let v47 := v46
            r0 := v47
            let v48 := ()
            b1 r0 r1 r2 r3
        if v16 then
          let v17 := r0
          let v18 := v17
          let v19 := r3
          let v20 ← checkedMul v18 v19
          let v21 := (UInt256.ofNat 12)
          let v22 := decide (v20 = v21)
          if v22 then
            let v23 := r0
            let v24 := v23
            let v25 := (UInt256.ofNat 100)
            let v26 ← checkedMul v24 v25
            let v27 := r3
            let v28 ← checkedAdd v26 v27
            let v29 := v28
            r1 := v29
            let v30 := ()
            let v31 := true
            let v32 := v31
            r2 := v32
            let v33 := ()
            b6 r0 r1 r2 r3
          else
            let v34 := ()
            let v35 := v34
            let v36 := (UInt256.ofNat 1)
            let v37 := r3
            let v38 ← checkedAdd v37 v36
            let v39 := v38
            r3 := v39
            let v40 := ()
            b4 r0 r1 r2 r3
        else
          b6 r0 r1 r2 r3
      partial_fixpoint
      b4 r0 r1 r2 r3
    else
      b3 r0 r1 r2 r3
  partial_fixpoint
  b1 r0 r1 r2 r3

def recv_BreakBoth (v0 : UInt256) : FeM UInt256 := do
  let v1 ← break_both v0
  return v1

def continue_outer (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : Bool := default
  let v1 := (UInt256.ofNat 0)
  let v2 := v1
  r0 := v2
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  r1 := v4
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : Bool) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let v5 := r0
    let v6 := v5
    let v7 := v0
    let v8 := decide (v6.toNat < v7.toNat)
    if v8 then
      let v9 := (UInt256.ofNat 1)
      let v10 := r0
      let v11 ← checkedAdd v10 v9
      let v12 := v11
      r0 := v12
      let v13 := ()
      let v14 := (UInt256.ofNat 0)
      let v15 := v14
      r2 := v15
      let v16 := false
      let v17 := v16
      r3 := v17
      let rec b4 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : Bool) : FeM UInt256 := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let v18 := r2
        let v19 := v18
        let v20 := r0
        let v21 := decide (v19.toNat < v20.toNat)
        let b6 : UInt256 → UInt256 → UInt256 → Bool → FeM UInt256 := fun r0 r1 r2 r3 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let v36 := r3
          if v36 then
            b1 r0 r1 r2 r3
          else
            let v37 := ()
            let v38 := v37
            let v39 := r0
            let v40 := v39
            let v41 := r1
            let v42 ← checkedAdd v41 v40
            let v43 := v42
            r1 := v43
            let v44 := ()
            b1 r0 r1 r2 r3
        if v21 then
          let v22 := r2
          let v23 := v22
          let v24 := (UInt256.ofNat 3)
          let v25 := decide (v23 = v24)
          if v25 then
            let v26 := true
            let v27 := v26
            r3 := v27
            let v28 := ()
            b6 r0 r1 r2 r3
          else
            let v29 := ()
            let v30 := v29
            let v31 := (UInt256.ofNat 1)
            let v32 := r2
            let v33 ← checkedAdd v32 v31
            let v34 := v33
            r2 := v34
            let v35 := ()
            b4 r0 r1 r2 r3
        else
          b6 r0 r1 r2 r3
      partial_fixpoint
      b4 r0 r1 r2 r3
    else
      let v45 := r1
      return v45
  partial_fixpoint
  b1 r0 r1 r2 r3

def recv_ContinueOuter (v0 : UInt256) : FeM UInt256 := do
  let v1 ← continue_outer v0
  return v1

def return_inner (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let v2 := v1
  r0 := v2
  let rec b1 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let v3 := r0
    let v4 := v3
    let v5 := v0
    let v6 := decide (v4.toNat < v5.toNat)
    if v6 then
      let v7 := (UInt256.ofNat 0)
      let v8 := v7
      r1 := v8
      let rec b4 (r0 : UInt256) (r1 : UInt256) : FeM UInt256 := do
        let mut r0 := r0
        let mut r1 := r1
        let v9 := r1
        let v10 := v9
        let v11 := v0
        let v12 := decide (v10.toNat < v11.toNat)
        if v12 then
          let v13 := r0
          let v14 := v13
          let v15 := r1
          let v16 ← checkedAdd v14 v15
          let v17 := (UInt256.ofNat 7)
          let v18 := decide (v16 = v17)
          if v18 then
            let v19 := r0
            let v20 := v19
            let v21 := (UInt256.ofNat 10)
            let v22 ← checkedMul v20 v21
            let v23 := r1
            let v24 ← checkedAdd v22 v23
            return v24
          else
            let v25 := ()
            let v26 := v25
            let v27 := (UInt256.ofNat 1)
            let v28 := r1
            let v29 ← checkedAdd v28 v27
            let v30 := v29
            r1 := v30
            let v31 := ()
            b4 r0 r1
        else
          let v32 := (UInt256.ofNat 1)
          let v33 := r0
          let v34 ← checkedAdd v33 v32
          let v35 := v34
          r0 := v35
          let v36 := ()
          b1 r0 r1
      partial_fixpoint
      b4 r0 r1
    else
      let v37 := (UInt256.ofNat 0)
      return v37
  partial_fixpoint
  b1 r0 r1

def recv_ReturnInner (v0 : UInt256) : FeM UInt256 := do
  let v1 ← return_inner v0
  return v1

def three_levels (v0 : UInt256) : FeM UInt256 := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  r0 := v4
  let v5 := (UInt256.ofNat 0)
  let v6 := v0
  let v7 := ()
  let v8 := (Range_Known_0_Unknown.mk v7 v6)
  r2 := v8
  let v10 ← Range_start_0 r2
  let v11 := v10
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v1 : UInt256) : FeM UInt256 := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let v13 ← Range_has_0 r2 v1
    if v13 then
      let v15 ← Range_at_0 r2 v1
      let v16 := v15
      let v17 := (UInt256.ofNat 0)
      let v18 := ()
      let v19 := (Range_Known_0_Unknown.mk v18 v16)
      r3 := v19
      let v21 ← Range_start_0 r3
      let v22 := v21
      let rec b5 (r0 : UInt256) (r1 : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v2 : UInt256) : FeM UInt256 := do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let v24 ← Range_has_0 r3 v2
        if v24 then
          let v26 ← Range_at_0 r3 v2
          let v27 := v26
          let v28 := (UInt256.ofNat 0)
          let v29 := v28
          r1 := v29
          let rec b9 (r0 : UInt256) (r1 : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) : FeM UInt256 := do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let v30 := r1
            let v31 := v30
            let v32 := decide (v31.toNat < v27.toNat)
            if v32 then
              let v33 ← checkedAdd v16 v27
              let v34 := r1
              let v35 ← checkedAdd v33 v34
              let v36 := (UInt256.ofNat 3)
              let v37 ← checkedRem v35 v36
              let v38 := (UInt256.ofNat 0)
              let v39 := decide (v37 = v38)
              let b14 : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun r0 r1 r2 r3 => do
                let mut r0 := r0
                let mut r1 := r1
                let mut r2 := r2
                let mut r3 := r3
                let v52 := (UInt256.ofNat 1)
                let v53 := r1
                let v54 ← checkedAdd v53 v52
                let v55 := v54
                r1 := v55
                let v56 := ()
                b9 r0 r1 r2 r3
              if v39 then
                let v40 := (UInt256.ofNat 1)
                let v41 := r0
                let v42 ← checkedAdd v41 v40
                let v43 := v42
                r0 := v43
                let v44 := ()
                let v45 := v44
                b14 r0 r1 r2 r3
              else
                let v46 := (UInt256.ofNat 2)
                let v47 := r0
                let v48 ← checkedAdd v47 v46
                let v49 := v48
                r0 := v49
                let v50 := ()
                let v51 := v50
                b14 r0 r1 r2 r3
            else
              let v57 := (UInt256.ofNat 2)
              let v58 ← checkedRem v27 v57
              let v59 := (UInt256.ofNat 0)
              let v60 := decide (v58 = v59)
              let b17 : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun r0 r1 r2 r3 => do
                let mut r0 := r0
                let mut r1 := r1
                let mut r2 := r2
                let mut r3 := r3
                let v70 ← Range_next_0 r3 v2
                let v71 := v70
                b5 r0 r1 r2 r3 v71
              if v60 then
                let v61 := (UInt256.ofNat 10)
                let v62 := r0
                let v63 ← checkedAdd v62 v61
                let v64 := v63
                r0 := v64
                let v65 := ()
                let v66 := v65
                b17 r0 r1 r2 r3
              else
                let v67 := ()
                let v68 := v67
                b17 r0 r1 r2 r3
          partial_fixpoint
          b9 r0 r1 r2 r3
        else
          let v72 := ()
          let v74 ← Range_next_0 r2 v1
          let v75 := v74
          b1 r0 r1 r2 r3 v75
      partial_fixpoint
      b5 r0 r1 r2 r3 v22
    else
      let v76 := r0
      return v76
  partial_fixpoint
  b1 r0 r1 r2 r3 v11

def recv_ThreeLevels (v0 : UInt256) : FeM UInt256 := do
  let v1 ← downcast_truncate_u256_usize v0
  let v2 ← three_levels v1
  return v2


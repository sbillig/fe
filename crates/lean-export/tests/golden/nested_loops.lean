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
  -- var acc: u256 = 1
  acc := (UInt256.ofNat 1)
  -- var index: usize = 0
  index := (UInt256.ofNat 0)
  -- var started = false
  started := false
  let rec loop (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
    let mut acc := acc
    let mut index := index
    let mut started := started
    let mut r3 := r3
    -- while index < n {
    if decide (index.toNat < n.toNat) then
      -- let end = if index + width + 1 < n { index + width + 1 } else { n }
      let v17 ← checkedAdd index width
      let v19 ← checkedAdd v17 (UInt256.ofNat 1)
      let join : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → UInt256 → FeM UInt256 := fun acc index started r3 «end» => do
        let mut acc := acc
        let mut index := index
        let mut started := started
        let mut r3 := r3
        let rec loop_1 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) : FeM UInt256 := do
          let mut acc := acc
          let mut index := index
          let mut started := started
          let mut r3 := r3
          -- if end % 2 == 1 {
          let v60 ← checkedRem «end» (UInt256.ofNat 2)
          let join_1 : UInt256 → UInt256 → Bool → Range_Unknown_Unknown → FeM UInt256 := fun acc index started r3 => do
            let mut acc := acc
            let mut index := index
            let mut started := started
            let mut r3 := r3
            -- index = end
            index := «end»
            loop acc index started r3
          if decide (v60 = (UInt256.ofNat 1)) then
            -- acc = acc + end as u256
            let acc_66 ← checkedAdd acc «end»
            acc := acc_66
            -- started = true
            started := true
            join_1 acc index started r3
          else
            join_1 acc index started r3
        partial_fixpoint
        -- if started {
        if started then
          -- for _ in index..end { acc = acc * 2 % 1000003 }
          r3 := (Range_Unknown_Unknown.mk index «end»)
          let v38 ← Range_Unknown_Unknown_start r3
          let rec loop_2 (acc : UInt256) (index : UInt256) (started : Bool) (r3 : Range_Unknown_Unknown) (v3 : UInt256) : FeM UInt256 := do
            let mut acc := acc
            let mut index := index
            let mut started := started
            let mut r3 := r3
            -- for _ in index..end { acc = acc * 2 % 1000003 }
            let v41 ← Range_Unknown_Unknown_has r3 v3
            if v41 then
              let _ ← Range_Unknown_Unknown_at r3 v3
              -- acc = acc * 2 % 1000003
              let v47 ← checkedMul acc (UInt256.ofNat 2)
              let acc_49 ← checkedRem v47 (UInt256.ofNat 1000003)
              acc := acc_49
              -- for _ in index..end { acc = acc * 2 % 1000003 }
              let v53 ← Range_Unknown_Unknown_next r3 v3
              loop_2 acc index started r3 v53
            else
              loop_1 acc index started r3
          partial_fixpoint
          loop_2 acc index started r3 v38
        else
          loop_1 acc index started r3
      if decide (v19.toNat < n.toNat) then
        -- index + width + 1
        let v25 ← checkedAdd index width
        let v27 ← checkedAdd v25 (UInt256.ofNat 1)
        join acc index started r3 v27
      else
        join acc index started r3 n
    else
      return acc
  partial_fixpoint
  loop acc index started r3

/-- `Windows { n, width } -> u256`, selector `0x6a39ed49` (`windows(uint256,uint256)`) -/
def recv_Windows (n : UInt256) (width : UInt256) : FeM UInt256 := do
  -- windows(n.downcast_truncate(), width.downcast_truncate())
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
  -- var i: u256 = 0
  i := (UInt256.ofNat 0)
  -- var found: u256 = 0
  found := (UInt256.ofNat 0)
  -- var done = false
  done := false
  let rec loop (i : UInt256) (found : UInt256) (done : Bool) (j : UInt256) : FeM UInt256 := do
    let mut i := i
    let mut found := found
    let mut done := done
    let mut j := j
    let join : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun i found done j => do
      let mut i := i
      let mut found := found
      let mut done := done
      let mut j := j
      return found
    -- while i < n {
    if decide (i.toNat < n.toNat) then
      -- var j: u256 = 0
      j := (UInt256.ofNat 0)
      let rec loop_1 (i : UInt256) (found : UInt256) (done : Bool) (j : UInt256) : FeM UInt256 := do
        let mut i := i
        let mut found := found
        let mut done := done
        let mut j := j
        let join_1 : UInt256 → UInt256 → Bool → UInt256 → FeM UInt256 := fun i found done j => do
          let mut i := i
          let mut found := found
          let mut done := done
          let mut j := j
          -- if done { break }
          if done then
            join i found done j
          else
            -- i += 1
            let i_46 ← checkedAdd i (UInt256.ofNat 1)
            i := i_46
            loop i found done j
        -- while j < i {
        if decide (j.toNat < i.toNat) then
          -- if i * j == 12 {
          let v20 ← checkedMul i j
          if decide (v20 = (UInt256.ofNat 12)) then
            -- found = i * 100 + j
            let v26 ← checkedMul i (UInt256.ofNat 100)
            let found_28 ← checkedAdd v26 j
            found := found_28
            -- done = true
            done := true
            join_1 i found done j
          else
            -- j += 1
            let j_38 ← checkedAdd j (UInt256.ofNat 1)
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
  -- break_both(n)
  let v1 ← break_both n
  return v1

/-- `nested_loops::continue_outer` -/
def continue_outer (n : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut total : UInt256 := default
  let mut j : UInt256 := default
  let mut skip : Bool := default
  -- var i: u256 = 0
  i := (UInt256.ofNat 0)
  -- var total: u256 = 0
  total := (UInt256.ofNat 0)
  let rec loop (i : UInt256) (total : UInt256) (j : UInt256) (skip : Bool) : FeM UInt256 := do
    let mut i := i
    let mut total := total
    let mut j := j
    let mut skip := skip
    -- while i < n {
    if decide (i.toNat < n.toNat) then
      -- i += 1
      let i_11 ← checkedAdd i (UInt256.ofNat 1)
      i := i_11
      -- var j: u256 = 0
      j := (UInt256.ofNat 0)
      -- var skip = false
      skip := false
      let rec loop_1 (i : UInt256) (total : UInt256) (j : UInt256) (skip : Bool) : FeM UInt256 := do
        let mut i := i
        let mut total := total
        let mut j := j
        let mut skip := skip
        let join : UInt256 → UInt256 → UInt256 → Bool → FeM UInt256 := fun i total j skip => do
          let mut i := i
          let mut total := total
          let mut j := j
          let mut skip := skip
          -- if skip { continue }
          if skip then
            loop i total j skip
          else
            -- total += i
            let total_42 ← checkedAdd total i
            total := total_42
            loop i total j skip
        -- while j < i {
        if decide (j.toNat < i.toNat) then
          -- if j == 3 {
          if decide (j = (UInt256.ofNat 3)) then
            -- skip = true
            skip := true
            join i total j skip
          else
            -- j += 1
            let j_33 ← checkedAdd j (UInt256.ofNat 1)
            j := j_33
            loop_1 i total j skip
        else
          join i total j skip
      partial_fixpoint
      loop_1 i total j skip
    else
      return total
  partial_fixpoint
  loop i total j skip

/-- `ContinueOuter { n } -> u256`, selector `0x0dec9eef` (`continueOuter(uint256)`) -/
def recv_ContinueOuter (n : UInt256) : FeM UInt256 := do
  -- continue_outer(n)
  let v1 ← continue_outer n
  return v1

/-- `nested_loops::return_inner` -/
def return_inner (n : UInt256) : FeM UInt256 := do
  let mut i : UInt256 := default
  let mut j : UInt256 := default
  -- var i: u256 = 0
  i := (UInt256.ofNat 0)
  let rec loop (i : UInt256) (j : UInt256) : FeM UInt256 := do
    let mut i := i
    let mut j := j
    -- while i < n {
    if decide (i.toNat < n.toNat) then
      -- var j: u256 = 0
      j := (UInt256.ofNat 0)
      let rec loop_1 (i : UInt256) (j : UInt256) : FeM UInt256 := do
        let mut i := i
        let mut j := j
        -- while j < n {
        if decide (j.toNat < n.toNat) then
          -- if i + j == 7 { return i * 10 + j }
          let v16 ← checkedAdd i j
          if decide (v16 = (UInt256.ofNat 7)) then
            -- return i * 10 + j
            let v22 ← checkedMul i (UInt256.ofNat 10)
            let v24 ← checkedAdd v22 j
            return v24
          else
            -- j += 1
            let j_29 ← checkedAdd j (UInt256.ofNat 1)
            j := j_29
            loop_1 i j
        else
          -- i += 1
          let i_34 ← checkedAdd i (UInt256.ofNat 1)
          i := i_34
          loop i j
      partial_fixpoint
      loop_1 i j
    else
      return UInt256.ofNat 0
  partial_fixpoint
  loop i j

/-- `ReturnInner { n } -> u256`, selector `0x92ba46ad` (`returnInner(uint256)`) -/
def recv_ReturnInner (n : UInt256) : FeM UInt256 := do
  -- return_inner(n)
  let v1 ← return_inner n
  return v1

/-- `nested_loops::three_levels` -/
def three_levels (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut k : UInt256 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : Range_Known_0_Unknown := default
  -- var total: u256 = 0
  total := (UInt256.ofNat 0)
  -- for i in 0..n {
  r2 := (Range_Known_0_Unknown.mk () n)
  let v10 ← Range_start_0 r2
  let rec loop (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v1 : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut k := k
    let mut r2 := r2
    let mut r3 := r3
    -- for i in 0..n {
    let v13 ← Range_has_0 r2 v1
    if v13 then
      let v15 ← Range_at_0 r2 v1
      let i := v15
      -- for j in 0..i {
      r3 := (Range_Known_0_Unknown.mk () i)
      let v21 ← Range_start_0 r3
      let rec loop_1 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) (v2 : UInt256) : FeM UInt256 := do
        let mut total := total
        let mut k := k
        let mut r2 := r2
        let mut r3 := r3
        -- for j in 0..i {
        let v24 ← Range_has_0 r3 v2
        if v24 then
          let v26 ← Range_at_0 r3 v2
          let j := v26
          -- var k: usize = 0
          k := (UInt256.ofNat 0)
          let rec loop_2 (total : UInt256) (k : UInt256) (r2 : Range_Known_0_Unknown) (r3 : Range_Known_0_Unknown) : FeM UInt256 := do
            let mut total := total
            let mut k := k
            let mut r2 := r2
            let mut r3 := r3
            -- while k < j {
            if decide (k.toNat < j.toNat) then
              -- if (i + j + k) % 3 == 0 { total += 1 } else { total += 2 }
              let v33 ← checkedAdd i j
              let v35 ← checkedAdd v33 k
              let v37 ← checkedRem v35 (UInt256.ofNat 3)
              let join : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun total k r2 r3 => do
                let mut total := total
                let mut k := k
                let mut r2 := r2
                let mut r3 := r3
                -- k += 1
                let k_54 ← checkedAdd k (UInt256.ofNat 1)
                k := k_54
                loop_2 total k r2 r3
              if decide (v37 = (UInt256.ofNat 0)) then
                -- total += 1
                let total_42 ← checkedAdd total (UInt256.ofNat 1)
                total := total_42
                join total k r2 r3
              else
                -- total += 2
                let total_48 ← checkedAdd total (UInt256.ofNat 2)
                total := total_48
                join total k r2 r3
            else
              -- if j % 2 == 0 { total += 10 }
              let v58 ← checkedRem j (UInt256.ofNat 2)
              let join_1 : UInt256 → UInt256 → Range_Known_0_Unknown → Range_Known_0_Unknown → FeM UInt256 := fun total k r2 r3 => do
                let mut total := total
                let mut k := k
                let mut r2 := r2
                let mut r3 := r3
                -- for j in 0..i {
                let v70 ← Range_next_0 r3 v2
                loop_1 total k r2 r3 v70
              if decide (v58 = (UInt256.ofNat 0)) then
                -- total += 10
                let total_63 ← checkedAdd total (UInt256.ofNat 10)
                total := total_63
                join_1 total k r2 r3
              else
                join_1 total k r2 r3
          partial_fixpoint
          loop_2 total k r2 r3
        else
          -- for i in 0..n {
          let v74 ← Range_next_0 r2 v1
          loop total k r2 r3 v74
      partial_fixpoint
      loop_1 total k r2 r3 v21
    else
      return total
  partial_fixpoint
  loop total k r2 r3 v10

/-- `ThreeLevels { n } -> u256`, selector `0x81037424` (`threeLevels(uint256)`) -/
def recv_ThreeLevels (n : UInt256) : FeM UInt256 := do
  -- three_levels(n.downcast_truncate())
  let v1 ← downcast_truncate_u256_usize n
  let v2 ← three_levels v1
  return v2


export Lib (Option_u256 Option_u256.None Option_u256.Some)

def init : FeM Unit := do
  return ()

def recv_Triangle (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let total_2 := v1
  total := total_2
  let v3 := (UInt256.ofNat 0)
  let i_4 := v3
  i := i_4
  let rec b1 (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    let i_5 := i
    let i_6 := i_5
    let v7 := decide (i_6.toNat < n.toNat)
    if v7 then
      let v8 := (UInt256.ofNat 1)
      let i_9 := i
      let v10 ← checkedAdd i_9 v8
      let i_11 := v10
      i := i_11
      let v12 := ()
      let i_13 := i
      let i_14 := i_13
      let total_15 := total
      let v16 ← checkedAdd total_15 i_14
      let total_17 := v16
      total := total_17
      let v18 := ()
      b1 total i
    else
      let total_19 := total
      return total_19
  partial_fixpoint
  b1 total i

def step (value_0 : UInt256) : FeM Option_u256 := do
  let value_3 := value_0
  let v4 := (UInt256.ofNat 1)
  let v5 := decide (value_3 = v4)
  let b3 : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if v5 then
    let v6 := Option_u256.None
    let v7 := v6
    b3 v7
  else
    let value_8 := value_0
    let v9 := (UInt256.ofNat 2)
    let v10 ← checkedRem value_8 v9
    let v11 := (UInt256.ofNat 0)
    let v12 := decide (v10 = v11)
    let b6 : Option_u256 → FeM Option_u256 := fun v2 => do
      let v25 := v2
      b3 v25
    if v12 then
      let value_13 := value_0
      let v14 := (UInt256.ofNat 2)
      let v15 ← checkedDiv value_13 v14
      let v16 := Option_u256.Some v15
      let v17 := v16
      b6 v17
    else
      let v18 := (UInt256.ofNat 3)
      let value_19 := value_0
      let v20 ← checkedMul v18 value_19
      let v21 := (UInt256.ofNat 1)
      let v22 ← checkedAdd v20 v21
      let v23 := Option_u256.Some v22
      let v24 := v23
      b6 v24

def recv_Collatz (start_0 : UInt256) (limit : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut steps : UInt256 := default
  let start_2 := start_0
  value := start_2
  let v3 := (UInt256.ofNat 0)
  let steps_4 := v3
  steps := steps_4
  let rec b1 (value : UInt256) (steps : UInt256) : FeM UInt256 := do
    let mut value := value
    let mut steps := steps
    let steps_5 := steps
    let steps_6 := steps_5
    let v7 := decide (steps_6 = limit)
    if v7 then
      return limit
    else
      let v8 := ()
      let v9 := v8
      let value_10 := value
      let value_11 := value_10
      let v12 ← step value_11
      match (← pure v12) with
      | .Some .. =>
        let v13 := v12.Some_0
        let next_14 := v13
        let next_15 := next_14
        value := next_15
        let v16 := ()
        let v17 := (UInt256.ofNat 1)
        let steps_18 := steps
        let v19 ← checkedAdd steps_18 v17
        let steps_20 := v19
        steps := steps_20
        let v21 := ()
        let v22 := v21
        b1 value steps
      | .None =>
        let steps_23 := steps
        return steps_23
  partial_fixpoint
  b1 value steps


export Lib (Option_u256 Option_u256.None Option_u256.Some)

def init : FeM Unit := do
  return ()

def recv_Triangle (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  total := (UInt256.ofNat 0)
  i := (UInt256.ofNat 0)
  let rec b1 (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    let i_5 := i
    let v7 := decide (i_5.toNat < n.toNat)
    if v7 then
      let i_9 := i
      let i_10 ← checkedAdd i_9 (UInt256.ofNat 1)
      i := i_10
      let i_13 := i
      let total_15 := total
      let total_16 ← checkedAdd total_15 i_13
      total := total_16
      b1 total i
    else
      let total_19 := total
      return total_19
  partial_fixpoint
  b1 total i

def step (value : UInt256) : FeM Option_u256 := do
  let v5 := decide (value = (UInt256.ofNat 1))
  let b3 : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if v5 then
    let v6 := Option_u256.None
    b3 v6
  else
    let v10 ← checkedRem value (UInt256.ofNat 2)
    let v12 := decide (v10 = (UInt256.ofNat 0))
    let b6 : Option_u256 → FeM Option_u256 := fun v2 => do
      b3 v2
    if v12 then
      let v15 ← checkedDiv value (UInt256.ofNat 2)
      let v16 := Option_u256.Some v15
      b6 v16
    else
      let v20 ← checkedMul (UInt256.ofNat 3) value
      let v22 ← checkedAdd v20 (UInt256.ofNat 1)
      let v23 := Option_u256.Some v22
      b6 v23

def recv_Collatz (start : UInt256) (limit : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut steps : UInt256 := default
  value := start
  steps := (UInt256.ofNat 0)
  let rec b1 (value : UInt256) (steps : UInt256) : FeM UInt256 := do
    let mut value := value
    let mut steps := steps
    let steps_5 := steps
    let v7 := decide (steps_5 = limit)
    if v7 then
      return limit
    else
      let value_10 := value
      let v12 ← step value_10
      match (← pure v12) with
      | .Some .. =>
        let next := v12.Some_0
        value := next
        let steps_18 := steps
        let steps_19 ← checkedAdd steps_18 (UInt256.ofNat 1)
        steps := steps_19
        b1 value steps
      | .None =>
        let steps_23 := steps
        return steps_23
  partial_fixpoint
  b1 value steps


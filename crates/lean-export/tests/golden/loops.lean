export Lib (Option_u256 Option_u256.None Option_u256.Some)

/-- The `init` of contract `Loops`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Triangle { n } -> u256`, selector `0x50c2698d` (`triangle(uint256)`) -/
def recv_Triangle (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  total := (UInt256.ofNat 0)
  i := (UInt256.ofNat 0)
  let rec loop (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    let i_5 := i
    if decide (i_5.toNat < n.toNat) then
      let i_9 := i
      let i_10 ← checkedAdd i_9 (UInt256.ofNat 1)
      i := i_10
      let i_13 := i
      let total_15 := total
      let total_16 ← checkedAdd total_15 i_13
      total := total_16
      loop total i
    else
      let total_19 := total
      return total_19
  partial_fixpoint
  loop total i

/-- `loops::step` -/
def step (value : UInt256) : FeM Option_u256 := do
  let join : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  if decide (value = (UInt256.ofNat 1)) then
    join Option_u256.None
  else
    let v10 ← checkedRem value (UInt256.ofNat 2)
    let join_1 : Option_u256 → FeM Option_u256 := fun v2 => do
      join v2
    if decide (v10 = (UInt256.ofNat 0)) then
      let v15 ← checkedDiv value (UInt256.ofNat 2)
      join_1 (Option_u256.Some v15)
    else
      let v20 ← checkedMul (UInt256.ofNat 3) value
      let v22 ← checkedAdd v20 (UInt256.ofNat 1)
      join_1 (Option_u256.Some v22)

/-- `Collatz { start, limit } -> u256`, selector `0xea732925` (`collatz(uint256,uint256)`) -/
def recv_Collatz (start : UInt256) (limit : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut steps : UInt256 := default
  value := start
  steps := (UInt256.ofNat 0)
  let rec loop (value : UInt256) (steps : UInt256) : FeM UInt256 := do
    let mut value := value
    let mut steps := steps
    let steps_5 := steps
    if decide (steps_5 = limit) then
      return limit
    else
      let value_10 := value
      let v12 ← step value_10
      match (← pure v12) with
      | .Some .. =>
        value := v12.Some_0
        let steps_18 := steps
        let steps_19 ← checkedAdd steps_18 (UInt256.ofNat 1)
        steps := steps_19
        loop value steps
      | .None =>
        let steps_23 := steps
        return steps_23
  partial_fixpoint
  loop value steps


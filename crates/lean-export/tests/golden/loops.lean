export Lib (Option_u256 Option_u256.None Option_u256.Some)

/-- The `init` of contract `Loops`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Triangle { n } -> u256`, selector `0x50c2698d` (`triangle(uint256)`) -/
def recv_Triangle (n : UInt256) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  -- var total: u256 = 0
  total := (UInt256.ofNat 0)
  -- var i: u256 = 0
  i := (UInt256.ofNat 0)
  let rec loop (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    -- while i < n {
    if decide (i.toNat < n.toNat) then
      -- i += 1
      let i_10 ← checkedAdd i (UInt256.ofNat 1)
      i := i_10
      -- total += i
      let total_16 ← checkedAdd total i
      total := total_16
      loop total i
    else
      return total
  partial_fixpoint
  loop total i

/-- `loops::step` -/
def step (value : UInt256) : FeM Option_u256 := do
  let join : Option_u256 → FeM Option_u256 := fun v1 => do
    return v1
  -- if value == 1 {
  if decide (value = (UInt256.ofNat 1)) then
    join Option_u256.None
  else
    let v10 ← checkedRem value (UInt256.ofNat 2)
    let join_1 : Option_u256 → FeM Option_u256 := fun v2 => do
      join v2
    if decide (v10 = (UInt256.ofNat 0)) then
      -- Option::Some(value / 2)
      let v15 ← checkedDiv value (UInt256.ofNat 2)
      join_1 (Option_u256.Some v15)
    else
      -- Option::Some(3 * value + 1)
      let v20 ← checkedMul (UInt256.ofNat 3) value
      let v22 ← checkedAdd v20 (UInt256.ofNat 1)
      join_1 (Option_u256.Some v22)

/-- `Collatz { start, limit } -> u256`, selector `0xea732925` (`collatz(uint256,uint256)`) -/
def recv_Collatz (start : UInt256) (limit : UInt256) : FeM UInt256 := do
  let mut value : UInt256 := default
  let mut steps : UInt256 := default
  -- var value = start
  value := start
  -- var steps: u256 = 0
  steps := (UInt256.ofNat 0)
  let rec loop (value : UInt256) (steps : UInt256) : FeM UInt256 := do
    let mut value := value
    let mut steps := steps
    -- if steps == limit {
    if decide (steps = limit) then
      return limit
    else
      -- match step(value) {
      let v12 ← step value
      match (← pure v12) with
      | .Some .. =>
        -- value = next
        value := v12.Some_0
        -- steps += 1
        let steps_19 ← checkedAdd steps (UInt256.ofNat 1)
        steps := steps_19
        loop value steps
      | .None =>
        return steps
  partial_fixpoint
  loop value steps


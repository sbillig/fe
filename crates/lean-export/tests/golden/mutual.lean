/-- The `init` of contract `Mutual`, which its source leaves out -/
def init : FeM Unit := do
  return ()

mutual
/-- `mutual::left` -/
def left (n : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if n == 0 {
  if decide (n = (UInt256.ofNat 0)) then
    join (UInt256.ofNat 0)
  else
    -- if n == 0 {
    let v10 ← checkedRem n (UInt256.ofNat 2)
    let join_1 : UInt256 → FeM UInt256 := fun v2 => do
      join v2
    -- if n == 0 {
    if decide (v10 = (UInt256.ofNat 1)) then
      -- step(n - 1)
      let v15 ← checkedSub n (UInt256.ofNat 1)
      let v16 ← step v15
      join_1 v16
    else
      -- right(n - 1)
      let v20 ← checkedSub n (UInt256.ofNat 1)
      let v21 ← right v20
      join_1 v21
partial_fixpoint

/-- `mutual::step` -/
def step (n : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if n == 0 {
  if decide (n = (UInt256.ofNat 0)) then
    join (UInt256.ofNat 1)
  else
    -- left(n - 1)
    let v9 ← checkedSub n (UInt256.ofNat 1)
    let v10 ← left v9
    join v10
partial_fixpoint

/-- `mutual::right` -/
def right (n : UInt256) : FeM UInt256 := do
  let join : UInt256 → FeM UInt256 := fun v1 => do
    return v1
  -- if n == 0 {
  if decide (n = (UInt256.ofNat 0)) then
    join (UInt256.ofNat 2)
  else
    -- step(n - 1)
    let v9 ← checkedSub n (UInt256.ofNat 1)
    let v10 ← step v9
    join v10
partial_fixpoint

end

/-- `Walk { n } -> u256`, selector `0x0581f59d` (`walk(uint256)`) -/
def recv_Walk (n : UInt256) : FeM UInt256 := do
  -- left(n)
  let v1 ← left n
  return v1


/-- The `init` of contract `Pairs`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `tuple::swap` -/
def swap (pair : (UInt256 × UInt256 × Bool)) : FeM (UInt256 × UInt256 × Bool) := do
  return (pair.snd.fst, pair.fst, pair.snd.snd)

/-- `Swap { a, b } -> u256`, selector `0xd96073cf` (`swap(uint256,uint256)`) -/
def recv_Swap (a : UInt256) (b : UInt256) : FeM UInt256 := do
  let mut pair : (UInt256 × UInt256 × Bool) := default
  let mut r1 : (UInt256 × UInt256 × Bool) := default
  r1 := (a, b, true)
  let pair_6 ← swap r1
  pair := pair_6
  let v10 ← checkedAdd pair.snd.fst (UInt256.ofNat 1)
  pair := { pair with snd := { pair.snd with fst := v10 } }
  let join : (UInt256 × UInt256 × Bool) → (UInt256 × UInt256 × Bool) → UInt256 → FeM UInt256 := fun pair r1 v2 => do
    let mut pair := pair
    let mut r1 := r1
    return v2
  if pair.snd.snd then
    let v15 ← checkedSub pair.fst pair.snd.fst
    join pair r1 v15
  else
    join pair r1 (UInt256.ofNat 0)


/-- The `init` of contract `Arrays`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `arrays::bump` -/
def bump (items : (Vector UInt256 4)) («at» : UInt256) («by» : UInt256) : FeM (Unit × (Vector UInt256 4)) := do
  let mut items := items
  checkIndex «at» 4
  let v6 ← checkedAdd items[«at».toNat]! «by»
  checkIndex «at» 4
  items := (items.set! «at».toNat v6)
  return ((), items)

/-- `Pick { index, value } -> u256`, selector `0xa62f4550` (`pick(uint256,uint256)`) -/
def recv_Pick (index : UInt256) (value : UInt256) : FeM UInt256 := do
  let mut items : (Vector UInt256 4) := default
  items := (Vector.replicate 4 value)
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 4
  items := (items.set! v5.toNat (UInt256.ofNat 7))
  let (v9, v9_0) ← bump items index (UInt256.ofNat 1)
  items := v9_0
  checkIndex index 4
  let v11 := (UInt256.ofNat 3)
  checkIndex v11 4
  let v13 ← checkedAdd items[index.toNat]! items[v11.toNat]!
  return v13

/-- `arrays::WEIGHTS` -/
def WEIGHTS : FeM (Vector UInt256 3) := do
  let v3 := #v[(UInt256.ofNat 1), (UInt256.ofNat 10), (UInt256.ofNat 100)]
  return v3

/-- `Weigh { values } -> u256`, selector `0x2f57374e` (`weigh(uint256[3])`) -/
def recv_Weigh (values : (Vector UInt256 3)) : FeM UInt256 := do
  let mut total : UInt256 := default
  let mut i : UInt256 := default
  total := (UInt256.ofNat 0)
  i := (UInt256.ofNat 0)
  let rec loop (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    if decide (i.toNat < (UInt256.ofNat 3).toNat) then
      let i_9 := i
      checkIndex i_9 3
      let v11 ← WEIGHTS
      let i_12 := i
      checkIndex i_12 3
      let v14 ← checkedMul values[i_9.toNat]! v11[i_12.toNat]!
      let total_16 ← checkedAdd total v14
      total := total_16
      let i_21 ← checkedAdd i (UInt256.ofNat 1)
      i := i_21
      loop total i
    else
      return total
  partial_fixpoint
  loop total i


/-- The `init` of contract `Arrays`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `arrays::bump` -/
def bump (items : (Vector UInt256 4)) («at» : UInt256) («by» : UInt256) : FeM (Unit × (Vector UInt256 4)) := do
  let mut items := items
  checkIndex «at» 4
  let v5 := items[«at».toNat]!
  let v6 ← checkedAdd v5 «by»
  checkIndex «at» 4
  items := (items.set! «at».toNat v6)
  return ((), items)

/-- `Pick { index, value } -> u256`, selector `0xa62f4550` (`pick(uint256,uint256)`) -/
def recv_Pick (index : UInt256) (value : UInt256) : FeM UInt256 := do
  let mut items : (Vector UInt256 4) := default
  let items_2 := Vector.replicate 4 value
  items := items_2
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 4
  items := (items.set! v5.toNat (UInt256.ofNat 7))
  let (v9, v9_0) ← bump items index (UInt256.ofNat 1)
  items := v9_0
  checkIndex index 4
  let v10 := items[index.toNat]!
  let v11 := (UInt256.ofNat 3)
  checkIndex v11 4
  let v12 := items[v11.toNat]!
  let v13 ← checkedAdd v10 v12
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
  let rec b1 (total : UInt256) (i : UInt256) : FeM UInt256 := do
    let mut total := total
    let mut i := i
    let i_5 := i
    let v8 := decide (i_5.toNat < (UInt256.ofNat 3).toNat)
    if v8 then
      let i_9 := i
      checkIndex i_9 3
      let v10 := values[i_9.toNat]!
      let v11 ← WEIGHTS
      let i_12 := i
      checkIndex i_12 3
      let v13 := v11[i_12.toNat]!
      let v14 ← checkedMul v10 v13
      let total_15 := total
      let total_16 ← checkedAdd total_15 v14
      total := total_16
      let i_20 := i
      let i_21 ← checkedAdd i_20 (UInt256.ofNat 1)
      i := i_21
      b1 total i
    else
      let total_24 := total
      return total_24
  partial_fixpoint
  b1 total i


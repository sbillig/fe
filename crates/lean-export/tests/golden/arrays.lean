def init : FeM Unit := do
  return ()

def bump (items : (Vector UInt256 4)) (at_1 : UInt256) (by_2 : UInt256) : FeM (Unit × (Vector UInt256 4)) := do
  let mut items := items
  let by_3 := by_2
  let at_4 := at_1
  checkIndex at_4 4
  let v5 := items[at_4.toNat]!
  let v6 ← checkedAdd v5 by_3
  checkIndex at_4 4
  items := (items.set! at_4.toNat v6)
  let v7 := ()
  return ((), items)

def recv_Pick (index : UInt256) (value : UInt256) : FeM UInt256 := do
  let mut items : (Vector UInt256 4) := default
  let v2 := Vector.replicate 4 value
  let items_3 := v2
  items := items_3
  let v4 := (UInt256.ofNat 7)
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 4
  items := (items.set! v5.toNat v4)
  let v6 := ()
  let v8 := (UInt256.ofNat 1)
  let (v9, v9_0) ← bump items index v8
  items := v9_0
  checkIndex index 4
  let v10 := items[index.toNat]!
  let v11 := (UInt256.ofNat 3)
  checkIndex v11 4
  let v12 := items[v11.toNat]!
  let v13 ← checkedAdd v10 v12
  return v13

def WEIGHTS : FeM (Vector UInt256 3) := do
  let v0 := (UInt256.ofNat 1)
  let v1 := (UInt256.ofNat 10)
  let v2 := (UInt256.ofNat 100)
  let v3 := #v[v0, v1, v2]
  return v3

def recv_Weigh (values : (Vector UInt256 3)) : FeM UInt256 := do
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
    let v7 := (UInt256.ofNat 3)
    let v8 := decide (i_6.toNat < v7.toNat)
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
      let v16 ← checkedAdd total_15 v14
      let total_17 := v16
      total := total_17
      let v18 := ()
      let v19 := (UInt256.ofNat 1)
      let i_20 := i
      let v21 ← checkedAdd i_20 v19
      let i_22 := v21
      i := i_22
      let v23 := ()
      b1 total i
    else
      let total_24 := total
      return total_24
  partial_fixpoint
  b1 total i


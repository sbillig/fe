export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_drop MemBuffer_index_mut MemBuffer_new MemBuffer_span_ref MemBuffer_try_new MemBuffer_write_le_u64 MemSlice_empty_u8 downcast_unchecked_u256_usize)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1 Output Output.mk SUCCESS host_allocation)

def init : FeM Unit := do
  return ()

def input_Evm_5 : FeM MemBuffer := do
  let mut r1 : MemBuffer := default
  let v1 ← Evm_calldatasize
  let v2 := v1
  let v3 ← downcast_unchecked_u256_usize v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 0)
  let (v10, v10_0) ← Evm_calldatacopy r1 v7 v8 v2
  r1 := v10_0
  let v11 := r1
  return v11

def copied_MemBuffer (v0 : MemBuffer) (v1 : UInt256) : FeM Output := do
  let mut r0 : MemBuffer := default
  let mut r1 : UInt256 := default
  let v2 ← MemBuffer_len_1 v0
  let v3 := v2
  let v4 ← MemBuffer_try_new v3
  let v5 := v4
  let v6 := v5
  match (← pure v6) with
  | .Some .. =>
    let v7 := v5.Some_0
    r0 := v7
    let v8 := (UInt256.ofNat 0)
    let v9 := v8
    r1 := v9
    let rec b5 (r0 : MemBuffer) (r1 : UInt256) : FeM Output := do
      let mut r0 := r0
      let mut r1 := r1
      let v10 := r1
      let v11 := v10
      let v12 := (UInt256.ofNat 8)
      let v13 ← checkedAdd v11 v12
      let v14 := decide (v13.toNat ≤ v3.toNat)
      if v14 then
        let v15 := r1
        let v16 := v15
        let v17 := r1
        let v18 := v17
        let v19 ← MemBuffer_lane_at v0 v18
        let (v21, v21_0) ← MemBuffer_write_le_u64 r0 v16 v19
        r0 := v21_0
        let v22 := (UInt256.ofNat 8)
        let v23 := r1
        let v24 ← checkedAdd v23 v22
        let v25 := v24
        r1 := v25
        let v26 := ()
        b5 r0 r1
      else
        let rec b8 (r0 : MemBuffer) (r1 : UInt256) : FeM Output := do
          let mut r0 := r0
          let mut r1 := r1
          let v27 := r1
          let v28 := v27
          let v29 := decide (v28.toNat < v3.toNat)
          if v29 then
            let v30 := r1
            let v31 := v30
            let v32 ← MemBuffer_byte_at v0 v31
            let v33 := r1
            let v34 := v33
            let (v36, v36_k) ← MemBuffer_index_mut r0 v34
            mstoreInt (.u 8) v36 v32
            let v36_0 ← v36_k v36
            r0 := v36_0
            let v37 := ()
            let v38 := (UInt256.ofNat 1)
            let v39 := r1
            let v40 ← checkedAdd v39 v38
            let v41 := v40
            r1 := v41
            let v42 := ()
            b8 r0 r1
          else
            let v43 ← SUCCESS
            let v44 := v1
            let v45 := r0
            let v46 := (Output.mk v43 v44 v45)
            return v46
        partial_fixpoint
        b8 r0 r1
    partial_fixpoint
    b5 r0 r1
  | _ =>
    let v47 ← host_allocation v1
    match (← pure v5) with
    | .Some .. =>
      let _ ← MemBuffer_drop v5.Some_0
      pure ()
    | _ => pure ()
    return v47

def recv0_arm0 : FeM Empty := do
  let mut r1 : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_5
  r2 := v0
  let v1 := (UInt256.ofNat 0)
  let v3 ← copied_MemBuffer r2 v1
  let v4 := v3
  r1 := v4
  let _ ← MemBuffer_drop r2
  let v5 := r1.status
  let v6 ← SUCCESS
  let v7 := decide (v5 ≠ v6)
  if v7 then
    let v8 ← MemSlice_empty_u8
    let v11 ← Evm_revert v8
    nomatch v11
  else
    let v12 := ()
    let v13 := v12
    let v16 ← MemBuffer_span_ref r1.output
    let v19 ← Evm_return_data v16
    nomatch v19

def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let _ ← recv0_arm0
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv0_arm0_entry
  else
    let selector ← calldataSelector
    recv0_arm0_entry

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


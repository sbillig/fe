export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_drop MemBuffer_index_mut MemBuffer_new MemBuffer_span_ref MemBuffer_try_new MemBuffer_write_le_u64 MemSlice_empty_u8 downcast_unchecked_u256_usize)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1 Output Output.mk SUCCESS host_allocation)

def init : FeM Unit := do
  return ()

def input_Evm_5 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let v1 ← Evm_calldatasize
  let len := v1
  let v3 ← downcast_unchecked_u256_usize len
  let v4 ← MemBuffer_new v3
  let input_5 := v4
  input := input_5
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 0)
  let (v10, v10_0) ← Evm_calldatacopy input v7 v8 len
  input := v10_0
  let input_11 := input
  return input_11

def copied_MemBuffer (source : MemBuffer) (gas_used_1 : UInt256) : FeM Output := do
  let mut output : MemBuffer := default
  let mut i : UInt256 := default
  let v2 ← MemBuffer_len_1 source
  let len := v2
  let v4 ← MemBuffer_try_new len
  let v5 := v4
  let v6 := v5
  match (← pure v6) with
  | .Some .. =>
    let output_7 := v5.Some_0
    output := output_7
    let v8 := (UInt256.ofNat 0)
    let i_9 := v8
    i := i_9
    let rec b5 (output : MemBuffer) (i : UInt256) : FeM Output := do
      let mut output := output
      let mut i := i
      let i_10 := i
      let i_11 := i_10
      let v12 := (UInt256.ofNat 8)
      let v13 ← checkedAdd i_11 v12
      let v14 := decide (v13.toNat ≤ len.toNat)
      if v14 then
        let i_15 := i
        let i_16 := i_15
        let i_17 := i
        let i_18 := i_17
        let v19 ← MemBuffer_lane_at source i_18
        let (v21, v21_0) ← MemBuffer_write_le_u64 output i_16 v19
        output := v21_0
        let v22 := (UInt256.ofNat 8)
        let i_23 := i
        let v24 ← checkedAdd i_23 v22
        let i_25 := v24
        i := i_25
        let v26 := ()
        b5 output i
      else
        let rec b8 (output : MemBuffer) (i : UInt256) : FeM Output := do
          let mut output := output
          let mut i := i
          let i_27 := i
          let i_28 := i_27
          let v29 := decide (i_28.toNat < len.toNat)
          if v29 then
            let i_30 := i
            let i_31 := i_30
            let v32 ← MemBuffer_byte_at source i_31
            let i_33 := i
            let i_34 := i_33
            let (v36, v36_k) ← MemBuffer_index_mut output i_34
            mstoreInt (.u 8) v36 v32
            let v36_0 ← v36_k v36
            output := v36_0
            let v37 := ()
            let v38 := (UInt256.ofNat 1)
            let i_39 := i
            let v40 ← checkedAdd i_39 v38
            let i_41 := v40
            i := i_41
            let v42 := ()
            b8 output i
          else
            let v43 ← SUCCESS
            let gas_used_44 := gas_used_1
            let output_45 := output
            let v46 := (Output.mk v43 gas_used_44 output_45)
            return v46
        partial_fixpoint
        b8 output i
    partial_fixpoint
    b5 output i
  | _ =>
    let v47 ← host_allocation gas_used_1
    match (← pure v5) with
    | .Some .. =>
      let _ ← MemBuffer_drop v5.Some_0
      pure ()
    | _ => pure ()
    return v47

def recv0_arm0 : FeM Empty := do
  let mut result : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_5
  r2 := v0
  let v1 := (UInt256.ofNat 0)
  let v3 ← copied_MemBuffer r2 v1
  let result_4 := v3
  result := result_4
  let _ ← MemBuffer_drop r2
  let v5 := result.status
  let v6 ← SUCCESS
  let v7 := decide (v5 ≠ v6)
  if v7 then
    let v8 ← MemSlice_empty_u8
    let v11 ← Evm_revert v8
    nomatch v11
  else
    let v12 := ()
    let v13 := v12
    let v16 ← MemBuffer_span_ref result.output
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


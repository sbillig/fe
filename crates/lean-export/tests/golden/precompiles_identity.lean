export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data Evm_revert MemBuffer MemBuffer_drop MemBuffer_index_mut MemBuffer_new MemBuffer_span_ref MemBuffer_try_new MemBuffer_write_le_u64 MemSlice_empty_u8 downcast_unchecked_u256_usize)
export Shared (MemBuffer_byte_at MemBuffer_lane_at MemBuffer_len_1 Output Output.mk SUCCESS host_allocation)

/-- The `init` of contract `Identity`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_5 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let len ← Evm_calldatasize
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  let input_11 := input
  return input_11

/-- `lib::copied<MemBuffer>` -/
def copied_MemBuffer (source : MemBuffer) (gas_used : UInt256) : FeM Output := do
  let mut output : MemBuffer := default
  let mut i : UInt256 := default
  let len ← MemBuffer_len_1 source
  let v4 ← MemBuffer_try_new len
  match (← pure v4) with
  | .Some .. =>
    output := v4.Some_0
    i := (UInt256.ofNat 0)
    let rec loop (output : MemBuffer) (i : UInt256) : FeM Output := do
      let mut output := output
      let mut i := i
      let i_10 := i
      let v13 ← checkedAdd i_10 (UInt256.ofNat 8)
      if decide (v13.toNat ≤ len.toNat) then
        let i_15 := i
        let i_17 := i
        let v19 ← MemBuffer_lane_at source i_17
        let (v21, v21_0) ← MemBuffer_write_le_u64 output i_15 v19
        output := v21_0
        let i_23 := i
        let i_24 ← checkedAdd i_23 (UInt256.ofNat 8)
        i := i_24
        loop output i
      else
        let rec loop_1 (output : MemBuffer) (i : UInt256) : FeM Output := do
          let mut output := output
          let mut i := i
          let i_27 := i
          if decide (i_27.toNat < len.toNat) then
            let i_30 := i
            let v32 ← MemBuffer_byte_at source i_30
            let i_33 := i
            let (v36, v36_k) ← MemBuffer_index_mut output i_33
            mstoreInt (.u 8) v36 v32
            let v36_0 ← v36_k v36
            output := v36_0
            let i_39 := i
            let i_40 ← checkedAdd i_39 (UInt256.ofNat 1)
            i := i_40
            loop_1 output i
          else
            let v43 ← SUCCESS
            let output_45 := output
            return Output.mk v43 gas_used output_45
        partial_fixpoint
        loop_1 output i
    partial_fixpoint
    loop output i
  | _ =>
    let v47 ← host_allocation gas_used
    match (← pure v4) with
    | .Some .. =>
      let _ ← MemBuffer_drop v4.Some_0
      pure ()
    | _ => pure ()
    return v47

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Empty := do
  let mut result : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_5
  r2 := v0
  let result_3 ← copied_MemBuffer r2 (UInt256.ofNat 0)
  result := result_3
  let _ ← MemBuffer_drop r2
  let v5 := result.status
  let v6 ← SUCCESS
  if decide (v5 ≠ v6) then
    let v8 ← MemSlice_empty_u8
    let v11 ← Evm_revert v8
    nomatch v11
  else
    let v16 ← MemBuffer_span_ref result.output
    let v19 ← Evm_return_data v16
    nomatch v19

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  nonPayable
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `Identity`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
  else
    let selector ← calldataSelector
    recv_fallback_entry

/-- The deployment entry of contract `Identity`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


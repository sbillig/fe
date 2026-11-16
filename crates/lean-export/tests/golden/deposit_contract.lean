export Lib (Bytes Bytes_len Bytes_span_ref Bytes_word_at CallData Evm_input Evm_log1 Evm_return_value_Sol_Bytes Evm_return_value_Sol_bool Evm_return_value_Sol_u256 Evm_value FixedBytes_32 FixedBytes_4 MemBuffer MemBuffer_drop MemBuffer_span_ref Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_at_0 Range_has_0 Range_next_0 Range_start_0 Sol_SELECTOR_SIZE Sol_validate_runtime_head Wei Wei_div Wei_eq Wei_ge Wei_rem Wei_zero abi_field_head_size_Bytes abi_field_head_size_FixedBytes_32 abi_field_head_size_FixedBytes_4 decode_msg_field_from_Sol_Bytes_CallData decode_msg_field_from_Sol_FixedBytes_32_CallData decode_msg_field_from_Sol_FixedBytes_4_CallData downcast_truncate_u256_u64 encode_abi_payload_tuple_Bytes_Bytes_Bytes_Bytes_Bytes ether gwei hash_pair hash_tree_root_ByteVector_48 hash_tree_root_ByteVector_96 len_CallData mix_in_length serialize_u64 tuple_u256_u256_u256_u256_merkleize u64_MASK u64_chunk)

def Field.store : Place :=
  .storage (UInt256.ofNat 0)

def Layout.DepositStore : Layout :=
  { storage := [0, 32, 33] }

structure DepositEvent where
  pubkey : Bytes
  withdrawal_credentials : Bytes
  amount : Bytes
  signature : Bytes
  index : Bytes
deriving Inhabited

structure Deposit where
  pubkey : Bytes
  withdrawal_credentials : Bytes
  signature : Bytes
  deposit_data_root : FixedBytes_32
deriving Inhabited

structure GetDepositRoot where
deriving Inhabited

structure GetDepositCount where
deriving Inhabited

structure SupportsInterface where
  interface_id : FixedBytes_4
deriving Inhabited

def DEPOSIT_CONTRACT_TREE_DEPTH : FeM UInt256 := do
  let v0 := (UInt256.ofNat 32)
  return v0

def init : FeM Unit := do
  let mut r1 : Range_Known_0_Unknown := default
  let v1 := (UInt256.ofNat 0)
  let v2 ← DEPOSIT_CONTRACT_TREE_DEPTH
  let v3 := (UInt256.ofNat 1)
  let v4 ← checkedSub v2 v3
  let v5 := ()
  let v6 := (Range_Known_0_Unknown.mk v5 v4)
  r1 := v6
  let v8 ← Range_start_0 r1
  let v9 := v8
  let rec b1 (r1 : Range_Known_0_Unknown) (v0 : UInt256) : FeM Unit := do
    let mut r1 := r1
    let v11 ← Range_has_0 r1 v0
    if v11 then
      let v13 ← Range_at_0 r1 v0
      let height := v13
      checkIndex height 32
      let v15 ← load ((Field.store.field Layout.DepositStore 2).index 1 height)
      let cur := v15
      let v17 ← hash_pair cur cur
      let v18 := (UInt256.ofNat 1)
      let v19 ← checkedAdd height v18
      checkIndex v19 32
      store ((Field.store.field Layout.DepositStore 2).index 1 v19) v17
      let v20 := ()
      let v22 ← Range_next_0 r1 v0
      let v23 := v22
      b1 r1 v23
    else
      let v24 := ()
      return ()
  partial_fixpoint
  b1 r1 v9

def compute_deposit_data_root (pubkey : Bytes) (withdrawal_credentials : Bytes) (signature : Bytes) (amount_gwei : UInt256) : FeM UInt256 := do
  let mut r0 : (UInt256 × UInt256 × UInt256 × UInt256) := default
  let v4 ← Bytes_len withdrawal_credentials
  let v5 := (UInt256.ofNat 32)
  let v6 := decide (v4 = v5)
  if v6 then
    let v7 := ()
    let v8 ← Bytes_span_ref pubkey
    let v9 := (UInt256.ofNat 0)
    let v10 ← Bytes_len pubkey
    let v11 := v10
    let v13 ← hash_tree_root_ByteVector_48 v8 v9 v11
    let v14 := (UInt256.ofNat 0)
    let v15 ← Bytes_word_at withdrawal_credentials v14
    let v16 ← u64_chunk amount_gwei
    let v17 ← Bytes_span_ref signature
    let v18 := (UInt256.ofNat 0)
    let v19 ← Bytes_len signature
    let v20 := v19
    let v22 ← hash_tree_root_ByteVector_96 v17 v18 v20
    let v23 := (v13, v15, v16, v22)
    r0 := v23
    let v25 ← tuple_u256_u256_u256_u256_merkleize r0
    return v25
  else
    M.panic 0x01

def DepositEvent_emit_Evm (self : DepositEvent) : FeM Unit := do
  let mut r0 : MemBuffer := default
  let v2 := self.pubkey
  let v3 := self.withdrawal_credentials
  let v4 := self.amount
  let v5 := self.signature
  let v6 := self.index
  let v7 := (v2, v3, v4, v5, v6)
  let v8 ← encode_abi_payload_tuple_Bytes_Bytes_Bytes_Bytes_Bytes v7
  r0 := v8
  let v10 ← MemBuffer_span_ref r0
  let v11 := (UInt256.ofNat 45506446345753628416285423056165511379837572639148407563471291220684748896453) -- a constant, as the compiler evaluates it
  let v13 ← Evm_log1 v10 v11
  let _ ← MemBuffer_drop r0
  return ()

def emit_Evm_DepositEvent (event_1 : DepositEvent) : FeM Unit := do
  let event_3 := event_1
  let v4 ← DepositEvent_emit_Evm event_3
  return ()

def MAX_DEPOSIT_COUNT : FeM UInt256 := do
  let v0 := (UInt256.ofNat 2)
  let v1 ← DEPOSIT_CONTRACT_TREE_DEPTH
  let v2 := v1
  let v3 ← IntTy.pow (.u 256) v0 v2
  let v4 := (UInt256.ofNat 1)
  let v5 ← checkedSub v3 v4
  return v5

def recv_Deposit (pubkey : Bytes) (withdrawal_credentials : Bytes) (signature : Bytes) (deposit_data_root : FixedBytes_32) : FeM Unit := do
  let mut pubkey := pubkey
  let mut withdrawal_credentials := withdrawal_credentials
  let mut signature := signature
  let mut value : Wei := default
  let mut node : UInt256 := default
  let mut size : UInt256 := default
  let mut r8 : Wei := default
  let mut r9 : Wei := default
  let mut r10 : Wei := default
  let mut r11 : Range_Known_0_Unknown := default
  let v3 ← Bytes_len pubkey
  let v4 := (UInt256.ofNat 48)
  let v5 := decide (v3 = v4)
  if v5 then
    let v6 := ()
    let v8 ← Bytes_len withdrawal_credentials
    let v9 := (UInt256.ofNat 32)
    let v10 := decide (v8 = v9)
    if v10 then
      let v11 := ()
      let v13 ← Bytes_len signature
      let v14 := (UInt256.ofNat 96)
      let v15 := decide (v13 = v14)
      if v15 then
        let v16 := ()
        let v18 ← Evm_value
        let value_19 := v18
        value := value_19
        let v20 := (UInt256.ofNat 1)
        let v21 ← ether v20
        r8 := v21
        let value_22 := value
        let v23 := r8
        let v24 ← Wei_ge value_22 v23
        if v24 then
          let v25 := ()
          let v26 := (UInt256.ofNat 1)
          let v27 ← gwei v26
          let value_28 := value
          let v29 ← Wei_rem value_28 v27
          r9 := v29
          let v30 ← Wei_zero
          r10 := v30
          let v31 := r9
          let v32 := r10
          let v33 ← Wei_eq v31 v32
          if v33 then
            let v34 := ()
            let v35 := (UInt256.ofNat 1)
            let v36 ← gwei v35
            let value_37 := value
            let v38 ← Wei_div value_37 v36
            let deposit_amount := v38
            let v40 ← u64_MASK
            let v41 := decide (deposit_amount.toNat ≤ v40.toNat)
            if v41 then
              let v42 := ()
              let v43 ← downcast_truncate_u256_u64 deposit_amount
              let amount_gwei := v43
              let v48 ← compute_deposit_data_root pubkey withdrawal_credentials signature amount_gwei
              let node_49 := v48
              node := node_49
              let node_50 := node
              let node_51 := node_50
              let v52 := deposit_data_root.val
              let v53 := decide (node_51 = v52)
              if v53 then
                let v54 := ()
                let pubkey_55 := pubkey
                let pubkey_56 := pubkey_55
                let withdrawal_credentials_57 := withdrawal_credentials
                let withdrawal_credentials_58 := withdrawal_credentials_57
                let v59 ← serialize_u64 amount_gwei
                let signature_60 := signature
                let signature_61 := signature_60
                let v62 ← load (Field.store.field Layout.DepositStore 1)
                let v63 ← downcast_truncate_u256_u64 v62
                let v64 ← serialize_u64 v63
                let v65 := (DepositEvent.mk pubkey_56 withdrawal_credentials_58 v59 signature_61 v64)
                let v67 ← emit_Evm_DepositEvent v65
                let v68 ← load (Field.store.field Layout.DepositStore 1)
                let v69 ← MAX_DEPOSIT_COUNT
                let v70 := decide (v68.toNat < v69.toNat)
                if v70 then
                  let v71 := ()
                  let v72 := (UInt256.ofNat 1)
                  let v73 ← load (Field.store.field Layout.DepositStore 1)
                  let v74 ← checkedAdd v73 v72
                  store (Field.store.field Layout.DepositStore 1) v74
                  let v75 := ()
                  let v76 ← load (Field.store.field Layout.DepositStore 1)
                  let size_77 := v76
                  size := size_77
                  let v78 := (UInt256.ofNat 0)
                  let v79 ← DEPOSIT_CONTRACT_TREE_DEPTH
                  let v80 := ()
                  let v81 := (Range_Known_0_Unknown.mk v80 v79)
                  r11 := v81
                  let v83 ← Range_start_0 r11
                  let v84 := v83
                  let rec b25 (pubkey : Bytes) (withdrawal_credentials : Bytes) (signature : Bytes) (value : Wei) (node : UInt256) (size : UInt256) (r8 : Wei) (r9 : Wei) (r10 : Wei) (r11 : Range_Known_0_Unknown) (v1 : UInt256) : FeM Unit := do
                    let mut pubkey := pubkey
                    let mut withdrawal_credentials := withdrawal_credentials
                    let mut signature := signature
                    let mut value := value
                    let mut node := node
                    let mut size := size
                    let mut r8 := r8
                    let mut r9 := r9
                    let mut r10 := r10
                    let mut r11 := r11
                    let v86 ← Range_has_0 r11 v1
                    if v86 then
                      let v88 ← Range_at_0 r11 v1
                      let height := v88
                      let size_90 := size
                      let size_91 := size_90
                      let v92 := (UInt256.ofNat 1)
                      let v93 := size_91 &&& v92
                      let v94 := (UInt256.ofNat 1)
                      let v95 := decide (v93 = v94)
                      if v95 then
                        let node_96 := node
                        let node_97 := node_96
                        checkIndex height 32
                        store ((Field.store.field Layout.DepositStore 0).index 1 height) node_97
                        let v98 := ()
                        return ()
                      else
                        let v99 := ()
                        let v100 := v99
                        checkIndex height 32
                        let v101 ← load ((Field.store.field Layout.DepositStore 0).index 1 height)
                        let node_102 := node
                        let node_103 := node_102
                        let v104 ← hash_pair v101 node_103
                        let node_105 := v104
                        node := node_105
                        let v106 := ()
                        let size_107 := size
                        let size_108 := size_107
                        let v109 := (UInt256.ofNat 2)
                        let v110 ← checkedDiv size_108 v109
                        let size_111 := v110
                        size := size_111
                        let v112 := ()
                        let v114 ← Range_next_0 r11 v1
                        let v115 := v114
                        b25 pubkey withdrawal_credentials signature value node size r8 r9 r10 r11 v115
                    else
                      let v116 := false
                      if v116 then
                        let v117 := ()
                        return ()
                      else
                        M.panic 0x01
                  partial_fixpoint
                  b25 pubkey withdrawal_credentials signature value node size r8 r9 r10 r11 v84
                else
                  M.revert (assertData "DepositContract: merkle tree full")
              else
                M.revert (assertData "DepositContract: reconstructed DepositData does not match supplied deposit_data_root")
            else
              M.revert (assertData "DepositContract: deposit value too high")
          else
            M.revert (assertData "DepositContract: deposit value not multiple of gwei")
        else
          M.revert (assertData "DepositContract: deposit value too low")
      else
        M.revert (assertData "DepositContract: invalid signature length")
    else
      M.revert (assertData "DepositContract: invalid withdrawal_credentials length")
  else
    M.revert (assertData "DepositContract: invalid pubkey length")

def recv_GetDepositRoot : FeM UInt256 := do
  let mut node : UInt256 := default
  let mut size : UInt256 := default
  let mut r3 : Range_Known_0_Unknown := default
  let v1 := (UInt256.ofNat 0)
  let node_2 := v1
  node := node_2
  let v3 ← load (Field.store.field Layout.DepositStore 1)
  let size_4 := v3
  size := size_4
  let v5 := (UInt256.ofNat 0)
  let v6 ← DEPOSIT_CONTRACT_TREE_DEPTH
  let v7 := ()
  let v8 := (Range_Known_0_Unknown.mk v7 v6)
  r3 := v8
  let v10 ← Range_start_0 r3
  let v11 := v10
  let rec b1 (node : UInt256) (size : UInt256) (r3 : Range_Known_0_Unknown) (v0 : UInt256) : FeM UInt256 := do
    let mut node := node
    let mut size := size
    let mut r3 := r3
    let v13 ← Range_has_0 r3 v0
    if v13 then
      let v15 ← Range_at_0 r3 v0
      let height := v15
      let size_17 := size
      let size_18 := size_17
      let v19 := (UInt256.ofNat 1)
      let v20 := size_18 &&& v19
      let v21 := (UInt256.ofNat 1)
      let v22 := decide (v20 = v21)
      let b7 : UInt256 → UInt256 → Range_Known_0_Unknown → FeM UInt256 := fun node size r3 => do
        let mut node := node
        let mut size := size
        let mut r3 := r3
        let size_37 := size
        let size_38 := size_37
        let v39 := (UInt256.ofNat 2)
        let v40 ← checkedDiv size_38 v39
        let size_41 := v40
        size := size_41
        let v42 := ()
        let v44 ← Range_next_0 r3 v0
        let v45 := v44
        b1 node size r3 v45
      if v22 then
        checkIndex height 32
        let v23 ← load ((Field.store.field Layout.DepositStore 0).index 1 height)
        let node_24 := node
        let node_25 := node_24
        let v26 ← hash_pair v23 node_25
        let node_27 := v26
        node := node_27
        let v28 := ()
        let v29 := v28
        b7 node size r3
      else
        let node_30 := node
        let node_31 := node_30
        checkIndex height 32
        let v32 ← load ((Field.store.field Layout.DepositStore 2).index 1 height)
        let v33 ← hash_pair node_31 v32
        let node_34 := v33
        node := node_34
        let v35 := ()
        let v36 := v35
        b7 node size r3
    else
      let node_46 := node
      let node_47 := node_46
      let v48 ← load (Field.store.field Layout.DepositStore 1)
      let v49 ← downcast_truncate_u256_u64 v48
      let v50 ← mix_in_length node_47 v49
      return v50
  partial_fixpoint
  b1 node size r3 v11

def recv_GetDepositCount : FeM Bytes := do
  let v0 ← load (Field.store.field Layout.DepositStore 1)
  let v1 ← downcast_truncate_u256_u64 v0
  let count := v1
  let v3 ← serialize_u64 count
  return v3

def recv_SupportsInterface (interface_id : FixedBytes_4) : FeM Bool := do
  let v2 := interface_id.val
  let v3 := (UInt256.ofNat 33540519)
  let v4 := decide (v2 = v3)
  let b3 : Bool → FeM Bool := fun v1 => do
    return v1
  let b1 : FeM Bool := do
    let v10 := true
    let v11 := v10
    b3 v11
  if v4 then
    b1
  else
    let v5 := interface_id.val
    let v6 := (UInt256.ofNat 2237925639)
    let v7 := decide (v5 = v6)
    if v7 then
      b1
    else
      let v8 := false
      let v9 := v8
      b3 v9

def Deposit_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_Bytes
  let v2 ← checkedAdd v0 v1
  let v3 ← abi_field_head_size_Bytes
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_Bytes
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_FixedBytes_32
  let v8 ← checkedAdd v6 v7
  return v8

def Deposit_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base_1 : UInt256) : FeM Deposit := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_Bytes_CallData __fe_msg_decode_input __fe_msg_decode_base_1 __fe_msg_decode_base_1 __fe_msg_decode_input_len
  let __fe_field_0_5 := v4
  let __fe_msg_decode_base_6 := __fe_msg_decode_base_1
  let v7 ← abi_field_head_size_Bytes
  let v8 ← checkedAdd __fe_msg_decode_base_6 v7
  let v9 ← decode_msg_field_from_Sol_Bytes_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v8 __fe_msg_decode_input_len
  let __fe_field_1_10 := v9
  let __fe_msg_decode_base_11 := __fe_msg_decode_base_1
  let v12 ← abi_field_head_size_Bytes
  let v13 ← checkedAdd __fe_msg_decode_base_11 v12
  let v14 ← abi_field_head_size_Bytes
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_Bytes_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v15 __fe_msg_decode_input_len
  let __fe_field_2_17 := v16
  let __fe_msg_decode_base_18 := __fe_msg_decode_base_1
  let v19 ← abi_field_head_size_Bytes
  let v20 ← checkedAdd __fe_msg_decode_base_18 v19
  let v21 ← abi_field_head_size_Bytes
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_Bytes
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_FixedBytes_32_CallData __fe_msg_decode_input __fe_msg_decode_base_1 v24 __fe_msg_decode_input_len
  let __fe_field_3 := v25
  let __fe_field_0_27 := __fe_field_0_5
  let __fe_field_1_28 := __fe_field_1_10
  let __fe_field_2_29 := __fe_field_2_17
  let v30 := (Deposit.mk __fe_field_0_27 __fe_field_1_28 __fe_field_2_29 __fe_field_3)
  return v30

def decode_from_prechecked_head_Deposit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Deposit := do
  let v3 ← Deposit_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_Deposit : FeM Deposit := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← Deposit_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Deposit_Sol_CallData input_11 base len
  return v12

def recv_Deposit_entry : FeM Empty := do
  let message ← decode_runtime_args_Evm_Sol_Deposit
  let _ ← recv_Deposit message.pubkey message.withdrawal_credentials message.signature message.deposit_data_root
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

def GetDepositRoot_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def GetDepositRoot_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM GetDepositRoot := do
  let v2 := (GetDepositRoot.mk)
  return v2

def decode_from_prechecked_head_GetDepositRoot_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM GetDepositRoot := do
  let v3 ← GetDepositRoot_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_GetDepositRoot : FeM GetDepositRoot := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← GetDepositRoot_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_GetDepositRoot_Sol_CallData input_11 base len
  return v12

def recv_GetDepositRoot_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_GetDepositRoot
  let result ← recv_GetDepositRoot
  Evm_return_value_Sol_u256 result

def GetDepositCount_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  return v0

def GetDepositCount_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM GetDepositCount := do
  let v2 := (GetDepositCount.mk)
  return v2

def decode_from_prechecked_head_GetDepositCount_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM GetDepositCount := do
  let v3 ← GetDepositCount_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_GetDepositCount : FeM GetDepositCount := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← GetDepositCount_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_GetDepositCount_Sol_CallData input_11 base len
  return v12

def recv_GetDepositCount_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_GetDepositCount
  let result ← recv_GetDepositCount
  Evm_return_value_Sol_Bytes result

def SupportsInterface_HEAD_SIZE : FeM UInt256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 ← abi_field_head_size_FixedBytes_4
  let v2 ← checkedAdd v0 v1
  return v2

def SupportsInterface_decode_from_CallData (__fe_msg_decode_input : CallData) (__fe_msg_decode_base : UInt256) : FeM SupportsInterface := do
  let v2 ← len_CallData __fe_msg_decode_input
  let __fe_msg_decode_input_len := v2
  let v4 ← decode_msg_field_from_Sol_FixedBytes_4_CallData __fe_msg_decode_input __fe_msg_decode_base __fe_msg_decode_base __fe_msg_decode_input_len
  let __fe_field_0 := v4
  let v6 := (SupportsInterface.mk __fe_field_0)
  return v6

def decode_from_prechecked_head_SupportsInterface_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM SupportsInterface := do
  let v3 ← SupportsInterface_decode_from_CallData input pos
  return v3

def decode_runtime_args_Evm_Sol_SupportsInterface : FeM SupportsInterface := do
  let mut input : CallData := default
  let v2 ← Evm_input
  let input_3 := v2
  input := input_3
  let v4 ← Sol_SELECTOR_SIZE
  let base := v4
  let input_6 := input
  let v7 ← len_CallData input_6
  let len := v7
  let v9 ← SupportsInterface_HEAD_SIZE
  let v10 ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_SupportsInterface_Sol_CallData input_11 base len
  return v12

def recv_SupportsInterface_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_SupportsInterface
  let result ← recv_SupportsInterface message.interface_id
  Evm_return_value_Sol_bool result

def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x01ffc9a7 then
      recv_SupportsInterface_entry
    else if selector = UInt256.ofNat 0x22895118 then
      recv_Deposit_entry
    else if selector = UInt256.ofNat 0x621fd130 then
      recv_GetDepositCount_entry
    else if selector = UInt256.ofNat 0xc5f2892f then
      recv_GetDepositRoot_entry
    else
      revertEmpty

def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


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

/-- `deposit_contract::DEPOSIT_CONTRACT_TREE_DEPTH` -/
def DEPOSIT_CONTRACT_TREE_DEPTH : FeM UInt256 := do
  return (UInt256.ofNat 32)

/-- `init()` of contract `DepositContract` -/
def init : FeM Unit := do
  let mut r1 : Range_Known_0_Unknown := default
  let v2 ← DEPOSIT_CONTRACT_TREE_DEPTH
  let v4 ← checkedSub v2 (UInt256.ofNat 1)
  let v6 := (Range_Known_0_Unknown.mk () v4)
  r1 := v6
  let v8 ← Range_start_0 r1
  let rec b1 (r1 : Range_Known_0_Unknown) (v0 : UInt256) : FeM Unit := do
    let mut r1 := r1
    let v11 ← Range_has_0 r1 v0
    if v11 then
      let v13 ← Range_at_0 r1 v0
      let height := v13
      checkIndex height 32
      let cur ← load ((Field.store.field Layout.DepositStore 2).index 1 height)
      let v17 ← hash_pair cur cur
      let v19 ← checkedAdd height (UInt256.ofNat 1)
      checkIndex v19 32
      store ((Field.store.field Layout.DepositStore 2).index 1 v19) v17
      let v22 ← Range_next_0 r1 v0
      b1 r1 v22
    else
      return ()
  partial_fixpoint
  b1 r1 v8

/-- `deposit_contract::compute_deposit_data_root` -/
def compute_deposit_data_root (pubkey : Bytes) (withdrawal_credentials : Bytes) (signature : Bytes) (amount_gwei : UInt256) : FeM UInt256 := do
  let mut r0 : (UInt256 × UInt256 × UInt256 × UInt256) := default
  let v4 ← Bytes_len withdrawal_credentials
  let v6 := decide (v4 = (UInt256.ofNat 32))
  if v6 then
    let v8 ← Bytes_span_ref pubkey
    let v10 ← Bytes_len pubkey
    let v11 := v10
    let v13 ← hash_tree_root_ByteVector_48 v8 (UInt256.ofNat 0) v11
    let v15 ← Bytes_word_at withdrawal_credentials (UInt256.ofNat 0)
    let v16 ← u64_chunk amount_gwei
    let v17 ← Bytes_span_ref signature
    let v19 ← Bytes_len signature
    let v20 := v19
    let v22 ← hash_tree_root_ByteVector_96 v17 (UInt256.ofNat 0) v20
    let v23 := (v13, v15, v16, v22)
    r0 := v23
    let v25 ← tuple_u256_u256_u256_u256_merkleize r0
    return v25
  else
    M.panic 0x01

/-- `<DepositEvent as Event>::emit<Evm>` -/
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
  let _ ← Evm_log1 v10 v11
  let _ ← MemBuffer_drop r0
  return ()

/-- `std::evm::effects::Log::emit<Evm, DepositEvent>` -/
def emit_Evm_DepositEvent (event : DepositEvent) : FeM Unit := do
  let _ ← DepositEvent_emit_Evm event
  return ()

/-- `deposit_contract::MAX_DEPOSIT_COUNT` -/
def MAX_DEPOSIT_COUNT : FeM UInt256 := do
  let v1 ← DEPOSIT_CONTRACT_TREE_DEPTH
  let v2 := v1
  let v3 ← IntTy.pow (.u 256) (UInt256.ofNat 2) v2
  let v5 ← checkedSub v3 (UInt256.ofNat 1)
  return v5

/-- `Deposit { pubkey, withdrawal_credentials, signature, deposit_data_root }`, selector `0x22895118` (`deposit(bytes,bytes,bytes,bytes32)`) -/
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
  let v5 := decide (v3 = (UInt256.ofNat 48))
  if v5 then
    let v8 ← Bytes_len withdrawal_credentials
    let v10 := decide (v8 = (UInt256.ofNat 32))
    if v10 then
      let v13 ← Bytes_len signature
      let v15 := decide (v13 = (UInt256.ofNat 96))
      if v15 then
        let value_18 ← Evm_value
        value := value_18
        let v21 ← ether (UInt256.ofNat 1)
        r8 := v21
        let value_22 := value
        let v23 := r8
        let v24 ← Wei_ge value_22 v23
        if v24 then
          let v27 ← gwei (UInt256.ofNat 1)
          let value_28 := value
          let v29 ← Wei_rem value_28 v27
          r9 := v29
          let v30 ← Wei_zero
          r10 := v30
          let v31 := r9
          let v32 := r10
          let v33 ← Wei_eq v31 v32
          if v33 then
            let v36 ← gwei (UInt256.ofNat 1)
            let value_37 := value
            let deposit_amount ← Wei_div value_37 v36
            let v40 ← u64_MASK
            let v41 := decide (deposit_amount.toNat ≤ v40.toNat)
            if v41 then
              let amount_gwei ← downcast_truncate_u256_u64 deposit_amount
              let node_48 ← compute_deposit_data_root pubkey withdrawal_credentials signature amount_gwei
              node := node_48
              let node_50 := node
              let v52 := deposit_data_root.val
              let v53 := decide (node_50 = v52)
              if v53 then
                let pubkey_55 := pubkey
                let withdrawal_credentials_57 := withdrawal_credentials
                let v59 ← serialize_u64 amount_gwei
                let signature_60 := signature
                let v62 ← load (Field.store.field Layout.DepositStore 1)
                let v63 ← downcast_truncate_u256_u64 v62
                let v64 ← serialize_u64 v63
                let v65 := (DepositEvent.mk pubkey_55 withdrawal_credentials_57 v59 signature_60 v64)
                let _ ← emit_Evm_DepositEvent v65
                let v68 ← load (Field.store.field Layout.DepositStore 1)
                let v69 ← MAX_DEPOSIT_COUNT
                let v70 := decide (v68.toNat < v69.toNat)
                if v70 then
                  let v73 ← load (Field.store.field Layout.DepositStore 1)
                  let v74 ← checkedAdd v73 (UInt256.ofNat 1)
                  store (Field.store.field Layout.DepositStore 1) v74
                  let size_76 ← load (Field.store.field Layout.DepositStore 1)
                  size := size_76
                  let v79 ← DEPOSIT_CONTRACT_TREE_DEPTH
                  let v81 := (Range_Known_0_Unknown.mk () v79)
                  r11 := v81
                  let v83 ← Range_start_0 r11
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
                      let v93 := size_90 &&& (UInt256.ofNat 1)
                      let v95 := decide (v93 = (UInt256.ofNat 1))
                      if v95 then
                        let node_96 := node
                        checkIndex height 32
                        store ((Field.store.field Layout.DepositStore 0).index 1 height) node_96
                        return ()
                      else
                        checkIndex height 32
                        let v101 ← load ((Field.store.field Layout.DepositStore 0).index 1 height)
                        let node_102 := node
                        let node_104 ← hash_pair v101 node_102
                        node := node_104
                        let size_107 := size
                        let size_110 ← checkedDiv size_107 (UInt256.ofNat 2)
                        size := size_110
                        let v114 ← Range_next_0 r11 v1
                        b25 pubkey withdrawal_credentials signature value node size r8 r9 r10 r11 v114
                    else
                      if false then
                        return ()
                      else
                        M.panic 0x01
                  partial_fixpoint
                  b25 pubkey withdrawal_credentials signature value node size r8 r9 r10 r11 v83
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

/-- `GetDepositRoot -> u256`, selector `0xc5f2892f` (`get_deposit_root()`) -/
def recv_GetDepositRoot : FeM UInt256 := do
  let mut node : UInt256 := default
  let mut size : UInt256 := default
  let mut r3 : Range_Known_0_Unknown := default
  node := (UInt256.ofNat 0)
  let size_3 ← load (Field.store.field Layout.DepositStore 1)
  size := size_3
  let v6 ← DEPOSIT_CONTRACT_TREE_DEPTH
  let v8 := (Range_Known_0_Unknown.mk () v6)
  r3 := v8
  let v10 ← Range_start_0 r3
  let rec b1 (node : UInt256) (size : UInt256) (r3 : Range_Known_0_Unknown) (v0 : UInt256) : FeM UInt256 := do
    let mut node := node
    let mut size := size
    let mut r3 := r3
    let v13 ← Range_has_0 r3 v0
    if v13 then
      let v15 ← Range_at_0 r3 v0
      let height := v15
      let size_17 := size
      let v20 := size_17 &&& (UInt256.ofNat 1)
      let v22 := decide (v20 = (UInt256.ofNat 1))
      let b7 : UInt256 → UInt256 → Range_Known_0_Unknown → FeM UInt256 := fun node size r3 => do
        let mut node := node
        let mut size := size
        let mut r3 := r3
        let size_37 := size
        let size_40 ← checkedDiv size_37 (UInt256.ofNat 2)
        size := size_40
        let v44 ← Range_next_0 r3 v0
        b1 node size r3 v44
      if v22 then
        checkIndex height 32
        let v23 ← load ((Field.store.field Layout.DepositStore 0).index 1 height)
        let node_24 := node
        let node_26 ← hash_pair v23 node_24
        node := node_26
        b7 node size r3
      else
        let node_30 := node
        checkIndex height 32
        let v32 ← load ((Field.store.field Layout.DepositStore 2).index 1 height)
        let node_33 ← hash_pair node_30 v32
        node := node_33
        b7 node size r3
    else
      let node_46 := node
      let v48 ← load (Field.store.field Layout.DepositStore 1)
      let v49 ← downcast_truncate_u256_u64 v48
      let v50 ← mix_in_length node_46 v49
      return v50
  partial_fixpoint
  b1 node size r3 v10

/-- `GetDepositCount -> Bytes`, selector `0x621fd130` (`get_deposit_count()`) -/
def recv_GetDepositCount : FeM Bytes := do
  let v0 ← load (Field.store.field Layout.DepositStore 1)
  let count ← downcast_truncate_u256_u64 v0
  let v3 ← serialize_u64 count
  return v3

/-- `SupportsInterface { interface_id } -> bool`, selector `0x01ffc9a7` (`supportsInterface(bytes4)`) -/
def recv_SupportsInterface (interface_id : FixedBytes_4) : FeM Bool := do
  let v2 := interface_id.val
  let v4 := decide (v2 = (UInt256.ofNat 33540519))
  let b3 : Bool → FeM Bool := fun v1 => do
    return v1
  let b1 : FeM Bool := do
    b3 true
  if v4 then
    b1
  else
    let v5 := interface_id.val
    let v7 := decide (v5 = (UInt256.ofNat 2237925639))
    if v7 then
      b1
    else
      b3 false

/-- `<Deposit as AbiSize>::HEAD_SIZE` -/
def Deposit_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_Bytes
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  let v3 ← abi_field_head_size_Bytes
  let v4 ← checkedAdd v2 v3
  let v5 ← abi_field_head_size_Bytes
  let v6 ← checkedAdd v4 v5
  let v7 ← abi_field_head_size_FixedBytes_32
  let v8 ← checkedAdd v6 v7
  return v8

/-- `<Deposit as Decode>::decode_from<CallData>` -/
def Deposit_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Deposit := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_Bytes_CallData v0 v1 v1 v2
  let v7 ← abi_field_head_size_Bytes
  let v8 ← checkedAdd v1 v7
  let v9 ← decode_msg_field_from_Sol_Bytes_CallData v0 v1 v8 v2
  let v12 ← abi_field_head_size_Bytes
  let v13 ← checkedAdd v1 v12
  let v14 ← abi_field_head_size_Bytes
  let v15 ← checkedAdd v13 v14
  let v16 ← decode_msg_field_from_Sol_Bytes_CallData v0 v1 v15 v2
  let v19 ← abi_field_head_size_Bytes
  let v20 ← checkedAdd v1 v19
  let v21 ← abi_field_head_size_Bytes
  let v22 ← checkedAdd v20 v21
  let v23 ← abi_field_head_size_Bytes
  let v24 ← checkedAdd v22 v23
  let v25 ← decode_msg_field_from_Sol_FixedBytes_32_CallData v0 v1 v24 v2
  let v30 := (Deposit.mk v4 v9 v16 v25)
  return v30

/-- `core::abi::Decode::decode_from_prechecked_head<Deposit, Sol, CallData>` -/
def decode_from_prechecked_head_Deposit_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Deposit := do
  let v3 ← Deposit_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Deposit>` -/
def decode_runtime_args_Evm_Sol_Deposit : FeM Deposit := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← Deposit_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_Deposit_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_Deposit`, for `deposit_contract::DepositMsg::Deposit` -/
def recv_Deposit_entry : FeM Empty := do
  let message ← decode_runtime_args_Evm_Sol_Deposit
  let _ ← recv_Deposit message.pubkey message.withdrawal_credentials message.signature message.deposit_data_root
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<GetDepositRoot as AbiSize>::HEAD_SIZE` -/
def GetDepositRoot_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<GetDepositRoot as Decode>::decode_from<CallData>` -/
def GetDepositRoot_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM GetDepositRoot := do
  let v2 := (GetDepositRoot.mk)
  return v2

/-- `core::abi::Decode::decode_from_prechecked_head<GetDepositRoot, Sol, CallData>` -/
def decode_from_prechecked_head_GetDepositRoot_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM GetDepositRoot := do
  let v3 ← GetDepositRoot_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, GetDepositRoot>` -/
def decode_runtime_args_Evm_Sol_GetDepositRoot : FeM GetDepositRoot := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← GetDepositRoot_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_GetDepositRoot_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_GetDepositRoot`, for `deposit_contract::DepositMsg::GetDepositRoot` -/
def recv_GetDepositRoot_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_GetDepositRoot
  let result ← recv_GetDepositRoot
  Evm_return_value_Sol_u256 result

/-- `<GetDepositCount as AbiSize>::HEAD_SIZE` -/
def GetDepositCount_HEAD_SIZE : FeM UInt256 := do
  return (UInt256.ofNat 0)

/-- `<GetDepositCount as Decode>::decode_from<CallData>` -/
def GetDepositCount_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM GetDepositCount := do
  let v2 := (GetDepositCount.mk)
  return v2

/-- `core::abi::Decode::decode_from_prechecked_head<GetDepositCount, Sol, CallData>` -/
def decode_from_prechecked_head_GetDepositCount_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM GetDepositCount := do
  let v3 ← GetDepositCount_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, GetDepositCount>` -/
def decode_runtime_args_Evm_Sol_GetDepositCount : FeM GetDepositCount := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← GetDepositCount_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_GetDepositCount_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_GetDepositCount`, for `deposit_contract::DepositMsg::GetDepositCount` -/
def recv_GetDepositCount_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_GetDepositCount
  let result ← recv_GetDepositCount
  Evm_return_value_Sol_Bytes result

/-- `<SupportsInterface as AbiSize>::HEAD_SIZE` -/
def SupportsInterface_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_FixedBytes_4
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<SupportsInterface as Decode>::decode_from<CallData>` -/
def SupportsInterface_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM SupportsInterface := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_FixedBytes_4_CallData v0 v1 v1 v2
  let v6 := (SupportsInterface.mk v4)
  return v6

/-- `core::abi::Decode::decode_from_prechecked_head<SupportsInterface, Sol, CallData>` -/
def decode_from_prechecked_head_SupportsInterface_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM SupportsInterface := do
  let v3 ← SupportsInterface_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, SupportsInterface>` -/
def decode_runtime_args_Evm_Sol_SupportsInterface : FeM SupportsInterface := do
  let mut input : CallData := default
  let input_2 ← Evm_input
  input := input_2
  let base ← Sol_SELECTOR_SIZE
  let input_6 := input
  let len ← len_CallData input_6
  let v9 ← SupportsInterface_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  let input_11 := input
  let v12 ← decode_from_prechecked_head_SupportsInterface_Sol_CallData input_11 base len
  return v12

/-- The entry of `recv_SupportsInterface`, for `deposit_contract::DepositMsg::SupportsInterface` -/
def recv_SupportsInterface_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_SupportsInterface
  let result ← recv_SupportsInterface message.interface_id
  Evm_return_value_Sol_bool result

/-- The runtime entry of contract `DepositContract`: the entry of the arm the call's selector picks -/
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

/-- The deployment entry of contract `DepositContract`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


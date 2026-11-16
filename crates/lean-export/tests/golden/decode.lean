export Lib (Evm_decode_args_tuple_u256_u256)

/-- The `init` of contract `Decoder`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `Probe -> u256`, selector `0xb74af5a9` (`probe()`) -/
def recv_Probe : FeM UInt256 := do
  -- let args: (u256, u256) = evm.decode_args()
  let args ← Evm_decode_args_tuple_u256_u256
  -- args.0 + args.1
  let v5 ← checkedAdd args.fst args.snd
  return v5


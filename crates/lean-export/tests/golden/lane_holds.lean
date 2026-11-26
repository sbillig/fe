export Lib (CallData Evm_input FixedBytes_4 FixedBytes_4.mk Option_mut_u8 Option_mut_u8.None Option_mut_u8.Some Sol_SELECTOR_SIZE Sol_validate_runtime_head abi_field_head_size_bool decode_msg_field_from_Sol_bool_CallData encode_bytes_return_from_calldata len_CallData)

def Field.a : Place :=
  .storage (UInt256.ofNat 0)

def Field.small : Place :=
  .storage (UInt256.ofNat 1)

def Layout.Small : Layout :=
  { storage := [0, 0] }

/-- `small.b` -/
abbrev Field.small.b : Place :=
  Field.small.field Layout.Small 1

def Field.c : Place :=
  .storage (UInt256.ofNat 0)

def Field.b : Place :=
  .storage (UInt256.ofNat 0)

/-- `small.a` -/
abbrev Field.small.a : Place :=
  Field.small.field Layout.Small 0

def Field.tagged : Place :=
  .storage (UInt256.ofNat 2)

def Layout.Tagged : Layout :=
  { storage := [0, 1] }

/-- `tagged.tag` -/
abbrev Field.tagged.tag : Place :=
  Field.tagged.field Layout.Tagged 0

structure Arg where
deriving Inhabited

structure Effect where
deriving Inhabited

structure Field where
deriving Inhabited

structure Maybe where
  h : Bool
deriving Inhabited

structure Pair where
deriving Inhabited

structure Forward where
deriving Inhabited

structure Session where
deriving Inhabited

structure PickThenWrite where
  c : Bool
deriving Inhabited

structure MaybeThenWrite where
  c : Bool
deriving Inhabited

structure DirectThenWrite where
deriving Inhabited

structure Tag where
deriving Inhabited

structure TagVal where
deriving Inhabited

/-- The `init` of contract `Lanes`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lane_holds::halt` -/
def halt : FeM Empty := do
  -- encode_bytes_return_from_calldata(data_ptr: 4, len: 32)
  let v2 ← encode_bytes_return_from_calldata (UInt256.ofNat 4) (UInt256.ofNat 32)
  nomatch v2

/-- `lane_holds::bump_halt` -/
def bump_halt_unwinds (x : UInt256) : FeM (Unwound Empty × UInt256) := do
  let mut x := x
  -- x += 1
  let v3 ← IntTy.add (.u 8) x (UInt256.ofNat 1)
  x := v3
  -- halt()
  let v5_u ← Fe.M.unwind (halt)
  let some v5 := v5_u.ret? | do
    return (.halt v5_u.data, x)
  nomatch v5

/-- `Arg`, selector `0x7435e911` (`arg()`) -/
def recv_Arg : FeM Empty := do
  let mut h0 : UInt256 := default
  -- bump_halt(mut a)
  h0 := (← loadIntAt (.u 8) Field.a ⟨0, 1, true⟩)
  let (v1_u, v1_0) ← bump_halt_unwinds h0
  h0 := v1_0
  let some v1 := v1_u.ret? | do
    storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ h0
    Fe.M.stop v1_u.data
  nomatch v1

/-- `lane_holds::bump_effect_halt` -/
def bump_effect_halt_unwinds (x : UInt256) : FeM (Unwound Empty × UInt256) := do
  let mut x := x
  -- x += 1
  let v2 ← IntTy.add (.u 8) x (UInt256.ofNat 1)
  x := v2
  -- halt()
  let v4_u ← Fe.M.unwind (halt)
  let some v4 := v4_u.ret? | do
    return (.halt v4_u.data, x)
  nomatch v4

/-- `Effect`, selector `0xdae42fd8` (`effect()`) -/
def recv_Effect : FeM Empty := do
  -- bump_effect_halt()
  let v0_e0 ← loadIntAt (.u 8) Field.small.b ⟨1, 1, true⟩
  let (v0_u, v0_0) ← bump_effect_halt_unwinds v0_e0
  storeIntAt (.u 8) Field.small.b ⟨1, 1, true⟩ v0_0
  let some v0 := v0_u.ret? | do
    Fe.M.stop v0_u.data
  nomatch v0

/-- `lane_holds::bump_effect_halt` -/
def bump_effect_halt : FeM Empty := do
  let mut a ← loadIntAt (.u 8) Field.a ⟨0, 1, true⟩
  -- x += 1
  let v2 ← IntTy.add (.u 8) a (UInt256.ofNat 1)
  a := v2
  -- halt()
  let v4_u ← Fe.M.unwind (halt)
  let some v4 := v4_u.ret? | do
    storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ a
    Fe.M.stop v4_u.data
  nomatch v4

/-- `Field`, selector `0x4f28bf0e` (`field()`) -/
def recv_Field : FeM Empty := do
  -- bump_effect_halt()
  let v0 ← bump_effect_halt
  nomatch v0

/-- `lane_holds::maybe_halt` -/
def maybe_halt (h : Bool) : FeM Unit := do
  -- if h { halt() }
  if h then
    -- halt()
    let v2 ← halt
    nomatch v2
  else
    return ()

/-- `lane_holds::bump_maybe_halt` -/
def bump_maybe_halt_unwinds (x : UInt256) (h : Bool) : FeM (Unwound Unit × UInt256) := do
  let mut x := x
  -- x += 1
  let v4 ← IntTy.add (.u 8) x (UInt256.ofNat 1)
  x := v4
  -- maybe_halt(h)
  let v6_u ← Fe.M.unwind (maybe_halt h)
  let some v6 := v6_u.ret? | do
    return (.halt v6_u.data, x)
  return (.ret (), x)

/-- `Maybe { h }`, selector `0xe13e4d4f` (`maybe(bool)`) -/
def recv_Maybe (h : Bool) : FeM Unit := do
  let mut h1 : UInt256 := default
  -- bump_maybe_halt(mut a, h)
  h1 := (← loadIntAt (.u 8) Field.a ⟨0, 1, true⟩)
  let (v2_u, v2_0) ← bump_maybe_halt_unwinds h1 h
  h1 := v2_0
  let some v2 := v2_u.ret? | do
    storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ h1
    Fe.M.stop v2_u.data
  storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ h1
  -- a += 10
  let a ← loadIntAt (.u 8) Field.a ⟨0, 1, true⟩
  let v5 ← IntTy.add (.u 8) a (UInt256.ofNat 10)
  storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ v5
  return ()

/-- `<u8 as Bump>::bump` -/
def u8_bump (self : UInt256) : FeM (Unit × UInt256) := do
  let mut self := self
  -- self += 1
  let v3 ← IntTy.add (.u 8) self (UInt256.ofNat 1)
  self := v3
  return ((), self)

/-- `lane_holds::pair<u8, u8>` -/
def pair_u8_u8 : FeM Unit := do
  let mut a ← loadIntAt (.u 8) Field.a ⟨0, 1, true⟩
  -- x.bump()
  let (v1, v1_0) ← u8_bump a
  a := v1_0
  -- y.bump()
  let (v3, v3_0) ← u8_bump a
  a := v3_0
  -- x.bump()
  let (v5, v5_0) ← u8_bump a
  a := v5_0
  storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ a
  return ()

/-- `<u16 as Bump>::bump` -/
def u16_bump (self : UInt256) : FeM (Unit × UInt256) := do
  let mut self := self
  -- self += 10
  let v3 ← IntTy.add (.u 16) self (UInt256.ofNat 10)
  self := v3
  return ((), self)

/-- `lane_holds::pair<u8, u16>` -/
def pair_u8_u16 : FeM Unit := do
  let mut a ← loadIntAt (.u 8) Field.a ⟨0, 1, true⟩
  let mut c ← loadIntAt (.u 16) Field.c ⟨2, 2, true⟩
  -- x.bump()
  let (v1, v1_0) ← u8_bump a
  a := v1_0
  -- y.bump()
  let (v3, v3_0) ← u16_bump c
  c := v3_0
  -- x.bump()
  let (v5, v5_0) ← u8_bump a
  a := v5_0
  storeIntAt (.u 8) Field.a ⟨0, 1, true⟩ a
  storeIntAt (.u 16) Field.c ⟨2, 2, true⟩ c
  return ()

/-- `Pair`, selector `0xa8aa1b31` (`pair()`) -/
def recv_Pair : FeM Unit := do
  -- pair<u8, u8>()
  let _ ← pair_u8_u8
  -- pair<u8, u16>()
  let _ ← pair_u8_u16
  return ()

/-- `lane_holds::bump_effect_halt` -/
def bump_effect_halt_unwinds_1 (b : UInt256) : FeM (Unwound Empty × UInt256) := do
  let mut b := b
  -- x += 1
  let v2 ← IntTy.add (.u 8) b (UInt256.ofNat 1)
  b := v2
  -- halt()
  let v4_u ← Fe.M.unwind (halt)
  let some v4 := v4_u.ret? | do
    return (.halt v4_u.data, b)
  nomatch v4

/-- `lane_holds::forward` -/
def forward : FeM Empty := do
  let mut b ← loadIntAt (.u 8) Field.b ⟨1, 1, true⟩
  -- x.bump()
  let (v1, v1_0) ← u8_bump b
  b := v1_0
  -- bump_effect_halt()
  let (v2_u, v2_0) ← bump_effect_halt_unwinds_1 b
  b := v2_0
  let some v2 := v2_u.ret? | do
    storeIntAt (.u 8) Field.b ⟨1, 1, true⟩ b
    Fe.M.stop v2_u.data
  nomatch v2

/-- `Forward`, selector `0xd264e05e` (`forward()`) -/
def recv_Forward : FeM Empty := do
  -- forward()
  let v0 ← forward
  nomatch v0

/-- `Small::a_mut` -/
def Small_a_mut (self : Place) : FeM (UInt256 × (UInt256 → FeM Unit)) := do
  let mut h1 : UInt256 := default
  -- mut self.a
  h1 := (← loadIntAt (.u 8) (self.field Layout.Small 0) ⟨0, 1, true⟩)
  let slide : UInt256 → FeM Unit := fun g => do
    let mut h1 := h1
    h1 := g
    storeIntAt (.u 8) (self.field Layout.Small 0) ⟨0, 1, true⟩ h1
    return ()
  return (h1, slide)

/-- `Session`, selector `0x5e3568b8` (`session()`) -/
def recv_Session : FeM Unit := do
  let mut v1 : UInt256 := default
  -- let q = mut small.a_mut()
  let v0 := Field.small
  let (v1_g, v1_k) ← Small_a_mut v0
  v1 := v1_g
  -- q = 5
  v1 := (UInt256.ofNat 5)
  -- maybe_halt(true)
  let v7_u ← Fe.M.unwind (maybe_halt true)
  let some v7 := v7_u.ret? | do
    let _ ← v1_k v1
    Fe.M.stop v7_u.data
  -- q += 1
  let v10 ← IntTy.add (.u 8) v1 (UInt256.ofNat 1)
  v1 := v10
  let _ ← v1_k v1
  return ()

/-- `Small::pick` -/
def Small_pick (self : Place) (c : Bool) : FeM (UInt256 × (UInt256 → FeM Unit)) := do
  let mut h2 : UInt256 := default
  let mut h3 : UInt256 := default
  -- if c { mut self.b } else { mut self.a }
  if c then
    -- mut self.b
    h2 := (← loadIntAt (.u 8) (self.field Layout.Small 1) ⟨1, 1, true⟩)
    let slide : UInt256 → FeM Unit := fun g => do
      let mut h2 := h2
      let mut h3 := h3
      h2 := g
      storeIntAt (.u 8) (self.field Layout.Small 1) ⟨1, 1, true⟩ h2
      return ()
    return (h2, slide)
  else
    -- mut self.a
    h3 := (← loadIntAt (.u 8) (self.field Layout.Small 0) ⟨0, 1, true⟩)
    let slide : UInt256 → FeM Unit := fun g => do
      let mut h2 := h2
      let mut h3 := h3
      h3 := g
      storeIntAt (.u 8) (self.field Layout.Small 0) ⟨0, 1, true⟩ h3
      return ()
    return (h3, slide)

/-- `PickThenWrite { c }`, selector `0x0e36f0c3` (`pickThenWrite(bool)`) -/
def recv_PickThenWrite (c : Bool) : FeM Empty := do
  let mut v2 : UInt256 := default
  -- let q = mut small.pick(c)
  let v1 := Field.small
  let (v2_g, v2_k) ← Small_pick v1 c
  v2 := v2_g
  -- q = 5
  v2 := (UInt256.ofNat 5)
  -- maybe_halt(false)
  let v8_u ← Fe.M.unwind (maybe_halt false)
  let some v8 := v8_u.ret? | do
    let _ ← v2_k v2
    Fe.M.stop v8_u.data
  -- q += 1
  let v11 ← IntTy.add (.u 8) v2 (UInt256.ofNat 1)
  v2 := v11
  let _ ← v2_k v2
  -- small.a = 7
  storeIntAt (.u 8) Field.small.a ⟨0, 1, true⟩ (UInt256.ofNat 7)
  -- small.b = 8
  storeIntAt (.u 8) Field.small.b ⟨1, 1, true⟩ (UInt256.ofNat 8)
  -- halt()
  let v17 ← halt
  nomatch v17

/-- `Small::maybe_b` -/
def Small_maybe_b (self : Place) (c : Bool) : FeM (Option_mut_u8 × (Option_mut_u8 → FeM Unit)) := do
  let mut h2 : UInt256 := default
  -- if c { Option::Some(mut self.b) } else { Option::None }
  if c then
    -- Option::Some(mut self.b)
    h2 := (← loadIntAt (.u 8) (self.field Layout.Small 1) ⟨1, 1, true⟩)
    let slide : Option_mut_u8 → FeM Unit := fun g => do
      let mut h2 := h2
      h2 := g.Some_0
      storeIntAt (.u 8) (self.field Layout.Small 1) ⟨1, 1, true⟩ h2
      return ()
    return ((Option_mut_u8.Some h2), slide)
  else
    let slide : Option_mut_u8 → FeM Unit := fun g => do
      let mut h2 := h2
      return ()
    return (Option_mut_u8.None, slide)

/-- `MaybeThenWrite { c }`, selector `0xe54a17e2` (`maybeThenWrite(bool)`) -/
def recv_MaybeThenWrite (c : Bool) : FeM Empty := do
  let mut v2 : Option_mut_u8 := default
  -- if let Option::Some(mut q) = small.maybe_b(c) {
  let v1 := Field.small
  let (v2_g, v2_k) ← Small_maybe_b v1 c
  v2 := v2_g
  let join : Option_mut_u8 → FeM Empty := fun v2 => do
    let mut v2 := v2
    -- small.b = 9
    storeIntAt (.u 8) Field.small.b ⟨1, 1, true⟩ (UInt256.ofNat 9)
    -- halt()
    let v19 ← halt
    nomatch v19
  match v2 with
  | .Some .. =>
    -- q = 5
    v2 := (v2.set_Some_0 (UInt256.ofNat 5))
    -- maybe_halt(false)
    let v9_u ← Fe.M.unwind (maybe_halt false)
    let some v9 := v9_u.ret? | do
      let _ ← v2_k v2
      Fe.M.stop v9_u.data
    -- q += 1
    let v12 ← IntTy.add (.u 8) v2.Some_0 (UInt256.ofNat 1)
    v2 := (v2.set_Some_0 v12)
    let _ ← v2_k v2
    join v2
  | _ =>
    let _ ← v2_k v2
    join v2

/-- `DirectThenWrite`, selector `0xe318c006` (`directThenWrite()`) -/
def recv_DirectThenWrite : FeM Empty := do
  let mut v1 : UInt256 := default
  -- let q = mut small.a_mut()
  let v0 := Field.small
  let (v1_g, v1_k) ← Small_a_mut v0
  v1 := v1_g
  -- q = 5
  v1 := (UInt256.ofNat 5)
  -- maybe_halt(false)
  let v7_u ← Fe.M.unwind (maybe_halt false)
  let some v7 := v7_u.ret? | do
    let _ ← v1_k v1
    Fe.M.stop v7_u.data
  -- q += 1
  let v10 ← IntTy.add (.u 8) v1 (UInt256.ofNat 1)
  v1 := v10
  let _ ← v1_k v1
  -- small.a = 7
  storeIntAt (.u 8) Field.small.a ⟨0, 1, true⟩ (UInt256.ofNat 7)
  -- halt()
  let v14 ← halt
  nomatch v14

/-- `lane_holds::set_tag` -/
def set_tag (tag : Place) («to» : FixedBytes_4) : FeM Unit := do
  -- tag = to
  storeIntAt (.u 256) tag ⟨0, 4, false⟩ «to».val
  return ()

/-- `Tag`, selector `0x51f91066` (`tag()`) -/
def recv_Tag : FeM Empty := do
  let mut r1 : FixedBytes_4 := default
  -- set_tag(mut tagged.tag, Bytes4 { val: 0x01ffc9a7 })
  let v0 := Field.tagged.tag
  r1 := (FixedBytes_4.mk (UInt256.ofNat 33540519))
  let _ ← set_tag v0 r1
  -- halt()
  let v5 ← halt
  nomatch v5

/-- `lane_holds::bump_word_halt` -/
def bump_word_halt_unwinds (x : UInt256) : FeM (Unwound Empty × UInt256) := do
  let mut x := x
  -- x += 1
  let v3 ← checkedAdd x (UInt256.ofNat 1)
  x := v3
  -- halt()
  let v5_u ← Fe.M.unwind (halt)
  let some v5 := v5_u.ret? | do
    return (.halt v5_u.data, x)
  nomatch v5

/-- `TagVal`, selector `0xb2be6dc2` (`tagVal()`) -/
def recv_TagVal : FeM Empty := do
  let mut h0 : UInt256 := default
  -- bump_word_halt(mut tagged.tag.val)
  h0 := (← loadIntAt (.u 256) Field.tagged.tag ⟨0, 4, false⟩)
  let (v1_u, v1_0) ← bump_word_halt_unwinds h0
  h0 := v1_0
  let some v1 := v1_u.ret? | do
    storeIntAt (.u 256) Field.tagged.tag ⟨0, 4, false⟩ h0
    Fe.M.stop v1_u.data
  nomatch v1

/-- `<Arg as AbiSize>::HEAD_SIZE` -/
def Arg_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Arg as Decode>::decode_from<CallData>` -/
def Arg_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Arg := do
  return Arg.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Arg, Sol, CallData>` -/
def decode_from_prechecked_head_Arg_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Arg := do
  -- Self::decode_from(input, pos)
  let v3 ← Arg_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Arg>` -/
def decode_runtime_args_Evm_Sol_Arg : FeM Arg := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Arg_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Arg_Sol_CallData input base len
  return v12

/-- The entry of `recv_Arg`, for `lane_holds::LanesMsg::Arg` -/
def recv_Arg_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Arg
  let _ ← recv_Arg
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Effect as AbiSize>::HEAD_SIZE` -/
def Effect_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Effect as Decode>::decode_from<CallData>` -/
def Effect_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Effect := do
  return Effect.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Effect, Sol, CallData>` -/
def decode_from_prechecked_head_Effect_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Effect := do
  -- Self::decode_from(input, pos)
  let v3 ← Effect_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Effect>` -/
def decode_runtime_args_Evm_Sol_Effect : FeM Effect := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Effect_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Effect_Sol_CallData input base len
  return v12

/-- The entry of `recv_Effect`, for `lane_holds::LanesMsg::Effect` -/
def recv_Effect_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Effect
  let _ ← recv_Effect
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Field as AbiSize>::HEAD_SIZE` -/
def Field_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Field as Decode>::decode_from<CallData>` -/
def Field_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Field := do
  return Field.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Field, Sol, CallData>` -/
def decode_from_prechecked_head_Field_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Field := do
  -- Self::decode_from(input, pos)
  let v3 ← Field_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Field>` -/
def decode_runtime_args_Evm_Sol_Field : FeM Field := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Field_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Field_Sol_CallData input base len
  return v12

/-- The entry of `recv_Field`, for `lane_holds::LanesMsg::Field` -/
def recv_Field_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Field
  let _ ← recv_Field
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Maybe as AbiSize>::HEAD_SIZE` -/
def Maybe_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_bool
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<Maybe as Decode>::decode_from<CallData>` -/
def Maybe_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Maybe := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_bool_CallData v0 v1 v1 v2
  return Maybe.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<Maybe, Sol, CallData>` -/
def decode_from_prechecked_head_Maybe_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Maybe := do
  -- Self::decode_from(input, pos)
  let v3 ← Maybe_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Maybe>` -/
def decode_runtime_args_Evm_Sol_Maybe : FeM Maybe := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Maybe_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Maybe_Sol_CallData input base len
  return v12

/-- The entry of `recv_Maybe`, for `lane_holds::LanesMsg::Maybe` -/
def recv_Maybe_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Maybe
  let _ ← recv_Maybe message.h
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Pair as AbiSize>::HEAD_SIZE` -/
def Pair_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Pair as Decode>::decode_from<CallData>` -/
def Pair_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Pair := do
  return Pair.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Pair, Sol, CallData>` -/
def decode_from_prechecked_head_Pair_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Pair := do
  -- Self::decode_from(input, pos)
  let v3 ← Pair_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Pair>` -/
def decode_runtime_args_Evm_Sol_Pair : FeM Pair := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Pair_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Pair_Sol_CallData input base len
  return v12

/-- The entry of `recv_Pair`, for `lane_holds::LanesMsg::Pair` -/
def recv_Pair_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Pair
  let _ ← recv_Pair
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Forward as AbiSize>::HEAD_SIZE` -/
def Forward_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Forward as Decode>::decode_from<CallData>` -/
def Forward_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Forward := do
  return Forward.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Forward, Sol, CallData>` -/
def decode_from_prechecked_head_Forward_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Forward := do
  -- Self::decode_from(input, pos)
  let v3 ← Forward_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Forward>` -/
def decode_runtime_args_Evm_Sol_Forward : FeM Forward := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Forward_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Forward_Sol_CallData input base len
  return v12

/-- The entry of `recv_Forward`, for `lane_holds::LanesMsg::Forward` -/
def recv_Forward_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Forward
  let _ ← recv_Forward
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Session as AbiSize>::HEAD_SIZE` -/
def Session_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Session as Decode>::decode_from<CallData>` -/
def Session_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Session := do
  return Session.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Session, Sol, CallData>` -/
def decode_from_prechecked_head_Session_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Session := do
  -- Self::decode_from(input, pos)
  let v3 ← Session_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Session>` -/
def decode_runtime_args_Evm_Sol_Session : FeM Session := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Session_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Session_Sol_CallData input base len
  return v12

/-- The entry of `recv_Session`, for `lane_holds::LanesMsg::Session` -/
def recv_Session_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Session
  let _ ← recv_Session
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<PickThenWrite as AbiSize>::HEAD_SIZE` -/
def PickThenWrite_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_bool
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<PickThenWrite as Decode>::decode_from<CallData>` -/
def PickThenWrite_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM PickThenWrite := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_bool_CallData v0 v1 v1 v2
  return PickThenWrite.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<PickThenWrite, Sol, CallData>` -/
def decode_from_prechecked_head_PickThenWrite_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM PickThenWrite := do
  -- Self::decode_from(input, pos)
  let v3 ← PickThenWrite_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, PickThenWrite>` -/
def decode_runtime_args_Evm_Sol_PickThenWrite : FeM PickThenWrite := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← PickThenWrite_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_PickThenWrite_Sol_CallData input base len
  return v12

/-- The entry of `recv_PickThenWrite`, for `lane_holds::LanesMsg::PickThenWrite` -/
def recv_PickThenWrite_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_PickThenWrite
  let _ ← recv_PickThenWrite message.c
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<MaybeThenWrite as AbiSize>::HEAD_SIZE` -/
def MaybeThenWrite_HEAD_SIZE : FeM UInt256 := do
  let v1 ← abi_field_head_size_bool
  let v2 ← checkedAdd (UInt256.ofNat 0) v1
  return v2

/-- `<MaybeThenWrite as Decode>::decode_from<CallData>` -/
def MaybeThenWrite_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM MaybeThenWrite := do
  let v2 ← len_CallData v0
  let v4 ← decode_msg_field_from_Sol_bool_CallData v0 v1 v1 v2
  return MaybeThenWrite.mk v4

/-- `core::abi::Decode::decode_from_prechecked_head<MaybeThenWrite, Sol, CallData>` -/
def decode_from_prechecked_head_MaybeThenWrite_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM MaybeThenWrite := do
  -- Self::decode_from(input, pos)
  let v3 ← MaybeThenWrite_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, MaybeThenWrite>` -/
def decode_runtime_args_Evm_Sol_MaybeThenWrite : FeM MaybeThenWrite := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← MaybeThenWrite_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_MaybeThenWrite_Sol_CallData input base len
  return v12

/-- The entry of `recv_MaybeThenWrite`, for `lane_holds::LanesMsg::MaybeThenWrite` -/
def recv_MaybeThenWrite_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_MaybeThenWrite
  let _ ← recv_MaybeThenWrite message.c
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<DirectThenWrite as AbiSize>::HEAD_SIZE` -/
def DirectThenWrite_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<DirectThenWrite as Decode>::decode_from<CallData>` -/
def DirectThenWrite_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM DirectThenWrite := do
  return DirectThenWrite.mk

/-- `core::abi::Decode::decode_from_prechecked_head<DirectThenWrite, Sol, CallData>` -/
def decode_from_prechecked_head_DirectThenWrite_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM DirectThenWrite := do
  -- Self::decode_from(input, pos)
  let v3 ← DirectThenWrite_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, DirectThenWrite>` -/
def decode_runtime_args_Evm_Sol_DirectThenWrite : FeM DirectThenWrite := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← DirectThenWrite_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_DirectThenWrite_Sol_CallData input base len
  return v12

/-- The entry of `recv_DirectThenWrite`, for `lane_holds::LanesMsg::DirectThenWrite` -/
def recv_DirectThenWrite_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_DirectThenWrite
  let _ ← recv_DirectThenWrite
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<Tag as AbiSize>::HEAD_SIZE` -/
def Tag_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<Tag as Decode>::decode_from<CallData>` -/
def Tag_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM Tag := do
  return Tag.mk

/-- `core::abi::Decode::decode_from_prechecked_head<Tag, Sol, CallData>` -/
def decode_from_prechecked_head_Tag_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM Tag := do
  -- Self::decode_from(input, pos)
  let v3 ← Tag_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, Tag>` -/
def decode_runtime_args_Evm_Sol_Tag : FeM Tag := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← Tag_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_Tag_Sol_CallData input base len
  return v12

/-- The entry of `recv_Tag`, for `lane_holds::LanesMsg::Tag` -/
def recv_Tag_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_Tag
  let _ ← recv_Tag
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- `<TagVal as AbiSize>::HEAD_SIZE` -/
def TagVal_HEAD_SIZE : FeM UInt256 := do
  return UInt256.ofNat 0

/-- `<TagVal as Decode>::decode_from<CallData>` -/
def TagVal_decode_from_CallData (v0 : CallData) (v1 : UInt256) : FeM TagVal := do
  return TagVal.mk

/-- `core::abi::Decode::decode_from_prechecked_head<TagVal, Sol, CallData>` -/
def decode_from_prechecked_head_TagVal_Sol_CallData (input : CallData) (pos : UInt256) (input_len : UInt256) : FeM TagVal := do
  -- Self::decode_from(input, pos)
  let v3 ← TagVal_decode_from_CallData input pos
  return v3

/-- `core::contracts::decode_runtime_args<Evm, Sol, TagVal>` -/
def decode_runtime_args_Evm_Sol_TagVal : FeM TagVal := do
  let mut input : CallData := default
  -- let input = host.input()
  let input_2 ← Evm_input
  input := input_2
  -- let base = A::SELECTOR_SIZE
  let base ← Sol_SELECTOR_SIZE
  -- let len = input.len()
  let len ← len_CallData input
  -- A::validate_runtime_head(input_len: len, base, head_size: T::HEAD_SIZE)
  let v9 ← TagVal_HEAD_SIZE
  let _ ← Sol_validate_runtime_head len base v9
  -- T::decode_from_prechecked_head(input, base, len)
  let v12 ← decode_from_prechecked_head_TagVal_Sol_CallData input base len
  return v12

/-- The entry of `recv_TagVal`, for `lane_holds::LanesMsg::TagVal` -/
def recv_TagVal_entry : FeM Empty := do
  nonPayable
  let message ← decode_runtime_args_Evm_Sol_TagVal
  let _ ← recv_TagVal
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `Lanes`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    revertEmpty
  else
    let selector ← calldataSelector
    if selector = UInt256.ofNat 0x0e36f0c3 then
      recv_PickThenWrite_entry
    else if selector = UInt256.ofNat 0x4f28bf0e then
      recv_Field_entry
    else if selector = UInt256.ofNat 0x51f91066 then
      recv_Tag_entry
    else if selector = UInt256.ofNat 0x5e3568b8 then
      recv_Session_entry
    else if selector = UInt256.ofNat 0x7435e911 then
      recv_Arg_entry
    else if selector = UInt256.ofNat 0xa8aa1b31 then
      recv_Pair_entry
    else if selector = UInt256.ofNat 0xb2be6dc2 then
      recv_TagVal_entry
    else if selector = UInt256.ofNat 0xd264e05e then
      recv_Forward_entry
    else if selector = UInt256.ofNat 0xdae42fd8 then
      recv_Effect_entry
    else if selector = UInt256.ofNat 0xe13e4d4f then
      recv_Maybe_entry
    else if selector = UInt256.ofNat 0xe318c006 then
      recv_DirectThenWrite_entry
    else if selector = UInt256.ofNat 0xe54a17e2 then
      recv_MaybeThenWrite_entry
    else
      revertEmpty

/-- The deployment entry of contract `Lanes`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


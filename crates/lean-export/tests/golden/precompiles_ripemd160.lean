export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data MemBuffer MemBuffer_drop MemBuffer_new MemBuffer_span_ref MemBuffer_write_word Range_Known_0_Known_64 Range_Known_0_Known_64.mk Range_Known_0_Known_8 Range_Known_0_Known_8.mk Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_at_0 Range_at_0_64 Range_at_0_8 Range_has_0 Range_has_0_64 Range_has_0_8 Range_next_0 Range_next_0_64 Range_next_0_8 Range_start_0 Range_start_0_64 Range_start_0_8 array_at_ref_u32_5 array_has_u32_5 array_next_u32_5 array_start_u32_5 downcast_truncate_u64_u8 downcast_unchecked_u256_usize downcast_unchecked_usize_u64)
export Shared (MemBuffer_byte_at MemBuffer_len_1)

/-- The `init` of contract `Ripemd160`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_2 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let len ← Evm_calldatasize
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  return input

/-- `lib::hashes::RIPEMD_IV` -/
def RIPEMD_IV : FeM (Vector UInt256 5) := do
  let v5 := #v[(UInt256.ofNat 1732584193), (UInt256.ofNat 4023233417), (UInt256.ofNat 2562383102), (UInt256.ofNat 271733878), (UInt256.ofNat 3285377520)]
  return v5

/-- `lib::hashes::padded_blocks` -/
def padded_blocks (len : UInt256) : FeM UInt256 := do
  let v3 ← checkedAdd len (UInt256.ofNat 9)
  let v5 ← checkedAdd v3 (UInt256.ofNat 63)
  let v7 ← checkedDiv v5 (UInt256.ofNat 64)
  return v7

/-- `lib::hashes::padded_block<MemBuffer>` -/
def padded_block_MemBuffer (input : MemBuffer) (offset : UInt256) (len : UInt256) (block_index : UInt256) (blocks : UInt256) : FeM (Vector UInt256 64) := do
  let mut block : (Vector UInt256 64) := default
  let mut r1 : Range_Known_0_Known_64 := default
  let mut r2 : Range_Known_0_Known_8 := default
  block := (Vector.replicate 64 (UInt256.ofNat 0))
  let start := block_index * (UInt256.ofNat 64)
  r1 := (Range_Known_0_Known_64.mk () ())
  let v20 ← Range_start_0_64 r1
  let rec loop (block : (Vector UInt256 64)) (r1 : Range_Known_0_Known_64) (r2 : Range_Known_0_Known_8) (v5 : UInt256) : FeM (Vector UInt256 64) := do
    let mut block := block
    let mut r1 := r1
    let mut r2 := r2
    let v23 ← Range_has_0_64 r1 v5
    if v23 then
      let v25 ← Range_at_0_64 r1 v5
      let i_26 := v25
      let position := start + i_26
      let join : (Vector UInt256 64) → Range_Known_0_Known_64 → Range_Known_0_Known_8 → FeM (Vector UInt256 64) := fun block r1 r2 => do
        let mut block := block
        let mut r1 := r1
        let mut r2 := r2
        let v45 ← Range_next_0_64 r1 v5
        loop block r1 r2 v45
      if decide (position.toNat < len.toNat) then
        let v33 ← MemBuffer_byte_at input (offset + position)
        checkIndex i_26 64
        block := (block.set! i_26.toNat v33)
        join block r1 r2
      else
        let join_1 : (Vector UInt256 64) → Range_Known_0_Known_64 → Range_Known_0_Known_8 → Unit → FeM (Vector UInt256 64) := fun block r1 r2 v6 => do
          let mut block := block
          let mut r1 := r1
          let mut r2 := r2
          join block r1 r2
        if decide (position = len) then
          checkIndex i_26 64
          block := (block.set! i_26.toNat (UInt256.ofNat 128))
          join_1 block r1 r2 ()
        else
          join_1 block r1 r2 ()
    else
      let v51 := decide (block_index = (blocks - (UInt256.ofNat 1)))
      let join_2 : (Vector UInt256 64) → Range_Known_0_Known_64 → Range_Known_0_Known_8 → FeM (Vector UInt256 64) := fun block r1 r2 => do
        let mut block := block
        let mut r1 := r1
        let mut r2 := r2
        return block
      if v51 then
        let bits ← downcast_unchecked_usize_u64 (len * (UInt256.ofNat 8))
        r2 := (Range_Known_0_Known_8.mk () ())
        let v62 ← Range_start_0_8 r2
        let rec loop_1 (block : (Vector UInt256 64)) (r1 : Range_Known_0_Known_64) (r2 : Range_Known_0_Known_8) (v7 : UInt256) : FeM (Vector UInt256 64) := do
          let mut block := block
          let mut r1 := r1
          let mut r2 := r2
          let v65 ← Range_has_0_8 r2 v7
          if v65 then
            let v67 ← Range_at_0_8 r2 v7
            let i_68 := v67
            let v71 ← downcast_unchecked_usize_u64 (i_68 * (UInt256.ofNat 8))
            let v73 ← downcast_truncate_u64_u8 (bits >>> v71)
            let v75 := (UInt256.ofNat 56) + i_68
            checkIndex v75 64
            block := (block.set! v75.toNat v73)
            let v78 ← Range_next_0_8 r2 v7
            loop_1 block r1 r2 v78
          else
            join_2 block r1 r2
        partial_fixpoint
        loop_1 block r1 r2 v62
      else
        join_2 block r1 r2
  partial_fixpoint
  loop block r1 r2 v20

/-- `lib::hashes::f0` -/
def f0 (x : UInt256) (y : UInt256) (z : UInt256) : FeM UInt256 := do
  return (x ^^^ y) ^^^ z

/-- `lib::hashes::rotate_left` -/
def rotate_left (value : UInt256) («by» : UInt256) : FeM UInt256 := do
  let v8 ← IntTy.sub (.u 32) (UInt256.ofNat 32) «by»
  return (IntTy.shl (.u 32) value «by») ||| (value >>> v8)

/-- `lib::hashes::f1` -/
def f1 (x : UInt256) (y : UInt256) (z : UInt256) : FeM UInt256 := do
  return (x &&& y) ||| ((IntTy.not (.u 32) x) &&& z)

/-- `lib::hashes::f2` -/
def f2 (x : UInt256) (y : UInt256) (z : UInt256) : FeM UInt256 := do
  return (x ||| (IntTy.not (.u 32) y)) ^^^ z

/-- `lib::hashes::f3` -/
def f3 (x : UInt256) (y : UInt256) (z : UInt256) : FeM UInt256 := do
  return (x &&& z) ||| (y &&& (IntTy.not (.u 32) z))

/-- `lib::hashes::f4` -/
def f4 (x : UInt256) (y : UInt256) (z : UInt256) : FeM UInt256 := do
  return x ^^^ (y ||| (IntTy.not (.u 32) z))

def ripemd_compress_k46 (lr77 : UInt256) (v1296 : UInt256) (rr77 : UInt256) (r78 : UInt256) (v2224 : UInt256) (t : UInt256) (v2244 : UInt256) (v2245 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  checkIndex v2245 5
  state := (state.set! v2245.toNat v2244)
  let v2247 := (UInt256.ofNat 4)
  checkIndex v2247 5
  let v2250 := IntTy.addW (.u 32) (IntTy.addW (.u 32) state[v2247.toNat]! lr77) (IntTy.addW (.u 32) v2224 rr77)
  let v2251 := (UInt256.ofNat 3)
  checkIndex v2251 5
  state := (state.set! v2251.toNat v2250)
  let v2253 := (UInt256.ofNat 0)
  checkIndex v2253 5
  let v2255 := IntTy.addW (.u 32) state[v2253.toNat]! (IntTy.addW (.u 32) v1296 lr77)
  let v2257 := (UInt256.ofNat 4)
  checkIndex v2257 5
  state := (state.set! v2257.toNat (IntTy.addW (.u 32) v2255 r78))
  let v2259 := (UInt256.ofNat 0)
  checkIndex v2259 5
  state := (state.set! v2259.toNat t)
  return ((), state)

def ripemd_compress_k45 (w9 : UInt256) (w11 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r75 : UInt256) (rr75 : UInt256) (r76 : UInt256) (rr76 : UInt256) (v2204 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let r77 := IntTy.addW (.u 32) v2204 rr75
  let rr77 ← rotate_left r75 (UInt256.ofNat 10)
  let v2210 ← f0 r77 r76 rr77
  let v2212 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr75 v2210) w9
  let v2214 ← rotate_left v2212 (UInt256.ofNat 11)
  let r78 := IntTy.addW (.u 32) v2214 rr76
  let rr78 ← rotate_left r76 (UInt256.ofNat 10)
  let v2220 ← f0 r78 r77 rr78
  let v2222 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr76 v2220) w11
  let v2224 ← rotate_left v2222 (UInt256.ofNat 11)
  let rr79 ← rotate_left r77 (UInt256.ofNat 10)
  let v2230 := (UInt256.ofNat 1)
  checkIndex v2230 5
  let t := IntTy.addW (.u 32) (IntTy.addW (.u 32) state[v2230.toNat]! l78) rr79
  let v2235 := (UInt256.ofNat 2)
  checkIndex v2235 5
  let v2238 := IntTy.addW (.u 32) (IntTy.addW (.u 32) state[v2235.toNat]! lr79) rr78
  let v2239 := (UInt256.ofNat 1)
  checkIndex v2239 5
  state := (state.set! v2239.toNat v2238)
  let v2241 := (UInt256.ofNat 3)
  checkIndex v2241 5
  let v2244 := IntTy.addW (.u 32) (IntTy.addW (.u 32) state[v2241.toNat]! lr78) rr77
  let v2245 := (UInt256.ofNat 2)
  ripemd_compress_k46 lr77 v1296 rr77 r78 v2224 t v2244 v2245 state

def ripemd_compress_k44 (w0 : UInt256) (w2 : UInt256) (w3 : UInt256) (w9 : UInt256) (w11 : UInt256) (w13 : UInt256) (w14 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r70 : UInt256) (rr70 : UInt256) (r71 : UInt256) (rr71 : UInt256) (r72 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let rr72 ← rotate_left r70 (UInt256.ofNat 10)
  let v2160 ← f0 r72 r71 rr72
  let v2162 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr70 v2160) w2
  let v2164 ← rotate_left v2162 (UInt256.ofNat 13)
  let r73 := IntTy.addW (.u 32) v2164 rr71
  let rr73 ← rotate_left r71 (UInt256.ofNat 10)
  let v2170 ← f0 r73 r72 rr73
  let v2172 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr71 v2170) w13
  let v2174 ← rotate_left v2172 (UInt256.ofNat 6)
  let r74 := IntTy.addW (.u 32) v2174 rr72
  let rr74 ← rotate_left r72 (UInt256.ofNat 10)
  let v2180 ← f0 r74 r73 rr74
  let v2182 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr72 v2180) w14
  let v2184 ← rotate_left v2182 (UInt256.ofNat 5)
  let r75 := IntTy.addW (.u 32) v2184 rr73
  let rr75 ← rotate_left r73 (UInt256.ofNat 10)
  let v2190 ← f0 r75 r74 rr75
  let v2192 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr73 v2190) w0
  let v2194 ← rotate_left v2192 (UInt256.ofNat 15)
  let r76 := IntTy.addW (.u 32) v2194 rr74
  let rr76 ← rotate_left r74 (UInt256.ofNat 10)
  let v2200 ← f0 r76 r75 rr76
  let v2202 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr74 v2200) w3
  let v2204 ← rotate_left v2202 (UInt256.ofNat 13)
  ripemd_compress_k45 w9 w11 lr77 l78 lr78 v1296 lr79 r75 rr75 r76 rr76 v2204 state

def ripemd_compress_k43 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w11 : UInt256) (w13 : UInt256) (w14 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (rr65 : UInt256) (r66 : UInt256) (rr66 : UInt256) (r67 : UInt256) (rr67 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v2110 ← f0 r67 r66 rr67
  let v2112 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr65 v2110) w1
  let v2114 ← rotate_left v2112 (UInt256.ofNat 12)
  let r68 := IntTy.addW (.u 32) v2114 rr66
  let rr68 ← rotate_left r66 (UInt256.ofNat 10)
  let v2120 ← f0 r68 r67 rr68
  let v2122 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr66 v2120) w5
  let v2124 ← rotate_left v2122 (UInt256.ofNat 5)
  let r69 := IntTy.addW (.u 32) v2124 rr67
  let rr69 ← rotate_left r67 (UInt256.ofNat 10)
  let v2130 ← f0 r69 r68 rr69
  let v2132 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr67 v2130) w8
  let v2134 ← rotate_left v2132 (UInt256.ofNat 14)
  let r70 := IntTy.addW (.u 32) v2134 rr68
  let rr70 ← rotate_left r68 (UInt256.ofNat 10)
  let v2140 ← f0 r70 r69 rr70
  let v2142 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr68 v2140) w7
  let v2144 ← rotate_left v2142 (UInt256.ofNat 6)
  let r71 := IntTy.addW (.u 32) v2144 rr69
  let rr71 ← rotate_left r69 (UInt256.ofNat 10)
  let v2150 ← f0 r71 r70 rr71
  let v2152 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr69 v2150) w6
  let v2154 ← rotate_left v2152 (UInt256.ofNat 8)
  let r72 := IntTy.addW (.u 32) v2154 rr70
  ripemd_compress_k44 w0 w2 w3 w9 w11 w13 w14 lr77 l78 lr78 v1296 lr79 r70 rr70 r71 rr71 r72 state

def ripemd_compress_k42 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r61 : UInt256) (rr61 : UInt256) (r62 : UInt256) (rr62 : UInt256) (v2060 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v2062 := IntTy.addW (.u 32) v2060 (UInt256.ofNat 2053994217)
  let v2064 ← rotate_left v2062 (UInt256.ofNat 8)
  let r63 := IntTy.addW (.u 32) v2064 rr61
  let rr63 ← rotate_left r61 (UInt256.ofNat 10)
  let v2070 ← f0 r63 r62 rr63
  let v2072 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr61 v2070) w12
  let v2074 ← rotate_left v2072 (UInt256.ofNat 8)
  let r64 := IntTy.addW (.u 32) v2074 rr62
  let rr64 ← rotate_left r62 (UInt256.ofNat 10)
  let v2080 ← f0 r64 r63 rr64
  let v2082 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr62 v2080) w15
  let v2084 ← rotate_left v2082 (UInt256.ofNat 5)
  let r65 := IntTy.addW (.u 32) v2084 rr63
  let rr65 ← rotate_left r63 (UInt256.ofNat 10)
  let v2090 ← f0 r65 r64 rr65
  let v2092 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr63 v2090) w10
  let v2094 ← rotate_left v2092 (UInt256.ofNat 12)
  let r66 := IntTy.addW (.u 32) v2094 rr64
  let rr66 ← rotate_left r64 (UInt256.ofNat 10)
  let v2100 ← f0 r66 r65 rr66
  let v2102 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr64 v2100) w4
  let v2104 ← rotate_left v2102 (UInt256.ofNat 9)
  let r67 := IntTy.addW (.u 32) v2104 rr65
  let rr67 ← rotate_left r65 (UInt256.ofNat 10)
  ripemd_compress_k43 w0 w1 w2 w3 w5 w6 w7 w8 w9 w11 w13 w14 lr77 l78 lr78 v1296 lr79 rr65 r66 rr66 r67 rr67 state

def ripemd_compress_k41 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r57 : UInt256) (rr57 : UInt256) (r58 : UInt256) (rr58 : UInt256) (v2012 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v2014 := IntTy.addW (.u 32) v2012 (UInt256.ofNat 2053994217)
  let v2016 ← rotate_left v2014 (UInt256.ofNat 9)
  let r59 := IntTy.addW (.u 32) v2016 rr57
  let rr59 ← rotate_left r57 (UInt256.ofNat 10)
  let v2022 ← f1 r59 r58 rr59
  let v2024 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr57 v2022) w9
  let v2026 := IntTy.addW (.u 32) v2024 (UInt256.ofNat 2053994217)
  let v2028 ← rotate_left v2026 (UInt256.ofNat 12)
  let r60 := IntTy.addW (.u 32) v2028 rr58
  let rr60 ← rotate_left r58 (UInt256.ofNat 10)
  let v2034 ← f1 r60 r59 rr60
  let v2036 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr58 v2034) w7
  let v2038 := IntTy.addW (.u 32) v2036 (UInt256.ofNat 2053994217)
  let v2040 ← rotate_left v2038 (UInt256.ofNat 5)
  let r61 := IntTy.addW (.u 32) v2040 rr59
  let rr61 ← rotate_left r59 (UInt256.ofNat 10)
  let v2046 ← f1 r61 r60 rr61
  let v2048 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr59 v2046) w10
  let v2050 := IntTy.addW (.u 32) v2048 (UInt256.ofNat 2053994217)
  let v2052 ← rotate_left v2050 (UInt256.ofNat 15)
  let r62 := IntTy.addW (.u 32) v2052 rr60
  let rr62 ← rotate_left r60 (UInt256.ofNat 10)
  let v2058 ← f1 r62 r61 rr62
  let v2060 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr60 v2058) w14
  ripemd_compress_k42 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r61 rr61 r62 rr62 v2060 state

def ripemd_compress_k40 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r53 : UInt256) (rr53 : UInt256) (r54 : UInt256) (rr54 : UInt256) (v1964 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1966 := IntTy.addW (.u 32) v1964 (UInt256.ofNat 2053994217)
  let v1968 ← rotate_left v1966 (UInt256.ofNat 14)
  let r55 := IntTy.addW (.u 32) v1968 rr53
  let rr55 ← rotate_left r53 (UInt256.ofNat 10)
  let v1974 ← f1 r55 r54 rr55
  let v1976 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr53 v1974) w5
  let v1978 := IntTy.addW (.u 32) v1976 (UInt256.ofNat 2053994217)
  let v1980 ← rotate_left v1978 (UInt256.ofNat 6)
  let r56 := IntTy.addW (.u 32) v1980 rr54
  let rr56 ← rotate_left r54 (UInt256.ofNat 10)
  let v1986 ← f1 r56 r55 rr56
  let v1988 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr54 v1986) w12
  let v1990 := IntTy.addW (.u 32) v1988 (UInt256.ofNat 2053994217)
  let v1992 ← rotate_left v1990 (UInt256.ofNat 9)
  let r57 := IntTy.addW (.u 32) v1992 rr55
  let rr57 ← rotate_left r55 (UInt256.ofNat 10)
  let v1998 ← f1 r57 r56 rr57
  let v2000 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr55 v1998) w2
  let v2002 := IntTy.addW (.u 32) v2000 (UInt256.ofNat 2053994217)
  let v2004 ← rotate_left v2002 (UInt256.ofNat 12)
  let r58 := IntTy.addW (.u 32) v2004 rr56
  let rr58 ← rotate_left r56 (UInt256.ofNat 10)
  let v2010 ← f1 r58 r57 rr58
  let v2012 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr56 v2010) w13
  ripemd_compress_k41 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r57 rr57 r58 rr58 v2012 state

def ripemd_compress_k39 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r49 : UInt256) (rr49 : UInt256) (r50 : UInt256) (rr50 : UInt256) (v1916 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1918 := IntTy.addW (.u 32) v1916 (UInt256.ofNat 2053994217)
  let v1920 ← rotate_left v1918 (UInt256.ofNat 11)
  let r51 := IntTy.addW (.u 32) v1920 rr49
  let rr51 ← rotate_left r49 (UInt256.ofNat 10)
  let v1926 ← f1 r51 r50 rr51
  let v1928 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr49 v1926) w3
  let v1930 := IntTy.addW (.u 32) v1928 (UInt256.ofNat 2053994217)
  let v1932 ← rotate_left v1930 (UInt256.ofNat 14)
  let r52 := IntTy.addW (.u 32) v1932 rr50
  let rr52 ← rotate_left r50 (UInt256.ofNat 10)
  let v1938 ← f1 r52 r51 rr52
  let v1940 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr50 v1938) w11
  let v1942 := IntTy.addW (.u 32) v1940 (UInt256.ofNat 2053994217)
  let v1944 ← rotate_left v1942 (UInt256.ofNat 14)
  let r53 := IntTy.addW (.u 32) v1944 rr51
  let rr53 ← rotate_left r51 (UInt256.ofNat 10)
  let v1950 ← f1 r53 r52 rr53
  let v1952 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr51 v1950) w15
  let v1954 := IntTy.addW (.u 32) v1952 (UInt256.ofNat 2053994217)
  let v1956 ← rotate_left v1954 (UInt256.ofNat 6)
  let r54 := IntTy.addW (.u 32) v1956 rr52
  let rr54 ← rotate_left r52 (UInt256.ofNat 10)
  let v1962 ← f1 r54 r53 rr54
  let v1964 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr52 v1962) w0
  ripemd_compress_k40 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r53 rr53 r54 rr54 v1964 state

def ripemd_compress_k38 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r45 : UInt256) (rr45 : UInt256) (r46 : UInt256) (rr46 : UInt256) (v1868 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1870 := IntTy.addW (.u 32) v1868 (UInt256.ofNat 1836072691)
  let v1872 ← rotate_left v1870 (UInt256.ofNat 5)
  let r47 := IntTy.addW (.u 32) v1872 rr45
  let rr47 ← rotate_left r45 (UInt256.ofNat 10)
  let v1878 ← f1 r47 r46 rr47
  let v1880 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr45 v1878) w8
  let v1882 := IntTy.addW (.u 32) v1880 (UInt256.ofNat 2053994217)
  let v1884 ← rotate_left v1882 (UInt256.ofNat 15)
  let r48 := IntTy.addW (.u 32) v1884 rr46
  let rr48 ← rotate_left r46 (UInt256.ofNat 10)
  let v1890 ← f1 r48 r47 rr48
  let v1892 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr46 v1890) w6
  let v1894 := IntTy.addW (.u 32) v1892 (UInt256.ofNat 2053994217)
  let v1896 ← rotate_left v1894 (UInt256.ofNat 5)
  let r49 := IntTy.addW (.u 32) v1896 rr47
  let rr49 ← rotate_left r47 (UInt256.ofNat 10)
  let v1902 ← f1 r49 r48 rr49
  let v1904 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr47 v1902) w4
  let v1906 := IntTy.addW (.u 32) v1904 (UInt256.ofNat 2053994217)
  let v1908 ← rotate_left v1906 (UInt256.ofNat 8)
  let r50 := IntTy.addW (.u 32) v1908 rr48
  let rr50 ← rotate_left r48 (UInt256.ofNat 10)
  let v1914 ← f1 r50 r49 rr50
  let v1916 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr48 v1914) w1
  ripemd_compress_k39 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r49 rr49 r50 rr50 v1916 state

def ripemd_compress_k37 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r41 : UInt256) (rr41 : UInt256) (r42 : UInt256) (rr42 : UInt256) (v1820 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1822 := IntTy.addW (.u 32) v1820 (UInt256.ofNat 1836072691)
  let v1824 ← rotate_left v1822 (UInt256.ofNat 14)
  let r43 := IntTy.addW (.u 32) v1824 rr41
  let rr43 ← rotate_left r41 (UInt256.ofNat 10)
  let v1830 ← f2 r43 r42 rr43
  let v1832 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr41 v1830) w10
  let v1834 := IntTy.addW (.u 32) v1832 (UInt256.ofNat 1836072691)
  let v1836 ← rotate_left v1834 (UInt256.ofNat 13)
  let r44 := IntTy.addW (.u 32) v1836 rr42
  let rr44 ← rotate_left r42 (UInt256.ofNat 10)
  let v1842 ← f2 r44 r43 rr44
  let v1844 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr42 v1842) w0
  let v1846 := IntTy.addW (.u 32) v1844 (UInt256.ofNat 1836072691)
  let v1848 ← rotate_left v1846 (UInt256.ofNat 13)
  let r45 := IntTy.addW (.u 32) v1848 rr43
  let rr45 ← rotate_left r43 (UInt256.ofNat 10)
  let v1854 ← f2 r45 r44 rr45
  let v1856 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr43 v1854) w4
  let v1858 := IntTy.addW (.u 32) v1856 (UInt256.ofNat 1836072691)
  let v1860 ← rotate_left v1858 (UInt256.ofNat 7)
  let r46 := IntTy.addW (.u 32) v1860 rr44
  let rr46 ← rotate_left r44 (UInt256.ofNat 10)
  let v1866 ← f2 r46 r45 rr46
  let v1868 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr44 v1866) w13
  ripemd_compress_k38 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r45 rr45 r46 rr46 v1868 state

def ripemd_compress_k36 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r37 : UInt256) (rr37 : UInt256) (r38 : UInt256) (rr38 : UInt256) (v1772 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1774 := IntTy.addW (.u 32) v1772 (UInt256.ofNat 1836072691)
  let v1776 ← rotate_left v1774 (UInt256.ofNat 14)
  let r39 := IntTy.addW (.u 32) v1776 rr37
  let rr39 ← rotate_left r37 (UInt256.ofNat 10)
  let v1782 ← f2 r39 r38 rr39
  let v1784 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr37 v1782) w11
  let v1786 := IntTy.addW (.u 32) v1784 (UInt256.ofNat 1836072691)
  let v1788 ← rotate_left v1786 (UInt256.ofNat 12)
  let r40 := IntTy.addW (.u 32) v1788 rr38
  let rr40 ← rotate_left r38 (UInt256.ofNat 10)
  let v1794 ← f2 r40 r39 rr40
  let v1796 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr38 v1794) w8
  let v1798 := IntTy.addW (.u 32) v1796 (UInt256.ofNat 1836072691)
  let v1800 ← rotate_left v1798 (UInt256.ofNat 13)
  let r41 := IntTy.addW (.u 32) v1800 rr39
  let rr41 ← rotate_left r39 (UInt256.ofNat 10)
  let v1806 ← f2 r41 r40 rr41
  let v1808 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr39 v1806) w12
  let v1810 := IntTy.addW (.u 32) v1808 (UInt256.ofNat 1836072691)
  let v1812 ← rotate_left v1810 (UInt256.ofNat 5)
  let r42 := IntTy.addW (.u 32) v1812 rr40
  let rr42 ← rotate_left r40 (UInt256.ofNat 10)
  let v1818 ← f2 r42 r41 rr42
  let v1820 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr40 v1818) w2
  ripemd_compress_k37 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r41 rr41 r42 rr42 v1820 state

def ripemd_compress_k35 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r33 : UInt256) (rr33 : UInt256) (r34 : UInt256) (rr34 : UInt256) (v1724 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1726 := IntTy.addW (.u 32) v1724 (UInt256.ofNat 1836072691)
  let v1728 ← rotate_left v1726 (UInt256.ofNat 11)
  let r35 := IntTy.addW (.u 32) v1728 rr33
  let rr35 ← rotate_left r33 (UInt256.ofNat 10)
  let v1734 ← f2 r35 r34 rr35
  let v1736 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr33 v1734) w7
  let v1738 := IntTy.addW (.u 32) v1736 (UInt256.ofNat 1836072691)
  let v1740 ← rotate_left v1738 (UInt256.ofNat 8)
  let r36 := IntTy.addW (.u 32) v1740 rr34
  let rr36 ← rotate_left r34 (UInt256.ofNat 10)
  let v1746 ← f2 r36 r35 rr36
  let v1748 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr34 v1746) w14
  let v1750 := IntTy.addW (.u 32) v1748 (UInt256.ofNat 1836072691)
  let v1752 ← rotate_left v1750 (UInt256.ofNat 6)
  let r37 := IntTy.addW (.u 32) v1752 rr35
  let rr37 ← rotate_left r35 (UInt256.ofNat 10)
  let v1758 ← f2 r37 r36 rr37
  let v1760 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr35 v1758) w6
  let v1762 := IntTy.addW (.u 32) v1760 (UInt256.ofNat 1836072691)
  let v1764 ← rotate_left v1762 (UInt256.ofNat 6)
  let r38 := IntTy.addW (.u 32) v1764 rr36
  let rr38 ← rotate_left r36 (UInt256.ofNat 10)
  let v1770 ← f2 r38 r37 rr38
  let v1772 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr36 v1770) w9
  ripemd_compress_k36 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r37 rr37 r38 rr38 v1772 state

def ripemd_compress_k34 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r29 : UInt256) (rr29 : UInt256) (r30 : UInt256) (rr30 : UInt256) (v1676 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1678 := IntTy.addW (.u 32) v1676 (UInt256.ofNat 1548603684)
  let v1680 ← rotate_left v1678 (UInt256.ofNat 11)
  let r31 := IntTy.addW (.u 32) v1680 rr29
  let rr31 ← rotate_left r29 (UInt256.ofNat 10)
  let v1686 ← f2 r31 r30 rr31
  let v1688 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr29 v1686) w15
  let v1690 := IntTy.addW (.u 32) v1688 (UInt256.ofNat 1836072691)
  let v1692 ← rotate_left v1690 (UInt256.ofNat 9)
  let r32 := IntTy.addW (.u 32) v1692 rr30
  let rr32 ← rotate_left r30 (UInt256.ofNat 10)
  let v1698 ← f2 r32 r31 rr32
  let v1700 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr30 v1698) w5
  let v1702 := IntTy.addW (.u 32) v1700 (UInt256.ofNat 1836072691)
  let v1704 ← rotate_left v1702 (UInt256.ofNat 7)
  let r33 := IntTy.addW (.u 32) v1704 rr31
  let rr33 ← rotate_left r31 (UInt256.ofNat 10)
  let v1710 ← f2 r33 r32 rr33
  let v1712 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr31 v1710) w1
  let v1714 := IntTy.addW (.u 32) v1712 (UInt256.ofNat 1836072691)
  let v1716 ← rotate_left v1714 (UInt256.ofNat 15)
  let r34 := IntTy.addW (.u 32) v1716 rr32
  let rr34 ← rotate_left r32 (UInt256.ofNat 10)
  let v1722 ← f2 r34 r33 rr34
  let v1724 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr32 v1722) w3
  ripemd_compress_k35 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r33 rr33 r34 rr34 v1724 state

def ripemd_compress_k33 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r25 : UInt256) (rr25 : UInt256) (r26 : UInt256) (rr26 : UInt256) (v1628 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1630 := IntTy.addW (.u 32) v1628 (UInt256.ofNat 1548603684)
  let v1632 ← rotate_left v1630 (UInt256.ofNat 7)
  let r27 := IntTy.addW (.u 32) v1632 rr25
  let rr27 ← rotate_left r25 (UInt256.ofNat 10)
  let v1638 ← f3 r27 r26 rr27
  let v1640 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr25 v1638) w4
  let v1642 := IntTy.addW (.u 32) v1640 (UInt256.ofNat 1548603684)
  let v1644 ← rotate_left v1642 (UInt256.ofNat 6)
  let r28 := IntTy.addW (.u 32) v1644 rr26
  let rr28 ← rotate_left r26 (UInt256.ofNat 10)
  let v1650 ← f3 r28 r27 rr28
  let v1652 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr26 v1650) w9
  let v1654 := IntTy.addW (.u 32) v1652 (UInt256.ofNat 1548603684)
  let v1656 ← rotate_left v1654 (UInt256.ofNat 15)
  let r29 := IntTy.addW (.u 32) v1656 rr27
  let rr29 ← rotate_left r27 (UInt256.ofNat 10)
  let v1662 ← f3 r29 r28 rr29
  let v1664 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr27 v1662) w1
  let v1666 := IntTy.addW (.u 32) v1664 (UInt256.ofNat 1548603684)
  let v1668 ← rotate_left v1666 (UInt256.ofNat 13)
  let r30 := IntTy.addW (.u 32) v1668 rr28
  let rr30 ← rotate_left r28 (UInt256.ofNat 10)
  let v1674 ← f3 r30 r29 rr30
  let v1676 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr28 v1674) w2
  ripemd_compress_k34 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r29 rr29 r30 rr30 v1676 state

def ripemd_compress_k32 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r21 : UInt256) (rr21 : UInt256) (r22 : UInt256) (rr22 : UInt256) (v1580 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1582 := IntTy.addW (.u 32) v1580 (UInt256.ofNat 1548603684)
  let v1584 ← rotate_left v1582 (UInt256.ofNat 11)
  let r23 := IntTy.addW (.u 32) v1584 rr21
  let rr23 ← rotate_left r21 (UInt256.ofNat 10)
  let v1590 ← f3 r23 r22 rr23
  let v1592 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr21 v1590) w14
  let v1594 := IntTy.addW (.u 32) v1592 (UInt256.ofNat 1548603684)
  let v1596 ← rotate_left v1594 (UInt256.ofNat 7)
  let r24 := IntTy.addW (.u 32) v1596 rr22
  let rr24 ← rotate_left r22 (UInt256.ofNat 10)
  let v1602 ← f3 r24 r23 rr24
  let v1604 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr22 v1602) w15
  let v1606 := IntTy.addW (.u 32) v1604 (UInt256.ofNat 1548603684)
  let v1608 ← rotate_left v1606 (UInt256.ofNat 7)
  let r25 := IntTy.addW (.u 32) v1608 rr23
  let rr25 ← rotate_left r23 (UInt256.ofNat 10)
  let v1614 ← f3 r25 r24 rr25
  let v1616 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr23 v1614) w8
  let v1618 := IntTy.addW (.u 32) v1616 (UInt256.ofNat 1548603684)
  let v1620 ← rotate_left v1618 (UInt256.ofNat 12)
  let r26 := IntTy.addW (.u 32) v1620 rr24
  let rr26 ← rotate_left r24 (UInt256.ofNat 10)
  let v1626 ← f3 r26 r25 rr26
  let v1628 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr24 v1626) w12
  ripemd_compress_k33 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r25 rr25 r26 rr26 v1628 state

def ripemd_compress_k31 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r17 : UInt256) (rr17 : UInt256) (r18 : UInt256) (rr18 : UInt256) (v1532 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1534 := IntTy.addW (.u 32) v1532 (UInt256.ofNat 1548603684)
  let v1536 ← rotate_left v1534 (UInt256.ofNat 7)
  let r19 := IntTy.addW (.u 32) v1536 rr17
  let rr19 ← rotate_left r17 (UInt256.ofNat 10)
  let v1542 ← f3 r19 r18 rr19
  let v1544 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr17 v1542) w0
  let v1546 := IntTy.addW (.u 32) v1544 (UInt256.ofNat 1548603684)
  let v1548 ← rotate_left v1546 (UInt256.ofNat 12)
  let r20 := IntTy.addW (.u 32) v1548 rr18
  let rr20 ← rotate_left r18 (UInt256.ofNat 10)
  let v1554 ← f3 r20 r19 rr20
  let v1556 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr18 v1554) w13
  let v1558 := IntTy.addW (.u 32) v1556 (UInt256.ofNat 1548603684)
  let v1560 ← rotate_left v1558 (UInt256.ofNat 8)
  let r21 := IntTy.addW (.u 32) v1560 rr19
  let rr21 ← rotate_left r19 (UInt256.ofNat 10)
  let v1566 ← f3 r21 r20 rr21
  let v1568 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr19 v1566) w5
  let v1570 := IntTy.addW (.u 32) v1568 (UInt256.ofNat 1548603684)
  let v1572 ← rotate_left v1570 (UInt256.ofNat 9)
  let r22 := IntTy.addW (.u 32) v1572 rr20
  let rr22 ← rotate_left r20 (UInt256.ofNat 10)
  let v1578 ← f3 r22 r21 rr22
  let v1580 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr20 v1578) w10
  ripemd_compress_k32 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r21 rr21 r22 rr22 v1580 state

def ripemd_compress_k30 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r13 : UInt256) (rr13 : UInt256) (r14 : UInt256) (rr14 : UInt256) (v1484 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1486 := IntTy.addW (.u 32) v1484 (UInt256.ofNat 1352829926)
  let v1488 ← rotate_left v1486 (UInt256.ofNat 6)
  let r15 := IntTy.addW (.u 32) v1488 rr13
  let rr15 ← rotate_left r13 (UInt256.ofNat 10)
  let v1494 ← f3 r15 r14 rr15
  let v1496 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr13 v1494) w6
  let v1498 := IntTy.addW (.u 32) v1496 (UInt256.ofNat 1548603684)
  let v1500 ← rotate_left v1498 (UInt256.ofNat 9)
  let r16 := IntTy.addW (.u 32) v1500 rr14
  let rr16 ← rotate_left r14 (UInt256.ofNat 10)
  let v1506 ← f3 r16 r15 rr16
  let v1508 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr14 v1506) w11
  let v1510 := IntTy.addW (.u 32) v1508 (UInt256.ofNat 1548603684)
  let v1512 ← rotate_left v1510 (UInt256.ofNat 13)
  let r17 := IntTy.addW (.u 32) v1512 rr15
  let rr17 ← rotate_left r15 (UInt256.ofNat 10)
  let v1518 ← f3 r17 r16 rr17
  let v1520 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr15 v1518) w3
  let v1522 := IntTy.addW (.u 32) v1520 (UInt256.ofNat 1548603684)
  let v1524 ← rotate_left v1522 (UInt256.ofNat 15)
  let r18 := IntTy.addW (.u 32) v1524 rr16
  let rr18 ← rotate_left r16 (UInt256.ofNat 10)
  let v1530 ← f3 r18 r17 rr18
  let v1532 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr16 v1530) w7
  ripemd_compress_k31 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r17 rr17 r18 rr18 v1532 state

def ripemd_compress_k29 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r9 : UInt256) (rr9 : UInt256) (r10 : UInt256) (rr10 : UInt256) (v1436 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1438 := IntTy.addW (.u 32) v1436 (UInt256.ofNat 1352829926)
  let v1440 ← rotate_left v1438 (UInt256.ofNat 11)
  let r11 := IntTy.addW (.u 32) v1440 rr9
  let rr11 ← rotate_left r9 (UInt256.ofNat 10)
  let v1446 ← f4 r11 r10 rr11
  let v1448 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr9 v1446) w1
  let v1450 := IntTy.addW (.u 32) v1448 (UInt256.ofNat 1352829926)
  let v1452 ← rotate_left v1450 (UInt256.ofNat 14)
  let r12 := IntTy.addW (.u 32) v1452 rr10
  let rr12 ← rotate_left r10 (UInt256.ofNat 10)
  let v1458 ← f4 r12 r11 rr12
  let v1460 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr10 v1458) w10
  let v1462 := IntTy.addW (.u 32) v1460 (UInt256.ofNat 1352829926)
  let v1464 ← rotate_left v1462 (UInt256.ofNat 14)
  let r13 := IntTy.addW (.u 32) v1464 rr11
  let rr13 ← rotate_left r11 (UInt256.ofNat 10)
  let v1470 ← f4 r13 r12 rr13
  let v1472 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr11 v1470) w3
  let v1474 := IntTy.addW (.u 32) v1472 (UInt256.ofNat 1352829926)
  let v1476 ← rotate_left v1474 (UInt256.ofNat 12)
  let r14 := IntTy.addW (.u 32) v1476 rr12
  let rr14 ← rotate_left r12 (UInt256.ofNat 10)
  let v1482 ← f4 r14 r13 rr14
  let v1484 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr12 v1482) w12
  ripemd_compress_k30 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r13 rr13 r14 rr14 v1484 state

def ripemd_compress_k28 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r5 : UInt256) (rr5 : UInt256) (r6 : UInt256) (rr6 : UInt256) (v1388 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1390 := IntTy.addW (.u 32) v1388 (UInt256.ofNat 1352829926)
  let v1392 ← rotate_left v1390 (UInt256.ofNat 5)
  let r7 := IntTy.addW (.u 32) v1392 rr5
  let rr7 ← rotate_left r5 (UInt256.ofNat 10)
  let v1398 ← f4 r7 r6 rr7
  let v1400 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr5 v1398) w13
  let v1402 := IntTy.addW (.u 32) v1400 (UInt256.ofNat 1352829926)
  let v1404 ← rotate_left v1402 (UInt256.ofNat 7)
  let r8 := IntTy.addW (.u 32) v1404 rr6
  let rr8 ← rotate_left r6 (UInt256.ofNat 10)
  let v1410 ← f4 r8 r7 rr8
  let v1412 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr6 v1410) w6
  let v1414 := IntTy.addW (.u 32) v1412 (UInt256.ofNat 1352829926)
  let v1416 ← rotate_left v1414 (UInt256.ofNat 7)
  let r9 := IntTy.addW (.u 32) v1416 rr7
  let rr9 ← rotate_left r7 (UInt256.ofNat 10)
  let v1422 ← f4 r9 r8 rr9
  let v1424 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr7 v1422) w15
  let v1426 := IntTy.addW (.u 32) v1424 (UInt256.ofNat 1352829926)
  let v1428 ← rotate_left v1426 (UInt256.ofNat 8)
  let r10 := IntTy.addW (.u 32) v1428 rr8
  let rr10 ← rotate_left r8 (UInt256.ofNat 10)
  let v1434 ← f4 r10 r9 rr10
  let v1436 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr8 v1434) w8
  ripemd_compress_k29 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r9 rr9 r10 rr10 v1436 state

def ripemd_compress_k27 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1296 : UInt256) (lr79 : UInt256) (r1 : UInt256) (rr1 : UInt256) (r2 : UInt256) (rr2 : UInt256) (v1340 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1342 := IntTy.addW (.u 32) v1340 (UInt256.ofNat 1352829926)
  let v1344 ← rotate_left v1342 (UInt256.ofNat 11)
  let r3 := IntTy.addW (.u 32) v1344 rr1
  let rr3 ← rotate_left r1 (UInt256.ofNat 10)
  let v1350 ← f4 r3 r2 rr3
  let v1352 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr1 v1350) w9
  let v1354 := IntTy.addW (.u 32) v1352 (UInt256.ofNat 1352829926)
  let v1356 ← rotate_left v1354 (UInt256.ofNat 13)
  let r4 := IntTy.addW (.u 32) v1356 rr2
  let rr4 ← rotate_left r2 (UInt256.ofNat 10)
  let v1362 ← f4 r4 r3 rr4
  let v1364 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr2 v1362) w2
  let v1366 := IntTy.addW (.u 32) v1364 (UInt256.ofNat 1352829926)
  let v1368 ← rotate_left v1366 (UInt256.ofNat 15)
  let r5 := IntTy.addW (.u 32) v1368 rr3
  let rr5 ← rotate_left r3 (UInt256.ofNat 10)
  let v1374 ← f4 r5 r4 rr5
  let v1376 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr3 v1374) w11
  let v1378 := IntTy.addW (.u 32) v1376 (UInt256.ofNat 1352829926)
  let v1380 ← rotate_left v1378 (UInt256.ofNat 15)
  let r6 := IntTy.addW (.u 32) v1380 rr4
  let rr6 ← rotate_left r4 (UInt256.ofNat 10)
  let v1386 ← f4 r6 r5 rr6
  let v1388 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr4 v1386) w4
  ripemd_compress_k28 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r5 rr5 r6 rr6 v1388 state

def ripemd_compress_k26 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr76 : UInt256) (l77 : UInt256) (lr77 : UInt256) (l78 : UInt256) (lr78 : UInt256) (v1290 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1292 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr76 v1290) w13
  let v1294 := IntTy.addW (.u 32) v1292 (UInt256.ofNat 2840853838)
  let v1296 ← rotate_left v1294 (UInt256.ofNat 6)
  let lr79 ← rotate_left l77 (UInt256.ofNat 10)
  let v1302 ← f4 lb lc ld
  let v1304 := IntTy.addW (.u 32) (IntTy.addW (.u 32) la v1302) w5
  let v1306 := IntTy.addW (.u 32) v1304 (UInt256.ofNat 1352829926)
  let v1308 ← rotate_left v1306 (UInt256.ofNat 8)
  let r0 := IntTy.addW (.u 32) v1308 le
  let rr0 ← rotate_left lc (UInt256.ofNat 10)
  let v1314 ← f4 r0 lb rr0
  let v1316 := IntTy.addW (.u 32) (IntTy.addW (.u 32) le v1314) w14
  let v1318 := IntTy.addW (.u 32) v1316 (UInt256.ofNat 1352829926)
  let v1320 ← rotate_left v1318 (UInt256.ofNat 9)
  let r1 := IntTy.addW (.u 32) v1320 ld
  let rr1 ← rotate_left lb (UInt256.ofNat 10)
  let v1326 ← f4 r1 r0 rr1
  let v1328 := IntTy.addW (.u 32) (IntTy.addW (.u 32) ld v1326) w7
  let v1330 := IntTy.addW (.u 32) v1328 (UInt256.ofNat 1352829926)
  let v1332 ← rotate_left v1330 (UInt256.ofNat 9)
  let r2 := IntTy.addW (.u 32) v1332 rr0
  let rr2 ← rotate_left r0 (UInt256.ofNat 10)
  let v1338 ← f4 r2 r1 rr2
  let v1340 := IntTy.addW (.u 32) (IntTy.addW (.u 32) rr0 v1338) w0
  ripemd_compress_k27 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 lr77 l78 lr78 v1296 lr79 r1 rr1 r2 rr2 v1340 state

def ripemd_compress_k25 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr72 : UInt256) (l73 : UInt256) (lr73 : UInt256) (l74 : UInt256) (lr74 : UInt256) (v1242 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1244 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr72 v1242) w8
  let v1246 := IntTy.addW (.u 32) v1244 (UInt256.ofNat 2840853838)
  let v1248 ← rotate_left v1246 (UInt256.ofNat 14)
  let l75 := IntTy.addW (.u 32) v1248 lr73
  let lr75 ← rotate_left l73 (UInt256.ofNat 10)
  let v1254 ← f4 l75 l74 lr75
  let v1256 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr73 v1254) w11
  let v1258 := IntTy.addW (.u 32) v1256 (UInt256.ofNat 2840853838)
  let v1260 ← rotate_left v1258 (UInt256.ofNat 11)
  let l76 := IntTy.addW (.u 32) v1260 lr74
  let lr76 ← rotate_left l74 (UInt256.ofNat 10)
  let v1266 ← f4 l76 l75 lr76
  let v1268 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr74 v1266) w6
  let v1270 := IntTy.addW (.u 32) v1268 (UInt256.ofNat 2840853838)
  let v1272 ← rotate_left v1270 (UInt256.ofNat 8)
  let l77 := IntTy.addW (.u 32) v1272 lr75
  let lr77 ← rotate_left l75 (UInt256.ofNat 10)
  let v1278 ← f4 l77 l76 lr77
  let v1280 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr75 v1278) w15
  let v1282 := IntTy.addW (.u 32) v1280 (UInt256.ofNat 2840853838)
  let v1284 ← rotate_left v1282 (UInt256.ofNat 5)
  let l78 := IntTy.addW (.u 32) v1284 lr76
  let lr78 ← rotate_left l76 (UInt256.ofNat 10)
  let v1290 ← f4 l78 l77 lr78
  ripemd_compress_k26 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr76 l77 lr77 l78 lr78 v1290 state

def ripemd_compress_k24 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr68 : UInt256) (l69 : UInt256) (lr69 : UInt256) (l70 : UInt256) (lr70 : UInt256) (v1194 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1196 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr68 v1194) w10
  let v1198 := IntTy.addW (.u 32) v1196 (UInt256.ofNat 2840853838)
  let v1200 ← rotate_left v1198 (UInt256.ofNat 12)
  let l71 := IntTy.addW (.u 32) v1200 lr69
  let lr71 ← rotate_left l69 (UInt256.ofNat 10)
  let v1206 ← f4 l71 l70 lr71
  let v1208 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr69 v1206) w14
  let v1210 := IntTy.addW (.u 32) v1208 (UInt256.ofNat 2840853838)
  let v1212 ← rotate_left v1210 (UInt256.ofNat 5)
  let l72 := IntTy.addW (.u 32) v1212 lr70
  let lr72 ← rotate_left l70 (UInt256.ofNat 10)
  let v1218 ← f4 l72 l71 lr72
  let v1220 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr70 v1218) w1
  let v1222 := IntTy.addW (.u 32) v1220 (UInt256.ofNat 2840853838)
  let v1224 ← rotate_left v1222 (UInt256.ofNat 12)
  let l73 := IntTy.addW (.u 32) v1224 lr71
  let lr73 ← rotate_left l71 (UInt256.ofNat 10)
  let v1230 ← f4 l73 l72 lr73
  let v1232 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr71 v1230) w3
  let v1234 := IntTy.addW (.u 32) v1232 (UInt256.ofNat 2840853838)
  let v1236 ← rotate_left v1234 (UInt256.ofNat 13)
  let l74 := IntTy.addW (.u 32) v1236 lr72
  let lr74 ← rotate_left l72 (UInt256.ofNat 10)
  let v1242 ← f4 l74 l73 lr74
  ripemd_compress_k25 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr72 l73 lr73 l74 lr74 v1242 state

def ripemd_compress_k23 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr64 : UInt256) (l65 : UInt256) (lr65 : UInt256) (l66 : UInt256) (lr66 : UInt256) (v1146 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1148 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr64 v1146) w9
  let v1150 := IntTy.addW (.u 32) v1148 (UInt256.ofNat 2840853838)
  let v1152 ← rotate_left v1150 (UInt256.ofNat 11)
  let l67 := IntTy.addW (.u 32) v1152 lr65
  let lr67 ← rotate_left l65 (UInt256.ofNat 10)
  let v1158 ← f4 l67 l66 lr67
  let v1160 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr65 v1158) w7
  let v1162 := IntTy.addW (.u 32) v1160 (UInt256.ofNat 2840853838)
  let v1164 ← rotate_left v1162 (UInt256.ofNat 6)
  let l68 := IntTy.addW (.u 32) v1164 lr66
  let lr68 ← rotate_left l66 (UInt256.ofNat 10)
  let v1170 ← f4 l68 l67 lr68
  let v1172 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr66 v1170) w12
  let v1174 := IntTy.addW (.u 32) v1172 (UInt256.ofNat 2840853838)
  let v1176 ← rotate_left v1174 (UInt256.ofNat 8)
  let l69 := IntTy.addW (.u 32) v1176 lr67
  let lr69 ← rotate_left l67 (UInt256.ofNat 10)
  let v1182 ← f4 l69 l68 lr69
  let v1184 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr67 v1182) w2
  let v1186 := IntTy.addW (.u 32) v1184 (UInt256.ofNat 2840853838)
  let v1188 ← rotate_left v1186 (UInt256.ofNat 13)
  let l70 := IntTy.addW (.u 32) v1188 lr68
  let lr70 ← rotate_left l68 (UInt256.ofNat 10)
  let v1194 ← f4 l70 l69 lr70
  ripemd_compress_k24 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr68 l69 lr69 l70 lr70 v1194 state

def ripemd_compress_k22 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr60 : UInt256) (l61 : UInt256) (lr61 : UInt256) (l62 : UInt256) (lr62 : UInt256) (v1098 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1100 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr60 v1098) w2
  let v1102 := IntTy.addW (.u 32) v1100 (UInt256.ofNat 2400959708)
  let v1104 ← rotate_left v1102 (UInt256.ofNat 12)
  let l63 := IntTy.addW (.u 32) v1104 lr61
  let lr63 ← rotate_left l61 (UInt256.ofNat 10)
  let v1110 ← f4 l63 l62 lr63
  let v1112 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr61 v1110) w4
  let v1114 := IntTy.addW (.u 32) v1112 (UInt256.ofNat 2840853838)
  let v1116 ← rotate_left v1114 (UInt256.ofNat 9)
  let l64 := IntTy.addW (.u 32) v1116 lr62
  let lr64 ← rotate_left l62 (UInt256.ofNat 10)
  let v1122 ← f4 l64 l63 lr64
  let v1124 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr62 v1122) w0
  let v1126 := IntTy.addW (.u 32) v1124 (UInt256.ofNat 2840853838)
  let v1128 ← rotate_left v1126 (UInt256.ofNat 15)
  let l65 := IntTy.addW (.u 32) v1128 lr63
  let lr65 ← rotate_left l63 (UInt256.ofNat 10)
  let v1134 ← f4 l65 l64 lr65
  let v1136 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr63 v1134) w5
  let v1138 := IntTy.addW (.u 32) v1136 (UInt256.ofNat 2840853838)
  let v1140 ← rotate_left v1138 (UInt256.ofNat 5)
  let l66 := IntTy.addW (.u 32) v1140 lr64
  let lr66 ← rotate_left l64 (UInt256.ofNat 10)
  let v1146 ← f4 l66 l65 lr66
  ripemd_compress_k23 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr64 l65 lr65 l66 lr66 v1146 state

def ripemd_compress_k21 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr56 : UInt256) (l57 : UInt256) (lr57 : UInt256) (l58 : UInt256) (lr58 : UInt256) (v1050 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1052 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr56 v1050) w15
  let v1054 := IntTy.addW (.u 32) v1052 (UInt256.ofNat 2400959708)
  let v1056 ← rotate_left v1054 (UInt256.ofNat 6)
  let l59 := IntTy.addW (.u 32) v1056 lr57
  let lr59 ← rotate_left l57 (UInt256.ofNat 10)
  let v1062 ← f3 l59 l58 lr59
  let v1064 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr57 v1062) w14
  let v1066 := IntTy.addW (.u 32) v1064 (UInt256.ofNat 2400959708)
  let v1068 ← rotate_left v1066 (UInt256.ofNat 8)
  let l60 := IntTy.addW (.u 32) v1068 lr58
  let lr60 ← rotate_left l58 (UInt256.ofNat 10)
  let v1074 ← f3 l60 l59 lr60
  let v1076 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr58 v1074) w5
  let v1078 := IntTy.addW (.u 32) v1076 (UInt256.ofNat 2400959708)
  let v1080 ← rotate_left v1078 (UInt256.ofNat 6)
  let l61 := IntTy.addW (.u 32) v1080 lr59
  let lr61 ← rotate_left l59 (UInt256.ofNat 10)
  let v1086 ← f3 l61 l60 lr61
  let v1088 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr59 v1086) w6
  let v1090 := IntTy.addW (.u 32) v1088 (UInt256.ofNat 2400959708)
  let v1092 ← rotate_left v1090 (UInt256.ofNat 5)
  let l62 := IntTy.addW (.u 32) v1092 lr60
  let lr62 ← rotate_left l60 (UInt256.ofNat 10)
  let v1098 ← f3 l62 l61 lr62
  ripemd_compress_k22 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr60 l61 lr61 l62 lr62 v1098 state

def ripemd_compress_k20 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr52 : UInt256) (l53 : UInt256) (lr53 : UInt256) (l54 : UInt256) (lr54 : UInt256) (v1002 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v1004 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr52 v1002) w4
  let v1006 := IntTy.addW (.u 32) v1004 (UInt256.ofNat 2400959708)
  let v1008 ← rotate_left v1006 (UInt256.ofNat 8)
  let l55 := IntTy.addW (.u 32) v1008 lr53
  let lr55 ← rotate_left l53 (UInt256.ofNat 10)
  let v1014 ← f3 l55 l54 lr55
  let v1016 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr53 v1014) w13
  let v1018 := IntTy.addW (.u 32) v1016 (UInt256.ofNat 2400959708)
  let v1020 ← rotate_left v1018 (UInt256.ofNat 9)
  let l56 := IntTy.addW (.u 32) v1020 lr54
  let lr56 ← rotate_left l54 (UInt256.ofNat 10)
  let v1026 ← f3 l56 l55 lr56
  let v1028 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr54 v1026) w3
  let v1030 := IntTy.addW (.u 32) v1028 (UInt256.ofNat 2400959708)
  let v1032 ← rotate_left v1030 (UInt256.ofNat 14)
  let l57 := IntTy.addW (.u 32) v1032 lr55
  let lr57 ← rotate_left l55 (UInt256.ofNat 10)
  let v1038 ← f3 l57 l56 lr57
  let v1040 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr55 v1038) w7
  let v1042 := IntTy.addW (.u 32) v1040 (UInt256.ofNat 2400959708)
  let v1044 ← rotate_left v1042 (UInt256.ofNat 5)
  let l58 := IntTy.addW (.u 32) v1044 lr56
  let lr58 ← rotate_left l56 (UInt256.ofNat 10)
  let v1050 ← f3 l58 l57 lr58
  ripemd_compress_k21 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr56 l57 lr57 l58 lr58 v1050 state

def ripemd_compress_k19 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l48 : UInt256) (lr48 : UInt256) (l49 : UInt256) (lr49 : UInt256) (v944 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v948 ← rotate_left (IntTy.addW (.u 32) v944 (UInt256.ofNat 2400959708)) (UInt256.ofNat 14)
  let l50 := IntTy.addW (.u 32) v948 lr48
  let lr50 ← rotate_left l48 (UInt256.ofNat 10)
  let v954 ← f3 l50 l49 lr50
  let v956 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr48 v954) w10
  let v960 ← rotate_left (IntTy.addW (.u 32) v956 (UInt256.ofNat 2400959708)) (UInt256.ofNat 15)
  let l51 := IntTy.addW (.u 32) v960 lr49
  let lr51 ← rotate_left l49 (UInt256.ofNat 10)
  let v966 ← f3 l51 l50 lr51
  let v968 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr49 v966) w0
  let v972 ← rotate_left (IntTy.addW (.u 32) v968 (UInt256.ofNat 2400959708)) (UInt256.ofNat 14)
  let l52 := IntTy.addW (.u 32) v972 lr50
  let lr52 ← rotate_left l50 (UInt256.ofNat 10)
  let v978 ← f3 l52 l51 lr52
  let v980 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr50 v978) w8
  let v984 ← rotate_left (IntTy.addW (.u 32) v980 (UInt256.ofNat 2400959708)) (UInt256.ofNat 15)
  let l53 := IntTy.addW (.u 32) v984 lr51
  let lr53 ← rotate_left l51 (UInt256.ofNat 10)
  let v990 ← f3 l53 l52 lr53
  let v992 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr51 v990) w12
  let v996 ← rotate_left (IntTy.addW (.u 32) v992 (UInt256.ofNat 2400959708)) (UInt256.ofNat 9)
  let l54 := IntTy.addW (.u 32) v996 lr52
  let lr54 ← rotate_left l52 (UInt256.ofNat 10)
  let v1002 ← f3 l54 l53 lr54
  ripemd_compress_k20 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr52 l53 lr53 l54 lr54 v1002 state

def ripemd_compress_k18 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l43 : UInt256) (lr43 : UInt256) (l44 : UInt256) (lr44 : UInt256) (v888 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let l45 := IntTy.addW (.u 32) v888 lr43
  let lr45 ← rotate_left l43 (UInt256.ofNat 10)
  let v894 ← f2 l45 l44 lr45
  let v896 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr43 v894) w5
  let v900 ← rotate_left (IntTy.addW (.u 32) v896 (UInt256.ofNat 1859775393)) (UInt256.ofNat 7)
  let l46 := IntTy.addW (.u 32) v900 lr44
  let lr46 ← rotate_left l44 (UInt256.ofNat 10)
  let v906 ← f2 l46 l45 lr46
  let v908 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr44 v906) w12
  let v912 ← rotate_left (IntTy.addW (.u 32) v908 (UInt256.ofNat 1859775393)) (UInt256.ofNat 5)
  let l47 := IntTy.addW (.u 32) v912 lr45
  let lr47 ← rotate_left l45 (UInt256.ofNat 10)
  let v918 ← f3 l47 l46 lr47
  let v920 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr45 v918) w1
  let v924 ← rotate_left (IntTy.addW (.u 32) v920 (UInt256.ofNat 2400959708)) (UInt256.ofNat 11)
  let l48 := IntTy.addW (.u 32) v924 lr46
  let lr48 ← rotate_left l46 (UInt256.ofNat 10)
  let v930 ← f3 l48 l47 lr48
  let v932 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr46 v930) w9
  let v936 ← rotate_left (IntTy.addW (.u 32) v932 (UInt256.ofNat 2400959708)) (UInt256.ofNat 12)
  let l49 := IntTy.addW (.u 32) v936 lr47
  let lr49 ← rotate_left l47 (UInt256.ofNat 10)
  let v942 ← f3 l49 l48 lr49
  let v944 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr47 v942) w11
  ripemd_compress_k19 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l48 lr48 l49 lr49 v944 state

def ripemd_compress_k17 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l38 : UInt256) (lr38 : UInt256) (l39 : UInt256) (lr39 : UInt256) (l40 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let lr40 ← rotate_left l38 (UInt256.ofNat 10)
  let v834 ← f2 l40 l39 lr40
  let v836 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr38 v834) w7
  let v840 ← rotate_left (IntTy.addW (.u 32) v836 (UInt256.ofNat 1859775393)) (UInt256.ofNat 8)
  let l41 := IntTy.addW (.u 32) v840 lr39
  let lr41 ← rotate_left l39 (UInt256.ofNat 10)
  let v846 ← f2 l41 l40 lr41
  let v848 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr39 v846) w0
  let v852 ← rotate_left (IntTy.addW (.u 32) v848 (UInt256.ofNat 1859775393)) (UInt256.ofNat 13)
  let l42 := IntTy.addW (.u 32) v852 lr40
  let lr42 ← rotate_left l40 (UInt256.ofNat 10)
  let v858 ← f2 l42 l41 lr42
  let v860 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr40 v858) w6
  let v864 ← rotate_left (IntTy.addW (.u 32) v860 (UInt256.ofNat 1859775393)) (UInt256.ofNat 6)
  let l43 := IntTy.addW (.u 32) v864 lr41
  let lr43 ← rotate_left l41 (UInt256.ofNat 10)
  let v870 ← f2 l43 l42 lr43
  let v872 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr41 v870) w13
  let v876 ← rotate_left (IntTy.addW (.u 32) v872 (UInt256.ofNat 1859775393)) (UInt256.ofNat 5)
  let l44 := IntTy.addW (.u 32) v876 lr42
  let lr44 ← rotate_left l42 (UInt256.ofNat 10)
  let v882 ← f2 l44 l43 lr44
  let v884 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr42 v882) w11
  let v888 ← rotate_left (IntTy.addW (.u 32) v884 (UInt256.ofNat 1859775393)) (UInt256.ofNat 12)
  ripemd_compress_k18 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l43 lr43 l44 lr44 v888 state

def ripemd_compress_k16 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr33 : UInt256) (l34 : UInt256) (lr34 : UInt256) (l35 : UInt256) (lr35 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v774 ← f2 l35 l34 lr35
  let v776 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr33 v774) w9
  let v780 ← rotate_left (IntTy.addW (.u 32) v776 (UInt256.ofNat 1859775393)) (UInt256.ofNat 14)
  let l36 := IntTy.addW (.u 32) v780 lr34
  let lr36 ← rotate_left l34 (UInt256.ofNat 10)
  let v786 ← f2 l36 l35 lr36
  let v788 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr34 v786) w15
  let v792 ← rotate_left (IntTy.addW (.u 32) v788 (UInt256.ofNat 1859775393)) (UInt256.ofNat 9)
  let l37 := IntTy.addW (.u 32) v792 lr35
  let lr37 ← rotate_left l35 (UInt256.ofNat 10)
  let v798 ← f2 l37 l36 lr37
  let v800 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr35 v798) w8
  let v804 ← rotate_left (IntTy.addW (.u 32) v800 (UInt256.ofNat 1859775393)) (UInt256.ofNat 13)
  let l38 := IntTy.addW (.u 32) v804 lr36
  let lr38 ← rotate_left l36 (UInt256.ofNat 10)
  let v810 ← f2 l38 l37 lr38
  let v812 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr36 v810) w1
  let v816 ← rotate_left (IntTy.addW (.u 32) v812 (UInt256.ofNat 1859775393)) (UInt256.ofNat 15)
  let l39 := IntTy.addW (.u 32) v816 lr37
  let lr39 ← rotate_left l37 (UInt256.ofNat 10)
  let v822 ← f2 l39 l38 lr39
  let v824 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr37 v822) w2
  let v828 ← rotate_left (IntTy.addW (.u 32) v824 (UInt256.ofNat 1859775393)) (UInt256.ofNat 14)
  let l40 := IntTy.addW (.u 32) v828 lr38
  ripemd_compress_k17 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l38 lr38 l39 lr39 l40 state

def ripemd_compress_k15 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr28 : UInt256) (l29 : UInt256) (lr29 : UInt256) (l30 : UInt256) (lr30 : UInt256) (v714 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v716 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr28 v714) w8
  let v720 ← rotate_left (IntTy.addW (.u 32) v716 (UInt256.ofNat 1518500249)) (UInt256.ofNat 12)
  let l31 := IntTy.addW (.u 32) v720 lr29
  let lr31 ← rotate_left l29 (UInt256.ofNat 10)
  let v726 ← f2 l31 l30 lr31
  let v728 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr29 v726) w3
  let v732 ← rotate_left (IntTy.addW (.u 32) v728 (UInt256.ofNat 1859775393)) (UInt256.ofNat 11)
  let l32 := IntTy.addW (.u 32) v732 lr30
  let lr32 ← rotate_left l30 (UInt256.ofNat 10)
  let v738 ← f2 l32 l31 lr32
  let v740 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr30 v738) w10
  let v744 ← rotate_left (IntTy.addW (.u 32) v740 (UInt256.ofNat 1859775393)) (UInt256.ofNat 13)
  let l33 := IntTy.addW (.u 32) v744 lr31
  let lr33 ← rotate_left l31 (UInt256.ofNat 10)
  let v750 ← f2 l33 l32 lr33
  let v752 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr31 v750) w14
  let v756 ← rotate_left (IntTy.addW (.u 32) v752 (UInt256.ofNat 1859775393)) (UInt256.ofNat 6)
  let l34 := IntTy.addW (.u 32) v756 lr32
  let lr34 ← rotate_left l32 (UInt256.ofNat 10)
  let v762 ← f2 l34 l33 lr34
  let v764 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr32 v762) w4
  let v768 ← rotate_left (IntTy.addW (.u 32) v764 (UInt256.ofNat 1859775393)) (UInt256.ofNat 7)
  let l35 := IntTy.addW (.u 32) v768 lr33
  let lr35 ← rotate_left l33 (UInt256.ofNat 10)
  ripemd_compress_k16 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr33 l34 lr34 l35 lr35 state

def ripemd_compress_k14 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l24 : UInt256) (lr24 : UInt256) (l25 : UInt256) (lr25 : UInt256) (v656 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v660 ← rotate_left (IntTy.addW (.u 32) v656 (UInt256.ofNat 1518500249)) (UInt256.ofNat 15)
  let l26 := IntTy.addW (.u 32) v660 lr24
  let lr26 ← rotate_left l24 (UInt256.ofNat 10)
  let v666 ← f1 l26 l25 lr26
  let v668 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr24 v666) w5
  let v672 ← rotate_left (IntTy.addW (.u 32) v668 (UInt256.ofNat 1518500249)) (UInt256.ofNat 9)
  let l27 := IntTy.addW (.u 32) v672 lr25
  let lr27 ← rotate_left l25 (UInt256.ofNat 10)
  let v678 ← f1 l27 l26 lr27
  let v680 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr25 v678) w2
  let v684 ← rotate_left (IntTy.addW (.u 32) v680 (UInt256.ofNat 1518500249)) (UInt256.ofNat 11)
  let l28 := IntTy.addW (.u 32) v684 lr26
  let lr28 ← rotate_left l26 (UInt256.ofNat 10)
  let v690 ← f1 l28 l27 lr28
  let v692 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr26 v690) w14
  let v696 ← rotate_left (IntTy.addW (.u 32) v692 (UInt256.ofNat 1518500249)) (UInt256.ofNat 7)
  let l29 := IntTy.addW (.u 32) v696 lr27
  let lr29 ← rotate_left l27 (UInt256.ofNat 10)
  let v702 ← f1 l29 l28 lr29
  let v704 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr27 v702) w11
  let v708 ← rotate_left (IntTy.addW (.u 32) v704 (UInt256.ofNat 1518500249)) (UInt256.ofNat 13)
  let l30 := IntTy.addW (.u 32) v708 lr28
  let lr30 ← rotate_left l28 (UInt256.ofNat 10)
  let v714 ← f1 l30 l29 lr30
  ripemd_compress_k15 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr28 l29 lr29 l30 lr30 v714 state

def ripemd_compress_k13 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l19 : UInt256) (lr19 : UInt256) (l20 : UInt256) (lr20 : UInt256) (v600 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let l21 := IntTy.addW (.u 32) v600 lr19
  let lr21 ← rotate_left l19 (UInt256.ofNat 10)
  let v606 ← f1 l21 l20 lr21
  let v608 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr19 v606) w15
  let v612 ← rotate_left (IntTy.addW (.u 32) v608 (UInt256.ofNat 1518500249)) (UInt256.ofNat 7)
  let l22 := IntTy.addW (.u 32) v612 lr20
  let lr22 ← rotate_left l20 (UInt256.ofNat 10)
  let v618 ← f1 l22 l21 lr22
  let v620 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr20 v618) w3
  let v624 ← rotate_left (IntTy.addW (.u 32) v620 (UInt256.ofNat 1518500249)) (UInt256.ofNat 15)
  let l23 := IntTy.addW (.u 32) v624 lr21
  let lr23 ← rotate_left l21 (UInt256.ofNat 10)
  let v630 ← f1 l23 l22 lr23
  let v632 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr21 v630) w12
  let v636 ← rotate_left (IntTy.addW (.u 32) v632 (UInt256.ofNat 1518500249)) (UInt256.ofNat 7)
  let l24 := IntTy.addW (.u 32) v636 lr22
  let lr24 ← rotate_left l22 (UInt256.ofNat 10)
  let v642 ← f1 l24 l23 lr24
  let v644 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr22 v642) w0
  let v648 ← rotate_left (IntTy.addW (.u 32) v644 (UInt256.ofNat 1518500249)) (UInt256.ofNat 12)
  let l25 := IntTy.addW (.u 32) v648 lr23
  let lr25 ← rotate_left l23 (UInt256.ofNat 10)
  let v654 ← f1 l25 l24 lr25
  let v656 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr23 v654) w9
  ripemd_compress_k14 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l24 lr24 l25 lr25 v656 state

def ripemd_compress_k12 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l14 : UInt256) (lr14 : UInt256) (l15 : UInt256) (lr15 : UInt256) (l16 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let lr16 ← rotate_left l14 (UInt256.ofNat 10)
  let v546 ← f1 l16 l15 lr16
  let v548 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr14 v546) w4
  let v552 ← rotate_left (IntTy.addW (.u 32) v548 (UInt256.ofNat 1518500249)) (UInt256.ofNat 6)
  let l17 := IntTy.addW (.u 32) v552 lr15
  let lr17 ← rotate_left l15 (UInt256.ofNat 10)
  let v558 ← f1 l17 l16 lr17
  let v560 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr15 v558) w13
  let v564 ← rotate_left (IntTy.addW (.u 32) v560 (UInt256.ofNat 1518500249)) (UInt256.ofNat 8)
  let l18 := IntTy.addW (.u 32) v564 lr16
  let lr18 ← rotate_left l16 (UInt256.ofNat 10)
  let v570 ← f1 l18 l17 lr18
  let v572 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr16 v570) w1
  let v576 ← rotate_left (IntTy.addW (.u 32) v572 (UInt256.ofNat 1518500249)) (UInt256.ofNat 13)
  let l19 := IntTy.addW (.u 32) v576 lr17
  let lr19 ← rotate_left l17 (UInt256.ofNat 10)
  let v582 ← f1 l19 l18 lr19
  let v584 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr17 v582) w10
  let v588 ← rotate_left (IntTy.addW (.u 32) v584 (UInt256.ofNat 1518500249)) (UInt256.ofNat 11)
  let l20 := IntTy.addW (.u 32) v588 lr18
  let lr20 ← rotate_left l18 (UInt256.ofNat 10)
  let v594 ← f1 l20 l19 lr20
  let v596 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr18 v594) w6
  let v600 ← rotate_left (IntTy.addW (.u 32) v596 (UInt256.ofNat 1518500249)) (UInt256.ofNat 9)
  ripemd_compress_k13 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l19 lr19 l20 lr20 v600 state

def ripemd_compress_k11 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr9 : UInt256) (l10 : UInt256) (lr10 : UInt256) (l11 : UInt256) (lr11 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v494 ← f0 l11 l10 lr11
  let v496 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr9 v494) w12
  let v498 ← rotate_left v496 (UInt256.ofNat 6)
  let l12 := IntTy.addW (.u 32) v498 lr10
  let lr12 ← rotate_left l10 (UInt256.ofNat 10)
  let v504 ← f0 l12 l11 lr12
  let v506 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr10 v504) w13
  let v508 ← rotate_left v506 (UInt256.ofNat 7)
  let l13 := IntTy.addW (.u 32) v508 lr11
  let lr13 ← rotate_left l11 (UInt256.ofNat 10)
  let v514 ← f0 l13 l12 lr13
  let v516 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr11 v514) w14
  let v518 ← rotate_left v516 (UInt256.ofNat 9)
  let l14 := IntTy.addW (.u 32) v518 lr12
  let lr14 ← rotate_left l12 (UInt256.ofNat 10)
  let v524 ← f0 l14 l13 lr14
  let v526 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr12 v524) w15
  let v528 ← rotate_left v526 (UInt256.ofNat 8)
  let l15 := IntTy.addW (.u 32) v528 lr13
  let lr15 ← rotate_left l13 (UInt256.ofNat 10)
  let v534 ← f1 l15 l14 lr15
  let v536 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr13 v534) w7
  let v540 ← rotate_left (IntTy.addW (.u 32) v536 (UInt256.ofNat 1518500249)) (UInt256.ofNat 7)
  let l16 := IntTy.addW (.u 32) v540 lr14
  ripemd_compress_k12 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l14 lr14 l15 lr15 l16 state

def ripemd_compress_k10 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (lr4 : UInt256) (l5 : UInt256) (lr5 : UInt256) (l6 : UInt256) (lr6 : UInt256) (v444 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v446 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr4 v444) w7
  let v448 ← rotate_left v446 (UInt256.ofNat 9)
  let l7 := IntTy.addW (.u 32) v448 lr5
  let lr7 ← rotate_left l5 (UInt256.ofNat 10)
  let v454 ← f0 l7 l6 lr7
  let v456 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr5 v454) w8
  let v458 ← rotate_left v456 (UInt256.ofNat 11)
  let l8 := IntTy.addW (.u 32) v458 lr6
  let lr8 ← rotate_left l6 (UInt256.ofNat 10)
  let v464 ← f0 l8 l7 lr8
  let v466 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr6 v464) w9
  let v468 ← rotate_left v466 (UInt256.ofNat 13)
  let l9 := IntTy.addW (.u 32) v468 lr7
  let lr9 ← rotate_left l7 (UInt256.ofNat 10)
  let v474 ← f0 l9 l8 lr9
  let v476 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr7 v474) w10
  let v478 ← rotate_left v476 (UInt256.ofNat 14)
  let l10 := IntTy.addW (.u 32) v478 lr8
  let lr10 ← rotate_left l8 (UInt256.ofNat 10)
  let v484 ← f0 l10 l9 lr10
  let v486 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr8 v484) w11
  let v488 ← rotate_left v486 (UInt256.ofNat 15)
  let l11 := IntTy.addW (.u 32) v488 lr9
  let lr11 ← rotate_left l9 (UInt256.ofNat 10)
  ripemd_compress_k11 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr9 l10 lr10 l11 lr11 state

def ripemd_compress_k9 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (la : UInt256) (lb : UInt256) (lc : UInt256) (ld : UInt256) (le : UInt256) (l0 : UInt256) (lr0 : UInt256) (l1 : UInt256) (lr1 : UInt256) (v394 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v398 ← rotate_left (IntTy.addW (.u 32) (IntTy.addW (.u 32) ld v394) w2) (UInt256.ofNat 15)
  let l2 := IntTy.addW (.u 32) v398 lr0
  let lr2 ← rotate_left l0 (UInt256.ofNat 10)
  let v404 ← f0 l2 l1 lr2
  let v406 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr0 v404) w3
  let v408 ← rotate_left v406 (UInt256.ofNat 12)
  let l3 := IntTy.addW (.u 32) v408 lr1
  let lr3 ← rotate_left l1 (UInt256.ofNat 10)
  let v414 ← f0 l3 l2 lr3
  let v416 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr1 v414) w4
  let v418 ← rotate_left v416 (UInt256.ofNat 5)
  let l4 := IntTy.addW (.u 32) v418 lr2
  let lr4 ← rotate_left l2 (UInt256.ofNat 10)
  let v424 ← f0 l4 l3 lr4
  let v426 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr2 v424) w5
  let v428 ← rotate_left v426 (UInt256.ofNat 8)
  let l5 := IntTy.addW (.u 32) v428 lr3
  let lr5 ← rotate_left l3 (UInt256.ofNat 10)
  let v434 ← f0 l5 l4 lr5
  let v436 := IntTy.addW (.u 32) (IntTy.addW (.u 32) lr3 v434) w6
  let v438 ← rotate_left v436 (UInt256.ofNat 7)
  let l6 := IntTy.addW (.u 32) v438 lr4
  let lr6 ← rotate_left l4 (UInt256.ofNat 10)
  let v444 ← f0 l6 l5 lr6
  ripemd_compress_k10 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le lr4 l5 lr5 l6 lr6 v444 state

def ripemd_compress_k8 (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (w14 : UInt256) (w15 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v354 := (UInt256.ofNat 0)
  checkIndex v354 5
  let la := state[v354.toNat]!
  let v357 := (UInt256.ofNat 1)
  checkIndex v357 5
  let lb := state[v357.toNat]!
  let v360 := (UInt256.ofNat 2)
  checkIndex v360 5
  let lc := state[v360.toNat]!
  let v363 := (UInt256.ofNat 3)
  checkIndex v363 5
  let ld := state[v363.toNat]!
  let v366 := (UInt256.ofNat 4)
  checkIndex v366 5
  let le := state[v366.toNat]!
  let v374 ← f0 lb lc ld
  let v378 ← rotate_left (IntTy.addW (.u 32) (IntTy.addW (.u 32) la v374) w0) (UInt256.ofNat 11)
  let l0 := IntTy.addW (.u 32) v378 le
  let lr0 ← rotate_left lc (UInt256.ofNat 10)
  let v384 ← f0 l0 lb lr0
  let v388 ← rotate_left (IntTy.addW (.u 32) (IntTy.addW (.u 32) le v384) w1) (UInt256.ofNat 14)
  let l1 := IntTy.addW (.u 32) v388 ld
  let lr1 ← rotate_left lb (UInt256.ofNat 10)
  let v394 ← f0 l1 l0 lr1
  ripemd_compress_k9 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 la lb lc ld le l0 lr0 l1 lr1 v394 state

def ripemd_compress_k7 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (w12 : UInt256) (w13 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v310 := (UInt256.ofNat 56)
  checkIndex v310 64
  let v313 := (UInt256.ofNat 57)
  checkIndex v313 64
  let v317 := IntTy.shl (.u 32) block[v313.toNat]! (UInt256.ofNat 8)
  let v319 := (UInt256.ofNat 58)
  checkIndex v319 64
  let v323 := IntTy.shl (.u 32) block[v319.toNat]! (UInt256.ofNat 16)
  let v325 := (UInt256.ofNat 59)
  checkIndex v325 64
  let v329 := IntTy.shl (.u 32) block[v325.toNat]! (UInt256.ofNat 24)
  let w14 := ((block[v310.toNat]! ||| v317) ||| v323) ||| v329
  let v332 := (UInt256.ofNat 60)
  checkIndex v332 64
  let v335 := (UInt256.ofNat 61)
  checkIndex v335 64
  let v339 := IntTy.shl (.u 32) block[v335.toNat]! (UInt256.ofNat 8)
  let v341 := (UInt256.ofNat 62)
  checkIndex v341 64
  let v345 := IntTy.shl (.u 32) block[v341.toNat]! (UInt256.ofNat 16)
  let v347 := (UInt256.ofNat 63)
  checkIndex v347 64
  let v351 := IntTy.shl (.u 32) block[v347.toNat]! (UInt256.ofNat 24)
  let w15 := ((block[v332.toNat]! ||| v339) ||| v345) ||| v351
  ripemd_compress_k8 w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 state

def ripemd_compress_k6 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (w10 : UInt256) (w11 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v266 := (UInt256.ofNat 48)
  checkIndex v266 64
  let v269 := (UInt256.ofNat 49)
  checkIndex v269 64
  let v273 := IntTy.shl (.u 32) block[v269.toNat]! (UInt256.ofNat 8)
  let v275 := (UInt256.ofNat 50)
  checkIndex v275 64
  let v279 := IntTy.shl (.u 32) block[v275.toNat]! (UInt256.ofNat 16)
  let v281 := (UInt256.ofNat 51)
  checkIndex v281 64
  let v285 := IntTy.shl (.u 32) block[v281.toNat]! (UInt256.ofNat 24)
  let w12 := ((block[v266.toNat]! ||| v273) ||| v279) ||| v285
  let v288 := (UInt256.ofNat 52)
  checkIndex v288 64
  let v291 := (UInt256.ofNat 53)
  checkIndex v291 64
  let v295 := IntTy.shl (.u 32) block[v291.toNat]! (UInt256.ofNat 8)
  let v297 := (UInt256.ofNat 54)
  checkIndex v297 64
  let v301 := IntTy.shl (.u 32) block[v297.toNat]! (UInt256.ofNat 16)
  let v303 := (UInt256.ofNat 55)
  checkIndex v303 64
  let v307 := IntTy.shl (.u 32) block[v303.toNat]! (UInt256.ofNat 24)
  let w13 := ((block[v288.toNat]! ||| v295) ||| v301) ||| v307
  ripemd_compress_k7 block w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 state

def ripemd_compress_k5 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (w8 : UInt256) (w9 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v222 := (UInt256.ofNat 40)
  checkIndex v222 64
  let v225 := (UInt256.ofNat 41)
  checkIndex v225 64
  let v229 := IntTy.shl (.u 32) block[v225.toNat]! (UInt256.ofNat 8)
  let v231 := (UInt256.ofNat 42)
  checkIndex v231 64
  let v235 := IntTy.shl (.u 32) block[v231.toNat]! (UInt256.ofNat 16)
  let v237 := (UInt256.ofNat 43)
  checkIndex v237 64
  let v241 := IntTy.shl (.u 32) block[v237.toNat]! (UInt256.ofNat 24)
  let w10 := ((block[v222.toNat]! ||| v229) ||| v235) ||| v241
  let v244 := (UInt256.ofNat 44)
  checkIndex v244 64
  let v247 := (UInt256.ofNat 45)
  checkIndex v247 64
  let v251 := IntTy.shl (.u 32) block[v247.toNat]! (UInt256.ofNat 8)
  let v253 := (UInt256.ofNat 46)
  checkIndex v253 64
  let v257 := IntTy.shl (.u 32) block[v253.toNat]! (UInt256.ofNat 16)
  let v259 := (UInt256.ofNat 47)
  checkIndex v259 64
  let v263 := IntTy.shl (.u 32) block[v259.toNat]! (UInt256.ofNat 24)
  let w11 := ((block[v244.toNat]! ||| v251) ||| v257) ||| v263
  ripemd_compress_k6 block w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 state

def ripemd_compress_k4 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (w6 : UInt256) (w7 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v178 := (UInt256.ofNat 32)
  checkIndex v178 64
  let v181 := (UInt256.ofNat 33)
  checkIndex v181 64
  let v185 := IntTy.shl (.u 32) block[v181.toNat]! (UInt256.ofNat 8)
  let v187 := (UInt256.ofNat 34)
  checkIndex v187 64
  let v191 := IntTy.shl (.u 32) block[v187.toNat]! (UInt256.ofNat 16)
  let v193 := (UInt256.ofNat 35)
  checkIndex v193 64
  let v197 := IntTy.shl (.u 32) block[v193.toNat]! (UInt256.ofNat 24)
  let w8 := ((block[v178.toNat]! ||| v185) ||| v191) ||| v197
  let v200 := (UInt256.ofNat 36)
  checkIndex v200 64
  let v203 := (UInt256.ofNat 37)
  checkIndex v203 64
  let v207 := IntTy.shl (.u 32) block[v203.toNat]! (UInt256.ofNat 8)
  let v209 := (UInt256.ofNat 38)
  checkIndex v209 64
  let v213 := IntTy.shl (.u 32) block[v209.toNat]! (UInt256.ofNat 16)
  let v215 := (UInt256.ofNat 39)
  checkIndex v215 64
  let v219 := IntTy.shl (.u 32) block[v215.toNat]! (UInt256.ofNat 24)
  let w9 := ((block[v200.toNat]! ||| v207) ||| v213) ||| v219
  ripemd_compress_k5 block w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 state

def ripemd_compress_k3 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (w4 : UInt256) (w5 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v134 := (UInt256.ofNat 24)
  checkIndex v134 64
  let v137 := (UInt256.ofNat 25)
  checkIndex v137 64
  let v141 := IntTy.shl (.u 32) block[v137.toNat]! (UInt256.ofNat 8)
  let v143 := (UInt256.ofNat 26)
  checkIndex v143 64
  let v147 := IntTy.shl (.u 32) block[v143.toNat]! (UInt256.ofNat 16)
  let v149 := (UInt256.ofNat 27)
  checkIndex v149 64
  let v153 := IntTy.shl (.u 32) block[v149.toNat]! (UInt256.ofNat 24)
  let w6 := ((block[v134.toNat]! ||| v141) ||| v147) ||| v153
  let v156 := (UInt256.ofNat 28)
  checkIndex v156 64
  let v159 := (UInt256.ofNat 29)
  checkIndex v159 64
  let v163 := IntTy.shl (.u 32) block[v159.toNat]! (UInt256.ofNat 8)
  let v165 := (UInt256.ofNat 30)
  checkIndex v165 64
  let v169 := IntTy.shl (.u 32) block[v165.toNat]! (UInt256.ofNat 16)
  let v171 := (UInt256.ofNat 31)
  checkIndex v171 64
  let v175 := IntTy.shl (.u 32) block[v171.toNat]! (UInt256.ofNat 24)
  let w7 := ((block[v156.toNat]! ||| v163) ||| v169) ||| v175
  ripemd_compress_k4 block w0 w1 w2 w3 w4 w5 w6 w7 state

def ripemd_compress_k2 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (w2 : UInt256) (w3 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v90 := (UInt256.ofNat 16)
  checkIndex v90 64
  let v93 := (UInt256.ofNat 17)
  checkIndex v93 64
  let v97 := IntTy.shl (.u 32) block[v93.toNat]! (UInt256.ofNat 8)
  let v99 := (UInt256.ofNat 18)
  checkIndex v99 64
  let v103 := IntTy.shl (.u 32) block[v99.toNat]! (UInt256.ofNat 16)
  let v105 := (UInt256.ofNat 19)
  checkIndex v105 64
  let v109 := IntTy.shl (.u 32) block[v105.toNat]! (UInt256.ofNat 24)
  let w4 := ((block[v90.toNat]! ||| v97) ||| v103) ||| v109
  let v112 := (UInt256.ofNat 20)
  checkIndex v112 64
  let v115 := (UInt256.ofNat 21)
  checkIndex v115 64
  let v119 := IntTy.shl (.u 32) block[v115.toNat]! (UInt256.ofNat 8)
  let v121 := (UInt256.ofNat 22)
  checkIndex v121 64
  let v125 := IntTy.shl (.u 32) block[v121.toNat]! (UInt256.ofNat 16)
  let v127 := (UInt256.ofNat 23)
  checkIndex v127 64
  let v131 := IntTy.shl (.u 32) block[v127.toNat]! (UInt256.ofNat 24)
  let w5 := ((block[v112.toNat]! ||| v119) ||| v125) ||| v131
  ripemd_compress_k3 block w0 w1 w2 w3 w4 w5 state

def ripemd_compress_k1 (block : (Vector UInt256 64)) (w0 : UInt256) (w1 : UInt256) (state : (Vector UInt256 5)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v46 := (UInt256.ofNat 8)
  checkIndex v46 64
  let v49 := (UInt256.ofNat 9)
  checkIndex v49 64
  let v53 := IntTy.shl (.u 32) block[v49.toNat]! (UInt256.ofNat 8)
  let v55 := (UInt256.ofNat 10)
  checkIndex v55 64
  let v59 := IntTy.shl (.u 32) block[v55.toNat]! (UInt256.ofNat 16)
  let v61 := (UInt256.ofNat 11)
  checkIndex v61 64
  let v65 := IntTy.shl (.u 32) block[v61.toNat]! (UInt256.ofNat 24)
  let w2 := ((block[v46.toNat]! ||| v53) ||| v59) ||| v65
  let v68 := (UInt256.ofNat 12)
  checkIndex v68 64
  let v71 := (UInt256.ofNat 13)
  checkIndex v71 64
  let v75 := IntTy.shl (.u 32) block[v71.toNat]! (UInt256.ofNat 8)
  let v77 := (UInt256.ofNat 14)
  checkIndex v77 64
  let v81 := IntTy.shl (.u 32) block[v77.toNat]! (UInt256.ofNat 16)
  let v83 := (UInt256.ofNat 15)
  checkIndex v83 64
  let v87 := IntTy.shl (.u 32) block[v83.toNat]! (UInt256.ofNat 24)
  let w3 := ((block[v68.toNat]! ||| v75) ||| v81) ||| v87
  ripemd_compress_k2 block w0 w1 w2 w3 state

/-- `lib::hashes::ripemd_compress` -/
def ripemd_compress (state : (Vector UInt256 5)) (block : (Vector UInt256 64)) : FeM (Unit × (Vector UInt256 5)) := do
  let mut state := state
  let v2 := (UInt256.ofNat 0)
  checkIndex v2 64
  let v5 := (UInt256.ofNat 1)
  checkIndex v5 64
  let v9 := IntTy.shl (.u 32) block[v5.toNat]! (UInt256.ofNat 8)
  let v11 := (UInt256.ofNat 2)
  checkIndex v11 64
  let v15 := IntTy.shl (.u 32) block[v11.toNat]! (UInt256.ofNat 16)
  let v17 := (UInt256.ofNat 3)
  checkIndex v17 64
  let v21 := IntTy.shl (.u 32) block[v17.toNat]! (UInt256.ofNat 24)
  let w0 := ((block[v2.toNat]! ||| v9) ||| v15) ||| v21
  let v24 := (UInt256.ofNat 4)
  checkIndex v24 64
  let v27 := (UInt256.ofNat 5)
  checkIndex v27 64
  let v31 := IntTy.shl (.u 32) block[v27.toNat]! (UInt256.ofNat 8)
  let v33 := (UInt256.ofNat 6)
  checkIndex v33 64
  let v37 := IntTy.shl (.u 32) block[v33.toNat]! (UInt256.ofNat 16)
  let v39 := (UInt256.ofNat 7)
  checkIndex v39 64
  let v43 := IntTy.shl (.u 32) block[v39.toNat]! (UInt256.ofNat 24)
  let w1 := ((block[v24.toNat]! ||| v31) ||| v37) ||| v43
  ripemd_compress_k1 block w0 w1 state

/-- `lib::hashes::ripemd160<MemBuffer>` -/
def ripemd160_MemBuffer (input : MemBuffer) : FeM UInt256 := do
  let mut state : (Vector UInt256 5) := default
  let mut digest : UInt256 := default
  let mut r2 : Range_Known_0_Unknown := default
  let mut r3 : (Vector UInt256 64) := default
  let state_3 ← RIPEMD_IV
  state := state_3
  let v5 ← MemBuffer_len_1 input
  let blocks ← padded_blocks v5
  r2 := (Range_Known_0_Unknown.mk () blocks)
  let v12 ← Range_start_0 r2
  let rec loop (state : (Vector UInt256 5)) (digest : UInt256) (r2 : Range_Known_0_Unknown) (r3 : (Vector UInt256 64)) (v1 : UInt256) : FeM UInt256 := do
    let mut state := state
    let mut digest := digest
    let mut r2 := r2
    let mut r3 := r3
    let v15 ← Range_has_0 r2 v1
    if v15 then
      let v17 ← Range_at_0 r2 v1
      let v21 ← MemBuffer_len_1 input
      let v22 ← padded_block_MemBuffer input (UInt256.ofNat 0) v21 v17 blocks
      r3 := v22
      let (v24, v24_0) ← ripemd_compress state r3
      state := v24_0
      let v26 ← Range_next_0 r2 v1
      loop state digest r2 r3 v26
    else
      digest := (UInt256.ofNat 0)
      let v31 ← array_start_u32_5 state
      let rec loop_1 (state : (Vector UInt256 5)) (digest : UInt256) (r2 : Range_Known_0_Unknown) (r3 : (Vector UInt256 64)) (v2 : UInt256) : FeM UInt256 := do
        let mut state := state
        let mut digest := digest
        let mut r2 := r2
        let mut r3 := r3
        let v34 ← array_has_u32_5 state v2
        if v34 then
          let v36 ← array_at_ref_u32_5 state v2
          let v41 := digest <<< (UInt256.ofNat 32)
          digest := (v41 ||| (IntTy.swapBytes (.u 32) v36))
          let v48 ← array_next_u32_5 state v2
          loop_1 state digest r2 r3 v48
        else
          return digest
      partial_fixpoint
      loop_1 state digest r2 r3 v31
  partial_fixpoint
  loop state digest r2 r3 v12

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Empty := do
  let mut output : MemBuffer := default
  let mut r2 : MemBuffer := default
  let output_1 ← MemBuffer_new (UInt256.ofNat 32)
  output := output_1
  let v4 ← input_Evm_2
  r2 := v4
  let v6 ← ripemd160_MemBuffer r2
  let (v8, v8_0) ← MemBuffer_write_word output (UInt256.ofNat 0) v6
  output := v8_0
  let _ ← MemBuffer_drop r2
  let v10 ← MemBuffer_span_ref output
  let v13 ← Evm_return_data v10
  nomatch v13

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  nonPayable
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `Ripemd160`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
  else
    let selector ← calldataSelector
    recv_fallback_entry

/-- The deployment entry of contract `Ripemd160`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


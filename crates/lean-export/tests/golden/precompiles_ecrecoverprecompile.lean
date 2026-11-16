export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data MemBuffer MemBuffer_drop MemBuffer_empty MemBuffer_new MemBuffer_span_ref MemBuffer_try_new MemBuffer_write_word Option_u256 Option_u256.None Option_u256.Some Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_at_0 Range_has_0 Range_next_0 Range_start_0 downcast_truncate_u256_u64 downcast_unchecked_u256_usize downcast_unchecked_usize_u64 «assert»)
export Shared (N Output Output.mk SUCCESS host_allocation recover word_at_MemBuffer)

structure Keccak256 where
  lanes : (Vector UInt256 25)
  position : UInt256
deriving Inhabited

/-- The `init` of contract `EcrecoverPrecompile`, which its source leaves out -/
def init : FeM Unit := do
  return ()

/-- `lib::input<Evm>` -/
def input_Evm_4 : FeM MemBuffer := do
  let mut input : MemBuffer := default
  let len ← Evm_calldatasize
  let v3 ← downcast_unchecked_u256_usize len
  let input_4 ← MemBuffer_new v3
  input := input_4
  let (v10, v10_0) ← Evm_calldatacopy input (UInt256.ofNat 0) (UInt256.ofNat 0) len
  input := v10_0
  return input

/-- `lib::words<1>` -/
def words_1 (values : (Vector UInt256 1)) (gas_used : UInt256) : FeM Output := do
  let mut output : MemBuffer := default
  let mut r1 : Range_Known_0_Unknown := default
  let v5 ← checkedMul (UInt256.ofNat 1) (UInt256.ofNat 32)
  let v6 ← MemBuffer_try_new v5
  match (← pure v6) with
  | .Some .. =>
    output := v6.Some_0
    r1 := (Range_Known_0_Unknown.mk () (UInt256.ofNat 1))
    let v15 ← Range_start_0 r1
    let rec loop (output : MemBuffer) (r1 : Range_Known_0_Unknown) (v2 : UInt256) : FeM Output := do
      let mut output := output
      let mut r1 := r1
      let v18 ← Range_has_0 r1 v2
      if v18 then
        let v20 ← Range_at_0 r1 v2
        let i := v20
        let v23 ← checkedMul i (UInt256.ofNat 32)
        checkIndex i 1
        let (v26, v26_0) ← MemBuffer_write_word output v23 values[i.toNat]!
        output := v26_0
        let v28 ← Range_next_0 r1 v2
        loop output r1 v28
      else
        let v30 ← SUCCESS
        let v33 := (Output.mk v30 gas_used output)
        return v33
    partial_fixpoint
    loop output r1 v15
  | _ =>
    let v34 ← host_allocation gas_used
    match (← pure v6) with
    | .Some .. =>
      let _ ← MemBuffer_drop v6.Some_0
      pure ()
    | _ => pure ()
    return v34

/-- `Keccak256::new` -/
def Keccak256_new : FeM Keccak256 := do
  let v3 := (Keccak256.mk (Vector.replicate 25 (UInt256.ofNat 0)) (UInt256.ofNat 0))
  return v3

/-- `lib::permutation::ROUND_CONSTANTS` -/
def ROUND_CONSTANTS : FeM (Vector UInt256 24) := do
  let v24 := #v[(UInt256.ofNat 1), (UInt256.ofNat 32898), (UInt256.ofNat 9223372036854808714), (UInt256.ofNat 9223372039002292224), (UInt256.ofNat 32907), (UInt256.ofNat 2147483649), (UInt256.ofNat 9223372039002292353), (UInt256.ofNat 9223372036854808585), (UInt256.ofNat 138), (UInt256.ofNat 136), (UInt256.ofNat 2147516425), (UInt256.ofNat 2147483658), (UInt256.ofNat 2147516555), (UInt256.ofNat 9223372036854775947), (UInt256.ofNat 9223372036854808713), (UInt256.ofNat 9223372036854808579), (UInt256.ofNat 9223372036854808578), (UInt256.ofNat 9223372036854775936), (UInt256.ofNat 32778), (UInt256.ofNat 9223372039002259466), (UInt256.ofNat 9223372039002292353), (UInt256.ofNat 9223372036854808704), (UInt256.ofNat 2147483649), (UInt256.ofNat 9223372039002292232)]
  return v24

/-- `lib::permutation::rotate` -/
def rotate (value : UInt256) («by» : UInt256) : FeM UInt256 := do
  let v8 ← IntTy.sub (.u 64) (UInt256.ofNat 64) «by»
  return (IntTy.shl (.u 64) value «by») ||| (value >>> v8)

def keccak_round_k9 (b20 : UInt256) (b21 : UInt256) (b23 : UInt256) (b24 : UInt256) (v366 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v366 25
  e := (e.set! v366.toNat (b23 ^^^ ((IntTy.not (.u 64) b24) &&& b20)))
  let v371 := (UInt256.ofNat 24)
  checkIndex v371 25
  e := (e.set! v371.toNat (b24 ^^^ ((IntTy.not (.u 64) b20) &&& b21)))
  return ((), e)

def keccak_round_k8 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (v318 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v318 25
  let b20 ← rotate (a[v318.toNat]! ^^^ d2) (UInt256.ofNat 62)
  let v324 := (UInt256.ofNat 8)
  checkIndex v324 25
  let b21 ← rotate (a[v324.toNat]! ^^^ d3) (UInt256.ofNat 55)
  let v330 := (UInt256.ofNat 14)
  checkIndex v330 25
  let b22 ← rotate (a[v330.toNat]! ^^^ d4) (UInt256.ofNat 39)
  let v336 := (UInt256.ofNat 15)
  checkIndex v336 25
  let b23 ← rotate (a[v336.toNat]! ^^^ d0) (UInt256.ofNat 41)
  let v342 := (UInt256.ofNat 21)
  checkIndex v342 25
  let b24 ← rotate (a[v342.toNat]! ^^^ d1) (UInt256.ofNat 2)
  let v351 := (UInt256.ofNat 20)
  checkIndex v351 25
  e := (e.set! v351.toNat (b20 ^^^ ((IntTy.not (.u 64) b21) &&& b22)))
  let v356 := (UInt256.ofNat 21)
  checkIndex v356 25
  e := (e.set! v356.toNat (b21 ^^^ ((IntTy.not (.u 64) b22) &&& b23)))
  let v361 := (UInt256.ofNat 22)
  checkIndex v361 25
  e := (e.set! v361.toNat (b22 ^^^ ((IntTy.not (.u 64) b23) &&& b24)))
  let v366 := (UInt256.ofNat 23)
  keccak_round_k9 b20 b21 b23 b24 v366 e

def keccak_round_k7 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b15 : UInt256) (b16 : UInt256) (v275 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v275 25
  let b17 ← rotate (a[v275.toNat]! ^^^ d1) (UInt256.ofNat 10)
  let v281 := (UInt256.ofNat 17)
  checkIndex v281 25
  let b18 ← rotate (a[v281.toNat]! ^^^ d2) (UInt256.ofNat 15)
  let v287 := (UInt256.ofNat 23)
  checkIndex v287 25
  let b19 ← rotate (a[v287.toNat]! ^^^ d3) (UInt256.ofNat 56)
  let v296 := (UInt256.ofNat 15)
  checkIndex v296 25
  e := (e.set! v296.toNat (b15 ^^^ ((IntTy.not (.u 64) b16) &&& b17)))
  let v301 := (UInt256.ofNat 16)
  checkIndex v301 25
  e := (e.set! v301.toNat (b16 ^^^ ((IntTy.not (.u 64) b17) &&& b18)))
  let v306 := (UInt256.ofNat 17)
  checkIndex v306 25
  e := (e.set! v306.toNat (b17 ^^^ ((IntTy.not (.u 64) b18) &&& b19)))
  let v311 := (UInt256.ofNat 18)
  checkIndex v311 25
  e := (e.set! v311.toNat (b18 ^^^ ((IntTy.not (.u 64) b19) &&& b15)))
  let v316 := (UInt256.ofNat 19)
  checkIndex v316 25
  e := (e.set! v316.toNat (b19 ^^^ ((IntTy.not (.u 64) b15) &&& b16)))
  let v318 := (UInt256.ofNat 2)
  keccak_round_k8 a d0 d1 d2 d3 d4 v318 e

def keccak_round_k6 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b10 : UInt256) (b11 : UInt256) (b12 : UInt256) (b13 : UInt256) (v232 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v232 25
  let b14 ← rotate (a[v232.toNat]! ^^^ d0) (UInt256.ofNat 18)
  let v241 := (UInt256.ofNat 10)
  checkIndex v241 25
  e := (e.set! v241.toNat (b10 ^^^ ((IntTy.not (.u 64) b11) &&& b12)))
  let v246 := (UInt256.ofNat 11)
  checkIndex v246 25
  e := (e.set! v246.toNat (b11 ^^^ ((IntTy.not (.u 64) b12) &&& b13)))
  let v251 := (UInt256.ofNat 12)
  checkIndex v251 25
  e := (e.set! v251.toNat (b12 ^^^ ((IntTy.not (.u 64) b13) &&& b14)))
  let v256 := (UInt256.ofNat 13)
  checkIndex v256 25
  e := (e.set! v256.toNat (b13 ^^^ ((IntTy.not (.u 64) b14) &&& b10)))
  let v261 := (UInt256.ofNat 14)
  checkIndex v261 25
  e := (e.set! v261.toNat (b14 ^^^ ((IntTy.not (.u 64) b10) &&& b11)))
  let v263 := (UInt256.ofNat 4)
  checkIndex v263 25
  let b15 ← rotate (a[v263.toNat]! ^^^ d4) (UInt256.ofNat 27)
  let v269 := (UInt256.ofNat 5)
  checkIndex v269 25
  let b16 ← rotate (a[v269.toNat]! ^^^ d0) (UInt256.ofNat 36)
  let v275 := (UInt256.ofNat 11)
  keccak_round_k7 a d0 d1 d2 d3 d4 b15 b16 v275 e

def keccak_round_k5 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b5 : UInt256) (b6 : UInt256) (b7 : UInt256) (b8 : UInt256) (b9 : UInt256) (v191 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v191 25
  e := (e.set! v191.toNat (b6 ^^^ ((IntTy.not (.u 64) b7) &&& b8)))
  let v196 := (UInt256.ofNat 7)
  checkIndex v196 25
  e := (e.set! v196.toNat (b7 ^^^ ((IntTy.not (.u 64) b8) &&& b9)))
  let v201 := (UInt256.ofNat 8)
  checkIndex v201 25
  e := (e.set! v201.toNat (b8 ^^^ ((IntTy.not (.u 64) b9) &&& b5)))
  let v206 := (UInt256.ofNat 9)
  checkIndex v206 25
  e := (e.set! v206.toNat (b9 ^^^ ((IntTy.not (.u 64) b5) &&& b6)))
  let v208 := (UInt256.ofNat 1)
  checkIndex v208 25
  let b10 ← rotate (a[v208.toNat]! ^^^ d1) (UInt256.ofNat 1)
  let v214 := (UInt256.ofNat 7)
  checkIndex v214 25
  let b11 ← rotate (a[v214.toNat]! ^^^ d2) (UInt256.ofNat 6)
  let v220 := (UInt256.ofNat 13)
  checkIndex v220 25
  let b12 ← rotate (a[v220.toNat]! ^^^ d3) (UInt256.ofNat 25)
  let v226 := (UInt256.ofNat 19)
  checkIndex v226 25
  let b13 ← rotate (a[v226.toNat]! ^^^ d4) (UInt256.ofNat 8)
  let v232 := (UInt256.ofNat 20)
  keccak_round_k6 a d0 d1 d2 d3 d4 b10 b11 b12 b13 v232 e

def keccak_round_k4 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b0 : UInt256) (b1 : UInt256) (b3 : UInt256) (b4 : UInt256) (v146 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v146 25
  e := (e.set! v146.toNat (b3 ^^^ ((IntTy.not (.u 64) b4) &&& b0)))
  let v151 := (UInt256.ofNat 4)
  checkIndex v151 25
  e := (e.set! v151.toNat (b4 ^^^ ((IntTy.not (.u 64) b0) &&& b1)))
  let v153 := (UInt256.ofNat 3)
  checkIndex v153 25
  let b5 ← rotate (a[v153.toNat]! ^^^ d3) (UInt256.ofNat 28)
  let v159 := (UInt256.ofNat 9)
  checkIndex v159 25
  let b6 ← rotate (a[v159.toNat]! ^^^ d4) (UInt256.ofNat 20)
  let v165 := (UInt256.ofNat 10)
  checkIndex v165 25
  let b7 ← rotate (a[v165.toNat]! ^^^ d0) (UInt256.ofNat 3)
  let v171 := (UInt256.ofNat 16)
  checkIndex v171 25
  let b8 ← rotate (a[v171.toNat]! ^^^ d1) (UInt256.ofNat 45)
  let v177 := (UInt256.ofNat 22)
  checkIndex v177 25
  let b9 ← rotate (a[v177.toNat]! ^^^ d2) (UInt256.ofNat 61)
  let v186 := (UInt256.ofNat 5)
  checkIndex v186 25
  e := (e.set! v186.toNat (b5 ^^^ ((IntTy.not (.u 64) b6) &&& b7)))
  let v191 := (UInt256.ofNat 6)
  keccak_round_k5 a d0 d1 d2 d3 d4 b5 b6 b7 b8 b9 v191 e

def keccak_round_k3 (a : (Vector UInt256 25)) (constant : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (v98 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let b0 := a[v98.toNat]! ^^^ d0
  let v102 := (UInt256.ofNat 6)
  checkIndex v102 25
  let b1 ← rotate (a[v102.toNat]! ^^^ d1) (UInt256.ofNat 44)
  let v108 := (UInt256.ofNat 12)
  checkIndex v108 25
  let b2 ← rotate (a[v108.toNat]! ^^^ d2) (UInt256.ofNat 43)
  let v114 := (UInt256.ofNat 18)
  checkIndex v114 25
  let b3 ← rotate (a[v114.toNat]! ^^^ d3) (UInt256.ofNat 21)
  let v120 := (UInt256.ofNat 24)
  checkIndex v120 25
  let b4 ← rotate (a[v120.toNat]! ^^^ d4) (UInt256.ofNat 14)
  let v130 := (b0 ^^^ ((IntTy.not (.u 64) b1) &&& b2)) ^^^ constant
  let v131 := (UInt256.ofNat 0)
  checkIndex v131 25
  e := (e.set! v131.toNat v130)
  let v136 := (UInt256.ofNat 1)
  checkIndex v136 25
  e := (e.set! v136.toNat (b1 ^^^ ((IntTy.not (.u 64) b2) &&& b3)))
  let v141 := (UInt256.ofNat 2)
  checkIndex v141 25
  e := (e.set! v141.toNat (b2 ^^^ ((IntTy.not (.u 64) b3) &&& b4)))
  let v146 := (UInt256.ofNat 3)
  keccak_round_k4 a d0 d1 d2 d3 d4 b0 b1 b3 b4 v146 e

def keccak_round_k2 (a : (Vector UInt256 25)) (constant : UInt256) (c0 : UInt256) (c1 : UInt256) (c2 : UInt256) (c3 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v63 := (UInt256.ofNat 4)
  checkIndex v63 25
  let v65 := (UInt256.ofNat 9)
  checkIndex v65 25
  let v68 := (UInt256.ofNat 14)
  checkIndex v68 25
  let v70 := (a[v63.toNat]! ^^^ a[v65.toNat]!) ^^^ a[v68.toNat]!
  let v71 := (UInt256.ofNat 19)
  checkIndex v71 25
  let v74 := (UInt256.ofNat 24)
  checkIndex v74 25
  let c4 := (v70 ^^^ a[v71.toNat]!) ^^^ a[v74.toNat]!
  let v79 ← rotate c1 (UInt256.ofNat 1)
  let d0 := c4 ^^^ v79
  let v83 ← rotate c2 (UInt256.ofNat 1)
  let d1 := c0 ^^^ v83
  let v87 ← rotate c3 (UInt256.ofNat 1)
  let d2 := c1 ^^^ v87
  let v91 ← rotate c4 (UInt256.ofNat 1)
  let d3 := c2 ^^^ v91
  let v95 ← rotate c0 (UInt256.ofNat 1)
  let d4 := c3 ^^^ v95
  let v98 := (UInt256.ofNat 0)
  checkIndex v98 25
  keccak_round_k3 a constant d0 d1 d2 d3 d4 v98 e

def keccak_round_k1 (a : (Vector UInt256 25)) (constant : UInt256) (c0 : UInt256) (c1 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v33 := (UInt256.ofNat 2)
  checkIndex v33 25
  let v35 := (UInt256.ofNat 7)
  checkIndex v35 25
  let v38 := (UInt256.ofNat 12)
  checkIndex v38 25
  let v40 := (a[v33.toNat]! ^^^ a[v35.toNat]!) ^^^ a[v38.toNat]!
  let v41 := (UInt256.ofNat 17)
  checkIndex v41 25
  let v44 := (UInt256.ofNat 22)
  checkIndex v44 25
  let c2 := (v40 ^^^ a[v41.toNat]!) ^^^ a[v44.toNat]!
  let v48 := (UInt256.ofNat 3)
  checkIndex v48 25
  let v50 := (UInt256.ofNat 8)
  checkIndex v50 25
  let v53 := (UInt256.ofNat 13)
  checkIndex v53 25
  let v55 := (a[v48.toNat]! ^^^ a[v50.toNat]!) ^^^ a[v53.toNat]!
  let v56 := (UInt256.ofNat 18)
  checkIndex v56 25
  let v59 := (UInt256.ofNat 23)
  checkIndex v59 25
  let c3 := (v55 ^^^ a[v56.toNat]!) ^^^ a[v59.toNat]!
  keccak_round_k2 a constant c0 c1 c2 c3 e

/-- `lib::permutation::keccak_round` -/
def keccak_round (a : (Vector UInt256 25)) (e : (Vector UInt256 25)) (constant : UInt256) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v3 := (UInt256.ofNat 0)
  checkIndex v3 25
  let v5 := (UInt256.ofNat 5)
  checkIndex v5 25
  let v8 := (UInt256.ofNat 10)
  checkIndex v8 25
  let v11 := (UInt256.ofNat 15)
  checkIndex v11 25
  let v13 := ((a[v3.toNat]! ^^^ a[v5.toNat]!) ^^^ a[v8.toNat]!) ^^^ a[v11.toNat]!
  let v14 := (UInt256.ofNat 20)
  checkIndex v14 25
  let c0 := v13 ^^^ a[v14.toNat]!
  let v18 := (UInt256.ofNat 1)
  checkIndex v18 25
  let v20 := (UInt256.ofNat 6)
  checkIndex v20 25
  let v23 := (UInt256.ofNat 11)
  checkIndex v23 25
  let v25 := (a[v18.toNat]! ^^^ a[v20.toNat]!) ^^^ a[v23.toNat]!
  let v26 := (UInt256.ofNat 16)
  checkIndex v26 25
  let v29 := (UInt256.ofNat 21)
  checkIndex v29 25
  let c1 := (v25 ^^^ a[v26.toNat]!) ^^^ a[v29.toNat]!
  keccak_round_k1 a constant c0 c1 e

/-- `lib::permutation::permute` -/
def permute (state : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut other : (Vector UInt256 25) := default
  let mut round : UInt256 := default
  let mut r2 : (Vector UInt256 25) := default
  let mut state := state
  other := (Vector.replicate 25 (UInt256.ofNat 0))
  round := (UInt256.ofNat 0)
  let rec loop (other : (Vector UInt256 25)) (round : UInt256) (r2 : (Vector UInt256 25)) (state : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
    let mut other := other
    let mut round := round
    let mut r2 := r2
    let mut state := state
    if decide (round.toNat < (UInt256.ofNat 24).toNat) then
      let v13 ← checkedRem round (UInt256.ofNat 2)
      let join : (Vector UInt256 25) → UInt256 → (Vector UInt256 25) → (Vector UInt256 25) → FeM (Unit × (Vector UInt256 25)) := fun other round r2 state => do
        let mut other := other
        let mut round := round
        let mut r2 := r2
        let mut state := state
        let round_35 ← checkedAdd round (UInt256.ofNat 1)
        round := round_35
        loop other round r2 state
      if decide (v13 = (UInt256.ofNat 0)) then
        let v18 ← ROUND_CONSTANTS
        let round_19 := round
        checkIndex round_19 24
        let (v22, v22_0) ← keccak_round state other v18[round_19.toNat]!
        other := v22_0
        join other round r2 state
      else
        r2 := other
        let v27 ← ROUND_CONSTANTS
        let round_28 := round
        checkIndex round_28 24
        let (v31, v31_0) ← keccak_round r2 state v27[round_28.toNat]!
        state := v31_0
        join other round r2 state
    else
      return ((), state)
  partial_fixpoint
  loop other round r2 state

/-- `Keccak256::advance` -/
def Keccak256_advance (self : Keccak256) (bytes : UInt256) : FeM (Unit × Keccak256) := do
  let mut self := self
  let v4 ← checkedAdd self.position bytes
  self := { self with position := v4 }
  let join : Keccak256 → FeM (Unit × Keccak256) := fun self => do
    let mut self := self
    return ((), self)
  if decide (self.position = (UInt256.ofNat 136)) then
    let (v10, v10_0) ← permute self.lanes
    self := { self with lanes := v10_0 }
    self := { self with position := (UInt256.ofNat 0) }
    join self
  else
    join self

/-- `Keccak256::update_word` -/
def Keccak256_update_word (self : Keccak256) (value : UInt256) (len : UInt256) : FeM (Unit × Keccak256) := do
  let mut start : UInt256 := default
  let mut remaining : UInt256 := default
  let mut self := self
  let _ ← «assert» (decide (len.toNat ≤ (UInt256.ofNat 32).toNat))
  let join : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
    let mut start := start
    let mut remaining := remaining
    let mut self := self
    let v134 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 192))
    let v139 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 128))
    let v144 ← downcast_truncate_u256_u64 (value >>> (UInt256.ofNat 64))
    let v146 ← downcast_truncate_u256_u64 value
    let words := #v[(IntTy.swapBytes (.u 64) v134), (IntTy.swapBytes (.u 64) v139), (IntTy.swapBytes (.u 64) v144), (IntTy.swapBytes (.u 64) v146), (UInt256.ofNat 0)]
    let start_153 ← checkedSub (UInt256.ofNat 32) len
    start := start_153
    remaining := len
    let rec loop (start : UInt256) (remaining : UInt256) (self : Keccak256) : FeM (Unit × Keccak256) := do
      let mut start := start
      let mut remaining := remaining
      let mut self := self
      if decide ((UInt256.ofNat 0).toNat < remaining.toNat) then
        let offset ← checkedRem self.position (UInt256.ofNat 8)
        let v168 ← checkedSub (UInt256.ofNat 8) offset
        let join_1 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self take => do
          let mut start := start
          let mut remaining := remaining
          let mut self := self
          let v180 ← checkedRem start (UInt256.ofNat 8)
          let v182 ← checkedMul v180 (UInt256.ofNat 8)
          let shift_183 ← downcast_unchecked_usize_u64 v182
          let v188 ← checkedDiv start (UInt256.ofNat 8)
          checkIndex v188 5
          let low := words[v188.toNat]! >>> shift_183
          let join_2 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self high => do
            let mut start := start
            let mut remaining := remaining
            let mut self := self
            let all := (UInt256.ofNat 18446744073709551615)
            let join_3 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self mask => do
              let mut start := start
              let mut remaining := remaining
              let mut self := self
              let v224 ← checkedMul offset (UInt256.ofNat 8)
              let v225 ← downcast_unchecked_usize_u64 v224
              let v229 ← checkedDiv self.position (UInt256.ofNat 8)
              checkIndex v229 25
              let v231 := self.lanes[v229.toNat]! ^^^ (IntTy.shl (.u 64) ((low ||| high) &&& mask) v225)
              checkIndex v229 25
              self := { self with lanes := (self.lanes.set! v229.toNat v231) }
              let start_234 ← checkedAdd start take
              start := start_234
              let remaining_238 ← checkedSub remaining take
              remaining := remaining_238
              let (v241, v241_0) ← Keccak256_advance self take
              self := v241_0
              loop start remaining self
            if decide (take = (UInt256.ofNat 8)) then
              join_3 start remaining self all
            else
              let v215 ← checkedMul take (UInt256.ofNat 8)
              let v216 ← checkedSub (UInt256.ofNat 64) v215
              let v217 ← downcast_unchecked_usize_u64 v216
              join_3 start remaining self (all >>> v217)
          if decide (shift_183 = (UInt256.ofNat 0)) then
            join_2 start remaining self (UInt256.ofNat 0)
          else
            let v199 ← checkedDiv start (UInt256.ofNat 8)
            let v201 ← checkedAdd v199 (UInt256.ofNat 1)
            checkIndex v201 5
            let v204 ← IntTy.sub (.u 64) (UInt256.ofNat 64) shift_183
            join_2 start remaining self (IntTy.shl (.u 64) words[v201.toNat]! v204)
        if decide (remaining.toNat < v168.toNat) then
          join_1 start remaining self remaining
        else
          let v174 ← checkedSub (UInt256.ofNat 8) offset
          join_1 start remaining self v174
      else
        return ((), self)
    partial_fixpoint
    loop start remaining self
  if decide (len ≠ (UInt256.ofNat 0)) then
    let v15 ← checkedAdd self.position len
    if decide (v15.toNat ≤ (UInt256.ofNat 136).toNat) then
      let v21 ← checkedSub (UInt256.ofNat 32) len
      let v23 ← checkedMul v21 (UInt256.ofNat 8)
      let leading := value <<< v23
      let v29 ← downcast_truncate_u256_u64 (leading >>> (UInt256.ofNat 192))
      let w0 := IntTy.swapBytes (.u 64) v29
      let v34 ← downcast_truncate_u256_u64 (leading >>> (UInt256.ofNat 128))
      let w1 := IntTy.swapBytes (.u 64) v34
      let v39 ← downcast_truncate_u256_u64 (leading >>> (UInt256.ofNat 64))
      let w2 := IntTy.swapBytes (.u 64) v39
      let v42 ← downcast_truncate_u256_u64 leading
      let w3 := IntTy.swapBytes (.u 64) v42
      let lane ← checkedDiv self.position (UInt256.ofNat 8)
      let v51 ← checkedRem self.position (UInt256.ofNat 8)
      let v53 ← checkedMul v51 (UInt256.ofNat 8)
      let shift_54 ← downcast_unchecked_usize_u64 v53
      let join_4 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
        let mut start := start
        let mut remaining := remaining
        let mut self := self
        let v117 ← checkedAdd self.position len
        self := { self with position := v117 }
        let join_5 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
          let mut start := start
          let mut remaining := remaining
          let mut self := self
          return ((), self)
        if decide (self.position = (UInt256.ofNat 136)) then
          let (v123, v123_0) ← permute self.lanes
          self := { self with lanes := v123_0 }
          self := { self with position := (UInt256.ofNat 0) }
          join_5 start remaining self
        else
          join_5 start remaining self
      if decide (shift_54 = (UInt256.ofNat 0)) then
        checkIndex lane 25
        checkIndex lane 25
        self := { self with lanes := (self.lanes.set! lane.toNat (self.lanes[lane.toNat]! ^^^ w0)) }
        let v62 ← checkedAdd lane (UInt256.ofNat 1)
        checkIndex v62 25
        checkIndex v62 25
        self := { self with lanes := (self.lanes.set! v62.toNat (self.lanes[v62.toNat]! ^^^ w1)) }
        let v67 ← checkedAdd lane (UInt256.ofNat 2)
        checkIndex v67 25
        checkIndex v67 25
        self := { self with lanes := (self.lanes.set! v67.toNat (self.lanes[v67.toNat]! ^^^ w2)) }
        let v72 ← checkedAdd lane (UInt256.ofNat 3)
        checkIndex v72 25
        checkIndex v72 25
        self := { self with lanes := (self.lanes.set! v72.toNat (self.lanes[v72.toNat]! ^^^ w3)) }
        join_4 start remaining self
      else
        let back ← IntTy.sub (.u 64) (UInt256.ofNat 64) shift_54
        checkIndex lane 25
        let v82 := self.lanes[lane.toNat]! ^^^ (IntTy.shl (.u 64) w0 shift_54)
        checkIndex lane 25
        self := { self with lanes := (self.lanes.set! lane.toNat v82) }
        let v88 ← checkedAdd lane (UInt256.ofNat 1)
        checkIndex v88 25
        let v90 := self.lanes[v88.toNat]! ^^^ ((IntTy.shl (.u 64) w1 shift_54) ||| (w0 >>> back))
        checkIndex v88 25
        self := { self with lanes := (self.lanes.set! v88.toNat v90) }
        let v96 ← checkedAdd lane (UInt256.ofNat 2)
        checkIndex v96 25
        let v98 := self.lanes[v96.toNat]! ^^^ ((IntTy.shl (.u 64) w2 shift_54) ||| (w1 >>> back))
        checkIndex v96 25
        self := { self with lanes := (self.lanes.set! v96.toNat v98) }
        let v104 ← checkedAdd lane (UInt256.ofNat 3)
        checkIndex v104 25
        let v106 := self.lanes[v104.toNat]! ^^^ ((IntTy.shl (.u 64) w3 shift_54) ||| (w2 >>> back))
        checkIndex v104 25
        self := { self with lanes := (self.lanes.set! v104.toNat v106) }
        let v110 ← checkedAdd lane (UInt256.ofNat 4)
        checkIndex v110 25
        checkIndex v110 25
        self := { self with lanes := (self.lanes.set! v110.toNat (self.lanes[v110.toNat]! ^^^ (w3 >>> back))) }
        join_4 start remaining self
    else
      join start remaining self
  else
    join start remaining self

/-- `Keccak256::pad` -/
def Keccak256_pad (self : Keccak256) : FeM (Unit × Keccak256) := do
  let mut self := self
  let v4 ← checkedRem self.position (UInt256.ofNat 8)
  let v6 ← checkedMul v4 (UInt256.ofNat 8)
  let v7 ← downcast_unchecked_usize_u64 v6
  let v11 ← checkedDiv self.position (UInt256.ofNat 8)
  checkIndex v11 25
  let v13 := self.lanes[v11.toNat]! ^^^ (IntTy.shl (.u 64) (UInt256.ofNat 1) v7)
  checkIndex v11 25
  self := { self with lanes := (self.lanes.set! v11.toNat v13) }
  let v16 := (UInt256.ofNat 16)
  checkIndex v16 25
  let v18 := self.lanes[v16.toNat]! ^^^ (UInt256.ofNat 9223372036854775808)
  checkIndex v16 25
  self := { self with lanes := (self.lanes.set! v16.toNat v18) }
  let (v21, v21_0) ← permute self.lanes
  self := { self with lanes := v21_0 }
  return ((), self)

/-- `Keccak256::finish_word` -/
def Keccak256_finish_word (self : Keccak256) : FeM UInt256 := do
  let mut self := self
  let (v1, v1_0) ← Keccak256_pad self
  self := v1_0
  let v2 := (UInt256.ofNat 0)
  checkIndex v2 25
  let v7 := (IntTy.swapBytes (.u 64) self.lanes[v2.toNat]!) <<< (UInt256.ofNat 192)
  let v8 := (UInt256.ofNat 1)
  checkIndex v8 25
  let v13 := (IntTy.swapBytes (.u 64) self.lanes[v8.toNat]!) <<< (UInt256.ofNat 128)
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 25
  let v20 := (IntTy.swapBytes (.u 64) self.lanes[v15.toNat]!) <<< (UInt256.ofNat 64)
  let v22 := (UInt256.ofNat 3)
  checkIndex v22 25
  let v26 := ((v7 ||| v13) ||| v20) ||| (IntTy.swapBytes (.u 64) self.lanes[v22.toNat]!)
  return v26

/-- `lib::recover_signer` -/
def recover_signer (hash : UInt256) (recovery_id : UInt256) (r : UInt256) (s : UInt256) : FeM Option_u256 := do
  let mut digest : Keccak256 := default
  let v4 ← recover hash recovery_id r s
  match v4 with
  | .Some .. =>
    let key := v4.Some_0
    let digest_7 ← Keccak256_new
    digest := digest_7
    let (v12, v12_0) ← Keccak256_update_word digest key.x (UInt256.ofNat 32)
    digest := v12_0
    let (v16, v16_0) ← Keccak256_update_word digest key.y (UInt256.ofNat 32)
    digest := v16_0
    let v19 ← Keccak256_finish_word digest
    let v21 := v19 &&& (UInt256.ofNat 1461501637330902918203684832716283019655932542975)
    return Option_u256.Some v21
  | _ =>
    return Option_u256.None

/-- `lib::ecrecover<MemBuffer>` -/
def ecrecover_MemBuffer (input : MemBuffer) : FeM Output := do
  let mut r0 : (Vector UInt256 1) := default
  let hash ← word_at_MemBuffer input (UInt256.ofNat 0)
  let v_1 ← word_at_MemBuffer input (UInt256.ofNat 32)
  let r ← word_at_MemBuffer input (UInt256.ofNat 64)
  let s ← word_at_MemBuffer input (UInt256.ofNat 96)
  let join : (Vector UInt256 1) → Bool → FeM Output := fun r0 malformed => do
    let mut r0 := r0
    let join_1 : (Vector UInt256 1) → Option_u256 → FeM Output := fun r0 signer => do
      let mut r0 := r0
      let join_2 : (Vector UInt256 1) → Output → FeM Output := fun r0 v3 => do
        let mut r0 := r0
        return v3
      match signer with
      | .Some .. =>
        r0 := #v[signer.Some_0]
        let v45 ← words_1 r0 (UInt256.ofNat 3000)
        join_2 r0 v45
      | .None =>
        let v47 ← SUCCESS
        let v49 ← MemBuffer_empty
        let v50 := (Output.mk v47 (UInt256.ofNat 3000) v49)
        join_2 r0 v50
    if malformed then
      join_1 r0 Option_u256.None
    else
      let v36 ← checkedSub v_1 (UInt256.ofNat 27)
      let v37 ← recover_signer hash v36 r s
      join_1 r0 v37
  let join_3 : (Vector UInt256 1) → FeM Output := fun r0 => do
    let mut r0 := r0
    join r0 true
  let join_4 : (Vector UInt256 1) → FeM Output := fun r0 => do
    let mut r0 := r0
    if decide (r = (UInt256.ofNat 0)) then
      join_3 r0
    else
      let v22 ← N
      if decide (v22.toNat ≤ r.toNat) then
        join_3 r0
      else
        if decide (s = (UInt256.ofNat 0)) then
          join_3 r0
        else
          let v26 ← N
          if decide (v26.toNat ≤ s.toNat) then
            join_3 r0
          else
            join r0 false
  if decide (v_1 ≠ (UInt256.ofNat 27)) then
    if decide (v_1 ≠ (UInt256.ofNat 28)) then
      join_3 r0
    else
      join_4 r0
  else
    join_4 r0

/-- `_`, for a call no other arm takes -/
def recv_fallback : FeM Empty := do
  let mut result : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_4
  r2 := v0
  let result_2 ← ecrecover_MemBuffer r2
  result := result_2
  let _ ← MemBuffer_drop r2
  let v6 ← MemBuffer_span_ref result.output
  let v9 ← Evm_return_data v6
  nomatch v9

/-- The entry of `recv_fallback`, for a call no other arm takes -/
def recv_fallback_entry : FeM Empty := do
  nonPayable
  let _ ← recv_fallback
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `EcrecoverPrecompile`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv_fallback_entry
  else
    let selector ← calldataSelector
    recv_fallback_entry

/-- The deployment entry of contract `EcrecoverPrecompile`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


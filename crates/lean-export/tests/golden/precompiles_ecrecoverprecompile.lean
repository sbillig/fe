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
  let input_11 := input
  return input_11

/-- `lib::words<1>` -/
def words_1 (values : (Vector UInt256 1)) (gas_used : UInt256) : FeM Output := do
  let mut output : MemBuffer := default
  let mut r1 : Range_Known_0_Unknown := default
  let v5 ← checkedMul (UInt256.ofNat 1) (UInt256.ofNat 32)
  let v6 ← MemBuffer_try_new v5
  match (← pure v6) with
  | .Some .. =>
    let output_9 := v6.Some_0
    output := output_9
    let v13 := (Range_Known_0_Unknown.mk () (UInt256.ofNat 1))
    r1 := v13
    let v15 ← Range_start_0 r1
    let rec b5 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (v2 : UInt256) : FeM Output := do
      let mut output := output
      let mut r1 := r1
      let v18 ← Range_has_0 r1 v2
      if v18 then
        let v20 ← Range_at_0 r1 v2
        let i := v20
        let v23 ← checkedMul i (UInt256.ofNat 32)
        checkIndex i 1
        let v24 := values[i.toNat]!
        let (v26, v26_0) ← MemBuffer_write_word output v23 v24
        output := v26_0
        let v28 ← Range_next_0 r1 v2
        b5 output r1 v28
      else
        let v30 ← SUCCESS
        let output_32 := output
        let v33 := (Output.mk v30 gas_used output_32)
        return v33
    partial_fixpoint
    b5 output r1 v15
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
  let v1 := Vector.replicate 25 (UInt256.ofNat 0)
  let v3 := (Keccak256.mk v1 (UInt256.ofNat 0))
  return v3

/-- `lib::permutation::ROUND_CONSTANTS` -/
def ROUND_CONSTANTS : FeM (Vector UInt256 24) := do
  let v24 := #v[(UInt256.ofNat 1), (UInt256.ofNat 32898), (UInt256.ofNat 9223372036854808714), (UInt256.ofNat 9223372039002292224), (UInt256.ofNat 32907), (UInt256.ofNat 2147483649), (UInt256.ofNat 9223372039002292353), (UInt256.ofNat 9223372036854808585), (UInt256.ofNat 138), (UInt256.ofNat 136), (UInt256.ofNat 2147516425), (UInt256.ofNat 2147483658), (UInt256.ofNat 2147516555), (UInt256.ofNat 9223372036854775947), (UInt256.ofNat 9223372036854808713), (UInt256.ofNat 9223372036854808579), (UInt256.ofNat 9223372036854808578), (UInt256.ofNat 9223372036854775936), (UInt256.ofNat 32778), (UInt256.ofNat 9223372039002259466), (UInt256.ofNat 9223372039002292353), (UInt256.ofNat 9223372036854808704), (UInt256.ofNat 2147483649), (UInt256.ofNat 9223372039002292232)]
  return v24

/-- `lib::permutation::rotate` -/
def rotate (value : UInt256) («by» : UInt256) : FeM UInt256 := do
  let v4 := IntTy.shl (.u 64) value «by»
  let v8 ← IntTy.sub (.u 64) (UInt256.ofNat 64) «by»
  let v9 := value >>> v8
  let v10 := v4 ||| v9
  return v10

def keccak_round_k15 (b20 : UInt256) (b21 : UInt256) (b22 : UInt256) (b23 : UInt256) (b24 : UInt256) (v355 : UInt256) (v356 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v356 25
  e := (e.set! v356.toNat v355)
  let v358 := IntTy.not (.u 64) b23
  let v359 := v358 &&& b24
  let v360 := b22 ^^^ v359
  let v361 := (UInt256.ofNat 22)
  checkIndex v361 25
  e := (e.set! v361.toNat v360)
  let v363 := IntTy.not (.u 64) b24
  let v364 := v363 &&& b20
  let v365 := b23 ^^^ v364
  let v366 := (UInt256.ofNat 23)
  checkIndex v366 25
  e := (e.set! v366.toNat v365)
  let v368 := IntTy.not (.u 64) b20
  let v369 := v368 &&& b21
  let v370 := b24 ^^^ v369
  let v371 := (UInt256.ofNat 24)
  checkIndex v371 25
  e := (e.set! v371.toNat v370)
  return ((), e)

def keccak_round_k14 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d4 : UInt256) (b20 : UInt256) (b21 : UInt256) (v330 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v330 25
  let v331 := a[v330.toNat]!
  let v332 := v331 ^^^ d4
  let b22 ← rotate v332 (UInt256.ofNat 39)
  let v336 := (UInt256.ofNat 15)
  checkIndex v336 25
  let v337 := a[v336.toNat]!
  let v338 := v337 ^^^ d0
  let b23 ← rotate v338 (UInt256.ofNat 41)
  let v342 := (UInt256.ofNat 21)
  checkIndex v342 25
  let v343 := a[v342.toNat]!
  let v344 := v343 ^^^ d1
  let b24 ← rotate v344 (UInt256.ofNat 2)
  let v348 := IntTy.not (.u 64) b21
  let v349 := v348 &&& b22
  let v350 := b20 ^^^ v349
  let v351 := (UInt256.ofNat 20)
  checkIndex v351 25
  e := (e.set! v351.toNat v350)
  let v353 := IntTy.not (.u 64) b22
  let v354 := v353 &&& b23
  let v355 := b21 ^^^ v354
  let v356 := (UInt256.ofNat 21)
  keccak_round_k15 b20 b21 b22 b23 b24 v355 v356 e

def keccak_round_k13 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b15 : UInt256) (b16 : UInt256) (b18 : UInt256) (b19 : UInt256) (v305 : UInt256) (v306 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  e := (e.set! v306.toNat v305)
  let v308 := IntTy.not (.u 64) b19
  let v309 := v308 &&& b15
  let v310 := b18 ^^^ v309
  let v311 := (UInt256.ofNat 18)
  checkIndex v311 25
  e := (e.set! v311.toNat v310)
  let v313 := IntTy.not (.u 64) b15
  let v314 := v313 &&& b16
  let v315 := b19 ^^^ v314
  let v316 := (UInt256.ofNat 19)
  checkIndex v316 25
  e := (e.set! v316.toNat v315)
  let v318 := (UInt256.ofNat 2)
  checkIndex v318 25
  let v319 := a[v318.toNat]!
  let v320 := v319 ^^^ d2
  let b20 ← rotate v320 (UInt256.ofNat 62)
  let v324 := (UInt256.ofNat 8)
  checkIndex v324 25
  let v325 := a[v324.toNat]!
  let v326 := v325 ^^^ d3
  let b21 ← rotate v326 (UInt256.ofNat 55)
  let v330 := (UInt256.ofNat 14)
  keccak_round_k14 a d0 d1 d4 b20 b21 v330 e

def keccak_round_k12 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b15 : UInt256) (b16 : UInt256) (b17 : UInt256) (v282 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v283 := v282 ^^^ d2
  let b18 ← rotate v283 (UInt256.ofNat 15)
  let v287 := (UInt256.ofNat 23)
  checkIndex v287 25
  let v288 := a[v287.toNat]!
  let v289 := v288 ^^^ d3
  let b19 ← rotate v289 (UInt256.ofNat 56)
  let v293 := IntTy.not (.u 64) b16
  let v294 := v293 &&& b17
  let v295 := b15 ^^^ v294
  let v296 := (UInt256.ofNat 15)
  checkIndex v296 25
  e := (e.set! v296.toNat v295)
  let v298 := IntTy.not (.u 64) b17
  let v299 := v298 &&& b18
  let v300 := b16 ^^^ v299
  let v301 := (UInt256.ofNat 16)
  checkIndex v301 25
  e := (e.set! v301.toNat v300)
  let v303 := IntTy.not (.u 64) b18
  let v304 := v303 &&& b19
  let v305 := b17 ^^^ v304
  let v306 := (UInt256.ofNat 17)
  checkIndex v306 25
  keccak_round_k13 a d0 d1 d2 d3 d4 b15 b16 b18 b19 v305 v306 e

def keccak_round_k11 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b10 : UInt256) (b11 : UInt256) (b14 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v258 := IntTy.not (.u 64) b10
  let v259 := v258 &&& b11
  let v260 := b14 ^^^ v259
  let v261 := (UInt256.ofNat 14)
  checkIndex v261 25
  e := (e.set! v261.toNat v260)
  let v263 := (UInt256.ofNat 4)
  checkIndex v263 25
  let v264 := a[v263.toNat]!
  let v265 := v264 ^^^ d4
  let b15 ← rotate v265 (UInt256.ofNat 27)
  let v269 := (UInt256.ofNat 5)
  checkIndex v269 25
  let v270 := a[v269.toNat]!
  let v271 := v270 ^^^ d0
  let b16 ← rotate v271 (UInt256.ofNat 36)
  let v275 := (UInt256.ofNat 11)
  checkIndex v275 25
  let v276 := a[v275.toNat]!
  let v277 := v276 ^^^ d1
  let b17 ← rotate v277 (UInt256.ofNat 10)
  let v281 := (UInt256.ofNat 17)
  checkIndex v281 25
  let v282 := a[v281.toNat]!
  keccak_round_k12 a d0 d1 d2 d3 d4 b15 b16 b17 v282 e

def keccak_round_k10 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b10 : UInt256) (b11 : UInt256) (b12 : UInt256) (b13 : UInt256) (b14 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v238 := IntTy.not (.u 64) b11
  let v239 := v238 &&& b12
  let v240 := b10 ^^^ v239
  let v241 := (UInt256.ofNat 10)
  checkIndex v241 25
  e := (e.set! v241.toNat v240)
  let v243 := IntTy.not (.u 64) b12
  let v244 := v243 &&& b13
  let v245 := b11 ^^^ v244
  let v246 := (UInt256.ofNat 11)
  checkIndex v246 25
  e := (e.set! v246.toNat v245)
  let v248 := IntTy.not (.u 64) b13
  let v249 := v248 &&& b14
  let v250 := b12 ^^^ v249
  let v251 := (UInt256.ofNat 12)
  checkIndex v251 25
  e := (e.set! v251.toNat v250)
  let v253 := IntTy.not (.u 64) b14
  let v254 := v253 &&& b10
  let v255 := b13 ^^^ v254
  let v256 := (UInt256.ofNat 13)
  checkIndex v256 25
  e := (e.set! v256.toNat v255)
  keccak_round_k11 a d0 d1 d2 d3 d4 b10 b11 b14 e

def keccak_round_k9 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (v208 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v208 25
  let v209 := a[v208.toNat]!
  let v210 := v209 ^^^ d1
  let b10 ← rotate v210 (UInt256.ofNat 1)
  let v214 := (UInt256.ofNat 7)
  checkIndex v214 25
  let v215 := a[v214.toNat]!
  let v216 := v215 ^^^ d2
  let b11 ← rotate v216 (UInt256.ofNat 6)
  let v220 := (UInt256.ofNat 13)
  checkIndex v220 25
  let v221 := a[v220.toNat]!
  let v222 := v221 ^^^ d3
  let b12 ← rotate v222 (UInt256.ofNat 25)
  let v226 := (UInt256.ofNat 19)
  checkIndex v226 25
  let v227 := a[v226.toNat]!
  let v228 := v227 ^^^ d4
  let b13 ← rotate v228 (UInt256.ofNat 8)
  let v232 := (UInt256.ofNat 20)
  checkIndex v232 25
  let v233 := a[v232.toNat]!
  let v234 := v233 ^^^ d0
  let b14 ← rotate v234 (UInt256.ofNat 18)
  keccak_round_k10 a d0 d1 d2 d3 d4 b10 b11 b12 b13 b14 e

def keccak_round_k8 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b5 : UInt256) (b6 : UInt256) (b7 : UInt256) (b8 : UInt256) (b9 : UInt256) (v188 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v189 := v188 &&& b8
  let v190 := b6 ^^^ v189
  let v191 := (UInt256.ofNat 6)
  checkIndex v191 25
  e := (e.set! v191.toNat v190)
  let v193 := IntTy.not (.u 64) b8
  let v194 := v193 &&& b9
  let v195 := b7 ^^^ v194
  let v196 := (UInt256.ofNat 7)
  checkIndex v196 25
  e := (e.set! v196.toNat v195)
  let v198 := IntTy.not (.u 64) b9
  let v199 := v198 &&& b5
  let v200 := b8 ^^^ v199
  let v201 := (UInt256.ofNat 8)
  checkIndex v201 25
  e := (e.set! v201.toNat v200)
  let v203 := IntTy.not (.u 64) b5
  let v204 := v203 &&& b6
  let v205 := b9 ^^^ v204
  let v206 := (UInt256.ofNat 9)
  checkIndex v206 25
  e := (e.set! v206.toNat v205)
  let v208 := (UInt256.ofNat 1)
  keccak_round_k9 a d0 d1 d2 d3 d4 v208 e

def keccak_round_k7 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b5 : UInt256) (v160 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v161 := v160 ^^^ d4
  let b6 ← rotate v161 (UInt256.ofNat 20)
  let v165 := (UInt256.ofNat 10)
  checkIndex v165 25
  let v166 := a[v165.toNat]!
  let v167 := v166 ^^^ d0
  let b7 ← rotate v167 (UInt256.ofNat 3)
  let v171 := (UInt256.ofNat 16)
  checkIndex v171 25
  let v172 := a[v171.toNat]!
  let v173 := v172 ^^^ d1
  let b8 ← rotate v173 (UInt256.ofNat 45)
  let v177 := (UInt256.ofNat 22)
  checkIndex v177 25
  let v178 := a[v177.toNat]!
  let v179 := v178 ^^^ d2
  let b9 ← rotate v179 (UInt256.ofNat 61)
  let v183 := IntTy.not (.u 64) b6
  let v184 := v183 &&& b7
  let v185 := b5 ^^^ v184
  let v186 := (UInt256.ofNat 5)
  checkIndex v186 25
  e := (e.set! v186.toNat v185)
  let v188 := IntTy.not (.u 64) b7
  keccak_round_k8 a d0 d1 d2 d3 d4 b5 b6 b7 b8 b9 v188 e

def keccak_round_k6 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b0 : UInt256) (b1 : UInt256) (b2 : UInt256) (b3 : UInt256) (b4 : UInt256) (v139 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v140 := b2 ^^^ v139
  let v141 := (UInt256.ofNat 2)
  checkIndex v141 25
  e := (e.set! v141.toNat v140)
  let v143 := IntTy.not (.u 64) b4
  let v144 := v143 &&& b0
  let v145 := b3 ^^^ v144
  let v146 := (UInt256.ofNat 3)
  checkIndex v146 25
  e := (e.set! v146.toNat v145)
  let v148 := IntTy.not (.u 64) b0
  let v149 := v148 &&& b1
  let v150 := b4 ^^^ v149
  let v151 := (UInt256.ofNat 4)
  checkIndex v151 25
  e := (e.set! v151.toNat v150)
  let v153 := (UInt256.ofNat 3)
  checkIndex v153 25
  let v154 := a[v153.toNat]!
  let v155 := v154 ^^^ d3
  let b5 ← rotate v155 (UInt256.ofNat 28)
  let v159 := (UInt256.ofNat 9)
  checkIndex v159 25
  let v160 := a[v159.toNat]!
  keccak_round_k7 a d0 d1 d2 d3 d4 b5 v160 e

def keccak_round_k5 (a : (Vector UInt256 25)) (constant : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b0 : UInt256) (b1 : UInt256) (b2 : UInt256) (v114 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v114 25
  let v115 := a[v114.toNat]!
  let v116 := v115 ^^^ d3
  let b3 ← rotate v116 (UInt256.ofNat 21)
  let v120 := (UInt256.ofNat 24)
  checkIndex v120 25
  let v121 := a[v120.toNat]!
  let v122 := v121 ^^^ d4
  let b4 ← rotate v122 (UInt256.ofNat 14)
  let v126 := IntTy.not (.u 64) b1
  let v127 := v126 &&& b2
  let v128 := b0 ^^^ v127
  let v130 := v128 ^^^ constant
  let v131 := (UInt256.ofNat 0)
  checkIndex v131 25
  e := (e.set! v131.toNat v130)
  let v133 := IntTy.not (.u 64) b2
  let v134 := v133 &&& b3
  let v135 := b1 ^^^ v134
  let v136 := (UInt256.ofNat 1)
  checkIndex v136 25
  e := (e.set! v136.toNat v135)
  let v138 := IntTy.not (.u 64) b3
  let v139 := v138 &&& b4
  keccak_round_k6 a d0 d1 d2 d3 d4 b0 b1 b2 b3 b4 v139 e

def keccak_round_k4 (a : (Vector UInt256 25)) (constant : UInt256) (c0 : UInt256) (c1 : UInt256) (c2 : UInt256) (c3 : UInt256) (c4 : UInt256) (v79 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
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
  let v99 := a[v98.toNat]!
  let b0 := v99 ^^^ d0
  let v102 := (UInt256.ofNat 6)
  checkIndex v102 25
  let v103 := a[v102.toNat]!
  let v104 := v103 ^^^ d1
  let b1 ← rotate v104 (UInt256.ofNat 44)
  let v108 := (UInt256.ofNat 12)
  checkIndex v108 25
  let v109 := a[v108.toNat]!
  let v110 := v109 ^^^ d2
  let b2 ← rotate v110 (UInt256.ofNat 43)
  let v114 := (UInt256.ofNat 18)
  keccak_round_k5 a constant d0 d1 d2 d3 d4 b0 b1 b2 v114 e

def keccak_round_k3 (a : (Vector UInt256 25)) (constant : UInt256) (c0 : UInt256) (c1 : UInt256) (c2 : UInt256) (v58 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v59 := (UInt256.ofNat 23)
  checkIndex v59 25
  let v60 := a[v59.toNat]!
  let c3 := v58 ^^^ v60
  let v63 := (UInt256.ofNat 4)
  checkIndex v63 25
  let v64 := a[v63.toNat]!
  let v65 := (UInt256.ofNat 9)
  checkIndex v65 25
  let v66 := a[v65.toNat]!
  let v67 := v64 ^^^ v66
  let v68 := (UInt256.ofNat 14)
  checkIndex v68 25
  let v69 := a[v68.toNat]!
  let v70 := v67 ^^^ v69
  let v71 := (UInt256.ofNat 19)
  checkIndex v71 25
  let v72 := a[v71.toNat]!
  let v73 := v70 ^^^ v72
  let v74 := (UInt256.ofNat 24)
  checkIndex v74 25
  let v75 := a[v74.toNat]!
  let c4 := v73 ^^^ v75
  let v79 ← rotate c1 (UInt256.ofNat 1)
  keccak_round_k4 a constant c0 c1 c2 c3 c4 v79 e

def keccak_round_k2 (a : (Vector UInt256 25)) (constant : UInt256) (c0 : UInt256) (c1 : UInt256) (v37 : UInt256) (v39 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v40 := v37 ^^^ v39
  let v41 := (UInt256.ofNat 17)
  checkIndex v41 25
  let v42 := a[v41.toNat]!
  let v43 := v40 ^^^ v42
  let v44 := (UInt256.ofNat 22)
  checkIndex v44 25
  let v45 := a[v44.toNat]!
  let c2 := v43 ^^^ v45
  let v48 := (UInt256.ofNat 3)
  checkIndex v48 25
  let v49 := a[v48.toNat]!
  let v50 := (UInt256.ofNat 8)
  checkIndex v50 25
  let v51 := a[v50.toNat]!
  let v52 := v49 ^^^ v51
  let v53 := (UInt256.ofNat 13)
  checkIndex v53 25
  let v54 := a[v53.toNat]!
  let v55 := v52 ^^^ v54
  let v56 := (UInt256.ofNat 18)
  checkIndex v56 25
  let v57 := a[v56.toNat]!
  let v58 := v55 ^^^ v57
  keccak_round_k3 a constant c0 c1 c2 v58 e

def keccak_round_k1 (a : (Vector UInt256 25)) (constant : UInt256) (c0 : UInt256) (v19 : UInt256) (v20 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v21 := a[v20.toNat]!
  let v22 := v19 ^^^ v21
  let v23 := (UInt256.ofNat 11)
  checkIndex v23 25
  let v24 := a[v23.toNat]!
  let v25 := v22 ^^^ v24
  let v26 := (UInt256.ofNat 16)
  checkIndex v26 25
  let v27 := a[v26.toNat]!
  let v28 := v25 ^^^ v27
  let v29 := (UInt256.ofNat 21)
  checkIndex v29 25
  let v30 := a[v29.toNat]!
  let c1 := v28 ^^^ v30
  let v33 := (UInt256.ofNat 2)
  checkIndex v33 25
  let v34 := a[v33.toNat]!
  let v35 := (UInt256.ofNat 7)
  checkIndex v35 25
  let v36 := a[v35.toNat]!
  let v37 := v34 ^^^ v36
  let v38 := (UInt256.ofNat 12)
  checkIndex v38 25
  let v39 := a[v38.toNat]!
  keccak_round_k2 a constant c0 c1 v37 v39 e

/-- `lib::permutation::keccak_round` -/
def keccak_round (a : (Vector UInt256 25)) (e : (Vector UInt256 25)) (constant : UInt256) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v3 := (UInt256.ofNat 0)
  checkIndex v3 25
  let v4 := a[v3.toNat]!
  let v5 := (UInt256.ofNat 5)
  checkIndex v5 25
  let v6 := a[v5.toNat]!
  let v7 := v4 ^^^ v6
  let v8 := (UInt256.ofNat 10)
  checkIndex v8 25
  let v9 := a[v8.toNat]!
  let v10 := v7 ^^^ v9
  let v11 := (UInt256.ofNat 15)
  checkIndex v11 25
  let v12 := a[v11.toNat]!
  let v13 := v10 ^^^ v12
  let v14 := (UInt256.ofNat 20)
  checkIndex v14 25
  let v15 := a[v14.toNat]!
  let c0 := v13 ^^^ v15
  let v18 := (UInt256.ofNat 1)
  checkIndex v18 25
  let v19 := a[v18.toNat]!
  let v20 := (UInt256.ofNat 6)
  checkIndex v20 25
  keccak_round_k1 a constant c0 v19 v20 e

/-- `lib::permutation::permute` -/
def permute (state : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut other : (Vector UInt256 25) := default
  let mut round : UInt256 := default
  let mut r2 : (Vector UInt256 25) := default
  let mut state := state
  let other_2 := Vector.replicate 25 (UInt256.ofNat 0)
  other := other_2
  round := (UInt256.ofNat 0)
  let rec b1 (other : (Vector UInt256 25)) (round : UInt256) (r2 : (Vector UInt256 25)) (state : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
    let mut other := other
    let mut round := round
    let mut r2 := r2
    let mut state := state
    let round_6 := round
    let v9 := decide (round_6.toNat < (UInt256.ofNat 24).toNat)
    if v9 then
      let round_10 := round
      let v13 ← checkedRem round_10 (UInt256.ofNat 2)
      let v15 := decide (v13 = (UInt256.ofNat 0))
      let b6 : (Vector UInt256 25) → UInt256 → (Vector UInt256 25) → (Vector UInt256 25) → FeM (Unit × (Vector UInt256 25)) := fun other round r2 state => do
        let mut other := other
        let mut round := round
        let mut r2 := r2
        let mut state := state
        let round_34 := round
        let round_35 ← checkedAdd round_34 (UInt256.ofNat 1)
        round := round_35
        b1 other round r2 state
      if v15 then
        let v18 ← ROUND_CONSTANTS
        let round_19 := round
        checkIndex round_19 24
        let v20 := v18[round_19.toNat]!
        let (v22, v22_0) ← keccak_round state other v20
        other := v22_0
        b6 other round r2 state
      else
        let other_24 := other
        r2 := other_24
        let v27 ← ROUND_CONSTANTS
        let round_28 := round
        checkIndex round_28 24
        let v29 := v27[round_28.toNat]!
        let (v31, v31_0) ← keccak_round r2 state v29
        state := v31_0
        b6 other round r2 state
    else
      return ((), state)
  partial_fixpoint
  b1 other round r2 state

/-- `Keccak256::advance` -/
def Keccak256_advance (self : Keccak256) (bytes : UInt256) : FeM (Unit × Keccak256) := do
  let mut self := self
  let v3 := self.position
  let v4 ← checkedAdd v3 bytes
  self := { self with position := v4 }
  let v6 := self.position
  let v8 := decide (v6 = (UInt256.ofNat 136))
  let b3 : Keccak256 → FeM (Unit × Keccak256) := fun self => do
    let mut self := self
    return ((), self)
  if v8 then
    let (v10, v10_0) ← permute self.lanes
    self := { self with lanes := v10_0 }
    self := { self with position := (UInt256.ofNat 0) }
    b3 self
  else
    b3 self

/-- `Keccak256::update_word` -/
def Keccak256_update_word (self : Keccak256) (value : UInt256) (len : UInt256) : FeM (Unit × Keccak256) := do
  let mut start : UInt256 := default
  let mut remaining : UInt256 := default
  let mut self := self
  let v8 := decide (len.toNat ≤ (UInt256.ofNat 32).toNat)
  let _ ← «assert» v8
  let v12 := decide (len ≠ (UInt256.ofNat 0))
  let b2 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
    let mut start := start
    let mut remaining := remaining
    let mut self := self
    let v133 := value >>> (UInt256.ofNat 192)
    let v134 ← downcast_truncate_u256_u64 v133
    let v135 := IntTy.swapBytes (.u 64) v134
    let v138 := value >>> (UInt256.ofNat 128)
    let v139 ← downcast_truncate_u256_u64 v138
    let v140 := IntTy.swapBytes (.u 64) v139
    let v143 := value >>> (UInt256.ofNat 64)
    let v144 ← downcast_truncate_u256_u64 v143
    let v145 := IntTy.swapBytes (.u 64) v144
    let v146 ← downcast_truncate_u256_u64 value
    let v147 := IntTy.swapBytes (.u 64) v146
    let words := #v[v135, v140, v145, v147, (UInt256.ofNat 0)]
    let start_153 ← checkedSub (UInt256.ofNat 32) len
    start := start_153
    remaining := len
    let rec b11 (start : UInt256) (remaining : UInt256) (self : Keccak256) : FeM (Unit × Keccak256) := do
      let mut start := start
      let mut remaining := remaining
      let mut self := self
      let remaining_157 := remaining
      let v160 := decide ((UInt256.ofNat 0).toNat < remaining_157.toNat)
      if v160 then
        let v161 := self.position
        let offset ← checkedRem v161 (UInt256.ofNat 8)
        let remaining_165 := remaining
        let v168 ← checkedSub (UInt256.ofNat 8) offset
        let v169 := decide (remaining_165.toNat < v168.toNat)
        let b16 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self take => do
          let mut start := start
          let mut remaining := remaining
          let mut self := self
          let start_177 := start
          let v180 ← checkedRem start_177 (UInt256.ofNat 8)
          let v182 ← checkedMul v180 (UInt256.ofNat 8)
          let shift_183 ← downcast_unchecked_usize_u64 v182
          let start_185 := start
          let v188 ← checkedDiv start_185 (UInt256.ofNat 8)
          checkIndex v188 5
          let v189 := words[v188.toNat]!
          let low := v189 >>> shift_183
          let v193 := decide (shift_183 = (UInt256.ofNat 0))
          let b19 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self high => do
            let mut start := start
            let mut remaining := remaining
            let mut self := self
            let all := (UInt256.ofNat 18446744073709551615)
            let v211 := decide (take = (UInt256.ofNat 8))
            let b22 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self mask => do
              let mut start := start
              let mut remaining := remaining
              let mut self := self
              let v221 := low ||| high
              let v222 := v221 &&& mask
              let v224 ← checkedMul offset (UInt256.ofNat 8)
              let v225 ← downcast_unchecked_usize_u64 v224
              let v226 := IntTy.shl (.u 64) v222 v225
              let v227 := self.position
              let v229 ← checkedDiv v227 (UInt256.ofNat 8)
              checkIndex v229 25
              let v230 := self.lanes[v229.toNat]!
              let v231 := v230 ^^^ v226
              checkIndex v229 25
              self := { self with lanes := (self.lanes.set! v229.toNat v231) }
              let start_233 := start
              let start_234 ← checkedAdd start_233 take
              start := start_234
              let remaining_237 := remaining
              let remaining_238 ← checkedSub remaining_237 take
              remaining := remaining_238
              let (v241, v241_0) ← Keccak256_advance self take
              self := v241_0
              b11 start remaining self
            if v211 then
              b22 start remaining self all
            else
              let v215 ← checkedMul take (UInt256.ofNat 8)
              let v216 ← checkedSub (UInt256.ofNat 64) v215
              let v217 ← downcast_unchecked_usize_u64 v216
              let v218 := all >>> v217
              b22 start remaining self v218
          if v193 then
            b19 start remaining self (UInt256.ofNat 0)
          else
            let start_196 := start
            let v199 ← checkedDiv start_196 (UInt256.ofNat 8)
            let v201 ← checkedAdd v199 (UInt256.ofNat 1)
            checkIndex v201 5
            let v202 := words[v201.toNat]!
            let v204 ← IntTy.sub (.u 64) (UInt256.ofNat 64) shift_183
            let v205 := IntTy.shl (.u 64) v202 v204
            b19 start remaining self v205
        if v169 then
          let remaining_170 := remaining
          b16 start remaining self remaining_170
        else
          let v174 ← checkedSub (UInt256.ofNat 8) offset
          b16 start remaining self v174
      else
        return ((), self)
    partial_fixpoint
    b11 start remaining self
  if v12 then
    let v13 := self.position
    let v15 ← checkedAdd v13 len
    let v17 := decide (v15.toNat ≤ (UInt256.ofNat 136).toNat)
    if v17 then
      let v21 ← checkedSub (UInt256.ofNat 32) len
      let v23 ← checkedMul v21 (UInt256.ofNat 8)
      let v24 := v23
      let leading := value <<< v24
      let v28 := leading >>> (UInt256.ofNat 192)
      let v29 ← downcast_truncate_u256_u64 v28
      let w0 := IntTy.swapBytes (.u 64) v29
      let v33 := leading >>> (UInt256.ofNat 128)
      let v34 ← downcast_truncate_u256_u64 v33
      let w1 := IntTy.swapBytes (.u 64) v34
      let v38 := leading >>> (UInt256.ofNat 64)
      let v39 ← downcast_truncate_u256_u64 v38
      let w2 := IntTy.swapBytes (.u 64) v39
      let v42 ← downcast_truncate_u256_u64 leading
      let w3 := IntTy.swapBytes (.u 64) v42
      let v45 := self.position
      let lane ← checkedDiv v45 (UInt256.ofNat 8)
      let v49 := self.position
      let v51 ← checkedRem v49 (UInt256.ofNat 8)
      let v53 ← checkedMul v51 (UInt256.ofNat 8)
      let shift_54 ← downcast_unchecked_usize_u64 v53
      let v57 := decide (shift_54 = (UInt256.ofNat 0))
      let b7 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
        let mut start := start
        let mut remaining := remaining
        let mut self := self
        let v116 := self.position
        let v117 ← checkedAdd v116 len
        self := { self with position := v117 }
        let v119 := self.position
        let v121 := decide (v119 = (UInt256.ofNat 136))
        let b10 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
          let mut start := start
          let mut remaining := remaining
          let mut self := self
          return ((), self)
        if v121 then
          let (v123, v123_0) ← permute self.lanes
          self := { self with lanes := v123_0 }
          self := { self with position := (UInt256.ofNat 0) }
          b10 start remaining self
        else
          b10 start remaining self
      if v57 then
        checkIndex lane 25
        let v58 := self.lanes[lane.toNat]!
        let v59 := v58 ^^^ w0
        checkIndex lane 25
        self := { self with lanes := (self.lanes.set! lane.toNat v59) }
        let v62 ← checkedAdd lane (UInt256.ofNat 1)
        checkIndex v62 25
        let v63 := self.lanes[v62.toNat]!
        let v64 := v63 ^^^ w1
        checkIndex v62 25
        self := { self with lanes := (self.lanes.set! v62.toNat v64) }
        let v67 ← checkedAdd lane (UInt256.ofNat 2)
        checkIndex v67 25
        let v68 := self.lanes[v67.toNat]!
        let v69 := v68 ^^^ w2
        checkIndex v67 25
        self := { self with lanes := (self.lanes.set! v67.toNat v69) }
        let v72 ← checkedAdd lane (UInt256.ofNat 3)
        checkIndex v72 25
        let v73 := self.lanes[v72.toNat]!
        let v74 := v73 ^^^ w3
        checkIndex v72 25
        self := { self with lanes := (self.lanes.set! v72.toNat v74) }
        b7 start remaining self
      else
        let back ← IntTy.sub (.u 64) (UInt256.ofNat 64) shift_54
        let v80 := IntTy.shl (.u 64) w0 shift_54
        checkIndex lane 25
        let v81 := self.lanes[lane.toNat]!
        let v82 := v81 ^^^ v80
        checkIndex lane 25
        self := { self with lanes := (self.lanes.set! lane.toNat v82) }
        let v84 := IntTy.shl (.u 64) w1 shift_54
        let v85 := w0 >>> back
        let v86 := v84 ||| v85
        let v88 ← checkedAdd lane (UInt256.ofNat 1)
        checkIndex v88 25
        let v89 := self.lanes[v88.toNat]!
        let v90 := v89 ^^^ v86
        checkIndex v88 25
        self := { self with lanes := (self.lanes.set! v88.toNat v90) }
        let v92 := IntTy.shl (.u 64) w2 shift_54
        let v93 := w1 >>> back
        let v94 := v92 ||| v93
        let v96 ← checkedAdd lane (UInt256.ofNat 2)
        checkIndex v96 25
        let v97 := self.lanes[v96.toNat]!
        let v98 := v97 ^^^ v94
        checkIndex v96 25
        self := { self with lanes := (self.lanes.set! v96.toNat v98) }
        let v100 := IntTy.shl (.u 64) w3 shift_54
        let v101 := w2 >>> back
        let v102 := v100 ||| v101
        let v104 ← checkedAdd lane (UInt256.ofNat 3)
        checkIndex v104 25
        let v105 := self.lanes[v104.toNat]!
        let v106 := v105 ^^^ v102
        checkIndex v104 25
        self := { self with lanes := (self.lanes.set! v104.toNat v106) }
        let v108 := w3 >>> back
        let v110 ← checkedAdd lane (UInt256.ofNat 4)
        checkIndex v110 25
        let v111 := self.lanes[v110.toNat]!
        let v112 := v111 ^^^ v108
        checkIndex v110 25
        self := { self with lanes := (self.lanes.set! v110.toNat v112) }
        b7 start remaining self
    else
      b2 start remaining self
  else
    b2 start remaining self

/-- `Keccak256::pad` -/
def Keccak256_pad (self : Keccak256) : FeM (Unit × Keccak256) := do
  let mut self := self
  let v2 := self.position
  let v4 ← checkedRem v2 (UInt256.ofNat 8)
  let v6 ← checkedMul v4 (UInt256.ofNat 8)
  let v7 ← downcast_unchecked_usize_u64 v6
  let v8 := IntTy.shl (.u 64) (UInt256.ofNat 1) v7
  let v9 := self.position
  let v11 ← checkedDiv v9 (UInt256.ofNat 8)
  checkIndex v11 25
  let v12 := self.lanes[v11.toNat]!
  let v13 := v12 ^^^ v8
  checkIndex v11 25
  self := { self with lanes := (self.lanes.set! v11.toNat v13) }
  let v16 := (UInt256.ofNat 16)
  checkIndex v16 25
  let v17 := self.lanes[v16.toNat]!
  let v18 := v17 ^^^ (UInt256.ofNat 9223372036854775808)
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
  let v3 := self.lanes[v2.toNat]!
  let v4 := IntTy.swapBytes (.u 64) v3
  let v5 := v4
  let v7 := v5 <<< (UInt256.ofNat 192)
  let v8 := (UInt256.ofNat 1)
  checkIndex v8 25
  let v9 := self.lanes[v8.toNat]!
  let v10 := IntTy.swapBytes (.u 64) v9
  let v11 := v10
  let v13 := v11 <<< (UInt256.ofNat 128)
  let v14 := v7 ||| v13
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 25
  let v16 := self.lanes[v15.toNat]!
  let v17 := IntTy.swapBytes (.u 64) v16
  let v18 := v17
  let v20 := v18 <<< (UInt256.ofNat 64)
  let v21 := v14 ||| v20
  let v22 := (UInt256.ofNat 3)
  checkIndex v22 25
  let v23 := self.lanes[v22.toNat]!
  let v24 := IntTy.swapBytes (.u 64) v23
  let v25 := v24
  let v26 := v21 ||| v25
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
    let v9 := key.x
    let (v12, v12_0) ← Keccak256_update_word digest v9 (UInt256.ofNat 32)
    digest := v12_0
    let v13 := key.y
    let (v16, v16_0) ← Keccak256_update_word digest v13 (UInt256.ofNat 32)
    digest := v16_0
    let digest_17 := digest
    let v19 ← Keccak256_finish_word digest_17
    let v21 := v19 &&& (UInt256.ofNat 1461501637330902918203684832716283019655932542975)
    let v22 := Option_u256.Some v21
    return v22
  | _ =>
    let v23 := Option_u256.None
    return v23

/-- `lib::ecrecover<MemBuffer>` -/
def ecrecover_MemBuffer (input : MemBuffer) : FeM Output := do
  let mut r0 : (Vector UInt256 1) := default
  let hash ← word_at_MemBuffer input (UInt256.ofNat 0)
  let v_1 ← word_at_MemBuffer input (UInt256.ofNat 32)
  let r ← word_at_MemBuffer input (UInt256.ofNat 64)
  let s ← word_at_MemBuffer input (UInt256.ofNat 96)
  let v17 := decide (v_1 ≠ (UInt256.ofNat 27))
  let b3 : (Vector UInt256 1) → Bool → FeM Output := fun r0 malformed => do
    let mut r0 := r0
    let b11 : (Vector UInt256 1) → Option_u256 → FeM Output := fun r0 signer => do
      let mut r0 := r0
      let b12 : (Vector UInt256 1) → Output → FeM Output := fun r0 v3 => do
        let mut r0 := r0
        return v3
      match signer with
      | .Some .. =>
        let address := signer.Some_0
        let v42 := #v[address]
        r0 := v42
        let v45 ← words_1 r0 (UInt256.ofNat 3000)
        b12 r0 v45
      | .None =>
        let v47 ← SUCCESS
        let v49 ← MemBuffer_empty
        let v50 := (Output.mk v47 (UInt256.ofNat 3000) v49)
        b12 r0 v50
    if malformed then
      let v33 := Option_u256.None
      b11 r0 v33
    else
      let v36 ← checkedSub v_1 (UInt256.ofNat 27)
      let v37 ← recover_signer hash v36 r s
      b11 r0 v37
  let b1 : (Vector UInt256 1) → FeM Output := fun r0 => do
    let mut r0 := r0
    b3 r0 true
  let b7 : (Vector UInt256 1) → FeM Output := fun r0 => do
    let mut r0 := r0
    let v21 := decide (r = (UInt256.ofNat 0))
    if v21 then
      b1 r0
    else
      let v22 ← N
      let v23 := decide (v22.toNat ≤ r.toNat)
      if v23 then
        b1 r0
      else
        let v25 := decide (s = (UInt256.ofNat 0))
        if v25 then
          b1 r0
        else
          let v26 ← N
          let v27 := decide (v26.toNat ≤ s.toNat)
          if v27 then
            b1 r0
          else
            b3 r0 false
  if v17 then
    let v19 := decide (v_1 ≠ (UInt256.ofNat 28))
    if v19 then
      b1 r0
    else
      b7 r0
  else
    b7 r0

/-- `_`, for a call no other arm takes -/
def recv0_arm0 : FeM Empty := do
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

/-- The entry of `recv0_arm0`, for a call no other arm takes -/
def recv0_arm0_entry : FeM Empty := do
  nonPayable
  let _ ← recv0_arm0
  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)

/-- The runtime entry of contract `EcrecoverPrecompile`: the entry of the arm the call's selector picks -/
def runtime : FeM Empty := do
  let size ← calldatasize
  if size.toNat < 4 then
    recv0_arm0_entry
  else
    let selector ← calldataSelector
    recv0_arm0_entry

/-- The deployment entry of contract `EcrecoverPrecompile`: `init` on the arguments after the creation code -/
def init_entry (args : ByteArray) : FeM ByteArray := do
  nonPayable
  let _ ← init
  return ByteArray.empty


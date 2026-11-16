export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data MemBuffer MemBuffer_drop MemBuffer_empty MemBuffer_new MemBuffer_span_ref MemBuffer_try_new MemBuffer_write_word Option_u256 Option_u256.None Option_u256.Some Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_at_0 Range_has_0 Range_next_0 Range_start_0 downcast_truncate_u256_u64 downcast_unchecked_u256_usize downcast_unchecked_usize_u64 «assert»)
export Shared (N Output Output.mk SUCCESS host_allocation recover word_at_MemBuffer)

structure Keccak256 where
  lanes : (Vector UInt256 25)
  position : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def input_Evm_4 : FeM MemBuffer := do
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

def words_1 (values : (Vector UInt256 1)) (gas_used_1 : UInt256) : FeM Output := do
  let mut output : MemBuffer := default
  let mut r1 : Range_Known_0_Unknown := default
  let v3 := (UInt256.ofNat 1)
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedMul v3 v4
  let v6 ← MemBuffer_try_new v5
  let v7 := v6
  let v8 := v7
  match (← pure v8) with
  | .Some .. =>
    let output_9 := v7.Some_0
    output := output_9
    let v10 := (UInt256.ofNat 0)
    let v11 := (UInt256.ofNat 1)
    let v12 := ()
    let v13 := (Range_Known_0_Unknown.mk v12 v11)
    r1 := v13
    let v15 ← Range_start_0 r1
    let v16 := v15
    let rec b5 (output : MemBuffer) (r1 : Range_Known_0_Unknown) (v2 : UInt256) : FeM Output := do
      let mut output := output
      let mut r1 := r1
      let v18 ← Range_has_0 r1 v2
      if v18 then
        let v20 ← Range_at_0 r1 v2
        let i := v20
        let v22 := (UInt256.ofNat 32)
        let v23 ← checkedMul i v22
        checkIndex i 1
        let v24 := values[i.toNat]!
        let (v26, v26_0) ← MemBuffer_write_word output v23 v24
        output := v26_0
        let v28 ← Range_next_0 r1 v2
        let v29 := v28
        b5 output r1 v29
      else
        let v30 ← SUCCESS
        let gas_used_31 := gas_used_1
        let output_32 := output
        let v33 := (Output.mk v30 gas_used_31 output_32)
        return v33
    partial_fixpoint
    b5 output r1 v16
  | _ =>
    let v34 ← host_allocation gas_used_1
    match (← pure v7) with
    | .Some .. =>
      let _ ← MemBuffer_drop v7.Some_0
      pure ()
    | _ => pure ()
    return v34

def Keccak256_new : FeM Keccak256 := do
  let v0 := (UInt256.ofNat 0)
  let v1 := Vector.replicate 25 v0
  let v2 := (UInt256.ofNat 0)
  let v3 := (Keccak256.mk v1 v2)
  return v3

def ROUND_CONSTANTS : FeM (Vector UInt256 24) := do
  let v0 := (UInt256.ofNat 1)
  let v1 := (UInt256.ofNat 32898)
  let v2 := (UInt256.ofNat 9223372036854808714)
  let v3 := (UInt256.ofNat 9223372039002292224)
  let v4 := (UInt256.ofNat 32907)
  let v5 := (UInt256.ofNat 2147483649)
  let v6 := (UInt256.ofNat 9223372039002292353)
  let v7 := (UInt256.ofNat 9223372036854808585)
  let v8 := (UInt256.ofNat 138)
  let v9 := (UInt256.ofNat 136)
  let v10 := (UInt256.ofNat 2147516425)
  let v11 := (UInt256.ofNat 2147483658)
  let v12 := (UInt256.ofNat 2147516555)
  let v13 := (UInt256.ofNat 9223372036854775947)
  let v14 := (UInt256.ofNat 9223372036854808713)
  let v15 := (UInt256.ofNat 9223372036854808579)
  let v16 := (UInt256.ofNat 9223372036854808578)
  let v17 := (UInt256.ofNat 9223372036854775936)
  let v18 := (UInt256.ofNat 32778)
  let v19 := (UInt256.ofNat 9223372039002259466)
  let v20 := (UInt256.ofNat 9223372039002292353)
  let v21 := (UInt256.ofNat 9223372036854808704)
  let v22 := (UInt256.ofNat 2147483649)
  let v23 := (UInt256.ofNat 9223372039002292232)
  let v24 := #v[v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23]
  return v24

def rotate (value_0 : UInt256) (by_1 : UInt256) : FeM UInt256 := do
  let value_2 := value_0
  let by_3 := by_1
  let v4 := IntTy.shl (.u 64) value_2 by_3
  let value_5 := value_0
  let v6 := (UInt256.ofNat 64)
  let by_7 := by_1
  let v8 ← IntTy.sub (.u 64) v6 by_7
  let v9 := value_5 >>> v8
  let v10 := v4 ||| v9
  return v10

def keccak_round_k19 (b20 : UInt256) (b21 : UInt256) (b23 : UInt256) (b24 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v363 := IntTy.not (.u 64) b24
  let v364 := v363 &&& b20
  let v365 := b23 ^^^ v364
  let v366 := (UInt256.ofNat 23)
  checkIndex v366 25
  e := (e.set! v366.toNat v365)
  let v367 := ()
  let v368 := IntTy.not (.u 64) b20
  let v369 := v368 &&& b21
  let v370 := b24 ^^^ v369
  let v371 := (UInt256.ofNat 24)
  checkIndex v371 25
  e := (e.set! v371.toNat v370)
  let v372 := ()
  return ((), e)

def keccak_round_k18 (b20 : UInt256) (b21 : UInt256) (b22 : UInt256) (b23 : UInt256) (v344 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v345 := (UInt256.ofNat 2)
  let v346 ← rotate v344 v345
  let b24 := v346
  let v348 := IntTy.not (.u 64) b21
  let v349 := v348 &&& b22
  let v350 := b20 ^^^ v349
  let v351 := (UInt256.ofNat 20)
  checkIndex v351 25
  e := (e.set! v351.toNat v350)
  let v352 := ()
  let v353 := IntTy.not (.u 64) b22
  let v354 := v353 &&& b23
  let v355 := b21 ^^^ v354
  let v356 := (UInt256.ofNat 21)
  checkIndex v356 25
  e := (e.set! v356.toNat v355)
  let v357 := ()
  let v358 := IntTy.not (.u 64) b23
  let v359 := v358 &&& b24
  let v360 := b22 ^^^ v359
  let v361 := (UInt256.ofNat 22)
  checkIndex v361 25
  e := (e.set! v361.toNat v360)
  let v362 := ()
  keccak_round_k19 b20 b21 b23 b24 e

def keccak_round_k17 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d3 : UInt256) (d4 : UInt256) (b20 : UInt256) (v324 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v324 25
  let v325 := a[v324.toNat]!
  let v326 := v325 ^^^ d3
  let v327 := (UInt256.ofNat 55)
  let v328 ← rotate v326 v327
  let b21 := v328
  let v330 := (UInt256.ofNat 14)
  checkIndex v330 25
  let v331 := a[v330.toNat]!
  let v332 := v331 ^^^ d4
  let v333 := (UInt256.ofNat 39)
  let v334 ← rotate v332 v333
  let b22 := v334
  let v336 := (UInt256.ofNat 15)
  checkIndex v336 25
  let v337 := a[v336.toNat]!
  let v338 := v337 ^^^ d0
  let v339 := (UInt256.ofNat 41)
  let v340 ← rotate v338 v339
  let b23 := v340
  let v342 := (UInt256.ofNat 21)
  checkIndex v342 25
  let v343 := a[v342.toNat]!
  let v344 := v343 ^^^ d1
  keccak_round_k18 b20 b21 b22 b23 v344 e

def keccak_round_k16 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b15 : UInt256) (b16 : UInt256) (b18 : UInt256) (b19 : UInt256) (v305 : UInt256) (v306 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  e := (e.set! v306.toNat v305)
  let v307 := ()
  let v308 := IntTy.not (.u 64) b19
  let v309 := v308 &&& b15
  let v310 := b18 ^^^ v309
  let v311 := (UInt256.ofNat 18)
  checkIndex v311 25
  e := (e.set! v311.toNat v310)
  let v312 := ()
  let v313 := IntTy.not (.u 64) b15
  let v314 := v313 &&& b16
  let v315 := b19 ^^^ v314
  let v316 := (UInt256.ofNat 19)
  checkIndex v316 25
  e := (e.set! v316.toNat v315)
  let v317 := ()
  let v318 := (UInt256.ofNat 2)
  checkIndex v318 25
  let v319 := a[v318.toNat]!
  let v320 := v319 ^^^ d2
  let v321 := (UInt256.ofNat 62)
  let v322 ← rotate v320 v321
  let b20 := v322
  let v324 := (UInt256.ofNat 8)
  keccak_round_k17 a d0 d1 d3 d4 b20 v324 e

def keccak_round_k15 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b15 : UInt256) (b16 : UInt256) (b17 : UInt256) (b18 : UInt256) (v287 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v288 := a[v287.toNat]!
  let v289 := v288 ^^^ d3
  let v290 := (UInt256.ofNat 56)
  let v291 ← rotate v289 v290
  let b19 := v291
  let v293 := IntTy.not (.u 64) b16
  let v294 := v293 &&& b17
  let v295 := b15 ^^^ v294
  let v296 := (UInt256.ofNat 15)
  checkIndex v296 25
  e := (e.set! v296.toNat v295)
  let v297 := ()
  let v298 := IntTy.not (.u 64) b17
  let v299 := v298 &&& b18
  let v300 := b16 ^^^ v299
  let v301 := (UInt256.ofNat 16)
  checkIndex v301 25
  e := (e.set! v301.toNat v300)
  let v302 := ()
  let v303 := IntTy.not (.u 64) b18
  let v304 := v303 &&& b19
  let v305 := b17 ^^^ v304
  let v306 := (UInt256.ofNat 17)
  checkIndex v306 25
  keccak_round_k16 a d0 d1 d2 d3 d4 b15 b16 b18 b19 v305 v306 e

def keccak_round_k14 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (v267 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let b15 := v267
  let v269 := (UInt256.ofNat 5)
  checkIndex v269 25
  let v270 := a[v269.toNat]!
  let v271 := v270 ^^^ d0
  let v272 := (UInt256.ofNat 36)
  let v273 ← rotate v271 v272
  let b16 := v273
  let v275 := (UInt256.ofNat 11)
  checkIndex v275 25
  let v276 := a[v275.toNat]!
  let v277 := v276 ^^^ d1
  let v278 := (UInt256.ofNat 10)
  let v279 ← rotate v277 v278
  let b17 := v279
  let v281 := (UInt256.ofNat 17)
  checkIndex v281 25
  let v282 := a[v281.toNat]!
  let v283 := v282 ^^^ d2
  let v284 := (UInt256.ofNat 15)
  let v285 ← rotate v283 v284
  let b18 := v285
  let v287 := (UInt256.ofNat 23)
  checkIndex v287 25
  keccak_round_k15 a d0 d1 d2 d3 d4 b15 b16 b17 b18 v287 e

def keccak_round_k13 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b10 : UInt256) (b11 : UInt256) (b13 : UInt256) (b14 : UInt256) (v250 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v251 := (UInt256.ofNat 12)
  checkIndex v251 25
  e := (e.set! v251.toNat v250)
  let v252 := ()
  let v253 := IntTy.not (.u 64) b14
  let v254 := v253 &&& b10
  let v255 := b13 ^^^ v254
  let v256 := (UInt256.ofNat 13)
  checkIndex v256 25
  e := (e.set! v256.toNat v255)
  let v257 := ()
  let v258 := IntTy.not (.u 64) b10
  let v259 := v258 &&& b11
  let v260 := b14 ^^^ v259
  let v261 := (UInt256.ofNat 14)
  checkIndex v261 25
  e := (e.set! v261.toNat v260)
  let v262 := ()
  let v263 := (UInt256.ofNat 4)
  checkIndex v263 25
  let v264 := a[v263.toNat]!
  let v265 := v264 ^^^ d4
  let v266 := (UInt256.ofNat 27)
  let v267 ← rotate v265 v266
  keccak_round_k14 a d0 d1 d2 d3 d4 v267 e

def keccak_round_k12 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b10 : UInt256) (b11 : UInt256) (b12 : UInt256) (b13 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v232 := (UInt256.ofNat 20)
  checkIndex v232 25
  let v233 := a[v232.toNat]!
  let v234 := v233 ^^^ d0
  let v235 := (UInt256.ofNat 18)
  let v236 ← rotate v234 v235
  let b14 := v236
  let v238 := IntTy.not (.u 64) b11
  let v239 := v238 &&& b12
  let v240 := b10 ^^^ v239
  let v241 := (UInt256.ofNat 10)
  checkIndex v241 25
  e := (e.set! v241.toNat v240)
  let v242 := ()
  let v243 := IntTy.not (.u 64) b12
  let v244 := v243 &&& b13
  let v245 := b11 ^^^ v244
  let v246 := (UInt256.ofNat 11)
  checkIndex v246 25
  e := (e.set! v246.toNat v245)
  let v247 := ()
  let v248 := IntTy.not (.u 64) b13
  let v249 := v248 &&& b14
  let v250 := b12 ^^^ v249
  keccak_round_k13 a d0 d1 d2 d3 d4 b10 b11 b13 b14 v250 e

def keccak_round_k11 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (v210 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v211 := (UInt256.ofNat 1)
  let v212 ← rotate v210 v211
  let b10 := v212
  let v214 := (UInt256.ofNat 7)
  checkIndex v214 25
  let v215 := a[v214.toNat]!
  let v216 := v215 ^^^ d2
  let v217 := (UInt256.ofNat 6)
  let v218 ← rotate v216 v217
  let b11 := v218
  let v220 := (UInt256.ofNat 13)
  checkIndex v220 25
  let v221 := a[v220.toNat]!
  let v222 := v221 ^^^ d3
  let v223 := (UInt256.ofNat 25)
  let v224 ← rotate v222 v223
  let b12 := v224
  let v226 := (UInt256.ofNat 19)
  checkIndex v226 25
  let v227 := a[v226.toNat]!
  let v228 := v227 ^^^ d4
  let v229 := (UInt256.ofNat 8)
  let v230 ← rotate v228 v229
  let b13 := v230
  keccak_round_k12 a d0 d1 d2 d3 d4 b10 b11 b12 b13 e

def keccak_round_k10 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b5 : UInt256) (b6 : UInt256) (b7 : UInt256) (b8 : UInt256) (b9 : UInt256) (v193 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v194 := v193 &&& b9
  let v195 := b7 ^^^ v194
  let v196 := (UInt256.ofNat 7)
  checkIndex v196 25
  e := (e.set! v196.toNat v195)
  let v197 := ()
  let v198 := IntTy.not (.u 64) b9
  let v199 := v198 &&& b5
  let v200 := b8 ^^^ v199
  let v201 := (UInt256.ofNat 8)
  checkIndex v201 25
  e := (e.set! v201.toNat v200)
  let v202 := ()
  let v203 := IntTy.not (.u 64) b5
  let v204 := v203 &&& b6
  let v205 := b9 ^^^ v204
  let v206 := (UInt256.ofNat 9)
  checkIndex v206 25
  e := (e.set! v206.toNat v205)
  let v207 := ()
  let v208 := (UInt256.ofNat 1)
  checkIndex v208 25
  let v209 := a[v208.toNat]!
  let v210 := v209 ^^^ d1
  keccak_round_k11 a d0 d1 d2 d3 d4 v210 e

def keccak_round_k9 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b5 : UInt256) (b6 : UInt256) (b7 : UInt256) (v173 : UInt256) (v174 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v175 ← rotate v173 v174
  let b8 := v175
  let v177 := (UInt256.ofNat 22)
  checkIndex v177 25
  let v178 := a[v177.toNat]!
  let v179 := v178 ^^^ d2
  let v180 := (UInt256.ofNat 61)
  let v181 ← rotate v179 v180
  let b9 := v181
  let v183 := IntTy.not (.u 64) b6
  let v184 := v183 &&& b7
  let v185 := b5 ^^^ v184
  let v186 := (UInt256.ofNat 5)
  checkIndex v186 25
  e := (e.set! v186.toNat v185)
  let v187 := ()
  let v188 := IntTy.not (.u 64) b7
  let v189 := v188 &&& b8
  let v190 := b6 ^^^ v189
  let v191 := (UInt256.ofNat 6)
  checkIndex v191 25
  e := (e.set! v191.toNat v190)
  let v192 := ()
  let v193 := IntTy.not (.u 64) b8
  keccak_round_k10 a d0 d1 d2 d3 d4 b5 b6 b7 b8 b9 v193 e

def keccak_round_k8 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (v153 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v154 := a[v153.toNat]!
  let v155 := v154 ^^^ d3
  let v156 := (UInt256.ofNat 28)
  let v157 ← rotate v155 v156
  let b5 := v157
  let v159 := (UInt256.ofNat 9)
  checkIndex v159 25
  let v160 := a[v159.toNat]!
  let v161 := v160 ^^^ d4
  let v162 := (UInt256.ofNat 20)
  let v163 ← rotate v161 v162
  let b6 := v163
  let v165 := (UInt256.ofNat 10)
  checkIndex v165 25
  let v166 := a[v165.toNat]!
  let v167 := v166 ^^^ d0
  let v168 := (UInt256.ofNat 3)
  let v169 ← rotate v167 v168
  let b7 := v169
  let v171 := (UInt256.ofNat 16)
  checkIndex v171 25
  let v172 := a[v171.toNat]!
  let v173 := v172 ^^^ d1
  let v174 := (UInt256.ofNat 45)
  keccak_round_k9 a d0 d1 d2 d3 d4 b5 b6 b7 v173 v174 e

def keccak_round_k7 (a : (Vector UInt256 25)) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b0 : UInt256) (b1 : UInt256) (b2 : UInt256) (b3 : UInt256) (b4 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v137 := ()
  let v138 := IntTy.not (.u 64) b3
  let v139 := v138 &&& b4
  let v140 := b2 ^^^ v139
  let v141 := (UInt256.ofNat 2)
  checkIndex v141 25
  e := (e.set! v141.toNat v140)
  let v142 := ()
  let v143 := IntTy.not (.u 64) b4
  let v144 := v143 &&& b0
  let v145 := b3 ^^^ v144
  let v146 := (UInt256.ofNat 3)
  checkIndex v146 25
  e := (e.set! v146.toNat v145)
  let v147 := ()
  let v148 := IntTy.not (.u 64) b0
  let v149 := v148 &&& b1
  let v150 := b4 ^^^ v149
  let v151 := (UInt256.ofNat 4)
  checkIndex v151 25
  e := (e.set! v151.toNat v150)
  let v152 := ()
  let v153 := (UInt256.ofNat 3)
  checkIndex v153 25
  keccak_round_k8 a d0 d1 d2 d3 d4 v153 e

def keccak_round_k6 (a : (Vector UInt256 25)) (constant_2 : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (b0 : UInt256) (b1 : UInt256) (b2 : UInt256) (v116 : UInt256) (v117 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v118 ← rotate v116 v117
  let b3 := v118
  let v120 := (UInt256.ofNat 24)
  checkIndex v120 25
  let v121 := a[v120.toNat]!
  let v122 := v121 ^^^ d4
  let v123 := (UInt256.ofNat 14)
  let v124 ← rotate v122 v123
  let b4 := v124
  let v126 := IntTy.not (.u 64) b1
  let v127 := v126 &&& b2
  let v128 := b0 ^^^ v127
  let constant_129 := constant_2
  let v130 := v128 ^^^ constant_129
  let v131 := (UInt256.ofNat 0)
  checkIndex v131 25
  e := (e.set! v131.toNat v130)
  let v132 := ()
  let v133 := IntTy.not (.u 64) b2
  let v134 := v133 &&& b3
  let v135 := b1 ^^^ v134
  let v136 := (UInt256.ofNat 1)
  checkIndex v136 25
  e := (e.set! v136.toNat v135)
  keccak_round_k7 a d0 d1 d2 d3 d4 b0 b1 b2 b3 b4 e

def keccak_round_k5 (a : (Vector UInt256 25)) (constant_2 : UInt256) (d0 : UInt256) (d1 : UInt256) (d2 : UInt256) (d3 : UInt256) (d4 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  let v98 := (UInt256.ofNat 0)
  checkIndex v98 25
  let v99 := a[v98.toNat]!
  let v100 := v99 ^^^ d0
  let b0 := v100
  let v102 := (UInt256.ofNat 6)
  checkIndex v102 25
  let v103 := a[v102.toNat]!
  let v104 := v103 ^^^ d1
  let v105 := (UInt256.ofNat 44)
  let v106 ← rotate v104 v105
  let b1 := v106
  let v108 := (UInt256.ofNat 12)
  checkIndex v108 25
  let v109 := a[v108.toNat]!
  let v110 := v109 ^^^ d2
  let v111 := (UInt256.ofNat 43)
  let v112 ← rotate v110 v111
  let b2 := v112
  let v114 := (UInt256.ofNat 18)
  checkIndex v114 25
  let v115 := a[v114.toNat]!
  let v116 := v115 ^^^ d3
  let v117 := (UInt256.ofNat 21)
  keccak_round_k6 a constant_2 d0 d1 d2 d3 d4 b0 b1 b2 v116 v117 e

def keccak_round_k4 (a : (Vector UInt256 25)) (constant_2 : UInt256) (c0 : UInt256) (c1 : UInt256) (c2 : UInt256) (c3 : UInt256) (v73 : UInt256) (v74 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v74 25
  let v75 := a[v74.toNat]!
  let v76 := v73 ^^^ v75
  let c4 := v76
  let v78 := (UInt256.ofNat 1)
  let v79 ← rotate c1 v78
  let v80 := c4 ^^^ v79
  let d0 := v80
  let v82 := (UInt256.ofNat 1)
  let v83 ← rotate c2 v82
  let v84 := c0 ^^^ v83
  let d1 := v84
  let v86 := (UInt256.ofNat 1)
  let v87 ← rotate c3 v86
  let v88 := c1 ^^^ v87
  let d2 := v88
  let v90 := (UInt256.ofNat 1)
  let v91 ← rotate c4 v90
  let v92 := c2 ^^^ v91
  let d3 := v92
  let v94 := (UInt256.ofNat 1)
  let v95 ← rotate c0 v94
  let v96 := c3 ^^^ v95
  let d4 := v96
  keccak_round_k5 a constant_2 d0 d1 d2 d3 d4 e

def keccak_round_k3 (a : (Vector UInt256 25)) (constant_2 : UInt256) (c0 : UInt256) (c1 : UInt256) (c2 : UInt256) (v55 : UInt256) (v56 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v56 25
  let v57 := a[v56.toNat]!
  let v58 := v55 ^^^ v57
  let v59 := (UInt256.ofNat 23)
  checkIndex v59 25
  let v60 := a[v59.toNat]!
  let v61 := v58 ^^^ v60
  let c3 := v61
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
  keccak_round_k4 a constant_2 c0 c1 c2 c3 v73 v74 e

def keccak_round_k2 (a : (Vector UInt256 25)) (constant_2 : UInt256) (c0 : UInt256) (c1 : UInt256) (v37 : UInt256) (v38 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v38 25
  let v39 := a[v38.toNat]!
  let v40 := v37 ^^^ v39
  let v41 := (UInt256.ofNat 17)
  checkIndex v41 25
  let v42 := a[v41.toNat]!
  let v43 := v40 ^^^ v42
  let v44 := (UInt256.ofNat 22)
  checkIndex v44 25
  let v45 := a[v44.toNat]!
  let v46 := v43 ^^^ v45
  let c2 := v46
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
  keccak_round_k3 a constant_2 c0 c1 c2 v55 v56 e

def keccak_round_k1 (a : (Vector UInt256 25)) (constant_2 : UInt256) (c0 : UInt256) (v19 : UInt256) (v20 : UInt256) (e : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut e := e
  checkIndex v20 25
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
  let v31 := v28 ^^^ v30
  let c1 := v31
  let v33 := (UInt256.ofNat 2)
  checkIndex v33 25
  let v34 := a[v33.toNat]!
  let v35 := (UInt256.ofNat 7)
  checkIndex v35 25
  let v36 := a[v35.toNat]!
  let v37 := v34 ^^^ v36
  let v38 := (UInt256.ofNat 12)
  keccak_round_k2 a constant_2 c0 c1 v37 v38 e

def keccak_round (a : (Vector UInt256 25)) (e : (Vector UInt256 25)) (constant_2 : UInt256) : FeM (Unit × (Vector UInt256 25)) := do
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
  let v16 := v13 ^^^ v15
  let c0 := v16
  let v18 := (UInt256.ofNat 1)
  checkIndex v18 25
  let v19 := a[v18.toNat]!
  let v20 := (UInt256.ofNat 6)
  keccak_round_k1 a constant_2 c0 v19 v20 e

def permute (state : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut other : (Vector UInt256 25) := default
  let mut round : UInt256 := default
  let mut r2 : (Vector UInt256 25) := default
  let mut state := state
  let v1 := (UInt256.ofNat 0)
  let v2 := Vector.replicate 25 v1
  let other_3 := v2
  other := other_3
  let v4 := (UInt256.ofNat 0)
  let round_5 := v4
  round := round_5
  let rec b1 (other : (Vector UInt256 25)) (round : UInt256) (r2 : (Vector UInt256 25)) (state : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
    let mut other := other
    let mut round := round
    let mut r2 := r2
    let mut state := state
    let round_6 := round
    let round_7 := round_6
    let v8 := (UInt256.ofNat 24)
    let v9 := decide (round_7.toNat < v8.toNat)
    if v9 then
      let round_10 := round
      let round_11 := round_10
      let v12 := (UInt256.ofNat 2)
      let v13 ← checkedRem round_11 v12
      let v14 := (UInt256.ofNat 0)
      let v15 := decide (v13 = v14)
      let b6 : (Vector UInt256 25) → UInt256 → (Vector UInt256 25) → (Vector UInt256 25) → FeM (Unit × (Vector UInt256 25)) := fun other round r2 state => do
        let mut other := other
        let mut round := round
        let mut r2 := r2
        let mut state := state
        let v33 := (UInt256.ofNat 1)
        let round_34 := round
        let v35 ← checkedAdd round_34 v33
        let round_36 := v35
        round := round_36
        let v37 := ()
        b1 other round r2 state
      if v15 then
        let v18 ← ROUND_CONSTANTS
        let round_19 := round
        checkIndex round_19 24
        let v20 := v18[round_19.toNat]!
        let (v22, v22_0) ← keccak_round state other v20
        other := v22_0
        let v23 := v22
        b6 other round r2 state
      else
        let other_24 := other
        let other_25 := other_24
        r2 := other_25
        let v27 ← ROUND_CONSTANTS
        let round_28 := round
        checkIndex round_28 24
        let v29 := v27[round_28.toNat]!
        let (v31, v31_0) ← keccak_round r2 state v29
        state := v31_0
        let v32 := v31
        b6 other round r2 state
    else
      let v38 := ()
      return ((), state)
  partial_fixpoint
  b1 other round r2 state

def Keccak256_advance (self : Keccak256) (bytes_1 : UInt256) : FeM (Unit × Keccak256) := do
  let mut self := self
  let bytes_2 := bytes_1
  let v3 := self.position
  let v4 ← checkedAdd v3 bytes_2
  self := { self with position := v4 }
  let v5 := ()
  let v6 := self.position
  let v7 := (UInt256.ofNat 136)
  let v8 := decide (v6 = v7)
  let b3 : Keccak256 → FeM (Unit × Keccak256) := fun self => do
    let mut self := self
    return ((), self)
  if v8 then
    let (v10, v10_0) ← permute self.lanes
    self := { self with lanes := v10_0 }
    let v11 := (UInt256.ofNat 0)
    self := { self with position := v11 }
    let v12 := ()
    let v13 := v12
    b3 self
  else
    let v14 := ()
    let v15 := v14
    b3 self

def Keccak256_update_word (self : Keccak256) (value_1 : UInt256) (len_2 : UInt256) : FeM (Unit × Keccak256) := do
  let mut start : UInt256 := default
  let mut remaining : UInt256 := default
  let mut self := self
  let len_6 := len_2
  let v7 := (UInt256.ofNat 32)
  let v8 := decide (len_6.toNat ≤ v7.toNat)
  let v9 ← «assert» v8
  let len_10 := len_2
  let v11 := (UInt256.ofNat 0)
  let v12 := decide (len_10 ≠ v11)
  let b2 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
    let mut start := start
    let mut remaining := remaining
    let mut self := self
    let v129 := ()
    let v130 := v129
    let value_131 := value_1
    let v132 := (UInt256.ofNat 192)
    let v133 := value_131 >>> v132
    let v134 ← downcast_truncate_u256_u64 v133
    let v135 := IntTy.swapBytes (.u 64) v134
    let value_136 := value_1
    let v137 := (UInt256.ofNat 128)
    let v138 := value_136 >>> v137
    let v139 ← downcast_truncate_u256_u64 v138
    let v140 := IntTy.swapBytes (.u 64) v139
    let value_141 := value_1
    let v142 := (UInt256.ofNat 64)
    let v143 := value_141 >>> v142
    let v144 ← downcast_truncate_u256_u64 v143
    let v145 := IntTy.swapBytes (.u 64) v144
    let v146 ← downcast_truncate_u256_u64 value_1
    let v147 := IntTy.swapBytes (.u 64) v146
    let v148 := (UInt256.ofNat 0)
    let v149 := #v[v135, v140, v145, v147, v148]
    let words := v149
    let v151 := (UInt256.ofNat 32)
    let len_152 := len_2
    let v153 ← checkedSub v151 len_152
    let start_154 := v153
    start := start_154
    let len_155 := len_2
    let len_156 := len_155
    remaining := len_156
    let rec b11 (start : UInt256) (remaining : UInt256) (self : Keccak256) : FeM (Unit × Keccak256) := do
      let mut start := start
      let mut remaining := remaining
      let mut self := self
      let remaining_157 := remaining
      let remaining_158 := remaining_157
      let v159 := (UInt256.ofNat 0)
      let v160 := decide (v159.toNat < remaining_158.toNat)
      if v160 then
        let v161 := self.position
        let v162 := (UInt256.ofNat 8)
        let v163 ← checkedRem v161 v162
        let offset := v163
        let remaining_165 := remaining
        let remaining_166 := remaining_165
        let v167 := (UInt256.ofNat 8)
        let v168 ← checkedSub v167 offset
        let v169 := decide (remaining_166.toNat < v168.toNat)
        let b16 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self v3 => do
          let mut start := start
          let mut remaining := remaining
          let mut self := self
          let take := v3
          let start_177 := start
          let start_178 := start_177
          let v179 := (UInt256.ofNat 8)
          let v180 ← checkedRem start_178 v179
          let v181 := (UInt256.ofNat 8)
          let v182 ← checkedMul v180 v181
          let v183 ← downcast_unchecked_usize_u64 v182
          let shift_184 := v183
          let start_185 := start
          let start_186 := start_185
          let v187 := (UInt256.ofNat 8)
          let v188 ← checkedDiv start_186 v187
          checkIndex v188 5
          let v189 := words[v188.toNat]!
          let v190 := v189 >>> shift_184
          let low := v190
          let v192 := (UInt256.ofNat 0)
          let v193 := decide (shift_184 = v192)
          let b19 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self v4 => do
            let mut start := start
            let mut remaining := remaining
            let mut self := self
            let high := v4
            let v208 := (UInt256.ofNat 18446744073709551615)
            let all_209 := v208
            let v210 := (UInt256.ofNat 8)
            let v211 := decide (take = v210)
            let b22 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun start remaining self v5 => do
              let mut start := start
              let mut remaining := remaining
              let mut self := self
              let mask := v5
              let v221 := low ||| high
              let v222 := v221 &&& mask
              let v223 := (UInt256.ofNat 8)
              let v224 ← checkedMul offset v223
              let v225 ← downcast_unchecked_usize_u64 v224
              let v226 := IntTy.shl (.u 64) v222 v225
              let v227 := self.position
              let v228 := (UInt256.ofNat 8)
              let v229 ← checkedDiv v227 v228
              checkIndex v229 25
              let v230 := self.lanes[v229.toNat]!
              let v231 := v230 ^^^ v226
              checkIndex v229 25
              self := { self with lanes := (self.lanes.set! v229.toNat v231) }
              let v232 := ()
              let start_233 := start
              let v234 ← checkedAdd start_233 take
              let start_235 := v234
              start := start_235
              let v236 := ()
              let remaining_237 := remaining
              let v238 ← checkedSub remaining_237 take
              let remaining_239 := v238
              remaining := remaining_239
              let v240 := ()
              let (v241, v241_0) ← Keccak256_advance self take
              self := v241_0
              b11 start remaining self
            if v211 then
              let all_212 := all_209
              b22 start remaining self all_212
            else
              let v213 := (UInt256.ofNat 64)
              let v214 := (UInt256.ofNat 8)
              let v215 ← checkedMul take v214
              let v216 ← checkedSub v213 v215
              let v217 ← downcast_unchecked_usize_u64 v216
              let v218 := all_209 >>> v217
              let v219 := v218
              b22 start remaining self v219
          if v193 then
            let v194 := (UInt256.ofNat 0)
            let v195 := v194
            b19 start remaining self v195
          else
            let start_196 := start
            let start_197 := start_196
            let v198 := (UInt256.ofNat 8)
            let v199 ← checkedDiv start_197 v198
            let v200 := (UInt256.ofNat 1)
            let v201 ← checkedAdd v199 v200
            checkIndex v201 5
            let v202 := words[v201.toNat]!
            let v203 := (UInt256.ofNat 64)
            let v204 ← IntTy.sub (.u 64) v203 shift_184
            let v205 := IntTy.shl (.u 64) v202 v204
            let v206 := v205
            b19 start remaining self v206
        if v169 then
          let remaining_170 := remaining
          let remaining_171 := remaining_170
          let remaining_172 := remaining_171
          b16 start remaining self remaining_172
        else
          let v173 := (UInt256.ofNat 8)
          let v174 ← checkedSub v173 offset
          let v175 := v174
          b16 start remaining self v175
      else
        let v242 := ()
        return ((), self)
    partial_fixpoint
    b11 start remaining self
  if v12 then
    let v13 := self.position
    let len_14 := len_2
    let v15 ← checkedAdd v13 len_14
    let v16 := (UInt256.ofNat 136)
    let v17 := decide (v15.toNat ≤ v16.toNat)
    if v17 then
      let value_18 := value_1
      let v19 := (UInt256.ofNat 32)
      let len_20 := len_2
      let v21 ← checkedSub v19 len_20
      let v22 := (UInt256.ofNat 8)
      let v23 ← checkedMul v21 v22
      let v24 := v23
      let v25 := value_18 <<< v24
      let leading := v25
      let v27 := (UInt256.ofNat 192)
      let v28 := leading >>> v27
      let v29 ← downcast_truncate_u256_u64 v28
      let v30 := IntTy.swapBytes (.u 64) v29
      let w0 := v30
      let v32 := (UInt256.ofNat 128)
      let v33 := leading >>> v32
      let v34 ← downcast_truncate_u256_u64 v33
      let v35 := IntTy.swapBytes (.u 64) v34
      let w1 := v35
      let v37 := (UInt256.ofNat 64)
      let v38 := leading >>> v37
      let v39 ← downcast_truncate_u256_u64 v38
      let v40 := IntTy.swapBytes (.u 64) v39
      let w2 := v40
      let v42 ← downcast_truncate_u256_u64 leading
      let v43 := IntTy.swapBytes (.u 64) v42
      let w3 := v43
      let v45 := self.position
      let v46 := (UInt256.ofNat 8)
      let v47 ← checkedDiv v45 v46
      let lane := v47
      let v49 := self.position
      let v50 := (UInt256.ofNat 8)
      let v51 ← checkedRem v49 v50
      let v52 := (UInt256.ofNat 8)
      let v53 ← checkedMul v51 v52
      let v54 ← downcast_unchecked_usize_u64 v53
      let shift_55 := v54
      let v56 := (UInt256.ofNat 0)
      let v57 := decide (shift_55 = v56)
      let b7 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
        let mut start := start
        let mut remaining := remaining
        let mut self := self
        let len_115 := len_2
        let v116 := self.position
        let v117 ← checkedAdd v116 len_115
        self := { self with position := v117 }
        let v118 := ()
        let v119 := self.position
        let v120 := (UInt256.ofNat 136)
        let v121 := decide (v119 = v120)
        let b10 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun start remaining self => do
          let mut start := start
          let mut remaining := remaining
          let mut self := self
          return ((), self)
        if v121 then
          let (v123, v123_0) ← permute self.lanes
          self := { self with lanes := v123_0 }
          let v124 := (UInt256.ofNat 0)
          self := { self with position := v124 }
          let v125 := ()
          let v126 := v125
          b10 start remaining self
        else
          let v127 := ()
          let v128 := v127
          b10 start remaining self
      if v57 then
        checkIndex lane 25
        let v58 := self.lanes[lane.toNat]!
        let v59 := v58 ^^^ w0
        checkIndex lane 25
        self := { self with lanes := (self.lanes.set! lane.toNat v59) }
        let v60 := ()
        let v61 := (UInt256.ofNat 1)
        let v62 ← checkedAdd lane v61
        checkIndex v62 25
        let v63 := self.lanes[v62.toNat]!
        let v64 := v63 ^^^ w1
        checkIndex v62 25
        self := { self with lanes := (self.lanes.set! v62.toNat v64) }
        let v65 := ()
        let v66 := (UInt256.ofNat 2)
        let v67 ← checkedAdd lane v66
        checkIndex v67 25
        let v68 := self.lanes[v67.toNat]!
        let v69 := v68 ^^^ w2
        checkIndex v67 25
        self := { self with lanes := (self.lanes.set! v67.toNat v69) }
        let v70 := ()
        let v71 := (UInt256.ofNat 3)
        let v72 ← checkedAdd lane v71
        checkIndex v72 25
        let v73 := self.lanes[v72.toNat]!
        let v74 := v73 ^^^ w3
        checkIndex v72 25
        self := { self with lanes := (self.lanes.set! v72.toNat v74) }
        let v75 := ()
        let v76 := v75
        b7 start remaining self
      else
        let v77 := (UInt256.ofNat 64)
        let v78 ← IntTy.sub (.u 64) v77 shift_55
        let back := v78
        let v80 := IntTy.shl (.u 64) w0 shift_55
        checkIndex lane 25
        let v81 := self.lanes[lane.toNat]!
        let v82 := v81 ^^^ v80
        checkIndex lane 25
        self := { self with lanes := (self.lanes.set! lane.toNat v82) }
        let v83 := ()
        let v84 := IntTy.shl (.u 64) w1 shift_55
        let v85 := w0 >>> back
        let v86 := v84 ||| v85
        let v87 := (UInt256.ofNat 1)
        let v88 ← checkedAdd lane v87
        checkIndex v88 25
        let v89 := self.lanes[v88.toNat]!
        let v90 := v89 ^^^ v86
        checkIndex v88 25
        self := { self with lanes := (self.lanes.set! v88.toNat v90) }
        let v91 := ()
        let v92 := IntTy.shl (.u 64) w2 shift_55
        let v93 := w1 >>> back
        let v94 := v92 ||| v93
        let v95 := (UInt256.ofNat 2)
        let v96 ← checkedAdd lane v95
        checkIndex v96 25
        let v97 := self.lanes[v96.toNat]!
        let v98 := v97 ^^^ v94
        checkIndex v96 25
        self := { self with lanes := (self.lanes.set! v96.toNat v98) }
        let v99 := ()
        let v100 := IntTy.shl (.u 64) w3 shift_55
        let v101 := w2 >>> back
        let v102 := v100 ||| v101
        let v103 := (UInt256.ofNat 3)
        let v104 ← checkedAdd lane v103
        checkIndex v104 25
        let v105 := self.lanes[v104.toNat]!
        let v106 := v105 ^^^ v102
        checkIndex v104 25
        self := { self with lanes := (self.lanes.set! v104.toNat v106) }
        let v107 := ()
        let v108 := w3 >>> back
        let v109 := (UInt256.ofNat 4)
        let v110 ← checkedAdd lane v109
        checkIndex v110 25
        let v111 := self.lanes[v110.toNat]!
        let v112 := v111 ^^^ v108
        checkIndex v110 25
        self := { self with lanes := (self.lanes.set! v110.toNat v112) }
        let v113 := ()
        let v114 := v113
        b7 start remaining self
    else
      b2 start remaining self
  else
    b2 start remaining self

def Keccak256_pad (self : Keccak256) : FeM (Unit × Keccak256) := do
  let mut self := self
  let v1 := (UInt256.ofNat 1)
  let v2 := self.position
  let v3 := (UInt256.ofNat 8)
  let v4 ← checkedRem v2 v3
  let v5 := (UInt256.ofNat 8)
  let v6 ← checkedMul v4 v5
  let v7 ← downcast_unchecked_usize_u64 v6
  let v8 := IntTy.shl (.u 64) v1 v7
  let v9 := self.position
  let v10 := (UInt256.ofNat 8)
  let v11 ← checkedDiv v9 v10
  checkIndex v11 25
  let v12 := self.lanes[v11.toNat]!
  let v13 := v12 ^^^ v8
  checkIndex v11 25
  self := { self with lanes := (self.lanes.set! v11.toNat v13) }
  let v14 := ()
  let v15 := (UInt256.ofNat 9223372036854775808)
  let v16 := (UInt256.ofNat 16)
  checkIndex v16 25
  let v17 := self.lanes[v16.toNat]!
  let v18 := v17 ^^^ v15
  checkIndex v16 25
  self := { self with lanes := (self.lanes.set! v16.toNat v18) }
  let v19 := ()
  let (v21, v21_0) ← permute self.lanes
  self := { self with lanes := v21_0 }
  return ((), self)

def Keccak256_finish_word (self : Keccak256) : FeM UInt256 := do
  let mut self := self
  let (v1, v1_0) ← Keccak256_pad self
  self := v1_0
  let v2 := (UInt256.ofNat 0)
  checkIndex v2 25
  let v3 := self.lanes[v2.toNat]!
  let v4 := IntTy.swapBytes (.u 64) v3
  let v5 := v4
  let v6 := (UInt256.ofNat 192)
  let v7 := v5 <<< v6
  let v8 := (UInt256.ofNat 1)
  checkIndex v8 25
  let v9 := self.lanes[v8.toNat]!
  let v10 := IntTy.swapBytes (.u 64) v9
  let v11 := v10
  let v12 := (UInt256.ofNat 128)
  let v13 := v11 <<< v12
  let v14 := v7 ||| v13
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 25
  let v16 := self.lanes[v15.toNat]!
  let v17 := IntTy.swapBytes (.u 64) v16
  let v18 := v17
  let v19 := (UInt256.ofNat 64)
  let v20 := v18 <<< v19
  let v21 := v14 ||| v20
  let v22 := (UInt256.ofNat 3)
  checkIndex v22 25
  let v23 := self.lanes[v22.toNat]!
  let v24 := IntTy.swapBytes (.u 64) v23
  let v25 := v24
  let v26 := v21 ||| v25
  return v26

def recover_signer (hash : UInt256) (recovery_id : UInt256) (r : UInt256) (s : UInt256) : FeM Option_u256 := do
  let mut digest : Keccak256 := default
  let v4 ← recover hash recovery_id r s
  match v4 with
  | .Some .. =>
    let v5 := v4.Some_0
    let key := v5
    let v7 ← Keccak256_new
    let digest_8 := v7
    digest := digest_8
    let v9 := key.x
    let v10 := (UInt256.ofNat 32)
    let (v12, v12_0) ← Keccak256_update_word digest v9 v10
    digest := v12_0
    let v13 := key.y
    let v14 := (UInt256.ofNat 32)
    let (v16, v16_0) ← Keccak256_update_word digest v13 v14
    digest := v16_0
    let digest_17 := digest
    let digest_18 := digest_17
    let v19 ← Keccak256_finish_word digest_18
    let v20 := (UInt256.ofNat 1461501637330902918203684832716283019655932542975)
    let v21 := v19 &&& v20
    let v22 := Option_u256.Some v21
    return v22
  | _ =>
    let v23 := Option_u256.None
    return v23

def ecrecover_MemBuffer (input : MemBuffer) : FeM Output := do
  let mut r0 : (Vector UInt256 1) := default
  let v4 := (UInt256.ofNat 0)
  let v5 ← word_at_MemBuffer input v4
  let hash := v5
  let v7 := (UInt256.ofNat 32)
  let v8 ← word_at_MemBuffer input v7
  let v_1 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 ← word_at_MemBuffer input v10
  let r := v11
  let v13 := (UInt256.ofNat 96)
  let v14 ← word_at_MemBuffer input v13
  let s := v14
  let v16 := (UInt256.ofNat 27)
  let v17 := decide (v_1 ≠ v16)
  let b3 : (Vector UInt256 1) → Bool → FeM Output := fun r0 v1 => do
    let mut r0 := r0
    let malformed := v1
    let b11 : (Vector UInt256 1) → Option_u256 → FeM Output := fun r0 v2 => do
      let mut r0 := r0
      let signer := v2
      let b12 : (Vector UInt256 1) → Output → FeM Output := fun r0 v3 => do
        let mut r0 := r0
        return v3
      match signer with
      | .Some .. =>
        let v40 := signer.Some_0
        let address := v40
        let v42 := #v[address]
        r0 := v42
        let v43 := (UInt256.ofNat 3000)
        let v45 ← words_1 r0 v43
        let v46 := v45
        b12 r0 v46
      | .None =>
        let v47 ← SUCCESS
        let v48 := (UInt256.ofNat 3000)
        let v49 ← MemBuffer_empty
        let v50 := (Output.mk v47 v48 v49)
        let v51 := v50
        b12 r0 v51
    if malformed then
      let v33 := Option_u256.None
      let v34 := v33
      b11 r0 v34
    else
      let v35 := (UInt256.ofNat 27)
      let v36 ← checkedSub v_1 v35
      let v37 ← recover_signer hash v36 r s
      let v38 := v37
      b11 r0 v38
  let b1 : (Vector UInt256 1) → FeM Output := fun r0 => do
    let mut r0 := r0
    let v30 := true
    let v31 := v30
    b3 r0 v31
  let b7 : (Vector UInt256 1) → FeM Output := fun r0 => do
    let mut r0 := r0
    let v20 := (UInt256.ofNat 0)
    let v21 := decide (r = v20)
    if v21 then
      b1 r0
    else
      let v22 ← N
      let v23 := decide (v22.toNat ≤ r.toNat)
      if v23 then
        b1 r0
      else
        let v24 := (UInt256.ofNat 0)
        let v25 := decide (s = v24)
        if v25 then
          b1 r0
        else
          let v26 ← N
          let v27 := decide (v26.toNat ≤ s.toNat)
          if v27 then
            b1 r0
          else
            let v28 := false
            let v29 := v28
            b3 r0 v29
  if v17 then
    let v18 := (UInt256.ofNat 28)
    let v19 := decide (v_1 ≠ v18)
    if v19 then
      b1 r0
    else
      b7 r0
  else
    b7 r0

def recv0_arm0 : FeM Empty := do
  let mut result : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_4
  r2 := v0
  let v2 ← ecrecover_MemBuffer r2
  let result_3 := v2
  result := result_3
  let _ ← MemBuffer_drop r2
  let v6 ← MemBuffer_span_ref result.output
  let v9 ← Evm_return_data v6
  nomatch v9

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


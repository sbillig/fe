export Lib (Evm_calldatacopy Evm_calldatasize Evm_return_data MemBuffer MemBuffer_drop MemBuffer_empty MemBuffer_new MemBuffer_span_ref MemBuffer_try_new MemBuffer_write_word Option_u256 Option_u256.None Option_u256.Some Range_Known_0_Unknown Range_Known_0_Unknown.mk Range_at_0 Range_has_0 Range_next_0 Range_start_0 assert downcast_truncate_u256_u64 downcast_unchecked_u256_usize downcast_unchecked_usize_u64)
export Shared (N Output Output.mk SUCCESS host_allocation recover word_at_MemBuffer)

structure Keccak256 where
  lanes : (Vector UInt256 25)
  position : UInt256
deriving Inhabited

def init : FeM Unit := do
  return ()

def input_Evm_4 : FeM MemBuffer := do
  let mut r1 : MemBuffer := default
  let v1 ← Evm_calldatasize
  let v2 := v1
  let v3 ← downcast_unchecked_u256_usize v2
  let v4 ← MemBuffer_new v3
  let v5 := v4
  r1 := v5
  let v7 := (UInt256.ofNat 0)
  let v8 := (UInt256.ofNat 0)
  let (v10, v10_0) ← Evm_calldatacopy r1 v7 v8 v2
  r1 := v10_0
  let v11 := r1
  return v11

def words_1 (v0 : (Vector UInt256 1)) (v1 : UInt256) : FeM Output := do
  let mut r0 : MemBuffer := default
  let mut r1 : Range_Known_0_Unknown := default
  let v3 := (UInt256.ofNat 1)
  let v4 := (UInt256.ofNat 32)
  let v5 ← checkedMul v3 v4
  let v6 ← MemBuffer_try_new v5
  let v7 := v6
  let v8 := v7
  match (← pure v8) with
  | .Some .. =>
    let v9 := v7.Some_0
    r0 := v9
    let v10 := (UInt256.ofNat 0)
    let v11 := (UInt256.ofNat 1)
    let v12 := ()
    let v13 := (Range_Known_0_Unknown.mk v12 v11)
    r1 := v13
    let v15 ← Range_start_0 r1
    let v16 := v15
    let rec b5 (r0 : MemBuffer) (r1 : Range_Known_0_Unknown) (v2 : UInt256) : FeM Output := do
      let mut r0 := r0
      let mut r1 := r1
      let v18 ← Range_has_0 r1 v2
      if v18 then
        let v20 ← Range_at_0 r1 v2
        let v21 := v20
        let v22 := (UInt256.ofNat 32)
        let v23 ← checkedMul v21 v22
        checkIndex v21 1
        let v24 := v0[v21.toNat]!
        let (v26, v26_0) ← MemBuffer_write_word r0 v23 v24
        r0 := v26_0
        let v28 ← Range_next_0 r1 v2
        let v29 := v28
        b5 r0 r1 v29
      else
        let v30 ← SUCCESS
        let v31 := v1
        let v32 := r0
        let v33 := (Output.mk v30 v31 v32)
        return v33
    partial_fixpoint
    b5 r0 r1 v16
  | _ =>
    let v34 ← host_allocation v1
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

def rotate (v0 : UInt256) (v1 : UInt256) : FeM UInt256 := do
  let v2 := v0
  let v3 := v1
  let v4 := IntTy.shl (.u 64) v2 v3
  let v5 := v0
  let v6 := (UInt256.ofNat 64)
  let v7 := v1
  let v8 ← IntTy.sub (.u 64) v6 v7
  let v9 := v5 >>> v8
  let v10 := v4 ||| v9
  return v10

def keccak_round_k19 (v323 : UInt256) (v329 : UInt256) (v341 : UInt256) (v347 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v363 := IntTy.not (.u 64) v347
  let v364 := v363 &&& v323
  let v365 := v341 ^^^ v364
  let v366 := (UInt256.ofNat 23)
  checkIndex v366 25
  v1 := (v1.set! v366.toNat v365)
  let v367 := ()
  let v368 := IntTy.not (.u 64) v323
  let v369 := v368 &&& v329
  let v370 := v347 ^^^ v369
  let v371 := (UInt256.ofNat 24)
  checkIndex v371 25
  v1 := (v1.set! v371.toNat v370)
  let v372 := ()
  return ((), v1)

def keccak_round_k18 (v323 : UInt256) (v329 : UInt256) (v335 : UInt256) (v341 : UInt256) (v344 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v345 := (UInt256.ofNat 2)
  let v346 ← rotate v344 v345
  let v347 := v346
  let v348 := IntTy.not (.u 64) v329
  let v349 := v348 &&& v335
  let v350 := v323 ^^^ v349
  let v351 := (UInt256.ofNat 20)
  checkIndex v351 25
  v1 := (v1.set! v351.toNat v350)
  let v352 := ()
  let v353 := IntTy.not (.u 64) v335
  let v354 := v353 &&& v341
  let v355 := v329 ^^^ v354
  let v356 := (UInt256.ofNat 21)
  checkIndex v356 25
  v1 := (v1.set! v356.toNat v355)
  let v357 := ()
  let v358 := IntTy.not (.u 64) v341
  let v359 := v358 &&& v347
  let v360 := v335 ^^^ v359
  let v361 := (UInt256.ofNat 22)
  checkIndex v361 25
  v1 := (v1.set! v361.toNat v360)
  let v362 := ()
  keccak_round_k19 v323 v329 v341 v347 v1

def keccak_round_k17 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v93 : UInt256) (v97 : UInt256) (v323 : UInt256) (v324 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  checkIndex v324 25
  let v325 := v0[v324.toNat]!
  let v326 := v325 ^^^ v93
  let v327 := (UInt256.ofNat 55)
  let v328 ← rotate v326 v327
  let v329 := v328
  let v330 := (UInt256.ofNat 14)
  checkIndex v330 25
  let v331 := v0[v330.toNat]!
  let v332 := v331 ^^^ v97
  let v333 := (UInt256.ofNat 39)
  let v334 ← rotate v332 v333
  let v335 := v334
  let v336 := (UInt256.ofNat 15)
  checkIndex v336 25
  let v337 := v0[v336.toNat]!
  let v338 := v337 ^^^ v81
  let v339 := (UInt256.ofNat 41)
  let v340 ← rotate v338 v339
  let v341 := v340
  let v342 := (UInt256.ofNat 21)
  checkIndex v342 25
  let v343 := v0[v342.toNat]!
  let v344 := v343 ^^^ v85
  keccak_round_k18 v323 v329 v335 v341 v344 v1

def keccak_round_k16 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v268 : UInt256) (v274 : UInt256) (v286 : UInt256) (v292 : UInt256) (v305 : UInt256) (v306 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  v1 := (v1.set! v306.toNat v305)
  let v307 := ()
  let v308 := IntTy.not (.u 64) v292
  let v309 := v308 &&& v268
  let v310 := v286 ^^^ v309
  let v311 := (UInt256.ofNat 18)
  checkIndex v311 25
  v1 := (v1.set! v311.toNat v310)
  let v312 := ()
  let v313 := IntTy.not (.u 64) v268
  let v314 := v313 &&& v274
  let v315 := v292 ^^^ v314
  let v316 := (UInt256.ofNat 19)
  checkIndex v316 25
  v1 := (v1.set! v316.toNat v315)
  let v317 := ()
  let v318 := (UInt256.ofNat 2)
  checkIndex v318 25
  let v319 := v0[v318.toNat]!
  let v320 := v319 ^^^ v89
  let v321 := (UInt256.ofNat 62)
  let v322 ← rotate v320 v321
  let v323 := v322
  let v324 := (UInt256.ofNat 8)
  keccak_round_k17 v0 v81 v85 v93 v97 v323 v324 v1

def keccak_round_k15 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v268 : UInt256) (v274 : UInt256) (v280 : UInt256) (v286 : UInt256) (v287 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v288 := v0[v287.toNat]!
  let v289 := v288 ^^^ v93
  let v290 := (UInt256.ofNat 56)
  let v291 ← rotate v289 v290
  let v292 := v291
  let v293 := IntTy.not (.u 64) v274
  let v294 := v293 &&& v280
  let v295 := v268 ^^^ v294
  let v296 := (UInt256.ofNat 15)
  checkIndex v296 25
  v1 := (v1.set! v296.toNat v295)
  let v297 := ()
  let v298 := IntTy.not (.u 64) v280
  let v299 := v298 &&& v286
  let v300 := v274 ^^^ v299
  let v301 := (UInt256.ofNat 16)
  checkIndex v301 25
  v1 := (v1.set! v301.toNat v300)
  let v302 := ()
  let v303 := IntTy.not (.u 64) v286
  let v304 := v303 &&& v292
  let v305 := v280 ^^^ v304
  let v306 := (UInt256.ofNat 17)
  checkIndex v306 25
  keccak_round_k16 v0 v81 v85 v89 v93 v97 v268 v274 v286 v292 v305 v306 v1

def keccak_round_k14 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v267 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v268 := v267
  let v269 := (UInt256.ofNat 5)
  checkIndex v269 25
  let v270 := v0[v269.toNat]!
  let v271 := v270 ^^^ v81
  let v272 := (UInt256.ofNat 36)
  let v273 ← rotate v271 v272
  let v274 := v273
  let v275 := (UInt256.ofNat 11)
  checkIndex v275 25
  let v276 := v0[v275.toNat]!
  let v277 := v276 ^^^ v85
  let v278 := (UInt256.ofNat 10)
  let v279 ← rotate v277 v278
  let v280 := v279
  let v281 := (UInt256.ofNat 17)
  checkIndex v281 25
  let v282 := v0[v281.toNat]!
  let v283 := v282 ^^^ v89
  let v284 := (UInt256.ofNat 15)
  let v285 ← rotate v283 v284
  let v286 := v285
  let v287 := (UInt256.ofNat 23)
  checkIndex v287 25
  keccak_round_k15 v0 v81 v85 v89 v93 v97 v268 v274 v280 v286 v287 v1

def keccak_round_k13 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v213 : UInt256) (v219 : UInt256) (v231 : UInt256) (v237 : UInt256) (v250 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v251 := (UInt256.ofNat 12)
  checkIndex v251 25
  v1 := (v1.set! v251.toNat v250)
  let v252 := ()
  let v253 := IntTy.not (.u 64) v237
  let v254 := v253 &&& v213
  let v255 := v231 ^^^ v254
  let v256 := (UInt256.ofNat 13)
  checkIndex v256 25
  v1 := (v1.set! v256.toNat v255)
  let v257 := ()
  let v258 := IntTy.not (.u 64) v213
  let v259 := v258 &&& v219
  let v260 := v237 ^^^ v259
  let v261 := (UInt256.ofNat 14)
  checkIndex v261 25
  v1 := (v1.set! v261.toNat v260)
  let v262 := ()
  let v263 := (UInt256.ofNat 4)
  checkIndex v263 25
  let v264 := v0[v263.toNat]!
  let v265 := v264 ^^^ v97
  let v266 := (UInt256.ofNat 27)
  let v267 ← rotate v265 v266
  keccak_round_k14 v0 v81 v85 v89 v93 v97 v267 v1

def keccak_round_k12 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v213 : UInt256) (v219 : UInt256) (v225 : UInt256) (v231 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v232 := (UInt256.ofNat 20)
  checkIndex v232 25
  let v233 := v0[v232.toNat]!
  let v234 := v233 ^^^ v81
  let v235 := (UInt256.ofNat 18)
  let v236 ← rotate v234 v235
  let v237 := v236
  let v238 := IntTy.not (.u 64) v219
  let v239 := v238 &&& v225
  let v240 := v213 ^^^ v239
  let v241 := (UInt256.ofNat 10)
  checkIndex v241 25
  v1 := (v1.set! v241.toNat v240)
  let v242 := ()
  let v243 := IntTy.not (.u 64) v225
  let v244 := v243 &&& v231
  let v245 := v219 ^^^ v244
  let v246 := (UInt256.ofNat 11)
  checkIndex v246 25
  v1 := (v1.set! v246.toNat v245)
  let v247 := ()
  let v248 := IntTy.not (.u 64) v231
  let v249 := v248 &&& v237
  let v250 := v225 ^^^ v249
  keccak_round_k13 v0 v81 v85 v89 v93 v97 v213 v219 v231 v237 v250 v1

def keccak_round_k11 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v210 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v211 := (UInt256.ofNat 1)
  let v212 ← rotate v210 v211
  let v213 := v212
  let v214 := (UInt256.ofNat 7)
  checkIndex v214 25
  let v215 := v0[v214.toNat]!
  let v216 := v215 ^^^ v89
  let v217 := (UInt256.ofNat 6)
  let v218 ← rotate v216 v217
  let v219 := v218
  let v220 := (UInt256.ofNat 13)
  checkIndex v220 25
  let v221 := v0[v220.toNat]!
  let v222 := v221 ^^^ v93
  let v223 := (UInt256.ofNat 25)
  let v224 ← rotate v222 v223
  let v225 := v224
  let v226 := (UInt256.ofNat 19)
  checkIndex v226 25
  let v227 := v0[v226.toNat]!
  let v228 := v227 ^^^ v97
  let v229 := (UInt256.ofNat 8)
  let v230 ← rotate v228 v229
  let v231 := v230
  keccak_round_k12 v0 v81 v85 v89 v93 v97 v213 v219 v225 v231 v1

def keccak_round_k10 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v158 : UInt256) (v164 : UInt256) (v170 : UInt256) (v176 : UInt256) (v182 : UInt256) (v193 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v194 := v193 &&& v182
  let v195 := v170 ^^^ v194
  let v196 := (UInt256.ofNat 7)
  checkIndex v196 25
  v1 := (v1.set! v196.toNat v195)
  let v197 := ()
  let v198 := IntTy.not (.u 64) v182
  let v199 := v198 &&& v158
  let v200 := v176 ^^^ v199
  let v201 := (UInt256.ofNat 8)
  checkIndex v201 25
  v1 := (v1.set! v201.toNat v200)
  let v202 := ()
  let v203 := IntTy.not (.u 64) v158
  let v204 := v203 &&& v164
  let v205 := v182 ^^^ v204
  let v206 := (UInt256.ofNat 9)
  checkIndex v206 25
  v1 := (v1.set! v206.toNat v205)
  let v207 := ()
  let v208 := (UInt256.ofNat 1)
  checkIndex v208 25
  let v209 := v0[v208.toNat]!
  let v210 := v209 ^^^ v85
  keccak_round_k11 v0 v81 v85 v89 v93 v97 v210 v1

def keccak_round_k9 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v158 : UInt256) (v164 : UInt256) (v170 : UInt256) (v173 : UInt256) (v174 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v175 ← rotate v173 v174
  let v176 := v175
  let v177 := (UInt256.ofNat 22)
  checkIndex v177 25
  let v178 := v0[v177.toNat]!
  let v179 := v178 ^^^ v89
  let v180 := (UInt256.ofNat 61)
  let v181 ← rotate v179 v180
  let v182 := v181
  let v183 := IntTy.not (.u 64) v164
  let v184 := v183 &&& v170
  let v185 := v158 ^^^ v184
  let v186 := (UInt256.ofNat 5)
  checkIndex v186 25
  v1 := (v1.set! v186.toNat v185)
  let v187 := ()
  let v188 := IntTy.not (.u 64) v170
  let v189 := v188 &&& v176
  let v190 := v164 ^^^ v189
  let v191 := (UInt256.ofNat 6)
  checkIndex v191 25
  v1 := (v1.set! v191.toNat v190)
  let v192 := ()
  let v193 := IntTy.not (.u 64) v176
  keccak_round_k10 v0 v81 v85 v89 v93 v97 v158 v164 v170 v176 v182 v193 v1

def keccak_round_k8 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v153 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v154 := v0[v153.toNat]!
  let v155 := v154 ^^^ v93
  let v156 := (UInt256.ofNat 28)
  let v157 ← rotate v155 v156
  let v158 := v157
  let v159 := (UInt256.ofNat 9)
  checkIndex v159 25
  let v160 := v0[v159.toNat]!
  let v161 := v160 ^^^ v97
  let v162 := (UInt256.ofNat 20)
  let v163 ← rotate v161 v162
  let v164 := v163
  let v165 := (UInt256.ofNat 10)
  checkIndex v165 25
  let v166 := v0[v165.toNat]!
  let v167 := v166 ^^^ v81
  let v168 := (UInt256.ofNat 3)
  let v169 ← rotate v167 v168
  let v170 := v169
  let v171 := (UInt256.ofNat 16)
  checkIndex v171 25
  let v172 := v0[v171.toNat]!
  let v173 := v172 ^^^ v85
  let v174 := (UInt256.ofNat 45)
  keccak_round_k9 v0 v81 v85 v89 v93 v97 v158 v164 v170 v173 v174 v1

def keccak_round_k7 (v0 : (Vector UInt256 25)) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v101 : UInt256) (v107 : UInt256) (v113 : UInt256) (v119 : UInt256) (v125 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v137 := ()
  let v138 := IntTy.not (.u 64) v119
  let v139 := v138 &&& v125
  let v140 := v113 ^^^ v139
  let v141 := (UInt256.ofNat 2)
  checkIndex v141 25
  v1 := (v1.set! v141.toNat v140)
  let v142 := ()
  let v143 := IntTy.not (.u 64) v125
  let v144 := v143 &&& v101
  let v145 := v119 ^^^ v144
  let v146 := (UInt256.ofNat 3)
  checkIndex v146 25
  v1 := (v1.set! v146.toNat v145)
  let v147 := ()
  let v148 := IntTy.not (.u 64) v101
  let v149 := v148 &&& v107
  let v150 := v125 ^^^ v149
  let v151 := (UInt256.ofNat 4)
  checkIndex v151 25
  v1 := (v1.set! v151.toNat v150)
  let v152 := ()
  let v153 := (UInt256.ofNat 3)
  checkIndex v153 25
  keccak_round_k8 v0 v81 v85 v89 v93 v97 v153 v1

def keccak_round_k6 (v0 : (Vector UInt256 25)) (v2 : UInt256) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v101 : UInt256) (v107 : UInt256) (v113 : UInt256) (v116 : UInt256) (v117 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v118 ← rotate v116 v117
  let v119 := v118
  let v120 := (UInt256.ofNat 24)
  checkIndex v120 25
  let v121 := v0[v120.toNat]!
  let v122 := v121 ^^^ v97
  let v123 := (UInt256.ofNat 14)
  let v124 ← rotate v122 v123
  let v125 := v124
  let v126 := IntTy.not (.u 64) v107
  let v127 := v126 &&& v113
  let v128 := v101 ^^^ v127
  let v129 := v2
  let v130 := v128 ^^^ v129
  let v131 := (UInt256.ofNat 0)
  checkIndex v131 25
  v1 := (v1.set! v131.toNat v130)
  let v132 := ()
  let v133 := IntTy.not (.u 64) v113
  let v134 := v133 &&& v119
  let v135 := v107 ^^^ v134
  let v136 := (UInt256.ofNat 1)
  checkIndex v136 25
  v1 := (v1.set! v136.toNat v135)
  keccak_round_k7 v0 v81 v85 v89 v93 v97 v101 v107 v113 v119 v125 v1

def keccak_round_k5 (v0 : (Vector UInt256 25)) (v2 : UInt256) (v81 : UInt256) (v85 : UInt256) (v89 : UInt256) (v93 : UInt256) (v97 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v98 := (UInt256.ofNat 0)
  checkIndex v98 25
  let v99 := v0[v98.toNat]!
  let v100 := v99 ^^^ v81
  let v101 := v100
  let v102 := (UInt256.ofNat 6)
  checkIndex v102 25
  let v103 := v0[v102.toNat]!
  let v104 := v103 ^^^ v85
  let v105 := (UInt256.ofNat 44)
  let v106 ← rotate v104 v105
  let v107 := v106
  let v108 := (UInt256.ofNat 12)
  checkIndex v108 25
  let v109 := v0[v108.toNat]!
  let v110 := v109 ^^^ v89
  let v111 := (UInt256.ofNat 43)
  let v112 ← rotate v110 v111
  let v113 := v112
  let v114 := (UInt256.ofNat 18)
  checkIndex v114 25
  let v115 := v0[v114.toNat]!
  let v116 := v115 ^^^ v93
  let v117 := (UInt256.ofNat 21)
  keccak_round_k6 v0 v2 v81 v85 v89 v93 v97 v101 v107 v113 v116 v117 v1

def keccak_round_k4 (v0 : (Vector UInt256 25)) (v2 : UInt256) (v17 : UInt256) (v32 : UInt256) (v47 : UInt256) (v62 : UInt256) (v73 : UInt256) (v74 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  checkIndex v74 25
  let v75 := v0[v74.toNat]!
  let v76 := v73 ^^^ v75
  let v77 := v76
  let v78 := (UInt256.ofNat 1)
  let v79 ← rotate v32 v78
  let v80 := v77 ^^^ v79
  let v81 := v80
  let v82 := (UInt256.ofNat 1)
  let v83 ← rotate v47 v82
  let v84 := v17 ^^^ v83
  let v85 := v84
  let v86 := (UInt256.ofNat 1)
  let v87 ← rotate v62 v86
  let v88 := v32 ^^^ v87
  let v89 := v88
  let v90 := (UInt256.ofNat 1)
  let v91 ← rotate v77 v90
  let v92 := v47 ^^^ v91
  let v93 := v92
  let v94 := (UInt256.ofNat 1)
  let v95 ← rotate v17 v94
  let v96 := v62 ^^^ v95
  let v97 := v96
  keccak_round_k5 v0 v2 v81 v85 v89 v93 v97 v1

def keccak_round_k3 (v0 : (Vector UInt256 25)) (v2 : UInt256) (v17 : UInt256) (v32 : UInt256) (v47 : UInt256) (v55 : UInt256) (v56 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  checkIndex v56 25
  let v57 := v0[v56.toNat]!
  let v58 := v55 ^^^ v57
  let v59 := (UInt256.ofNat 23)
  checkIndex v59 25
  let v60 := v0[v59.toNat]!
  let v61 := v58 ^^^ v60
  let v62 := v61
  let v63 := (UInt256.ofNat 4)
  checkIndex v63 25
  let v64 := v0[v63.toNat]!
  let v65 := (UInt256.ofNat 9)
  checkIndex v65 25
  let v66 := v0[v65.toNat]!
  let v67 := v64 ^^^ v66
  let v68 := (UInt256.ofNat 14)
  checkIndex v68 25
  let v69 := v0[v68.toNat]!
  let v70 := v67 ^^^ v69
  let v71 := (UInt256.ofNat 19)
  checkIndex v71 25
  let v72 := v0[v71.toNat]!
  let v73 := v70 ^^^ v72
  let v74 := (UInt256.ofNat 24)
  keccak_round_k4 v0 v2 v17 v32 v47 v62 v73 v74 v1

def keccak_round_k2 (v0 : (Vector UInt256 25)) (v2 : UInt256) (v17 : UInt256) (v32 : UInt256) (v37 : UInt256) (v38 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  checkIndex v38 25
  let v39 := v0[v38.toNat]!
  let v40 := v37 ^^^ v39
  let v41 := (UInt256.ofNat 17)
  checkIndex v41 25
  let v42 := v0[v41.toNat]!
  let v43 := v40 ^^^ v42
  let v44 := (UInt256.ofNat 22)
  checkIndex v44 25
  let v45 := v0[v44.toNat]!
  let v46 := v43 ^^^ v45
  let v47 := v46
  let v48 := (UInt256.ofNat 3)
  checkIndex v48 25
  let v49 := v0[v48.toNat]!
  let v50 := (UInt256.ofNat 8)
  checkIndex v50 25
  let v51 := v0[v50.toNat]!
  let v52 := v49 ^^^ v51
  let v53 := (UInt256.ofNat 13)
  checkIndex v53 25
  let v54 := v0[v53.toNat]!
  let v55 := v52 ^^^ v54
  let v56 := (UInt256.ofNat 18)
  keccak_round_k3 v0 v2 v17 v32 v47 v55 v56 v1

def keccak_round_k1 (v0 : (Vector UInt256 25)) (v2 : UInt256) (v17 : UInt256) (v19 : UInt256) (v20 : UInt256) (v1 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  checkIndex v20 25
  let v21 := v0[v20.toNat]!
  let v22 := v19 ^^^ v21
  let v23 := (UInt256.ofNat 11)
  checkIndex v23 25
  let v24 := v0[v23.toNat]!
  let v25 := v22 ^^^ v24
  let v26 := (UInt256.ofNat 16)
  checkIndex v26 25
  let v27 := v0[v26.toNat]!
  let v28 := v25 ^^^ v27
  let v29 := (UInt256.ofNat 21)
  checkIndex v29 25
  let v30 := v0[v29.toNat]!
  let v31 := v28 ^^^ v30
  let v32 := v31
  let v33 := (UInt256.ofNat 2)
  checkIndex v33 25
  let v34 := v0[v33.toNat]!
  let v35 := (UInt256.ofNat 7)
  checkIndex v35 25
  let v36 := v0[v35.toNat]!
  let v37 := v34 ^^^ v36
  let v38 := (UInt256.ofNat 12)
  keccak_round_k2 v0 v2 v17 v32 v37 v38 v1

def keccak_round (v0 : (Vector UInt256 25)) (v1 : (Vector UInt256 25)) (v2 : UInt256) : FeM (Unit × (Vector UInt256 25)) := do
  let mut v1 := v1
  let v3 := (UInt256.ofNat 0)
  checkIndex v3 25
  let v4 := v0[v3.toNat]!
  let v5 := (UInt256.ofNat 5)
  checkIndex v5 25
  let v6 := v0[v5.toNat]!
  let v7 := v4 ^^^ v6
  let v8 := (UInt256.ofNat 10)
  checkIndex v8 25
  let v9 := v0[v8.toNat]!
  let v10 := v7 ^^^ v9
  let v11 := (UInt256.ofNat 15)
  checkIndex v11 25
  let v12 := v0[v11.toNat]!
  let v13 := v10 ^^^ v12
  let v14 := (UInt256.ofNat 20)
  checkIndex v14 25
  let v15 := v0[v14.toNat]!
  let v16 := v13 ^^^ v15
  let v17 := v16
  let v18 := (UInt256.ofNat 1)
  checkIndex v18 25
  let v19 := v0[v18.toNat]!
  let v20 := (UInt256.ofNat 6)
  keccak_round_k1 v0 v2 v17 v19 v20 v1

def permute (v0 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
  let mut r0 : (Vector UInt256 25) := default
  let mut r1 : UInt256 := default
  let mut r2 : (Vector UInt256 25) := default
  let mut v0 := v0
  let v1 := (UInt256.ofNat 0)
  let v2 := Vector.replicate 25 v1
  let v3 := v2
  r0 := v3
  let v4 := (UInt256.ofNat 0)
  let v5 := v4
  r1 := v5
  let rec b1 (r0 : (Vector UInt256 25)) (r1 : UInt256) (r2 : (Vector UInt256 25)) (v0 : (Vector UInt256 25)) : FeM (Unit × (Vector UInt256 25)) := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut v0 := v0
    let v6 := r1
    let v7 := v6
    let v8 := (UInt256.ofNat 24)
    let v9 := decide (v7.toNat < v8.toNat)
    if v9 then
      let v10 := r1
      let v11 := v10
      let v12 := (UInt256.ofNat 2)
      let v13 ← checkedRem v11 v12
      let v14 := (UInt256.ofNat 0)
      let v15 := decide (v13 = v14)
      let b6 : (Vector UInt256 25) → UInt256 → (Vector UInt256 25) → (Vector UInt256 25) → FeM (Unit × (Vector UInt256 25)) := fun r0 r1 r2 v0 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut v0 := v0
        let v33 := (UInt256.ofNat 1)
        let v34 := r1
        let v35 ← checkedAdd v34 v33
        let v36 := v35
        r1 := v36
        let v37 := ()
        b1 r0 r1 r2 v0
      if v15 then
        let v18 ← ROUND_CONSTANTS
        let v19 := r1
        checkIndex v19 24
        let v20 := v18[v19.toNat]!
        let (v22, v22_0) ← keccak_round v0 r0 v20
        r0 := v22_0
        let v23 := v22
        b6 r0 r1 r2 v0
      else
        let v24 := r0
        let v25 := v24
        r2 := v25
        let v27 ← ROUND_CONSTANTS
        let v28 := r1
        checkIndex v28 24
        let v29 := v27[v28.toNat]!
        let (v31, v31_0) ← keccak_round r2 v0 v29
        v0 := v31_0
        let v32 := v31
        b6 r0 r1 r2 v0
    else
      let v38 := ()
      return ((), v0)
  partial_fixpoint
  b1 r0 r1 r2 v0

def Keccak256_advance (v0 : Keccak256) (v1 : UInt256) : FeM (Unit × Keccak256) := do
  let mut v0 := v0
  let v2 := v1
  let v3 := v0.position
  let v4 ← checkedAdd v3 v2
  v0 := { v0 with position := v4 }
  let v5 := ()
  let v6 := v0.position
  let v7 := (UInt256.ofNat 136)
  let v8 := decide (v6 = v7)
  let b3 : Keccak256 → FeM (Unit × Keccak256) := fun v0 => do
    let mut v0 := v0
    return ((), v0)
  if v8 then
    let (v10, v10_0) ← permute v0.lanes
    v0 := { v0 with lanes := v10_0 }
    let v11 := (UInt256.ofNat 0)
    v0 := { v0 with position := v11 }
    let v12 := ()
    let v13 := v12
    b3 v0
  else
    let v14 := ()
    let v15 := v14
    b3 v0

def Keccak256_update_word (v0 : Keccak256) (v1 : UInt256) (v2 : UInt256) : FeM (Unit × Keccak256) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut v0 := v0
  let v6 := v2
  let v7 := (UInt256.ofNat 32)
  let v8 := decide (v6.toNat ≤ v7.toNat)
  let v9 ← assert v8
  let v10 := v2
  let v11 := (UInt256.ofNat 0)
  let v12 := decide (v10 ≠ v11)
  let b2 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun r0 r1 v0 => do
    let mut r0 := r0
    let mut r1 := r1
    let mut v0 := v0
    let v129 := ()
    let v130 := v129
    let v131 := v1
    let v132 := (UInt256.ofNat 192)
    let v133 := v131 >>> v132
    let v134 ← downcast_truncate_u256_u64 v133
    let v135 := IntTy.swapBytes (.u 64) v134
    let v136 := v1
    let v137 := (UInt256.ofNat 128)
    let v138 := v136 >>> v137
    let v139 ← downcast_truncate_u256_u64 v138
    let v140 := IntTy.swapBytes (.u 64) v139
    let v141 := v1
    let v142 := (UInt256.ofNat 64)
    let v143 := v141 >>> v142
    let v144 ← downcast_truncate_u256_u64 v143
    let v145 := IntTy.swapBytes (.u 64) v144
    let v146 ← downcast_truncate_u256_u64 v1
    let v147 := IntTy.swapBytes (.u 64) v146
    let v148 := (UInt256.ofNat 0)
    let v149 := #v[v135, v140, v145, v147, v148]
    let v150 := v149
    let v151 := (UInt256.ofNat 32)
    let v152 := v2
    let v153 ← checkedSub v151 v152
    let v154 := v153
    r0 := v154
    let v155 := v2
    let v156 := v155
    r1 := v156
    let rec b11 (r0 : UInt256) (r1 : UInt256) (v0 : Keccak256) : FeM (Unit × Keccak256) := do
      let mut r0 := r0
      let mut r1 := r1
      let mut v0 := v0
      let v157 := r1
      let v158 := v157
      let v159 := (UInt256.ofNat 0)
      let v160 := decide (v159.toNat < v158.toNat)
      if v160 then
        let v161 := v0.position
        let v162 := (UInt256.ofNat 8)
        let v163 ← checkedRem v161 v162
        let v164 := v163
        let v165 := r1
        let v166 := v165
        let v167 := (UInt256.ofNat 8)
        let v168 ← checkedSub v167 v164
        let v169 := decide (v166.toNat < v168.toNat)
        let b16 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun r0 r1 v0 v3 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut v0 := v0
          let v176 := v3
          let v177 := r0
          let v178 := v177
          let v179 := (UInt256.ofNat 8)
          let v180 ← checkedRem v178 v179
          let v181 := (UInt256.ofNat 8)
          let v182 ← checkedMul v180 v181
          let v183 ← downcast_unchecked_usize_u64 v182
          let v184 := v183
          let v185 := r0
          let v186 := v185
          let v187 := (UInt256.ofNat 8)
          let v188 ← checkedDiv v186 v187
          checkIndex v188 5
          let v189 := v150[v188.toNat]!
          let v190 := v189 >>> v184
          let v191 := v190
          let v192 := (UInt256.ofNat 0)
          let v193 := decide (v184 = v192)
          let b19 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun r0 r1 v0 v4 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut v0 := v0
            let v207 := v4
            let v208 := (UInt256.ofNat 18446744073709551615)
            let v209 := v208
            let v210 := (UInt256.ofNat 8)
            let v211 := decide (v176 = v210)
            let b22 : UInt256 → UInt256 → Keccak256 → UInt256 → FeM (Unit × Keccak256) := fun r0 r1 v0 v5 => do
              let mut r0 := r0
              let mut r1 := r1
              let mut v0 := v0
              let v220 := v5
              let v221 := v191 ||| v207
              let v222 := v221 &&& v220
              let v223 := (UInt256.ofNat 8)
              let v224 ← checkedMul v164 v223
              let v225 ← downcast_unchecked_usize_u64 v224
              let v226 := IntTy.shl (.u 64) v222 v225
              let v227 := v0.position
              let v228 := (UInt256.ofNat 8)
              let v229 ← checkedDiv v227 v228
              checkIndex v229 25
              let v230 := v0.lanes[v229.toNat]!
              let v231 := v230 ^^^ v226
              checkIndex v229 25
              v0 := { v0 with lanes := (v0.lanes.set! v229.toNat v231) }
              let v232 := ()
              let v233 := r0
              let v234 ← checkedAdd v233 v176
              let v235 := v234
              r0 := v235
              let v236 := ()
              let v237 := r1
              let v238 ← checkedSub v237 v176
              let v239 := v238
              r1 := v239
              let v240 := ()
              let (v241, v241_0) ← Keccak256_advance v0 v176
              v0 := v241_0
              b11 r0 r1 v0
            if v211 then
              let v212 := v209
              b22 r0 r1 v0 v212
            else
              let v213 := (UInt256.ofNat 64)
              let v214 := (UInt256.ofNat 8)
              let v215 ← checkedMul v176 v214
              let v216 ← checkedSub v213 v215
              let v217 ← downcast_unchecked_usize_u64 v216
              let v218 := v209 >>> v217
              let v219 := v218
              b22 r0 r1 v0 v219
          if v193 then
            let v194 := (UInt256.ofNat 0)
            let v195 := v194
            b19 r0 r1 v0 v195
          else
            let v196 := r0
            let v197 := v196
            let v198 := (UInt256.ofNat 8)
            let v199 ← checkedDiv v197 v198
            let v200 := (UInt256.ofNat 1)
            let v201 ← checkedAdd v199 v200
            checkIndex v201 5
            let v202 := v150[v201.toNat]!
            let v203 := (UInt256.ofNat 64)
            let v204 ← IntTy.sub (.u 64) v203 v184
            let v205 := IntTy.shl (.u 64) v202 v204
            let v206 := v205
            b19 r0 r1 v0 v206
        if v169 then
          let v170 := r1
          let v171 := v170
          let v172 := v171
          b16 r0 r1 v0 v172
        else
          let v173 := (UInt256.ofNat 8)
          let v174 ← checkedSub v173 v164
          let v175 := v174
          b16 r0 r1 v0 v175
      else
        let v242 := ()
        return ((), v0)
    partial_fixpoint
    b11 r0 r1 v0
  if v12 then
    let v13 := v0.position
    let v14 := v2
    let v15 ← checkedAdd v13 v14
    let v16 := (UInt256.ofNat 136)
    let v17 := decide (v15.toNat ≤ v16.toNat)
    if v17 then
      let v18 := v1
      let v19 := (UInt256.ofNat 32)
      let v20 := v2
      let v21 ← checkedSub v19 v20
      let v22 := (UInt256.ofNat 8)
      let v23 ← checkedMul v21 v22
      let v24 := v23
      let v25 := v18 <<< v24
      let v26 := v25
      let v27 := (UInt256.ofNat 192)
      let v28 := v26 >>> v27
      let v29 ← downcast_truncate_u256_u64 v28
      let v30 := IntTy.swapBytes (.u 64) v29
      let v31 := v30
      let v32 := (UInt256.ofNat 128)
      let v33 := v26 >>> v32
      let v34 ← downcast_truncate_u256_u64 v33
      let v35 := IntTy.swapBytes (.u 64) v34
      let v36 := v35
      let v37 := (UInt256.ofNat 64)
      let v38 := v26 >>> v37
      let v39 ← downcast_truncate_u256_u64 v38
      let v40 := IntTy.swapBytes (.u 64) v39
      let v41 := v40
      let v42 ← downcast_truncate_u256_u64 v26
      let v43 := IntTy.swapBytes (.u 64) v42
      let v44 := v43
      let v45 := v0.position
      let v46 := (UInt256.ofNat 8)
      let v47 ← checkedDiv v45 v46
      let v48 := v47
      let v49 := v0.position
      let v50 := (UInt256.ofNat 8)
      let v51 ← checkedRem v49 v50
      let v52 := (UInt256.ofNat 8)
      let v53 ← checkedMul v51 v52
      let v54 ← downcast_unchecked_usize_u64 v53
      let v55 := v54
      let v56 := (UInt256.ofNat 0)
      let v57 := decide (v55 = v56)
      let b7 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun r0 r1 v0 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut v0 := v0
        let v115 := v2
        let v116 := v0.position
        let v117 ← checkedAdd v116 v115
        v0 := { v0 with position := v117 }
        let v118 := ()
        let v119 := v0.position
        let v120 := (UInt256.ofNat 136)
        let v121 := decide (v119 = v120)
        let b10 : UInt256 → UInt256 → Keccak256 → FeM (Unit × Keccak256) := fun r0 r1 v0 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut v0 := v0
          return ((), v0)
        if v121 then
          let (v123, v123_0) ← permute v0.lanes
          v0 := { v0 with lanes := v123_0 }
          let v124 := (UInt256.ofNat 0)
          v0 := { v0 with position := v124 }
          let v125 := ()
          let v126 := v125
          b10 r0 r1 v0
        else
          let v127 := ()
          let v128 := v127
          b10 r0 r1 v0
      if v57 then
        checkIndex v48 25
        let v58 := v0.lanes[v48.toNat]!
        let v59 := v58 ^^^ v31
        checkIndex v48 25
        v0 := { v0 with lanes := (v0.lanes.set! v48.toNat v59) }
        let v60 := ()
        let v61 := (UInt256.ofNat 1)
        let v62 ← checkedAdd v48 v61
        checkIndex v62 25
        let v63 := v0.lanes[v62.toNat]!
        let v64 := v63 ^^^ v36
        checkIndex v62 25
        v0 := { v0 with lanes := (v0.lanes.set! v62.toNat v64) }
        let v65 := ()
        let v66 := (UInt256.ofNat 2)
        let v67 ← checkedAdd v48 v66
        checkIndex v67 25
        let v68 := v0.lanes[v67.toNat]!
        let v69 := v68 ^^^ v41
        checkIndex v67 25
        v0 := { v0 with lanes := (v0.lanes.set! v67.toNat v69) }
        let v70 := ()
        let v71 := (UInt256.ofNat 3)
        let v72 ← checkedAdd v48 v71
        checkIndex v72 25
        let v73 := v0.lanes[v72.toNat]!
        let v74 := v73 ^^^ v44
        checkIndex v72 25
        v0 := { v0 with lanes := (v0.lanes.set! v72.toNat v74) }
        let v75 := ()
        let v76 := v75
        b7 r0 r1 v0
      else
        let v77 := (UInt256.ofNat 64)
        let v78 ← IntTy.sub (.u 64) v77 v55
        let v79 := v78
        let v80 := IntTy.shl (.u 64) v31 v55
        checkIndex v48 25
        let v81 := v0.lanes[v48.toNat]!
        let v82 := v81 ^^^ v80
        checkIndex v48 25
        v0 := { v0 with lanes := (v0.lanes.set! v48.toNat v82) }
        let v83 := ()
        let v84 := IntTy.shl (.u 64) v36 v55
        let v85 := v31 >>> v79
        let v86 := v84 ||| v85
        let v87 := (UInt256.ofNat 1)
        let v88 ← checkedAdd v48 v87
        checkIndex v88 25
        let v89 := v0.lanes[v88.toNat]!
        let v90 := v89 ^^^ v86
        checkIndex v88 25
        v0 := { v0 with lanes := (v0.lanes.set! v88.toNat v90) }
        let v91 := ()
        let v92 := IntTy.shl (.u 64) v41 v55
        let v93 := v36 >>> v79
        let v94 := v92 ||| v93
        let v95 := (UInt256.ofNat 2)
        let v96 ← checkedAdd v48 v95
        checkIndex v96 25
        let v97 := v0.lanes[v96.toNat]!
        let v98 := v97 ^^^ v94
        checkIndex v96 25
        v0 := { v0 with lanes := (v0.lanes.set! v96.toNat v98) }
        let v99 := ()
        let v100 := IntTy.shl (.u 64) v44 v55
        let v101 := v41 >>> v79
        let v102 := v100 ||| v101
        let v103 := (UInt256.ofNat 3)
        let v104 ← checkedAdd v48 v103
        checkIndex v104 25
        let v105 := v0.lanes[v104.toNat]!
        let v106 := v105 ^^^ v102
        checkIndex v104 25
        v0 := { v0 with lanes := (v0.lanes.set! v104.toNat v106) }
        let v107 := ()
        let v108 := v44 >>> v79
        let v109 := (UInt256.ofNat 4)
        let v110 ← checkedAdd v48 v109
        checkIndex v110 25
        let v111 := v0.lanes[v110.toNat]!
        let v112 := v111 ^^^ v108
        checkIndex v110 25
        v0 := { v0 with lanes := (v0.lanes.set! v110.toNat v112) }
        let v113 := ()
        let v114 := v113
        b7 r0 r1 v0
    else
      b2 r0 r1 v0
  else
    b2 r0 r1 v0

def Keccak256_pad (v0 : Keccak256) : FeM (Unit × Keccak256) := do
  let mut v0 := v0
  let v1 := (UInt256.ofNat 1)
  let v2 := v0.position
  let v3 := (UInt256.ofNat 8)
  let v4 ← checkedRem v2 v3
  let v5 := (UInt256.ofNat 8)
  let v6 ← checkedMul v4 v5
  let v7 ← downcast_unchecked_usize_u64 v6
  let v8 := IntTy.shl (.u 64) v1 v7
  let v9 := v0.position
  let v10 := (UInt256.ofNat 8)
  let v11 ← checkedDiv v9 v10
  checkIndex v11 25
  let v12 := v0.lanes[v11.toNat]!
  let v13 := v12 ^^^ v8
  checkIndex v11 25
  v0 := { v0 with lanes := (v0.lanes.set! v11.toNat v13) }
  let v14 := ()
  let v15 := (UInt256.ofNat 9223372036854775808)
  let v16 := (UInt256.ofNat 16)
  checkIndex v16 25
  let v17 := v0.lanes[v16.toNat]!
  let v18 := v17 ^^^ v15
  checkIndex v16 25
  v0 := { v0 with lanes := (v0.lanes.set! v16.toNat v18) }
  let v19 := ()
  let (v21, v21_0) ← permute v0.lanes
  v0 := { v0 with lanes := v21_0 }
  return ((), v0)

def Keccak256_finish_word (r0 : Keccak256) : FeM UInt256 := do
  let mut r0 := r0
  let (v1, v1_0) ← Keccak256_pad r0
  r0 := v1_0
  let v2 := (UInt256.ofNat 0)
  checkIndex v2 25
  let v3 := r0.lanes[v2.toNat]!
  let v4 := IntTy.swapBytes (.u 64) v3
  let v5 := v4
  let v6 := (UInt256.ofNat 192)
  let v7 := v5 <<< v6
  let v8 := (UInt256.ofNat 1)
  checkIndex v8 25
  let v9 := r0.lanes[v8.toNat]!
  let v10 := IntTy.swapBytes (.u 64) v9
  let v11 := v10
  let v12 := (UInt256.ofNat 128)
  let v13 := v11 <<< v12
  let v14 := v7 ||| v13
  let v15 := (UInt256.ofNat 2)
  checkIndex v15 25
  let v16 := r0.lanes[v15.toNat]!
  let v17 := IntTy.swapBytes (.u 64) v16
  let v18 := v17
  let v19 := (UInt256.ofNat 64)
  let v20 := v18 <<< v19
  let v21 := v14 ||| v20
  let v22 := (UInt256.ofNat 3)
  checkIndex v22 25
  let v23 := r0.lanes[v22.toNat]!
  let v24 := IntTy.swapBytes (.u 64) v23
  let v25 := v24
  let v26 := v21 ||| v25
  return v26

def recover_signer (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) (v3 : UInt256) : FeM Option_u256 := do
  let mut r0 : Keccak256 := default
  let v4 ← recover v0 v1 v2 v3
  match v4 with
  | .Some .. =>
    let v5 := v4.Some_0
    let v6 := v5
    let v7 ← Keccak256_new
    let v8 := v7
    r0 := v8
    let v9 := v6.x
    let v10 := (UInt256.ofNat 32)
    let (v12, v12_0) ← Keccak256_update_word r0 v9 v10
    r0 := v12_0
    let v13 := v6.y
    let v14 := (UInt256.ofNat 32)
    let (v16, v16_0) ← Keccak256_update_word r0 v13 v14
    r0 := v16_0
    let v17 := r0
    let v18 := v17
    let v19 ← Keccak256_finish_word v18
    let v20 := (UInt256.ofNat 1461501637330902918203684832716283019655932542975)
    let v21 := v19 &&& v20
    let v22 := Option_u256.Some v21
    return v22
  | _ =>
    let v23 := Option_u256.None
    return v23

def ecrecover_MemBuffer (v0 : MemBuffer) : FeM Output := do
  let mut r0 : (Vector UInt256 1) := default
  let v4 := (UInt256.ofNat 0)
  let v5 ← word_at_MemBuffer v0 v4
  let v6 := v5
  let v7 := (UInt256.ofNat 32)
  let v8 ← word_at_MemBuffer v0 v7
  let v9 := v8
  let v10 := (UInt256.ofNat 64)
  let v11 ← word_at_MemBuffer v0 v10
  let v12 := v11
  let v13 := (UInt256.ofNat 96)
  let v14 ← word_at_MemBuffer v0 v13
  let v15 := v14
  let v16 := (UInt256.ofNat 27)
  let v17 := decide (v9 ≠ v16)
  let b3 : (Vector UInt256 1) → Bool → FeM Output := fun r0 v1 => do
    let mut r0 := r0
    let v32 := v1
    let b11 : (Vector UInt256 1) → Option_u256 → FeM Output := fun r0 v2 => do
      let mut r0 := r0
      let v39 := v2
      let b12 : (Vector UInt256 1) → Output → FeM Output := fun r0 v3 => do
        let mut r0 := r0
        return v3
      match v39 with
      | .Some .. =>
        let v40 := v39.Some_0
        let v41 := v40
        let v42 := #v[v41]
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
    if v32 then
      let v33 := Option_u256.None
      let v34 := v33
      b11 r0 v34
    else
      let v35 := (UInt256.ofNat 27)
      let v36 ← checkedSub v9 v35
      let v37 ← recover_signer v6 v36 v12 v15
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
    let v21 := decide (v12 = v20)
    if v21 then
      b1 r0
    else
      let v22 ← N
      let v23 := decide (v22.toNat ≤ v12.toNat)
      if v23 then
        b1 r0
      else
        let v24 := (UInt256.ofNat 0)
        let v25 := decide (v15 = v24)
        if v25 then
          b1 r0
        else
          let v26 ← N
          let v27 := decide (v26.toNat ≤ v15.toNat)
          if v27 then
            b1 r0
          else
            let v28 := false
            let v29 := v28
            b3 r0 v29
  if v17 then
    let v18 := (UInt256.ofNat 28)
    let v19 := decide (v9 ≠ v18)
    if v19 then
      b1 r0
    else
      b7 r0
  else
    b7 r0

def recv0_arm0 : FeM Empty := do
  let mut r1 : Output := default
  let mut r2 : MemBuffer := default
  let v0 ← input_Evm_4
  r2 := v0
  let v2 ← ecrecover_MemBuffer r2
  let v3 := v2
  r1 := v3
  let _ ← MemBuffer_drop r2
  let v6 ← MemBuffer_span_ref r1.output
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


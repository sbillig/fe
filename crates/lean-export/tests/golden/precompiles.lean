export Lib (MemBuffer MemBuffer_index_ref MemBuffer_len MemBuffer_read_le_u64 downcast_truncate_i128_i64 downcast_truncate_i128_u64 downcast_truncate_i64_u64 downcast_truncate_u128_u64 downcast_truncate_u64_i64 u128_wrapping_sub)

structure Transition where
  u : UInt256
  v : UInt256
  q : UInt256
  r : UInt256
deriving Inhabited

def mul_wide (v0 : UInt256) (v1 : UInt256) : FeM (UInt256 × UInt256) := do
  let v2 := v0
  let v3 := v2
  let v4 := v1
  let v5 := v4
  let v6 ← IntTy.mul (.u 128) v3 v5
  let v7 := v6
  let v8 ← downcast_truncate_u128_u64 v7
  let v9 := (UInt256.ofNat 64)
  let v10 := v7 >>> v9
  let v11 ← downcast_truncate_u128_u64 v10
  let v12 := (v8, v11)
  return v12

def adc (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 := v0
  let v4 := v3
  let v5 := v1
  let v6 := v5
  let v7 ← IntTy.add (.u 128) v4 v6
  let v8 := v2
  let v9 := v8
  let v10 ← IntTy.add (.u 128) v7 v9
  let v11 := v10
  let v12 ← downcast_truncate_u128_u64 v11
  let v13 := (UInt256.ofNat 64)
  let v14 := v11 >>> v13
  let v15 ← downcast_truncate_u128_u64 v14
  let v16 := (v12, v15)
  return v16

def sbb (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × UInt256) := do
  let v3 := v0
  let v4 := v3
  let v5 := v1
  let v6 := v5
  let v7 ← u128_wrapping_sub v4 v6
  let v8 := v2
  let v9 := v8
  let v10 ← u128_wrapping_sub v7 v9
  let v11 := v10
  let v12 ← downcast_truncate_u128_u64 v11
  let v13 := (UInt256.ofNat 127)
  let v14 := v11 >>> v13
  let v15 ← downcast_truncate_u128_u64 v14
  let v16 := (v12, v15)
  return v16

def M62 : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4611686018427387903)
  return v0

def is_zero_5 (v0 : (Vector UInt256 5)) : FeM Bool := do
  let mut r0 : UInt256 := default
  let v1 := (UInt256.ofNat 0)
  let v2 := v1
  r0 := v2
  let rec b1 (r0 : UInt256) : FeM Bool := do
    let mut r0 := r0
    let v3 := r0
    let v4 := v3
    let v5 := (UInt256.ofNat 5)
    let v6 := decide (v4.toNat < v5.toNat)
    if v6 then
      let v7 := r0
      checkIndex v7 5
      let v8 := v0[v7.toNat]!
      let v9 := (UInt256.ofNat 0)
      let v10 := decide (v8 ≠ v9)
      if v10 then
        let v11 := false
        return v11
      else
        let v12 := ()
        let v13 := v12
        let v14 := (UInt256.ofNat 1)
        let v15 := r0
        let v16 ← checkedAdd v15 v14
        let v17 := v16
        r0 := v17
        let v18 := ()
        b1 r0
    else
      let v19 := true
      return v19
  partial_fixpoint
  b1 r0

def divsteps (v0 : UInt256) (v1 : UInt256) (v2 : UInt256) : FeM (UInt256 × Transition) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : UInt256 := default
  let mut r4 : UInt256 := default
  let mut r5 : UInt256 := default
  let mut r6 : UInt256 := default
  let mut r7 : UInt256 := default
  let mut r8 : UInt256 := default
  let v5 := (UInt256.ofNat 1)
  let v6 := v5
  r0 := v6
  let v7 := (UInt256.ofNat 0)
  let v8 := v7
  r1 := v8
  let v9 := (UInt256.ofNat 0)
  let v10 := v9
  r2 := v10
  let v11 := (UInt256.ofNat 1)
  let v12 := v11
  r3 := v12
  let v13 := v1
  let v14 := v13
  r4 := v14
  let v15 := v2
  let v16 := v15
  r5 := v16
  let v17 := v0
  let v18 := v17
  r6 := v18
  let v19 := (UInt256.ofNat 62)
  let v20 := v19
  r7 := v20
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : UInt256) (r5 : UInt256) (r6 : UInt256) (r7 : UInt256) (r8 : UInt256) : FeM (UInt256 × Transition) := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    let v21 := r5
    let v22 := v21
    let v23 := (UInt256.ofNat 18446744073709551615)
    let v24 := r7
    let v25 := IntTy.shl (.u 64) v23 v24
    let v26 := v22 ||| v25
    let v27 := IntTy.trailingZeros (.u 64) v26
    let v28 := v27
    let v29 := r5
    let v30 := v29
    let v31 := v30 >>> v28
    let v32 := v31
    r5 := v32
    let v33 := ()
    let v34 := r0
    let v35 := v34
    let v36 := IntTy.shl (.u 64) v35 v28
    let v37 := v36
    r0 := v37
    let v38 := ()
    let v39 := r1
    let v40 := v39
    let v41 := IntTy.shl (.u 64) v40 v28
    let v42 := v41
    r1 := v42
    let v43 := ()
    let v44 ← downcast_truncate_u64_i64 v28
    let v45 := r6
    let v46 := IntTy.subW (.i 64) v45 v44
    let v47 := v46
    r6 := v47
    let v48 := ()
    let v49 := r7
    let v50 := IntTy.subW (.u 64) v49 v28
    let v51 := v50
    r7 := v51
    let v52 := ()
    let v53 := r7
    let v54 := v53
    let v55 := (UInt256.ofNat 0)
    let v56 := decide (v54 = v55)
    if v56 then
      let v57 := r6
      let v58 := v57
      let v59 := r0
      let v60 := v59
      let v61 ← downcast_truncate_u64_i64 v60
      let v62 := r1
      let v63 := v62
      let v64 ← downcast_truncate_u64_i64 v63
      let v65 := r2
      let v66 := v65
      let v67 ← downcast_truncate_u64_i64 v66
      let v68 := r3
      let v69 := v68
      let v70 ← downcast_truncate_u64_i64 v69
      let v71 := (Transition.mk v61 v64 v67 v70)
      let v72 := (v58, v71)
      return v72
    else
      let v73 := ()
      let v74 := v73
      let v75 := (UInt256.ofNat 15)
      let v76 := v75
      r8 := v76
      let v77 := r6
      let v78 := v77
      let v79 := (UInt256.ofNat 0)
      let v80 := UInt256.sltBool v78 v79
      let b9 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 => do
        let mut r0 := r0
        let mut r1 := r1
        let mut r2 := r2
        let mut r3 := r3
        let mut r4 := r4
        let mut r5 := r5
        let mut r6 := r6
        let mut r7 := r7
        let mut r8 := r8
        let v122 := r6
        let v123 := v122
        let v124 := (UInt256.ofNat 1)
        let v125 := IntTy.addW (.i 64) v123 v124
        let v126 ← downcast_truncate_i64_u64 v125
        let v127 := v126
        let v128 := r7
        let v129 := decide (v127.toNat < v128.toNat)
        let b12 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v3 => do
          let mut r0 := r0
          let mut r1 := r1
          let mut r2 := r2
          let mut r3 := r3
          let mut r4 := r4
          let mut r5 := r5
          let mut r6 := r6
          let mut r7 := r7
          let mut r8 := r8
          let v134 := v3
          let v135 := (UInt256.ofNat 18446744073709551615)
          let v136 := (UInt256.ofNat 64)
          let v137 := IntTy.subW (.u 64) v136 v134
          let v138 := v135 >>> v137
          let v139 := r8
          let v140 := v138 &&& v139
          let v141 := v140
          let v142 := r8
          let v143 := v142
          let v144 := (UInt256.ofNat 63)
          let v145 := decide (v143 = v144)
          let b15 : UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → UInt256 → FeM (UInt256 × Transition) := fun r0 r1 r2 r3 r4 r5 r6 r7 r8 v4 => do
            let mut r0 := r0
            let mut r1 := r1
            let mut r2 := r2
            let mut r3 := r3
            let mut r4 := r4
            let mut r5 := r5
            let mut r6 := r6
            let mut r7 := r7
            let mut r8 := r8
            let v176 := v4
            let v177 := r4
            let v178 := v177
            let v179 := IntTy.mulW (.u 64) v178 v176
            let v180 := r5
            let v181 := IntTy.addW (.u 64) v180 v179
            let v182 := v181
            r5 := v182
            let v183 := ()
            let v184 := r0
            let v185 := v184
            let v186 := IntTy.mulW (.u 64) v185 v176
            let v187 := r2
            let v188 := IntTy.addW (.u 64) v187 v186
            let v189 := v188
            r2 := v189
            let v190 := ()
            let v191 := r1
            let v192 := v191
            let v193 := IntTy.mulW (.u 64) v192 v176
            let v194 := r3
            let v195 := IntTy.addW (.u 64) v194 v193
            let v196 := v195
            r3 := v196
            let v197 := ()
            b1 r0 r1 r2 r3 r4 r5 r6 r7 r8
          if v145 then
            let v146 := r4
            let v147 := v146
            let v148 := r5
            let v149 := IntTy.mulW (.u 64) v147 v148
            let v150 := r4
            let v151 := v150
            let v152 := r4
            let v153 := IntTy.mulW (.u 64) v151 v152
            let v154 := (UInt256.ofNat 2)
            let v155 := IntTy.subW (.u 64) v153 v154
            let v156 := IntTy.mulW (.u 64) v149 v155
            let v157 := v156 &&& v141
            let v158 := v157
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 v158
          else
            let v159 := (UInt256.ofNat 0)
            let v160 := r4
            let v161 := v160
            let v162 := r4
            let v163 := v162
            let v164 := (UInt256.ofNat 1)
            let v165 := IntTy.addW (.u 64) v163 v164
            let v166 := (UInt256.ofNat 4)
            let v167 := v165 &&& v166
            let v168 := (UInt256.ofNat 1)
            let v169 := IntTy.shl (.u 64) v167 v168
            let v170 := IntTy.addW (.u 64) v161 v169
            let v171 := IntTy.subW (.u 64) v159 v170
            let v172 := r5
            let v173 := IntTy.mulW (.u 64) v171 v172
            let v174 := v173 &&& v141
            let v175 := v174
            b15 r0 r1 r2 r3 r4 r5 r6 r7 r8 v175
        if v129 then
          let v130 := v127
          b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 v130
        else
          let v131 := r7
          let v132 := v131
          let v133 := v132
          b12 r0 r1 r2 r3 r4 r5 r6 r7 r8 v133
      if v80 then
        let v81 := (UInt256.ofNat 0)
        let v82 := r6
        let v83 := IntTy.subW (.i 64) v81 v82
        let v84 := v83
        r6 := v84
        let v85 := ()
        let v86 := r4
        let v87 := v86
        let v88 := r5
        let v89 := v88
        let v90 := v89
        r4 := v90
        let v91 := ()
        let v92 := (UInt256.ofNat 0)
        let v93 := IntTy.subW (.u 64) v92 v87
        let v94 := v93
        r5 := v94
        let v95 := ()
        let v96 := r0
        let v97 := v96
        let v98 := r2
        let v99 := v98
        let v100 := v99
        r0 := v100
        let v101 := ()
        let v102 := (UInt256.ofNat 0)
        let v103 := IntTy.subW (.u 64) v102 v97
        let v104 := v103
        r2 := v104
        let v105 := ()
        let v106 := r1
        let v107 := v106
        let v108 := r3
        let v109 := v108
        let v110 := v109
        r1 := v110
        let v111 := ()
        let v112 := (UInt256.ofNat 0)
        let v113 := IntTy.subW (.u 64) v112 v107
        let v114 := v113
        r3 := v114
        let v115 := ()
        let v116 := (UInt256.ofNat 63)
        let v117 := v116
        r8 := v117
        let v118 := ()
        let v119 := v118
        b9 r0 r1 r2 r3 r4 r5 r6 r7 r8
      else
        let v120 := ()
        let v121 := v120
        b9 r0 r1 r2 r3 r4 r5 r6 r7 r8
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8

def M62_SIGNED : FeM UInt256 := do
  let v0 := (UInt256.ofNat 4611686018427387903)
  return v0

def update_de_5_k6 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v130 : (Vector UInt256 5)) (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let mut r5 := r5
  let mut r6 := r6
  let v131 := v130
  r4 := v131
  let v132 := (UInt256.ofNat 0)
  let v133 := Vector.replicate 5 v132
  let v134 := v133
  r5 := v134
  let v135 := (UInt256.ofNat 1)
  let v136 := v135
  r6 := v136
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let v137 := r6
    let v138 := v137
    let v139 := (UInt256.ofNat 5)
    let v140 := decide (v138.toNat < v139.toNat)
    if v140 then
      let v141 := v2.u
      let v142 := v141
      let v143 := r6
      checkIndex v143 5
      let v144 := v0[v143.toNat]!
      let v145 := v144
      let v146 := IntTy.mulW (.i 128) v142 v145
      let v147 := v2.v
      let v148 := v147
      let v149 := r6
      checkIndex v149 5
      let v150 := v1[v149.toNat]!
      let v151 := v150
      let v152 := IntTy.mulW (.i 128) v148 v151
      let v153 := IntTy.addW (.i 128) v146 v152
      let v154 := r6
      checkIndex v154 5
      let v155 := v3[v154.toNat]!
      let v156 := v155
      let v157 := r0
      let v158 := v157
      let v159 := IntTy.mulW (.i 128) v156 v158
      let v160 := IntTy.addW (.i 128) v153 v159
      let v161 := r2
      let v162 := IntTy.addW (.i 128) v161 v160
      let v163 := v162
      r2 := v163
      let v164 := ()
      let v165 := v2.q
      let v166 := v165
      let v167 := r6
      checkIndex v167 5
      let v168 := v0[v167.toNat]!
      let v169 := v168
      let v170 := IntTy.mulW (.i 128) v166 v169
      let v171 := v2.r
      let v172 := v171
      let v173 := r6
      checkIndex v173 5
      let v174 := v1[v173.toNat]!
      let v175 := v174
      let v176 := IntTy.mulW (.i 128) v172 v175
      let v177 := IntTy.addW (.i 128) v170 v176
      let v178 := r6
      checkIndex v178 5
      let v179 := v3[v178.toNat]!
      let v180 := v179
      let v181 := r1
      let v182 := v181
      let v183 := IntTy.mulW (.i 128) v180 v182
      let v184 := IntTy.addW (.i 128) v177 v183
      let v185 := r3
      let v186 := IntTy.addW (.i 128) v185 v184
      let v187 := v186
      r3 := v187
      let v188 := ()
      let v189 := r2
      let v190 := v189
      let v191 ← downcast_truncate_i128_i64 v190
      let v192 := v191
      let v193 := r3
      let v194 := v193
      let v195 ← downcast_truncate_i128_i64 v194
      let v196 := v195
      let v197 ← M62_SIGNED
      let v198 := v192 &&& v197
      let v199 := r6
      let v200 := v199
      let v201 := (UInt256.ofNat 1)
      let v202 := v200 - v201
      checkIndex v202 5
      r4 := (r4.set! v202.toNat v198)
      let v203 := ()
      let v204 ← M62_SIGNED
      let v205 := v196 &&& v204
      let v206 := r6
      let v207 := v206
      let v208 := (UInt256.ofNat 1)
      let v209 := v207 - v208
      checkIndex v209 5
      r5 := (r5.set! v209.toNat v205)
      let v210 := ()
      let v211 := r2
      let v212 := v211
      let v213 := (UInt256.ofNat 62)
      let v214 := IntTy.shr (.i 128) v212 v213
      let v215 := v214
      r2 := v215
      let v216 := ()
      let v217 := r3
      let v218 := v217
      let v219 := (UInt256.ofNat 62)
      let v220 := IntTy.shr (.i 128) v218 v219
      let v221 := v220
      r3 := v221
      let v222 := ()
      let v223 := (UInt256.ofNat 1)
      let v224 := r6
      let v225 := v224 + v223
      let v226 := v225
      r6 := v226
      let v227 := ()
      b1 r0 r1 r2 r3 r4 r5 r6
    else
      let v228 := r2
      let v229 := v228
      let v230 ← downcast_truncate_i128_i64 v229
      let v231 := (UInt256.ofNat 5)
      let v232 := (UInt256.ofNat 1)
      let v233 := v231 - v232
      checkIndex v233 5
      r4 := (r4.set! v233.toNat v230)
      let v234 := ()
      let v235 := r3
      let v236 := v235
      let v237 ← downcast_truncate_i128_i64 v236
      let v238 := (UInt256.ofNat 5)
      let v239 := (UInt256.ofNat 1)
      let v240 := v238 - v239
      checkIndex v240 5
      r5 := (r5.set! v240.toNat v237)
      let v241 := ()
      let v242 := r4
      let v243 := v242
      let v244 := r5
      let v245 := (v243, v244)
      return v245
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6

def update_de_5_k5 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v109 : UInt256) (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let mut r5 := r5
  let mut r6 := r6
  let v110 := r1
  let v111 := v110
  let v112 := IntTy.mulW (.i 128) v109 v111
  let v113 := r3
  let v114 := IntTy.addW (.i 128) v113 v112
  let v115 := v114
  r3 := v115
  let v116 := ()
  let v117 := r2
  let v118 := v117
  let v119 := (UInt256.ofNat 62)
  let v120 := IntTy.shr (.i 128) v118 v119
  let v121 := v120
  r2 := v121
  let v122 := ()
  let v123 := r3
  let v124 := v123
  let v125 := (UInt256.ofNat 62)
  let v126 := IntTy.shr (.i 128) v124 v125
  let v127 := v126
  r3 := v127
  let v128 := ()
  let v129 := (UInt256.ofNat 0)
  let v130 := Vector.replicate 5 v129
  update_de_5_k6 v0 v1 v2 v3 v130 r0 r1 r2 r3 r4 r5 r6

def update_de_5_k4 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v88 : UInt256) (v90 : UInt256) (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let mut r5 := r5
  let mut r6 := r6
  let v91 := v90
  r0 := v91
  let v92 := ()
  let v93 := r1
  let v94 := IntTy.subW (.i 64) v93 v88
  let v95 := v94
  r1 := v95
  let v96 := ()
  let v97 := (UInt256.ofNat 0)
  checkIndex v97 5
  let v98 := v3[v97.toNat]!
  let v99 := v98
  let v100 := r0
  let v101 := v100
  let v102 := IntTy.mulW (.i 128) v99 v101
  let v103 := r2
  let v104 := IntTy.addW (.i 128) v103 v102
  let v105 := v104
  r2 := v105
  let v106 := ()
  let v107 := (UInt256.ofNat 0)
  checkIndex v107 5
  let v108 := v3[v107.toNat]!
  let v109 := v108
  update_de_5_k5 v0 v1 v2 v3 v109 r0 r1 r2 r3 r4 r5 r6

def update_de_5_k3 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v4 : UInt256) (v62 : UInt256) (v66 : UInt256) (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let mut r5 := r5
  let mut r6 := r6
  let v67 := r0
  let v68 := v67
  let v69 ← downcast_truncate_i64_u64 v68
  let v70 := v69
  let v71 := r1
  let v72 := v71
  let v73 ← downcast_truncate_i64_u64 v72
  let v74 := v73
  let v75 := v4
  let v76 := IntTy.mulW (.u 64) v75 v62
  let v77 := IntTy.addW (.u 64) v76 v70
  let v78 ← M62
  let v79 := v77 &&& v78
  let v80 ← downcast_truncate_u64_i64 v79
  let v81 := v80
  let v82 := v4
  let v83 := IntTy.mulW (.u 64) v82 v66
  let v84 := IntTy.addW (.u 64) v83 v74
  let v85 ← M62
  let v86 := v84 &&& v85
  let v87 ← downcast_truncate_u64_i64 v86
  let v88 := v87
  let v89 := r0
  let v90 := IntTy.subW (.i 64) v89 v81
  update_de_5_k4 v0 v1 v2 v3 v88 v90 r0 r1 r2 r3 r4 r5 r6

def update_de_5_k2 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v4 : UInt256) (v45 : UInt256) (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let mut r5 := r5
  let mut r6 := r6
  let v46 := v45
  let v47 := (UInt256.ofNat 0)
  checkIndex v47 5
  let v48 := v0[v47.toNat]!
  let v49 := v48
  let v50 := IntTy.mulW (.i 128) v46 v49
  let v51 := v2.r
  let v52 := v51
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let v54 := v1[v53.toNat]!
  let v55 := v54
  let v56 := IntTy.mulW (.i 128) v52 v55
  let v57 := IntTy.addW (.i 128) v50 v56
  let v58 := v57
  r3 := v58
  let v59 := r2
  let v60 := v59
  let v61 ← downcast_truncate_i128_u64 v60
  let v62 := v61
  let v63 := r3
  let v64 := v63
  let v65 ← downcast_truncate_i128_u64 v64
  let v66 := v65
  update_de_5_k3 v0 v1 v2 v3 v4 v62 v66 r0 r1 r2 r3 r4 r5 r6

def update_de_5_k1 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v4 : UInt256) (v11 : UInt256) (v18 : UInt256) (v25 : UInt256) (r0 : UInt256) (r1 : UInt256) (r2 : UInt256) (r3 : UInt256) (r4 : (Vector UInt256 5)) (r5 : (Vector UInt256 5)) (r6 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let mut r5 := r5
  let mut r6 := r6
  let v26 := v25 &&& v11
  let v27 := v2.r
  let v28 := v27 &&& v18
  let v29 := IntTy.addW (.i 64) v26 v28
  let v30 := v29
  r1 := v30
  let v31 := v2.u
  let v32 := v31
  let v33 := (UInt256.ofNat 0)
  checkIndex v33 5
  let v34 := v0[v33.toNat]!
  let v35 := v34
  let v36 := IntTy.mulW (.i 128) v32 v35
  let v37 := v2.v
  let v38 := v37
  let v39 := (UInt256.ofNat 0)
  checkIndex v39 5
  let v40 := v1[v39.toNat]!
  let v41 := v40
  let v42 := IntTy.mulW (.i 128) v38 v41
  let v43 := IntTy.addW (.i 128) v36 v42
  let v44 := v43
  r2 := v44
  let v45 := v2.q
  update_de_5_k2 v0 v1 v2 v3 v4 v45 r0 r1 r2 r3 r4 r5 r6

def update_de_5 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v3 : (Vector UInt256 5)) (v4 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : UInt256 := default
  let mut r3 : UInt256 := default
  let mut r4 : (Vector UInt256 5) := default
  let mut r5 : (Vector UInt256 5) := default
  let mut r6 : UInt256 := default
  let v5 := (UInt256.ofNat 5)
  let v6 := (UInt256.ofNat 1)
  let v7 := v5 - v6
  checkIndex v7 5
  let v8 := v0[v7.toNat]!
  let v9 := (UInt256.ofNat 63)
  let v10 := IntTy.shr (.i 64) v8 v9
  let v11 := v10
  let v12 := (UInt256.ofNat 5)
  let v13 := (UInt256.ofNat 1)
  let v14 := v12 - v13
  checkIndex v14 5
  let v15 := v1[v14.toNat]!
  let v16 := (UInt256.ofNat 63)
  let v17 := IntTy.shr (.i 64) v15 v16
  let v18 := v17
  let v19 := v2.u
  let v20 := v19 &&& v11
  let v21 := v2.v
  let v22 := v21 &&& v18
  let v23 := IntTy.addW (.i 64) v20 v22
  let v24 := v23
  r0 := v24
  let v25 := v2.q
  update_de_5_k1 v0 v1 v2 v3 v4 v11 v18 v25 r0 r1 r2 r3 r4 r5 r6

def update_fg_5_k2 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (r0 : UInt256) (r1 : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let v43 := (UInt256.ofNat 0)
  let v44 := Vector.replicate 5 v43
  let v45 := v44
  r2 := v45
  let v46 := (UInt256.ofNat 0)
  let v47 := Vector.replicate 5 v46
  let v48 := v47
  r3 := v48
  let v49 := (UInt256.ofNat 1)
  let v50 := v49
  r4 := v50
  let rec b1 (r0 : UInt256) (r1 : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let v51 := r4
    let v52 := v51
    let v53 := (UInt256.ofNat 5)
    let v54 := decide (v52.toNat < v53.toNat)
    if v54 then
      let v55 := v2.u
      let v56 := v55
      let v57 := r4
      checkIndex v57 5
      let v58 := v0[v57.toNat]!
      let v59 := v58
      let v60 := IntTy.mulW (.i 128) v56 v59
      let v61 := v2.v
      let v62 := v61
      let v63 := r4
      checkIndex v63 5
      let v64 := v1[v63.toNat]!
      let v65 := v64
      let v66 := IntTy.mulW (.i 128) v62 v65
      let v67 := IntTy.addW (.i 128) v60 v66
      let v68 := r0
      let v69 := IntTy.addW (.i 128) v68 v67
      let v70 := v69
      r0 := v70
      let v71 := ()
      let v72 := v2.q
      let v73 := v72
      let v74 := r4
      checkIndex v74 5
      let v75 := v0[v74.toNat]!
      let v76 := v75
      let v77 := IntTy.mulW (.i 128) v73 v76
      let v78 := v2.r
      let v79 := v78
      let v80 := r4
      checkIndex v80 5
      let v81 := v1[v80.toNat]!
      let v82 := v81
      let v83 := IntTy.mulW (.i 128) v79 v82
      let v84 := IntTy.addW (.i 128) v77 v83
      let v85 := r1
      let v86 := IntTy.addW (.i 128) v85 v84
      let v87 := v86
      r1 := v87
      let v88 := ()
      let v89 := r0
      let v90 := v89
      let v91 ← downcast_truncate_i128_i64 v90
      let v92 := v91
      let v93 := r1
      let v94 := v93
      let v95 ← downcast_truncate_i128_i64 v94
      let v96 := v95
      let v97 ← M62_SIGNED
      let v98 := v92 &&& v97
      let v99 := r4
      let v100 := v99
      let v101 := (UInt256.ofNat 1)
      let v102 := v100 - v101
      checkIndex v102 5
      r2 := (r2.set! v102.toNat v98)
      let v103 := ()
      let v104 ← M62_SIGNED
      let v105 := v96 &&& v104
      let v106 := r4
      let v107 := v106
      let v108 := (UInt256.ofNat 1)
      let v109 := v107 - v108
      checkIndex v109 5
      r3 := (r3.set! v109.toNat v105)
      let v110 := ()
      let v111 := r0
      let v112 := v111
      let v113 := (UInt256.ofNat 62)
      let v114 := IntTy.shr (.i 128) v112 v113
      let v115 := v114
      r0 := v115
      let v116 := ()
      let v117 := r1
      let v118 := v117
      let v119 := (UInt256.ofNat 62)
      let v120 := IntTy.shr (.i 128) v118 v119
      let v121 := v120
      r1 := v121
      let v122 := ()
      let v123 := (UInt256.ofNat 1)
      let v124 := r4
      let v125 := v124 + v123
      let v126 := v125
      r4 := v126
      let v127 := ()
      b1 r0 r1 r2 r3 r4
    else
      let v128 := r0
      let v129 := v128
      let v130 ← downcast_truncate_i128_i64 v129
      let v131 := (UInt256.ofNat 5)
      let v132 := (UInt256.ofNat 1)
      let v133 := v131 - v132
      checkIndex v133 5
      r2 := (r2.set! v133.toNat v130)
      let v134 := ()
      let v135 := r1
      let v136 := v135
      let v137 ← downcast_truncate_i128_i64 v136
      let v138 := (UInt256.ofNat 5)
      let v139 := (UInt256.ofNat 1)
      let v140 := v138 - v139
      checkIndex v140 5
      r3 := (r3.set! v140.toNat v137)
      let v141 := ()
      let v142 := r2
      let v143 := v142
      let v144 := r3
      let v145 := (v143, v144)
      return v145
  partial_fixpoint
  b1 r0 r1 r2 r3 r4

def update_fg_5_k1 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) (v22 : UInt256) (r0 : UInt256) (r1 : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : UInt256) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 := r0
  let mut r1 := r1
  let mut r2 := r2
  let mut r3 := r3
  let mut r4 := r4
  let v23 := v2.r
  let v24 := v23
  let v25 := (UInt256.ofNat 0)
  checkIndex v25 5
  let v26 := v1[v25.toNat]!
  let v27 := v26
  let v28 := IntTy.mulW (.i 128) v24 v27
  let v29 := IntTy.addW (.i 128) v22 v28
  let v30 := v29
  r1 := v30
  let v31 := r0
  let v32 := v31
  let v33 := (UInt256.ofNat 62)
  let v34 := IntTy.shr (.i 128) v32 v33
  let v35 := v34
  r0 := v35
  let v36 := ()
  let v37 := r1
  let v38 := v37
  let v39 := (UInt256.ofNat 62)
  let v40 := IntTy.shr (.i 128) v38 v39
  let v41 := v40
  r1 := v41
  let v42 := ()
  update_fg_5_k2 v0 v1 v2 r0 r1 r2 r3 r4

def update_fg_5 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : Transition) : FeM ((Vector UInt256 5) × (Vector UInt256 5)) := do
  let mut r0 : UInt256 := default
  let mut r1 : UInt256 := default
  let mut r2 : (Vector UInt256 5) := default
  let mut r3 : (Vector UInt256 5) := default
  let mut r4 : UInt256 := default
  let v3 := v2.u
  let v4 := v3
  let v5 := (UInt256.ofNat 0)
  checkIndex v5 5
  let v6 := v0[v5.toNat]!
  let v7 := v6
  let v8 := IntTy.mulW (.i 128) v4 v7
  let v9 := v2.v
  let v10 := v9
  let v11 := (UInt256.ofNat 0)
  checkIndex v11 5
  let v12 := v1[v11.toNat]!
  let v13 := v12
  let v14 := IntTy.mulW (.i 128) v10 v13
  let v15 := IntTy.addW (.i 128) v8 v14
  let v16 := v15
  r0 := v16
  let v17 := v2.q
  let v18 := v17
  let v19 := (UInt256.ofNat 0)
  checkIndex v19 5
  let v20 := v0[v19.toNat]!
  let v21 := v20
  let v22 := IntTy.mulW (.i 128) v18 v21
  update_fg_5_k1 v0 v1 v2 v22 r0 r1 r2 r3 r4

def add_if_negative_5 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r0 : (Vector UInt256 5) := default
  let mut r1 : UInt256 := default
  let v2 := v0
  let v3 := v2
  r0 := v3
  let v4 := (UInt256.ofNat 5)
  let v5 := (UInt256.ofNat 1)
  let v6 := v4 - v5
  checkIndex v6 5
  let v7 := r0[v6.toNat]!
  let v8 := (UInt256.ofNat 63)
  let v9 := IntTy.shr (.i 64) v7 v8
  let v10 := v9
  let v11 := (UInt256.ofNat 0)
  let v12 := v11
  r1 := v12
  let rec b1 (r0 : (Vector UInt256 5)) (r1 : UInt256) : FeM (Vector UInt256 5) := do
    let mut r0 := r0
    let mut r1 := r1
    let v13 := r1
    let v14 := v13
    let v15 := (UInt256.ofNat 5)
    let v16 := decide (v14.toNat < v15.toNat)
    if v16 then
      let v17 := r1
      checkIndex v17 5
      let v18 := v1[v17.toNat]!
      let v19 := v18 &&& v10
      let v20 := r1
      checkIndex v20 5
      let v21 := r0[v20.toNat]!
      let v22 := IntTy.addW (.i 64) v21 v19
      let v23 := r1
      checkIndex v23 5
      r0 := (r0.set! v23.toNat v22)
      let v24 := ()
      let v25 := (UInt256.ofNat 1)
      let v26 := r1
      let v27 := v26 + v25
      let v28 := v27
      r1 := v28
      let v29 := ()
      b1 r0 r1
    else
      let v30 := r0
      return v30
  partial_fixpoint
  b1 r0 r1

def carry_5 (v0 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r0 : (Vector UInt256 5) := default
  let mut r1 : UInt256 := default
  let v1 := v0
  let v2 := v1
  r0 := v2
  let v3 := (UInt256.ofNat 0)
  let v4 := v3
  r1 := v4
  let rec b1 (r0 : (Vector UInt256 5)) (r1 : UInt256) : FeM (Vector UInt256 5) := do
    let mut r0 := r0
    let mut r1 := r1
    let v5 := r1
    let v6 := v5
    let v7 := (UInt256.ofNat 1)
    let v8 := v6 + v7
    let v9 := (UInt256.ofNat 5)
    let v10 := decide (v8.toNat < v9.toNat)
    if v10 then
      let v11 := r1
      checkIndex v11 5
      let v12 := r0[v11.toNat]!
      let v13 := (UInt256.ofNat 62)
      let v14 := IntTy.shr (.i 64) v12 v13
      let v15 := r1
      let v16 := v15
      let v17 := (UInt256.ofNat 1)
      let v18 := v16 + v17
      checkIndex v18 5
      let v19 := r0[v18.toNat]!
      let v20 := IntTy.addW (.i 64) v19 v14
      checkIndex v18 5
      r0 := (r0.set! v18.toNat v20)
      let v21 := ()
      let v22 ← M62_SIGNED
      let v23 := r1
      checkIndex v23 5
      let v24 := r0[v23.toNat]!
      let v25 := v24 &&& v22
      let v26 := r1
      checkIndex v26 5
      r0 := (r0.set! v26.toNat v25)
      let v27 := ()
      let v28 := (UInt256.ofNat 1)
      let v29 := r1
      let v30 := v29 + v28
      let v31 := v30
      r1 := v31
      let v32 := ()
      b1 r0 r1
    else
      let v33 := r0
      return v33
  partial_fixpoint
  b1 r0 r1

def normalize_5 (v0 : (Vector UInt256 5)) (v1 : UInt256) (v2 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
  let mut r0 : (Vector UInt256 5) := default
  let mut r1 : UInt256 := default
  let mut r2 : (Vector UInt256 5) := default
  let mut r3 : (Vector UInt256 5) := default
  let mut r4 : (Vector UInt256 5) := default
  let v3 ← add_if_negative_5 v0 v2
  let v4 := v3
  r0 := v4
  let v5 := v1
  let v6 := (UInt256.ofNat 63)
  let v7 := IntTy.shr (.i 64) v5 v6
  let v8 := v7
  let v9 := (UInt256.ofNat 0)
  let v10 := v9
  r1 := v10
  let rec b1 (r0 : (Vector UInt256 5)) (r1 : UInt256) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let v11 := r1
    let v12 := v11
    let v13 := (UInt256.ofNat 5)
    let v14 := decide (v12.toNat < v13.toNat)
    if v14 then
      let v15 := r1
      checkIndex v15 5
      let v16 := r0[v15.toNat]!
      let v17 := v16 ^^^ v8
      let v18 := IntTy.subW (.i 64) v17 v8
      let v19 := r1
      checkIndex v19 5
      r0 := (r0.set! v19.toNat v18)
      let v20 := ()
      let v21 := (UInt256.ofNat 1)
      let v22 := r1
      let v23 := v22 + v21
      let v24 := v23
      r1 := v24
      let v25 := ()
      b1 r0 r1 r2 r3 r4
    else
      let v26 := r0
      let v27 := v26
      r2 := v27
      let v29 ← carry_5 r2
      r3 := v29
      let v31 ← add_if_negative_5 r3 v2
      r4 := v31
      let v33 ← carry_5 r4
      return v33
  partial_fixpoint
  b1 r0 r1 r2 r3 r4

def invert_5 (v0 : (Vector UInt256 5)) (v1 : (Vector UInt256 5)) (v2 : UInt256) : FeM (Vector UInt256 5) := do
  let mut r0 : (Vector UInt256 5) := default
  let mut r1 : (Vector UInt256 5) := default
  let mut r2 : (Vector UInt256 5) := default
  let mut r3 : (Vector UInt256 5) := default
  let mut r4 : UInt256 := default
  let mut r5 : Transition := default
  let mut r6 : (Vector UInt256 5) := default
  let mut r7 : (Vector UInt256 5) := default
  let mut r8 : (Vector UInt256 5) := default
  let mut r9 : (Vector UInt256 5) := default
  let mut r10 : (Vector UInt256 5) := default
  let mut r11 : (Vector UInt256 5) := default
  let v3 := (UInt256.ofNat 0)
  let v4 := Vector.replicate 5 v3
  let v5 := v4
  r0 := v5
  let v6 := (UInt256.ofNat 0)
  let v7 := Vector.replicate 5 v6
  let v8 := v7
  r1 := v8
  let v9 := (UInt256.ofNat 1)
  let v10 := (UInt256.ofNat 0)
  checkIndex v10 5
  r1 := (r1.set! v10.toNat v9)
  let v11 := ()
  let v12 := v1
  let v13 := v12
  r2 := v13
  let v14 := v0
  let v15 := v14
  r3 := v15
  let v16 := (IntTy.wrap (.i 256) (-1))
  let v17 := v16
  r4 := v17
  let rec b1 (r0 : (Vector UInt256 5)) (r1 : (Vector UInt256 5)) (r2 : (Vector UInt256 5)) (r3 : (Vector UInt256 5)) (r4 : UInt256) (r5 : Transition) (r6 : (Vector UInt256 5)) (r7 : (Vector UInt256 5)) (r8 : (Vector UInt256 5)) (r9 : (Vector UInt256 5)) (r10 : (Vector UInt256 5)) (r11 : (Vector UInt256 5)) : FeM (Vector UInt256 5) := do
    let mut r0 := r0
    let mut r1 := r1
    let mut r2 := r2
    let mut r3 := r3
    let mut r4 := r4
    let mut r5 := r5
    let mut r6 := r6
    let mut r7 := r7
    let mut r8 := r8
    let mut r9 := r9
    let mut r10 := r10
    let mut r11 := r11
    let v18 := r3
    let v19 := v18
    r6 := v19
    let v21 ← is_zero_5 r6
    let v22 := !v21
    if v22 then
      let v23 := r4
      let v24 := v23
      let v25 := (UInt256.ofNat 0)
      checkIndex v25 5
      let v26 := r2[v25.toNat]!
      let v27 ← downcast_truncate_i64_u64 v26
      let v28 := (UInt256.ofNat 0)
      checkIndex v28 5
      let v29 := r3[v28.toNat]!
      let v30 ← downcast_truncate_i64_u64 v29
      let v31 ← divsteps v24 v27 v30
      let v32 := v31.fst
      let v33 := v32
      let v34 := v31.snd
      let v35 := v34
      r5 := v35
      let v36 := v33
      r4 := v36
      let v37 := ()
      let v38 := r0
      let v39 := v38
      r7 := v39
      let v40 := r1
      let v41 := v40
      r8 := v41
      let v45 ← update_de_5 r7 r8 r5 v1 v2
      let v46 := v45.fst
      let v47 := v46
      let v48 := v45.snd
      let v49 := v48
      let v50 := v47
      r0 := v50
      let v51 := ()
      let v52 := v49
      r1 := v52
      let v53 := ()
      let v54 := r2
      let v55 := v54
      r9 := v55
      let v56 := r3
      let v57 := v56
      r10 := v57
      let v61 ← update_fg_5 r9 r10 r5
      let v62 := v61.fst
      let v63 := v62
      let v64 := v61.snd
      let v65 := v64
      let v66 := v63
      r2 := v66
      let v67 := ()
      let v68 := v65
      r3 := v68
      let v69 := ()
      b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11
    else
      let v70 := r0
      let v71 := v70
      r11 := v71
      let v72 := (UInt256.ofNat 5)
      let v73 := (UInt256.ofNat 1)
      let v74 := v72 - v73
      checkIndex v74 5
      let v75 := r2[v74.toNat]!
      let v77 ← normalize_5 r11 v75 v1
      return v77
  partial_fixpoint
  b1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11

def invert_256_k5 (v76 : UInt256) (v85 : UInt256) (v94 : UInt256) (v98 : UInt256) (v100 : UInt256) (v101 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v102 := IntTy.shl (.u 64) v100 v101
  let v103 := v98 ||| v102
  let v104 := #v[v76, v85, v94, v103]
  return v104

def invert_256_k4 (v69 : (Vector UInt256 5)) (v76 : UInt256) (v80 : UInt256) (v81 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v82 := v69[v81.toNat]!
  let v83 := (UInt256.ofNat 60)
  let v84 := IntTy.shl (.u 64) v82 v83
  let v85 := v80 ||| v84
  let v86 := (UInt256.ofNat 2)
  checkIndex v86 5
  let v87 := v69[v86.toNat]!
  let v88 := (UInt256.ofNat 4)
  let v89 := v87 >>> v88
  let v90 := (UInt256.ofNat 3)
  checkIndex v90 5
  let v91 := v69[v90.toNat]!
  let v92 := (UInt256.ofNat 58)
  let v93 := IntTy.shl (.u 64) v91 v92
  let v94 := v89 ||| v93
  let v95 := (UInt256.ofNat 3)
  checkIndex v95 5
  let v96 := v69[v95.toNat]!
  let v97 := (UInt256.ofNat 6)
  let v98 := v96 >>> v97
  let v99 := (UInt256.ofNat 4)
  checkIndex v99 5
  let v100 := v69[v99.toNat]!
  let v101 := (UInt256.ofNat 56)
  invert_256_k5 v76 v85 v94 v98 v100 v101 r0

def invert_256_k3 (v52 : (Vector UInt256 5)) (v55 : UInt256) (v58 : UInt256) (v61 : UInt256) (v62 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v63 := v52[v62.toNat]!
  let v64 ← downcast_truncate_i64_u64 v63
  let v65 := (UInt256.ofNat 4)
  checkIndex v65 5
  let v66 := v52[v65.toNat]!
  let v67 ← downcast_truncate_i64_u64 v66
  let v68 := #v[v55, v58, v61, v64, v67]
  let v69 := v68
  let v70 := (UInt256.ofNat 0)
  checkIndex v70 5
  let v71 := v69[v70.toNat]!
  let v72 := (UInt256.ofNat 1)
  checkIndex v72 5
  let v73 := v69[v72.toNat]!
  let v74 := (UInt256.ofNat 62)
  let v75 := IntTy.shl (.u 64) v73 v74
  let v76 := v71 ||| v75
  let v77 := (UInt256.ofNat 1)
  checkIndex v77 5
  let v78 := v69[v77.toNat]!
  let v79 := (UInt256.ofNat 2)
  let v80 := v78 >>> v79
  let v81 := (UInt256.ofNat 2)
  checkIndex v81 5
  invert_256_k4 v69 v76 v80 v81 r0

def invert_256_k2 (v0 : (Vector UInt256 4)) (v1 : (Vector UInt256 5)) (v2 : UInt256) (v7 : UInt256) (v19 : UInt256) (v31 : UInt256) (v43 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v44 := (UInt256.ofNat 3)
  checkIndex v44 4
  let v45 := v0[v44.toNat]!
  let v46 := (UInt256.ofNat 56)
  let v47 := v45 >>> v46
  let v48 ← downcast_truncate_u64_i64 v47
  let v49 := #v[v7, v19, v31, v43, v48]
  r0 := v49
  let v51 ← invert_5 r0 v1 v2
  let v52 := v51
  let v53 := (UInt256.ofNat 0)
  checkIndex v53 5
  let v54 := v52[v53.toNat]!
  let v55 ← downcast_truncate_i64_u64 v54
  let v56 := (UInt256.ofNat 1)
  checkIndex v56 5
  let v57 := v52[v56.toNat]!
  let v58 ← downcast_truncate_i64_u64 v57
  let v59 := (UInt256.ofNat 2)
  checkIndex v59 5
  let v60 := v52[v59.toNat]!
  let v61 ← downcast_truncate_i64_u64 v60
  let v62 := (UInt256.ofNat 3)
  checkIndex v62 5
  invert_256_k3 v52 v55 v58 v61 v62 r0

def invert_256_k1 (v0 : (Vector UInt256 4)) (v1 : (Vector UInt256 5)) (v2 : UInt256) (v7 : UInt256) (v19 : UInt256) (v21 : UInt256) (v22 : UInt256) (r0 : (Vector UInt256 5)) : FeM (Vector UInt256 4) := do
  let mut r0 := r0
  let v23 := v21 >>> v22
  let v24 := (UInt256.ofNat 2)
  checkIndex v24 4
  let v25 := v0[v24.toNat]!
  let v26 := (UInt256.ofNat 4)
  let v27 := IntTy.shl (.u 64) v25 v26
  let v28 := v23 ||| v27
  let v29 ← M62
  let v30 := v28 &&& v29
  let v31 ← downcast_truncate_u64_i64 v30
  let v32 := (UInt256.ofNat 2)
  checkIndex v32 4
  let v33 := v0[v32.toNat]!
  let v34 := (UInt256.ofNat 58)
  let v35 := v33 >>> v34
  let v36 := (UInt256.ofNat 3)
  checkIndex v36 4
  let v37 := v0[v36.toNat]!
  let v38 := (UInt256.ofNat 6)
  let v39 := IntTy.shl (.u 64) v37 v38
  let v40 := v35 ||| v39
  let v41 ← M62
  let v42 := v40 &&& v41
  let v43 ← downcast_truncate_u64_i64 v42
  invert_256_k2 v0 v1 v2 v7 v19 v31 v43 r0

def invert_256 (v0 : (Vector UInt256 4)) (v1 : (Vector UInt256 5)) (v2 : UInt256) : FeM (Vector UInt256 4) := do
  let mut r0 : (Vector UInt256 5) := default
  let v3 := (UInt256.ofNat 0)
  checkIndex v3 4
  let v4 := v0[v3.toNat]!
  let v5 ← M62
  let v6 := v4 &&& v5
  let v7 ← downcast_truncate_u64_i64 v6
  let v8 := (UInt256.ofNat 0)
  checkIndex v8 4
  let v9 := v0[v8.toNat]!
  let v10 := (UInt256.ofNat 62)
  let v11 := v9 >>> v10
  let v12 := (UInt256.ofNat 1)
  checkIndex v12 4
  let v13 := v0[v12.toNat]!
  let v14 := (UInt256.ofNat 2)
  let v15 := IntTy.shl (.u 64) v13 v14
  let v16 := v11 ||| v15
  let v17 ← M62
  let v18 := v16 &&& v17
  let v19 ← downcast_truncate_u64_i64 v18
  let v20 := (UInt256.ofNat 1)
  checkIndex v20 4
  let v21 := v0[v20.toNat]!
  let v22 := (UInt256.ofNat 60)
  invert_256_k1 v0 v1 v2 v7 v19 v21 v22 r0

def MemBuffer_len_1 (v0 : MemBuffer) : FeM UInt256 := do
  let v1 ← MemBuffer_len v0
  return v1

def MemBuffer_lane_at (v0 : MemBuffer) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_read_le_u64 v0 v1
  return v2

def MemBuffer_byte_at (v0 : MemBuffer) (v1 : UInt256) : FeM UInt256 := do
  let v2 ← MemBuffer_index_ref v0 v1
  let v3 ← mloadInt (.u 8) v2
  return v3


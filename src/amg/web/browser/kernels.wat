;; Matrix kernels for engine.js, using WebAssembly SIMD (4 floats per instruction).
;; Build: npx wat2wasm kernels.wat -o kernels.wasm, then base64 the
;; result into KERNELS_WASM in engine.js (python -m amg.export_web does not need this;
;; the compiled bytes are already embedded).
(module
  (import "env" "memory" (memory 1))

  ;; Out[r*outDim + j] = b[j] + sum_i X[r*inDim + i] * Wt[j*inDim + i]
  ;; for r < n, j < outDim. Wt is the kernel transposed to (outDim, inDim).
  ;; All pointers are byte offsets; inDim must be a multiple of 8.
  (func (export "matmat")
    (param $x i32) (param $n i32) (param $w i32) (param $b i32)
    (param $in i32) (param $out i32) (param $o i32)
    (local $j i32) (local $r i32) (local $i i32)
    (local $row i32) (local $xr i32) (local $rowBytes i32)
    (local $acc0 v128) (local $acc1 v128) (local $bj f32)
    (local.set $rowBytes (i32.shl (local.get $in) (i32.const 2)))
    (local.set $j (i32.const 0))
    (block $jdone
      (loop $jloop
        (br_if $jdone (i32.ge_u (local.get $j) (local.get $out)))
        (local.set $row (i32.add (local.get $w) (i32.mul (local.get $j) (local.get $rowBytes))))
        (local.set $bj (f32.load (i32.add (local.get $b) (i32.shl (local.get $j) (i32.const 2)))))
        (local.set $r (i32.const 0))
        (block $rdone
          (loop $rloop
            (br_if $rdone (i32.ge_u (local.get $r) (local.get $n)))
            (local.set $xr (i32.add (local.get $x) (i32.mul (local.get $r) (local.get $rowBytes))))
            (local.set $acc0 (v128.const f32x4 0 0 0 0))
            (local.set $acc1 (v128.const f32x4 0 0 0 0))
            (local.set $i (i32.const 0))
            (block $idone
              (loop $iloop
                (br_if $idone (i32.ge_u (local.get $i) (local.get $rowBytes)))
                (local.set $acc0 (f32x4.add (local.get $acc0)
                  (f32x4.mul (v128.load (i32.add (local.get $xr) (local.get $i)))
                             (v128.load (i32.add (local.get $row) (local.get $i))))))
                (local.set $acc1 (f32x4.add (local.get $acc1)
                  (f32x4.mul (v128.load offset=16 (i32.add (local.get $xr) (local.get $i)))
                             (v128.load offset=16 (i32.add (local.get $row) (local.get $i))))))
                (local.set $i (i32.add (local.get $i) (i32.const 32)))
                (br $iloop)))
            (local.set $acc0 (f32x4.add (local.get $acc0) (local.get $acc1)))
            (f32.store
              (i32.add (local.get $o)
                (i32.shl (i32.add (i32.mul (local.get $r) (local.get $out)) (local.get $j)) (i32.const 2)))
              (f32.add (local.get $bj)
                (f32.add
                  (f32.add (f32x4.extract_lane 0 (local.get $acc0)) (f32x4.extract_lane 1 (local.get $acc0)))
                  (f32.add (f32x4.extract_lane 2 (local.get $acc0)) (f32x4.extract_lane 3 (local.get $acc0))))))
            (local.set $r (i32.add (local.get $r) (i32.const 1)))
            (br $rloop)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $jloop))))
)

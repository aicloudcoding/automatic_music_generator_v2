/*
 * Automatic Music Generator: the trained Transformer running in JavaScript.
 *
 * Mirrors amg.model.build_transformer: token + position (+ composer) embeddings,
 * pre-norm blocks of causal multi-head attention and a GELU feed-forward layer,
 * a final LayerNorm and a softmax over the 97-token vocabulary.
 *
 * Generation feeds one token at a time and keeps each layer's keys and values
 * (a "KV cache"), so a new token costs one small pass instead of re-reading the
 * whole window. Because the model was trained to predict at every position of
 * its window, starting from a shorter opening is in-distribution.
 *
 * Works in a browser, a Web Worker and Node (for tests).
 */
(function (root) {
  "use strict";

  // ---------- weights ----------
  const HALF = (() => {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
           : e === 31 ? (f ? NaN : s * Infinity)
           : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return t;
  })();

  function decodeWeights(buffer, dtype) {
    if (dtype === "float32") return new Float32Array(buffer);
    const h = new Uint16Array(buffer), out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) out[i] = HALF[h[i]];
    return out;
  }

  // ---------- math helpers ----------
  // Kernels arrive as (inDim, outDim); store them as (outDim, inDim) so each output is one
  // contiguous dot product with its running sum kept in a register.
  function transpose(W, inDim, outDim) {
    const T = new Float32Array(W.length);
    for (let i = 0; i < inDim; i++) for (let j = 0; j < outDim; j++) T[j * inDim + i] = W[i * outDim + j];
    return T;
  }

  // Many inputs at once (rows of X, n of them): reads each weight row once for all inputs.
  function matmat(X, n, Wt, b, inDim, outDim, Out) {
    for (let j = 0; j < outDim; j++) {
      const row = j * inDim, bj = b[j];
      for (let r = 0; r < n; r++) {
        const xr = r * inDim;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let i = 0; i < inDim; i += 4) {
          s0 += X[xr + i] * Wt[row + i];
          s1 += X[xr + i + 1] * Wt[row + i + 1];
          s2 += X[xr + i + 2] * Wt[row + i + 2];
          s3 += X[xr + i + 3] * Wt[row + i + 3];
        }
        Out[r * outDim + j] = bj + (s0 + s1) + (s2 + s3);
      }
    }
    return Out;
  }

  function layerNorm(x, gamma, beta, eps, out) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
    const inv = 1 / Math.sqrt(v / n + eps);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    return out;
  }

  // erf with ~1e-7 accuracy (Numerical Recipes), for the exact GELU Keras uses.
  function erf(x) {
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

  // ---------- fast path: WebAssembly SIMD ----------
  // kernels.wat compiled with wat2wasm; computes the same thing as matmat() above,
  // four multiply-adds per instruction.
  const KERNELS_WASM = "AGFzbQEAAAABCwFgB39/f39/f38AAg8BA2VudgZtZW1vcnkCAAEDAgEABwoBBm1hdG1hdAAACpUCAZICAwZ/AnsBfSAEQQJ0IQxBACEHAkADQCAHIAVPDQEgAiAHIAxsaiEKIAMgB0ECdGoqAgAhD0EAIQgCQANAIAggAU8NASAAIAggDGxqIQv9DAAAAAAAAAAAAAAAAAAAAAAhDf0MAAAAAAAAAAAAAAAAAAAAACEOQQAhCQJAA0AgCSAMTw0BIA0gCyAJav0ABAAgCiAJav0ABAD95gH95AEhDSAOIAsgCWr9AAQQIAogCWr9AAQQ/eYB/eQBIQ4gCUEgaiEJDAALCyANIA795AEhDSAGIAggBWwgB2pBAnRqIA8gDf0fACAN/R8BkiAN/R8CIA39HwOSkpI4AgAgCEEBaiEIDAALCyAHQQFqIQcMAAsLCw==";

  function decodeBase64(b64) {
    if (typeof atob === "function") return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  /** One block of memory for all weights and buffers; WebAssembly-backed when possible. */
  function makeArena(floats, useWasm) {
    const bytes = floats * 4 + 1024;
    let buffer = null, kernel = null;
    if (useWasm && typeof WebAssembly === "object") {
      try {
        const memory = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) });
        const instance = new WebAssembly.Instance(new WebAssembly.Module(decodeBase64(KERNELS_WASM)), { env: { memory } });
        buffer = memory.buffer; kernel = instance.exports.matmat;
      } catch (e) { kernel = null; }   // no SIMD support: plain JavaScript below
    }
    if (!kernel) buffer = new ArrayBuffer(bytes);
    let offset = 0;
    return {
      kernel,
      alloc(n, from) {
        const v = new Float32Array(buffer, offset, n);
        if (from) v.set(from);
        offset += Math.ceil(n / 4) * 16;   // keep 16-byte alignment for SIMD loads
        return v;
      },
    };
  }

  // ---------- the model ----------
  class Transformer {
    /** options.wasm = false forces the plain JavaScript path (used by tests). */
    constructor(manifest, weights, options = {}) {
      Object.assign(this, {
        L: manifest.seq_len, d: manifest.d_model, H: manifest.heads, dk: manifest.key_dim,
        nLayers: manifest.layers, ff: manifest.ff_dim, eps: manifest.eps, V: manifest.vocab_size,
      });
      const { L, d, ff, V } = this;
      const buffers = 5 * d + ff + V + this.nLayers * 2 * L * d + 5 * L * d + L * ff;
      // The SIMD kernel reads 8 floats at a time, so every input size must be a multiple of 8.
      const simdOk = d % 8 === 0 && ff % 8 === 0;
      const arena = makeArena(weights.length + manifest.tensors.length * 4 + buffers, options.wasm !== false && simdOk);
      this.kernel = arena.kernel;
      this.fast = !!arena.kernel;
      const w = {};
      for (const t of manifest.tensors) {
        const size = t.shape.reduce((a, b) => a * b, 1);
        const src = weights.subarray(t.offset, t.offset + size);
        // Matrices are transposed to (out, in) for the dot-product kernels; the rest is copied as is.
        const isMatrix = t.shape.length === 2 && /(\.w|^out\.w)$/.test(t.name);
        w[t.name] = arena.alloc(size, isMatrix ? transpose(src, t.shape[0], t.shape[1]) : src);
      }
      this.w = w;
      this.outW = w["out.w"];
      this.blocks = [];
      for (let i = 1; i <= this.nLayers; i++) {
        const g = (k) => w[`b${i}.${k}`];
        this.blocks.push({
          n1g: g("norm1.gamma"), n1b: g("norm1.beta"),
          qw: g("q.w"), qb: g("q.b"), kw: g("k.w"), kb: g("k.b"), vw: g("v.w"), vb: g("v.b"),
          ow: g("o.w"), ob: g("o.b"), n2g: g("norm2.gamma"), n2b: g("norm2.beta"),
          f1w: g("ff1.w"), f1b: g("ff1.b"), f2w: g("ff2.w"), f2b: g("ff2.b"),
          K: arena.alloc(L * d), Vc: arena.alloc(L * d),
        });
      }
      this.x = arena.alloc(d); this.h = arena.alloc(d); this.q = arena.alloc(d);
      this.a = arena.alloc(d); this.o = arena.alloc(d); this.f = arena.alloc(ff);
      this.logits = arena.alloc(V); this.scores = new Float32Array(L);
      // Work space for prime(): up to a full window of tokens at once.
      this.P = { X: arena.alloc(L * d), Hn: arena.alloc(L * d), Q: arena.alloc(L * d),
                 A: arena.alloc(L * d), O: arena.alloc(L * d), F: arena.alloc(L * ff) };
      this.reset(null);
    }

    /** Out = X·W + b for n rows, through WebAssembly when available. */
    mm(X, n, Wt, b, inDim, outDim, Out) {
      if (this.kernel) this.kernel(X.byteOffset, n, Wt.byteOffset, b.byteOffset, inDim, outDim, Out.byteOffset);
      else matmat(X, n, Wt, b, inDim, outDim, Out);
      return Out;
    }

    /** Start a new sequence, optionally in a composer's style. */
    reset(composerId) {
      this.t = 0;
      const ce = this.w.composer_embedding;
      this.comp = ce && composerId != null ? ce.subarray(composerId * this.d, (composerId + 1) * this.d) : null;
    }

    /** Feed one token; returns the probabilities for the next one. */
    step(token) {
      const { d, H, dk, eps } = this, t = this.t;
      if (t >= this.L) throw new Error("Context full; reset and prime again.");
      const x = this.x, h = this.h, q = this.q, a = this.a, o = this.o, f = this.f;
      const emb = this.w.embedding, pos = this.w.positions;
      for (let i = 0; i < d; i++) x[i] = emb[token * d + i] + pos[t * d + i] + (this.comp ? this.comp[i] : 0);
      const scale = 1 / Math.sqrt(dk), scores = this.scores;

      for (const B of this.blocks) {
        layerNorm(x, B.n1g, B.n1b, eps, h);
        this.mm(h, 1, B.qw, B.qb, d, d, q);
        this.mm(h, 1, B.kw, B.kb, d, d, B.K.subarray(t * d, (t + 1) * d));
        this.mm(h, 1, B.vw, B.vb, d, d, B.Vc.subarray(t * d, (t + 1) * d));
        for (let head = 0; head < H; head++) {
          const off = head * dk;
          let max = -Infinity;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            const kj = j * d + off;
            for (let c = 0; c < dk; c++) s += q[off + c] * B.K[kj + c];
            s *= scale;
            scores[j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j <= t; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }
          for (let c = 0; c < dk; c++) a[off + c] = 0;
          for (let j = 0; j <= t; j++) {
            const p = scores[j] / sum, vj = j * d + off;
            for (let c = 0; c < dk; c++) a[off + c] += p * B.Vc[vj + c];
          }
        }
        this.mm(a, 1, B.ow, B.ob, d, d, o);
        for (let i = 0; i < d; i++) x[i] += o[i];
        layerNorm(x, B.n2g, B.n2b, eps, h);
        this.mm(h, 1, B.f1w, B.f1b, d, this.ff, f);
        for (let i = 0; i < this.ff; i++) f[i] = gelu(f[i]);
        this.mm(f, 1, B.f2w, B.f2b, this.ff, d, o);
        for (let i = 0; i < d; i++) x[i] += o[i];
      }
      layerNorm(x, this.w["final_norm.gamma"], this.w["final_norm.beta"], eps, h);
      this.mm(h, 1, this.outW, this.w["out.b"], d, this.V, this.logits);
      this.t = t + 1;
      return softmax(this.logits);
    }

    /**
     * Feed a whole opening at once (fills the key/value cache for every position);
     * returns the probabilities after its last token. Same result as calling step()
     * for each token, but each weight is read once per layer instead of once per token.
     */
    prime(tokens) {
      const n = tokens.length, { d, H, dk, eps, ff } = this, t0 = this.t;
      if (!n) return null;
      if (t0 + n > this.L) throw new Error("Opening longer than the model's window.");
      const P = this.P;
      const X = P.X.subarray(0, n * d), Hn = P.Hn.subarray(0, n * d), Q = P.Q.subarray(0, n * d);
      const A = P.A.subarray(0, n * d), O = P.O.subarray(0, n * d), F = P.F.subarray(0, n * ff);
      const emb = this.w.embedding, pos = this.w.positions, scale = 1 / Math.sqrt(dk);
      for (let r = 0; r < n; r++)
        for (let i = 0; i < d; i++)
          X[r * d + i] = emb[tokens[r] * d + i] + pos[(t0 + r) * d + i] + (this.comp ? this.comp[i] : 0);
      const scores = new Float32Array(this.L);
      const norm = (src, g, b, dst) => {
        for (let r = 0; r < n; r++) layerNorm(src.subarray(r * d, (r + 1) * d), g, b, eps, dst.subarray(r * d, (r + 1) * d));
      };
      for (const B of this.blocks) {
        norm(X, B.n1g, B.n1b, Hn);
        this.mm(Hn, n, B.qw, B.qb, d, d, Q);
        this.mm(Hn, n, B.kw, B.kb, d, d, B.K.subarray(t0 * d, (t0 + n) * d));
        this.mm(Hn, n, B.vw, B.vb, d, d, B.Vc.subarray(t0 * d, (t0 + n) * d));
        for (let r = 0; r < n; r++) {
          const t = t0 + r;
          for (let head = 0; head < H; head++) {
            const off = head * dk, qo = r * d + off;
            let max = -Infinity;
            for (let j = 0; j <= t; j++) {
              let s = 0;
              const kj = j * d + off;
              for (let c = 0; c < dk; c++) s += Q[qo + c] * B.K[kj + c];
              s *= scale;
              scores[j] = s;
              if (s > max) max = s;
            }
            let sum = 0;
            for (let j = 0; j <= t; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }
            const ao = r * d + off;
            for (let c = 0; c < dk; c++) A[ao + c] = 0;
            for (let j = 0; j <= t; j++) {
              const p = scores[j] / sum, vj = j * d + off;
              for (let c = 0; c < dk; c++) A[ao + c] += p * B.Vc[vj + c];
            }
          }
        }
        this.mm(A, n, B.ow, B.ob, d, d, O);
        for (let i = 0; i < n * d; i++) X[i] += O[i];
        norm(X, B.n2g, B.n2b, Hn);
        this.mm(Hn, n, B.f1w, B.f1b, d, ff, F);
        for (let i = 0; i < n * ff; i++) F[i] = gelu(F[i]);
        this.mm(F, n, B.f2w, B.f2b, ff, d, O);
        for (let i = 0; i < n * d; i++) X[i] += O[i];
      }
      this.t = t0 + n;
      // Only the last position's output is needed.
      layerNorm(X.subarray((n - 1) * d, n * d), this.w["final_norm.gamma"], this.w["final_norm.beta"], eps, this.h);
      this.mm(this.h, 1, this.outW, this.w["out.b"], d, this.V, this.logits);
      return softmax(this.logits);
    }
  }

  function softmax(logits) {
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    const out = new Float64Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
    for (let i = 0; i < out.length; i++) out[i] /= sum;
    return out;
  }

  // ---------- sampling (same rules as amg.model.sample_next) ----------
  function sampleNext(probs, temperature, topK, banned, rng) {
    const n = probs.length, p = Float64Array.from(probs);
    for (const b of banned) p[b] = 0;
    if (temperature <= 0) {
      let best = 0;
      for (let i = 1; i < n; i++) if (p[i] > p[best]) best = i;
      return best;
    }
    const logits = new Float64Array(n);
    for (let i = 0; i < n; i++) logits[i] = Math.log(Math.max(p[i], 1e-12)) / temperature;
    for (const b of banned) logits[b] = -Infinity;
    if (topK && topK > 0 && topK < n) {
      const cutoff = Array.from(logits).sort((x, y) => y - x)[topK - 1];
      for (let i = 0; i < n; i++) if (logits[i] < cutoff) logits[i] = -Infinity;
    }
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let sum = 0;
    const wts = new Float64Array(n);
    for (let i = 0; i < n; i++) { wts[i] = Math.exp(logits[i] - max); sum += wts[i]; }
    let r = rng() * sum;
    for (let i = 0; i < n; i++) { r -= wts[i]; if (r <= 0 && wts[i] > 0) return i; }
    for (let i = n - 1; i >= 0; i--) if (wts[i] > 0) return i;
    return 0;
  }

  function seededRandom(seed) {  // mulberry32
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- tokens -> notes (same rules as amg.tokens.tokens_to_elements) ----------
  const isPitch = (t) => /^P\d+$/.test(t);
  const isShift = (t) => /^T\d+(\.\d+)?$/.test(t);

  class OnsetCounter {
    constructor() { this.count = 0; this.open = false; }
    add(tok) {
      if (isPitch(tok)) this.open = true;
      else if (isShift(tok) && this.open) { this.count++; this.open = false; }
      return this.count;
    }
  }

  /** [{start, dur, pitches}] in beats (quarter notes). */
  function tokensToMoments(tokens) {
    const out = [];
    let offset = 0, pending = [], holding = null;
    const flush = (len) => {
      holding = { start: offset, dur: len, pitches: [...new Set(pending)].sort((x, y) => x - y) };
      out.push(holding);
      pending = [];
    };
    for (const tok of tokens) {
      if (tok.startsWith("<")) continue;
      if (isPitch(tok)) { holding = null; pending.push(+tok.slice(1)); }
      else if (isShift(tok)) {
        const step = +tok.slice(1);
        if (pending.length) flush(step);
        else if (holding) holding.dur += step;
        offset += step;
      }
    }
    if (pending.length) flush(1.0);
    return out;
  }

  const NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const noteName = (m) => `${NAMES[m % 12]}${Math.floor(m / 12) - 1}`;

  function momentsToEvents(moments) {
    return moments.map((m) => ({
      token: `${m.pitches.map(noteName).join(" ")} · ${+m.dur.toFixed(4)}`,
      start: m.start, dur: m.dur, pitches: m.pitches,
    }));
  }

  // ---------- MIDI file (format 0, one piano track) ----------
  function midiFile(moments, bpm) {
    const PPQ = 480, events = [];
    for (const m of moments) {
      const on = Math.round(m.start * PPQ), off = Math.round((m.start + m.dur) * PPQ);
      for (const p of m.pitches) { events.push([on, 1, p]); events.push([off, 0, p]); }
    }
    events.sort((x, y) => x[0] - y[0] || x[1] - y[1]);  // note-offs before note-ons at the same tick
    const bytes = [];
    const vlq = (n) => {
      const stack = [n & 0x7f];
      while ((n >>= 7)) stack.push((n & 0x7f) | 0x80);
      while (stack.length) bytes.push(stack.pop());
    };
    const us = Math.round(60000000 / bpm);
    bytes.push(0x00, 0xff, 0x51, 0x03, (us >> 16) & 0xff, (us >> 8) & 0xff, us & 0xff);  // tempo
    bytes.push(0x00, 0xc0, 0x00);                                                           // piano
    let last = 0;
    for (const [tick, on, p] of events) {
      vlq(tick - last); last = tick;
      bytes.push(on ? 0x90 : 0x80, p, on ? 90 : 0);
    }
    bytes.push(0x00, 0xff, 0x2f, 0x00);
    const n = bytes.length;
    const head = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, (PPQ >> 8) & 0xff, PPQ & 0xff,
                  0x4d, 0x54, 0x72, 0x6b, (n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff];
    return Uint8Array.from(head.concat(bytes));
  }

  function base64(bytes) {
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return typeof btoa === "function" ? btoa(s) : Buffer.from(bytes).toString("base64");
  }

  // ---------- generation ----------
  /**
   * Write `notes` moments. Returns the same fields as the Python server's
   * /api/generate, so the page can't tell the difference.
   */
  function generate(model, manifest, seeds, req) {
    const itos = manifest.vocab, composers = manifest.composers;
    const rng = req.random_seed != null ? seededRandom(req.random_seed) : Math.random;
    let composer = null, pool = seeds[""] || [];
    if (composers.length) {
      composer = req.composer || composers[0];
      const id = composers.indexOf(composer);
      if (id < 0) throw new Error(`Unknown composer '${composer}'.`);
      pool = seeds[composer] && seeds[composer].length ? seeds[composer] : [].concat(...Object.values(seeds));
      model.reset(id);
    } else {
      model.reset(null);
    }
    const seed = pool[Math.floor(rng() * pool.length)];
    let probs = model.prime(seed);
    const out = [], counter = new OnsetCounter(), banned = [manifest.unk_id];
    const half = Math.floor(model.L / 2);
    for (let i = 0; i < req.notes * 20; i++) {
      const next = sampleNext(probs, req.temperature, req.top_k, banned, rng);
      out.push(next);
      if (counter.add(itos[next]) >= req.notes) break;
      if (model.t >= model.L) {  // window full: keep the most recent half and carry on
        const context = seed.concat(out).slice(-half);
        model.reset(composer != null ? composers.indexOf(composer) : null);
        probs = model.prime(context);
      } else {
        probs = model.step(next);
      }
    }
    const tokens = out.map((i) => itos[i]);
    const moments = tokensToMoments(tokens);
    const bpm = req.bpm || 90;
    const d = new Date(), pad = (n) => String(n).padStart(2, "0");
    const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return {
      tokens, events: momentsToEvents(moments), bpm, composer,
      filename: `amg${composer ? "_" + composer : ""}_${req.notes}notes_${stamp}.mid`,
      midi_base64: base64(midiFile(moments, bpm)),
    };
  }

  const api = { decodeWeights, Transformer, matmat, sampleNext, seededRandom, OnsetCounter, tokensToMoments,
                momentsToEvents, midiFile, generate, erf, gelu };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AMG = api;
})(typeof self !== "undefined" ? self : this);

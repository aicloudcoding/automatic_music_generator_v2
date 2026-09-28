/* Runs the model in a background thread so the page stays responsive. */
importScripts("engine.js" + self.location.search);  // same version tag as this file

let ready = null;  // Promise of {model, manifest, seeds}

/**
 * Fetch the weights, reporting progress so the page can show "Loading model… 45%".
 * Progress is measured against the size the manifest expects, not Content-Length:
 * servers like GitHub Pages compress the file in transit, and then Content-Length
 * is the compressed size while the stream delivers the uncompressed bytes.
 */
async function download(url, expectedBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Couldn't download the model (${r.status}).`);
  if (!r.body) return r.arrayBuffer();
  const reader = r.body.getReader(), chunks = [];
  let loaded = 0, lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const pct = Math.min(99, Math.floor((loaded / expectedBytes) * 100));
    if (pct !== lastPct) { lastPct = pct; postMessage({ type: "progress", pct }); }
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out.buffer;
}

function load(base, v = "") {
  ready = (async () => {
    const manifest = await fetch(base + "model/manifest.json" + v).then((r) => r.json());
    const count = manifest.tensors.reduce((n, t) => n + t.shape.reduce((a, b) => a * b, 1), 0);
    const bytes = count * (manifest.dtype === "float32" ? 4 : 2);
    const [seeds, weights] = await Promise.all([
      fetch(base + "model/seeds.json" + v).then((r) => r.json()),
      download(base + "model/weights.bin" + v, bytes),
    ]);
    if (weights.byteLength !== bytes) {
      throw new Error(`The model download was incomplete (${weights.byteLength} of ${bytes} bytes). Please reload the page.`);
    }
    const model = new AMG.Transformer(manifest, AMG.decodeWeights(weights, manifest.dtype));
    return { model, manifest, seeds };
  })();
  ready.then(() => postMessage({ type: "ready" }), (e) => postMessage({ type: "failed", error: String(e.message || e) }));
}

onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "load") return load(msg.base, msg.v || "");
  if (msg.type === "generate") {
    try {
      const { model, manifest, seeds } = await ready;
      const t0 = Date.now();
      const result = AMG.generate(model, manifest, seeds, msg.request);
      result.ms = Date.now() - t0;
      postMessage({ type: "result", id: msg.id, result });
    } catch (err) {
      postMessage({ type: "error", id: msg.id, error: String(err.message || err) });
    }
  }
};

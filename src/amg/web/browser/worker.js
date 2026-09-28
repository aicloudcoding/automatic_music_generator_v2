/* Runs the model in a background thread so the page stays responsive. */
importScripts("engine.js");

let ready = null;  // Promise of {model, manifest, seeds}

/** Fetch the weights, reporting progress so the page can show "Loading model… 45%". */
async function download(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Couldn't download the model (${r.status}).`);
  const total = Number(r.headers.get("Content-Length")) || 0;
  if (!r.body || !total) return r.arrayBuffer();
  const reader = r.body.getReader(), out = new Uint8Array(total);
  let loaded = 0, lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > total) return (await new Response(out.slice(0, loaded)).arrayBuffer()); // size changed
    out.set(value, loaded);
    loaded += value.length;
    const pct = Math.floor((loaded / total) * 100);
    if (pct !== lastPct) { lastPct = pct; postMessage({ type: "progress", pct }); }
  }
  return out.buffer.slice(0, loaded);
}

function load(base) {
  ready = (async () => {
    const [manifest, seeds, weights] = await Promise.all([
      fetch(base + "model/manifest.json").then((r) => r.json()),
      fetch(base + "model/seeds.json").then((r) => r.json()),
      download(base + "model/weights.bin"),
    ]);
    const model = new AMG.Transformer(manifest, AMG.decodeWeights(weights, manifest.dtype));
    return { model, manifest, seeds };
  })();
  ready.then(() => postMessage({ type: "ready" }), (e) => postMessage({ type: "failed", error: String(e.message || e) }));
}

onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "load") return load(msg.base);
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

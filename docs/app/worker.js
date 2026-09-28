/* Runs the model in a background thread so the page stays responsive. */
importScripts("engine.js");

let ready = null;  // Promise of {model, manifest, seeds}

function load(base) {
  ready = (async () => {
    const [manifest, seeds, weights] = await Promise.all([
      fetch(base + "model/manifest.json").then((r) => r.json()),
      fetch(base + "model/seeds.json").then((r) => r.json()),
      fetch(base + "model/weights.bin").then((r) => {
        if (!r.ok) throw new Error(`Couldn't download the model (${r.status}).`);
        return r.arrayBuffer();
      }),
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

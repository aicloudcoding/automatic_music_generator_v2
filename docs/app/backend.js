/*
 * Browser backend for the static site. The page was written for the Python
 * server's /api/info and /api/generate; this answers those same requests
 * locally, with the model running in a Web Worker, so the page itself is
 * unchanged.
 */
(function () {
  "use strict";
  const base = new URL(".", document.baseURI).href;
  const worker = new Worker(base + "app/worker.js");
  const pending = new Map();
  let nextId = 1, loadError = null;
  const whenReady = new Promise((resolve) => {
    worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "ready") resolve();
      else if (m.type === "failed") { loadError = m.error; resolve(); }
      else if (pending.has(m.id)) {
        const { ok, fail } = pending.get(m.id);
        pending.delete(m.id);
        m.type === "result" ? ok(m.result) : fail(new Error(m.error));
      }
    });
  });
  worker.postMessage({ type: "load", base });

  const manifest = fetch(base + "model/manifest.json").then((r) => r.json());
  const info = manifest.then((m) => ({
    run: m.run, arch: m.arch, params: m.params, pieces: m.pieces, results: m.results,
    composer_names: m.composer_names, vocab_size: m.vocab_size, seq_len: m.seq_len,
    durations: true, encoding: "events", composers: m.composers, trained_on: m.trained_on,
    min_notes: 5, max_notes: 50, static: true,
  }));

  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

  async function generate(req) {
    const notes = Number(req.notes);
    if (!(notes >= 5 && notes <= 50)) return json({ detail: "notes must be between 5 and 50." }, 422);
    await whenReady;
    if (loadError) return json({ detail: loadError }, 503);
    const request = {
      notes, temperature: Number(req.temperature ?? 0.9), top_k: Number(req.top_k ?? 20),
      composer: req.composer || null, bpm: Number(req.bpm ?? 90), random_seed: req.random_seed ?? null,
    };
    try {
      const result = await new Promise((ok, fail) => {
        const id = nextId++;
        pending.set(id, { ok, fail });
        worker.postMessage({ type: "generate", id, request });
      });
      return json(result);
    } catch (err) {
      return json({ detail: err.message }, 422);
    }
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    const path = new URL(url, location.href).pathname;
    if (path.endsWith("/api/info")) return json(await info);
    if (path.endsWith("/api/generate")) return generate(JSON.parse((init && init.body) || "{}"));
    return realFetch(input, init);
  };
})();

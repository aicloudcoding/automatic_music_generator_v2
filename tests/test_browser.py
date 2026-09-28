"""The browser version must match the Python model. Needs Node.js; skipped without it."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from amg.model import build_transformer
from amg.tokens import midi_to_tokens, tokens_to_elements, tokens_to_midi
from amg.train import event_vocab

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
ENGINE = str((__import__("amg").__path__[0])) + "/web/browser/engine.js"
PHRASE = ["P36", "P60", "P64", "P67", "T1.0", "P62", "T0.5", "T0.5", "P43", "P59", "P65", "T0.25", "P67", "T2.0"]


def _node(script: str, *args) -> str:
    return subprocess.run([NODE, "-e", script, ENGINE, *map(str, args)], capture_output=True, text=True,
                          check=True, timeout=120).stdout


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    from amg.export_web import export

    run = tmp_path_factory.mktemp("run")
    vocab = event_vocab()
    vocab.save(run / "vocab.json")
    model = build_transformer(len(vocab), 32, d_model=16, num_layers=2, heads=2, num_composers=3)
    model.save(run / "model.keras")
    (run / "config.json").write_text(json.dumps({"composers": ["bach", "chopin", "liszt"], "use_composer": True,
                                                 "encoding": "events", "arch": "transformer"}))
    seeds = [{"composer": c, "tokens": (PHRASE * 3)[:30]} for c in ("bach", "chopin", "liszt")]
    (run / "seeds.json").write_text(json.dumps(seeds))
    out = tmp_path_factory.mktemp("site")
    export(run, out, seed_len=20, dtype="float32")
    return run, out, model, vocab


READ = """
const fs = require('fs'), AMG = require(process.argv[1]), dir = process.argv[2];
const man = JSON.parse(fs.readFileSync(dir + '/model/manifest.json'));
const buf = fs.readFileSync(dir + '/model/weights.bin');
const w = AMG.decodeWeights(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), man.dtype);
"""


def test_export_writes_a_complete_site(site):
    _, out, _, _ = site
    for f in ("index.html", "app/engine.js", "app/worker.js", "app/backend.js", "model/manifest.json",
              "model/weights.bin", "model/seeds.json", "static/fonts/nunito-latin-wght-normal.woff2", ".nojekyll"):
        assert (out / f).exists(), f
    page = (out / "index.html").read_text()
    assert '<script src="app/backend.js?v=' in page and 'url("static/fonts/' in page
    man = json.loads((out / "model" / "manifest.json").read_text())
    assert man["composers"] == ["bach", "chopin", "liszt"] and man["seq_len"] == 32


@pytest.mark.parametrize("wasm", [True, False])
def test_javascript_matches_keras_at_every_position(site, wasm):
    _, out, model, vocab = site
    ids = vocab.encode((PHRASE * 3)[:32])
    ref = np.asarray(model({"tokens": np.array([ids], "int32"), "composer": np.array([[2]], "int32")}))[0]
    got = json.loads(_node(READ + f"""
        const m = new AMG.Transformer(man, w, {{wasm: {str(wasm).lower()}}});
        m.reset(2);
        const ids = {json.dumps(ids)}, out = [];
        for (const t of ids) out.push(Array.from(m.step(t)));
        m.reset(2);
        const primed = Array.from(m.prime(ids.slice(0, 20)));
        console.log(JSON.stringify({{steps: out, primed}}));
    """, out))
    assert np.allclose(np.array(got["steps"]), ref, atol=1e-5)
    assert np.allclose(np.array(got["primed"]), ref[19], atol=1e-5)  # batched priming == step by step


def test_decoding_and_midi_match_python(site, tmp_path):
    _, out, _, _ = site
    got = json.loads(_node(READ + f"""
        const toks = {json.dumps(PHRASE)};
        const moments = AMG.tokensToMoments(toks);
        fs.writeFileSync(process.argv[3], Buffer.from(AMG.midiFile(moments, 90)));
        const c = new AMG.OnsetCounter();
        console.log(JSON.stringify({{moments, counts: toks.map((t) => c.add(t))}}));
    """, out, tmp_path / "js.mid"))
    python = [(float(o), float(el.quarterLength), sorted(p.midi for p in el.pitches))
              for o, el in tokens_to_elements(PHRASE)]
    assert [(m["start"], m["dur"], m["pitches"]) for m in got["moments"]] == python
    assert got["counts"][-1] == 4
    # The MIDI file written in the browser reads back as the same music as Python's.
    tokens_to_midi(PHRASE, tmp_path / "py.mid")
    js = midi_to_tokens(tmp_path / "js.mid", encoding="events")
    assert js == midi_to_tokens(tmp_path / "py.mid", encoding="events")
    assert js[:6] == PHRASE[:6] and js[-2:] == PHRASE[-2:]


def test_generate_returns_what_the_page_expects(site):
    _, out, _, _ = site
    res = json.loads(_node(READ + """
        const seeds = JSON.parse(fs.readFileSync(dir + '/model/seeds.json'));
        const m = new AMG.Transformer(man, w);
        const r = AMG.generate(m, man, seeds, {notes: 12, temperature: 0.9, top_k: 10, composer: 'chopin', bpm: 100, random_seed: 3});
        const again = AMG.generate(m, man, seeds, {notes: 12, temperature: 0.9, top_k: 10, composer: 'chopin', bpm: 100, random_seed: 3});
        let bad = null; try { AMG.generate(m, man, seeds, {notes: 5, composer: 'nobody'}); } catch (e) { bad = e.message; }
        console.log(JSON.stringify({r, same: JSON.stringify(r.tokens) === JSON.stringify(again.tokens), bad}));
    """, out))
    r = res["r"]
    assert set(r) >= {"tokens", "events", "bpm", "composer", "filename", "midi_base64"}
    assert r["composer"] == "chopin" and r["bpm"] == 100 and "<UNK>" not in r["tokens"]
    assert len(r["events"]) <= 12 and r["filename"].endswith(".mid")
    assert res["same"] and "Unknown composer" in res["bad"]

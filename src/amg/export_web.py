"""Build the browser version of the web app (a static site with no server).

    python -m amg.export_web --run pretrained/transformer --out docs

Writes the same page the Python server shows, plus:
  model/manifest.json   sizes, vocabulary, composers, results and where each tensor sits
  model/weights.bin     all weights as little-endian float16 (half the size of float32)
  model/seeds.json      a few openings per composer, as token ids
  app/*.js              the Transformer, sampling and MIDI writer in JavaScript
The page then runs the model in the visitor's browser, so it can be hosted on
GitHub Pages or any static host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent / "web"
PAGE = HERE / "static" / "index.html"
FONTS = HERE / "static" / "fonts"
BROWSER_JS = HERE / "browser"


def _dense(layer) -> tuple[np.ndarray, np.ndarray]:
    kernel, bias = (np.asarray(w) for w in layer.get_weights())
    return kernel, bias


def transformer_tensors(model) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """Architecture sizes plus the weights, in the order the JavaScript reads them.

    Attention projections are reshaped to plain matrices: the (d, heads, key_dim)
    query/key/value kernels become (d, heads*key_dim) and the (heads, key_dim, d)
    output kernel becomes (heads*key_dim, d), so head h uses columns h*key_dim..
    """
    names = {layer.name for layer in model.layers}
    if "positions" not in names:
        raise SystemExit("Only Transformer runs (--arch transformer) can be exported for the browser.")
    n_layers = sum(1 for n in names if n.endswith("_attention") and n.startswith("block"))
    tokens = model.get_layer("embedding").get_weights()[0]
    positions = model.get_layer("positions").get_weights()[0]
    mha = model.get_layer("block1_attention")
    heads, key_dim = mha.num_heads, mha.key_dim
    d = tokens.shape[1]
    ff = model.get_layer("block1_ff1").get_weights()[0].shape[1]
    eps = float(model.get_layer("block1_norm1").epsilon)

    out: list[tuple[str, np.ndarray]] = [("embedding", tokens), ("positions", positions)]
    if "composer_embedding" in names:
        out.append(("composer_embedding", model.get_layer("composer_embedding").get_weights()[0]))
    for i in range(1, n_layers + 1):
        norm1 = model.get_layer(f"block{i}_norm1").get_weights()
        norm2 = model.get_layer(f"block{i}_norm2").get_weights()
        att = model.get_layer(f"block{i}_attention")
        q, k, v, o = att._query_dense, att._key_dense, att._value_dense, att._output_dense
        wq, bq = _dense(q)
        wk, bk = _dense(k)
        wv, bv = _dense(v)
        wo, bo = _dense(o)
        w1, b1 = _dense(model.get_layer(f"block{i}_ff1"))
        w2, b2 = _dense(model.get_layer(f"block{i}_ff2"))
        out += [
            (f"b{i}.norm1.gamma", norm1[0]), (f"b{i}.norm1.beta", norm1[1]),
            (f"b{i}.q.w", wq.reshape(d, heads * key_dim)), (f"b{i}.q.b", bq.reshape(-1)),
            (f"b{i}.k.w", wk.reshape(d, heads * key_dim)), (f"b{i}.k.b", bk.reshape(-1)),
            (f"b{i}.v.w", wv.reshape(d, heads * key_dim)), (f"b{i}.v.b", bv.reshape(-1)),
            (f"b{i}.o.w", wo.reshape(heads * key_dim, d)), (f"b{i}.o.b", bo.reshape(-1)),
            (f"b{i}.norm2.gamma", norm2[0]), (f"b{i}.norm2.beta", norm2[1]),
            (f"b{i}.ff1.w", w1), (f"b{i}.ff1.b", b1),
            (f"b{i}.ff2.w", w2), (f"b{i}.ff2.b", b2),
        ]
    final = model.get_layer("final_norm").get_weights()
    wout, bout = _dense(model.get_layer("next_token"))
    out += [("final_norm.gamma", final[0]), ("final_norm.beta", final[1]),
            ("out.w", wout), ("out.b", bout)]
    sizes = {"seq_len": int(positions.shape[0]), "d_model": int(d), "heads": int(heads),
             "key_dim": int(key_dim), "layers": n_layers, "ff_dim": int(ff), "eps": eps,
             "vocab_size": int(tokens.shape[0])}
    return sizes, out


def trim_seed(ids: list[int], itos: list[str], length: int) -> list[int]:
    """The last `length` tokens of a seed, starting cleanly at a moment boundary."""
    tail = ids[-length:]
    for i, t in enumerate(tail[:-1]):
        if itos[t].startswith("T") and i < len(tail) // 2:
            return tail[i + 1:]
    return tail


def export(run: str | Path, out: str | Path, seed_len: int = 256, seeds_per_composer: int = 10,
           dtype: str = "float16") -> Path:
    from .generate import load_run
    from .web.service import _read_seeds, display_name, run_results

    run, out = Path(run), Path(out)
    model, vocab, config, raw_seeds = load_run(run)
    sizes, tensors = transformer_tensors(model)
    if seed_len >= sizes["seq_len"]:
        raise SystemExit("--seed-len must be shorter than the model's window.")

    composers = list(config.get("composers", [])) if config.get("use_composer") else []
    seeds, by_composer = _read_seeds(raw_seeds, 1)
    seed_ids: dict[str, list[list[int]]] = {}
    groups = {c: by_composer.get(c, []) for c in composers} if composers else {"": seeds}
    for name, group in groups.items():
        picked = [trim_seed(vocab.encode(s), vocab.itos, seed_len) for s in group[:seeds_per_composer]]
        seed_ids[name] = [s for s in picked if len(s) >= 16]
    if not any(seed_ids.values()):
        raise SystemExit("No usable seeds in this run.")

    np_dtype = {"float16": "<f2", "float32": "<f4"}[dtype]
    (out / "model").mkdir(parents=True, exist_ok=True)
    index, offset = [], 0
    with open(out / "model" / "weights.bin", "wb") as f:
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            f.write(arr.astype(np_dtype).tobytes())
            index.append({"name": name, "shape": list(arr.shape), "offset": offset})
            offset += arr.size
    pieces = len(config.get("train_pieces", [])) + len(config.get("val_pieces", []))
    manifest = {
        "format": 1, "arch": "transformer", "dtype": dtype, **sizes,
        "vocab": vocab.itos, "unk_id": vocab.unk_id,
        "composers": composers, "composer_names": {c: display_name(c) for c in composers},
        "trained_on": config.get("composers", []), "pieces": pieces,
        "params": int(model.count_params()), "results": run_results(run, config),
        "run": run.name, "tensors": index,
    }
    (out / "model" / "manifest.json").write_text(json.dumps(manifest))
    (out / "model" / "seeds.json").write_text(json.dumps(seed_ids, separators=(",", ":")))

    # The page: the same one the Python server shows, plus the in-browser backend.
    (out / "app").mkdir(exist_ok=True)
    for js in BROWSER_JS.glob("*.js"):
        shutil.copy(js, out / "app" / js.name)
    shutil.copytree(FONTS, out / "static" / "fonts", dirs_exist_ok=True)
    # A version tag in every URL, so browsers fetch fresh files after an update instead of
    # reusing cached ones (GitHub Pages lets browsers cache for about 10 minutes).
    digest = hashlib.sha256()
    for f in sorted((out / "app").glob("*.js")) + sorted((out / "model").iterdir()):
        digest.update(f.read_bytes())
    version = digest.hexdigest()[:10]
    page = PAGE.read_text()
    marker = "<script>\nconst $ = "
    if marker not in page:
        raise SystemExit("Couldn't find where to add the browser backend in index.html.")
    page = page.replace(marker, f'<script src="app/backend.js?v={version}" data-version="{version}"></script>\n' + marker, 1)
    (out / "index.html").write_text(page)
    (out / ".nojekyll").write_text("")  # serve files as they are on GitHub Pages
    size = (out / "model" / "weights.bin").stat().st_size / 1e6
    print(f"Exported {run} -> {out}  ({offset:,} weights, {size:.1f} MB {dtype}, "
          f"{sum(map(len, seed_ids.values()))} seeds)")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the static browser version of the web app.")
    p.add_argument("--run", default="pretrained/transformer", help="Transformer run folder to export.")
    p.add_argument("--out", default="docs", help="Output folder (docs/ is what GitHub Pages serves).")
    p.add_argument("--seed-len", type=int, default=256,
                   help="Tokens of each opening kept (shorter = faster start in the browser).")
    p.add_argument("--seeds-per-composer", type=int, default=10)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = p.parse_args(argv)
    export(args.run, args.out, args.seed_len, args.seeds_per_composer, args.dtype)


if __name__ == "__main__":
    main()

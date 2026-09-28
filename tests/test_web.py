"""Web app tests. They use a tiny untrained model, so they check plumbing, not musicality."""

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from amg.model import build_model  # noqa: E402
from amg.vocab import UNK, Vocab  # noqa: E402

SEQ = 8
TOKENS = ["C4_0.5", "E4_0.5", "G4_1.0", "0.4.7_1.0", "D4_0.25", "F4_0.5", "2.5.9_2.0", "B3_0.5"]


def _make_run(path, composers=None):
    path.mkdir()
    vocab = Vocab([UNK, *TOKENS])
    vocab.save(path / "vocab.json")
    seeds = [TOKENS[i:] + TOKENS[:i] for i in range(4)]
    config = {"durations": True, "composers": ["schubert"]}
    if composers:
        import keras
        from keras import layers

        tok = keras.Input(shape=(SEQ,), dtype="int32", name="tokens")
        comp = keras.Input(shape=(1,), dtype="int32", name="composer")
        x = layers.LSTM(8)(layers.Embedding(len(vocab), 8)(tok))
        c = layers.Flatten()(layers.Embedding(len(composers), 4)(comp))
        out = layers.Dense(len(vocab), activation="softmax")(layers.Concatenate()([x, c]))
        model = keras.Model([tok, comp], out)
        # Same layout as a real composer-conditioned run from amg.train.
        config = {"durations": True, "composers": composers, "use_composer": True}
        seeds = [{"composer": name, "tokens": s} for name in composers for s in seeds]
    else:
        model = build_model(len(vocab), SEQ, embed_dim=8, units=8, num_layers=1)
    model.save(path / "model.keras")
    (path / "config.json").write_text(json.dumps(config))
    (path / "seeds.json").write_text(json.dumps(seeds))
    return path


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from amg.web.app import create_app
    from amg.web.service import MusicService

    run = _make_run(tmp_path_factory.mktemp("runs") / "tiny")
    return TestClient(create_app(MusicService(run)))


def test_info_and_page(client):
    info = client.get("/api/info").json()
    assert info["min_notes"] == 5 and info["max_notes"] == 50
    assert info["composers"] == []
    assert "Automatic Music Generator" in client.get("/").text


@pytest.mark.parametrize("n", [5, 23, 50])
def test_generate_exact_note_count(client, n):
    res = client.post("/api/generate", json={"notes": n, "random_seed": 1})
    assert res.status_code == 200
    data = res.json()
    assert len(data["tokens"]) == n and len(data["events"]) == n
    midi = base64.b64decode(data["midi_base64"])
    assert midi[:4] == b"MThd" and data["filename"].endswith(".mid")
    assert all(UNK not in t for t in data["tokens"])


def test_same_seed_same_piece(client):
    a = client.post("/api/generate", json={"notes": 12, "random_seed": 7}).json()["tokens"]
    b = client.post("/api/generate", json={"notes": 12, "random_seed": 7}).json()["tokens"]
    assert a == b


@pytest.mark.parametrize("n", [4, 51])
def test_note_count_out_of_range(client, n):
    assert client.post("/api/generate", json={"notes": n}).status_code == 422


def test_composer_conditioned_run(tmp_path):
    from amg.web.app import create_app
    from amg.web.service import MusicService

    client = TestClient(create_app(MusicService(_make_run(tmp_path / "cond", ["bach", "chopin"]))))
    assert client.get("/api/info").json()["composers"] == ["bach", "chopin"]
    data = client.post("/api/generate", json={"notes": 10, "composer": "chopin"}).json()
    assert len(data["tokens"]) == 10 and data["composer"] == "chopin"
    assert client.post("/api/generate", json={"notes": 10, "composer": "nobody"}).status_code == 422


@pytest.mark.parametrize("raw", [
    [TOKENS],
    [{"composer": "bach", "tokens": TOKENS}],
    {"bach": [TOKENS]},
])
def test_read_seeds_layouts(raw):
    from amg.web.service import _read_seeds

    seeds, by_composer = _read_seeds(raw, SEQ)
    assert seeds == [TOKENS]
    assert by_composer in ({}, {"bach": [TOKENS]})
    assert _read_seeds(raw, SEQ + 1)[0] == []  # too short for the model is dropped


def _make_events_run(path):
    """A tiny untrained run using the events encoding, laid out like amg.train writes it."""
    from amg.train import event_vocab

    path.mkdir()
    vocab = event_vocab()
    vocab.save(path / "vocab.json")
    phrase = ["P48", "P60", "P64", "T0.5", "P67", "T0.5", "P43", "P62", "P65", "T1.0"]
    seed = (phrase * 3)[:SEQ * 3]
    model = build_model(len(vocab), len(seed), embed_dim=8, units=8, num_layers=1)
    model.save(path / "model.keras")
    (path / "config.json").write_text(json.dumps({"durations": True, "encoding": "events",
                                                  "composers": ["schubert"]}))
    (path / "seeds.json").write_text(json.dumps([{"composer": "schubert", "tokens": seed}]))
    return path


def test_events_run_counts_moments(tmp_path):
    from amg.tokens import count_onsets
    from amg.web.app import create_app
    from amg.web.service import MusicService

    client = TestClient(create_app(MusicService(_make_events_run(tmp_path / "ev"))))
    assert client.get("/api/info").json()["encoding"] == "events"
    for n in (5, 17):
        data = client.post("/api/generate", json={"notes": n, "random_seed": n}).json()
        # An untrained model may emit rests/holds, but never more moments than asked for.
        assert count_onsets(data["tokens"]) <= n
        assert all(e["pitches"] for e in data["events"])
        assert base64.b64decode(data["midi_base64"])[:4] == b"MThd"


def test_transformer_run_in_web_app(tmp_path):
    from amg.model import build_transformer
    from amg.train import event_vocab
    from amg.web.app import create_app
    from amg.web.service import MusicService

    run = tmp_path / "tf"
    run.mkdir()
    vocab = event_vocab()
    vocab.save(run / "vocab.json")
    seed = ["P48", "P60", "P64", "T0.5", "P67", "T0.5"] * 4
    build_transformer(len(vocab), len(seed), d_model=16, num_layers=1, heads=2,
                      num_composers=2).save(run / "model.keras")
    (run / "config.json").write_text(json.dumps({"encoding": "events", "arch": "transformer",
                                                 "composers": ["bach", "chopin"], "use_composer": True}))
    (run / "seeds.json").write_text(json.dumps([{"composer": c, "tokens": seed} for c in ("bach", "chopin")]))
    client = TestClient(create_app(MusicService(run)))
    info = client.get("/api/info").json()
    assert info["arch"] == "transformer" and info["composers"] == ["bach", "chopin"]
    assert client.post("/api/generate", json={"notes": 8, "composer": "bach"}).status_code == 200


def test_run_results_from_history_when_stopped_early(tmp_path):
    from amg.web.service import display_name, run_results

    (tmp_path / "history.csv").write_text(
        "epoch,val_accuracy,val_last_accuracy,val_top5\n"
        "0,0.50,0.52,0.80\n1,0.67,0.70,0.88\n2,0.66,0.71,0.87\n")
    res = run_results(tmp_path, {"monitor": "val_accuracy"})
    assert res == {"accuracy": 0.70, "top5": 0.88, "epoch": 2}  # best by the monitored number
    (tmp_path / "metrics.json").write_text(json.dumps({"val": {"accuracy": 0.5, "top5": 0.79}}))
    assert run_results(tmp_path, {})["accuracy"] == 0.5
    assert display_name("beeth") == "Beethoven" and display_name("new_person") == "New Person"


def test_page_and_fonts_are_served(client):
    page = client.get("/").text
    assert "/static/fonts/quicksand-latin-wght-normal.woff2" in page
    assert client.get("/static/fonts/quicksand-latin-wght-normal.woff2").status_code == 200
    info = client.get("/api/info").json()
    assert info["params"] > 0 and "results" in info


def test_launcher_prefers_your_runs_then_the_pretrained_model(tmp_path, monkeypatch):
    from amg.web import __main__ as launcher

    monkeypatch.chdir(tmp_path)
    assert launcher.latest_run() is None
    pre = tmp_path / "pretrained" / "transformer"
    pre.mkdir(parents=True)
    (pre / "vocab.json").write_text("[]")
    assert launcher.latest_run() == Path("pretrained/transformer")
    run = tmp_path / "runs" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "vocab.json").write_text("[]")
    assert launcher.latest_run() == Path("pretrained/transformer")  # no model file yet
    (run / "best.keras").write_bytes(b"")
    assert launcher.latest_run() == Path("runs/20260101-000000")

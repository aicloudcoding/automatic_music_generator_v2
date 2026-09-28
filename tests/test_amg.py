import numpy as np
import pytest
from music21 import chord, note

from amg.data import find_midi_files, make_windows, split_by_piece
from amg.model import build_model, sample_next
from amg.tokens import (
    DURATIONS,
    transpose_tokens,
    midi_to_tokens,
    quantize_duration,
    token_to_element,
    tokens_to_midi,
)
from amg.vocab import UNK, Vocab


def test_quantize_duration_snaps_to_allowed_values():
    assert quantize_duration(0.26) == 0.25
    assert quantize_duration(0.9) == 1.0
    assert quantize_duration(10) == 4.0
    assert all(quantize_duration(d) == d for d in DURATIONS)


def test_token_to_element():
    n = token_to_element("E4_0.5")
    assert isinstance(n, note.Note) and n.nameWithOctave == "E4" and n.quarterLength == 0.5
    c = token_to_element("4.7.11_1.0")
    assert isinstance(c, chord.Chord)
    assert sorted(p.pitchClass for p in c.pitches) == [4, 7, 11]
    assert [p.midi for p in c.pitches] == sorted(p.midi for p in c.pitches)
    r = token_to_element("R_2.0")
    assert isinstance(r, note.Rest) and r.quarterLength == 2.0
    assert token_to_element(UNK) is None
    # Tokens without a duration (v1 style) fall back to a default.
    assert token_to_element("C#5").quarterLength > 0


def test_midi_roundtrip(tmp_path):
    tokens = ["C4_1.0", "E4_0.5", "0.4.7_2.0", "G4_0.25", "C3_4.0"]
    path = tokens_to_midi(tokens, tmp_path / "t.mid")
    assert midi_to_tokens(path) == tokens
    assert midi_to_tokens(path, durations=False) == ["C4", "E4", "0.4.7", "G4", "C3"]


def test_rest_becomes_a_longer_gap(tmp_path):
    path = tokens_to_midi(["C4_1.0", "R_1.0", "E4_1.0"], tmp_path / "r.mid")
    assert midi_to_tokens(path) == ["C4_2.0", "E4_1.0"]


def test_vocab_unknowns_and_save(tmp_path):
    v = Vocab.build([["a", "a", "b"], ["a", "c"]], min_count=2)
    assert v.itos == [UNK, "a"]
    assert v.encode(["a", "b", "zzz"]) == [1, 0, 0]
    v.save(tmp_path / "v.json")
    assert Vocab.load(tmp_path / "v.json").itos == v.itos


def test_make_windows():
    x, y = make_windows([[1, 2, 3, 4, 5], [9, 9]], seq_len=3)
    assert x.tolist() == [[1, 2, 3], [2, 3, 4]]
    assert y.tolist() == [4, 5]


def test_split_by_piece_has_no_overlap():
    pieces = [f"p{i}" for i in range(20)]
    train, val = split_by_piece(pieces, 0.2, seed=1)
    assert len(val) == 4 and not set(train) & set(val)
    assert set(train) | set(val) == set(pieces)
    assert split_by_piece(pieces, 0.2, seed=1) == (train, val)


def test_sample_next():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    assert sample_next(probs, temperature=0) == 0
    assert sample_next(probs, temperature=0, banned=[0]) == 1
    rng = np.random.default_rng(0)
    picks = {sample_next(probs, 1.0, top_k=2, rng=rng) for _ in range(200)}
    assert picks <= {0, 1}


def test_model_output_shape():
    model = build_model(vocab_size=30, seq_len=8, embed_dim=8, units=16)
    out = model(np.zeros((2, 8), dtype=np.int32))
    assert out.shape == (2, 30)
    assert np.allclose(np.sum(out, axis=1), 1.0, atol=1e-4)


def test_find_midi_files_is_case_insensitive(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.mid").write_bytes(b"")
    (tmp_path / "x" / "b.MID").write_bytes(b"")
    (tmp_path / "x" / "c.txt").write_bytes(b"")
    assert [p.name for p in find_midi_files(tmp_path, ["x"])] == ["a.mid", "b.MID"]
    with pytest.raises(FileNotFoundError):
        find_midi_files(tmp_path, ["nobody"])


def test_transpose_notes_and_chords():
    assert transpose_tokens(["E4_0.5", "C#4_1.0"], -1) == ["E-4_0.5", "C4_1.0"]
    assert transpose_tokens(["0.4.7_1.0"], 2) == ["2.6.9_1.0"]
    # Chords are re-put in normal order after shifting (C major -> B major).
    assert transpose_tokens(["0.4.7_1.0"], -1) == ["11.3.6_1.0"]
    assert transpose_tokens(["E4", "0.4.7"], 12) == ["E5", "0.4.7"]
    assert transpose_tokens(["C4_1.0"], 0) == ["C4_1.0"]


def test_transpose_matches_the_tokenizer(tmp_path):
    """A transposed sequence must equal tokenizing the transposed music."""
    original = ["C4_1.0", "E-4_0.5", "0.4.7_2.0", "F#3_0.25", "2.5.9_1.0"]
    for shift in (-5, -1, 3, 6):
        shifted = transpose_tokens(original, shift)
        path = tokens_to_midi(shifted, tmp_path / f"s{shift}.mid")
        assert midi_to_tokens(path) == shifted


def test_transpose_off_the_keyboard_returns_none():
    assert transpose_tokens(["C8_1.0"], 1) is None
    assert transpose_tokens(["A0_1.0"], -1) is None


def test_make_windows_with_composer_labels():
    x, y, c = make_windows([[1, 2, 3, 4], [5, 6, 7, 8, 9]], seq_len=3, labels=[0, 2])
    assert x.tolist() == [[1, 2, 3], [5, 6, 7], [6, 7, 8]]
    assert y.tolist() == [4, 8, 9]
    assert c.tolist() == [0, 2, 2]


def test_model_with_composer_input():
    model = build_model(vocab_size=30, seq_len=8, embed_dim=8, units=16, num_composers=4)
    assert [i.name for i in model.inputs] == ["tokens", "composer"]
    out = model({"tokens": np.zeros((2, 8), np.int32), "composer": np.array([[0], [3]], np.int32)})
    assert out.shape == (2, 30)


# ---- events encoding (tokenizer v5) ----

from amg.data import WindowDataset, transpose_table  # noqa: E402
from amg.tokens import OnsetCounter, count_onsets, detect_encoding, shift_tokens, tokens_to_elements  # noqa: E402


def test_events_roundtrip_keeps_voicing(tmp_path):
    # Bass C2 under a C major chord two octaves up, then a melody note.
    tokens = ["P36", "P60", "P64", "P67", "T1.0", "P72", "T0.5", "P36", "P79", "T2.0"]
    path = tokens_to_midi(tokens, tmp_path / "e.mid")
    assert midi_to_tokens(path, encoding="events") == tokens
    first = tokens_to_elements(tokens)[0][1]
    assert [p.midi for p in first.pitches] == [36, 60, 64, 67]  # exact octaves kept


def test_events_real_piece_keeps_every_pitch():
    path = find_midi_files("data/midi", ["schubert"])[0]
    from amg.tokens import _onsets
    onsets, _ = _onsets(path)
    tokens = midi_to_tokens(path, encoding="events")
    decoded = [sorted(p.midi for p in el.pitches) for _, el in tokens_to_elements(tokens)]
    original = [sorted({min(max(p.midi, 21), 108) for p in onsets[t]}) for t in sorted(onsets)]
    assert decoded == original


def test_shift_tokens_split_long_gaps():
    assert shift_tokens(0.5) == ["T0.5"]
    assert shift_tokens(9.0) == ["T4.0", "T4.0", "T1.0"]
    assert shift_tokens(0.01) == ["T0.25"]


def test_long_rests_hold_the_last_moment():
    els = tokens_to_elements(["P60", "T4.0", "T4.0", "P62", "T1.0"])
    assert [(o, el.quarterLength) for o, el in els] == [(0.0, 8.0), (8.0, 1.0)]


def test_transpose_events():
    assert transpose_tokens(["P60", "P64", "T0.5"], 3) == ["P63", "P67", "T0.5"]
    assert transpose_tokens(["P108", "T1.0"], 1) is None


def test_onset_counting():
    tokens = ["P36", "P60", "T1.0", "T1.0", "P62", "T0.5", "P64"]
    counter = OnsetCounter()
    assert [counter.add(t) for t in tokens] == [0, 0, 1, 1, 1, 2, 2]
    assert count_onsets(tokens) == 3  # the trailing note still plays
    assert count_onsets(["C4_1.0", "R_1.0", "0.4.7_1.0"]) == 2
    assert detect_encoding(["<UNK>", "P60", "T0.5"]) == "events"
    assert detect_encoding(["<UNK>", "C4_0.5"]) == "chords"


def test_window_dataset_and_transposition():
    v = Vocab([UNK, "P60", "P62", "P64", "T0.5"])
    seq = v.encode(["P60", "T0.5", "P62", "T0.5", "P64", "T0.5", "P60", "T0.5"])
    ds = WindowDataset([seq], seq_len=3, batch_size=8, shuffle=False)
    x, y = ds[0]
    assert x.tolist()[0] == seq[:3] and y.tolist()[0] == seq[3]
    assert len(ds.starts) == len(seq) - 3
    table = transpose_table(v.itos, 2)
    assert table[2 + 2, v.stoi["P60"]] == v.stoi["P62"]  # +2 semitones
    assert table[2 - 2, v.stoi["P60"]] == -1             # P58 isn't in this vocabulary
    moved = WindowDataset([seq], seq_len=3, batch_size=8, transpose=2, table=table, seed=0)
    for _ in range(5):
        xb, yb = moved[0]
        assert ((xb > 0) | (xb == 0)).all() and (yb >= 0).all()  # never an invalid id


def test_window_dataset_labels_and_sampling():
    ds = WindowDataset([[1, 2, 3, 4], [5, 6, 7, 8, 9]], seq_len=3, batch_size=10,
                       labels=[0, 2], windows_per_epoch=2)
    x, y = ds[0]
    assert x["tokens"].shape == (2, 3) and x["composer"].shape == (2, 1)
    assert len(ds) == 1


# ---- Transformer ----

from amg.model import build_transformer, next_token_probs  # noqa: E402


def test_transformer_is_causal_and_predicts_every_position():
    model = build_transformer(vocab_size=30, seq_len=12, d_model=16, num_layers=2, heads=2)
    x = np.random.default_rng(0).integers(0, 30, (2, 12)).astype(np.int32)
    out = np.asarray(model(x))
    assert out.shape == (2, 12, 30)
    x2 = x.copy()
    x2[:, 7] = (x2[:, 7] + 1) % 30
    out2 = np.asarray(model(x2))
    assert np.allclose(out[:, :7], out2[:, :7], atol=1e-5)  # earlier positions can't see later tokens
    assert next_token_probs(out[:1]).shape == (30,)


def test_transformer_with_composer_saves_and_loads(tmp_path):
    import keras

    from amg import layers  # noqa: F401

    model = build_transformer(vocab_size=30, seq_len=8, d_model=16, num_layers=1, heads=2, num_composers=3)
    x = {"tokens": np.zeros((1, 8), np.int32), "composer": np.array([[2]], np.int32)}
    model.save(tmp_path / "t.keras")
    again = keras.models.load_model(tmp_path / "t.keras", compile=False)
    assert np.allclose(model(x), again(x), atol=1e-6)


def test_all_targets_windows_shift_by_one():
    ds = WindowDataset([[1, 2, 3, 4, 5, 6]], seq_len=3, batch_size=8, shuffle=False, all_targets=True)
    x, y = ds[0]
    assert x.tolist()[0] == [1, 2, 3] and y.tolist()[0] == [2, 3, 4]


def test_last_token_accuracy():
    from amg.layers import LastTokenAccuracy

    m = LastTokenAccuracy()
    y_true = np.array([[1, 2], [0, 1]])
    y_pred = np.zeros((2, 2, 3), np.float32)
    y_pred[0, 1, 2] = 1  # right at the last position
    y_pred[1, 1, 0] = 1  # wrong at the last position
    m.update_state(y_true, y_pred)
    assert float(m.result()) == 0.5

"""Converting MIDI files to token sequences and token sequences back to MIDI.

Two encodings are supported:

"events" (default since tokenizer v5): every struck note is its own token
with its exact MIDI pitch ("P60" = middle C), listed low to high, and a
time-step token ("T0.5") moves to the next moment. A C major chord with a
bass note is "P36 P60 P64 P67 T1.0". Chord voicing, octaves and the gap
between bass and melody are all kept, and the vocabulary is ~100 tokens.

"chords" (the original v2 encoding): one token per moment: a single note
keeps its octave ("E4"), two or more pitch classes become a normal-order
chord ("4.7.11") with no octave, plus the time to the next moment
("E4_0.5", "4.7.11_1.0"). A rest token ("R") is also understood.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

from music21 import chord, converter, instrument, note, pitch, stream, tempo

# Bump when tokenization changes so cached token files are rebuilt.
TOKENIZER_VERSION = 5
ENCODINGS = ("events", "chords")
PITCH, SHIFT = "P", "T"  # event-token prefixes: P<midi>, T<quarter notes>

# Allowed durations in quarter notes (0.25 = sixteenth, 4.0 = whole note).
DURATIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
DEFAULT_DURATION = 0.5
REST = "R"
SEP = "_"
# Piano range (A0 to C8) as MIDI note numbers.
PIANO_LOW, PIANO_HIGH = 21, 108


def quantize_duration(quarter_length: float) -> float:
    """Snap a duration to the nearest allowed value."""
    return min(DURATIONS, key=lambda d: abs(d - quarter_length))


def _piano_stream(score: stream.Score) -> stream.Stream:
    """Return the piano part(s) of a score, or the whole score if none is labeled."""
    parts = instrument.partitionByInstrument(score)
    if parts is not None:
        piano = [p for p in parts.parts if "Piano" in str(p.getInstrument())]
        if piano:
            merged = stream.Score()
            for p in piano:
                merged.insert(0, p)
            return merged
    return score


def pitches_token(pitches) -> str:
    """Token for the set of pitches struck at one moment.

    One pitch (or one pitch doubled in octaves) keeps its octave, e.g. "E4".
    Two or more pitch classes become a chord in normal order, e.g. "4.7.11".
    """
    classes = {p.pitchClass for p in pitches}
    if len(classes) == 1:
        return min(pitches, key=lambda p: p.midi).nameWithOctave
    return ".".join(str(n) for n in chord.Chord(pitches).normalOrder)


def _onsets(path: str | Path) -> tuple[dict[float, list], float]:
    """Pitches struck at each moment of a MIDI file, and when the last note ends."""
    with warnings.catch_warnings():
        # Public MIDI files often have track names music21 can't map to an instrument.
        warnings.simplefilter("ignore")
        score = converter.parse(str(path))
    onsets: dict[float, list] = {}
    end = 0.0  # when the last sounding note (including tied parts) ends
    for el in _piano_stream(score).flatten().notes:
        ql = float(el.quarterLength)
        if ql <= 0:  # grace notes
            continue
        end = max(end, float(el.offset) + ql)
        # Skip the tied continuation of a note that crosses a barline: it isn't struck again.
        struck = [n.pitch for n in (el.notes if isinstance(el, chord.Chord) else [el])
                  if not (n.tie and n.tie.type in ("continue", "stop"))]
        if not struck:
            continue
        t = round(float(el.offset), 4)
        onsets.setdefault(t, []).extend(struck)
    return onsets, end


def shift_tokens(gap: float) -> list[str]:
    """Time-step tokens covering a gap: whole 4-beat steps, then the quantized rest."""
    out = []
    while gap > DURATIONS[-1] + 1e-6:
        out.append(f"{SHIFT}{DURATIONS[-1]}")
        gap -= DURATIONS[-1]
    if gap >= DURATIONS[0] / 2 or not out:
        out.append(f"{SHIFT}{quantize_duration(gap)}")
    return out


def midi_to_tokens(path: str | Path, durations: bool = True, encoding: str = "chords") -> list[str]:
    """Parse a MIDI file into a list of tokens.

    Notes from every piano voice are grouped by the moment they start. Each
    moment's duration is the time until the next moment starts, so
    sustained notes are not re-struck and the rhythm of the piece is kept.
    Silences show up as longer durations.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown encoding {encoding!r}; choose from {ENCODINGS}.")
    onsets, end = _onsets(path)
    times = sorted(onsets)
    tokens: list[str] = []
    for i, t in enumerate(times):
        gap = times[i + 1] - t if i + 1 < len(times) else end - t
        if encoding == "events":
            midis = sorted({min(max(p.midi, PIANO_LOW), PIANO_HIGH) for p in onsets[t]})
            tokens.extend(f"{PITCH}{m}" for m in midis)
            tokens.extend(shift_tokens(gap) if durations else [f"{SHIFT}{DEFAULT_DURATION}"])
            continue
        base = pitches_token(onsets[t])
        if not durations:
            tokens.append(base)
            continue
        tokens.append(f"{base}{SEP}{quantize_duration(gap)}")
    return tokens


def is_pitch_token(token: str) -> bool:
    return token[:1] == PITCH and token[1:].isdigit()


def is_shift_token(token: str) -> bool:
    if token[:1] != SHIFT:
        return False
    try:
        float(token[1:])
        return True
    except ValueError:
        return False


def detect_encoding(tokens) -> str:
    """"events" if the tokens (e.g. a vocabulary) contain event tokens, else "chords"."""
    return "events" if any(is_pitch_token(t) for t in tokens) else "chords"


def split_token(token: str) -> tuple[str, float]:
    """Split a token into its musical part and its duration."""
    if SEP in token:
        base, dur = token.rsplit(SEP, 1)
        return base, float(dur)
    return token, DEFAULT_DURATION


def token_to_element(token: str):
    """Build a music21 note, chord or rest from a token. None for special tokens."""
    if token.startswith("<"):
        return None
    base, dur = split_token(token)
    if base == REST:
        el = note.Rest()
    elif "." in base or base.isdigit():
        # Normal-order pitch classes carry no octave; voice the chord upward from C4.
        pcs = [int(p) for p in base.split(".")]
        midi, prev = [], None
        for pc in pcs:
            m = 60 + pc
            while prev is not None and m <= prev:
                m += 12
            midi.append(m)
            prev = m
        el = chord.Chord(midi)
    else:
        el = note.Note(base)
    el.quarterLength = dur
    return el


def tokens_to_elements(tokens: list[str]) -> list[tuple[float, object]]:
    """(offset, music21 element) pairs for a token sequence of either encoding.

    Event tokens: pitches collect until a time step, which plays them together
    for that long. Extra time steps with no new pitches let the notes ring on.
    """
    out: list[tuple[float, object]] = []
    offset = 0.0
    pending: list[int] = []
    holding = None  # element that consecutive time steps extend

    def flush(length: float):
        nonlocal holding
        uniq = sorted(set(pending))
        el = note.Note(uniq[0]) if len(uniq) == 1 else chord.Chord(uniq)
        el.quarterLength = length
        out.append((offset, el))
        pending.clear()
        holding = el

    for tok in tokens:
        if tok.startswith("<"):
            continue
        if is_pitch_token(tok):
            holding = None
            pending.append(int(tok[1:]))
        elif is_shift_token(tok):
            step = float(tok[1:])
            if pending:
                flush(step)
            elif holding is not None:
                holding.quarterLength += step
            offset += step
        else:
            holding = None
            el = token_to_element(tok)
            if el is None:
                continue
            out.append((offset, el))
            offset += el.quarterLength
    if pending:  # trailing notes without a time step
        flush(DEFAULT_DURATION * 2)
    return out


def count_onsets(tokens: list[str]) -> int:
    """How many moments (a note or a chord struck together) a sequence contains."""
    return sum(1 for _, el in tokens_to_elements(tokens) if not isinstance(el, note.Rest))


class OnsetCounter:
    """Counts finished moments while tokens are generated one at a time.

    An event moment is finished by the time step after its pitches; a chord
    token is a whole moment by itself.
    """

    def __init__(self):
        self.count = 0
        self._open = False

    def add(self, token: str) -> int:
        if is_pitch_token(token):
            self._open = True
        elif is_shift_token(token):
            if self._open:
                self.count += 1
                self._open = False
        elif not token.startswith("<") and split_token(token)[0] != REST:
            self.count += 1
        return self.count


def tokens_to_midi(tokens: list[str], path: str | Path, bpm: int = 90) -> Path:
    """Write a token sequence to a MIDI file and return its path."""
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=bpm))
    for offset, el in tokens_to_elements(tokens):
        part.insert(offset, el)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part.write("midi", fp=str(path))
    return path


@lru_cache(maxsize=None)
def _transpose_base(base: str, semitones: int) -> str | None:
    """Transpose the musical part of a token. None if a note leaves the piano's range."""
    if base == REST or base.startswith("<") or is_shift_token(base):
        return base
    if is_pitch_token(base):
        midi = int(base[1:]) + semitones
        return f"{PITCH}{midi}" if PIANO_LOW <= midi <= PIANO_HIGH else None
    if "." in base or base.isdigit():
        pcs = sorted({(int(p) + semitones) % 12 for p in base.split(".")})
        if len(pcs) == 1:
            return str(pcs[0])
        # Recompute normal order rather than shifting it, so tokens match the tokenizer exactly.
        return ".".join(str(n) for n in chord.Chord(pcs).normalOrder)
    midi = pitch.Pitch(base).midi + semitones
    if not PIANO_LOW <= midi <= PIANO_HIGH:
        return None
    return pitch.Pitch(midi=midi).nameWithOctave


def transpose_tokens(tokens: list[str], semitones: int) -> list[str] | None:
    """Shift a whole token sequence by some semitones (a key change).

    Returns None if any note would fall off the piano keyboard, so the
    caller can skip that transposition.
    """
    if semitones == 0:
        return list(tokens)
    out = []
    for tok in tokens:
        if SEP in tok and not tok.startswith("<"):
            base, dur = tok.rsplit(SEP, 1)
            new = _transpose_base(base, semitones)
            if new is None:
                return None
            out.append(f"{new}{SEP}{dur}")
        else:
            new = _transpose_base(tok, semitones)
            if new is None:
                return None
            out.append(new)
    return out

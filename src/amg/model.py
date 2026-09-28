"""The LSTM model and the sampling used for generation."""

from __future__ import annotations

import numpy as np


def build_model(
    vocab_size: int,
    seq_len: int,
    embed_dim: int = 128,
    units: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_composers: int = 0,
    composer_dim: int = 16,
):
    """Embedding -> stacked LSTMs -> Dense -> softmax over the vocabulary.

    With num_composers > 0 the model takes a second input, the composer id.
    Its learned embedding is attached to every time step, so the whole
    window is read "in the style of" that composer.

    LSTM layers keep Keras defaults (tanh/sigmoid, no recurrent_dropout) so
    they run on the fast cuDNN kernel when a GPU is available. Dropout is
    applied between layers instead.
    """
    import keras
    from keras import layers

    tokens = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(tokens)
    inputs = [tokens]
    if num_composers > 0:
        composer = keras.Input(shape=(1,), dtype="int32", name="composer")
        c = layers.Embedding(num_composers, composer_dim, name="composer_embedding")(composer)
        c = layers.Flatten()(c)
        c = layers.RepeatVector(seq_len)(c)
        x = layers.Concatenate()([x, c])
        inputs.append(composer)
    for i in range(num_layers):
        last = i == num_layers - 1
        x = layers.LSTM(units, return_sequences=not last, name=f"lstm_{i + 1}")(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    # float32 output keeps the softmax stable under mixed precision.
    outputs = layers.Dense(vocab_size, activation="softmax", dtype="float32", name="next_token")(x)
    return keras.Model(inputs if len(inputs) > 1 else tokens, outputs, name="amg_lstm")


def build_transformer(
    vocab_size: int,
    seq_len: int,
    d_model: int = 256,
    num_layers: int = 6,
    heads: int = 4,
    ff_dim: int | None = None,
    dropout: float = 0.2,
    num_composers: int = 0,
):
    """A small decoder-only Transformer (GPT-style) that predicts every position.

    Tokens get a learned embedding plus a learned position vector (and, with
    num_composers > 0, the composer's embedding at every step). Each block is
    pre-norm causal self-attention followed by a feed-forward layer, so a
    position only sees tokens before it. The output is a probability
    distribution over the vocabulary at every position: (batch, seq_len, vocab).
    """
    import keras
    from keras import layers

    from .layers import PositionEmbedding

    ff_dim = ff_dim or 4 * d_model
    tokens = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size, d_model, name="embedding")(tokens)
    x = PositionEmbedding(seq_len, d_model, name="positions")(x)
    inputs = [tokens]
    if num_composers > 0:
        composer = keras.Input(shape=(1,), dtype="int32", name="composer")
        c = layers.Embedding(num_composers, d_model, name="composer_embedding")(composer)
        c = layers.RepeatVector(seq_len)(layers.Flatten()(c))
        x = layers.Add()([x, c])
        inputs.append(composer)
    x = layers.Dropout(dropout)(x)
    for i in range(num_layers):
        h = layers.LayerNormalization(epsilon=1e-5, name=f"block{i + 1}_norm1")(x)
        h = layers.MultiHeadAttention(heads, d_model // heads, dropout=dropout,
                                      name=f"block{i + 1}_attention")(h, h, use_causal_mask=True)
        x = layers.Add()([x, layers.Dropout(dropout)(h)])
        h = layers.LayerNormalization(epsilon=1e-5, name=f"block{i + 1}_norm2")(x)
        h = layers.Dense(ff_dim, activation="gelu", name=f"block{i + 1}_ff1")(h)
        h = layers.Dense(d_model, name=f"block{i + 1}_ff2")(h)
        x = layers.Add()([x, layers.Dropout(dropout)(h)])
    x = layers.LayerNormalization(epsilon=1e-5, name="final_norm")(x)
    outputs = layers.Dense(vocab_size, activation="softmax", dtype="float32", name="next_token")(x)
    return keras.Model(inputs if len(inputs) > 1 else tokens, outputs, name="amg_transformer")


def next_token_probs(output) -> np.ndarray:
    """The distribution for the next token from one model call on a single window.

    LSTM models return (1, vocab); Transformers return (1, seq_len, vocab),
    where the last position is the prediction after the whole window.
    """
    probs = np.asarray(output)[0]
    return probs[-1] if probs.ndim == 2 else probs


def sample_next(
    probs: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    banned: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Pick the next token id from a probability vector.

    temperature < 1 plays it safe, > 1 takes more risks; 0 means always the
    most likely token. top_k keeps only the k most likely tokens.
    """
    rng = rng or np.random.default_rng()
    p = np.asarray(probs, dtype=np.float64).copy()
    if banned:
        p[banned] = 0.0
    if temperature <= 0:
        return int(np.argmax(p))
    logits = np.log(np.maximum(p, 1e-12)) / temperature
    if banned:
        logits[banned] = -np.inf
    if top_k is not None and 0 < top_k < len(logits):
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits[logits < cutoff] = -np.inf
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return int(rng.choice(len(weights), p=weights))

"""Custom Keras layers and metrics. Importing this module registers them, so saved
Transformer models can be loaded with keras.models.load_model."""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="amg")
class PositionEmbedding(keras.layers.Layer):
    """Adds a learned vector for each position 0..seq_len-1 to the input."""

    def __init__(self, seq_len: int, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len, self.dim = seq_len, dim

    def build(self, input_shape):
        self.table = self.add_weight(name="table", shape=(self.seq_len, self.dim),
                                     initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        length = ops.shape(x)[1]
        return x + ops.cast(self.table[:length], x.dtype)

    def get_config(self):
        return {**super().get_config(), "seq_len": self.seq_len, "dim": self.dim}


@keras.saving.register_keras_serializable(package="amg")
class LastTokenAccuracy(keras.metrics.Mean):
    """Accuracy of the prediction at the last position only.

    A Transformer predicts every position of the window, and early positions
    have little context. This number matches what the LSTM's accuracy
    measures (the token after a full window), so runs can be compared.
    """

    def __init__(self, name="last_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) == 3:
            y_true, y_pred = y_true[:, -1], y_pred[:, -1, :]
        hit = ops.cast(ops.equal(ops.cast(y_true, "int32"),
                                 ops.cast(ops.argmax(y_pred, axis=-1), "int32")), "float32")
        return super().update_state(hit, sample_weight=sample_weight)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}

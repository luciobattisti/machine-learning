import tensorflow as tf

from attention import GlobalSelfAttention
from feed_forward import FeedForward
from positional_encoding import PositionalEmbedding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


if __name__ == "__main__":

    # Import dependencies
    import numpy as np

    # Instantiate the encoder.
    sample_encoder = Encoder(
        num_layers=4, d_model=512, num_heads=8, dff=2048, vocab_size=8500
    )

    pt = tf.cast(
        np.random.randint(0, 2048, size=(64, 67)),
    )
    sample_encoder_output = sample_encoder(pt, training=False)

    # Print the shape.
    print(pt.shape)
    print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.

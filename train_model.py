

import tensorflow as tf
from create_dataset import create_dataset  # Import dataset creation function

# Define PositionalEncoding class
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.pos_embedding = tf.keras.layers.Embedding(max_length + 1, d_model, mask_zero=True)

    def build(self, input_shape):
        self.pos_embedding.build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(0, seq_len, dtype=tf.int32)
        return inputs + self.pos_embedding(positions)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({'max_length': self.max_length, 'd_model': self.d_model})
        return config

# Define TransformerDecoder class
@tf.keras.utils.register_keras_serializable()
class TransformerDecoder(tf.keras.models.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, max_length, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = PositionalEncoding(max_length, embed_dim)
        self.encoder_projection = tf.keras.layers.Dense(embed_dim, activation="relu")
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.ffn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.max_length = max_length
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def build(self, input_shape):
        decoder_input_shape, encoder_input_shape = input_shape
        self.embedding.build(decoder_input_shape)
        embed_output_shape = [None, self.max_length, self.embed_dim]
        self.pos_encoding.build(embed_output_shape)
        self.encoder_projection.build(encoder_input_shape)
        encoder_output_shape = [None, self.max_length, self.embed_dim]
        self.attention.build(query_shape=embed_output_shape, value_shape=encoder_output_shape)
        self.ffn.build(embed_output_shape)
        self.layernorm1.build(embed_output_shape)
        self.layernorm2.build(embed_output_shape)
        self.final_layer.build(embed_output_shape)
        self.built = True

    def call(self, inputs, training=False):
        decoder_inputs, encoder_outputs = inputs
        batch_size = tf.shape(decoder_inputs)[0]
        seq_len = tf.shape(decoder_inputs)[-1]
        decoder_inputs = tf.reshape(decoder_inputs, [batch_size, seq_len])
        
        x = self.embedding(decoder_inputs)
        x = self.pos_encoding(x)
        
        encoder_outputs = self.encoder_projection(encoder_outputs)
        encoder_outputs = tf.reshape(encoder_outputs, [batch_size, self.embed_dim])
        
        encoder_outputs = tf.expand_dims(encoder_outputs, 1)
        multiples = [1, seq_len, 1]
        encoder_outputs = tf.tile(encoder_outputs, multiples)
        
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attn_output = self.attention(query=x, value=encoder_outputs, key=encoder_outputs,
                                    attention_mask=causal_mask)
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output, training=training))
        return self.final_layer(x)

    def train_step(self, data):
        (decoder_inputs, encoder_outputs), targets = data
        batch_size = tf.shape(decoder_inputs)[0]
        seq_len = tf.shape(decoder_inputs)[-1]
        decoder_inputs = tf.reshape(decoder_inputs, [batch_size, seq_len])
        encoder_outputs = tf.reshape(encoder_outputs, [batch_size, self.embed_dim])
        targets = tf.reshape(targets, [batch_size, seq_len])
        
        with tf.GradientTape() as tape:
            predictions = self([decoder_inputs, encoder_outputs], training=True)
            loss = self.compiled_loss(targets, predictions)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(targets, predictions)
        
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def test_step(self, data):
        (decoder_inputs, encoder_outputs), targets = data
        batch_size = tf.shape(decoder_inputs)[0]
        seq_len = tf.shape(decoder_inputs)[-1]
        decoder_inputs = tf.reshape(decoder_inputs, [batch_size, seq_len])
        encoder_outputs = tf.reshape(encoder_outputs, [batch_size, self.embed_dim])
        targets = tf.reshape(targets, [batch_size, seq_len])
        
        predictions = self([decoder_inputs, encoder_outputs], training=False)
        loss = self.compiled_loss(targets, predictions)
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(targets, predictions)
        
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.embedding.input_dim,
            'embed_dim': self.embed_dim,
            'num_heads': self.attention.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'max_length': self.max_length,
            'dropout_rate': self.dropout1.rate
        })
        return config

def train_model():
    # Load dataset
    dataset, _, _ = create_dataset()

    # Model parameters
    vocab_size = 15000
    embed_dim = 512
    num_heads = 12
    ff_dim = 2048
    max_length = 80
    dropout_rate = 0.2

    # Build fresh model
    decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, ff_dim, max_length, dropout_rate)
    decoder.build([(None, max_length), (None, 512)])

    # Dummy forward pass to initialize weights
    dummy_decoder_input = tf.zeros((1, max_length), dtype=tf.int32)
    dummy_encoder_input = tf.zeros((1, 512), dtype=tf.float32)
    _ = decoder([dummy_decoder_input, dummy_encoder_input])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, clipnorm=1.0, weight_decay=1e-4)
    decoder.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"])

    # Split dataset into train and validation
    dataset_size = 158817  # Based on your previous outputs
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset = dataset.take(train_size // 32).shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size // 32).prefetch(tf.data.AUTOTUNE)

    # Train with validation
    print("Training new model with current data and validation...")
    decoder.fit(train_dataset, epochs=15, validation_data=val_dataset)

    # Save model
    decoder.save(r"D:\Image_Captioning_using_Deep_Learning\transformer_decoder_model_fresh.keras")
    print("Model saved at D:\\Image_Captioning_using_Deep_Learning\\transformer_decoder_model_fresh.keras")

    # Verify weights
    print("Embedding layer weights shape:", decoder.embedding.get_weights()[0].shape)

if __name__ == "__main__":
    train_model()
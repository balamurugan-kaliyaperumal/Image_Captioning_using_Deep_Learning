import tensorflow as tf
import numpy as np
import json

# Define global vocab_size (used in inference functions)
vocab_size = 15000  # Match your model's vocab_size


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

# Define TransformerDecoder class (fixed)
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

# Load the trained model with custom objects defined
decoder = tf.keras.models.load_model(r"C:\Users\BALAMURUGAN\Downloads\transformer_decoder_model_fresh (2).keras", custom_objects={
    "PositionalEncoding": PositionalEncoding,
    "TransformerDecoder": TransformerDecoder
})
print("Model loaded successfully from C:\\Users\\BALAMURUGAN\\Downloads\\transformer_decoder_model_fresh (2).keras")

# Load annotations and tokenizer
with open(r"D:\Project\dataset\annotations_english.json", "r") as file:
    captions_data = json.load(file)
with open(r"C:\Users\BALAMURUGAN\Downloads\tokenizer (5).json", "r") as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Load ViT features
vit_features = np.load(r"D:\grok\Image_Captioning_using_Deep_Learning\vit_features_merged.npy")[:31764]
image_to_feature = {filename: vit_features[i] for i, filename in enumerate(captions_data.keys())}
image_filenames = [img for img, details in captions_data.items() for _ in details["comments"]]
vit_features_expanded = np.array([image_to_feature[filename] for filename in image_filenames], dtype=np.float32)
vit_features_expanded = (vit_features_expanded - np.mean(vit_features_expanded, axis=0)) / np.std(vit_features_expanded, axis=0)

# Sampling-based caption generation
def generate_caption_sampling(image_feature, max_length=80, penalty_steps=5, min_length=15):
    decoder_input = tf.constant([[1]])  # <start> token
    caption_ids = [1]
    image_feature = tf.expand_dims(image_feature, 0)
    used_ids = set()
    recent_ids = []

    for step in range(max_length - 1):
        predictions = decoder([decoder_input, image_feature], training=False)
        logits = predictions[:, -1, :]
        
        adjusted_logits = logits.numpy().copy()
        for i in range(vocab_size):
            if i in recent_ids:
                adjusted_logits[0, i] -= 5.0
            elif i in used_ids:
                adjusted_logits[0, i] -= 1.0
            if i == 2:
                adjusted_logits[0, i] -= 20.0 * (min_length - len(caption_ids)) / min_length if len(caption_ids) < min_length else 0.0
            if i == 3:
                adjusted_logits[0, i] -= 2.0
        
        adjusted_logits /= 0.05  # Sharper predictions
        probs = tf.nn.softmax(adjusted_logits).numpy()
        
        top_k = 150  # Wider pool
        top_k_probs, top_k_ids = tf.nn.top_k(probs[0], k=top_k)
        top_k_probs = top_k_probs.numpy()
        top_k_ids = top_k_ids.numpy()
        
        mask = top_k_ids != 3
        if np.any(mask):
            top_k_probs = top_k_probs[mask]
            top_k_ids = top_k_ids[mask]
            top_k_probs /= top_k_probs.sum()
        
        predicted_id = np.random.choice(top_k_ids, p=top_k_probs)
        
        top_5 = tf.nn.top_k(probs[0], k=5)
        print(f"Sampling Step {step + 1} - Top 5 Predictions: IDs {top_5.indices.numpy()}, Probs {top_5.values.numpy()}, Words {[tokenizer.index_word.get(id, '<unk>') for id in top_5.indices.numpy()]}")
        
        caption_ids.append(predicted_id)
        used_ids.add(predicted_id)
        recent_ids.append(predicted_id)
        if len(recent_ids) > penalty_steps:
            recent_ids.pop(0)
        
        decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=1)
        
        if predicted_id == 2 and len(caption_ids) >= min_length:
            break
    
    caption = " ".join([tokenizer.index_word.get(id, "<unk>") for id in caption_ids[1:]])
    return caption, caption_ids

# Beam search-based caption generation
def generate_caption_beam_search(image_feature, max_length=80, beam_width=5):
    image_feature = tf.expand_dims(image_feature, 0)
    start_seq = [[1], 0.0]  # [token_ids, score]
    candidates = [start_seq]
    
    for step in range(max_length - 1):
        all_candidates = []
        for seq, score in candidates:
            decoder_input = tf.constant([seq])
            predictions = decoder([decoder_input, image_feature], training=False)
            logits = predictions[:, -1, :]
            probs = tf.nn.softmax(logits / 0.1).numpy()[0]
            top_k_probs, top_k_ids = tf.nn.top_k(probs, k=beam_width)
            
            if seq == candidates[0][0]:
                top_5 = tf.nn.top_k(probs, k=5)
                print(f"Beam Search Step {step + 1} (Best Candidate) - Top 5 Predictions: IDs {top_5.indices.numpy()}, Probs {top_5.values.numpy()}, Words {[tokenizer.index_word.get(id, '<unk>') for id in top_5.indices.numpy()]}")
            
            for i in range(beam_width):
                new_seq = seq + [top_k_ids[i].numpy()]
                length_penalty = 0.5 * (len(new_seq) - 1) / max_length  # Encourage longer sequences
                new_score = score + np.log(top_k_probs[i].numpy() + 1e-10) - length_penalty
                all_candidates.append([new_seq, new_score])
        
        candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if any(seq[-1] == 2 for seq, _ in candidates):
            break
    
    best_seq, _ = candidates[0]
    end_idx = best_seq.index(2) if 2 in best_seq else len(best_seq)
    caption = " ".join([tokenizer.index_word.get(id, "<unk>") for id in best_seq[1:end_idx]])
    return caption, best_seq[:end_idx]


test_images = ["1000092795.jpg", "10002456.jpg"]
for img_name in test_images:
    idx = image_filenames.index(img_name)
    feature = vit_features_expanded[idx]
    print(f"\nGenerating caption for {img_name} at index {idx} (feature from caption index {idx})")
    print(f"Starting sequence: [1]")
    print(f"Image feature mean: {np.mean(feature)}, std: {np.std(feature)}")
    
    # Sampling-based generation
    print("\n--- Sampling-Based Generation ---")
    caption_sampling, caption_ids_sampling = generate_caption_sampling(feature)
    print(f"Generated Caption IDs (Sampling): {caption_ids_sampling}")
    print(f"Generated Caption (Sampling): {caption_sampling}")
    
    # Beam search-based generation
    print("\n--- Beam Search-Based Generation ---")
    caption_beam, caption_ids_beam = generate_caption_beam_search(feature)
    print(f"Generated Caption IDs (Beam Search): {caption_ids_beam}")
    print(f"Generated Caption (Beam Search): {caption_beam}")
    
    # Actual annotations
    actual_captions = captions_data[img_name]["comments"]
    print(f"Actual Annotations for {img_name}: {actual_captions}")

# Verify tokenizer vocabulary
print("\nTokenizer Vocabulary Check:")
print(f"Total words: {len(tokenizer.word_index)}")
print(f"Sample mappings: {dict(list(tokenizer.word_index.items())[:10])}")
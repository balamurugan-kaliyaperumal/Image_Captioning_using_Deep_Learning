# app.py
# Flask application to upload an image, extract feature vectors, and generate captions using a trained TransformerDecoder model.

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf
import json

app = Flask(__name__)

# Define TransformerDecoder and PositionalEncoding classes (from train_model.py)
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

# Load the trained model and tokenizer
decoder = tf.keras.models.load_model(r"C:\Users\BALAMURUGAN\Downloads\transformer_decoder_model_fresh (2).keras", custom_objects={
    "PositionalEncoding": PositionalEncoding,
    "TransformerDecoder": TransformerDecoder
})
with open(r"C:\Users\BALAMURUGAN\Downloads\tokenizer (5).json", "r") as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
vocab_size = 15000  # Match your model's vocab_size

# Define PatchPositionEmbedding for ViT feature extraction
class PatchPositionEmbedding(nn.Module):
    def __init__(self, num_patches=196, embedding_dim=512):
        super(PatchPositionEmbedding, self).__init__()
        self.position_embedding = self.generate_sinusoidal_embeddings(num_patches, embedding_dim)

    @staticmethod
    def generate_sinusoidal_embeddings(num_patches, embedding_dim):
        embeddings = torch.zeros((num_patches, embedding_dim))
        for pos in range(num_patches):
            pos_tensor = torch.tensor(float(pos), dtype=torch.float32)
            for i in range(0, embedding_dim, 2):
                i_tensor = torch.tensor(float(i), dtype=torch.float32)
                denom = torch.pow(torch.tensor(10000.0, dtype=torch.float32), (2 * i_tensor) / embedding_dim)
                embeddings[pos, i] = torch.sin(pos_tensor / denom)
                if i + 1 < embedding_dim:
                    embeddings[pos, i + 1] = torch.cos(pos_tensor / denom)
        return nn.Parameter(embeddings, requires_grad=False)

    def forward(self, x):
        return x + self.position_embedding

# Flask app setup
UPLOAD_FOLDER = "static/uploads"
RESIZED_FOLDER = "static/resized_images"
PATCHES_FOLDER = "static/patches"
NORMALIZED_FOLDER = "static/normalized_images"
TEXT_OUTPUT_FOLDER = "static/text_output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESIZED_FOLDER, exist_ok=True)
os.makedirs(PATCHES_FOLDER, exist_ok=True)
os.makedirs(NORMALIZED_FOLDER, exist_ok=True)
os.makedirs(TEXT_OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_jpg(image_path):
    img = Image.open(image_path)
    rgb_img = img.convert("RGB")
    new_path = os.path.splitext(image_path)[0] + ".jpg"
    rgb_img.save(new_path, "JPEG")
    return new_path

def resize_image(image_path):
    img = Image.open(image_path)
    img_resized = img.resize((224, 224))
    resized_path = os.path.join(RESIZED_FOLDER, os.path.basename(image_path))
    img_resized.save(resized_path)
    return resized_path, img_resized

def normalize_image(image):
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_img = (img_array - mean) / std
    normalized_img = np.clip(normalized_img, 0, 1)
    normalized_pil = Image.fromarray((normalized_img * 255).astype(np.uint8))
    normalized_path = os.path.join(NORMALIZED_FOLDER, "normalized.jpg")
    normalized_pil.save(normalized_path)
    return normalized_path, normalized_pil

linear_proj_layer = nn.Linear(768, 512)  # For ViT feature projection

def apply_linear_projection(flattened_patches, model):
    patches_tensor = torch.tensor(flattened_patches, dtype=torch.float32)
    projected_patches = model(patches_tensor)
    np.savetxt(os.path.join(TEXT_OUTPUT_FOLDER, "768_token_embeddings.txt"), flattened_patches, delimiter=",")
    np.savetxt(os.path.join(TEXT_OUTPUT_FOLDER, "512_token_embeddings.txt"), projected_patches.detach().cpu().numpy(), delimiter=",")
    return projected_patches

def apply_patch_position_embedding(projected_embeddings):
    num_patches = projected_embeddings.shape[0]
    embedding_dim = projected_embeddings.shape[1]
    position_model = PatchPositionEmbedding(num_patches, embedding_dim)
    position_encoded = position_model(projected_embeddings.unsqueeze(0)).squeeze(0)
    with open(os.path.join(TEXT_OUTPUT_FOLDER, "512_tokens_position_encoding.txt"), "w") as f:
        for embedding in position_encoded.detach().numpy():
            f.write(','.join(map(str, embedding)) + "\n")
    return position_encoded

def apply_mhsa(patches_tensor, embed_dim=512, num_heads=8):
    head_dim = embed_dim // num_heads
    assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

    W_q = nn.Linear(embed_dim, embed_dim, bias=False)
    W_k = nn.Linear(embed_dim, embed_dim, bias=False)
    W_v = nn.Linear(embed_dim, embed_dim, bias=False)
    W_out = nn.Linear(embed_dim, embed_dim, bias=False)

    Q = W_q(patches_tensor)
    K = W_k(patches_tensor)
    V = W_v(patches_tensor)

    Q = Q.view(-1, num_heads, head_dim).transpose(0, 1)
    K = K.view(-1, num_heads, head_dim).transpose(0, 1)
    V = V.view(-1, num_heads, head_dim).transpose(0, 1)

    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
    attention_weights = softmax(attention_scores, dim=-1)

    attn_output = torch.matmul(attention_weights, V)
    attn_output = attn_output.transpose(0, 1).reshape(-1, embed_dim)
    attn_output = W_out(attn_output)
    return attn_output.detach().numpy()

def apply_layer_norm(input_tensor):
    norm_layer = nn.LayerNorm(input_tensor.shape[-1])
    return norm_layer(input_tensor)

def apply_feed_forward_network(input_tensor, hidden_dim=2048, output_dim=512):
    ffn_layer1 = nn.Linear(input_tensor.shape[-1], hidden_dim)
    ffn_layer2 = nn.Linear(hidden_dim, output_dim)

    hidden = torch.relu(ffn_layer1(input_tensor))
    output = ffn_layer2(hidden)
    return output

def add_and_norm(ffn_output, input_tensor):
    residual = ffn_output + input_tensor
    norm_layer = nn.LayerNorm(ffn_output.shape[-1])
    normalized_output = norm_layer(residual)
    return normalized_output

def extract_patches(image):
    patch_size = 16
    img_array = np.array(image)
    h, w, _ = img_array.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    patch_paths = []
    flattened_patches = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = img_array[i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size, :]
            patch_img = Image.fromarray(patch)
            patch_filename = f"patch_{i}_{j}.jpg"
            patch_path = os.path.join(PATCHES_FOLDER, patch_filename)
            patch_img.save(patch_path)
            patch_paths.append("/" + patch_path)
            patch_vector = patch.flatten().tolist()
            flattened_patches.append(patch_vector)

    projected_patches = apply_linear_projection(flattened_patches, linear_proj_layer)
    embedded_patches = apply_patch_position_embedding(projected_patches)
    patches_tensor = embedded_patches.clone().detach()
    attn_output = apply_mhsa(patches_tensor)
    normalized_output = apply_layer_norm(torch.tensor(attn_output, dtype=torch.float32))
    ffn_output = apply_feed_forward_network(normalized_output)
    final_output = add_and_norm(ffn_output, normalized_output)

    # Average across patches to get a single 512D feature vector
    feature_vector = final_output.mean(dim=0).detach().numpy()
    return patch_paths, feature_vector

# Caption generation functions
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
        
        adjusted_logits /= 0.05
        probs = tf.nn.softmax(adjusted_logits).numpy()
        
        top_k = 150
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
    return caption

def generate_caption_beam_search(image_feature, max_length=80, beam_width=5):
    image_feature = tf.expand_dims(image_feature, 0)
    start_seq = [[1], 0.0]
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
                length_penalty = 0.5 * (len(new_seq) - 1) / max_length
                new_score = score + np.log(top_k_probs[i].numpy() + 1e-10) - length_penalty
                all_candidates.append([new_seq, new_score])
        
        candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if any(seq[-1] == 2 for seq, _ in candidates):
            break
    
    best_seq, _ = candidates[0]
    end_idx = best_seq.index(2) if 2 in best_seq else len(best_seq)
    caption = " ".join([tokenizer.index_word.get(id, "<unk>") for id in best_seq[1:end_idx]])
    return caption

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        if not filename.lower().endswith(".jpg"):
            filepath = convert_to_jpg(filepath)

        resized_path, img_resized = resize_image(filepath)
        normalized_path, img_normalized = normalize_image(img_resized)
        patch_paths, feature_vector = extract_patches(img_normalized)

        # Generate captions using the feature vector
        caption_sampling = generate_caption_sampling(feature_vector)
        caption_beam = generate_caption_beam_search(feature_vector)

        return jsonify({
            "caption": caption_sampling,  # Display sampling caption in index.html
            "caption_beam": caption_beam,  # Optional: return beam search caption too
            "image_path": "/" + filepath,
            "resized_image_path": "/" + resized_path,
            "normalized_image_path": "/" + normalized_path,
            "patches": patch_paths,
            "768_token_embeddings_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "768_token_embeddings.txt"),
            "512_token_embeddings_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "512_token_embeddings.txt"),
            "512_tokens_position_encoding_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "512_tokens_position_encoding.txt"),
            "final_encoder_output_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "final_encoder_output.txt")
        })

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
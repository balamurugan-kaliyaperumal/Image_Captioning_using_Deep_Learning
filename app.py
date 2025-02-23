from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

linear_proj_layer = nn.Linear(768, 512)  # Define globally


linear_proj_layer = nn.Linear(768, 512)  # Define globally

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

    return attn_output.detach().numpy().tolist()

'''
def apply_mhsa(patches_tensor, embed_dim=512, num_heads=8):
    head_dim = embed_dim // num_heads
    W_q = torch.randn(embed_dim, embed_dim)
    W_k = torch.randn(embed_dim, embed_dim)
    W_v = torch.randn(embed_dim, embed_dim)

    Q = torch.matmul(patches_tensor, W_q)
    K = torch.matmul(patches_tensor, W_k)
    V = torch.matmul(patches_tensor, W_v)

    attention_scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
    attention_weights = softmax(attention_scores, dim=-1)

    attn_output = torch.matmul(attention_weights, V)

    with open(os.path.join(TEXT_OUTPUT_FOLDER, "mhsa_output.txt"), "w") as f:
        for row in attn_output.detach().numpy():
            f.write(",".join(map(str, row)) + "\n")

    return attn_output.detach().numpy().tolist()
    '''
def apply_layer_norm(input_tensor):
    norm_layer = nn.LayerNorm(input_tensor.shape[-1])
    return norm_layer(input_tensor)

def apply_feed_forward_network(input_tensor, hidden_dim=2048, output_dim=512):
    ffn_layer1 = nn.Linear(input_tensor.shape[-1], hidden_dim)
    ffn_layer2 = nn.Linear(hidden_dim, output_dim)

    hidden = torch.relu(ffn_layer1(input_tensor))
    output = ffn_layer2(hidden)

    with open(os.path.join(TEXT_OUTPUT_FOLDER, "ffn_output.txt"), "w") as f:
        for row in output.detach().numpy():
            f.write(",".join(map(str, row)) + "\n")

    return output

def add_and_norm(ffn_output, input_tensor):
    residual = ffn_output + input_tensor  # Add
    norm_layer = nn.LayerNorm(ffn_output.shape[-1])
    normalized_output = norm_layer(residual)  # Normalize

    with open(os.path.join(TEXT_OUTPUT_FOLDER, "add_norm_output.txt"), "w") as f:
        for row in normalized_output.detach().numpy():
            f.write(",".join(map(str, row)) + "\n")

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

    #patches_tensor = torch.tensor(embedded_patches, dtype=torch.float32).clone().detach()
    patches_tensor = embedded_patches.clone().detach()

    attn_output = apply_mhsa(patches_tensor)

    patches_file = os.path.join(TEXT_OUTPUT_FOLDER, "patch_position_embeddings.txt")
    with open(patches_file, "w") as f:
        for patch in attn_output:
            f.write(",".join(map(str, patch)) + "\n")

    residual = patches_tensor
    mhsa_output = torch.tensor(attn_output, dtype=torch.float32)
    add_norm_output = mhsa_output + residual
    normalized_output = apply_layer_norm(add_norm_output)

    ffn_output = apply_feed_forward_network(normalized_output)
    final_output = add_and_norm(ffn_output,normalized_output)

    with open(os.path.join(TEXT_OUTPUT_FOLDER, "final_encoder_output.txt"), "w") as f:
        for row in final_output.detach().numpy():
            f.write(",".join(map(str, row)) + "\n")

    return patch_paths, attn_output, patches_file

@app.route('/')
def home():
    return render_template('index.html')

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
        patch_paths, attn_output, patches_file = extract_patches(img_normalized)

        return jsonify({
            "caption": "This is a sample caption.",
            "image_path": "/" + filepath,
            "resized_image_path": "/" + resized_path,
            "normalized_image_path": "/" + normalized_path,
            "patches": patch_paths,
            "manual_mhsa_output": attn_output,
            "patch_position_embeddings_file": "/" + patches_file,
            "768_token_embeddings_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "768_token_embeddings.txt"),
            "512_token_embeddings_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "512_token_embeddings.txt"),
            "mhsa_output_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "mhsa_output.txt"),
            "add_norm_output_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "add_norm_output.txt"),
            "ffn_output_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "ffn_output.txt"),
            "final_encoder_output_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "final_encoder_output.txt")
        })

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
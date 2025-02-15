from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

class LinearProjection(nn.Module):
    def __init__(self):
        super(LinearProjection, self).__init__()
        self.fc = nn.Linear(768, 512) 

    def forward(self, x):
        return self.fc(x)

class PatchPositionEmbedding(nn.Module):
    def __init__(self, num_patches=196, embedding_dim=512):
        super(PatchPositionEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))  

    def forward(self, x):
        return x + self.position_embedding  

def apply_linear_projection(flattened_patches):
    model = LinearProjection()
    patches_tensor = torch.tensor(flattened_patches, dtype=torch.float32)
    projected_patches = model(patches_tensor)  

    # Save 768-token embeddings
    with open(os.path.join(TEXT_OUTPUT_FOLDER, "768_token_embeddings.txt"), "w") as f:
        for patch in flattened_patches:
            f.write(",".join(map(str, patch)) + "\n")

    # Save 512-token embeddings
    with open(os.path.join(TEXT_OUTPUT_FOLDER, "512_token_embeddings.txt"), "w") as f:
        for patch in projected_patches.detach().numpy():
            f.write(",".join(map(str, patch)) + "\n")

    return projected_patches

def apply_patch_position_embedding(flattened_patches):
    patch_position_embedding = PatchPositionEmbedding()
    embedded_patches = patch_position_embedding(flattened_patches.unsqueeze(0))  
    return embedded_patches.squeeze(0).tolist()

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

    projected_patches = apply_linear_projection(flattened_patches)

    embedded_patches = apply_patch_position_embedding(projected_patches)

    patches_file = os.path.join(TEXT_OUTPUT_FOLDER, "patch_position_embeddings.txt")
    with open(patches_file, "w") as f:
        for patch in embedded_patches:
            f.write(",".join(map(str, patch)) + "\n")

    return patch_paths, embedded_patches, patches_file

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
        patch_paths, embedded_patches, patches_file = extract_patches(img_normalized)

        return jsonify({
            "caption": "This is a sample caption.",
            "image_path": "/" + filepath,
            "resized_image_path": "/" + resized_path,
            "normalized_image_path": "/" + normalized_path,
            "patches": patch_paths,
            "embedded_patches": embedded_patches,
            "patch_position_embeddings_file": "/" + patches_file,
            "768_token_embeddings_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "768_token_embeddings.txt"),
            "512_token_embeddings_file": "/" + os.path.join(TEXT_OUTPUT_FOLDER, "512_token_embeddings.txt"),
        })

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
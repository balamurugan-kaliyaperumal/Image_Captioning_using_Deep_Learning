from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESIZED_FOLDER = "static/resized_images"
NORMALIZED_FOLDER = "static/normalized_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESIZED_FOLDER, exist_ok=True)
os.makedirs(NORMALIZED_FOLDER, exist_ok=True)

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
    return normalized_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_normalize', methods=['POST'])
def upload_normalize():
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
        normalized_path = normalize_image(img_resized)

        return jsonify({
            "original_image": "/" + filepath,
            "resized_image": "/" + resized_path,
            "normalized_image": "/" + normalized_path
        })

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)

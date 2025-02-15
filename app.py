from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_jpg(image_path):
    img = Image.open(image_path)
    rgb_img = img.convert("RGB")
    new_path = os.path.splitext(image_path)[0] + ".jpg"
    rgb_img.save(new_path, "JPEG")
    return new_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
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

        return jsonify({"message": "Image uploaded and converted to JPG successfully.", "image_path": "/" + filepath})

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
import os

app = Flask(__name__)

# Directory to save images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_resize', methods=['POST'])
def upload_resize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(original_path)

        
        img = Image.open(original_path)
        jpg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted.jpg')
        img.convert('RGB').save(jpg_path, 'JPEG')

        
        resized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized.jpg')
        img_resized = img.resize((224, 224))
        img_resized.convert('RGB').save(resized_path, 'JPEG')

        
        original_url = url_for('static', filename='uploads/converted.jpg')
        resized_url = url_for('static', filename='uploads/resized.jpg')

        return jsonify({'original_image': original_url, 'resized_image': resized_url})

    return jsonify({'error': 'File processing failed'}), 500


if __name__ == '__main__':
    app.run(debug=True)

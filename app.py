import os
import secrets
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

IMG_SIZE = (224, 224)
model = None

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

DISEASE_RECOMMENDATIONS = {
    'healthy': [
        'Continue regular watering and care',
        'Ensure adequate sunlight and nutrients',
        'Monitor for any changes in appearance',
        'Maintain good air circulation'
    ],
    'default_disease': [
        'Isolate affected plants to prevent spread',
        'Consult with an agricultural expert',
        'Consider appropriate treatment methods',
        'Monitor other plants for similar symptoms'
    ]
}


def load_model():
    global model
    model_path = 'mobilenetv2_best.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully from", model_path)
    else:
        print(f"WARNING: Model file '{model_path}' not found. Predictions will not work.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(image_path):
    if model is None:
        raise ValueError("Model not loaded.")
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    class_name = CLASS_NAMES[predicted_index]

    parts = class_name.split('___')
    plant_type = parts[0].replace('_', ' ').replace('(', '').replace(')', '').strip()
    condition = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'

    is_healthy = 'healthy' in condition.lower()
    recommendations = DISEASE_RECOMMENDATIONS['healthy'] if is_healthy else DISEASE_RECOMMENDATIONS['default_disease']

    return {
        'plant_type': plant_type,
        'condition': condition,
        'confidence': round(confidence * 100, 2),
        'is_healthy': is_healthy,
        'recommendations': recommendations,
        'class_name': class_name
    }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            prediction = predict_image(filepath)

            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config['STATIC_FOLDER'], 'images', static_filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            Image.open(filepath).save(static_path)

            session['prediction'] = prediction
            session['image_path'] = f'images/{static_filename}'

            os.remove(filepath)

            return jsonify({'success': True})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG.'}), 400


@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')

    if not prediction:
        return redirect(url_for('upload'))

    return render_template('result.html', prediction=prediction, image_path=image_path)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs(os.path.join('static', 'images'), exist_ok=True)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

"""
app.py
------
Flask web app for face recognition.
Routes:
GET  /          — upload form
POST /predict   — upload image → recognition result
GET  /train     — trigger training from the web  
"""


import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from utils import load_single_image
from pca import PCA
from ann import ANN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024   # 8 MB max upload
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'pgm'}
IMG_SIZE      = (50, 50)
CONF_THRESHOLD = 20.0   # % confidence below which we call it an imposter

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------------------------------------------------ #
#  Load models at startup (if they exist)
# ------------------------------------------------------------------ #
pca_model   = None
ann_model   = None
label_names = []

def load_models():
    global pca_model, ann_model, label_names
    try:
        with open('label_names.pkl', 'rb') as f:
            label_names = pickle.load(f)
        pca_model = PCA()
        pca_model.load('pca_model.pkl')
        ann_model = ANN(layer_sizes=[])
        ann_model.load('ann_model.pkl')
        print(f"Models loaded. Classes: {label_names}")
    except FileNotFoundError:
        print("No saved models found — train first via /train or train_model.py")

load_models()

# ------------------------------------------------------------------ #
#  Load Iris models at startup (if they exist)
# ------------------------------------------------------------------ #
iris_pca   = None
iris_ann   = None
iris_labels = []
def load_iris_models():
    global iris_pca, iris_ann, iris_labels
    try:
        with open('iris_label_names.pkl', 'rb') as f:
            iris_labels = pickle.load(f)
        iris_pca = PCA()
        iris_pca.load('iris_pca_model.pkl')
        iris_ann = ANN(layer_sizes=[])
        iris_ann.load('iris_ann_model.pkl')
        print(f"Iris models loaded. Classes: {iris_labels}")
    except FileNotFoundError:
        print("No Iris models found - run iris_classify.py first")
load_iris_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #
@app.route('/')
def index():
    models_ready = pca_model is not None and ann_model is not None
    return render_template('index.html',
                           models_ready=models_ready,
                           label_names=label_names)


@app.route('/predict', methods=['POST'])
def predict():
    if pca_model is None or ann_model is None:
        return jsonify({'error': 'Models not loaded. Please train first.'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use jpg/png/bmp/pgm.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Step 1–3: Load, flatten, mean-zero, project
        face_vec  = load_single_image(filepath, IMG_SIZE)   # (mn, 1)
        signature = pca_model.transform(face_vec)           # (k, 1)

        proba      = ann_model.predict_proba(signature)     # (n_classes, 1)
        confidence = float(np.max(proba)) * 100
        pred_idx   = int(np.argmax(proba))

        is_imposter = confidence < CONF_THRESHOLD
        name = "Unknown (Imposter)" if is_imposter else label_names[pred_idx]

        # Top-3 predictions
        top3_idx  = np.argsort(proba[:, 0])[::-1][:3]
        top3 = [
            {'name': label_names[i], 'confidence': round(float(proba[i, 0]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify({
            'prediction': name,
            'confidence': round(confidence, 2),
            'is_imposter': is_imposter,
            'top3': top3
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/train', methods=['POST'])
def train():
    """
    Lightweight training trigger via web.
    POST body (JSON): { "k": 50, "epochs": 1000, "hidden": 128 }
    """
    import subprocess, sys
    data    = request.get_json(silent=True) or {}
    k       = int(data.get('k', 50))
    epochs  = int(data.get('epochs', 1000))
    
    hidden  = int(data.get('hidden', 128))

    try:
        result = subprocess.run(
            [sys.executable, 'train_model.py'],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, 'K': str(k), 'EPOCHS': str(epochs), 'HIDDEN': str(hidden)}
        )
        if result.returncode == 0:
            load_models()
            return jsonify({'status': 'ok', 'log': result.stdout[-3000:]})
        else:
            return jsonify({'status': 'error', 'log': result.stderr[-3000:]}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'log': 'Training timed out (>10 min).'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
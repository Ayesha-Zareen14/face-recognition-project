# Face Recognition — PCA + ANN

Internship project implementing face recognition from scratch using:
- **PCA** (Principal Component Analysis / Eigenfaces method — Turk & Pentland 1991)
- **ANN** (Feedforward Neural Network with Backpropagation)
- **Flask** web interface for live testing

---

## Project Structure

```
face_recognition_project/
├── pca.py            ← PCA implementation (eigenfaces)
├── ann.py            ← ANN with backpropagation
├── utils.py          ← dataset loader, image helpers
├── train_model.py    ← train + save models
├── test_model.py     ← evaluate saved models / recognise one image
├── evaluate.py       ← accuracy-vs-k plot + imposter analysis
├── main.py           ← single entry point for all commands
├── app.py            ← Flask web app
├── Templates/
│   └── index.html
├── static/
│   └── style.css
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Make sure your dataset is at:
```
dataset/
└── faces/
    ├── person1/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── person2/
    │   └── ...
    └── ...
```

---

## Run

### 1. Train the model
```bash
python main.py train
# or directly:
python train_model.py
```
Saves: `pca_model.pkl`, `ann_model.pkl`, `label_names.pkl`

### 2. Evaluate (accuracy vs k + imposter detection)
```bash
python main.py evaluate
# or directly:
python evaluate.py
```
Generates: `accuracy_vs_k.png`, `imposter_analysis.png`

### 3. Test on full test set
```bash
python main.py test
```

### 4. Recognise a single image
```bash
python main.py test path/to/face.jpg
```

### 5. Launch web app
```bash
python main.py web
# Open http://localhost:5000
```

---

## Algorithm Steps (as per requirements)

| Step | What happens | File |
|------|-------------|------|
| 1 | Build face database `(mn × p)` | `utils.py` |
| 2 | Calculate mean face `(mn × 1)` | `pca.py` |
| 3 | Mean-zero subtraction `Φ = Face_Db − M` | `pca.py` |
| 4 | Surrogate covariance `C = ΦᵀΦ/p` (p×p) | `pca.py` |
| 5 | Eigen-decomposition of C | `pca.py` |
| 6 | Select top-k eigenvectors | `pca.py` |
| 7 | Generate eigenfaces `W = Vᵀ × Φᵀ` | `pca.py` |
| 8 | Compute signatures `Ω = W × Φ` | `pca.py` |
| 9 | Train ANN on signatures | `ann.py` |
| Test | Project test face → ANN predict | `test_model.py` |

---

## Key settings (in `train_model.py`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `K` | 50 | Number of eigenfaces |
| `HIDDEN_UNITS` | 128 | Neurons in hidden layer |
| `LEARNING_RATE` | 0.01 | ANN learning rate |
| `EPOCHS` | 2000 | Training iterations |
| `TEST_RATIO` | 0.40 | 40% test, 60% train |
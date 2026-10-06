"""
test_model.py
-------------
Load saved PCA + ANN models and test a single image or the full test set.

Usage
-----
  python test_model.py                      # evaluates full test set
  python test_model.py path/to/image.jpg    # recognise one image
"""

import sys
import os
import pickle
import numpy as np

from utils import load_face_dataset, stratified_split, load_single_image
from pca import PCA
from ann import ANN

# ------------------------------------------------------------------ #
#  Config
# ------------------------------------------------------------------ #
DATASET_PATH = 'dataset/faces'
IMG_SIZE     = (50, 50)
TEST_RATIO   = 0.40
THRESHOLD    = 15.0   # max ANN output confidence to accept (for imposter check)

# ------------------------------------------------------------------ #
#  Load saved models
# ------------------------------------------------------------------ #
def load_models():
    with open('label_names.pkl', 'rb') as f:
        label_names = pickle.load(f)

    pca = PCA()
    pca.load('pca_model.pkl')

    ann = ANN(layer_sizes=[])
    ann.load('ann_model.pkl')

    return pca, ann, label_names


# ------------------------------------------------------------------ #
#  Recognise a single image
# ------------------------------------------------------------------ #
def recognise_image(img_path, pca, ann, label_names, threshold=THRESHOLD):
    """
    Returns (predicted_name, confidence_percent, is_imposter)
    """
    # Step 1 & 2: load + mean-zero is done inside pca.transform
    face_vec = load_single_image(img_path, IMG_SIZE)        # (mn, 1)
    signature = pca.transform(face_vec)                     # (k, 1)

    proba = ann.predict_proba(signature)                    # (n_classes, 1)
    confidence = float(np.max(proba)) * 100
    pred_idx   = int(np.argmax(proba))

    # Imposter detection — if max confidence below threshold, reject
    is_imposter = confidence < threshold
    name = label_names[pred_idx] if not is_imposter else "Unknown (Imposter)"

    return name, confidence, is_imposter


# ------------------------------------------------------------------ #
#  Evaluate full test split
# ------------------------------------------------------------------ #
def evaluate_test_set(pca, ann, label_names):
    face_db, labels, _ = load_face_dataset(DATASET_PATH, IMG_SIZE)
    _, _, face_test, labels_test = stratified_split(face_db, labels, TEST_RATIO)

    sigs_test = pca.transform(face_test)
    preds     = ann.predict(sigs_test)

    correct = sum(p == t for p, t in zip(preds, labels_test))
    acc     = correct / len(labels_test) * 100

    print(f"\nTest set accuracy: {acc:.2f}%  ({correct}/{len(labels_test)})\n")

    # Per-class breakdown
    from collections import defaultdict
    per_class = defaultdict(lambda: [0, 0])   # {label: [correct, total]}
    for p, t in zip(preds, labels_test):
        per_class[t][1] += 1
        if p == t:
            per_class[t][0] += 1

    print(f"{'Person':<25} {'Correct':>7} {'Total':>7} {'Acc %':>8}")
    print("-" * 52)
    for lbl in sorted(per_class):
        c, tot = per_class[lbl]
        print(f"{label_names[lbl]:<25} {c:>7} {tot:>7} {c/tot*100:>7.1f}%")


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    pca, ann, label_names = load_models()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}")
            sys.exit(1)
        name, conf, is_imp = recognise_image(img_path, pca, ann, label_names)
        print(f"\nImage     : {img_path}")
        print(f"Prediction: {name}")
        print(f"Confidence: {conf:.2f}%")
        print(f"Imposter  : {'YES' if is_imp else 'NO'}")
    else:
        evaluate_test_set(pca, ann, label_names)
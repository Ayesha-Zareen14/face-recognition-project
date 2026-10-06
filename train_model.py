"""
train_model.py
--------------
1. Load face dataset
2. Train/test split (60% train, 40% test)
3. Apply PCA
4. Train ANN with backpropagation
5. Save PCA + ANN models and label names
"""

import os
import pickle
import numpy as np

from utils import load_face_dataset, stratified_split
from pca import PCA
from ann import ANN

# ------------------------------------------------------------------ #
#  Configuration — adjust these as needed
# ------------------------------------------------------------------ #
DATASET_PATH  = 'dataset/faces'   # folder with one sub-folder per person
IMG_SIZE      = (50, 50)           # resize every face to this (width, height)
K             = 100               # number of principal components
HIDDEN_UNITS  = 256               # neurons in hidden layer
LEARNING_RATE = 0.05
EPOCHS        = 5000
TEST_RATIO    = 0.40               # 40% test, 60% train


# ------------------------------------------------------------------ #
#  Load dataset
# ------------------------------------------------------------------ #
print("=" * 55)
print("  FACE RECOGNITION — TRAINING")
print("=" * 55)

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATASET_PATH}'.\n"
        "Make sure the 'faces' folder is inside 'dataset/'.")

face_db, labels, label_names = load_face_dataset(DATASET_PATH, IMG_SIZE)

# Save label names for use during inference
with open('label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)
print(f"Label names saved -> label_names.pkl  ({len(label_names)} classes)")

# ------------------------------------------------------------------ #
#  Train / test split
# ------------------------------------------------------------------ #
face_train, labels_train, face_test, labels_test = stratified_split(
    face_db, labels, test_ratio=TEST_RATIO)

# ------------------------------------------------------------------ #
#  PCA
# ------------------------------------------------------------------ #
print(f"\nApplying PCA (k={K}) …")
pca = PCA(n_components=K)
sigs_train = pca.fit_transform(face_train)   # (k, n_train)
sigs_test  = pca.transform(face_test)        # (k, n_test)
pca.save('pca_model.pkl')

# ------------------------------------------------------------------ #
#  ANN training
# ------------------------------------------------------------------ #
n_classes   = len(label_names)
layer_sizes = [K, HIDDEN_UNITS, n_classes]

print(f"\nTraining ANN {layer_sizes}  lr={LEARNING_RATE}  epochs={EPOCHS} …")
ann = ANN(layer_sizes=layer_sizes,
          learning_rate=LEARNING_RATE,
          epochs=EPOCHS)
losses = ann.fit(sigs_train, labels_train)
ann.save('ann_model.pkl')

# ------------------------------------------------------------------ #
#  Quick evaluation on test set
# ------------------------------------------------------------------ #
preds   = ann.predict(sigs_test)
correct = sum(p == t for p, t in zip(preds, labels_test))
acc     = correct / len(labels_test) * 100

print("\n" + "=" * 55)
print(f"  Test accuracy (k={K}): {acc:.2f}%  ({correct}/{len(labels_test)})")
print("=" * 55)
print("\nDone! Run evaluate.py to see accuracy vs k plot.")
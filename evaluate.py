"""
evaluate.py
-----------
(a) Vary k and plot accuracy vs k
(b) Add imposters (unknown faces) and test rejection
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works without a display
import matplotlib.pyplot as plt

from utils import load_face_dataset, stratified_split
from pca import PCA
from ann import ANN

# ------------------------------------------------------------------ #
#  Config
# ------------------------------------------------------------------ #
DATASET_PATH  = 'dataset/faces'
IMG_SIZE      = (50, 50)
TEST_RATIO    = 0.40
K_VALUES      = [5, 10, 20, 30, 50, 75, 100, 150]
HIDDEN_UNITS  = 128
LEARNING_RATE = 0.01
EPOCHS        = 1000           # fewer epochs for the sweep; increase if needed
CONF_THRESHOLD = 40.0          # % — below this → imposter


# ------------------------------------------------------------------ #
#  Load data once
# ------------------------------------------------------------------ #
print("Loading dataset …")
face_db, labels, label_names = load_face_dataset(DATASET_PATH, IMG_SIZE)
n_classes = len(label_names)
mn, p     = face_db.shape

face_train, labels_train, face_test, labels_test = stratified_split(
    face_db, labels, TEST_RATIO)


# ------------------------------------------------------------------ #
#  (a) Accuracy vs k sweep
# ------------------------------------------------------------------ #
print("\n--- Part (a): Accuracy vs k ---")

results = []   # list of (k, accuracy)
for k in K_VALUES:
    k = min(k, face_train.shape[1] - 1)   # can't exceed training samples

    pca = PCA(n_components=k)
    sigs_train = pca.fit_transform(face_train)
    sigs_test  = pca.transform(face_test)

    ann = ANN(layer_sizes=[k, HIDDEN_UNITS, n_classes],
              learning_rate=LEARNING_RATE,
              epochs=EPOCHS)
    ann.fit(sigs_train, labels_train)

    preds   = ann.predict(sigs_test)
    correct = sum(p_ == t for p_, t in zip(preds, labels_test))
    acc     = correct / len(labels_test) * 100

    results.append((k, acc))
    print(f"  k = {k:>4d}  →  accuracy = {acc:.2f}%")

ks, accs = zip(*results)

plt.figure(figsize=(10, 6))
plt.plot(ks, accs, 'o-', color='steelblue', linewidth=2.5, markersize=8)
plt.xlabel('Number of Principal Components (k)', fontsize=13)
plt.ylabel('Recognition Accuracy (%)', fontsize=13)
plt.title('Face Recognition Accuracy vs Number of Eigenfaces (k)', fontsize=14)
plt.xticks(ks)
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('accuracy_vs_k.png', dpi=150)
print("\nPlot saved → accuracy_vs_k.png")


# ------------------------------------------------------------------ #
#  (b) Imposter detection
# ------------------------------------------------------------------ #
print("\n--- Part (b): Imposter detection ---")

# Use the best k from the sweep
best_k, best_acc = max(results, key=lambda x: x[1])
print(f"Using best k={best_k} (accuracy={best_acc:.2f}%)")

pca = PCA(n_components=best_k)
sigs_train = pca.fit_transform(face_train)
sigs_test  = pca.transform(face_test)

ann = ANN(layer_sizes=[best_k, HIDDEN_UNITS, n_classes],
          learning_rate=LEARNING_RATE,
          epochs=EPOCHS)
ann.fit(sigs_train, labels_train)

# --- Genuine test set ---
genuine_preds = ann.predict(sigs_test)
genuine_proba = ann.predict_proba(sigs_test)
genuine_conf  = np.max(genuine_proba, axis=0) * 100

genuine_correct = sum(p_ == t for p_, t in zip(genuine_preds, labels_test))
genuine_acc     = genuine_correct / len(labels_test) * 100

# --- Imposters: random noise images (simulate unknown persons) ---
n_imposters    = 20
mn_dim         = face_db.shape[0]
imposter_faces = np.random.randint(0, 256, (mn_dim, n_imposters)).astype(np.float64)
imposter_sigs  = pca.transform(imposter_faces)
imposter_proba = ann.predict_proba(imposter_sigs)
imposter_conf  = np.max(imposter_proba, axis=0) * 100

# Classify as imposter if max confidence < threshold
correctly_rejected = int(np.sum(imposter_conf < CONF_THRESHOLD))
rejection_rate     = correctly_rejected / n_imposters * 100

print(f"\n  Genuine accuracy     : {genuine_acc:.2f}%")
print(f"  Imposter rejection   : {correctly_rejected}/{n_imposters} "
      f"({rejection_rate:.1f}%) at threshold={CONF_THRESHOLD}%")

# Confidence distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(genuine_conf, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(CONF_THRESHOLD, color='red', linestyle='--', linewidth=1.8, label=f'Threshold={CONF_THRESHOLD}%')
axes[0].set_title('Genuine Faces — Confidence Distribution', fontsize=12)
axes[0].set_xlabel('Confidence (%)')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].hist(imposter_conf, bins=15, color='tomato', edgecolor='white', alpha=0.85)
axes[1].axvline(CONF_THRESHOLD, color='blue', linestyle='--', linewidth=1.8, label=f'Threshold={CONF_THRESHOLD}%')
axes[1].set_title('Imposter Faces — Confidence Distribution', fontsize=12)
axes[1].set_xlabel('Confidence (%)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.suptitle('Imposter vs Genuine Confidence Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('imposter_analysis.png', dpi=150)
print("  Plot saved → imposter_analysis.png")

print("\nEvaluation complete.")
# coding: utf-8
"""
iris_classify.py
----------------
Apply PCA + ANN to the Iris dataset.
Iris has 150 samples, 4 features, 3 classes.

Usage
-----
  python iris_classify.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pca import PCA
from ann import ANN

# ------------------------------------------------------------------ #
#  Config
# ------------------------------------------------------------------ #
IRIS_PATH     = os.path.join('dataset', 'Iris', 'iris.data')
TEST_RATIO    = 0.40
HIDDEN_UNITS  = 16
LEARNING_RATE = 0.1
EPOCHS        = 3000
K_VALUES      = [1, 2, 3, 4]   # Iris only has 4 features max


# ------------------------------------------------------------------ #
#  Load Iris dataset
# ------------------------------------------------------------------ #
def load_iris(path):
    """
    Returns
    -------
    X          : np.ndarray (4, 150)  — features
    labels     : list[int]            — 0/1/2 class index
    label_names: list[str]            — class name per index
    """
    data, labels, label_names, label_map = [], [], [], {}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            features = list(map(float, parts[:4]))
            name = parts[4]
            if name not in label_map:
                label_map[name] = len(label_map)
                label_names.append(name)
            data.append(features)
            labels.append(label_map[name])

    X = np.array(data).T          # (4, 150)
    print(f"Loaded Iris: {X.shape[1]} samples | "
          f"{X.shape[0]} features | {len(label_names)} classes")
    print(f"  Classes: {label_names}")
    return X, labels, label_names


# ------------------------------------------------------------------ #
#  Stratified split
# ------------------------------------------------------------------ #
def split(X, labels, test_ratio=0.40, seed=42):
    from collections import defaultdict
    np.random.seed(seed)
    class_idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_idx[lbl].append(i)

    train_idx, test_idx = [], []
    for lbl, idxs in class_idx.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_ratio))
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())

    X_train = X[:, train_idx]
    X_test  = X[:, test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test  = [labels[i] for i in test_idx]
    print(f"Split -> train: {len(y_train)} | test: {len(y_test)}")
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
print("=" * 50)
print("  IRIS CLASSIFICATION - PCA + ANN")
print("=" * 50)

X, labels, label_names = load_iris(IRIS_PATH)
n_classes = len(label_names)

X_train, y_train, X_test, y_test = split(X, labels, TEST_RATIO)

# ------------------------------------------------------------------ #
#  Accuracy vs k
# ------------------------------------------------------------------ #
print("\n--- Accuracy vs k ---")
results = []
for k in K_VALUES:
    k = min(k, X_train.shape[1])
    pca = PCA(n_components=k)
    sigs_train = pca.fit_transform(X_train)
    sigs_test  = pca.transform(X_test)

    ann = ANN(layer_sizes=[k, HIDDEN_UNITS, n_classes],
              learning_rate=LEARNING_RATE,
              epochs=EPOCHS)
    ann.fit(sigs_train, y_train)

    preds   = ann.predict(sigs_test)
    correct = sum(p == t for p, t in zip(preds, y_test))
    acc     = correct / len(y_test) * 100
    results.append((k, acc))
    print(f"  k={k}  ->  accuracy={acc:.2f}%")

# ------------------------------------------------------------------ #
#  Without PCA (use all 4 features directly)
# ------------------------------------------------------------------ #
print("\n--- Without PCA (all 4 features) ---")
ann_full = ANN(layer_sizes=[4, HIDDEN_UNITS, n_classes],
               learning_rate=LEARNING_RATE,
               epochs=EPOCHS)
ann_full.fit(X_train, y_train)
preds_full = ann_full.predict(X_test)
acc_full   = sum(p == t for p, t in zip(preds_full, y_test)) / len(y_test) * 100
print(f"  Accuracy (no PCA): {acc_full:.2f}%")

# ------------------------------------------------------------------ #
#  Plot accuracy vs k
# ------------------------------------------------------------------ #
ks, accs = zip(*results)

plt.figure(figsize=(8, 5))
plt.plot(ks, accs, 'o-', color='steelblue', linewidth=2.5, markersize=9, label='With PCA')
plt.axhline(acc_full, color='tomato', linestyle='--', linewidth=1.8, label=f'No PCA ({acc_full:.1f}%)')
plt.xlabel('Number of Principal Components (k)', fontsize=13)
plt.ylabel('Classification Accuracy (%)', fontsize=13)
plt.title('Iris Classification Accuracy vs k (PCA + ANN)', fontsize=14)
plt.xticks(ks)
plt.ylim(0, 105)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('iris_accuracy_vs_k.png', dpi=150)
print("\nPlot saved -> iris_accuracy_vs_k.png")

# ------------------------------------------------------------------ #
#  Best result summary
# ------------------------------------------------------------------ #
best_k, best_acc = max(results, key=lambda x: x[1])
print("\n" + "=" * 50)
print(f"  Best Iris accuracy: {best_acc:.2f}%  (k={best_k})")
print(f"  Without PCA      : {acc_full:.2f}%")
print("=" * 50)
print("\nIris classification complete.")

# ------------------------------------------------------------------ #
#  Save best model for web app
# ------------------------------------------------------------------ #
print("\nSaving best Iris model for web app...")
best_k, _ = max(results, key=lambda x: x[1])

pca_best = PCA(n_components=best_k)
sigs_train_best = pca_best.fit_transform(X_train)
ann_best = ANN(layer_sizes=[best_k, HIDDEN_UNITS, n_classes],
               learning_rate=LEARNING_RATE, epochs=EPOCHS)
ann_best.fit(sigs_train_best, y_train)

import pickle
with open('iris_pca_model.pkl', 'wb') as f:
    pickle.dump({'mean': pca_best.mean, 'eigenfaces': pca_best.eigenfaces,
                 'n_components': pca_best.n_components,
                 'eigenvalues': pca_best.eigenvalues}, f)
ann_best.save('iris_ann_model.pkl')
with open('iris_label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)
print("Iris models saved -> iris_pca_model.pkl, iris_ann_model.pkl")
# Save best model for web app
import pickle
best_k, _ = max(results, key=lambda x: x[1])
pca_best = PCA(n_components=best_k)
sigs_train_best = pca_best.fit_transform(X_train)
ann_best = ANN(layer_sizes=[best_k, HIDDEN_UNITS, n_classes],
               learning_rate=LEARNING_RATE, epochs=EPOCHS)
ann_best.fit(sigs_train_best, y_train)
with open('iris_pca_model.pkl', 'wb') as f:
    pickle.dump({'mean': pca_best.mean, 'eigenfaces': pca_best.eigenfaces,
                 'n_components': pca_best.n_components}, f)
ann_best.save('iris_ann_model.pkl')
with open('iris_label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)
print("Iris models saved!")
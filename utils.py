import os
import cv2
import numpy as np


# ------------------------------------------------------------------ #
#  Face dataset loader
# ------------------------------------------------------------------ #
def load_face_dataset(dataset_path, img_size=(50, 50)):
    """
    Load face images from a folder structured as:
        dataset_path/
            person_A/
                img1.jpg
                img2.jpg
            person_B/
                ...

    Returns
    -------
    face_db    : np.ndarray  shape (mn, p)  — each column is a flattened face
    labels     : list[int]                  — class index per image
    label_names: list[str]                  — class name per index
    """
    face_db = []
    labels = []
    label_names = []

    for label_idx, person_name in enumerate(sorted(os.listdir(dataset_path))):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        img_files = [
            f for f in sorted(os.listdir(person_path))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm', '.bmp'))
        ]

        if not img_files:
            continue

        label_names.append(person_name)

        for img_file in img_files:
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Could not read {img_path} — skipping")
                continue
            img = cv2.resize(img, img_size)
            face_db.append(img.flatten().astype(np.float64))
            labels.append(label_idx)

    if not face_db:
        raise ValueError(f"No images found in {dataset_path}. "
                         "Check the folder structure.")

    face_db = np.array(face_db).T          # (mn, p)
    print(f"Loaded {face_db.shape[1]} images | "
          f"{len(label_names)} persons | "
          f"image size {img_size}")
    return face_db, labels, label_names


# ------------------------------------------------------------------ #
#  Single image → column vector
# ------------------------------------------------------------------ #
def load_single_image(img_path, img_size=(50, 50)):
    """
    Load one image and return it as a column vector (mn, 1).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, img_size)
    return img.flatten().astype(np.float64).reshape(-1, 1)


# ------------------------------------------------------------------ #
#  Train / test split (stratified — keeps class balance)
# ------------------------------------------------------------------ #
def stratified_split(face_db, labels, test_ratio=0.4, seed=42):
    """
    Returns
    -------
    face_train, labels_train, face_test, labels_test
    """
    np.random.seed(seed)
    from collections import defaultdict

    class_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)

    train_idx, test_idx = [], []
    for lbl, idxs in class_indices.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_ratio))
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())

    face_train = face_db[:, train_idx]
    face_test  = face_db[:, test_idx]
    labels_train = [labels[i] for i in train_idx]
    labels_test  = [labels[i] for i in test_idx]

    print(f"Split -> train: {len(labels_train)} | test: {len(labels_test)}")
    return face_train, labels_train, face_test, labels_test

import numpy as np
import pickle


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.eigenfaces = None
        self.eigenvalues = None

    def fit(self, face_db):
        """
        face_db: numpy array of shape (mn, p)
                 mn = image pixels (rows*cols), p = number of images
        """
        mn, p = face_db.shape

        # Step 2: Calculate mean face
        self.mean = np.mean(face_db, axis=1, keepdims=True)  # (mn, 1)

        # Step 3: Mean-zero — subtract mean from every face
        phi = face_db - self.mean  # (mn, p)

        # Step 4: Surrogate covariance (p×p) — Turk & Pentland trick
        C = (phi.T @ phi) / p  # (p, p)

        # Step 5: Eigenvalue / eigenvector decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)  # (p,), (p, p)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # (p, p)

        self.eigenvalues = eigenvalues

        # Step 6: Select k best directions
        k = self.n_components if self.n_components else p
        k = min(k, p)
        self.n_components = k

        feature_vectors = eigenvectors[:, :k]  # (p, k)

        # Step 7: Generate eigenfaces — project mean-zero faces onto feature vectors
        # eigenfaces shape: (k, mn)
        self.eigenfaces = feature_vectors.T @ phi.T  # (k, p) @ (p, mn) = (k, mn)

        # Normalise each eigenface
        norms = np.linalg.norm(self.eigenfaces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.eigenfaces /= norms

        return self

    def transform(self, face_db):
        """
        Project faces into eigenface space.
        face_db : (mn, p)  — can be training OR test images
        returns  : (k, p)  — signatures / projections
        """
        phi = face_db - self.mean           # (mn, p)
        return self.eigenfaces @ phi        # (k, mn) @ (mn, p) = (k, p)

    def fit_transform(self, face_db):
        self.fit(face_db)
        return self.transform(face_db)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'eigenfaces': self.eigenfaces,
                'eigenvalues': self.eigenvalues,
                'n_components': self.n_components
            }, f)
        print(f"PCA model saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.mean = data['mean']
        self.eigenfaces = data['eigenfaces']
        self.eigenvalues = data.get('eigenvalues', None)
        self.n_components = data['n_components']
        return self

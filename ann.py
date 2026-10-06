import numpy as np
import pickle


class ANN:
    """
    Fully-connected feedforward ANN trained with backpropagation.
    Hidden layers use Sigmoid activation; output layer uses Softmax.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000):
        """
        layer_sizes : list e.g. [k, 128, n_classes]
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight initialisation
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # Xavier initialisation
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_out, fan_in))
            b = np.zeros((fan_out, 1))
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------ #
    #  Activations
    # ------------------------------------------------------------------ #
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1.0 - a)

    @staticmethod
    def softmax(z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def _forward(self, X):
        """
        X : (features, samples)
        Returns list of activations for every layer including input.
        """
        activations = [X]
        A = X
        for i in range(len(self.weights) - 1):
            Z = self.weights[i] @ A + self.biases[i]
            A = self.sigmoid(Z)
            activations.append(A)
        # Output layer — Softmax
        Z = self.weights[-1] @ A + self.biases[-1]
        A = self.softmax(Z)
        activations.append(A)
        return activations

    # ------------------------------------------------------------------ #
    #  Backward pass
    # ------------------------------------------------------------------ #
    def _backward(self, activations, Y):
        """
        Y : one-hot (n_classes, samples)
        Updates weights and biases in-place.
        """
        m = Y.shape[1]
        n_layers = len(self.weights)
        deltas = [None] * n_layers

        # Output layer delta (cross-entropy + softmax combined)
        deltas[-1] = activations[-1] - Y  # (n_classes, samples)

        # Hidden layer deltas — propagate backwards
        for i in range(n_layers - 2, -1, -1):
            d = (self.weights[i + 1].T @ deltas[i + 1]) * self.sigmoid_deriv(activations[i + 1])
            deltas[i] = d

        # Gradient descent update
        for i in range(n_layers):
            dW = (deltas[i] @ activations[i].T) / m
            db = np.mean(deltas[i], axis=1, keepdims=True)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i]  -= self.learning_rate * db

    # ------------------------------------------------------------------ #
    #  One-hot encoding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _one_hot(labels, n_classes):
        m = len(labels)
        Y = np.zeros((n_classes, m))
        for j, lbl in enumerate(labels):
            Y[lbl, j] = 1.0
        return Y

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def fit(self, X, labels):
        """
        X      : (features, samples)
        labels : list of int class indices
        """
        n_classes = self.layer_sizes[-1]
        Y = self._one_hot(labels, n_classes)
        losses = []

        for epoch in range(self.epochs):
            activations = self._forward(X)
            self._backward(activations, Y)

            if epoch % 100 == 0:
                out = activations[-1]
                loss = -np.mean(np.sum(Y * np.log(out + 1e-8), axis=0))
                losses.append(loss)
                print(f"  Epoch {epoch:>5d} | Loss: {loss:.4f}")

        return losses

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Returns class index for each sample. X: (features, samples)"""
        activations = self._forward(X)
        return np.argmax(activations[-1], axis=0).tolist()

    def predict_proba(self, X):
        """Returns softmax probability matrix (n_classes, samples)."""
        return self._forward(X)[-1]

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'layer_sizes': self.layer_sizes,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs
            }, f)
        print(f"ANN model saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.biases = data['biases']
        self.layer_sizes = data['layer_sizes']
        self.learning_rate = data['learning_rate']
        self.epochs = data['epochs']
        return self

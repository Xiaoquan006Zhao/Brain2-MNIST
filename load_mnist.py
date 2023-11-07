from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_mnist():
    if not os.path.exists('mnist.pkl'):
        mnist = fetch_openml("mnist_784", version=1)
        with open("mnist.pkl", "wb") as f:
            pickle.dump(mnist, f)
    else:
        with open("mnist.pkl", "rb") as f:
            mnist = pickle.load(f)

    X, y = mnist["data"], mnist["target"]

    X_2D = np.reshape(X.values, (-1, 28, 28))
    y = y.astype(int)

    y_range = set(y)
    y_separated = []
    X_separated = []

    for y_index in y_range:
        indices = (y == y_index)
        y_separated.append(indices)
        X_separated.append(X_2D[indices])
    return (X_separated, y_separated)

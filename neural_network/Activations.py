import numpy as np


class Tanh:

    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward(self, dY):
        dX = dY * (1 - np.tanh(self.X) ** 2)
        return dX


class Sigmoid:

    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X

        return 1 / (1 + np.exp(-X))

    def backward(self, dY):
        s = 1 / (1 + np.exp(-self.X))

        dX = dY * s * (1 - s)

        return dX


class ReLU:

    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X

        return np.maximum(0, X)

    def backward(self, dY):
        dX = dY * (self.X > 0).astype(float)

        return dX
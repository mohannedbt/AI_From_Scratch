import numpy as np


class MSELoss:

    def __init__(self):
        self.Y_pred = None
        self.Y_true = None

    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true

        return 0.5 * np.mean((Y_pred - Y_true) ** 2)

    def backward(self):
        N = self.Y_true.size

        return (self.Y_pred - self.Y_true) / N


class BCELoss:

    def __init__(self):
        self.Y_pred = None
        self.Y_true = None

    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true

        eps = 1e-12
        Y_pred = np.clip(Y_pred, eps, 1 - eps)

        L = (
            Y_true * np.log(Y_pred)
            + (1 - Y_true) * np.log(1 - Y_pred)
        )

        return -np.mean(L)

    def backward(self):
        Y_pred = np.clip(self.Y_pred, 1e-12, 1 - 1e-12)
        Y_true = self.Y_true

        N = Y_true.size

        return -(
            (Y_true / Y_pred)
            - ((1 - Y_true) / (1 - Y_pred))
        ) / N

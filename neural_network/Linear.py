import numpy as np


class Linear:

    def __init__(self, input_size, output_size):
        # W: (input_size, output_size)
        self.W = np.random.randn(input_size, output_size) * 0.01

        # b: (1, output_size)
        self.b = np.zeros((1, output_size))

        self.X = None

        self.dW = None
        self.db = None

    def forward(self, X):
        """
        X: (batch_size, input_size)

        Returns:
            Y: (batch_size, output_size)
        """

        self.X = X

        return X @ self.W + self.b

    def backward(self, dY):
        """
        dY: (batch_size, output_size)

        Returns:
            dX: (batch_size, input_size)
        """

        # (input_size, batch_size)
        #      @
        # (batch_size, output_size)
        #
        # -> (input_size, output_size)
        self.dW = self.X.T @ dY

        # Sum gradients over the batch
        self.db = np.sum(dY, axis=0, keepdims=True)

        # (batch_size, output_size)
        #      @
        # (output_size, input_size)
        #
        # -> (batch_size, input_size)
        dX = dY @ self.W.T

        return dX

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]
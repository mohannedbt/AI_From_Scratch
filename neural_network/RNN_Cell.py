import numpy as np


class RNN_Cell:

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros(hidden_size)

        # Gradients
        self.dWxh = None
        self.dWhh = None
        self.dbh = None

        # Cache for backward pass
        self.cache = {
            "x_t": None,
            "h_prev": None,
            "a_t": None,
            "h_t": None
        }

    def forward(self, x_t, h_prev):
        """
        One RNN timestep.

        x_t:
            shape = (input_size,)

        h_prev:
            shape = (hidden_size,)

        Returns:
            h_t:
                shape = (hidden_size,)
        """

        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h

        h_t = np.tanh(a_t)

        # Store values needed for backward
        self.cache["x_t"] = x_t
        self.cache["h_prev"] = h_prev
        self.cache["a_t"] = a_t
        self.cache["h_t"] = h_t

        return h_t

    def backward(self, dh_t):
        """
        Backward pass for one timestep.

        dh_t:
            gradient of the loss with respect to h_t

        Returns:
            dx_t:
                gradient with respect to x_t

            dh_prev:
                gradient with respect to h_prev
        """

        a_t = self.cache["a_t"]
        x_t = self.cache["x_t"]
        h_prev = self.cache["h_prev"]

        # tanh derivative
        da_t = dh_t * (1 - np.tanh(a_t) ** 2)

        # Parameter gradients
        self.dWxh = da_t @ x_t.T
        self.dWhh = da_t @ h_prev.T
        self.dbh = da_t

        # Gradients propagated backward
        dx_t = self.W_xh.T @ da_t
        dh_prev = self.W_hh.T @ da_t

        return dx_t, dh_prev

    def parameters(self):
        return [
            (self.W_xh, self.dWxh),
            (self.W_hh, self.dWhh),
            (self.b_h, self.dbh)
        ]
import numpy as np

from neural_network.RNN_Cell import RNN_Cell


class RNN_sequence:

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # One cell reused for every timestep
        self.cell = RNN_Cell(input_size, hidden_size)

        # Sequence-level cache
        self.cache = []

    def forward(self, X, h0=None):
        """
        Run the RNN through an entire sequence.

        X:
            shape = (sequence_length, input_size)

        h0:
            shape = (hidden_size,)
            Initial hidden state.
            If None, initialized to zeros.

        Returns:
            H:
                shape = (sequence_length, hidden_size)
        """

        sequence_length = X.shape[0]

        # Initial hidden state
        if h0 is None:
            h_prev = np.zeros(self.hidden_size)
        else:
            h_prev = h0

        # Clear previous sequence cache
        self.cache = []

        # Store all hidden states
        H = []

        for t in range(sequence_length):

            # Current input
            x_t = X[t]

            # One RNN step
            h_t = self.cell.forward(x_t, h_prev)

            # Save a copy of this timestep's cache
            self.cache.append({
                "x_t": self.cell.cache["x_t"].copy(),
                "h_prev": self.cell.cache["h_prev"].copy(),
                "a_t": self.cell.cache["a_t"].copy(),
                "h_t": self.cell.cache["h_t"].copy()
            })

            # Store hidden state
            H.append(h_t)

            # Current hidden state becomes
            # previous hidden state for next timestep
            h_prev = h_t

        return np.array(H)

    def parameters(self):
        return self.cell.parameters()

    def backward(self, dH):

        sequence_length = len(self.cache)

        dX = np.zeros(
            (sequence_length, self.input_size)
        )

        dh_future = np.zeros(self.hidden_size)

        dWxh_total = np.zeros_like(self.cell.W_xh)
        dWhh_total = np.zeros_like(self.cell.W_hh)
        dbh_total = np.zeros_like(self.cell.b_h)

        for t in reversed(range(sequence_length)):

            dh_t = dH[t] + dh_future

            self.cell.cache = self.cache[t]

            dx_t, dh_prev = self.cell.backward(dh_t)

            dX[t] = dx_t

            dh_future = dh_prev

            dWxh_total += self.cell.dWxh
            dWhh_total += self.cell.dWhh
            dbh_total += self.cell.dbh

        self.cell.dWxh = dWxh_total
        self.cell.dWhh = dWhh_total
        self.cell.dbh = dbh_total

        return dX

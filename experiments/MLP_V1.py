from neural_network import Linear
import numpy as np


# Activation functions
def tanh(z):
    return np.tanh(z)


def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


def ReLU(z):
    return np.maximum(0, z)


def ReLU_deriv(z):
    return (z > 0).astype(float)


# Neural network Class
class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        self.input = Linear(input_size, hidden_size)
        self.hidden = Linear(hidden_size, output_size)

        self.z = None

    def forward(self, X, act):
        self.z = self.input.forward(X)

        a = act(self.z)

        return self.hidden.forward(a)

    def backward(self, dY, act_deriv):
        
        da = self.hidden.backward(dY)
        # because act(z) element by element so the chain rule 
        # is applied el by el hence the usage of * instead of @
        
        dz = da * act_deriv(self.z) 

        return self.input.backward(dz)
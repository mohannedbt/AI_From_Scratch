from .Linear import Linear
class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        inter = X

        for layer in self.layers:
            inter = layer.forward(inter)

        return inter

    def backward(self, dY):
        inter = dY

        for layer in reversed(self.layers):
            inter = layer.backward(inter)

        return inter

    def parameters(self):

        L=[]

        for layer in self.layers:
            if(isinstance(layer,Linear)):
                L.extend(layer.parameters()) # not append to avoid nested lists
        return L


        
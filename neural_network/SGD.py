class SGD:
    def __init__(self, rate):
        self.learning_rate = rate

    def step(self, params):
        for p, dp in params:
            p -= self.learning_rate * dp
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self): # to easily use it in iterations
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]

            X_batch = []
            Y_batch = []

            for i in batch_indices:
                X, Y = self.dataset[i]
                X_batch.append(X)
                Y_batch.append(Y)

            yield np.array(X_batch), np.array(Y_batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
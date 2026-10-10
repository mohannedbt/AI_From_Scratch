from .DataLoader import DataLoader
from .Sequential import Sequential
from .SGD import SGD


class Trainer:

    def __init__(
        self,
        model: Sequential,
        loss_function,
        optimizer: SGD,
        dataloader: DataLoader,
        epoches
    ):
        self.model = model
        self.loss = loss_function
        self.optimizer = optimizer
        self.epochs = epoches
        self.losses = []
        self.epoch_losses = []
        self.dataloader = dataloader

    def train(self):

        for i in range(self.epochs):

            epoch_loss = 0.0
            batch_count = 0

            for X, Y in self.dataloader:

                prediction = self.model.forward(X)

                loss = self.loss.forward(prediction, Y)

                self.losses.append(loss)

                epoch_loss += loss
                batch_count += 1

                gradient = self.loss.backward()

                self.model.backward(gradient)

                self.optimizer.step(
                    self.model.parameters()
                )

            average_loss = epoch_loss / batch_count

            self.epoch_losses.append(average_loss)

            print(
                f"Epoch {i + 1}/{self.epochs} "
                f"- Loss: {average_loss}"
            )
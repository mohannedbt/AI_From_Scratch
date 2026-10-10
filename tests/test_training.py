import numpy as np

from neural_network import DataLoader, Dataset, Linear, MSELoss, SGD, Sequential, Tanh, Trainer


# --------------------------------------------------
# 1. Training data
# --------------------------------------------------

X = np.array([[1.0]])
Y = np.array([[2.0]])


# --------------------------------------------------
# 2. Model
# --------------------------------------------------

model = Sequential([
    Linear(1, 4),
    Tanh(),
    Linear(4, 1)
])


# --------------------------------------------------
# 3. Loss
# --------------------------------------------------

loss = MSELoss()


# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

optimizer = SGD(rate=0.01)


# --------------------------------------------------
# 5. Trainer
# --------------------------------------------------

dataset = Dataset(X, Y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

trainer = Trainer(
    model=model,
    loss_function=loss,
    optimizer=optimizer,
    dataloader=dataloader,
    epoches=5000
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# 7. Check loss
# --------------------------------------------------

print("Initial loss:", trainer.losses[0])
print("Final loss:", trainer.losses[-1])


# --------------------------------------------------
# 8. Test prediction
# --------------------------------------------------

prediction = model.forward(X)

print("Input:", X)
print("Expected:", Y)
print("Prediction:", prediction)


# --------------------------------------------------
# 9. Simple verification
# --------------------------------------------------

if trainer.losses[-1] < trainer.losses[0]:
    print("PASS: loss decreased.")
else:
    print("FAIL: loss did not decrease.")
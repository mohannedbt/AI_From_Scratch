import numpy as np

from neural_network import (
    DataLoader,
    Dataset,
    Linear,
    MSELoss,
    SGD,
    Sequential,
    Tanh,
    Trainer,
)


# --------------------------------------------------
# 1. Create training data
# --------------------------------------------------

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0]
])

# Target: Y = 2X
Y = np.array([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0],
    [12.0],
    [14.0],
    [16.0]
])


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

dataset = Dataset(X, Y)


# --------------------------------------------------
# 3. DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True
)


# --------------------------------------------------
# 4. Model
# --------------------------------------------------

model = Sequential([
    Linear(1, 4),
    Tanh(),
    Linear(4, 1)
])


# --------------------------------------------------
# 5. Loss
# --------------------------------------------------

loss = MSELoss()


# --------------------------------------------------
# 6. Optimizer
# --------------------------------------------------

optimizer = SGD(
    rate=0.01
)


# --------------------------------------------------
# 7. Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    loss_function=loss,
    optimizer=optimizer,
    dataloader=dataloader,
    epoches=1000
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# 9. Results
# --------------------------------------------------

print("Initial loss:", trainer.losses[0])
print("Final loss:", trainer.losses[-1])


# --------------------------------------------------
# 10. Test predictions
# --------------------------------------------------

X_test = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

prediction = model.forward(X_test)

print("\nInput:")
print(X_test)

print("\nPrediction:")
print(prediction)

print("\nExpected:")
print(2 * X_test)


# --------------------------------------------------
# 11. Check whether training worked
# --------------------------------------------------

if trainer.losses[-1] < trainer.losses[0]:
    print("\nPASS: loss decreased.")
else:
    print("\nFAIL: loss did not decrease.")
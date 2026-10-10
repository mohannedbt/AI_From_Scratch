import numpy as np

from neural_network import DataLoader, Dataset, Linear


# --------------------------------------------------
# 1. Create some fake data
# --------------------------------------------------

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0]
])

Y = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])


print("Original data")
print("X:")
print(X)
print("X shape:", X.shape)

print("\nY:")
print(Y)
print("Y shape:", Y.shape)


# --------------------------------------------------
# 2. Create Dataset
# --------------------------------------------------

dataset = Dataset(X, Y)

print("\nDataset")
print("Number of samples:", len(dataset))

print("First sample:")
print(dataset[0])


# --------------------------------------------------
# 3. Create DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=False
)

print("\nDataLoader")
print("Number of batches:", len(dataloader))


# --------------------------------------------------
# 4. Create Linear layer
# --------------------------------------------------

linear = Linear(
    input_size=2,
    output_size=1
)


# --------------------------------------------------
# 5. Test every batch
# --------------------------------------------------

for batch_number, (X_batch, Y_batch) in enumerate(dataloader):

    print("\n" + "=" * 40)
    print("BATCH", batch_number + 1)
    print("=" * 40)

    print("X_batch:")
    print(X_batch)

    print("X_batch shape:", X_batch.shape)

    print("Y_batch:")
    print(Y_batch)

    print("Y_batch shape:", Y_batch.shape)


    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    Y_pred = linear.forward(X_batch)

    print("\nForward")

    print("Y_pred:")
    print(Y_pred)

    print("Y_pred shape:", Y_pred.shape)


    # --------------------------------------------------
    # Fake gradient from the loss
    # --------------------------------------------------

    dY = np.ones_like(Y_pred)

    print("\ndY shape:", dY.shape)


    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    dX = linear.backward(dY)

    print("\nBackward")

    print("dW:")
    print(linear.dW)

    print("dW shape:", linear.dW.shape)

    print("\ndb:")
    print(linear.db)

    print("db shape:", linear.db.shape)

    print("\ndX:")
    print(dX)

    print("dX shape:", dX.shape)


# --------------------------------------------------
# 6. Final checks
# --------------------------------------------------

print("\n" + "=" * 40)
print("SHAPE CHECKS")
print("=" * 40)

expected_dW_shape = (1, 2)
expected_db_shape = (1, 1)
expected_dX_shape = (2, 2)

if linear.dW.shape == expected_dW_shape:
    print("PASS: dW shape")
else:
    print("FAIL: dW shape")


if linear.db.shape == expected_db_shape:
    print("PASS: db shape")
else:
    print("FAIL: db shape")


if dX.shape == expected_dX_shape:
    print("PASS: dX shape")
else:
    print("FAIL: dX shape")
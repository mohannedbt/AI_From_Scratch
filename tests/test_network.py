import numpy as np

from neural_network import Linear, ReLU, Sequential


# ============================================================
# Create layers
# ============================================================

layer1 = Linear(2, 2)
layer2 = Linear(2, 1)


# ============================================================
# Set known parameters
# ============================================================

layer1.W = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

layer1.b = np.array([
    [1.0],
    [1.0]
])

layer2.W = np.array([
    [2.0, 3.0]
])

layer2.b = np.array([
    [1.0]
])


# ============================================================
# Build model
# ============================================================

model = Sequential([
    layer1,
    ReLU(),
    layer2
])


# ============================================================
# Input
# ============================================================

X = np.array([
    [1.0],
    [2.0]
])


# ============================================================
# Forward test
# ============================================================

calculated_Y = model.forward(X)

expected_Y = np.array([
    [49.0]
])


print("========== FORWARD TEST ==========")

print("Calculated:")
print(calculated_Y)

print("\nExpected:")
print(expected_Y)

print("\nMatch:")
print(np.allclose(calculated_Y, expected_Y))


# ============================================================
# Backward test
# ============================================================

dY = np.array([
    [1.0]
])

calculated_dX = model.backward(dY)

expected_dX = np.array([
    [11.0],
    [16.0]
])


print("\n========== BACKWARD TEST ==========")

print("Calculated dX:")
print(calculated_dX)

print("\nExpected dX:")
print(expected_dX)

print("\nMatch:")
print(np.allclose(calculated_dX, expected_dX))


# ============================================================
# Final result
# ============================================================

forward_ok = np.allclose(calculated_Y, expected_Y)
backward_ok = np.allclose(calculated_dX, expected_dX)

print("\n========== FINAL RESULT ==========")

if forward_ok:
    print("Forward: PASS")
else:
    print("Forward: FAIL")

if backward_ok:
    print("Backward: PASS")
else:
    print("Backward: FAIL")

if forward_ok and backward_ok:
    print("\nAll tests passed.")
else:
    print("\nSome tests failed.")
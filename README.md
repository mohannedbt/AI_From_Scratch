# NumPy Neural Network Learning Project

This repository is a compact, educational neural-network project built with Python and NumPy. It is designed to teach the core ideas behind forward propagation, backpropagation, minibatch training, and recurrent sequence modeling without relying on a heavy framework such as PyTorch or TensorFlow.

The code is intentionally small and readable. It contains reusable building blocks for dense feedforward networks, an RNN cell and sequence model, a small training loop, and a few separate learning exercises. The goal is not to be production-ready, but to make the underlying math easy to inspect and experiment with.

## What is in this project?

This repository contains two main model styles:

- Feedforward networks built from `Linear` layers and activation functions.
- A simple recurrent architecture based on an RNN cell that is unrolled over time.

The core library lives in `neural_network/` and contains the reusable components used for learning and experimentation. The project also includes example scripts under `tests/`, earlier experiments under `experiments/`, and small programming drills under `learning_exercises/`.

## Architecture overview

The library is built around a simple layer-based design.

### 1. Feedforward network architecture

The dense-network path is the main training architecture used by the current implementation:

```text
Dataset
  -> DataLoader (mini-batches)
  -> Trainer
  -> Sequential([...layers...])
      -> Linear(input_size, output_size)
      -> Activation layer (Tanh, Sigmoid, ReLU)
      -> Linear(...)
      -> Loss function (MSELoss / BCELoss)
      -> Backpropagation
      -> SGD optimizer updates
```

A typical model is constructed as a stack of layers:

```python
model = Sequential([
    Linear(1, 4),
    Tanh(),
    Linear(4, 1)
])
```

The forward pass is computed layer by layer:

- `Linear.forward(X)` computes `X @ W + b`
- the activation applies a nonlinear transform such as `tanh`, `sigmoid`, or `ReLU`
- the final output is compared against the target using a loss function

During backpropagation, each layer stores the information it needs to compute gradients, and `Sequential.backward()` applies the chain rule in reverse order. The built-in optimizer (`SGD`) then updates the weights and bias values in place.

### 2. RNN architecture

The repository also includes a sequence model built around an RNN cell:

```text
RNN_sequence
  -> reuses one RNN_Cell across time steps
  -> for each timestep t:
       a_t = W_xh @ x_t + W_hh @ h_{t-1} + b_h
       h_t = tanh(a_t)
  -> stores cache for each timestep
  -> applies backpropagation through time (BPTT)
```

This is implemented in `neural_network/RNN_Cell.py` and `neural_network/RNN_sequence.py`.

The `RNN_Cell` has three learnable parameter matrices:

- `W_xh`: input-to-hidden weights
- `W_hh`: hidden-to-hidden recurrent weights
- `b_h`: bias vector

At each timestep:

```python
a_t = W_xh @ x_t + W_hh @ h_prev + b_h
h_t = tanh(a_t)
```

`RNN_sequence` loops through a whole input sequence, stores the cached values for each timestep, and then backpropagates through the unrolled sequence. This means the model can learn temporal dependencies by carrying hidden state from one step to the next.

## Training flow

The training pipeline is intentionally simple:

1. `Dataset` stores matching input and target pairs.
2. `DataLoader` splits them into minibatches and optionally shuffles samples.
3. `Trainer.train()` runs through every epoch and every batch.
4. The model produces predictions, the loss is computed, gradients are propagated, and `SGD.step()` updates parameters.
5. Average loss per epoch is recorded in `trainer.epoch_losses`.

The main loop is implemented in `neural_network/Trainer.py` and follows the standard pattern:

```python
prediction = model.forward(X)
loss = loss_function.forward(prediction, Y)
gradient = loss_function.backward()
model.backward(gradient)
optimizer.step(model.parameters())
```

## Project layout

```text
.
|-- neural_network/          # Reusable learning components
|   |-- __init__.py          # Exports the public package API
|   |-- Activations.py       # Tanh, Sigmoid, and ReLU layers
|   |-- DataLoader.py        # Mini-batch generation and optional shuffling
|   |-- Dataset.py           # Paired input/target arrays
|   |-- Linear.py            # Dense affine layer with gradients
|   |-- Losses.py            # MSE and binary cross-entropy losses
|   |-- RNN_Cell.py          # One recurrent timestep implementation
|   |-- RNN_sequence.py      # Full sequence forward/backward pass
|   |-- Sequential.py        # Layer composition and backpropagation
|   |-- SGD.py               # Stochastic gradient descent optimizer
|   |-- Trainer.py           # Epoch-based training loop
|-- tests/                   # Demo scripts and checks
|-- experiments/             # Earlier or alternative model implementations
|-- learning_exercises/      # Small standalone coding exercises
|-- README.md                # Project overview and usage notes
`--
```

## Files and responsibilities

- `Linear.py`: dense matrix-based layer with weight and bias parameters.
- `Sequential.py`: composes many layers and runs them in order during forward and reverse order during backward.
- `Trainer.py`: manages epochs, batch iteration, loss tracking, and optimization.
- `RNN_Cell.py`: one recurrent timestep using the hidden state recurrence.
- `RNN_sequence.py`: unrolls the recurrent computation across the sequence.
- `DataLoader.py` and `Dataset.py`: prepare and feed training data.
- `SGD.py`: basic parameter update rule.

## Requirements

- Python 3.10 or newer is recommended.
- NumPy

Install dependencies with:

```powershell
python -m pip install numpy
```

## Running the examples

The scripts in `tests/` are small learning demos and print values and pass/fail style messages. They are useful for checking that the math behaves sensibly, although they are not a full automated test suite.

Run them from the project root:

```powershell
python -m tests.test_network
python -m tests.test_batch
python -m tests.test_training
python -m tests.test_pipeline
```

## Example: feedforward network

```python
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

inputs = np.array([[1.0], [2.0], [3.0], [4.0]])
targets = 2.0 * inputs

dataset = Dataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = Sequential([
    Linear(1, 4),
    Tanh(),
    Linear(4, 1),
])

trainer = Trainer(
    model=model,
    loss_function=MSELoss(),
    optimizer=SGD(rate=0.01),
    dataloader=dataloader,
    epoches=100,
)

trainer.train()
predictions = model.forward(inputs)
print(predictions)
```

## Example: RNN sequence

```python
import numpy as np
from neural_network.RNN_sequence import RNN_sequence

X = np.array([
    [1.0, 0.0],
    [0.5, 0.2],
    [0.2, 0.8],
])

rnn = RNN_sequence(input_size=2, hidden_size=4)
H = rnn.forward(X)
print(H.shape)
```

This creates a sequence of hidden states, one per timestep, and reuses the same recurrent cell over the full input sequence.

## Learning and experiment files

- `experiments/MLP_V1.py` contains an earlier MLP-style implementation. It is kept separate from the current package API.
- `learning_exercises/Counter.py` is a straightforward counter exercise.
- `learning_exercises/Vector.py` demonstrates basic vector operations and algebra.

These are teaching artifacts rather than core training machinery.

## Current limitations

- This is an educational project, not a production deep-learning framework.
- There is no automatic differentiation engine, GPU acceleration, model checkpointing, or serialization.
- The tests are lightweight demos rather than full automated test suites.
- Layer instances store cached forward values, so they are best used in a single sequential pass rather than reused across overlapping computations.
- The code is intentionally simple and focuses on understanding the math over optimizing performance.

## Summary

This project is best understood as a hands-on implementation of the core ideas of machine learning:

- matrix multiplication as a learnable transformation
- activations for nonlinearity
- loss functions for measuring error
- gradient descent for parameter updates
- recurrent hidden state for sequence learning

It is ideal for learning how neural networks work from the ground up, especially in a NumPy-only environment where each step is easy to inspect.
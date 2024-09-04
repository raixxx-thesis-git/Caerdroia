# Nodeleys (in-dev)
![Nodeleys_Logo](nodeleys_logo.jpg)
Nodeleys is a new lightweight deep-learning framework that works on top of Numpy. Nodeleys supports automatic differentiation, static computational graphs, and dynamic computational graphs. At this milestone, Nodeleys supports CUDA with the use of Cupy. However, Cupy may be altered in the future.


# Genesis BackEnd
![Nodeleys_Logo](genesis_logo.jpg)
Genesis plays a crucial role as Nodeleys's backend, having a main focus on computational graphs and gradient flow algorithms which are the key features of a deep learning framework. I would like to say that Genesis is the "minimal" product of Nodeleys. With this backend as the foundation for Nodeleys, the framework currently only works for 2D Tensor data (matrices) and also is still "under development". A layering system, initializers, multi-input or outputs, loss functions, optimizers, etc are YET to be developed in this backend. In the future, Genesis would be replaced with a more robust and advanced backend, Exodus, supporting a higher rank of tensor data, a wider range of customization, rich layer variations, etc.

Exodus will be the "viable" product of Nodeleys.


## Features in this version
1. Node system. This version can create a node that holds a gradient, a tensor, and other states related to forward/backward propagation.
2. Forward propagation. This version supports forward propagation, although for limited operations (Supports basic operators: addition, subtraction, multiplication, and division; linear algebra operators: matrix multiplication, broadcasting (limited to the mentioned operators); and an aggregate function: reduce summation. Does not support trigonometric functions, logarithms, and other advanced functions just yet).
3. Backward propagation. Supports gradient computation for the mentioned operators.
4. Skip connections.
5. Dynamic graphs (still under development).
6. Maximum tensor rank of 2.

## What to be done
1. Dynamic graphs.
2. Support for a wider range of operations (both for forward and backward propagation).
3. Initializers.
4. Weights update mechanism.
5. Layering system.
6. Testing one input one output.
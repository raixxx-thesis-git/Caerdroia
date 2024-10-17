# Nodeleys (in-dev)
<img src="nodeleys_logo.jpg" alt="Nodeleys Logo" style="width:50%;">
Nodeleys is a new lightweight deep-learning framework that works on top of CuPy (CUDA supported NumPy). Nodeleys supports automatic differentiation and dynamic computational graphs. At this milestone, Nodeleys supports CUDA with the use of CuPy. However, CuPy may be altered in the future version with a better backend written directly from C/C++ CUDA. The author of this project is currently still learning how to do so.


# Genesis BackEnd
<img src="genesis_logo.jpg" alt="Genesis Logo" style="width:50%;">
Genesis plays a crucial role as Nodeleys's backend, having a main focus on computational graphs and gradient flow algorithms which are the key features of a deep learning framework. I would like to say that Genesis is the "minimal" product of Nodeleys. With this backend as the foundation for Nodeleys, the framework currently only works on some mathematical expressions listed in the "Supported Operations" section. A layering system, initializers, multi-input or outputs, loss functions, optimizers, etc also has been developed, yet the variations is still so limited. In the future, Genesis might be replaced with a more robust and advanced backend, Exodus, supporting wider range of mathematical expressions; a wider range of customization; and rich layer, loss, and optimizer variations.


## Supported Operations
How to read: if a tensor has $`(A, B)`$, it means it is in $`\mathbb{R}^{A\times B}`$ and if a tensor has $`\emptyset`$, it is a scalar.
1. Add ($`+`$) operations
    1. $`\mathbb{R}^{A\times B} + \mathbb{R}^{A\times B}`$
    2. $`\mathbb{R}^{A\times B} + \mathbb{R}`$ and vice versa
    3. $`\mathbb{R}^{A\times B} + \mathbb{R}^{1\times B}`$ and vice versa
2. Substraction ($`-`$) operations
    1. $`\mathbb{R}^{A\times B} - \mathbb{R}^{A\times B}`$
    2. $`\mathbb{R}^{A\times B} - \mathbb{R}^{1\times B}`$ and vice versa
    3. $`\mathbb{R}^{A\times B} - \mathbb{R}^{A\times 1}`$
3. Multiplication ($`\cdot`$) operations:
    1. $`\mathbb{R}^{A\times B} \cdot \mathbb{R}^{A\times B}`$
    2. $`\mathbb{R}^{A\times B} \cdot \mathbb{R}`$ and vice versa
4. Division ($`/`$) operations:
    1. $`\mathbb{R}^{A\times B} / \mathbb{R}^{A\times B}`$
    2. $`\mathbb{R}^{A\times B} / \mathbb{R}`$ and vice versa
    3. $`\mathbb{R}^{A\times B} / \mathbb{R}^{A\times 1}`$
5. Matrix multiplication ($`@`$) operation: $`\mathbb{R}^{A\times B} @ \mathbb{R}^{B\times C}`$
6. Power (^) operations: $`(\mathbb{R}^{A\times B})^{\mathbb{R}}`$ and vice versa
7. Reduce summation  ($`\text{redsum}`$) operation: 
    1. $`\text{redsum}(\mathbb{R}^{A\times B}, \text{axis}=0)`$
    2. $`\text{redsum}(\mathbb{R}^{A\times B}, \text{axis}=1)`$
8. Flatten ($`\text{flatten}`$) operation: $`\text{flatten}(\mathbb{R}^{N\times C\times H\times W})`$
9. Convolution 2D ($`\circledast`$) operation: $`\mathbb{R}^{N\times C\times H\times W} \circledast \mathbb{R}^{N\times C' \times R \times S}`$
10. Maxpooling 2D ($`\text{maxpool2d}`$) operation: $`\text{maxpool2d}(\mathbb{R}^{N\times C\times H\times W})`$
11. Concatenation ($`\text{concat}`$) operation: $`\text{concat}(\mathbb{R}^{A\times B\times C\times ...}, \mathbb{R}^{A\times B\times C\times ...}, \text{axis}=\text{any})`$
12. ReLU ($`\text{relu}`$) operation: $`\text{relu}(\mathbb{R}^{A\times B\times C\times ...})`$
13. LeakyReLU ($`\text{leakyrelu}`$) operation: $`\text{leakyrelu}(\mathbb{R}^{A\times B\times C\times ...})`$

## Features in this version
1. Dynamic computation graph
2. Automatic differentiation
3. Mathematical expressions (see Supported Operations section)
4. Pre-built core layers: Dense, Add, Concatenate, Conv2D, MaxPool2D, Flatten
5. Pre-built activation layers: ReLU, LeakyReLU, Sigmoid, Softmax
6. Pre-built initializers: RandomNormal, XavierNormal, XavierUniform
7. Pre-built loss: CategoricalCrossEntropy
8. Subclassing API
9. Model construction
10. Model training
11. Multi input and/or output model

## What to be done next in Exodus BackEnd
1. Adding wider range mathematical expressions
2. Adding wider range pre-built core layers such as AvgPool2D, RNN, and LSTM
3. Adding wider range pre-built activation functions
4. Optimizing memory usage
5. Altering CuPy to C/C++ CUDA directly
6. Expanding model.py built-in methods to optimize Nodeleys model
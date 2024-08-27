from __future__ import annotations
from typing import TYPE_CHECKING
from cupy import ndarray

import cupy

if TYPE_CHECKING:
  from mytensor import TensorNode

class BackwardMath():
  def __init__(self): pass

  def get_attributes(self, var: TensorNode):
    child: TensorNode = var.child
    node_type = var.node_type

    if node_type == 'p':
      partner: TensorNode = var.child.parent[1]
    elif node_type == 's':
      partner: TensorNode = var.child.parent[0]

 
    prev_grad: ndarray = cupy.array([[1.]])
    A: ndarray = var.tensor
    B: ndarray|None = None

    if child != None:
      prev_grad = child.grad
    if partner != None:
      B = partner.tensor

    return A, B, prev_grad, node_type

  def grad_for_matmul(self, var: TensorNode) -> ndarray:
    _, B, prev_grad, node_type = self.get_attributes(var)

    if node_type == 'p':
      return prev_grad @ B.T
    elif node_type == 's':
      return B.T @ prev_grad
    
  def grad_for_reduce_sum(self, var: TensorNode) -> ndarray:
    A, _, prev_grad, _ = self.get_attributes(var)

    if ((A.shape[0] != 1 and A.shape[1] == prev_grad.shape[1]) or 
        (A.shape[0] == 1 and A.shape[0] == prev_grad.shape[0])):
      return cupy.ones(shape=A.shape) * prev_grad
    
    print(f'WARNING: Tensor {var.name} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_add(self, var: TensorNode):
    A, B, prev_grad, node_type = self.get_attributes(var)

    # for space of A == space of B == space of C == R^{PxQ} where C=A+B
    if A.shape == B.shape:
      return prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')
    
  def grad_for_sub(self, var: TensorNode):
    A, B, prev_grad, node_type = self.get_attributes(var)

    # for space of A == space of B == space of C == R^{PxQ} where C=A-B
    if A.shape == B.shape and node_type == 'p':
      return prev_grad
    elif A.shape == B.shape and node_type == 's':
      return -1 * prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_truediv(self, var: TensorNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)
    print(var.name)
    if type(prev_grad) == type(None):
      return 1.0 / B
    # for space of A == space of B == space of C == R^{PxQ} where C=A/B 
    if A.shape == B.shape and node_type == 'p':
      return (1 / B) * prev_grad
    elif A.shape == B.shape and node_type == 's':
      print('xxx', B, A)
      print(var.child.parent)
      return -1 * (B / A**2) * prev_grad
    elif B.shape == () and node_type == 'p':
      return (1/B) * prev_grad
    elif B.shape == () and node_type == 's':
      return - (B/(A**2)) * prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_mul(self, var: TensorNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)

    if type(prev_grad) == type(None):
      return B
    elif A.shape == B.shape:
      return B * prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')
# future: skip connections
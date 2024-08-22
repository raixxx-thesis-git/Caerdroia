from __future__ import annotations
from typing import TYPE_CHECKING
from cupy import ndarray

import cupy

if TYPE_CHECKING:
  from mytensor import TensorNode

class BackwardMath():
  def __init__(self): pass

  def compute_grad(self, var: TensorNode) -> ndarray:
    operator_to_method = {
      '@': self.grad_for_matmul,
      '+': self.grad_for_add,
      'redsum': self.grad_for_reduce_sum,
      '/': self.grad_for_truediv
    }
    return operator_to_method[var.child.operation](var)
  
  def dummy(self, x): pass

  def get_attributes(self, var: TensorNode):
    child: TensorNode = var.child
    partner: TensorNode = var.child.parent[1]

    node_type = var.node_type
    prev_grad: ndarray|None = None
    A:ndarray = var.tensor
    B:ndarray|None = None

    if child != None:
      prev_grad = child.grad
    if partner != None:
      B = partner.tensor

    return A, B, prev_grad, node_type

  def grad_for_matmul(self, var: TensorNode) -> ndarray:
    if var.node_type == 'p' and type(var.child.grad) != type(None):
      return var.child.grad @ var.child.parent[1].tensor.T
    elif var.node_type == 's' and type(var.child.grad) != type(None):
      return var.child.parent[0].tensor.T @ var.child.grad
    
  def grad_for_reduce_sum(self, var: TensorNode) -> ndarray:
    if type(var.child.grad) == type(None):
      # SELF.child.tensor is a scalar
      return cupy.ones(var.tensor.shape)
    # SELF 
    return var.child.grad * cupy.ones(var.tensor.shape)
    
  def grad_for_add(self, var: TensorNode):
    A, B, prev_grad, node_type = self.get_attributes(var)

    # for space of A == space of B == space of C == R^{PxQ} where C=A+B
    if A.shape == B.shape:
      return prev_grad

  def grad_for_truediv(self, var: TensorNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)

    if type(prev_grad) == type(None):
      return 1.0 / B

    # for space of A == space of B == space of C == R^{PxQ} where C=A/B 
    if A.shape == B.shape and node_type == 'p':
      return (1 / B) * prev_grad
    elif A.shape == B.shape and node_type == 's':
      return -1 * (B / A**2) * prev_grad
    elif B.shape == () and node_type == 'p':
      return B * prev_grad
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not havea a gradient!')
    print(f'Your expression might be unsupported!')

# future: skip connections
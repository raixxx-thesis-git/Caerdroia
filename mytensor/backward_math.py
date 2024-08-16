from __future__ import annotations
from typing import TYPE_CHECKING
from cupy import ndarray

import cupy

if TYPE_CHECKING:
  from mytensor import TensorNode

class BackwardMath():
  def __init__(self): pass

  def grad_for_matmul_primary(self, var: TensorNode) -> ndarray:
    return var.child.grad @ var.child.parent[1].tensor.T
  
  def grad_for_matmul_secondary(self, var: TensorNode) -> ndarray:
    return var.child.parent[0].tensor.T @ var.child.grad
  
  def grad_for_reduce_sum(self, var: TensorNode) -> ndarray:
    if type(var.child.grad) == type(None):
      # SELF.child.tensor is a vector
      return cupy.ones(var.tensor.shape)
    # SELF 
    return var.child.grad * cupy.ones(var.tensor.shape)
    
  def grad_for_add(self, var: TensorNode):
    return var.child.tensor

  def grad_for_truediv(self, var: TensorNode) -> ndarray:
    if type(var.child.grad) == type(None):
      # SELF.child.tensor is a scalar
      # SELF.tensor is a scalar
      # SELF.child.parent[1].tensor is a scalar
      return 1.0/var.child.parent[1].tensor
    # SELF.child.parent[1].tensor is a scalar since it is a SELF's denominator
    # SELF.grad could be a matrix / vector / scalar
    return 1.0/var.child.parent[1].tensor * var.child.grad
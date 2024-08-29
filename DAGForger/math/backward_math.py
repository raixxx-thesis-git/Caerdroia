from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from cupy import ndarray

import cupy

if TYPE_CHECKING:
  from DAGForger import DAGNode

class BackwardMath():
  def __init__(self): pass

  def get_attributes(self, var: DAGNode) -> Tuple[ndarray, ndarray|None, ndarray, str]:
    child: DAGNode = var.child
    node_type = var.node_type

    if node_type == 'p':
      partner: DAGNode = var.child.parent[1]
    elif node_type == 's':
      partner: DAGNode = var.child.parent[0]
 
    prev_grad: ndarray = cupy.array([[1.]])
    A: ndarray = var.tensor
    B: ndarray|None = None

    if child != None:
      prev_grad = child.grad
    if partner != None:
      B = partner.tensor

    return A, B, prev_grad, node_type

  def grad_for_matmul(self, var: DAGNode) -> ndarray:
    _, B, prev_grad, node_type = self.get_attributes(var)

    if node_type == 'p':
      return prev_grad @ B.T
    elif node_type == 's':
      return B.T @ prev_grad
    
  def grad_for_reduce_sum(self, var: DAGNode) -> ndarray:
    A, _, prev_grad, _ = self.get_attributes(var)

    if ((A.shape[0] != 1 and A.shape[1] == prev_grad.shape[1]) or 
        (A.shape[0] == 1 and A.shape[0] == prev_grad.shape[0])):
      return cupy.ones(shape=A.shape) * prev_grad
    
    print(f'WARNING: Tensor {var.name} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_add(self, var: DAGNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)

    if (A.shape == B.shape) or (A.shape[0] != 1 and A.shape[1] == B.shape[1]):
      # if A + B = C or B + A = C where A is R^{MxN} and B is R^{MxN} or R^{1xN}
      return prev_grad
    
    elif A.shape[0] == 1 and A.shape[1] == B.shape[1]:
      # if A + B = C or B + A = C where A is R^{1xN} and B is R^{MxN}
      return cupy.sum(prev_grad, axis=0, keepdims=True)
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')
    
  def grad_for_sub(self, var: DAGNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)

    if A.shape == B.shape and node_type == 'p':
      # if A - B = C where A is R^{MxN} and B is R^{MxN}
      return prev_grad
    
    elif A.shape == B.shape and node_type == 's':
      # if B - A = C where A is R^{MxN} and B is R^{MxN}
      return -1 * prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_truediv(self, var: DAGNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)
    
    if (A.shape == B.shape or B.shape == ()) and node_type == 'p':
      # if A/B = C where A is R^{MxN} and B is R^{MxN} or R^{1x1}
      return (1 / B) * prev_grad
    
    elif (A.shape == B.shape or B.shape == ()) and node_type == 's':
      # if B/A = C where A is R^{MxN} and B is R^{MxN} or R^{1x1}
      return -1 * (B / A**2) * prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_mul(self, var: DAGNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)
    print('xxx', B)
    if A.shape == B.shape or B.shape == ():
      # if A ☉ B = C where A and B are R^{MxN}
      return B * prev_grad
    
    print(f'WARNINGx: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported! ')

  def grad_for_pow(self, var: DAGNode) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)

    # if A^B = C where A is R^{MxN} and B is R^{1x1}
    return B * A**(B - 1) * prev_grad
  
  def grad_for_log(self, var: DAGNode) -> ndarray:
    A, _, prev_grad, _ = self.get_attributes(var)
    basis = var.temp_state_log_basis
    
    # if log(A) = C where A is R^{MxN}
    return (1/cupy.log(basis)) * prev_grad / A
  
  def grad_for_abs(self, var: DAGNode) -> ndarray:
    A, _, prev_grad, _ = self.get_attributes(var)

    # if |A| = C where A is R^{MxN}
    return (A/cupy.absolute(A)) * prev_grad

# future: skip connections

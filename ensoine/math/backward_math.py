from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from cupy import ndarray

import cupy

if TYPE_CHECKING:
  from ensoine import Node

class BackwardMath():
  def __init__(self): pass

  def get_attributes(self, var: Node) -> Tuple[ndarray, ndarray|None, ndarray, str]:
    child: Node = var.child
    node_type = var.node_type

    if node_type == 'p':
      partner: Node = var.child.parent[1]
    elif node_type == 's':
      partner: Node = var.child.parent[0]
 
    prev_grad: ndarray = cupy.array([[1.]])
    A: ndarray = var.tensor
    B: ndarray|None = None

    if child != None:
      prev_grad = child.grad
    if partner != None:
      B = partner.tensor

    return A, B, prev_grad, node_type
    
  def grad_for_matmul(self, L: ndarray, R: ndarray, prev_grad: ndarray) -> ndarray:
    grad_L = prev_grad @ R.T
    grad_R = L.T @ prev_grad
    return (grad_L, grad_R)
    
  def grad_for_reduce_sum(self, L: ndarray, prev_grad: ndarray) -> ndarray:
    if ((L.shape[0] != 1 and L.shape[1] == prev_grad.shape[1]) or 
        (L.shape[0] == 1 and L.shape[0] == prev_grad.shape[0])):
      grad_L = cupy.ones(shape=L.shape) * prev_grad

    return grad_L

  def grad_for_add(self, L: ndarray, R: ndarray, prev_grad: ndarray) -> ndarray:
    equal_space_operands = L.shape == R.shape
    L_broadcast = (L.shape[0] == 1 and R.shape[0] != 1) and (L.shape[1] == R.shape[1])
    R_broadcast = (L.shape[0] != 1 and R.shape[0] == 1) and (L.shape[1] == R.shape[1])

    if equal_space_operands:
      grad_L = prev_grad
      grad_R = prev_grad
    elif L_broadcast:
      grad_L = cupy.sum(prev_grad, axis=0, keepdims=True)
      grad_R = prev_grad
    elif R_broadcast:
      grad_L = prev_grad
      grad_R = cupy.sum(prev_grad, axis=0, keepdims=True)

    return (grad_L, grad_R)
    
  def grad_for_sub(self, L: ndarray, R: ndarray, prev_grad: ndarray) -> ndarray:
    equal_space_operands = L.shape == R.shape
    L_broadcast = (L.shape[0] == 1 and R.shape[0] != 1) and (L.shape[1] == R.shape[1])
    R_broadcast = (L.shape[0] != 1 and R.shape[0] == 1) and (L.shape[1] == R.shape[1])

    if equal_space_operands:
      grad_L = prev_grad
      grad_R = -1.0 * prev_grad
    elif L_broadcast:
      grad_L = cupy.sum(prev_grad, axis=0, keepdims=True)
      grad_R = -1.0 * prev_grad
    elif R_broadcast:
      grad_L = prev_grad
      grad_R = -1.0 * cupy.sum(prev_grad, axis=0, keepdims=True)

    return (grad_L, grad_R)

  def grad_for_truediv(self, var: Node) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)
    
    if (A.shape == B.shape or B.shape == ()) and node_type == 'p':
      # if A/B = C where A is R^{MxN} and B is R^{MxN} or R^{1x1}
      return (1 / B) * prev_grad
    
    elif (A.shape == B.shape or B.shape == ()) and node_type == 's':
      # if B/A = C where A is R^{MxN} and B is R^{MxN} or R^{1x1}
      return -1 * (B / A**2) * prev_grad
    
    print(f'WARNING: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported!')

  def grad_for_mul(self, var: Node) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)
    print('xxx', B)
    if A.shape == B.shape or B.shape == ():
      # if A ☉ B = C where A and B are R^{MxN}
      return B * prev_grad
    
    print(f'WARNINGx: Tensor {var.name} with node_type {node_type} does not have a gradient!')
    print(f'Your expression might be unsupported! ')

  def grad_for_pow(self, var: Node) -> ndarray:
    A, B, prev_grad, node_type = self.get_attributes(var)

    # if A^B = C where A is R^{MxN} and B is R^{1x1}
    return B * A**(B - 1) * prev_grad
  
  def grad_for_log(self, var: Node) -> ndarray:
    A, _, prev_grad, _ = self.get_attributes(var)
    basis = var.temp_state_log_basis
    
    # if log(A) = C where A is R^{MxN}
    return (1/cupy.log(basis)) * prev_grad / A
  
  def grad_for_abs(self, var: Node) -> ndarray:
    A, _, prev_grad, _ = self.get_attributes(var)

    # if |A| = C where A is R^{MxN}
    return (A/cupy.absolute(A)) * prev_grad
  

# future: skip connections

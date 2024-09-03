from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from cupy import ndarray
import cupy

if TYPE_CHECKING:
  from ensoine.graph import Node

def consider(grad: ndarray, is_constant: bool) -> bool:
  return grad if not is_constant else None

def LR_init(L: Node, R: Node) -> Tuple[ndarray, ndarray, bool, bool]:
  return (L.tensor, R.tensor, L.get_is_constant(), R.get_is_constant())

def L_init(L: Node) -> Tuple[ndarray, bool]:
  return (L.tensor, L.get_is_constant())

def grad_for_matmul(L: Node, R: Node, prev_grad: ndarray) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  grad_L = consider(prev_grad @ R.T, L_is_constant)
  grad_R = consider(L.T @ prev_grad, R_is_constant)

  return (grad_L, grad_R)
  
def grad_for_reduce_sum(L: Node, prev_grad: ndarray) -> ndarray:
  L, L_is_constant = L_init(L)

  if ((L.shape[0] != 1 and L.shape[1] == prev_grad.shape[1]) or 
      (L.shape[0] == 1 and L.shape[0] == prev_grad.shape[0])):
    grad_L = consider(cupy.ones(shape=L.shape) * prev_grad, L_is_constant)

  return grad_L

def grad_for_add(L: Node, R: Node, prev_grad: ndarray) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  equal_space_operands = L.shape == R.shape
  L_broadcast = (L.shape[0] == 1 and R.shape[0] != 1) and (L.shape[1] == R.shape[1])
  R_broadcast = (L.shape[0] != 1 and R.shape[0] == 1) and (L.shape[1] == R.shape[1])

  if equal_space_operands:
    grad_L = consider(prev_grad, L_is_constant)
    grad_R = consider(prev_grad, R_is_constant)
  elif L_broadcast:
    grad_L = consider(cupy.sum(prev_grad, axis=0, keepdims=True), L_is_constant)
    grad_R = consider(prev_grad, R_is_constant)
  elif R_broadcast:
    grad_L = consider(prev_grad, L_is_constant)
    grad_R = consider(cupy.sum(prev_grad, axis=0, keepdims=True), R_is_constant)

  return (grad_L, grad_R)
  
def grad_for_sub(L: Node, R: Node, prev_grad: ndarray) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  equal_space_operands = L.shape == R.shape
  L_broadcast = (L.shape[0] == 1 and R.shape[0] != 1) and (L.shape[1] == R.shape[1])
  R_broadcast = (L.shape[0] != 1 and R.shape[0] == 1) and (L.shape[1] == R.shape[1])

  if equal_space_operands:
    grad_L = consider(prev_grad, L_is_constant)
    grad_R = consider(-1.0 * prev_grad, R_is_constant)
  elif L_broadcast:
    grad_L = consider(cupy.sum(prev_grad, axis=0, keepdims=True), L_is_constant)
    grad_R = consider(-1.0 * prev_grad, R_is_constant)
  elif R_broadcast:
    grad_L = consider(prev_grad, L_is_constant)
    grad_R = consider(-1.0 * cupy.sum(prev_grad, axis=0, keepdims=True), R_is_constant)

  return (grad_L, grad_R)

def grad_for_div(L: Node, R: Node, prev_grad: ndarray) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  if L.shape == R.shape:
    grad_L = consider((1/R) * prev_grad, L_is_constant)
    grad_R = consider(-1.0*L/(R**2) * prev_grad, R_is_constant)
  elif R.shape == ():
    grad_L = consider((1/R) * prev_grad, L_is_constant)
    grad_R = consider((-1/R**2) * cupy.sum(prev_grad * L, keepdims=True), R_is_constant)
  elif L.shape == ():
    pass

  return (grad_L, grad_R)

def grad_for_mul(L: Node, R: Node, prev_grad: ndarray) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  if L.shape == R.shape:
    grad_L = consider(R * prev_grad, L_is_constant)
    grad_R = consider(L * prev_grad, R_is_constant)
  elif R.shape == ():
    grad_L = consider(R * prev_grad, L_is_constant)
    grad_R = consider(cupy.sum(prev_grad * L, keepdims=True), R_is_constant)
  elif L.shape == ():
    pass

  return (grad_L, grad_R)
  
def grad_for_pow(L: Node, R: Node, prev_grad: ndarray) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  grad_L = consider(R * L**(R-1) * prev_grad, L_is_constant)
  grad_R = None # in the future, fix this!

  return (grad_L, grad_R)
from __future__ import annotations
from typing import TYPE_CHECKING
from cupy import ndarray

import operator 
import cupy

if TYPE_CHECKING:
  from mytensor import TensorNode


class TensorMathError(Exception):
  def __init__(self, msg):
    super().__init__(msg)

class ForwardMath():
  def __init__(self):
    pass

  def add(self, A: ndarray, B: ndarray) -> ndarray:
    return operator.add(A, B)
  
  def sub(self, A: ndarray, B: ndarray) -> ndarray:
    return operator.sub(A, B)

  def matmul(self, A: ndarray, B: ndarray) -> ndarray:
    return operator.matmul(A, B) 

  def mul(self, A: ndarray, B: ndarray) -> ndarray:
    return operator.mul(A, B) 

  def truediv(self, A: ndarray, B: ndarray) -> ndarray:
    return operator.truediv(A, B) 
    
  def pow(self, A: ndarray, B: ndarray) -> ndarray:
    return operator.pow(A, B)
  
  def log_(self, A: ndarray, basis: int) -> ndarray:
    return cupy.log(A)/cupy.log(basis)
  
  def abs_(self, A: ndarray) -> ndarray:
    return cupy.absolute(A)
  
  def reduce_sum_(self, A: ndarray, axis: int) -> ndarray:
    return cupy.sum(A, axis=axis, keepdims=True)
  
    
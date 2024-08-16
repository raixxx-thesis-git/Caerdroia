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

  def basic_linalg(self, tensor_node: TensorNode, partner: TensorNode, operation: str) -> ndarray:
    operator_dict = {'+': operator.add, 
                     '-': operator.sub, 
                     '@': operator.matmul, 
                     '*': operator.mul,
                     '/': operator.truediv,
                     '**': operator.pow}

    return operator_dict[operation](tensor_node.tensor, partner.tensor)
  
  def reduce_sum(self, tensor_node: TensorNode, axis: int) -> ndarray:
    return cupy.sum(tensor_node.tensor, axis=axis, keepdims=True)
    
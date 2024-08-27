from __future__ import annotations 
from typing import TYPE_CHECKING
from cupy import ndarray
from typing import List, Any, Tuple

from mytensor import BackwardMath
from mytensor import NodeComputer

if TYPE_CHECKING:
  from mytensor.tensor_node import TensorNode

import cupy

class TensorNodeError(Exception):
  def __init__(self, msg):
    super().__init__(msg)

class TensorNode(NodeComputer):
  def __init__(self, tensor: ndarray | List[Any], 
               name: str = None, 
               parent: Tuple[int|None, int|None] = (None, None), 
               child: int|None = None,
               grad: ndarray|None = None,
               is_weight: bool = False,
               operation: str|None = None,
               is_constant: bool = False,
               node_type: str = 'p'):
    if type(tensor) == ndarray: 
      self.tensor: ndarray = tensor
    elif type(tensor) == list or type(tensor) == float: 
      self.tensor: ndarray = cupy.array(tensor)
    else: raise TensorNodeError('Unknown tensor.')

    self.backward_math = BackwardMath()
    self.name = name
    self.parent = parent
    self.child = child
    self.grad = grad
    self.is_weight = is_weight
    self.operation = operation
    self.is_constant = is_constant
    self.updated = False
    self.node_type = node_type
  
  def __repr__(self) -> str:
    return (f'TensorNode\nName:{self.name}\n'
            f'Memory:{id(self)}\n'
            f'is_weight:{self.is_weight}\n'
            f'is_constant:{self.is_constant}\n'
            f'data_shape:{cupy.shape(self.tensor)}')

  @property
  def T(self):
    return TensorNode(self.tensor.T, name=f'{self.name}.T')
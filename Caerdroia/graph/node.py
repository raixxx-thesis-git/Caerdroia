from __future__ import annotations 
from typing import TYPE_CHECKING, Optional, Union
from cupy import ndarray
from typing import List, Any, Tuple
from Caerdroia.system import System

if TYPE_CHECKING:
  from Caerdroia.graph.node import Node
  from Caerdroia.graph import Triplet, Duplet, Dynamic

import cupy

class NodeError(Exception):
  def __init__(self, msg):
    super().__init__(msg)

class Node(System):
  def __init__(self, tensor: ndarray | List[Any], 
               name: str = None, 
               is_weight: bool = False,
               operation: str|None = None,
               is_constant: bool = False):
    if type(tensor) == ndarray: 
      self.tensor: ndarray = tensor
    elif type(tensor) == list or type(tensor) == float: 
      self.tensor: ndarray = cupy.array(tensor)
    else: raise NodeError('Unknown tensor.')

    super().__init__()
    self.name = name
    self.is_weight = is_weight
    self.operation = operation
    self.is_constant = is_constant
    self.adic: Optional[Union[Triplet, Duplet]] = None
    self.grad_pool = []
  
  def __repr__(self) -> str:
    return (f'Node\nName:{self.name}\n'
            f'Memory:{id(self)}\n'
            f'is_weight:{self.is_weight}\n'
            f'is_constant:{self.is_constant}\n'
            f'data_shape:{cupy.shape(self.tensor)}')

  def set_adic(self, adic: Union[Duplet, Triplet, Dynamic]) -> None:
    self.adic = adic
  
  def get_adic(self) -> Union[Duplet, Triplet, Dynamic]:
    return self.adic
  
  def get_is_constant(self) -> bool:
    return self.is_constant
  
  def add_gradient(self, grad: Union[ndarray, None]) -> None:
    if type(grad) != type(None):
      self.grad_pool.append(grad)

  def get_last_gradient(self) -> None:
    return self.grad_pool[-1]
  
  def get_gradient(self) -> ndarray:
    if len(self.grad_pool) == 0:
      print('No gradient is provided.')
      return
    
    total_gradient = self.grad_pool[0]
    for grad in self.grad_pool[1:]:
      total_gradient = total_gradient + grad
    return total_gradient
  
  @property
  def T(self):
    return Node(self.tensor.T, name=f'{self.name}.T')


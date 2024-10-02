from __future__ import annotations 
from typing import TYPE_CHECKING, Optional, Union, Dict
from cupy import ndarray
from typing import List, Any, Tuple
from nodeleys.system import System

if TYPE_CHECKING:
  from nodeleys.graph.node import Node
  from nodeleys.graph import Triplet, Duplet, Virtual

import cupy

class NodeError(Exception):
  def __init__(self, msg):
    super().__init__(msg)

class Node(System):
  def __init__(self, tensor: ndarray | List[Any], 
               name: str = None, 
               is_trainable: bool = False,
               operation: str|None = None,
               is_constant: bool = False):
    if type(tensor) == ndarray: 
      self.tensor: ndarray = tensor
    elif type(tensor) == list or type(tensor) == float: 
      self.tensor: ndarray = cupy.array(tensor)
    else: raise NodeError('Unknown tensor.')

    super().__init__()
    self.name = name
    self.is_trainable = is_trainable
    self.operation = operation
    self.is_constant = is_constant
    self.adic: Optional[Union[Triplet, Duplet]] = None
    self.grad_pool = []
    self.virtual_grad_pool = {}
    self.metadata = {}
  
  def __repr__(self) -> str:
    return (f'Node\nName:{self.name}\n'
            f'Memory:{id(self)}\n'
            f'is_trainable:{self.is_trainable}\n'
            f'is_constant:{self.is_constant}\n'
            f'data_shape:{cupy.shape(self.tensor)}')
  
  def set_adic(self, adic: Union[Duplet, Triplet, Virtual]) -> None:
    self.adic = adic
  
  def get_adic(self) -> Union[Duplet, Triplet, Virtual]:
    return self.adic
  
  def get_is_constant(self) -> bool:
    return self.is_constant
   
  def add_gradient(self, grad: Union[ndarray, None]) -> None: 
    if type(grad) != type(None):
      self.grad_pool.append(grad)

  def add_virtual_gradient(self, grad: Union[ndarray, None], idx: int) -> None:
    try:
      # Virtual_grad_pool holds the gradient pools of each corresponding conditions represented
      # by <idx> (virtual sessions).
      self.virtual_grad_pool[idx].append(grad)

    except KeyError:
      # This error is expected when <virtual_grad_pool> has not been initialized. We initialize
      # the virtual gradient pool with a key that corresponds to the virtual session / condition (<idx>) 
      # and a value of the passed gradient.
      self.virtual_grad_pool: Dict[int, List[ndarray]]
      self.virtual_grad_pool[idx] = [grad]

    except AttributeError:
      self.virtual_grad_pool = {}
      self.virtual_grad_pool[idx] = [grad]

  def sum_virtual_gradient_by_session(self, idx: int) -> None:
    try:
      # [Explanation]
      # This block is expected to get the final gradient of a dependent variable. By considering the 
      # chain rule, the gradient is the sum of all gradient taken by different paths.

      # [Admonition]
      # The summation of the gradient does NOT constitute the gradient of the node w.r.t the loss function, 
      # but rather w.r.t the loss function taken from the virtual paths (and static path if static path exists 
      # before the virtual node in the backpropagation process). The actual gradient is the meshed form of
      # all virtual gradient pools from all virtual session (<idx>). The meshing process is determined by
      # <Virtual.masked_ids>.
      self.virtual_grad_pool[idx] = cupy.sum(cupy.array(self.virtual_grad_pool[idx]), axis=0)

    except KeyError:
      self.virtual_grad_pool[idx] = None

  def get_virtual_gradient_by_session(self, idx: int) -> Union[List[ndarray], ndarray]:
    return self.virtual_grad_pool[idx]

  def get_last_virtual_gradient(self, idx: int) -> None:
    return self.virtual_grad_pool[idx][-1]
  
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
  
  def assign_metadata(self, key: str, val: Any) -> None:
    self.metadata[key] = val

  def get_metadata(self, key: str) -> Any:
    return self.metadata[key]
    
  @property
  def T(self):
    return Node(self.tensor.T, name=f'{self.name}.T')


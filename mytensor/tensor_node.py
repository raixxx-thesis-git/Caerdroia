from __future__ import annotations 
from typing import TYPE_CHECKING
from cupy import ndarray
from typing import List, Any, Tuple

from mytensor import ForwardMath
from mytensor import BackwardMath

if TYPE_CHECKING:
  from mytensor.tensor_node import TensorNode

import cupy

class TensorNodeError(Exception):
  def __init__(self, msg):
    super().__init__(msg)

class TensorNode():
  def __init__(self, tensor: ndarray | List[Any], 
               name: str = None, 
               parent: Tuple[int|None, int|None] = (None, None), 
               child: int|None = None,
               grad: ndarray|None = None,
               is_weight: bool = False,
               operation: str|None = None,
               is_constant: bool = False):
    if type(tensor) == ndarray: self.tensor = tensor
    elif type(tensor) == list or type(tensor) == float: self.tensor = cupy.array(tensor)
    else: raise TensorNodeError('Unknown tensor.')

    self.math = ForwardMath()
    self.name = name
    self.parent = parent
    self.child = child
    self.grad = grad
    self.is_weight = is_weight
    self.operation = operation
    self.is_constant = is_constant


  def make_child(self, partner, operation, value) -> TensorNode:
    if partner == None: name = f'{operation}({self.name})'
    else: name = f'({self.name}{operation}{partner.name})'

    child = TensorNode(value,
                       parent=(self, partner),
                       name=name,
                       operation=operation)

    self.child = child
    if partner != None: partner.child = child

    return child
  
  def partner_assure_tensornode(self, partner: TensorNode | float) -> TensorNode:
    if type(partner) == float:
      partner = TensorNode(partner, 
                          name=str(partner),
                          is_constant=True)
    
    return partner
    
  def __repr__(self) -> str:
    return (f'TensorNode\nName:{self.name}\n'
            f'Memory:{id(self)}\n'
            f'is_weight:{self.is_weight}\n'
            f'is_constant:{self.is_constant}\n'
            f'data_shape:{cupy.shape(self.tensor)}')

  def __add__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.math.basic_linalg(self, partner, '+')
    return self.make_child(partner, '+', value)

  def __sub__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.math.basic_linalg(self, partner, '-')
    return self.make_child(partner, '-', value)
  
  def __matmul__(self, partner: TensorNode) -> TensorNode:
    value = self.math.basic_linalg(self, partner, '@')
    return self.make_child(partner, '@', value)

  def __mul__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.math.basic_linalg(self, partner, '*')
    return self.make_child(partner, '*', value)

  def __truediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.math.basic_linalg(self, partner, '/')
    return self.make_child(partner, '/', value)
  
  def __pow__(self,  partner: TensorNode | float) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.math.basic_linalg(self, partner, '**')
    return self.make_child(partner, '**', value)

  def reduce_sum(self, axis: int) -> TensorNode:
    value = self.math.reduce_sum(self, axis)
    return self.make_child(None, 'redsum', value)
  
  def backprop(self):
    child = self.child
    if child == None:
      self.grad = None
      self.parent[0].backprop()
      return
    
    backward_math = BackwardMath()
    operator_to_method = {
      '@p': backward_math.grad_for_matmul_primary,
      '@s': backward_math.grad_for_matmul_secondary,
      '+p': backward_math.grad_for_add,
      '+s': backward_math.grad_for_add,
      'redsump': backward_math.grad_for_reduce_sum,
      '/p': backward_math.grad_for_truediv,
      '/s': backward_math.grad_for_truediv
    }

    if self == child.parent[0]:
      self.grad = operator_to_method[child.operation + 'p'](self)
      self.pass_by = child.operation + 'p'
      if child.parent[1] == None or child.parent[1].is_constant:
        # issue: severs the taping
        # some tensor node may be introduced in the middle of the graph
        # but this design would deter the taping to do the backprop all the way through
        # till the end by cutting it right where the tensor node is introduced
        # in the middle of the graph (since parent is none).
        if self.parent[0] == None: print('exit at:', self.name); return
        self.parent[0].backprop()
        return
      if child.parent[1] == None: print('exit at:', self.name); return
      child.parent[1].backprop()
      return
    
    if self == child.parent[1]:
      self.grad = operator_to_method[child.operation + 's'](self)
      self.pass_by = child.operation + 's'
      if self.parent[0] == None: print('exit at:', self.name); return
      self.parent[0].backprop()


  @property
  def T(self):
    return TensorNode(self.tensor.T, name=f'{self.name}.T')
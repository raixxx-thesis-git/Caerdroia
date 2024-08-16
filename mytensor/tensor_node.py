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
               is_constant: bool = False,
               node_type: str = 'p'):
    if type(tensor) == ndarray: self.tensor = tensor
    elif type(tensor) == list or type(tensor) == float: self.tensor = cupy.array(tensor)
    else: raise TensorNodeError('Unknown tensor.')

    self.forward_math = ForwardMath()
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


  def make_child(self, partner: TensorNode | None, operation: str, value: ndarray) -> TensorNode:
    if partner == None: name = f'{operation}({self.name})'
    else: name = f'({self.name}{operation}{partner.name})'

    child = TensorNode(value,
                       parent=(self, partner),
                       name=name,
                       operation=operation)

    self.child = child
    self.node_type = 'p'

    if partner != None: 
      partner.child = child
      partner.node_type = 's'

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
    value = self.forward_math.basic_linalg(self, partner, '+')
    return self.make_child(partner, '+', value)

  def __sub__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.basic_linalg(self, partner, '-')
    return self.make_child(partner, '-', value)
  
  def __matmul__(self, partner: TensorNode) -> TensorNode:
    value = self.forward_math.basic_linalg(self, partner, '@')
    return self.make_child(partner, '@', value)

  def __mul__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.basic_linalg(self, partner, '*')
    return self.make_child(partner, '*', value)

  def __truediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.basic_linalg(self, partner, '/')
    return self.make_child(partner, '/', value)
  
  def __pow__(self,  partner: TensorNode | float) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.basic_linalg(self, partner, '**')
    return self.make_child(partner, '**', value)

  def reduce_sum(self, axis: int) -> TensorNode:
    value = self.forward_math.reduce_sum(self, axis)
    return self.make_child(None, 'redsum', value)
  
  def propagate(self):
    p_parent: TensorNode | None = self.parent[0]
    s_parent: TensorNode | None = self.parent[1]
    child: TensorNode | None = self.child
    if child == None:
      pass # do something!
    partner: TensorNode| None = child.parent[1]

    if not p_parent.updated:
      # move to p_parent
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      p_parent.propagate()
      return
    
    if self.node_type == 'p' and p_parent == None:
      # move to partner
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      if partner == None:
        # perch to child
        child.propagate()
        return
      partner.propagate()
      
    if self.node_type == 's' and p_parent == None:
      # perch to child
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      child.propagate()
      return

    if self.node_type == 'p' and p_parent.updated:
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      if s_parent == None:
        # perch to child
        child.propagate()
      else:
        # move to partner
        partner.propagate()
    
    


  def backprop(self):
    child = self.child
    parent = self.parent

    if parent[0] == None:
      # an end node
      # apply gradient here if is_weight
      if self == child.parent[0]:
        child.parent[1].backprop()
      if self == child.parent[1]:
        child.child.parent[1].backprop()
      return
    
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
      child.parent[0].backprop()
      return
    
    if self == child.parent[1]:
      self.grad = operator_to_method[child.operation + 's'](self)
      self.pass_by = child.operation + 's'
      self.parent[0].backprop()


  @property
  def T(self):
    return TensorNode(self.tensor.T, name=f'{self.name}.T')
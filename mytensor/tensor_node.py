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
    if type(tensor) == ndarray: 
      self.tensor: ndarray = tensor
    elif type(tensor) == list or type(tensor) == float: 
      self.tensor: ndarray = cupy.array(tensor)
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
    operator = operation[0]
    operation_flag = operation[1]

    if partner == None: 
      name = f'{operator}({self.name})'
      self.node_type = 'p'
    elif operation_flag == 'r': 
      name = f'({partner.name}{operator}{self.name})'
      self.node_type = 's'
    else:
      name = f'({self.name}{operator}{partner.name})'
      self.node_type = 'p'

    child = TensorNode(value, parent=(self, partner), name=name, operation=operator)

    self.child = child
    if partner != None: 
      partner.child = child
      partner.node_type = 's' if self.node_type == 'p' else 'p'

    return child
  
  def partner_assure_tensornode(self, partner: TensorNode | float) -> TensorNode:
    if type(partner) == float:
      partner = TensorNode(partner, name=str(partner), is_constant=True)
    return partner
  
  def __repr__(self) -> str:
    return (f'TensorNode\nName:{self.name}\n'
            f'Memory:{id(self)}\n'
            f'is_weight:{self.is_weight}\n'
            f'is_constant:{self.is_constant}\n'
            f'data_shape:{cupy.shape(self.tensor)}')

  def __add__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.add(self.tensor, partner.tensor)
    return self.make_child(partner, '+.', value)

  def __sub__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.sub(self.tensor, partner.tensor)
    return self.make_child(partner, '-.', value)
  
  def __matmul__(self, partner: TensorNode) -> TensorNode:
    value = self.forward_math.matmul(self.tensor, partner.tensor)
    return self.make_child(partner, '@.', value)

  def __mul__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.mul(self.tensor, partner.tensor)
    return self.make_child(partner, '*.', value)

  def __truediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.truediv(self.tensor, partner.tensor)
    return self.make_child(partner, '/.', value)
  
  def __rtruediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.truediv(partner.tensor, self.tensor)
    return self.make_child(partner, '/r', value)
  
  def __pow__(self,  partner: TensorNode | float) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.basic_linalg(self, partner, '**')
    return self.make_child(partner, '**', value)

  def reduce_sum(self, axis: int) -> TensorNode:
    value = self.forward_math.reduce_sum(self, axis)
    return self.make_child(None, 'redsum', value)
  
  def backprop(self):
    p_parent: TensorNode | None = self.parent[0]
    s_parent: TensorNode | None = self.parent[1]
    child: TensorNode | None = self.child
    if child == None:
      pass # do something!
    partner: TensorNode| None = child.parent[1]

    if self.node_type == 'p' and p_parent == None:
      # move to partner
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      if partner == None:
        # perch to child
        child.backprop()
      else:
        partner.backprop()
      
    elif self.node_type == 's' and p_parent == None:
      # perch to child
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      child.backprop()

    elif not p_parent.updated:
      # move to p_parent
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      p_parent.backprop()

    elif self.node_type == 'p' and p_parent.updated:
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      if s_parent == None:
        # set parent's update state to default
        p_parent.updated = False
        # perch to child
        child.backprop()
      elif s_parent.updated:
        # set parents' update state to defualt
        p_parent.updated = False
        s_parent.updated = False
        # move to partner
        if partner == None: return
        else: partner.backprop()

    elif self.node_type == 's' and p_parent.updated:
      if not self.updated: self.grad = self.backward_math.compute_grad(self)
      self.updated = True
      if s_parent == None:
        # set parent's update state to default
        p_parent.updated = False
        # perch to child
      elif s_parent.updated:
        # set parents' update state to defualt
        p_parent.updated = False
        s_parent.updated = False
      child.backprop()

  @property
  def T(self):
    return TensorNode(self.tensor.T, name=f'{self.name}.T')
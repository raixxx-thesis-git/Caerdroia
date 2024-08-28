from __future__ import annotations
from typing import TYPE_CHECKING
from cupy import ndarray
from mytensor import ForwardMath
from mytensor import BackwardMath
if TYPE_CHECKING:
  from mytensor import TensorNode

import cupy

class NodeComputer(ForwardMath, BackwardMath):
  def __init__(self):
    pass

  def make_child(self, partner: TensorNode | None, operation: str, value: ndarray) -> TensorNode:
    # import TensorNode module is put here to avoid a circular import.
    from mytensor import TensorNode

    operator = operation[:-1]
    operation_flag = operation[-1]

    if partner == None: 
      name = f'{operator}({self.name})'
      self.node_type = 'p'
      parent = (self, partner)
    elif operation_flag == 'r': 
      name = f'({partner.name}{operator}{self.name})'
      self.node_type = 's'
      parent = (partner, self)
    else:
      name = f'({self.name}{operator}{partner.name})'
      self.node_type = 'p'
      parent = (self, partner)

    child = TensorNode(value, parent=parent, name=name, operation=operator)

    self.child = child
    if partner != None: 
      partner.child = child
      partner.node_type = 's' if self.node_type == 'p' else 'p'

    return child

  def partner_assure_tensornode(self, partner: TensorNode | float) -> TensorNode:
    # import TensorNode module is put here to avoid a circular import.
    from mytensor import TensorNode

    if type(partner) == float:
      partner = TensorNode(partner, name=str(partner), is_constant=True)
    return partner
  
  def __add__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.add(self.tensor, partner.tensor)
    return self.make_child(partner, '+.', value)

  def __sub__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.sub(self.tensor, partner.tensor)
    return self.make_child(partner, '-.', value)
  
  def __matmul__(self, partner: TensorNode) -> TensorNode:
    value = self.matmul(self.tensor, partner.tensor)
    return self.make_child(partner, '@.', value)

  def __mul__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.mul(self.tensor, partner.tensor)
    return self.make_child(partner, '*.', value)

  def __truediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.truediv(self.tensor, partner.tensor)
    return self.make_child(partner, '/.', value)
  
  def __rtruediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.truediv(partner.tensor, self.tensor)
    return self.make_child(partner, '/r', value)
  
  def __pow__(self,  partner: TensorNode | float) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.pow(self.tensor, partner.tensor)
    return self.make_child(partner, '**.', value)

  def reduce_sum(self, axis: int) -> TensorNode:
    value = self.reduce_sum_(self, axis)
    return self.make_child(None, 'redsum.', value)
  
  def compute_grad(self):
    operation = self.child.operation
    operation_operator = {
      '+': self.grad_for_add,
      '-': self.grad_for_sub,
      '/': self.grad_for_truediv,
      '*': self.grad_for_mul,
      '@': self.grad_for_matmul,
      '**': self.grad_for_pow,
      'redsum': self.grad_for_reduce_sum
    }
    return operation_operator[operation](self)

  def backprop(self):
    # for pylance
    self.child: None | TensorNode

    p_parent: TensorNode | None = self.parent[0]
    s_parent: TensorNode | None = self.parent[1]
    child: None | TensorNode = self.child
    
    if child == None:
      partner = None
    else:
      partner: TensorNode| None = child.parent[1]

    if self.node_type == 'p' and p_parent == None:
      # move to partner
      if not self.updated: self.grad = self.compute_grad()
      self.updated = True
      if partner == None:
        # perch to child
        child.backprop()
      else:
        partner.backprop()
      
    elif self.node_type == 's' and p_parent == None:
      # perch to child
      if not self.updated: self.grad = self.compute_grad()
      self.updated = True
      child.backprop()

    elif not p_parent.updated:
      # move to p_parent
      if child == None: self.grad = cupy.array([[1.]])
      elif not self.updated: self.grad = self.compute_grad()
      self.updated = True
      p_parent.backprop()

    elif self.node_type == 'p' and p_parent.updated:
      if not self.updated: self.grad = self.compute_grad()
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
      if not self.updated: self.grad = self.compute_grad()
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
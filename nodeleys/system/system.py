from __future__ import annotations
from typing import TYPE_CHECKING, List, Union
from cupy import ndarray
from nodeleys.system import secure_type
from nodeleys.math import ForwardMath, BackwardMath

if TYPE_CHECKING:
  from nodeleys import Node

import cupy

class System(ForwardMath, BackwardMath):
  def __init__(self):
    self.operation_operator = {
      '+': self.grad_for_add,
      '-': self.grad_for_sub,
      '/': self.grad_for_truediv,
      '*': self.grad_for_mul,
      '@': self.grad_for_matmul,
      '**': self.grad_for_pow,
      'log': self.grad_for_log,
      'abs': self.grad_for_abs,
      'redsum': self.grad_for_reduce_sum
    }

  def complete_adic(self: Node, coop: Node | None, operation: str, 
                    outcome: ndarray) -> Node:
    from nodeleys.graph import Triplet, Duplet
    # import Node module is put here to avoid a circular import.
    from nodeleys import Node

    operator = operation[:-1]
    operation_flag = operation[-1]

    outcome_node = Node(outcome)

    if coop == None:
      adic = Duplet(self, outcome_node, operator)
      adic.set_prev(self.adic)
    elif operation_flag == 'r':
      adic = Triplet(coop, self, outcome_node, operator)
      adic.set_prev((coop.adic, self.adic))
    elif operation_flag == '.':
      adic = Triplet(self, coop, outcome_node, operator)
      adic.set_prev((self.adic, coop.adic))

    outcome_node.adic = adic

    return outcome_node


  def make_child(self, partner: Node | None, operation: str, value: ndarray) -> Node:
    # import Node module is put here to avoid a circular import.
    from nodeleys import Node

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

    child = Node(value, parent=parent, name=name, operation=operator)

    self.child = child
    if partner != None: 
      partner.child = child
      partner.node_type = 's' if self.node_type == 'p' else 'p'

    return child

  def partner_assure_Node(self, partner: Node | float) -> Node:
    # import Node module is put here to avoid a circular import.
    from nodeleys import Node

    if type(partner) == float:
      partner = Node(partner, name=str(partner), is_constant=True)
    return partner
  
  def __add__(self, coop: Node) -> Node:
    coop = secure_type(coop)
    outcome = self.add(self.tensor, coop.tensor)
    return self.complete_adic(coop, '+.', outcome)

  def __sub__(self, partner: Node) -> Node:
    partner = self.partner_assure_Node(partner)
    value = self.sub(self.tensor, partner.tensor)
    return self.make_child(partner, '-.', value)
  
  def __matmul__(self, partner: Node) -> Node:
    value = self.matmul(self.tensor, partner.tensor)
    return self.make_child(partner, '@.', value)

  def __mul__(self, partner: Node) -> Node:
    partner = self.partner_assure_Node(partner)
    value = self.mul(self.tensor, partner.tensor)
    return self.make_child(partner, '*.', value)

  def __truediv__(self, partner: Node) -> Node:
    partner = self.partner_assure_Node(partner)
    value = self.truediv(self.tensor, partner.tensor)
    return self.make_child(partner, '/.', value)
  
  def __rtruediv__(self, partner: Node) -> Node:
    partner = self.partner_assure_Node(partner)
    value = self.truediv(partner.tensor, self.tensor)
    return self.make_child(partner, '/r', value)
  
  def __pow__(self,  partner: Node | float) -> Node:
    partner = self.partner_assure_Node(partner)
    value = self.pow(self.tensor, partner.tensor)
    return self.make_child(partner, '**.', value)

  def reduce_sum(self, axis: int) -> Node:
    value = self.reduce_sum_(self.tensor, axis)
    return self.make_child(None, 'redsum.', value)
  
  def log(self, basis: int) -> Node:
    self.temp_state_log_basis = basis
    value = self.log_(self.tensor, basis)
    return self.make_child(None, f'log.', value)
  
  def abs(self) -> Node:
    value = self.abs_(self.tensor)
    return self.make_child(None, f'abs.', value)
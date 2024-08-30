from __future__ import annotations
from typing import TYPE_CHECKING, List
from cupy import ndarray
from Caerdroia.system import secure_type
from Caerdroia.math import ForwardMath, BackwardMath
from Caerdroia.graph import Triplet, Duplet

if TYPE_CHECKING:
  from Caerdroia import Node

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
      'redsum': self.grad_for_reduce_sum,
      'condmap': self.grad_for_conditional_map
    }

  def complete_adic(self: Node, coop: Node | None, operation: str, 
                    outcome: ndarray) -> Node:
    # import Node module is put here to avoid a circular import.
    from Caerdroia import Node

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
    from Caerdroia import Node

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
    from Caerdroia import Node

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
  
  def conditional_map(self, maps: List[Node], conditions: List[str]) -> Node:
    new_matrix = cupy.array(self.tensor)
    
    self.temp_state_conditional_func_conditions = conditions
    self.temp_state_conditional_func_maps = maps

    for map, condition in zip(maps, conditions):
      condition_ = cupy.argwhere(eval(condition.replace('X', 'self.tensor')))
      map.child = self
      new_matrix[condition_[:,0], condition_[:,1]] = map.tensor[condition_[:,0], condition_[:,1]]
    
    return self.make_child(None, f'condmap.', new_matrix)
  
  def grad_for_conditional_map(self, var: Node) -> ndarray:
    maps = var.temp_state_conditional_func_maps
    conditions = var.temp_state_conditional_func_conditions
    new_matrix = self.tensor

    for map, condition in zip(maps, conditions):
      condition_ = cupy.argwhere(eval(condition.replace('X', 'self.tensor')))
      print(map.parent)
      grad_map = self.operation_operator[map.operation](map.parent[0])
      new_matrix[condition_[:,0], condition_[:,1]] = grad_map.tensor[condition_[:,0], condition_[:,1]]

    return new_matrix
  
  def compute_grad(self):
    operation = self.child.operation
    return self.operation_operator[operation](self)

  def update(self):
    if not self.updated: self.grad = self.compute_grad()
    self.updated = True

  def backprop(self):
    # for pylance
    self.child: None | Node

    p_parent: Node | None = self.parent[0]
    s_parent: Node | None = self.parent[1]
    child: None | Node = self.child
    
    if child == None:
      partner = None if not p_parent.updated else -1
    else:
      partner: Node| None = child.parent[1]

    print(self.name, self.node_type)
    if self.node_type == 'p' and p_parent == None:
      self.update()
      if partner == None or partner.is_constant:
        # move to child
        child.backprop()
      else:
        # move to partner
        partner.backprop()
    
    elif self.node_type == 's' and p_parent == None:
      # move to child
      self.update()
      child.backprop()

    elif not p_parent.updated and not p_parent.is_constant:
      if child == None: self.grad = cupy.array([[1.]])
      elif not self.updated: self.grad = self.compute_grad()
      
      # move to p_parent
      self.updated = True
      p_parent.backprop()
      return
        
    elif self.node_type == 'p' and (p_parent.updated or p_parent.is_constant):
      self.update()
      if s_parent == None:
        p_parent.updated = False
        # move to child
        if partner == -1: return
        child.backprop()
        return
      elif s_parent.updated or s_parent.is_constant:
        # set parents' update state to defualt
        p_parent.updated = False
        s_parent.updated = False
        # move to partner
        if partner == -1:
          return
        elif partner == None or partner.is_constant: 
          child.backprop()
          return
        else: 
          partner.backprop()
          return
      print('err', self.name, s_parent.is_constant)

    elif self.node_type == 's' and (p_parent.updated or p_parent.is_constant):
      self.update()
      if s_parent == None:
        # set parent's update state to default
        p_parent.updated = False
        # perch to child
      elif s_parent.updated:
        # set parents' update state to defualt
        p_parent.updated = False
        s_parent.updated = False
      child.backprop()

    elif not s_parent.updated and not s_parent.is_constant:
      if child == None: self.grad = cupy.array([[1.]])
      elif not self.updated: self.grad = self.compute_grad()
      
      # move to p_parent
      self.updated = True
      if not p_parent.is_constant:
        p_parent.backprop()
      elif not s_parent.is_constant:
        s_parent.backprop()
      return
      
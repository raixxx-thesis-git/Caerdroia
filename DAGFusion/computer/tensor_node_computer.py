from __future__ import annotations
from typing import TYPE_CHECKING, List
from cupy import ndarray
from DAGFusion.computer import secure_type
from DAGFusion.math import ForwardMath, BackwardMath
from DAGFusion.node_structures import Triad, Dyad

if TYPE_CHECKING:
  from DAGFusion import TensorNode

import cupy

class NodeComputer(ForwardMath, BackwardMath):
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

  def complete_adic(self: TensorNode, coop: TensorNode | None, operation: str, 
                    outcome: ndarray) -> TensorNode:
    # import TensorNode module is put here to avoid a circular import.
    from DAGFusion import TensorNode

    operator = operation[:-1]
    operation_flag = operation[-1]

    outcome_node = TensorNode(outcome)

    if coop == None:
      adic = Dyad(self, outcome_node, operator)
      adic.set_prev(self.adic)
    elif operation_flag == 'r':
      adic = Triad(coop, self, outcome_node, operator)
      adic.set_prev((coop.adic, self.adic))
    elif operation_flag == '.':
      adic = Triad(self, coop, outcome_node, operator)
      adic.set_prev((self.adic, coop.adic))

    outcome_node.adic = adic

    return outcome_node


  def make_child(self, partner: TensorNode | None, operation: str, value: ndarray) -> TensorNode:
    # import TensorNode module is put here to avoid a circular import.
    from DAGFusion import TensorNode

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
    from DAGFusion import TensorNode

    if type(partner) == float:
      partner = TensorNode(partner, name=str(partner), is_constant=True)
    return partner
  
  def __add__(self, coop: TensorNode) -> TensorNode:
    coop = secure_type(coop)
    outcome = self.add(self.tensor, coop.tensor)
    return self.complete_adic(coop, '+.', outcome)

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
    value = self.reduce_sum_(self.tensor, axis)
    return self.make_child(None, 'redsum.', value)
  
  def log(self, basis: int) -> TensorNode:
    self.temp_state_log_basis = basis
    value = self.log_(self.tensor, basis)
    return self.make_child(None, f'log.', value)
  
  def abs(self) -> TensorNode:
    value = self.abs_(self.tensor)
    return self.make_child(None, f'abs.', value)
  
  def conditional_map(self, maps: List[TensorNode], conditions: List[str]) -> TensorNode:
    new_matrix = cupy.array(self.tensor)
    
    self.temp_state_conditional_func_conditions = conditions
    self.temp_state_conditional_func_maps = maps

    for map, condition in zip(maps, conditions):
      condition_ = cupy.argwhere(eval(condition.replace('X', 'self.tensor')))
      map.child = self
      new_matrix[condition_[:,0], condition_[:,1]] = map.tensor[condition_[:,0], condition_[:,1]]
    
    return self.make_child(None, f'condmap.', new_matrix)
  
  def grad_for_conditional_map(self, var: TensorNode) -> ndarray:
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
    self.child: None | TensorNode

    p_parent: TensorNode | None = self.parent[0]
    s_parent: TensorNode | None = self.parent[1]
    child: None | TensorNode = self.child
    
    if child == None:
      partner = None if not p_parent.updated else -1
    else:
      partner: TensorNode| None = child.parent[1]

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
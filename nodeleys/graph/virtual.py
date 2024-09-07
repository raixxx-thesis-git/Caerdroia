from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Tuple
from nodeleys.graph import Node
from cupy import ndarray
import cupy
import re

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet

def or_(param0: str, param1: str) -> ndarray:
  param0 = re.sub(r'(\[\d+\])', r'domain_vars\1.tensor', param0)
  param1 = re.sub(r'(\[\d+\])', r'domain_vars\1.tensor', param1)
  return f'cupy.logical_or({param0}, {param1})'

def and_(param0: str, param1: str) -> ndarray:
  param0 = re.sub(r'(\[\d+\])', r'domain_vars\1.tensor', param0)
  param1 = re.sub(r'(\[\d+\])', r'domain_vars\1.tensor', param1)
  return f'cupy.logical_and({param0}, {param1})'

class Virtual():
  def __init__(self, domain_vars: List[Node], virtual_graphs: List[Node], conditions: List[str], name: str='') -> Node:
    self.mask_ids = []

    whiteboard = cupy.zeros(shape=virtual_graphs[0].tensor.shape)
    for map, condition in zip(virtual_graphs, conditions):
      if 'cupy' not in condition:
        condition = re.sub(r'(\[\d+\])', r'domain_vars\1.tensor', condition)

      condition = eval(condition)
      indices = cupy.argwhere(condition)
      self.mask_ids.append(indices)

      replacement = map.tensor 
    
      whiteboard[indices[:,0], indices[:,1]] = replacement[indices[:,0], indices[:,1]]
    
    self.tensor = whiteboard
    self.domain_vars = domain_vars
    self.virtual_graphs = virtual_graphs
    self.prev = tuple([domain_var.adic for domain_var in domain_vars])
    self.prev_names = [domain_var.name for domain_var in domain_vars]
    self.name = name

  def compile(self: Virtual):
    new_node = Node(self.tensor, name=self.name)
    new_node.set_adic(self)
    self.o = new_node
    return new_node

  def get_prev(self) -> Tuple[Union[Duplet, Triplet, None]]:
    return self.prev

  def get_outcome(self, idx: int) -> Node:
    return self.virtual_graphs[idx]

  def __repr__(self):
    return_string = 'Virtual('
    for prev_name in self.prev_names:
      return_string += prev_name + ', '
    return_string += f'{self.name}; multi_ops)'
    return return_string

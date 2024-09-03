from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Tuple
from ensoine.graph import Node
from cupy import ndarray
import cupy
import re

if TYPE_CHECKING:
  from ensoine.graph import Duplet, Triplet

def or_(param0: str, param1: str) -> ndarray:
  param0 = re.sub(r'(\[\d+\])', r'vars\1.tensor', param0)
  param1 = re.sub(r'(\[\d+\])', r'vars\1.tensor', param1)
  return f'cupy.logical_or({param0}, {param1})'

def and_(param0: str, param1: str) -> ndarray:
  param0 = re.sub(r'(\[\d+\])', r'vars\1.tensor', param0)
  param1 = re.sub(r'(\[\d+\])', r'vars\1.tensor', param1)
  return f'cupy.logical_and({param0}, {param1})'

class Dynamic():
  def __init__(self, vars: List[Node], maps: List[Node], conditions: List[str], name: str='') -> Node:
    vars = vars

    whiteboard = cupy.zeros(shape=maps[0].tensor.shape)

    for map, condition in zip(maps, conditions):
      if 'cupy' not in condition:
        condition = re.sub(r'(\[\d+\])', r'vars\1.tensor', condition)
    
      condition = eval(condition)
      indices = cupy.argwhere(condition)
      replacement = map.tensor
    
      whiteboard[indices[:,0], indices[:,1]] = replacement[indices[:,0], indices[:,1]]
    
    self.tensor = whiteboard
    self.graphs = maps
    self.conditions = conditions
    self.prev = tuple([var.adic for var in vars])
    self.prev_names = [var.name for var in vars]
    self.name = name

  def compile(self):
    new_node = Node(self.tensor, name=self.name)
    new_node.set_adic(self)
    return new_node

  def get_prev(self) -> Tuple[Union[Duplet, Triplet, None]]:
    return self.prev

  def __repr__(self):
    return_string = 'Dynamic('
    for prev_name in self.prev_names:
      return_string += prev_name + ', '
    return_string += f'{self.name}; multi_ops)'
    return return_string

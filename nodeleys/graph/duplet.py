from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Optional, List
from nodeleys.system import DupletBackpropSystem
from cupy import ndarray

if TYPE_CHECKING:
  from nodeleys import Node
  from nodeleys.graph import Triplet

class Duplet(DupletBackpropSystem):
  def __init__(self, operand: Node, outcome: Node, operator: str) -> None:
    self.l = operand
    self.o = outcome
    self.operator = operator

  def set_next(self, next_adic: Union[Triplet, Duplet]) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adic: Optional[Union[Triplet, Duplet]]) -> None:
    self.prev = prev_adic

  def operand_add_gradient(self, grad_L: ndarray) -> None:
    self.l.add_gradient(grad_L)

  def operand_add_virtual_gradient(self, grad_L: ndarray, idx: int) -> None:
    self.l.add_virtual_gradient(grad_L, idx)

  def get_prev(self) -> Optional[Union[Duplet, Triplet]]:
    return self.prev

  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> Node:
    return self.o
  
  def get_operand(self) -> Node:
    return self.l
  
  def __repr__(self) -> str:
    return f'Duplet({self.l.name}, {self.o.name}; {self.operator})'
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Optional, List
from cupy import ndarray
from nodeleys.system import TripletBackpropSystem

if TYPE_CHECKING:
  from nodeleys import Node
  from nodeleys.graph import Duplet

class Triplet(TripletBackpropSystem):
  def __init__(self, l_operand: Node, r_operand: Node, 
               outcome: Node, operator: str) -> None:
    self.l = l_operand
    self.r = r_operand
    self.o = outcome
    self.operator = operator
    self.touched = False

  def set_next(self, next_adic: Union[Triplet, Duplet]) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adics: Tuple[Optional[Union[Triplet, Duplet]], ...]) -> None:
    self.prev = prev_adics

  def operands_add_gradient(self, grad_L: ndarray, grad_R: ndarray) -> None:
    self.l.add_gradient(grad_L)
    self.r.add_gradient(grad_R)
  
  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> Node:
    return self.o
  
  def get_prev(self) -> Tuple[Optional[Union[Duplet, Triplet]], ...]:
    return self.prev
  
  def get_operands(self) -> Tuple[Node, Node]:
    return (self.l, self.r)

  def __repr__(self) -> str:
    return f'Triplet({self.l.name}, {self.r.name}, {self.o.name}; {self.operator})'
    
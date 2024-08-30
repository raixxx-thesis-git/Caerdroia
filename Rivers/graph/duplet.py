from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Optional

if TYPE_CHECKING:
  from Rivers import Node
  from Rivers.graph import Triplet

class Duplet():
  def __init__(self, operand: Node, outcome: Node, operator: str) -> None:
    self.l = operand
    self.o = outcome
    self.operator = operator

  def set_next(self, next_adic: Union[Triplet, Duplet]) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adic: Optional[Union[Triplet, Duplet]]) -> None:
    self.prev = prev_adic

  def get_prev(self) -> Optional[Union[Triplet, Duplet]]:
    return self.prev
  
  def get_next(self) -> Union[Triplet, Duplet]:
    return self.next
  
  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> Node:
    return self.o
  
  def __repr__(self) -> str:
    return f'Duplet({self.l.name}, {self.o.name}; {self.operator})'
  
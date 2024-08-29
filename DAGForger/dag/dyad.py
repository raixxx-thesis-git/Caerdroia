from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from DAGForger import DAGNode
  from DAGForger.dag import Triad

class Dyad():
  def __init__(self, operand: DAGNode, outcome: DAGNode, operator: str) -> None:
    self.l = operand
    self.o = outcome
    self.operator = operator

  def set_next(self, next_adic: Triad|Dyad) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adic: Triad|Dyad) -> None:
    self.prev = prev_adic

  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> DAGNode:
    return self.o
  
  def __repr__(self) -> str:
    return f'Dyad({self.l.name}, {self.o.name}; {self.operator})'
  
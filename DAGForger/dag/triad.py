from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Optional, List
from copy import deepcopy
if TYPE_CHECKING:
  from DAGForger import DAGNode
  from DAGForger.dag import Dyad

class Triad():
  def __init__(self, l_operand: DAGNode, r_operand: DAGNode, 
               outcome: DAGNode, operator: str) -> None:
    self.l = l_operand
    self.r = r_operand
    self.o = outcome
    self.operator = operator
    self.touched = False

  def set_next(self, next_adic: Union[Triad, Dyad]) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adics: Tuple[Optional[Union[Triad, Dyad]]]) -> None:
    self.prev = prev_adics

  # def get_prev(self) -> Tuple[Optional[Union[Triad, Dyad]]]:
  #   return self.prev
  
  # def get_next(self) -> Union[Triad, Dyad]:
  #   return self.next
  
  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> DAGNode:
    return self.o
  
  def __repr__(self) -> str:
    return f'Triad({self.l.name}, {self.r.name}, {self.o.name}; {self.operator})'
  
  def propagate(self, passed_adics: List[Union[Triad, Dyad]]=[], bonds: List[Tuple[Union[Triad, Dyad]]]=[], 
                checkpoints: List[Optional[Union[Triad, Dyad]]]=[], from_jump: bool=False):
    passed_adics = deepcopy(passed_adics)
    checkpoints = deepcopy(checkpoints)
    bonds = deepcopy(bonds)

    passed_adics.append(self)
    print(self, self.prev, from_jump, checkpoints)
  
    if self.prev[1] != None and self.prev[0] != None and from_jump == False:
      checkpoints.append(self)
    
    if self.prev[0] == None and self.prev[1] == None and len(checkpoints) != 0:
      last_checkpoint = checkpoints.pop()
      return last_checkpoint.propagate(passed_adics, bonds, checkpoints, True)
    
    if not from_jump:
      if self.prev[0] == None and self.prev[1] == None:
        return passed_adics, bonds
      elif self.prev[0] == None and self.prev[1] != None:
        new_bond = (self, self.prev[1])
        if new_bond not in bonds: bonds.append(new_bond)
        return self.prev[1].propagate(passed_adics, bonds, checkpoints, False)
      elif self.prev[0] != None:
        new_bond = (self, self.prev[0])
        if new_bond not in bonds: bonds.append(new_bond)
        return self.prev[0].propagate(passed_adics, bonds, checkpoints, False)
      
    else:
      new_bond = (self, self.prev[1])
      if new_bond not in bonds: bonds.append(new_bond)
      return self.prev[1].propagate(passed_adics, bonds, checkpoints, False)
    
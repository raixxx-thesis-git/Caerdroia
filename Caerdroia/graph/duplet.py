from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Optional, List

if TYPE_CHECKING:
  from Caerdroia import Node
  from Caerdroia.graph import Triplet

class Duplet():
  def __init__(self, operand: Node, outcome: Node, operator: str) -> None:
    self.l = operand
    self.o = outcome
    self.operator = operator

  def set_next(self, next_adic: Union[Triplet, Duplet]) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adic: Optional[Union[Triplet, Duplet]]) -> None:
    self.prev = prev_adic

  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> Node:
    return self.o
  
  def end_duplet(self) -> bool:
    '''
    is true as long as the current duplet is D[n,...] and D[n+1,...] is None.
    '...' is optional.
    '''
    return self.prev == None
  
  def __repr__(self) -> str:
    return f'Duplet({self.l.name}, {self.o.name}; {self.operator})'
  
  def propagate(self, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False):
    passed_adics.append(self)
    print(self)
    if not self.end_duplet():
      '''
      This happens if the current adic D[n,...] is connected to D[n+1,...]. We move
      D[n,...] to D[n+1,...] as long as D[n+1,...] is not None.
      '''
      bonds.append((self,self.prev))
      return self.prev.propagate(passed_adics, bonds, checkpoints, False)

    elif self.end_duplet():
      if len(checkpoints) != 0:
        '''
        This happens if the current adic is D[n,a,...]. We move D[n,a,...] to
        T[a,...].
        '''
        leap_to = checkpoints.pop()
        return leap_to.propagate(passed_adics, bonds, checkpoints, True)
      
      '''
      This is executed if the current adic is D[n] and D[n+1,...] is None.
      '''
      return passed_adics, bonds
    pass
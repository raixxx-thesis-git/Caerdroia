from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Optional, List
from copy import deepcopy
if TYPE_CHECKING:
  from Caerdroia import Node
  from Caerdroia.graph import Duplet

class Triplet():
  def __init__(self, l_operand: Node, r_operand: Node, 
               outcome: Node, operator: str) -> None:
    self.l = l_operand
    self.r = r_operand
    self.o = outcome
    self.operator = operator
    self.touched = False

  def set_next(self, next_adic: Union[Triplet, Duplet]) -> None:
    self.next = next_adic
  
  def set_prev(self, prev_adics: Tuple[Optional[Union[Triplet, Duplet]]]) -> None:
    self.prev = prev_adics
  
  def get_operator(self) -> str:
    return self.operator
  
  def get_outcome(self) -> Node:
    return self.o
  
  def complete_triplet(self) -> bool:
    ''' 
    when T[n,...] is connected to T[n+1,n,...] and T[n+1,...]. 
    '''
    return self.prev[0] != None and self.prev[1] != None
  
  def end_triplet(self) -> bool:
    '''
    when T[n,...] is connected to T[n+1,n,...] and T[n+1,...] but T[n+1,n,...] 
    and T[n+1,...] are None. '...' is optional.
    '''
    return self.prev[0] == None and self.prev[1] == None

  
  def __repr__(self) -> str:
    return f'Triplet({self.l.name}, {self.r.name}, {self.o.name}; {self.operator})'
  
  def propagate(self, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False):
    passed_adics.append(self)
  
    if self.complete_triplet() and from_leap == False:
      '''
      add T[n,...] as a checkpoint if T[n,...] is a complete triplet and is coming from
      by-turns propagation (not from leap propagation).
      '''
      checkpoints.append(self)
    
    if self.end_triplet() and len(checkpoints) != 0:
      '''
      if current T is T[n,a,...], leap to T[a,...]
      '''
      leap_to = checkpoints.pop()
      return leap_to.propagate(passed_adics, bonds, checkpoints, True)
    
    if not from_leap:
      if self.end_triplet():
        '''
        by reaching this point means that the adic is T[n]. T[n+1,n] and T[n+1]
        are None.
        '''
        return passed_adics, bonds
  
      elif self.prev[0] == None and self.prev[1] != None:
        '''
        This happens if T[n,...] = T(x, y, z, op) where x does not inherit T/D while
        y inherits T/D. We move T[n,...] to T[n+1,...].
        '''
        new_bond = (self, self.prev[1])
        if new_bond not in bonds:
          bonds.append(new_bond)
        return self.prev[1].propagate(passed_adics, bonds, checkpoints, False)
      
      elif self.prev[0] != None:
        '''
        This happens if T[n,...] = T(x, y, z, op) either where y does or does not inherit 
        T/D while x inherits T/D. If y does inherit, we move T[n,...] to T[n,n+1,...]. Otherwise, we move
        T[n,...] to T[n+1,...].
        '''
        new_bond = (self, self.prev[0])
        if new_bond not in bonds:
          bonds.append(new_bond)
        return self.prev[0].propagate(passed_adics, bonds, checkpoints, False)
      
    else:
      '''
      When this condition is true, T[n,...] must be T(x, y, z, op) and x and y inherit T/D. Since
      T[n,...] is connected to T[n,n+1,...] and T[n+1,...], we move to T[n+1,...]
      as T[n,n+1,...] has been passed.
      '''
      new_bond = (self, self.prev[1])
      if new_bond not in bonds:
        bonds.append(new_bond)
      return self.prev[1].propagate(passed_adics, bonds, checkpoints, False)
    
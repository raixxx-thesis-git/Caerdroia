from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple, Optional, Set
from nodeleys.system import compute_grad

import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet

class TripletBackpropSystem():
  def __init__(self): pass

  def complete_triplet(self) -> bool:
    ''' 
    when T[n,...] is connected to T[n+1,n,...] and T[n+1,...]. 
    '''
    self: Triplet
    return self.prev[0] != None and self.prev[1] != None
  
  def end_triplet(self) -> bool:
    '''
    when T[n,...] is connected to T[n+1,n,...] and T[n+1,...] but T[n+1,n,...] 
    and T[n+1,...] are None or interrupt(s). '...' is optional.
    '''
    self: Triplet
    prev0_is_none = self.prev[0] == None
    prev1_is_none = self.prev[1] == None

    return prev0_is_none and prev1_is_none
  
  def is_an_interrupt(self: Triplet, interrupts: Set[Union[Duplet, Triplet, None]]):
    print(self in interrupts)
    return self in interrupts
  
  def propagate(self: Triplet, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False,
                interrupts: Set[Union[Duplet, Triplet, None]]={}):
    passed_adics.append(self)

    self_is_an_interrupt = self.is_an_interrupt(interrupts)
    
    if not from_leap and not self_is_an_interrupt: 
      compute_grad(self)

    is_end = self.end_triplet() or self_is_an_interrupt
  
    if self.complete_triplet() and from_leap == False:
      '''
      add T[n,...] as a checkpoint if T[n,...] is a complete triplet and is coming from
      by-turns propagation (not from leap propagation).
      '''
      checkpoints.append(self)
    
    if is_end and len(checkpoints) != 0:
      '''
      if current T is T[n,a,...], leap to T[a,...]
      '''
      leap_to = checkpoints.pop()
      return leap_to.propagate(passed_adics, bonds, checkpoints, True, interrupts=interrupts)
    
    if not from_leap:
      if is_end:
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
        return self.prev[1].propagate(passed_adics, bonds, checkpoints, False, interrupts=interrupts)
      
      elif self.prev[0] != None:
        '''
        This happens if T[n,...] = T(x, y, z, op) either where y does or does not inherit 
        T/D while x inherits T/D. If y does inherit, we move T[n,...] to T[n,n+1,...]. Otherwise, we move
        T[n,...] to T[n+1,...].
        '''
        new_bond = (self, self.prev[0])
        if new_bond not in bonds:
          bonds.append(new_bond)
        return self.prev[0].propagate(passed_adics, bonds, checkpoints, False, interrupts=interrupts)
      
    else:
      '''
      When this condition is true, T[n,...] must be T(x, y, z, op) and x and y inherit T/D. Since
      T[n,...] is connected to T[n,n+1,...] and T[n+1,...], we move to T[n+1,...]
      as T[n,n+1,...] has been passed.
      '''
      new_bond = (self, self.prev[1])
      if new_bond not in bonds:
        bonds.append(new_bond)
      return self.prev[1].propagate(passed_adics, bonds, checkpoints, False, interrupts=interrupts)
    
  def set_as_objective(self: Triplet):
    self.get_outcome().add_gradient(cupy.array([[1.]]))

  def begin_backprop(self, interrupts: List[Union[Duplet, Triplet, None]]=[]):
    return self.propagate([], [], [], interrupts=interrupts)
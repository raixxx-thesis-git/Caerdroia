from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple, Optional
from Caerdroia.system import compute_grad

import cupy

if TYPE_CHECKING:
  from Caerdroia.graph import Duplet, Triplet

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
    and T[n+1,...] are None. '...' is optional.
    '''
    self: Triplet
    return self.prev[0] == None and self.prev[1] == None
  
  def propagate(self, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False):
    self: Triplet
    passed_adics.append(self)

    if not from_leap: compute_grad(self)
  
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
    
  def set_as_objective(self: Triplet):
    self.get_outcome().add_gradient(cupy.array([[1.]]))

  def begin_backprop(self):
    return self.propagate([], [], [])
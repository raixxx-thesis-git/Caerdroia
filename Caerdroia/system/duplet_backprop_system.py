from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Tuple, Optional
from Caerdroia.system import compute_grad
import cupy

if TYPE_CHECKING:
  from Caerdroia.graph import Duplet, Triplet

class DupletBackpropSystem:
  def __init__(self): pass
  
  def end_duplet(self) -> bool:
    '''
    is true as long as the current duplet is D[n,...] and D[n+1,...] is None.
    '...' is optional.
    '''
    self: Duplet
    return self.prev == None
  
  def propagate(self, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False):
    self: Duplet
    passed_adics.append(self)

    if not from_leap:
      compute_grad(self)

    if not self.end_duplet():
      '''
      This happens if the current adic D[n,...] is connected to D[n+1,...]. We move
      D[n,...] to D[n+1,...] as long as D[n+1,...] is not None.
      '''
      new_bond = (self,self.prev)
      if new_bond not in bonds:
        bonds.append(new_bond)
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
      This is executed if the current adic is D[n] and D[n+1] is None.
      '''
      return passed_adics, bonds
    pass

  def set_as_objective(self: Triplet):
    self.get_outcome().add_gradient(cupy.array([[1.]]))

  def begin_backprop(self):
    return self.propagate([], [], [])
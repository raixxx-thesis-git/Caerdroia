from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Tuple, Optional, Set
from nodeleys.system import compute_grad
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet, Node

class DupletBackpropSystem():
  def __init__(self): pass
  
  def end_duplet(self: Duplet) -> bool:
    '''
    is true as long as the current duplet is D[n,...] and D[n+1,...] is None or
    an interrupt. '...' is optional.
    '''
    return self.prev == None
  
  def propagate(self: Duplet, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False,
                interrupts: Set[Union[Duplet, Triplet, None]]={},
                is_virtually: bool=False,
                idx: int=-1,
                trainable_nodes: Union[Node]=[],
                tracing: bool=False):
    # This will be used to pass down the kwargs to the other propagates
    kwargs = locals()
    del kwargs['self']

    # Tracking flow
    passed_adics.append(self)
    
    # To check whether the operands is a trainable or not. If it's trainable,
    # then register it to the tracked trainable nodes list.
    operand = self.get_operand()
    if operand.get_adic() == None and operand.is_trainable: trainable_nodes.append(operand)

    # When using a Switch, the interrupts are assigned. Hence the process
    # must regard these interrutps.
    self_is_an_interrupt = self.get_outcome() in interrupts

    if (not from_leap and not self_is_an_interrupt) and (not tracing):
      compute_grad(self, is_virtually, idx)

    is_end = self.end_duplet() or self_is_an_interrupt

    if not is_end:
      '''
      This happens if the current adic D[n,...] is connected to D[n+1,...]. We move
      D[n,...] to D[n+1,...] as long as D[n+1,...] is not None.
      '''
      new_bond = (self,self.prev)
      if new_bond not in bonds:
        bonds.append(new_bond)
      if self.prev.get_adic_type() == 'Switch': 
        return self.prev.propagate(passed_adics=passed_adics,
                                   bonds=bonds,
                                   checkpoints=checkpoints,
                                   from_leap=from_leap,
                                   tracing=tracing,
                                   trainable_nodes=trainable_nodes)
      kwargs['from_leap'] = False
      return self.prev.propagate(**kwargs)

    elif is_end:
      if len(checkpoints) != 0:
        '''
        This happens if the current adic is D[n,a,...]. We move D[n,a,...] to
        T[a,...].
        '''
        leap_to = checkpoints.pop()
        kwargs['from_leap'] = False
        return self.prev.propagate(**kwargs)
      
      '''
      This is executed if the current adic is D[n] and D[n+1] is None.
      '''
      return passed_adics, bonds
    pass

  def set_as_objective(self: Triplet):
    self.get_outcome().add_gradient(cupy.array([[1.]]))

  def begin_backprop(self, tracing: bool=False, traces=[]):
    return self.propagate([], [], [], tracing=tracing, trainable_nodes=traces)
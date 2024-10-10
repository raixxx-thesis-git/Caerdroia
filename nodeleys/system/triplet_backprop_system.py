from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple, Optional, Set
from nodeleys.system import compute_grad

import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet, Node

class TripletBackpropSystem():
  def __init__(self): pass

  def complete_triplet(self) -> bool:
    # When T[n,...] is connected to T[n+1,n,...] and T[n+1,...]. 

    self: Triplet
    return self.prev[0] != None and self.prev[1] != None
  
  def end_triplet(self) -> bool:
    # When T[n,...] is connected to T[n+1,n,...] and T[n+1,...] but T[n+1,n,...] 
    # and T[n+1,...] are None or interrupt(s). '...' is optional.

    self: Triplet
    prev0_is_none = self.prev[0] == None
    prev1_is_none = self.prev[1] == None

    return prev0_is_none and prev1_is_none
  
  def propagate(self: Triplet, 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False,
                interrupts: Set[Union[Duplet, Triplet, None]]={},
                is_virtually: bool=False,
                idx: int=-1,
                trainable_nodes: Union[Node]=[],
                tracing=False):
    # This will be used to pass down the kwargs to the other propagates
    kwargs = locals()
    del kwargs['self']

    # Tracking flow
    passed_adics.append(self)

    # To check whether the operands is a trainable or not. If it's trainable,
    # then register it to the tracked trainable nodes list.
    operands = self.get_operands()
    if operands[0].get_adic() == None and operands[0].is_trainable: trainable_nodes.append(operands[0])
    if operands[1].get_adic() == None and operands[1].is_trainable: trainable_nodes.append(operands[1])

    # When using a Switch, the interrupts are assigned. Hence the process
    # must regard these interrutps.
    self_is_an_interrupt = self.get_outcome() in interrupts
    
    if (not from_leap and not self_is_an_interrupt) and (not tracing): 
      compute_grad(self, is_virtually, idx)

    # Reaching the end of the node or an interrupt if using a Switch.
    is_end = self.end_triplet() or self_is_an_interrupt
  
    if self.complete_triplet() and from_leap == False:
      # Add T[n,...] as a checkpoint if T[n,...] is a complete triplet and is coming from
      # by-turns propagation (not from leap propagation).
      checkpoints.append(self)
    
    if is_end and len(checkpoints) != 0:
      # if current T is T[n,a,...], leap to T[a,...]
      leap_to = checkpoints.pop()

      kwargs['from_leap'] = True
      return leap_to.propagate(**kwargs)
    
    if not from_leap:
      if is_end:
        # By reaching this point means that the adic is T[n]. T[n+1,n] and T[n+1]
        # are None.
        return passed_adics, bonds
      
      elif self.prev[0] == None and self.prev[1] != None:
        # This happens if T[n,...] = T(x, y, z, op) where x does not inherit T/D while
        # y inherits T/D. We move T[n,...] to T[n+1,...].
        new_bond = (self, self.prev[1])
        if new_bond not in bonds:
          bonds.append(new_bond)

        if self.prev[1].get_adic_type() == 'Switch': 
          return self.prev[1].propagate(passed_adics=passed_adics,
                                        bonds=bonds,
                                        checkpoints=checkpoints,
                                        from_leap=from_leap,
                                        tracing=tracing,
                                        trainable_nodes=trainable_nodes)
        kwargs['from_leap'] = False
        return self.prev[1].propagate(**kwargs)
      
      elif self.prev[0] != None:
        # This happens if T[n,...] = T(x, y, z, op) either where y does or does not inherit 
        # T/D while x inherits T/D. If y does inherit, we move T[n,...] to T[n,n+1,...]. Otherwise, we move
        # T[n,...] to T[n+1,...].

        new_bond = (self, self.prev[0])
        if new_bond not in bonds:
          bonds.append(new_bond)
        # print(self.prev[0].get_adic_type(), is_virtually)
        if self.prev[0].get_adic_type() == 'Switch': 
          return self.prev[0].propagate(passed_adics=passed_adics,
                                        bonds=bonds,
                                        checkpoints=checkpoints,
                                        from_leap=from_leap,
                                        tracing=tracing,
                                        trainable_nodes=trainable_nodes)
        kwargs['from_leap'] = False
        return self.prev[0].propagate(**kwargs)
      
    else:
      # When this condition is true, T[n,...] must be T(x, y, z, op) and x and y inherit T/D. Since
      # T[n,...] is connected to T[n,n+1,...] and T[n+1,...], we move to T[n+1,...]
      # as T[n,n+1,...] has been passed.

      new_bond = (self, self.prev[1])
      if new_bond not in bonds:
        bonds.append(new_bond)
        
      if self.prev[1].get_adic_type() == 'Switch': 
        return self.prev[1].propagate(passed_adics=passed_adics,
                                      bonds=bonds,
                                      checkpoints=checkpoints,
                                      from_leap=from_leap,
                                      tracing=tracing,
                                      trainable_nodes=trainable_nodes)
      kwargs['from_leap'] = False
      return self.prev[1].propagate(**kwargs)
    
  def set_as_objective(self: Triplet):
    self.get_outcome().add_gradient(cupy.array([[1.]]))

  def begin_backprop(self, interrupts: List[Union[Duplet, Triplet, None]]=[], tracing: bool=False, traces=[]):
    self.propagate([], [], [], interrupts=interrupts, tracing=tracing, trainable_nodes=traces)
    return 

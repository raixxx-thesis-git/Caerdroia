from __future__ import annotations
from typing import TYPE_CHECKING, List, Union 
import cupy
import re

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet, Virtual

class VirtualBackpropSystem():
  def __init__(self): pass
  
  def gradient_masking(self: Virtual):
    for domain_var in self.domain_vars:
      gradient_whiteboard = cupy.zeros(shape=domain_var.tensor.shape)

      '''
      Each ids is linked with its condition accordingly. In this for loop,
      we are filling the gradient whiteboard with the virutal_grad_pool using 
      a masking method. The variable 'mask_ids' (mask indices) stores the indices
      upon which the condition holds.
      '''
      for idx, ids in enumerate(self.ids):
        virtual_grad_pool = domain_var.virtual_grad_pool[idx]

        # the masking process begins here
        if type(virtual_grad_pool) != type(None):
          gradient_whiteboard[ids[:,0], ids[:,1]] = domain_var.virtual_grad_pool[idx][ids[:,0], ids[:,1]]
      
      '''
      After the masking process is completed, we move the filled gradient whiteboard
      from a dependent variable into its gradient pool.
      '''
      domain_var.add_gradient(gradient_whiteboard)

  def propagate(self: Union[Virtual, VirtualBackpropSystem]):
    idx = 0
    for idx, virtual_graph in enumerate(self.virtual_graphs):
      '''
      This is a subsystem routine. We do not pass the inherited backprop states to
      simulate back propagation routines of virtual graphs (dynamic).
      Since the graphs are virtual, we do not record the back propagation traces.
      '''
      _, _ = virtual_graph.get_adic().propagate([], [], [], interrupts=self.vars, 
                                                is_virtually=True, idx=idx)

      '''
      All of the interrupt adics are updated. At this point, now we are summing all of
      the gradient and do the gradient masking on each dependent variables.
      '''
      for domain_var in self.domain_vars:
        domain_var.sum_virtual_gradient_by_session(idx)

    
    self.gradient_masking()
  pass
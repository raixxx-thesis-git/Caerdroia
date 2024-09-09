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
      # We create a whiteboard for the masking process. The whiteboard has the same
      # size as the <domain_var> shape.

      gradient_whiteboard = cupy.zeros(shape=domain_var.tensor.shape)

      for idx, mask_idx in enumerate(self.mask_ids):
        virtual_grad_pool = domain_var.virtual_grad_pool[idx]

        # The masking process begins here.
        if type(virtual_grad_pool) != type(None):
          # print(domain_var.tensor.shape)
          # print(gradient_whiteboard.shape)
          # print(domain_var.virtual_grad_pool)
          gradient_whiteboard[mask_idx[:,0], mask_idx[:,1]] = domain_var.virtual_grad_pool[idx][mask_idx[:,0], mask_idx[:,1]]
      
      # After the masking process is completed, we move the filled gradient whiteboard
      # from a dependent variable into its gradient pool. This gradient value is in equivalence
      # with the gradient of the loss function w.r.t to the <domain_var>.
      domain_var.add_gradient(gradient_whiteboard)

  def propagate(self: Union[Virtual, VirtualBackpropSystem]):
    # This is the sub-process propagation.

    for idx, virtual_graph in enumerate(self.virtual_graphs):
      # Given by:
      #     t = Virtual([a, b], [c, d], [S0, S1]).compile()
      # we have conditions <S0> and <S1> and its corresponding virtual graph <c> and <d>.
      # In this for loop, we itereate <virtual_graph> from <[S0, S1]>.

      # This line of code executes one of the virtual graph's outcomes and add the virtual gradient
      # of the previous gradient.
      virtual_graph.get_adic().get_outcome().add_virtual_gradient(self.o.get_last_gradient(), idx)

      # We update all of the gradients in the virtual graph. The iterrupts are given to draw
      # the boundary of the virtual graph.
      _, _ = virtual_graph.get_adic().propagate([], [], [], interrupts=self.domain_vars, 
                                                is_virtually=True, idx=idx)

      # All of the interrupt adics are updated at this point. Next up, we should sum all of
      # the gradient and do the gradient masking on each dependent variables.

      for domain_var in self.domain_vars:
        domain_var.sum_virtual_gradient_by_session(idx)
    
    self.gradient_masking()
  pass
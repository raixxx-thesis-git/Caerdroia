from __future__ import annotations
from typing import TYPE_CHECKING, Union
from nodeleys.math import BackwardMath
from nodeleys.math import gradients
if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet, Virtual

GRADIENT_METHODS = {
  '+': gradients.grad_for_add,
  '-': gradients.grad_for_sub,
  '/': gradients.grad_for_div,
  '*': gradients.grad_for_mul,
  '@': gradients.grad_for_matmul,
  '**': gradients.grad_for_pow,
  # 'log': self.grad_for_log,
  # 'abs': self.grad_for_abs,
  'redsum': gradients.grad_for_reduce_sum
}

def compute_grad(adic: Union[Duplet, Triplet, Virtual], is_virtually: bool=False, idx: int=-1):
  from nodeleys.graph import Duplet, Triplet

  operation = adic.get_operator()

  '''
  Suppose we have a node Nx represented by an adic A[x,...], a node Ny represnted
  by A[y,...], and there is a direct connection between these two adics, 
  A[x,...] -> A[y,...] (forward propagation perspective). To obtain the gradient for
  Nx, we require the gradient of Ny. Considering there are intermediate paths where 
  these adics were updated more than once accordingly, the chain rule obliges us to
  isolate each gradient flow from each path.

  To be more concrete, let A[z,...] has S ways to reach A[y,...] -> A[x,...]. For each
  S, the gradient flow is isolated from one to the others. These gradients are pooled 
  in a list structure. Therefore, during any S, to update Nx, we require the gradient
  of Ny accordingly. Yet since the gradient of Ny is a list, we take the last gradient.
  '''

  if not is_virtually:
    prev_grad = adic.get_outcome().get_last_gradient()
  else:
    if isinstance(adic, Virtual):
      prev_grad = adic.get_outcome(idx).get_last_virtual_gradient()
    prev_grad = adic.get_outcome().get_last_virtual_gradient()
  
  if isinstance(adic, Triplet):
    l_operand, r_operand = adic.get_operands()
    grad_L, grad_R = GRADIENT_METHODS[operation](l_operand, r_operand, prev_grad)

    if not is_virtually:
      adic.operands_add_gradient(grad_L, grad_R)
    else:
      adic.operands_add_virtual_gradient(grad_L, grad_R, idx)

  elif isinstance(adic, Duplet):
    l_operand = adic.get_operand()
    grad_L = GRADIENT_METHODS[operation](l_operand, prev_grad)
    
    if not is_virtually:
      adic.operand_add_gradient(grad_L)
    else:
      adic.operand_add_virtual_gradient(grad_L, idx)
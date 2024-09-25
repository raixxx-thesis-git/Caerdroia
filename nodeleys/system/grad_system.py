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
  'redsum': gradients.grad_for_reduce_sum,
  'flatten': gradients.grad_for_flatten,
  'conv2d': gradients.grad_for_conv2d
}

def compute_grad(adic: Union[Duplet, Triplet, Virtual], is_virtually: bool=False, idx: int=-1):
  from nodeleys.graph import Duplet, Triplet, Virtual

  operation = adic.get_operator()

  if adic.get_adic_type() == 'Switch':
    # Since the virtual adic has multiple outcomes, unlike triplet and duplet that can be
    # easily obtained through <adic.get_outcome()>, therefore the we should create a sub-process
    # where we treat the adic as a pivot. This is why the sub-process is called "virtual".
    adic.propagate()
    return

  if not is_virtually:
    prev_grad = adic.get_outcome().get_last_gradient()
  else:
    prev_grad = adic.get_outcome().get_last_virtual_gradient(idx)
  
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
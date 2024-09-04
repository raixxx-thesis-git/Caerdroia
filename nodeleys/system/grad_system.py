from __future__ import annotations
from typing import TYPE_CHECKING, Union
from nodeleys.math import BackwardMath
from nodeleys.math import gradients
if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet

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

def compute_grad(adic: Union[Duplet, Triplet]):
  from nodeleys.graph import Duplet, Triplet

  operation = adic.get_operator()
  prev_grad = adic.get_outcome().get_last_gradient()
  
  if isinstance(adic, Triplet):
    l_operand, r_operand = adic.get_operands()
    grad_L, grad_R = GRADIENT_METHODS[operation](l_operand, r_operand, prev_grad)
    adic.operands_add_gradient(grad_L, grad_R)

  elif isinstance(adic, Duplet):
    l_operand = adic.get_operand()
    grad_L = GRADIENT_METHODS[operation](l_operand, prev_grad)
    adic.operand_add_gradient(grad_L)

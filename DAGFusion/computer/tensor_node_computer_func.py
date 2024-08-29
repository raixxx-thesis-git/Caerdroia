from __future__ import annotations
from DAGFusion.math import ForwardMath
from DAGFusion.node_structures import Triad, Dyad
from DAGFusion.computer import secure_type
from typing import TYPE_CHECKING
from cupy import ndarray

import operator
import cupy

if TYPE_CHECKING:
  from DAGFusion import TensorNode

def complete_adic_func(l_operand: TensorNode, r_operand: TensorNode|None, operator: str, outcome: ndarray, name: str):
  from DAGFusion import TensorNode
  outcome_node = TensorNode(outcome, name=name)

  if r_operand == None:
    adic = Dyad(l_operand, outcome_node, operator)
    adic.set_prev(l_operand.adic)
  else:
    adic = Triad(l_operand, r_operand, outcome_node, operator)
    adic.set_prev((l_operand.adic, r_operand.adic))
  
  outcome_node.adic = adic

  return outcome_node

def node_add(l_operand: TensorNode|float, r_operand: TensorNode|float, name: str) -> TensorNode:
  l_operand = secure_type(l_operand)
  r_operand = secure_type(r_operand)
  outcome = operator.add(l_operand.tensor, r_operand. tensor)
  return complete_adic_func(l_operand, r_operand, '+', outcome, name)

def node_ln(l_operand: TensorNode|float, name: str) -> TensorNode:
  l_operand = secure_type(l_operand)
  outcome = cupy.log(l_operand.tensor)
  return complete_adic_func(l_operand, None, 'ln', outcome, name)
from __future__ import annotations
from DAGForger.math import ForwardMath
from DAGForger.dag import Triad, Dyad
from DAGForger.forger import secure_type
from typing import TYPE_CHECKING
from cupy import ndarray

import operator
import cupy

if TYPE_CHECKING:
  from DAGForger import DAGNode

def complete_adic_func(l_operand: DAGNode, r_operand: DAGNode|None, operator: str, outcome: ndarray, name: str):
  from DAGForger import DAGNode
  outcome_node = DAGNode(outcome, name=name)

  if r_operand == None:
    adic = Dyad(l_operand, outcome_node, operator)
    adic.set_prev(l_operand.adic)
  else:
    adic = Triad(l_operand, r_operand, outcome_node, operator)
    adic.set_prev((l_operand.adic, r_operand.adic))
  
  outcome_node.adic = adic
  return outcome_node

def node_add(l_operand: DAGNode|float, r_operand: DAGNode|float, name: str) -> DAGNode:
  l_operand = secure_type(l_operand)
  r_operand = secure_type(r_operand)
  outcome = operator.add(l_operand.tensor, r_operand. tensor)
  return complete_adic_func(l_operand, r_operand, '+', outcome, name)

def node_ln(l_operand: DAGNode|float, name: str) -> DAGNode:
  l_operand = secure_type(l_operand)
  outcome = cupy.log(l_operand.tensor)
  return complete_adic_func(l_operand, None, 'ln', outcome, name)
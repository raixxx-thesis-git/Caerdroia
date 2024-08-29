from __future__ import annotations
from DAGForger.math import ForwardMath
from DAGForger.dag import Triad, Dyad
from DAGForger.forger import secure_type
from typing import TYPE_CHECKING, Optional, Union
from cupy import ndarray

import operator
import cupy

if TYPE_CHECKING:
  from DAGForger import DAGNode

def complete_adic_func(l_operand: DAGNode, r_operand: Optional[DAGNode], 
                       operator: str, outcome: ndarray, name: str):
  from DAGForger import DAGNode
  outcome_node = DAGNode(outcome, name=name)

  if r_operand == None:
    # create a dyad D[n](L, O; op)
    adic = Dyad(l_operand, outcome_node, operator)

    # connect D[n] with D[n-1]
    adic.set_prev(l_operand.adic)

    if l_operand.adic != None:
      # connect D[n-1] with D[n] if D[n-1] exists
      l_operand.adic.set_next(adic)
  else:
    # create a dyad T[n](L, O; op)
    adic = Triad(l_operand, r_operand, outcome_node, operator)

    # connect T[n] with T0[n-1] and T1[n-1]
    adic.set_prev((l_operand.adic, r_operand.adic))

    if l_operand.adic != None:
      # connect T0[n-1] with T[n] if T0[n-1] exists
      l_operand.adic.set_next(adic)
    elif r_operand.adic != None:
      # connect T1[n-1] with T[n] if T1[n-1] exists
      r_operand.adic.set_next(adic)
  
  outcome_node.adic = adic
  return outcome_node

def node_add(l_operand: Union[DAGNode, float], r_operand: Union[DAGNode, float], name: str) -> DAGNode:
  l_operand = secure_type(l_operand)
  r_operand = secure_type(r_operand)
  outcome = operator.add(l_operand.tensor, r_operand. tensor)
  return complete_adic_func(l_operand, r_operand, '+', outcome, name)

def node_ln(l_operand: Union[DAGNode, float], name: str) -> DAGNode:
  l_operand = secure_type(l_operand)
  outcome = cupy.log(l_operand.tensor)
  return complete_adic_func(l_operand, None, 'ln', outcome, name)


from __future__ import annotations
from Rivers.math import ForwardMath
from Rivers.graph import Triad, Duplet
from Rivers.system import secure_type
from typing import TYPE_CHECKING, Optional, Union
from cupy import ndarray

import operator
import cupy

if TYPE_CHECKING:
  from Rivers import Node

def complete_adic_func(l_operand: Node, r_operand: Optional[Node], 
                       operator: str, outcome: ndarray, name: str):
  from Rivers import Node
  outcome_node = Node(outcome, name=name)

  if r_operand == None:
    # create a Duplet D[n](L, O; op)
    adic = Duplet(l_operand, outcome_node, operator)

    # connect D[n] with D[n-1]
    adic.set_prev(l_operand.adic)

    if l_operand.adic != None:
      # connect D[n-1] with D[n] if D[n-1] exists
      l_operand.adic.set_next(adic)
  else:
    # create a Duplet T[n](L, O; op)
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

def node_add(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str) -> Node:
  l_operand = secure_type(l_operand)
  r_operand = secure_type(r_operand)
  outcome = operator.add(l_operand.tensor, r_operand. tensor)
  return complete_adic_func(l_operand, r_operand, '+', outcome, name)

def node_ln(l_operand: Union[Node, float], name: str) -> Node:
  l_operand = secure_type(l_operand)
  outcome = cupy.log(l_operand.tensor)
  return complete_adic_func(l_operand, None, 'ln', outcome, name)


from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from DAGForger import DAGNode

def secure_type(node: DAGNode | float) -> DAGNode:
  from DAGForger import DAGNode
  if type(node) == float:
    node = DAGNode(node, name=str(node), is_constant=True)
  return node
  
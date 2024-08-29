from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from DAGFusion import TensorNode

def secure_type(node: TensorNode | float) -> TensorNode:
  from DAGFusion import TensorNode
  if type(node) == float:
    node = TensorNode(node, name=str(node), is_constant=True)
  return node
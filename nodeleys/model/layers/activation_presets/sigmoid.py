from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.graph import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.layers import LayerBase

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Sigmoid(LayerBase):
  def __init__(self, name: str=''):
    super().__init__(name=name, no_register=True)

  def call(self, x: Node) -> Node:
    return node_div(1., node_add(1., x), self.name)
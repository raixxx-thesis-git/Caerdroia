from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.graph import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.layers import LayerBase

if TYPE_CHECKING:
  from nodeleys.graph import Node

class ReLU(LayerBase):
  def __init__(self, slope: int=1.0, name: str=''):
    super().__init__(name=name, no_register=True)
    self.slope = slope

  def call(self, x: Node) -> Node:
    return node_relu(x, name=self.name, slope=self.slope)
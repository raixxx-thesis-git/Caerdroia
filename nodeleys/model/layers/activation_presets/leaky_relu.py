from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.graph import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.layers import LayerBase

if TYPE_CHECKING:
  from nodeleys.graph import Node

class LeakyReLU(LayerBase):
  def __init__(self, slope_minval: int=0.01, slope_posval: int=1.0, name: str=''):
    super().__init__(name=name, no_register=True)
    self.slope_minval = slope_minval
    self.slope_posval = slope_posval
  
  def call(self, x: Node) -> Node:
    return node_leaky_relu(x, self.slope_minval, self.slope_posval, self.name)
from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.graph import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.layers import LayerBase

if TYPE_CHECKING:
  from nodeleys.graph import Node

class ReLU(LayerBase):
  def __init__(self, slope: int=1.0, name: str=''):
    super().__init__(name=name, nobuild=True)
    self.slope = slope

  def call(self, x: Node) -> Node:
    return node_relu(x, name=self.name, slope=self.slope)
  
class LeakyReLU(LayerBase):
  def __init__(self, slope_minval: int=0.01, slope_posval: int=1.0, name: str=''):
    super().__init__(name=name, nobuild=True)
    self.slope_minval = slope_minval
    self.slope_posval = slope_posval
  
  def call(self, x: Node) -> Node:
    return node_leaky_relu(x, self.slope_minval, self.slope_posval, self.name)
  
class Sigmoid(LayerBase):
  def __init__(self, name: str=''):
    super().__init__(name=name, nobuild=True)

  def call(self, x: Node) -> Node:
    return node_div(1., node_add(1., x), self.name)
  
class Softmax(LayerBase):
  def __init__(self, name: str=''):
    super().__init__(name=name, nobuild=True)
    self.name = name

  def call(self, x: Node) -> Node:
    # denominator = Node(node_redsum(x, 1).tensor) # severs connection
    return node_div(x, node_redsum(x, axis=1))
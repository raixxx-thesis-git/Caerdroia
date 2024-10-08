from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.graph import Node
from nodeleys.math.forward_math_func import *

if TYPE_CHECKING:
  from nodeleys.graph import Node

class ReLU():
  def __init__(self, slope: int=1.0, name: str=''):
    self.name = name
    self.slope = slope

  def __call__(self, x: Node):
    return node_relu(x, name=self.name, slope=self.slope)
  
class LeakyReLU():
  def __init__(self, slope_minval: int=0.01, slope_posval: int=1.0, name: str=''):
    self.name = name
    self.slope_minval = slope_minval
    self.slope_posval = slope_posval
  
  def __call__(self, x: Node):
    return node_leaky_relu(x, self.slope_minval, self.slope_posval, self.name)
  
class Sigmoid():
  def __init__(self, name: str=''):
    self.name = name

  def __call__(self, x: Node) -> Node:
    return node_div(1., node_add(1., x), self.name)
  
class Softmax():
  def __init__(self, name: str=''):
    self.name = name

  def __call__(self, x: Node) -> Node:
    return node_div(x, node_redsum(x, 1), self.name)
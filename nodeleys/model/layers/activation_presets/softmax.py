from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.graph import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.layers import LayerBase
import math

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Softmax(LayerBase):
  def __init__(self, name: str=''):
    super().__init__(name=name, no_register=True)
    self.name = name

  def call(self, x: Node) -> Node:
    numerator = node_pow(math.e, x)
    denominator = node_redsum(numerator, axis=1)
    return node_div(numerator, denominator)
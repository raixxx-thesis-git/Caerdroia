from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.initializer_presets import *
from nodeleys.model.layers import LayerBase
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Flatten(LayerBase):
  def __init__(self, name: str=''):
    super().__init__(name=name, no_register=True)

  def call(self, tensor_in: Node):
    return node_flatten(tensor_in, self.name)
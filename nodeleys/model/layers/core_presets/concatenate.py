from __future__ import annotations
from typing import TYPE_CHECKING, List
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.initializer_presets import *
from nodeleys.model.layers import LayerBase
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Concatenate(LayerBase):
  def __init__(self, axis: int, name: str=''):
    super().__init__(name=name, no_register=True)
    self.axis = axis

  def call(self, tensor_in: List[Node]):
    total_tensor = len(tensor_in)
    x = node_concat(tensor_in[0], tensor_in[1], axis=self.axis)

    for i in range(2, total_tensor):
      x = node_concat(x, tensor_in[i], axis=self.axis)

    return x
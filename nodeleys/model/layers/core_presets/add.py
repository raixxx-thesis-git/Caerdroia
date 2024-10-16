from __future__ import annotations
from typing import TYPE_CHECKING, List
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.initializer_presets import *
from nodeleys.model.layers import LayerBase
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Add(LayerBase):
  def __init__(self, name: str=''):
    super().__init__(name=name, no_register=True)

  def call(self, tensor_in: List[Node]):
    total_tensor = len(tensor_in)
    x = node_add(tensor_in[0], tensor_in[1])

    for i in range(2, total_tensor):
      x = node_add(x, tensor_in[i])
    return x
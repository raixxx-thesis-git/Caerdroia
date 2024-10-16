from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.initializer_presets import *
from nodeleys.model.layers import LayerBase
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class MaxPool2D(LayerBase):
  def __init__(self, pool_size: Tuple[int,int], strides: Tuple[int,int]=(1,1), 
               name: str=''):
    super().__init__(name=name, no_register=True)
    self.pool_size = pool_size
    self.strides = strides

  def call(self, tensor_in: Node):
    return node_maxpool2d(tensor_in, self.pool_size, self.strides, self.name)
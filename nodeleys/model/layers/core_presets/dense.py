from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.math.initializers import *
from nodeleys.model.layers import LayerBase
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Dense(LayerBase):
  def __init__(self, units: int, name: str='', initializers=XavierUniform()):
    super().__init__(name=name, initializers=initializers)
    self.units = units

  def register(self, tensor_in: Node):
    inter_shape = tensor_in.tensor.shape[-1]
    self.weights = Node(tensor=self.initializers((inter_shape, self.units)), 
                        name=f'weights-{self.name}',
                        is_trainable=True)

  def call(self, tensor_in: Node):
    return node_matmul(tensor_in, self.weights)
from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.math.initializers import *

import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Dense():
  def __init__(self, units: int, name: str='', initializers=XavierUniform()):
    self.name = name
    self.units = units
    self.initializers = initializers
    self.weights = None

  def build(self, tensor_in: Node):
    inter_shape = tensor_in.tensor.shape[-1]
    self.weights = Node(self.initializers((inter_shape, self.units)), 
                        name=f'weights-{self.name}',
                        is_trainable=True)
    
  def get_weights(self) -> Node:
    return self.weights

  def __call__(self, tensor_in: Node):
    if self.weights == None: self.build(tensor_in)
    return node_matmul(tensor_in, self.weights)
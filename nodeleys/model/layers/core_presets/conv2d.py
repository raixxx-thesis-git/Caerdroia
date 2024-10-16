from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.model.initializer_presets import *
from nodeleys.model.layers import LayerBase
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

class Conv2D(LayerBase):
  def __init__(self, total_kernels: int, kernel_size: Tuple[int,int], strides: Tuple[int,int]=(1,1), 
               name: str='', initializers=XavierUniform()):
    super().__init__(name=name, initializers=initializers)
    self.total_kernels = total_kernels
    self.kernel_size = kernel_size
    self.strides = strides

  def register(self, tensor_in: Node):
    total_channel= tensor_in.tensor.shape[1]
    kernel_height = self.kernel_size[0]
    kernel_width = self.kernel_size[1]
    self.kernels = Node(tensor=self.initializers((self.total_kernels, total_channel, kernel_height, kernel_width)),
                       name=f'kernels-{self.name}',
                       is_trainable=True)

  def call(self, tensor_in: Node):
    return node_conv2d(tensor_in, self.kernels, self.strides, self.name)
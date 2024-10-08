from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.math.initializers import *

if TYPE_CHECKING:
  from nodeleys.graph import Node

class LayerBase():
  def __init__(self, name: str='', initializers = None, nobuild: bool=False):
    self.initializers = initializers
    self.name = name
    self.nobuild = nobuild
    self.built = False

  def __call__(self, tensors_in: Node):
    if not self.built and not self.nobuild: 
      self.build(tensors_in)
      self.built = True
    return self.call(tensors_in)
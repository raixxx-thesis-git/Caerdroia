from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys import Node
from nodeleys.math.forward_math_func import *
from nodeleys.math.initializers import *

if TYPE_CHECKING:
  from nodeleys.graph import Node

class LayerBase():
  def __init__(self, name: str='', initializers = None, no_register: bool=False):
    self.initializers = initializers
    self.name = name
    self.no_register = no_register
    self.registered = False

  def __call__(self, tensors_in: Node):
    if not self.registered and not self.no_register: 
      self.register(tensors_in)
      self.registered = True
    return self.call(tensors_in)
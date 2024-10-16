from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from nodeleys.graph import Node

class OptimizerBase():
  def __call__(self, trainable_var: Node):
    self.call(trainable_var)
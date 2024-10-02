from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
  from nodeleys.graph import Node

class NodeleysModel():
  def __init__(self): pass

  def set_outputs(self, output: Node):
    self.output_adic = output.get_adic()
    self.output_adic.set_as_objective()
  
  def train(self):
    self.trainable_variables = []
    _, _ = self.output_adic.begin_backprop(tracing=False, traces=self.trainable_variables)


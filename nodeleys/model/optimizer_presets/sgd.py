from __future__ import annotations
from typing import TYPE_CHECKING, List
from nodeleys.model.optimizer_presets import OptimizerBase

if TYPE_CHECKING:
  from nodeleys.graph import Node

class SGD(OptimizerBase):
  def __init__(self, learning_rate: float):
    self.learning_rate = learning_rate
    pass

  def call(self, trainable_var: Node):
    trainable_var.tensor = trainable_var.tensor - self.learning_rate * trainable_var.get_gradient()
    trainable_var.clear_grad()
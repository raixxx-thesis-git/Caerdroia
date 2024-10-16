from __future__ import annotations
from typing import TYPE_CHECKING
from nodeleys.math.forward_math_func import *

if TYPE_CHECKING:
  from nodeleys.graph import Node

class CategoricalCrossEntropy():
  def __call__(self, predicted: Node, real_value: Node):
    ln_predicted = node_ln(predicted)
    products = node_mul(real_value, ln_predicted)
    cross_entropy = node_redsum(products, axis=1)
    cross_entropy_sum_over_batch = node_mul(node_redsum(cross_entropy, axis=0), -1.0)
    batch_size = float(cross_entropy.tensor.shape[0])
    loss = node_div(cross_entropy_sum_over_batch, batch_size)
    
    return loss

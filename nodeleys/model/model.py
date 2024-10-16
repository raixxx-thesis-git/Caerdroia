from __future__ import annotations
from typing import TYPE_CHECKING, List, Any, Union, Set
if TYPE_CHECKING:
  from nodeleys.graph import Node
  from nodeleys.model.optimizer_presets import OptimizerBase

class NodeleysModel():
  def __init__(self):
    self.trainable_vars: Set[Node] = []

  def add_loss(self, loss):
    self.loss = loss

  def __call__(self, inputs: Union[List[Node], Node]):
    return self.call(inputs)
  
  # This feature is still under consideration whether it should be continued
  # or halted.
  # 
  # def auto_update(self, inputs: Union[List[Node], Node], 
  #                 outputs: Union[List[Node], Node], learning_rate: int):
  #   model = self.call(inputs)
  #   loss: Node = self.loss(model, outputs)

  #   self.trainable_vars: List[Node] = []
  #   loss.adic.set_as_objective()
  #   loss.adic.begin_backprop(tracing=False, traces=self.trainable_vars)
    
  #   for trainable_var in self.trainable_vars:
  #     trainable_var.tensor += learning_rate * trainable_var.get_gradient()
  #     trainable_var.clear_grad()

  def compute_grads(self, loss: Node):
    for trainable_var in self.trainable_vars:
      trainable_var.clear_grad()
      
    self.trainable_vars: List[Node] = []
    loss.adic.set_as_objective()
    loss.adic.begin_backprop(tracing=False, traces=self.trainable_vars)
    self.trainable_vars = set(self.trainable_vars)

  def update(self, optimizer: OptimizerBase, weights: List[Node]=-1):
    iter_from = weights if weights != -1 else self.trainable_vars
    
    for trainable_var in iter_from: optimizer(trainable_var)
    self.trainable_variables = [] 

  def set_outputs(self, output: Node):
    self.output_adic = output.get_adic()
    self.output_adic.set_as_objective()
  
  def train(self):
    self.trainable_variables = []
    _, _ = self.output_adic.begin_backprop(tracing=False, traces=self.trainable_variables)


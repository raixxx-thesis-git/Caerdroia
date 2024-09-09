from __future__ import annotations
from typing import TYPE_CHECKING, Union, List
from cupy import ndarray
from nodeleys.graph import Node, Duplet, Triplet
from nodeleys.system import SwitchBackpropSystem

import cupy

def or_(param0: str, param1: str) -> ndarray:
  param0 = param0.replace('x', 'domain.tensor')
  param1 = param1.replace('x', 'domain.tensor')
  return f'cupy.logical_or({param0}, {param1})'

def and_(param0: str, param1: str) -> ndarray:
  param0 = param0.replace('x', 'domain.tensor')
  param1 = param1.replace('x', 'domain.tensor')
  return f'cupy.logical_and({param0}, {param1})'

class Switch(SwitchBackpropSystem):
  def __init__(self, domain: Node, constants: List[Node], graphs: List[Node],
               conditions: List[str], name:str='') -> None:
    self.name = name
    self.domain = domain
    self.graphs = graphs
    self.constants = constants

    # A whiteboard is a matrix filled with zeros which will be filled by
    # the outcome of each graph according to its condition. For instance:

    self.whiteboard = cupy.zeros(shape=(domain.tensor.shape))
    for graph, condition in zip(graphs, conditions):
      condition = condition if 'cupy' in condition else condition.replace('x', 'domain.tensor')
      print(condition)
      # A segregated indices are the indices of the domain tensor where its value conforms
      # with its corresponding condition. We fill the whiteboard with these particular indices.
      
      segregated_idx = cupy.argwhere(eval(condition))
      self.whiteboard[segregated_idx[:,0], segregated_idx[:,1]] = graph.tensor[segregated_idx[:,0], segregated_idx[:,1]]

  def compile(self) -> Node:
    outcome = Node(self.whiteboard, name=self.name)
    outcome.set_adic(self)
    self.o = outcome
    return outcome
  
  def __repr__(self):
    return f'Switch({self.domain.name}, {self.o.name})'


    

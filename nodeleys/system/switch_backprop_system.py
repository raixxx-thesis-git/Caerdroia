from __future__ import annotations
from typing import TYPE_CHECKING, Union, List

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet, Node, Switch

class SwitchBackpropSystem:
  def __init__(self): pass

  def propagate(self: Union[Switch, SwitchBackpropSystem]) -> None:
    for idx, graph in enumerate(self.graphs):
      _, _ = graph.adic.propagate([], [], [], False, self.constants, True, idx)
    pass

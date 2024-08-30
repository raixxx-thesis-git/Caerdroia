from __future__ import annotations
from graphviz import Digraph
from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
  from Rivers.graph import Triplet, Duplet

class Grapher():
  def __init__(self): pass

  def graph(self, adic: Optional[Union[Triplet, Duplet]]):
    pass
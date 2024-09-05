from __future__ import annotations
from typing import TYPE_CHECKING
from cupy import ndarray
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Expression

class Expression():
  def __init__(self, data: ndarray, name: str):
    self.data = data
    self.name = name

  def __repr__(self):
    return f'Expression({self.name})'
  
  def __and__(self: Expression, other: Expression):
    data = cupy.logical_and(self.data, other.data)
    return Expression(data, name='and')
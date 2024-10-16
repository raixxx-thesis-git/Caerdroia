from typing import Tuple
from cupy import ndarray

import cupy

class RandomNormal():
  def __init__(self, mean: float=0.0, std: float=1.0):
    self.mean = mean
    self.std = std

  def __call__(self, shape: Tuple[int, int]) -> ndarray:
    return cupy.random.normal(loc=self.mean, scale=self.std, size=shape)
  
from typing import Tuple
from cupy import ndarray

import cupy

class RandomNormal():
  def __init__(self, mean: float=0.0, std: float=1.0):
    self.mean = mean
    self.std = std

  def __call__(self, shape: Tuple[int, int]) -> ndarray:
    return cupy.random.normal(loc=self.mean, scale=self.std, size=shape)
  
class XavierNormal():
  def __init__(self): pass

  def __call__(self, shape: Tuple[int, int]):
    fan_in, fan_out = shape
    std = cupy.sqrt(2/(fan_in + fan_out))
    return cupy.random.normal(loc=0, scale=std, size=shape)

class XavierUniform():
  def __init__(self): pass

  def __call__(self, shape: Tuple[int, int]):
    fan_in, fan_out = shape
    end_range = cupy.sqrt(6/(fan_in + fan_out))
    return cupy.random.uniform(low=-1*end_range, high=end_range, size=shape)
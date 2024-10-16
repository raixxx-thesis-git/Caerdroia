from typing import Tuple
from cupy import ndarray

import cupy

class XavierUniform():
  def __init__(self): pass

  def __call__(self, shape: Tuple[int, int]):
    fan_in, fan_out = shape[-2:]
    end_range = cupy.sqrt(6/(fan_in + fan_out))
    return cupy.random.uniform(low=-1*end_range, high=end_range, size=shape)
from typing import Tuple
from cupy import ndarray

import cupy
  
class XavierNormal():
  def __init__(self): pass

  def __call__(self, shape: Tuple[int, int]):
    fan_in, fan_out = shape
    std = cupy.sqrt(2/(fan_in + fan_out))
    return cupy.random.normal(loc=0, scale=std, size=shape)
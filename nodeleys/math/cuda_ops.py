from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from numpy import ndarray

import numpy as np
import time

def to_device(numpy_data: ndarray) -> DeviceNDArray:
  return cuda.to_device(numpy_data)

def tensor_add_2d(l_operand: DeviceNDArray, r_operand: DeviceNDArray) -> DeviceNDArray:
  outcome = cuda.device_array_like(l_operand)
  threads_per_block = (32,32)
  blocks_per_grid = ((l_operand.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                     (l_operand.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])
  

  ops_tensor_add_2d[blocks_per_grid, threads_per_block](l_operand, r_operand, outcome)
  cuda.synchronize()
  return outcome

@cuda.jit
def ops_tensor_add_2d(l_operand: DeviceNDArray, r_operand: DeviceNDArray, outcome: DeviceNDArray) -> None:
  i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  
  if i < l_operand.shape[0] and j < l_operand.shape[1]:
    outcome[i, j] = l_operand[i, j] + r_operand[i, j]



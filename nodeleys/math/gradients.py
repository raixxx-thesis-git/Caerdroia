# b42o.vercel.app -------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Dict, Any
from cupy import ndarray
from numba import jit
from nodeleys.system.misc import block_stride_view
import cupy
import time
from numba import cuda, float32

if TYPE_CHECKING:
  from nodeleys.graph import Node

def consider(grad: ndarray, is_constant: bool) -> bool:
  return grad if not is_constant else None

def LR_init(L: Node, R: Node) -> Tuple[ndarray, ndarray, bool, bool]:
  return (L.tensor, R.tensor, L.get_is_constant(), R.get_is_constant())

def L_init(L: Node) -> Tuple[ndarray, bool]:
  return (L.tensor, L.get_is_constant())  

def grad_for_matmul(L: Node, R: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, R, = L.tensor, R.tensor

  grad_L = prev_grad @ R.T
  grad_R = L.T @ prev_grad

  return (grad_L, grad_R)
   
def grad_for_reduce_sum(L: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L = L.tensor

  if ((L.shape[0] != 1 and L.shape[1] == prev_grad.shape[1]) or 
      (L.shape[0] == 1 and L.shape[0] == prev_grad.shape[0])):
    grad_L = cupy.ones(shape=L.shape) * prev_grad

  return grad_L

def grad_for_add(L: Node, R: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, R,  = L.tensor, R.tensor

  equal_space_operands = L.shape == R.shape
  try:
    L_broadcast = (L.shape[0] == 1 and R.shape[0] != 1) and (L.shape[1] == R.shape[1])
    R_broadcast = (L.shape[0] != 1 and R.shape[0] == 1) and (L.shape[1] == R.shape[1])
  except: pass
  L_is_scalar = L.shape == () 
  R_is_scalar = R.shape == ()

  if equal_space_operands:
    grad_L = prev_grad
    grad_R = prev_grad
  elif L_is_scalar:
    grad_L = cupy.sum(prev_grad, keepdims=False)
    grad_R = prev_grad
  elif R_is_scalar:
    grad_L = prev_grad
    grad_R =  cupy.sum(prev_grad, keepdims=False)
  elif L_broadcast:
    grad_L = cupy.sum(prev_grad, axis=0, keepdims=True)
    grad_R = prev_grad
  elif R_broadcast:
    grad_L = prev_grad
    grad_R = cupy.sum(prev_grad, axis=0, keepdims=True)
  
  return (grad_L, grad_R)

def grad_for_sub(L: Node, R: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  equal_space_operands = L.shape == R.shape
  L_broadcast = (L.shape[0] == 1 and R.shape[0] != 1) and (L.shape[1] == R.shape[1])
  R_broadcast = (L.shape[0] != 1 and R.shape[0] == 1) and (L.shape[1] == R.shape[1])

  if equal_space_operands:
    grad_L = consider(prev_grad, L_is_constant)
    grad_R = consider(-1.0 * prev_grad, R_is_constant)
  elif L_broadcast:
    grad_L = consider(cupy.sum(prev_grad, axis=0, keepdims=True), L_is_constant)
    grad_R = consider(-1.0 * prev_grad, R_is_constant)
  elif R_broadcast:
    grad_L = consider(prev_grad, L_is_constant)
    grad_R = consider(-1.0 * cupy.sum(prev_grad, axis=0, keepdims=True), R_is_constant)

  return (grad_L, grad_R)

def grad_for_div(L: Node, R: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  if L.shape == R.shape:
    grad_L = consider((1/R) * prev_grad, L_is_constant)
    grad_R = consider(-1.0*L/(R**2) * prev_grad, R_is_constant)
  elif R.shape == ():
    grad_L = consider((1/R) * prev_grad, L_is_constant)
    grad_R = consider((-1/R**2) * cupy.sum(prev_grad * L, keepdims=True), R_is_constant)
  elif L.shape == ():
    grad_L = consider(cupy.sum((1/R) * prev_grad, keepdims=True), L_is_constant)
    grad_R = consider(prev_grad * (-L)/(R**2), R_is_constant)

  print(L.shape, R.shape)
  print(grad_L)
  print(grad_R)
  return (grad_L, grad_R)

def grad_for_mul(L: Node, R: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)

  if L.shape == R.shape:
    grad_L = consider(R * prev_grad, L_is_constant)
    grad_R = consider(L * prev_grad, R_is_constant)
  elif R.shape == ():
    grad_L = consider(R * prev_grad, L_is_constant)
    grad_R = consider(cupy.sum(prev_grad * L, keepdims=True), R_is_constant)
  elif L.shape == ():
    pass

  return (grad_L, grad_R)

def grad_for_flatten(L: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, L_is_constant = L_init(L)
  grad_L = consider(cupy.reshape(prev_grad, newshape=L.shape), L_is_constant)
  return grad_L

def grad_for_pow(L: Node, R: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, R, L_is_constant, R_is_constant = LR_init(L, R)
  
  L_is_shapeless = L.shape == ()
  R_is_shapeless = R.shape == ()

  if R_is_shapeless:
    grad_L = consider(R * L**(R-1) * prev_grad, L_is_constant)
    grad_R = consider(cupy.sum(prev_grad * (L**R) * cupy.log(L)), R_is_constant)
  elif L_is_shapeless:
    None
              
  return (grad_L, grad_R)

def grad_for_relu(L: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  L, _ = L_init(L)
  cond0 = ((L < 0) * 0.0)
  cond1 = ((L >= 0) * metadata['slope'] * prev_grad)
  grad_L = cond0 + cond1
  return grad_L

def grad_for_leaky_relu(L: Node, prev_grad: ndarray, metadata: Dict[str,Any]={}) -> ndarray:
  L, _ = L_init(L)
  cond0 = ((L < 0) * metadata['slope_minval'] * prev_grad)
  cond1 = ((L >= 0) * metadata['slope_posval'] * prev_grad)
  grad_L = cond0 + cond1
  return grad_L

def grad_for_conv2d(blocks: Node, kernels: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> ndarray:
  blocks_, kernels_, _, _ = LR_init(blocks, kernels)

  # Calculating the gradient for the kernel
  kernel_height = kernels_.shape[2]
  kernel_width = kernels_.shape[3]

  stride_height = metadata['strides'][0]
  stride_width = metadata['strides'][1]
  
  sub_blocks = block_stride_view(blocks_, (kernel_height, kernel_width), (stride_height, stride_width))
  subscript = 'bahw,hwbcrs' #a == k_0
  grad_kernels = cupy.einsum(subscript, prev_grad, sub_blocks)


  # Calculating the gradient for the blocks
  original_shape = metadata['original_shape']
  subscript = 'bjhw,jcrs->hwbcrs'
  grad_blocks = cupy.einsum(subscript, prev_grad, kernels_)

  whiteboard = cupy.zeros(shape=original_shape)
    
  # device_whiteboard = cuda.to_device(whiteboard)
  # device_base = cuda.to_device(grad_blocks)
  # threads_per_block = (10,10,10)
  # blocks_per_grid = ((grad_blocks.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
  #                    (grad_blocks.shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
  #                    (grad_blocks.shape[2] + threads_per_block[2] - 1) // threads_per_block[2])

  # shift_add[blocks_per_grid, threads_per_block](device_whiteboard, 
  #                                               device_base, 
  #                                               (stride_height, stride_width), 
  #                                               (kernel_height, kernel_width))
  # cuda.synchronize()

  # whiteboard = cupy.array(device_whiteboard.copy_to_host())

  for hx in range(0, grad_blocks.shape[0], 1):
    hstart = hx*stride_height
    for wx in range(0, grad_blocks.shape[1], 1):
      wstart = wx*stride_width
      whiteboard[:,:,hstart:hstart+kernel_height,wstart:wstart+kernel_width] += grad_blocks[hx,wx,:,:,:,:]

  return (whiteboard, grad_kernels)

def grad_for_maxpool2d(blocks: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}) -> Tuple[ndarray]:
  blocks, _ = L_init(blocks)
  
  pool_height = metadata['pool_size'][0]
  pool_width = metadata['pool_size'][1]
  stride_height = metadata['strides'][0]
  stride_width = metadata['strides'][1]

  strided_blocks = block_stride_view(blocks, (pool_height, pool_width), (stride_height, stride_width))
  height, width, batch_size, channel, height_in, width_in = strided_blocks.shape
  
  grid_indexes = cupy.indices([height, width, batch_size, channel])
  indexes_permutation = cupy.moveaxis(grid_indexes, 0, -1).reshape(-1, 4)
  
  # generates indexes in the flat space of an argmax
  flat_argmax_indexes = cupy.argmax(strided_blocks, axis=[-2, -1]).ravel()

  # unravels the flat space into (H x W) space indicies.
  unravel_height, unravel_width = cupy.unravel_index(flat_argmax_indexes, (height_in, width_in))
  unravel_height_width = cupy.concatenate([unravel_height[:, None], unravel_width[:, None]], axis=1)

  # final argmax
  argmax = cupy.concatenate([indexes_permutation, unravel_height_width], axis=-1)

  zeros = cupy.zeros(shape=strided_blocks.shape)
  zeros[argmax[:,0], argmax[:,1], argmax[:,2], 
        argmax[:,3], argmax[:,4], argmax[:,5]] = prev_grad[argmax[:,2], argmax[:,3], argmax[:,0], argmax[:,1]]
  
  whiteboard = cupy.zeros(blocks.shape)
  
  # device_whiteboard = cuda.to_device(whiteboard)
  # device_base = cuda.to_device(zeros)
  # threads_per_block = (10,10,10)
  # blocks_per_grid = ((zeros.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
  #                    (zeros.shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
  #                    (zeros.shape[2] + threads_per_block[2] - 1) // threads_per_block[2])

  # shift_add[blocks_per_grid, threads_per_block](device_whiteboard, 
  #                                               device_base, 
  #                                               (stride_height, stride_width), 
  #                                               (pool_height, pool_width))
  # cuda.synchronize()

  # whiteboard = cupy.array(device_whiteboard.copy_to_host())
  for hx in range(0, zeros.shape[0], 1):
    hstart = hx*stride_height
    for wx in range(0, zeros.shape[1], 1):
      wstart = wx*stride_width
      whiteboard[:,:,hstart:hstart+pool_height,wstart:wstart+pool_width] += zeros[hx,wx,:,:,:,:]

  return whiteboard

def grad_for_concat(primary_tensor: Node, secondary_tensor: Node, prev_grad: ndarray, metadata: Dict[str, Any]={}):
  axis_split = metadata['axis']
  total_dims = len(primary_tensor.tensor.shape)

  slices = [slice(None)] * total_dims
  slices[axis_split] = slice(0,primary_tensor.tensor.shape[axis_split])
  grad_for_primary = prev_grad[tuple(slices)]

  slices[axis_split] = slice(primary_tensor.tensor.shape[axis_split], None)
  grad_for_secondary = prev_grad[tuple(slices)]

  return (grad_for_primary, grad_for_secondary)

@cuda.jit
def shift_add(whiteboard, base, strides, kernel_size):
  hx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  wx = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  bx = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
  
  stride_height = strides[0]
  stride_width = strides[1]
  kernel_height = kernel_size[0]
  kernel_width = kernel_size[1]
  
  if hx < base.shape[0] and wx < base.shape[1] and bx < base.shape[2]:
    for h_expand in range(kernel_height):
      for w_expand in range(kernel_width):
        for b_ in range(whiteboard.shape[1]):
          whiteboard[bx, b_, (hx*stride_height)+h_expand, (wx*stride_width)+w_expand] += base[hx, wx, bx ,b_ ,h_expand, w_expand]
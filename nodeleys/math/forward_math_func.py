from __future__ import annotations
from nodeleys.system import secure_type
from nodeleys.system.misc import block_stride_view
from typing import TYPE_CHECKING, Optional, Union, List, Tuple, Dict, Any
from cupy import ndarray

import operator
import cupy
import re

if TYPE_CHECKING:
  from nodeleys import Node

def complete_adic_func(l_operand: Node, r_operand: Optional[Node], 
                       operator: str, outcome: ndarray, name: str, metadata: Dict[str, Any]={}) -> Node:
  from nodeleys import Node
  from nodeleys.graph import Triplet, Duplet
  outcome_node = Node(outcome, name=name)

  if r_operand == None:
    # create a Duplet D[n](L, O; op)
    adic = Duplet(l_operand, outcome_node, operator)

    # connect D[n] with D[n-1]
    adic.set_prev(l_operand.get_adic())

  else:
    # create a Duplet T[n](L, O; op)
    adic = Triplet(l_operand, r_operand, outcome_node, operator)

    # connect T[n] with T0[n-1] and T1[n-1]
    adic.set_prev((l_operand.get_adic(), r_operand.get_adic()))
  
  outcome_node.set_adic(adic)
  outcome_node.assign_metadata(metadata)
  return outcome_node

def secure_operands(l_operand: Union[Node, float], r_operand: Union[Node, float]):
  return secure_type(l_operand), secure_type(r_operand)

def node_add(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str='') -> Node:
  l_operand, r_operand = secure_operands(l_operand, r_operand)
  outcome = operator.add(l_operand.tensor, r_operand.tensor)
  return complete_adic_func(l_operand, r_operand, '+', outcome, name)

def node_sub(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str='') -> Node:
  l_operand, r_operand = secure_operands(l_operand, r_operand)
  outcome = operator.sub(l_operand.tensor, r_operand.tensor)
  return complete_adic_func(l_operand, r_operand, '-', outcome, name)

def node_mul(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str='') -> Node:
  l_operand, r_operand = secure_operands(l_operand, r_operand)
  outcome = operator.mul(l_operand.tensor, r_operand.tensor)
  return complete_adic_func(l_operand, r_operand, '*', outcome, name)

def node_div(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str='') -> Node:
  l_operand, r_operand = secure_operands(l_operand, r_operand)
  outcome = operator.truediv(l_operand.tensor, r_operand.tensor)
  return complete_adic_func(l_operand, r_operand, '/', outcome, name)

def node_matmul(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str='') -> Node:
  l_operand, r_operand = secure_operands(l_operand, r_operand)
  outcome = l_operand.tensor @ r_operand.tensor
  return complete_adic_func(l_operand, r_operand, '@', outcome, name)

def node_pow(l_operand: Union[Node, float], r_operand: Union[Node, float], name: str='') -> Node:
  l_operand, r_operand = secure_operands(l_operand, r_operand)
  outcome = l_operand.tensor ** r_operand.tensor
  return complete_adic_func(l_operand, r_operand, '**', outcome, name)

def node_redsum(l_operand: Union[Node, float], axis: int, name: str='') -> Node:
  l_operand = secure_type(l_operand)
  outcome = cupy.sum(l_operand.tensor, axis=axis, keepdims=True)
  return complete_adic_func(l_operand, None, 'redsum', outcome, name)

def node_ln(l_operand: Union[Node, float], eps: int=1e-7, name: str='') -> Node:
  l_operand = secure_type(l_operand)
  outcome = cupy.log(l_operand.tensor + 1e-30)
  return complete_adic_func(l_operand, None, 'ln', outcome, name)

def node_relu(l_operand: Node, slope: int=1, name: str='') -> Node:
  l_operand = secure_type(l_operand)
  cond0 = ((l_operand.tensor < 0.0) * 0.0)
  cond1 = ((l_operand.tensor >= 0.0) * slope * l_operand.tensor)
  add_elemwise_kernel = cupy.ElementwiseKernel(
     'float64 x, float64 y', 'float64 z',
     '''
     z = x + y
     ''', 'my_kernel')
  outcome = add_elemwise_kernel(cond0, cond1)

  metadata = {
    'slope': slope
  }
  return complete_adic_func(l_operand, None, 'relu', outcome, name, metadata)

def node_leaky_relu(l_operand: Node, slope_minval :int=0.01, slope_posval: int=1.0, name: str='') -> Node:
  l_operand = secure_type(l_operand)
  cond0 = ((l_operand.tensor < 0) * slope_minval * l_operand.tensor)
  cond1 = ((l_operand.tensor >= 0) * slope_posval * l_operand.tensor)
  outcome = cond0 + cond1
  
  metadata = {
    'slope_minval': slope_minval,
    'slope_posval': slope_posval
  }
  return complete_adic_func(l_operand, None, 'leakyrelu', outcome, name, metadata)

def node_flatten(l_operand: Union[Node, ndarray], name: str='') -> Node:
  l_operand = secure_type(l_operand)
  outcome = cupy.reshape(l_operand.tensor, newshape=(l_operand.tensor.shape[0], -1))
  return complete_adic_func(l_operand, None, 'flatten', outcome, name)

def node_conv2d(blocks: Union[Node, ndarray], kernels: Union[Node, ndarray], strides: Tuple[int]=(1,1), name: str='') -> Node:
  blocks, kernels = secure_operands(blocks, kernels)

  sub_blocks = block_stride_view(blocks=blocks.tensor, 
                                 view_size=(kernels.tensor.shape[2], kernels.tensor.shape[3]), 
                                 strides=strides)

  metadata = {
    'strides': strides,
    'original_shape': blocks.tensor.shape
  }

  outcome = cupy.einsum('ijklmn,rlmn', sub_blocks, kernels.tensor)
  outcome = cupy.transpose(outcome, axes=(2,3,0,1))
  return complete_adic_func(blocks, kernels, 'conv2d', outcome, name, metadata)

def node_maxpool2d(blocks: Union[Node, ndarray], pool_size: Tuple[int]=(2,2), strides: Tuple[int]=(1,1), name: str='') -> Node:
  blocks = secure_type(blocks)
  sub_blocks = block_stride_view(blocks=blocks.tensor, view_size=pool_size, strides=strides)

  metadata = {
    'strides': strides,
    'pool_size': pool_size
  }

  outcome = cupy.transpose(cupy.max(sub_blocks, axis=[-2,-1]), axes=(2,3,0,1))
  return complete_adic_func(blocks, None, 'maxpool2d', outcome, name, metadata)

def node_concat(primary_tensor: Union[Node, ndarray], secondary_tensor: Union[Node, ndarray], axis: int, name: str=''):
  primary_tensor, secondary_tensor = secure_operands(primary_tensor, secondary_tensor)
  outcome = cupy.concatenate((primary_tensor.tensor, secondary_tensor.tensor), axis=axis)

  metadata = {
    'axis': axis
  }
  return complete_adic_func(primary_tensor, secondary_tensor, 'concat', outcome, name, metadata)
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Node

def block_stride_view(blocks: Node, view_size: Tuple[int], strides: Tuple[int]=(1,1)):
  batch_size = blocks.shape[0]
  channel = blocks.shape[1]
  domain_height = blocks.shape[2]
  domain_width = blocks.shape[3]

  view_height = view_size[0]
  view_width = view_size[1]
 
  stride_height = strides[0]
  stride_width = strides[1]

  new_width = 1 + int((domain_width - view_width)/stride_width)
  new_height = 1 + int((domain_height - view_height)/stride_height)

  pick_shape = (new_height, new_width, batch_size, channel, view_height, view_width)

  blocks_strides = cupy.array((blocks.strides + blocks.strides)[2:])
  blocks_stride_strides = cupy.array((blocks.strides[2]*(stride_height-1), blocks.strides[3]*(stride_width-1), 0, 0,0,0))
  # memory_skip = (blocks.strides + blocks.strides)[2:]
  memory_skip = (blocks_strides + blocks_stride_strides).get()

  sub_blocks = cupy.lib.stride_tricks.as_strided(blocks, pick_shape, memory_skip)
  
  return sub_blocks
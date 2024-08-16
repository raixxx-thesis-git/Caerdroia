from cupy import ndarray
from mytensor import TensorNode
from mytensor import BackwardMath

import cupy

class GradComputerError(Exception):
  def __init__(self, msg):
    super().__init__(msg)

class GradComputer():
	def __init__(self):
		math = BackwardMath()
		self.operator_to_method = {
			'@p': math.grad_for_matmul_primary,
			'@s': math.grad_for_matmul_secondary,
			'+': math.grad_for_add,
			'+*': math.grad_for_redsum
		}
		pass

	'''
	For developer: 
	Issue #1: This method still lacks of proper end-node handling.
	The end-node does not have a child, hence "child.operation" is invalid.
	'''

	def decide_and_calc_gradient(self, tensor_node: TensorNode):
		child = tensor_node.child
		# temporary fix for issue no. 1
		if child == None and tensor_node.parent[0] == None:
			raise GradComputerError('Unconnected Tensor')
		
		primary_var = child.parent[0]
		secondary_var = child.parent[1]
		operation = child.operation

		if tensor_node == primary_var and secondary_var != None: 
			self.operator_to_method[operation + 'p'](secondary_var)
		elif tensor_node == primary_var and secondary_var == None:
			# special case: functional mapping
			self.operator_to_method[operation + 'p'](secondary_var)
		elif tensor_node == secondary_var: 
			self.operator_to_method[operation + 's'](primary_var)
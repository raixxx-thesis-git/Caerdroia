from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mytensor import TensorNode

class NodeComputer():
  def __init__(self):
    pass

  def __add__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.add(self.tensor, partner.tensor)
    return self.make_child(partner, '+.', value)

  def __sub__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.sub(self.tensor, partner.tensor)
    return self.make_child(partner, '-.', value)
  
  def __matmul__(self, partner: TensorNode) -> TensorNode:
    value = self.forward_math.matmul(self.tensor, partner.tensor)
    return self.make_child(partner, '@.', value)

  def __mul__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.mul(self.tensor, partner.tensor)
    return self.make_child(partner, '*.', value)

  def __truediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.truediv(self.tensor, partner.tensor)
    return self.make_child(partner, '/.', value)
  
  def __rtruediv__(self, partner: TensorNode) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.truediv(partner.tensor, self.tensor)
    return self.make_child(partner, '/r', value)
  
  def __pow__(self,  partner: TensorNode | float) -> TensorNode:
    partner = self.partner_assure_tensornode(partner)
    value = self.forward_math.basic_linalg(self, partner, '**')
    return self.make_child(partner, '**', value)

  def reduce_sum(self, axis: int) -> TensorNode:
    value = self.forward_math.reduce_sum(self, axis)
    return self.make_child(None, 'redsum', value)
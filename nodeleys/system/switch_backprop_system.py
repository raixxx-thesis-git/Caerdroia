from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple, Optional
import cupy

if TYPE_CHECKING:
  from nodeleys.graph import Duplet, Triplet, Node, Switch

class SwitchBackpropSystem:
  def __init__(self): pass

  def domain_mask_gradient(self: Switch) -> None:
    # In this case, we are doing another whiteboarding system. Since the gradient of the dependent
    # node is constructed via the virtual graphs, we should do the masking process. Henceforth,
    # this is required.
    if not self.from_external['tracing']:
      whiteboard = cupy.zeros(shape=self.whiteboard.shape)
      for idx, segregated_idx in enumerate(self.segregated_idxs):
        self.domain.sum_virtual_gradient_by_session(idx)
        gradient = self.domain.get_virtual_gradient_by_session(idx)
        whiteboard[segregated_idx[:,0], segregated_idx[:,1]] = gradient[segregated_idx[:,0], segregated_idx[:,1]]

      self.domain.add_gradient(whiteboard)
    return self.domain.get_adic().propagate(passed_adics=self.from_external['passed_adics'],
                                            bonds=self.from_external['bonds'],
                                            checkpoints=self.from_external['checkpoints'],
                                            from_leap=self.from_external['from_leap'],
                                            trainable_nodes=self.from_external['trainable_nodes'],
                                            tracing=self.from_external['tracing'])

  def propagate(self: Union[Switch, SwitchBackpropSystem], 
                passed_adics: List[Union[Triplet, Duplet]]=[], 
                bonds: List[Tuple[Union[Triplet, Duplet]]]=[], 
                checkpoints: List[Optional[Union[Triplet, Duplet]]]=[], 
                from_leap: bool=False,
                trainable_nodes: Union[Node]=[],
                tracing: bool=False) -> None:
    # The backward propagation algorithm in <Switch> adic is different from those of
    # <Duplet> and <Triplet>. Unlike the other two, <Switch> contains multiple independent
    # graphs within itself. These graphs are called sub-graphs (<Switch.sub_graphs>).
    self.from_external = {
      'passed_adics': passed_adics,
      'bonds': bonds,
      'checkpoints': checkpoints,
      'from_leap': from_leap,
      'trainable_nodes': trainable_nodes,
      'tracing': tracing
    }

    interrupts = self.dependent_vars
    interrupts.append(self.domain)

    for idx, sub_graph in enumerate(self.sub_graphs):
      # Upon each sub-graph, we do the backward propagation process.
      #
      # [WARNING!] It is still unclear the process would work under a condition where another 
      # <Switch> adic within the sub-graph exists.
      #
      # We pass the <interrupts>, <is_virtually>, and <idx> parameters. This parameters would
      # make sure that the backward propagation process in the <Duplet> and <Triplet> adic would
      # consider themselves as a 'virtual'. This concept of virtual is equivalent to the sub-graphs.
      if not tracing:
        sub_graph.add_virtual_gradient(self.o.get_last_gradient(), idx)
      _, _ = sub_graph.get_adic().propagate([], [], [], False,
                                            interrupts=interrupts, is_virtually=True, idx=idx,
                                            tracing=tracing, trainable_nodes=trainable_nodes)
    
    self.domain_mask_gradient()
    pass
from typing import get_origin, get_args, List, Dict, Any

class TypeChecker(Exception):
  def __init__(self, msg):
    super().__init__(msg)

def type_checker(annotations: Dict[str, Any], locals_: Dict[str, Any], skips: List[str]) -> None:
  for local in locals_:
    # check if the required data is a list, dict, or tuple
    required_structure = get_origin(annotations[local])
    required_key_datatype = get_args(annotations[local])
    passed_structure = type(locals_[local])

    if required_structure != None and required_structure != passed_structure:
      raise TypeChecker((f'Data structure mismatch. Data structure {required_structure} is required, ',
                         f'but got {passed_structure}'))
    
    if required_structure == dict and type(locals_[local]) != required_key_datatype[1]:
      raise TypeChecker((f'Data structure mismatch. Data structure {required_structure} is required, ',
                         f'but got {passed_structure}'))
    

  pass
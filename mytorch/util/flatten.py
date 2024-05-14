import numpy as np
from mytorch import Tensor

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    data = np.flatten(x.data)
    req_grad = x.requires_grad
    depends_on = [x] if req_grad else []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

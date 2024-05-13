import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    data = np.tanh(x.data)
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data ** 2)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

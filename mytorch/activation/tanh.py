import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    s = x.exp() - (-x).exp()
    m = x.exp() + (-x).exp()
    m = m ** -1
    return s * m

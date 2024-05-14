import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    exp_x = x.exp()
    exp_neg_x = (-x).exp()
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

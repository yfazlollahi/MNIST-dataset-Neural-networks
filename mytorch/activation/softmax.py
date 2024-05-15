import numpy as np
from mytorch import Tensor, Dependency

from mytorch.tensor import _tensor_exp, _tensor_sum, _tensor_pow


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    exp_x = x.exp()
    s = exp_x @ (np.ones((exp_x.shape[-1], 1)))
    return exp_x * (s ** -1)

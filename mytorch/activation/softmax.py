import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    exp_values = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    softmax_values = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Softmax gradient calculation
            sm = softmax_values.reshape(-1, softmax_values.shape[-1])
            grad_reshaped = grad.reshape(-1, softmax_values.shape[-1])
            jac = np.diagflat(sm) - np.dot(sm[:, None], sm[:, None].T)
            grad = np.dot(grad_reshaped, jac)
            return grad.reshape(*x.shape)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=softmax_values, requires_grad=req_grad, depends_on=depends_on)

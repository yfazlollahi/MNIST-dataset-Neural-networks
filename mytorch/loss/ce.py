from mytorch import Tensor
import numpy as np
from ..activation import softmax

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    return -(label * preds.log()).sum()

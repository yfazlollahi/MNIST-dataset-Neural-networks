from mytorch import Tensor
import numpy as np
from ..activation import softmax

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    epsilon = 1e-15
    preds = np.clip(preds, epsilon, 1 - epsilon)
    
    cross_entropy = -np.sum(label * np.log(preds)) / label.shape[0]
    
    return cross_entropy

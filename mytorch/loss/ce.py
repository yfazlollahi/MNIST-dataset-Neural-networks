from mytorch import Tensor

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    exp_preds = preds.exp()
    softmax_preds = exp_preds / exp_preds.sum(axis=-1, keepdims=True)

    ce = -(label * softmax_preds.log()).sum(axis=-1).mean()
    return ce

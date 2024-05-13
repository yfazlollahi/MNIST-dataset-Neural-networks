from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    diff = preds - actual
    mse = (diff ** 2).mean()
    return mse

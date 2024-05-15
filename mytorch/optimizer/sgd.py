from typing import List

import numpy as np

from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for layer in self.layers:
            layer.weight = layer.weight - layer.weight.grad * self.learning_rate
            if layer.need_bias:
                layer.bias = layer.bias - layer.bias.grad * self.learning_rate

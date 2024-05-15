from typing import List

import numpy as np

from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

"TODO: (optional) implement Momentum optimizer"


class Momentum(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.01, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self):
        """Perform one step of the Momentum optimization algorithm."""
        for layer in self.layers:
            if layer.parameters():
                for param in layer.parameters():
                    grad = param.grad.data

                    if not hasattr(param, 'velocity'):
                        param.velocity = np.zeros_like(param.data)

                    param.velocity = self.momentum * param.velocity - self.learning_rate * grad

                    param.data += param.velocity

from typing import List

import numpy as np

from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

"TODO: (optional) implement RMSprop optimizer"


class RMSprop(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, rho=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def step(self):
        """Perform one step of the RMSprop optimization algorithm."""
        for layer in self.layers:
            if layer.parameters():
                for param in layer.parameters():
                    # Compute gradient
                    grad = param.grad.data

                    # Initialize moving average of squared gradients
                    if not hasattr(param, 'squared_grad_avg'):
                        param.squared_grad_avg = np.zeros_like(param.data)

                    # Update moving average of squared gradients
                    param.squared_grad_avg = self.rho * param.squared_grad_avg + (1 - self.rho) * grad ** 2

                    # Normalize gradients
                    grad /= np.sqrt(param.squared_grad_avg**2 + self.epsilon)

                    # Update parameters
                    param.data -= self.learning_rate * grad

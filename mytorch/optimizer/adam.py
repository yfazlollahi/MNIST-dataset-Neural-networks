from typing import List

import numpy as np

from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

"TODO: (optional) implement Adam optimizer"


class Adam(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Initialize time step

    def step(self):
        """Perform one step of the Adam optimization algorithm."""
        self.t += 1  # Increment time step
        for layer in self.layers:
            if layer.parameters():
                for param in layer.parameters():
                    # Compute gradient
                    grad = param.grad.data

                    # Initialize moving averages
                    if not hasattr(param, 'm'):
                        param.m = np.zeros_like(param.data)
                        param.v = np.zeros_like(param.data)

                    # Update biased first moment estimate
                    param.m = self.beta1 * param.m + (1 - self.beta1) * grad

                    # Update biased second raw moment estimate
                    param.v = self.beta2 * param.v + (1 - self.beta2) * (grad ** 2)

                    # Compute bias-corrected first and second moment estimates
                    m_hat = param.m / (1 - self.beta1 ** self.t)
                    v_hat = param.v / (1 - self.beta2 ** self.t)

                    # Update parameters
                    param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

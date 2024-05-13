from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for layer in self.layers:
            for param_name, param in layer.parameters().items():
                param.data -= self.learning_rate * param.grad.data

    def zero_grad(self):
        for layer in self.layers:
                layer.zero_grad()

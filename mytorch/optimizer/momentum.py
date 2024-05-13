from mytorch.optimizer import Optimizer

"TODO: (optional) implement Momentum optimizer"
class Momentum(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        for layer in self.layers:
            for param_name, param in layer.parameters().items():
                # Initialize velocities if not exists
                if param_name not in self.velocities:
                    self.velocities[param_name] = np.zeros_like(param.data)
                # Update velocities
                self.velocities[param_name] = self.momentum * self.velocities[param_name] - self.learning_rate * param.grad.data
                # Update parameters
                param.data += self.velocities[param_name]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


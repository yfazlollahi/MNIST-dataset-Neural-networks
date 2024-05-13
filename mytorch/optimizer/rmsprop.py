from mytorch.optimizer import Optimizer

"TODO: (optional) implement RMSprop optimizer"
class RMSprop(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradients = {}

    def step(self):
        for layer in self.layers:
            for param_name, param in layer.parameters().items():
                if param_name not in self.squared_gradients:
                    self.squared_gradients[param_name] = param.grad.data ** 2
                else:
                    self.squared_gradients[param_name] = self.beta * self.squared_gradients[param_name] + (1 - self.beta) * (param.grad.data ** 2)
                param.data -= self.learning_rate * (param.grad.data / (np.sqrt(self.squared_gradients[param_name]) + self.epsilon))

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

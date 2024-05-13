from mytorch.optimizer import Optimizer

"TODO: (optional) implement Adam optimizer"
class Adam(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = {}
        self.moment2 = {}
        self.timestep = 0
    
    def step(self):
        self.timestep += 1
        for layer in self.layers:
            for param_name, param in layer.parameters().items():
                # Initialize moment estimates if not exists
                if param_name not in self.moment1:
                    self.moment1[param_name] = np.zeros_like(param.data)

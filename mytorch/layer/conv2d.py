from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        output_height = (x.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (x.shape[3] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        output = np.zeros((x.shape[0], self.out_channels, output_height, output_width))

        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                output[:, :, h, w] = np.sum(x[:, :, h_start:h_end, w_start:w_end] * self.weight.data, axis=(2, 3))
                if self.need_bias:
                    output[:, :, h, w] += self.bias.data
        return Tensor(output)
    
    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer((self.out_channels, self.in_channels, *self.kernel_size), mode=self.initialize_mode),
            requires_grad=True
        )

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.out_channels, 1, 1), mode="zeros"),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        if self.weight:
            self.weight.grad = None
        if self.bias:
            self.bias.grad = None

    def parameters(self):
        "TODO: return weights and bias"
        params = [self.weight]
        if self.need_bias:
            params.append(self.bias)
        return params
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)

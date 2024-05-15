from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

from mytorch.util import initializer


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1),
                 need_bias: bool = False, mode="xavier") -> None:
        super().__init__(need_bias)
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
        batch_size, in_height, in_width = x.data.shape[0], x.data.shape[1], x.data.shape[2]
        out_height = (in_height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (in_width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        out_tensor_data = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride[0] - self.padding[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w_out * self.stride[1] - self.padding[1]
                        w_end = w_start + self.kernel_size[1]
                        x_slice = x.data[b, :, h_start:h_end, w_start:w_end]

                        out_tensor_data[b, c_out, h_out, w_out] = np.sum(x_slice * self.weight.data[c_out, :, :, :])

                        if self.need_bias:
                            out_tensor_data[b, c_out, h_out, w_out] += self.bias.data[c_out]

        return Tensor(out_tensor_data)

    def initialize(self):
        "TODO: initialize weights"
        if self.initialize_mode == "xavier":
            fan_in = self.in_channels * np.prod(self.kernel_size)
            fan_out = self.out_channels * np.prod(self.kernel_size)
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            self.weight = Tensor(
                np.random.uniform(-bound, bound, size=(self.out_channels, self.in_channels) + self.kernel_size))
        else:
            raise NotImplementedError("Initialization mode {} not implemented".format(self.initialize_mode))

        if self.need_bias:
            self.bias = Tensor(np.zeros((self.out_channels,)))

    def zero_grad(self):
        "TODO: implement zero grad"
        if self.weight is not None:
            self.weight.grad = np.zeros_like(self.weight.data)
        if self.bias is not None:
            self.bias.grad = np.zeros_like(self.bias.data)

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

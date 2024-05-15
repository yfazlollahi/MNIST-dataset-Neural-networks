from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np


class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, in_channels, in_height, in_width = x.data.shape
        out_height = (in_height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (in_width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        out_tensor_data = np.zeros((batch_size, in_channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(in_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride[0] - self.padding[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w_out * self.stride[1] - self.padding[1]
                        w_end = w_start + self.kernel_size[1]
                        x_slice = x.data[b, c, h_start:h_end, w_start:w_end]

                        out_tensor_data[b, c, h_out, w_out] = np.mean(x_slice)
        return Tensor(out_tensor_data)

    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)

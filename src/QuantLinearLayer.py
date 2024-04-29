import torch                    # type: ignore
import torch.nn as nn           # type: ignore
import torch.nn.functional as F # type: ignore

class QuantLinear(nn.Module):

    def __init__(self, Weight, bias):
        super(QuantLinear, self).__init__()
        
        self.register_buffer('quantized_weight', Weight)
        self.bias = bias

    def forward(self, input):

        return F.linear(input, self.quantized_weight, self.bias)
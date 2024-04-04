import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantLinear(nn.Module):

    def __init__(self, QuantizationObject, bias):
        super(QuantLinear, self).__init__()
        self.QuantizationObject = QuantizationObject
        self.bias = bias

    def forward(self, input):

        weight = self.QuantizationObject.dequantize()
        weight = torch.tensor(weight, dtype=torch.float32)
    
        return F.linear(input, weight, self.bias)

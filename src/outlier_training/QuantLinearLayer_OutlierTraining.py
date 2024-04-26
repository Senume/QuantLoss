import torch                        #type:ignore
import torch.nn as nn               #type:ignore
import torch.nn.functional as F     #type:ignore

class OutlierQuantLinear(nn.Module):

    def __init__(self, QuantizedWeight, Outliers, Error, bias):
        super(OutlierQuantLinear, self).__init__()

        # Constants
        self.register_buffer('constant_weight', QuantizedWeight)
        self.register_buffer('outliers', Outliers)

        # Trainable Parameters
        self.bias = bias
        self.Updates = nn.Parameter(Error)

    def forward(self, input_x):

        with torch.no_grad():
            OutlierWeight = self.outliers*self.Updates
            Weight = OutlierWeight + self.constant_weight
            
        return F.linear(input_x, Weight, self.bias)

import torch                        #type:ignore
import torch.nn as nn               #type:ignore
import torch.nn.functional as F     #type:ignore

class QuantLinear(nn.Module):

    def __init__(self, QuantizationObject, bias, error):
        super(QuantLinear, self).__init__()
        self.QuantizationObject = QuantizationObject
        self.bias = bias

        self.GradUpdates = nn.Parameter(torch.tensor(error, dtype=torch.float32))

    def forward(self, input):

        weight = self.QuantizationObject.dequantize()
        weight = torch.tensor(weight, dtype=torch.float32)

        Outlier = torch.Tensor(self.QuantizationObject.outlierIndex.toarray(), dtype=torch.float32)

        with torch.no_grad():
            Outlier = Outlier*self.GradUpdates
            weight += Outlier
            
        return F.linear(input, weight, self.bias)

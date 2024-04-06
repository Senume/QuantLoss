import torch
from  QuantLinearLayer_OutlierTraining import QuantLinear
from  CustomQuantization_OutlierTraining import CustomQuantization as quant

def convertDenseLayer(ModelModule):
    """
    Finds the dense layer in the given module.

    Parameters
    ----------
    ModelModule : torch.nn.Module
        The module in which the dense layer is present.

    Returns
    -------
    torch.nn.Module
        Return the updated module instance
    """
    # For each sub module in the module hierarchyly
    for name, child in ModelModule.named_children():
        # If the sub module is a linear layer
        if isinstance(child, torch.nn.Linear):
            # Replaces with a linear layer with Quantization Layer
            QuantizationObject  = quant()
            layerweight = child.weight.detach().numpy().copy()

            QuantizationObject.extractRange(layerweight)
            QuantizationObject.proceedQuantization(layerweight)
            Error = layerweight - QuantizationObject.dequantize()
            
            QuantLayer = QuantLinear(QuantizationObject, child.bias, Error)
            setattr(ModelModule, name, QuantLayer)
        else:
            # Checking in the next level
            convertDenseLayer(child)

    # Returning next module
    return ModelModule


import torch                                                                        #type:ignore
from  outlier_training.QuantLinearLayer_OutlierTraining import OutlierQuantLinear
from  outlier_training.CustomQuantization_OutlierTraining import CustomQuantization as quant


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
            layerweight = child.weight.clone().detach()

            QuantizationObject.extractRange(layerweight)
            QuantizationObject.proceedQuantization(layerweight)

            outlier = QuantizationObject.outlierIndex
            quant_weight = QuantizationObject.dequantize()
            error = layerweight - quant_weight
            
            QuantLayer = OutlierQuantLinear(quant_weight,outlier, error, child.bias )
            setattr(ModelModule, name, QuantLayer)
        else:
            # Checking in the next level
            convertDenseLayer(child)

    # Returning next module
    return ModelModule


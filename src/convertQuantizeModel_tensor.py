import torch                                                        # type: ignore
from  QuantLinearLayer import QuantLinear
from CustomQuantization_tensor import CustomQuantization as quant
import os

def convertDenseLayer(ModelModule, specific_name, requires_plot=False):
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
    if requires_plot:
        try:
            os.mkdir(f'./{name}_plots/')
        except:
            print("Folder 'plots/' already exists.")

    # For each sub module in the module hierarchyly
    for name, child in ModelModule.named_children():
        # If the sub module is a linear layer
        if isinstance(child, torch.nn.Linear):
            print('\nLayer Name:', name)

            # Replaces with a linear layer with Quantization Layer
            QuantizationObject  = quant()
            layerweight = child.weight.clone().detach()
            QuantizationObject.extractRange(layerweight, save_plot=requires_plot, plot_path=f'./plots/{name}.png', sensitivity=0.5)
            QuantizationObject.proceedQuantization(layerweight)
            
            QuantLayer = QuantLinear(QuantizationObject.dequantize(), child.bias)
            setattr(ModelModule, name, QuantLayer)
        else:
            # Checking in the next level
            convertDenseLayer(child, specific_name, requires_plot=requires_plot)

    # Returning next module
    return ModelModule


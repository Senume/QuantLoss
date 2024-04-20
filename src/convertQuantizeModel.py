import torch # type: ignore
from  QuantLinearLayer import QuantLinear
from CustomQuantization import CustomQuantization as quant
import os

def convertDenseLayer(ModelModule, requires_plot=False):
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
            os.mkdir('./plots/')
        except:
            print("Folder 'plots/' already exists.")

    # For each sub module in the module hierarchyly
    for name, child in ModelModule.named_children():
        # If the sub module is a linear layer
        if isinstance(child, torch.nn.Linear):
            print('Layer Name:', name)
            # Replaces with a linear layer with Quantization Layer
            QuantizationObject  = quant()
            layerweight = child.weight.detach().numpy().copy()

            QuantizationObject.extractRange(layerweight, save_plot=requires_plot, plot_path=f'./plots/{name}.png')

            QuantizationObject.proceedQuantization(layerweight)
            
            QuantLayer = QuantLinear(QuantizationObject, child.bias)
            setattr(ModelModule, name, QuantLayer)
        else:
            # Checking in the next level
            convertDenseLayer(child)

    # Returning next module
    return ModelModule
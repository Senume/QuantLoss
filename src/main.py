import argparse
import torch                                                # type: ignore
from convertQuantizeModel_tensor import convertDenseLayer as tensorQuant
# from numpy_support.convertQuantizeModel import convertDenseLayer as cpuQuant

def main():
    parser = argparse.ArgumentParser(description='Quantization command')

    parser.add_argument('--modelpath', type=str,
                    help="model's directory which needs to be quantized.")    
    parser.add_argument('--outputpath', type=str,
                    help="directory where the quantized model to be saved.")
    parser.add_argument('--name', type=str,
                    help="name of the model.")  
    parser.add_argument('--tensor', type=str, default="True", 
                    help="check if tensor is needed to use.")  
    
    args = parser.parse_args()

    model = torch.load(args.modelpath)

    if args.tensor == "True":
        quant_model = tensorQuant(model)
    # elif args.tensor == "False":
    #     quant_model = cpuQuant(model)
    else:
        raise ValueError("Please enter either True or False")
        
    torch.save(quant_model, args.outputpath + args.name + ".pt")

if __name__ == "__main__":
    main()




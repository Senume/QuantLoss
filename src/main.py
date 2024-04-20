import argparse
import torch                                                # type: ignore
from convertQuantizeModel_tensor import convertDenseLayer

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--modelpath', type=str,
                    help="model's directory which needs to be quantized.")    
    parser.add_argument('--outputpath', type=str,
                    help="directory where the quantized model to be saved.")
    parser.add_argument('--name', type=str,
                    help="name of the model.")  
    args = parser.parse_args()

    model = torch.load(args.modelpath)
    quant_model = convertDenseLayer(model)
    torch.save(quant_model, args.outputpath + args.name + ".pt")



    

if __name__ == "__main__":
    main()




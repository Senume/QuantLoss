import argparse
import torch                                                                                # type: ignore
from convertQuantizeModel_tensor import convertDenseLayer as tensorQuant
from transformers import AutoModelForCausalLM, AutoTokenizer                                # type: ignore

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

    if '.pt' in args.modelpath:
        model = torch.load(args.modelpath)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.modelpath, trust_remote_code=True)


    if args.tensor == "True":
        quant_model = tensorQuant(model, specific_name= args.name, requires_plot=False)
    # elif args.tensor == "False":
    #     quant_model = cpuQuant(model)
    else:
        raise ValueError("Please enter either True or False")
        
    if '.pt' in args.modelpath:
        model = torch.save(quant_model, args.modelpath + args.name + '_quant.pt')
    else:
       quant_model.save_model(args.outputpath)

if __name__ == "__main__":
    main()




from transformers import LlamaTokenizer, LlamaForCausalLM                                   # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer                                # type: ignore
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration  # type: ignore
import torch                                                                                # type: ignore

# Define the model name
# bert_model_names = ['bert-base-uncased', 'bert-large-uncased', 'bert-base-chinese', 'bert-base-multilingual-uncased']
# t5_model_names = ['google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large']
openllama_names = ['openlm-research/open_llama_3b']
phi_names = ["microsoft/Phi-3-mini-128k-instruct"]


# for bert_model_name in bert_model_names:
#     print("Downloading model: ", bert_model_name)
#     # Bert Model
#     safe_model_name = bert_model_name.replace("/", "-")
#     bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#     bert_model = BertModel.from_pretrained(bert_model_name)

#     # Save Bert models
#     torch.save(bert_model, f"./model_saved/{safe_model_name}.pt")
#     torch.save(bert_tokenizer, f"./model_saved/{safe_model_name}_tokenizer.pt")

# for t5_model_name in t5_model_names:
#     print("Downloading model: ", t5_model_name)
#     # T5 Model - Replace '/' with '_'
#     safe_t5_model_name = t5_model_name.replace("/", "-")  # replace '/' with '-' for filename compatibility

#     # T5 Tokenizer and Model
#     t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
#     t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

#     # Save T5 models with modified names
#     torch.save(t5_model, f"./model_saved/{safe_t5_model_name}.pt")
#     torch.save(t5_tokenizer, f"./model_saved/{safe_t5_model_name}_tokenizer.pt")


# for openllama in openllama_names:

#     # Openllama and Model
#     llama_tokenizer = LlamaTokenizer.from_pretrained(openllama)
#     llama_model = LlamaForCausalLM.from_pretrained(openllama, torch_dtype=torch.float16)

#     # Save T5 models with modified names
#     torch.save(llama_model, f"./model_saved/{openllama}.pt")
#     torch.save(llama_tokenizer, f"./model_saved/{openllama}_tokenizer.pt")

# # Download and save Phi models
# for phi_name in phi_names:
#     model = AutoModelForCausalLM.from_pretrained(phi_name, torch_dtype="auto", trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained(phi_name, trust_remote_code=True)
#     torch.save(model, f"./model_saved/{phi_name}.pt")
#     torch.save(tokenizer, f"./model_saved/{phi_name}_tokenizer.pt")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

print(model)
for name, module in model.named_children():
    print(name)

# model.save_model(f"./model_saved/microsoft/phi2")
# tokenizer.save_model(f"./model_saved/microsoft/phi-2_tokenizer")

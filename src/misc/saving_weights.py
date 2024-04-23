import torch, pickle
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration

# Model name and directory name
bert_model_name = 'bert-base-uncased'
t5_model_name = 'google-t5/t5-small'

# loading each model
bert_model = BertModel.from_pretrained(f'./src/saved/{bert_model_name}')
t5_model = T5ForConditionalGeneration.from_pretrained(f'./src/saved/{t5_model_name}')

bert_weight_dict = {}
for name, para in bert_model.named_parameters():
    bert_weight_dict[name] = para.detach().numpy()


t5_weight_dict = {}
for name, para in t5_model.named_parameters():
    t5_weight_dict[name] = para.detach().numpy()

with open(f'src/saved/weights_only/{bert_model_name}_weights.pkl', 'wb') as file:
    pickle.dump(bert_weight_dict, file)

with open(f'src/saved/weights_only/google-t5-small_weights.pkl', 'wb') as file:
    pickle.dump(t5_weight_dict, file)
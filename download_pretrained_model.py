from transformers import BertModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration

# Define the model name
bert_model_name = 'bert-base-uncased'
t5_model_name = 'google-t5/t5-small'

# Bert Model
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# T5 small Model
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Optionally, save the model to a specific directory
bert_model.save_pretrained(f'./src/saved/{bert_model_name}')
t5_model.save_pretrained(f'./src/saved/{t5_model_name}')


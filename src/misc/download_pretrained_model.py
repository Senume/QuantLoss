from transformers import BertModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration       # type: ignore
import torch                                                                                     # type: ignore

# Define the model name
bert_model_names = ['bert-base-uncased', 'bert-large-uncased']
t5_model_names = ['google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large', 'google-bert/bert-large-uncased']

for bert_model_name in bert_model_names:
    print("Downloading model: ", bert_model_name)
    # Bert Model
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)

    torch.save(bert_model, f"./model_saved/{bert_model_name}.pt")
    torch.save(bert_tokenizer, f"./model_saved/{bert_model_name}_tokenizer.pt")

for t5_model_name in t5_model_names:
    print("Downloading model: ", t5_model_name)
    # T5 small Model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    torch.save(t5_model, f"./model_saved/{t5_model_name}.pt")
    torch.save(t5_tokenizer, f"./model_saved/{t5_model_name}_tokenizer.pt")
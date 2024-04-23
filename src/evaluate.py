from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# Import library
import numpy as np
import torch
import sys

# Adding path to py files
sys.path.append('../src')
from transformers import BertModel, BertTokenizer

# Importing the custom quantization module
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

#output file

# get the embeddings
dataset = load_dataset("glue", "sst2")

train_sentences = dataset["train"]["sentence"]
test_sentences = dataset["test"]["sentence"]

train_labels = dataset["train"]["label"]
test_labels = dataset["test"]["label"]

nos_train = len(train_sentences)
nos_test = len(test_sentences)

# get the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

path_models = ['bert-large-uncased','bert-large-uncased-quant.pt']

# determine the ANN
ANN = Sequential()

# For orignal model and quantized model
for model in path_models:

    with open(f'results.txt', 'w') as results:

        embeddings = []
        if "pt" in model:
            TestModel = torch.load(model)
            model = model.split(".")[0]
        else:
            TestModel = BertModel.from_pretrained(model)

        results.write(f"{model} \n")

        for i , sentence in enumerate(train_sentences):
            if i%1000 == 0:
                print(f"{i}/{nos_train+nos_test}")
            tokenized_text = tokenizer(sentence, return_tensors='pt')

            with torch.no_grad():
                outputs = TestModel(**tokenized_text)

            hidden_states = outputs.last_hidden_state
            sentence_embedding = torch.mean(hidden_states, dim=1)
            embeddings.append(sentence_embedding)

            if i == nos_train-1 :
                np.save(f"{model}Train",np.array(embeddings).reshape(nos_train,len(embeddings[0])))
                embeddings = []
            elif i == nos_test + nos_train -1 :
                np.save(f"{model}Test",np.array(embeddings).reshape(nos_test,len(embeddings[0])))
                embeddings = []

        train_embeddings = np.load(f"{model}Train.npy")
        test_embeddings = np.load(f"{model}Test.npy")
        
        ANN.add(Dense(128, input_shape=(train_embeddings.shape[1],), activation='relu'))
        ANN.add(Dense(64, activation='relu'))  
        ANN.add(Dense(1, activation='sigmoid')) 
        ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        ANN.fit(train_embeddings, train_labels, epochs=15, batch_size=32, validation_split=0.2)


        train_loss, train_accuracy = ANN.evaluate(train_embeddings, train_labels)
        test_loss, test_accuracy = ANN.evaluate(test_embeddings, test_labels)

        results.write(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}\n")
        results.write(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n \n")

        predictions = ANN.predict(test_embeddings)

        results.write("Confusion Matrix \n"+ str(confusion_matrix(test_labels, predictions)) + "\n")
        results.write(str(classification_report(test_labels, predictions)) + "\n")
    
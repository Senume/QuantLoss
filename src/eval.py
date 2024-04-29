import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
# Import library
import numpy as np
import torch.nn as nn
import sys
# Adding path to py files
sys.path.append('../src')
from transformers import BertModel, BertTokenizer
# Importing the custom quantization module
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
from tensorflow.keras.utils import to_categorical

# Set GPU devices
torch_device = torch.device("cuda:1")
tf_device = tf.device("/GPU:0")
print("Torch Device:", torch_device)
print("TensorFlow Device:", tf_device)


parser = argparse.ArgumentParser(description='Example CLI')
parser.add_argument('-o', '--output', type=str, default='results.txt', help='Output file name (default: output.txt)')
parser.add_argument('-d', '--dataset', type=str, default='glue', help='Dataset')
parser.add_argument('-l', '--bert_large', type=str, default='bert-large-uncased-quant.pt', help='Quantized bert large location')
parser.add_argument('-b', '--bert_base', type=str, default='bert-base-uncased-quant.pt', help='Quantized bert base location')
parser.add_argument('-e', '--epochs', type=int, default=10, help='train epochs')
parser.add_argument('-s', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-eb', '--embeddings', action='store_true', help='embeddings state (True if embeddings exist)')
args = parser.parse_args()

#output file
directory = os.path.dirname(args.output) 
if not os.path.exists(directory):
    os.makedirs(directory)
results = open(args.output, 'w')


models = ['bert-large-uncased',args.bert_large,'bert-base-uncased',args.bert_base]
# models = ['bert-base-uncased']


if args.dataset == "glue":
    dataset = load_dataset("glue", "sst2")
    sentences = dataset["train"]["sentence"] + dataset["validation"]["sentence"]
    labels = dataset["train"]["label"] + dataset["validation"]["label"]
    nos_train = 51165
    orig_train_labels = labels[:nos_train]
    orig_test_labels = labels[nos_train:]
else:
    dataset = pd.read_csv(args.dataset)
    sentiment = list(dataset.label)
    sentences = list(dataset.content)
    train_sentences,test_sentences,train_labels,test_labels = train_test_split(sentences,sentiment, test_size=0.3, random_state=37)
    orig_train_sentences,orig_test_sentences,orig_train_labels,orig_test_labels = train_sentences,test_sentences,train_labels,test_labels
data = args.dataset.split("/")[-1].split(".")[0]
print("Loaded Data :",data)
nos_train = len(orig_train_labels)

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

batch_size = 512
from tqdm import tqdm
# models
for model in models:
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(f'bert-{"base" if "base" in model else "large"}-uncased')
    embeddings = []
    if "pt" in model:
        TestModel = torch.load(model)
        modelName = model.split("/")[-1].split(".")[0]
        TestModel = TestModel
    else:
        TestModel = BertModel.from_pretrained(model)
        modelName = model
        TestModel = TestModel
    TestModel = TestModel.to(torch_device)  
    
    results.write(f"{modelName} {data} \n")

    if args.embeddings == False:
        # Batch processing
        all_embeddings = []
        TestModel = TestModel
        for i in tqdm(range(0, len(sentences), batch_size), desc="Batches"):
            batch_sentences = sentences[i: i + batch_size]
            tokenized_batch = tokenizer(batch_sentences, pad_to_max_length=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                tokenized_batch = tokenized_batch.to(torch_device)
                outputs = TestModel(**tokenized_batch)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                all_embeddings.append(batch_embeddings)
        
        embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()

        train_embeddings = embeddings[:nos_train]
        test_embeddings = embeddings[nos_train:]
        np.save(f"embeddings/{modelName}-{data}-train.npy",train_embeddings)
        np.save(f"embeddings/{modelName}-{data}-test.npy",test_embeddings)
    train_embeddings = np.array(np.load(f"embeddings/{modelName}-{data}-train.npy"))
    test_embeddings = np.array(np.load(f"embeddings/{modelName}-{data}-test.npy"))

    ANN = keras.Sequential()
    ANN.add(Dense(128, input_shape=(train_embeddings.shape[1],), activation='relu'))
    ANN.add(Dense(64, activation='relu'))
    
    if data == "glue":
        ANN.add(Dense(1, activation='sigmoid'))
        ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        train_labels = orig_train_labels
        test_labels = orig_test_labels
    else:
        ANN.add(Dense(13, activation='softmax'))
        ANN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        train_labels = to_categorical(orig_train_labels, num_classes=13)
        test_labels_hot = to_categorical(orig_test_labels, num_classes=13)

    with tf_device:
        ANN.fit(train_embeddings,np.array(train_labels), epochs=10, batch_size=32)
 
    train_loss, train_accuracy = ANN.evaluate(train_embeddings, np.array(train_labels))
    results.write(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}\n")

    predictions = ANN.predict(test_embeddings)
    if data == "glue":
        predictions = (np.squeeze(predictions)>0.5).astype(int)
    else:
        predictions = (np.argmax(np.squeeze(predictions),axis=1))
        print(predictions)
        
    results.write("Confusion Matrix \n"+ str(confusion_matrix(test_labels, predictions)) + "\n")
    print(str(confusion_matrix(test_labels, predictions)))
    results.write(str(classification_report(test_labels, predictions)) + "\n \n \n ")
    print(str(classification_report(test_labels, predictions)))
results.close()
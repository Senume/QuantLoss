from outlier_training.convertQuantizeModel_OutlierTraining import convertDenseLayer

from datasets import load_dataset                       #type: ignore

from torch.utils.data import Dataset, DataLoader        #type: ignore
from torch.optim import Adam                            #type: ignore
from torch.nn import CrossEntropyLoss                   #type: ignore
import torch.nn as nn, torch                            #type: ignore

from tqdm import tqdm                                   #type: ignore
from sklearn.metrics import classification_report       #type: ignore
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################################################

parser = argparse.ArgumentParser(description='Inference the quantised model command')

parser.add_argument('--path', type=str,
                help="directory where the model and inference is saved.")    
parser.add_argument('--Token_max_length', type=int,
                help="Maximun number of tokens considered.")
parser.add_argument('--ModelPath', type=str,
                help="Model Path.")  
parser.add_argument('--TokenizerPath', type=str, default="True", 
                help="Tokenizer Path.")  
parser.add_argument('--TokenizerPath', type=str, default="True", 
                help="Tokenizer Path.")
parser.add_argument('--EpochCycle', type=int, default=5, 
                help="Number of Epochs.")
parser.add_argument('--Embeddings_length', type=int, default=768, 
                help="Embeddings length.")

args = parser.parse_args()

path = args.path
Token_max_length = args.Token_max_length
ModelPath = args.ModelPath
TokenizerPath = args.TokenizerPath
EpochCycle = args.EpochCycle
Embeddings_length = args.Embeddings_length

#########################################################################################

class TextDataset(Dataset):
    def __init__(self, data):
        texts, labels = data
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label
    
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_classes)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sig(out)

        return out
    
#########################################################################################

# Dataset Loading
dataset = load_dataset("glue", "sst2")
train_sentences, train_labels = dataset["train"]["sentence"], dataset["train"]["label"]
test_sentences, test_labels = dataset["validation"]["sentence"], dataset["validation"]["label"]
unique_class_count = len(set(train_labels))

train_dataset = TextDataset(data=(train_sentences, train_labels))
test_dataset = TextDataset(data=(test_sentences, test_labels))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

#########################################################################################

# Pretrained model weights and tokenizer
Pretrained_model = torch.load(ModelPath).to(device)
tokenizer = torch.load(TokenizerPath)

# Classifier Model
Classifier_model = SimpleClassifier(input_size=Embeddings_length, output_classes=unique_class_count).to(device)

# Optimizer and Loss function
optimizer = Adam(Classifier_model.parameters(), lr=5e-3)
loss_fn = CrossEntropyLoss()

#########################################################################################
print("Training the Classifier model")
for epoch in range(EpochCycle): 
    
    running_loss = 0.0
    correct = 0
    total = 0

    for text, labels in tqdm(train_dataloader):
        optimizer.zero_grad()

        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = Pretrained_model(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )
        Loss = loss_fn(Prediction.to(torch.float64).to(device), labels.to(torch.float64).to(device))

        Loss.backward()
        optimizer.step()

        running_loss += Loss.item()
        total += labels.size(0)
        correct += (Prediction == labels).sum().item()

    print('Epoch:', epoch+1)
    print('Training Loss:', running_loss / len(train_dataloader))
    print('Training Accuracy:', correct / total)

#########################################################################################
print("Inferencing the original model with Classifier model")

y_pred_test = []
y_actual_test = []

print("---------------Training DataSet---------------")
with torch.no_grad():

    for text, labels in tqdm(train_dataloader):
        
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = Pretrained_model(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )

        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_train = classification_report(y_actual_test, y_pred_test)

print("---------------Testing DataSet---------------")

y_pred_test = []
y_actual_test = []

with torch.no_grad():

    for text, labels in tqdm(test_dataloader):
        
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = Pretrained_model(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )

        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_test = classification_report(y_actual_test, y_pred_test)

# Save the classification report to a text file
with open(path + 'classification_report.txt', 'w') as f:
    f.write("Train set Classification Report\n")
    f.write(report_train)
    f.write("\n")
    f.write("Test set Classification Report\n")
    f.write(report_test)
    f.write("\n")

print("Classification report for original model saved to 'classification_report.txt'")

#########################################################################################

QuantOutlierModel = convertDenseLayer(Pretrained_model.to('cpu')).to(device)
Outliers_optimizer = Adam(QuantOutlierModel.parameters(), lr=5e-3)

#########################################################################################

for epoch in range(EpochCycle): 
    
    running_loss = 0.0
    correct = 0
    total = 0

    for text, labels in train_dataloader:
        Outliers_optimizer.zero_grad()

        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = QuantOutlierModel(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )
        Loss = loss_fn(Prediction.to(torch.float64).to(device), labels.to(torch.float64).to(device))

        Loss.backward()
        Outliers_optimizer.step()

        running_loss += Loss.item()
        total += labels.size(0)
        correct += (Prediction == labels).sum().item()

    print('Epoch:', epoch+1)
    print('Training Loss:', running_loss / len(train_dataloader))
    print('Training Accuracy:', correct / total)

#########################################################################################

print("Inferencing the Quantized model with Classifier model")

y_pred_test = []
y_actual_test = []

print("---------------Training DataSet---------------")
with torch.no_grad():

    for text, labels in tqdm(train_dataloader):
        
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = QuantOutlierModel(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )

        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_train = classification_report(y_actual_test, y_pred_test)

print("---------------Testing DataSet---------------")

y_pred_test = []
y_actual_test = []

with torch.no_grad():

    for text, labels in tqdm(test_dataloader):
        
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = QuantOutlierModel(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )

        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_test = classification_report(y_actual_test, y_pred_test)

# Save the classification report to a text file
with open(path + 'classification_report_quant.txt', 'w') as f:
    f.write("Train set Classification Report\n")
    f.write(report_train)
    f.write("\n")
    f.write("Test set Classification Report\n")
    f.write(report_test)
    f.write("\n")

print("Classification report for original model saved to 'classification_report_quant.txt'")

#########################################################################################

torch.save(Classifier_model, path + 'classifier_model.pt')
torch.save(QuantOutlierModel, path + 'quantized_model.pt')

########################################################################################
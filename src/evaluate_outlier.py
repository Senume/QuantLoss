from outlier_training.convertQuantizeModel_OutlierTraining import convertDenseLayer

from datasets import load_dataset                       #type: ignore

from torch.utils.data import Dataset, DataLoader        #type: ignore
from torch.optim import Adam                            #type: ignore
from torch.nn import CrossEntropyLoss                   #type: ignore
import torch.nn as nn, torch                            #type: ignore

from tqdm import tqdm                                   #type: ignore
from sklearn.metrics import classification_report       #type: ignore
from sklearn.metrics import confusion_matrix            #type: ignore
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'

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
    def __init__(self, data, num_classes):
        texts, labels = data
        self.texts = texts
        self.labels = labels
        self.num_classes = num_classes

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
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        return out
    
#########################################################################################

# Dataset Loading
dataset = load_dataset("glue", "sst2")
train_sentences, train_labels = dataset["train"]["sentence"], dataset["train"]["label"]
# test_sentences, test_labels = dataset["validation"]["sentence"], dataset["validation"]["label"]

unique_class_count = len(set(train_labels))

train_dataset = TextDataset(data=(train_sentences, train_labels), num_classes=unique_class_count)
# test_dataset = TextDataset(data=(test_sentences, test_labels),  num_classes=unique_class_count)

train_dataset, test_dataset= torch.utils.data.random_split(train_dataset, [0.7, 0.3])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

#########################################################################################

# Pretrained model weights and tokenizer
Pretrained_model = torch.load(ModelPath).to(device)
tokenizer = torch.load(TokenizerPath)

# Classifier Model
Classifier_model = SimpleClassifier(input_size=Embeddings_length, output_classes=unique_class_count).to(device)
Classifier_model = Classifier_model.train()

# Optimizer and Loss function
optimizer = Adam(Classifier_model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()


Pretrained_model.train()
Classifier_model.train()
#########################################################################################
print("Training the Classifier model")
for epoch in range(EpochCycle): 
    
    running_loss = 0.0
    correct = 0
    total = 0

    for text, labels in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        labels = labels.to(device)
        
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')
        #print(Tokenised_Sentence)

        Tokenised_Sentence.to(device)
        Output = Pretrained_model(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Loss = loss_fn(Classifier_output, labels)

        Loss.backward()
        optimizer.step()

        running_loss += Loss.item()
        total += labels.size(0)
        pred = torch.argmax(Classifier_output, dim = 1 ).to(torch.float32)
        # gt = torch.argmax(labels, dim = 1 ).to(torch.float32)
        correct += (pred == labels).sum().item()

    print('Epoch:', epoch+1)
    print('Training Loss:', running_loss / len(train_dataloader))
    print('Training Accuracy:', correct / total)

#########################################################################################
Pretrained_model.eval()
Classifier_model.eval()

print("Inferencing the original model with Classifier model")

y_pred_train = []
y_actual_train = []

print("---------------Training DataSet---------------")
with torch.no_grad():

    for text, labels in tqdm(train_dataloader):
        labels = labels.to(device)
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = Pretrained_model(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )
        # labels = torch.argmax(labels, dim = 1 ).to(torch.float32)

        y_pred_train.extend(Prediction.tolist())
        y_actual_train.extend(labels.tolist())


report_train = classification_report(y_actual_train, y_pred_train)
matrix_train = confusion_matrix(y_actual_train, y_pred_train)

print("---------------Testing DataSet---------------")

y_pred_test = []
y_actual_test = []

with torch.no_grad():

    for text, labels in tqdm(test_dataloader):
        labels = labels.to(device)
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = Pretrained_model(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )
        # labels = torch.argmax(labels, dim = 1 ).to(torch.float32)
        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_test = classification_report(y_actual_test, y_pred_test)
matrix_test = confusion_matrix(y_actual_test, y_pred_test)

# Save the classification report to a text file
with open(path + 'classification_report_Rerun.txt', 'w') as f:
    f.write("Train set Classification Report\n")
    f.write(report_train)
    f.write("Confusion matrix on Train")
    f.write(matrix_train.__repr__())
    f.write("\n")
    f.write("Test set Classification Report\n")
    f.write(report_test)
    f.write("Confusion matrix on Train")
    f.write(matrix_test.__repr__())
    f.write("\n")

print("Classification report for original model saved to 'classification_report.txt'")

#########################################################################################

QuantOutlierModel = convertDenseLayer(Pretrained_model.to('cpu')).to(device)
Outliers_optimizer = Adam(QuantOutlierModel.parameters(), lr=0.001)

# Classifier Model
Quant_Classifier_model = SimpleClassifier(input_size=Embeddings_length, output_classes=unique_class_count).to(device)
Quant_Classifier_optimizer = Adam(QuantOutlierModel.parameters(), lr=0.001)

#########################################################################################

QuantOutlierModel.train()
Quant_Classifier_model.train()

for epoch in range(EpochCycle): 
    
    running_loss = 0.0
    correct = 0
    total = 0

    for text, labels in tqdm(train_dataloader):

        Outliers_optimizer.zero_grad()
        labels = labels.to(device)
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = QuantOutlierModel(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Quant_Classifier_model(Embeddings_Output)

        Loss = loss_fn(Classifier_output, labels)

        Loss.backward()
        Outliers_optimizer.step()
        Quant_Classifier_optimizer.step()

        running_loss += Loss.item()
        total += labels.size(0)
        pred = torch.argmax(Classifier_output, dim = 1 ).to(torch.float32)
        correct += (pred == labels).sum().item()

    print('Epoch:', epoch+1)
    print('Training Loss:', running_loss / len(train_dataloader))
    print('Training Accuracy:', correct / total)

#########################################################################################
QuantOutlierModel.eval()
Quant_Classifier_model.eval()

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

        Classifier_output = Quant_Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )
        labels = labels.to(device)
        # labels = torch.argmax(labels, dim = 1 ).to(torch.float32)
        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_train = classification_report(y_actual_test, y_pred_test)
matrix_train = confusion_matrix(y_actual_test, y_pred_test)

print("---------------Testing DataSet---------------")

y_pred_test = []
y_actual_test = []

with torch.no_grad():

    for text, labels in tqdm(test_dataloader):
        
        Tokenised_Sentence = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

        Tokenised_Sentence.to(device)
        Output = QuantOutlierModel(**Tokenised_Sentence)
        Embeddings_Output = torch.sum(Output.last_hidden_state, dim= 1)

        Classifier_output = Quant_Classifier_model(Embeddings_Output)
        Prediction = torch.argmax(Classifier_output, dim = 1 )
        labels = labels.to(device)
        # labels = torch.argmax(labels, dim = 1 ).to(torch.float32)
        y_pred_test.extend(Prediction.tolist())
        y_actual_test.extend(labels.tolist())


report_test = classification_report(y_actual_test, y_pred_test)
matrix_test = confusion_matrix(y_actual_test, y_pred_test)

# Save the classification report to a text file
with open(path + 'classification_report_quant_rerun.txt', 'w') as f:
    f.write("Train set Classification Report\n")
    f.write(report_train)
    f.write("Confusion matrix on Train")
    f.write(matrix_train.__repr__())
    f.write("\n")
    f.write("Test set Classification Report\n")
    f.write(report_test)
    f.write("Confusion matrix on Train")
    f.write(matrix_test.__repr__())
    f.write("\n")

print("Classification report for original model saved to 'classification_report_quant.txt'")

#########################################################################################

torch.save(Classifier_model, path + '_classifier_model_rerun.pt')
torch.save(QuantOutlierModel, path + '_quantized_model_rerun.pt')
torch.save(Quant_Classifier_model, path + '_quant_classifier_model_rerun.pt')


########################################################################################


import torch
import spacy

import numpy as np
import sklearn as sk
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")

def text2token(file_name: str):
    with open(file_name, 'r') as file:
        raw_data = file.readlines()
    doc = [nlp(sentence) for sentence in raw_data]
    all_tokens = [token for token in doc]
    flat_tokens = [token for sublist in all_tokens for token in sublist]
    token_vectors = [token.vector for token in flat_tokens]
    return token_vectors

def create_data_loaders(vector_list: list, train_or_test: str, order: str):
    vectorised_data = np.array(vector_list)
    label_list = generate_labels(len(vectorised_data) // 2, order)
    label_list.append(1)
    label_array = np.array(label_list)
    data_train, data_test, label_train, label_test \
    = train_test_split(vectorised_data, label_array, test_size = 0.2, random_state = None)

    data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
    data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train, dtype=torch.long)
    label_test_tensor = torch.tensor(label_test, dtype=torch.long)

    train_dataset = TensorDataset(data_train_tensor, label_train_tensor)
    test_dataset = TensorDataset(data_test_tensor, label_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if train_or_test == "train":
        return train_loader
    else:
        return test_loader
    

def count(tokenList: list):
    return Counter(tokenList)

def generate_labels(num_items: int, first: str):
    binary_labels = []
    if first == 'eng':
        for num in range(0, num_items):
            binary_labels.append(1)
        for num in range(0, num_items):
            binary_labels.append(0)
    else:
        for num in range(0, num_items):
            binary_labels.append(0)
        for num in range(0, num_items):
            binary_labels.append(1)
    return binary_labels

class Mad_Hatter(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Mad_Hatter, self).__init__()
        self.fc_layer_1 = nn.Linear(input_size, hidden_size)
        self.fc_layer_2 = nn.Linear(hidden_size, 1) 
        self.relu = nn.ReLU() 

    def feedforward(self, input_batch):
        input_batch = self.fc_layer_1(input_batch)  
        input_batch = self.relu(input_batch)  
        input_batch = self.fc_layer_2(input_batch)  
        return input_batch  
    
    def train_binary_classification(model, data_loader, loss_f, optim, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            loss_accumulation = 0.0 # track total loss for entire epoch
            for data, labels in data_loader:
                optim.zero_grad() # reset gradients of all parameters every iteration
                outputs = model.feedforward(data) # forward pass of input through model (produces raw output scores)
                loss = loss_f(outputs.squeeze(), labels.float()) # compute binary loss between predicted outputs and true labels
                loss.backward() # backward pass for autograd of each parameter
                optim.step() # adjust learning rate
                loss_accumulation += loss.item() # add loss value for current batch 
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_accumulation/len(data_loader):.4f}")

    def evaluate_model(model, data_loader):
        model.eval() # set model to evaluation mode
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, label in data_loader:
                output = model.feedforward(data)
                prob = torch.sigmoid(output)
                predict = (prob > 0.5).float()
                all_predictions.extend(predict.squeeze().tolist())  
                all_labels.extend(label.tolist())
                correct += (predict.squeeze() == label).sum().item()
                total += label.size(0)
            accuracy = 100 * correct / total
            print(f"Testing Accuracy: {accuracy:.2f}%")
            return all_predictions, all_labels, accuracy  


    #def print_results(predict_list, label_list)
                
classifier = Mad_Hatter(96, 64)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr = 0.001)

test = text2token("/Users/ananya/Desktop/training.csv")
loader = create_data_loaders(test, "train", "gib")
classifier.train_binary_classification(loader, loss_function, optimizer, 10)
classifier.evaluate_model(loader)







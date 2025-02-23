
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
dataset_size = 100

def text2token(file_name: str):
    with open(file_name, 'r') as file:
        raw_data = file.readlines()
    doc = [nlp(sentence) for sentence in raw_data]
    all_tokens = [token for token in doc]
    flat_tokens = [token for sublist in all_tokens for token in sublist]
    token_vectors = [token.vector for token in flat_tokens]
    return token_vectors

def create_data_loaders(vector_list: list):
    vectorised_data = np.array(vector_list)
    label_array = np.array(generate_labels(dataset_size, "eng"))
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
    

def count(tokenList: list):
    return Counter(tokenList)

def generate_labels(num_items: int, first: str):
    binary_labels = []
    if first == 'eng':
        for num in range(0, num_items + 1):
            binary_labels.append(1)
        for num in range(0, num_items + 1):
            binary_labels.append(0)
    else:
        for num in range(0, num_items + 1):
            binary_labels.append(0)
        for num in range(0, num_items + 1):
            binary_labels.append(1)
    return binary_labels










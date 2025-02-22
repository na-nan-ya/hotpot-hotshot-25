import torch
import spacy 
import sklearn as sk
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nlp = spacy.load("en_core_web_sm")

def text2vec(file_name: str):
    # read text file dataset 
    with open(file_name, 'r') as file:
        raw_data = file.readlines()
    vectorised_text = []
    for sentence in raw_data:
        # convert to lowercase
        lowered = sentence.lower()
        # treat sentence as a spaCy object
        doc = nlp(sentence)
        # convert tokens in sentence to vector representations (numPy array)
        vectors = [token.vector for token in doc if not token.is_punct]
        # rejoin processed sentence + append to list
        vectorised_text.append(vectors)
    return vectorised_text

def 






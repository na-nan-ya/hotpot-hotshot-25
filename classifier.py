import torch
import spacy 
import sklearn as sk
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nlp = spacy.load("en_core_web_sm")

# converts raw text data (sentences) to a list of their tokenised forms
def text2token(file_name: str):
    # read input file
    with open(file_name, 'r') as file:
        raw_data = file.readlines()
    tokenised_text = []
    # tokenise each sentence individually
    for sentence in raw_data:
        # convert to lowercase
        lowered = sentence.lower()
        doc = nlp(sentence) # treat sentence as a spaCy object
        # convert sentence to a list of tokens
        tokens = [token.text for token in doc if not token.is_punct] # ignore punctuation
        tokenised_text.append(" ".join(tokens)) # add list of tokens to main list
    return tokenised_text






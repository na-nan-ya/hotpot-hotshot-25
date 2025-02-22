import torch
import spacy 
import sklearn as sk
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nlp = spacy.load("en_core_web_sm")

# converts raw text data (sentences) to a list of their tokenised forms
def process_text(file_name: str):
    # read input file
    with open(file_name, 'r') as file:
        raw_data = file.readlines()
    processed_text = []
    # process each sentence individually
    for sentence in raw_data:
        # convert to lowercase
        lowered = sentence.lower()
        doc = nlp(sentence) # treat sentence as a spaCy object
        # convert sentence to a list of tokens and lemmatise before adding to list
        tokens = [token.lemma_ for token in doc if not token.is_punct] # ignore punctuation
        # convert list of tokens to a list of embedded vectors
        vectors = token2vector(tokens)
        processed_text.append(vectors) # add list of vectors to main list
    return processed_text

# helper method to convert a list of tokens to respective embedded vectors
def token2vector(tokenList: list):
    embedded_vector = [token.vector for token in tokenList]
    return embedded_vector
    







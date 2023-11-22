# Importing standard libraries for every machine/deep learning pipeline
import pandas as pd
import torch
from tqdm import tqdm, trange
import numpy as np


# Importing specific libraries for data prerpcessing, model archtecture choice, training and evaluation
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import AdamW
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns



from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("camembert-base")

def tokenize_texts(texts):
    return [tokenizer(text, padding=True, truncation=True, max_length=512) for text in texts]

# Supposons que `cleaned_texts` est votre liste de phrases nettoyées
tokenized_texts = tokenize_texts(data_cleaning)


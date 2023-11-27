#!pip3 install tokenizer
#!pip3 install sentencepiece
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import CamembertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from tqdm import trange
import nltk
import tokenizer as tokenizer_2
import re
from nltk.tokenize import word_tokenize
import string

epochs = 20
MAX_LEN = 128
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset, I selected only 5000 sample because of memory limitation
df = pd.read_csv('training_data.csv').reset_index(drop=True)
df.head()

# Mapping des valeurs de la colonne "difficulty"
difficulty_mapping = {
    'A1': 0,
    'A2': 1,
    'B1': 2,
    'B2': 3,
    'C1': 4,
    'C2': 5
}

# Utiliser la fonction map pour encoder les valeurs
df['difficulty_encoded'] = df['difficulty'].map(difficulty_mapping)

unique_labels = df['difficulty_encoded'].unique()
print(unique_labels)

# Creates list of texts and labels
text = df['sentence'].to_list()
labels = df['difficulty_encoded'].to_list()  # Utilisez les labels encodés

# Utilisez le tokenizer Camembert
tokenizer = CamembertTokenizer.from_pretrained("camembert-base", do_lower_case=True)


# Utilisez le tokenizer pour convertir les phrases en tokens
input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True, truncation=True) for sent in text]

# Créez des masques d'attention
attention_masks = []
# Créez un masque de 1 pour chaque token suivi de 0 pour le padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

    # Convertissez les listes en tenseurs PyTorch
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

# Créez un DataLoader pour gérer les lots de données
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset))

# Vous pouvez maintenant utiliser dataloader pour l'entraînement de votre modèle.
# Use train_test_split to split our data into train and validation sets for training
train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(input_ids, labels, attention_masks,
                                                            random_state=42, test_size=0.2)


# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs).to(device)
validation_inputs = torch.tensor(validation_inputs).to(device)
train_labels = torch.tensor(train_labels).to(device)
validation_labels = torch.tensor(validation_labels).to(device)
train_masks = torch.tensor(train_masks).to(device)
validation_masks = torch.tensor(validation_masks).to(device)


# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=6)
model.to(device)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
from transformers import AdamW
from sklearn.metrics import accuracy_score

# Define the optimizer and set the learning rate
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


# Store our loss and accuracy for plotting if we want to visualize training evolution per epochs after the training process
train_loss_set = []

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):  
    # Tracking variables for training
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
  
    # Train the model
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Add batch to device CPU or GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        # Get loss value
        loss = outputs.loss
        # Add it to train loss list
        train_loss_set.append(loss.item())    
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
# Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # Tracking variables for validation
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Validation of the model
    model.eval()
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to device CPU or GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs =  model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
    
        # Move logits and labels to CPU if GPU is used
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# Charger le nouveau jeu de données
# Remplacez 'new_data.csv' par le chemin de votre fichier de nouvelles phrases
new_df = pd.read_csv('unlabelled_test_data.csv')
new_texts = new_df['sentence'].tolist()  # Assurez-vous que la colonne contient les phrases

# Préparer les données pour le modèle
tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
new_input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True, truncation=True) for sent in new_texts]
new_attention_masks = [[float(i > 0) for i in seq] for seq in new_input_ids]

# Convertir en tenseurs
new_input_ids = torch.tensor(new_input_ids)
new_attention_masks = torch.tensor(new_attention_masks)

# Créer un DataLoader
prediction_data = TensorDataset(new_input_ids, new_attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prédiction
model.eval()
predictions = []

for batch in prediction_dataloader:
    # Ajouter batch à GPU
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    predictions.append(logits)

# Convertir les prédictions en étiquettes de difficulté
predicted_labels = [np.argmax(p, axis=1).flatten() for p in predictions]
predicted_labels = np.concatenate(predicted_labels)

# Créer un DataFrame pour le CSV
output_df = pd.DataFrame({
    'id': new_df.index,  # ou une autre colonne d'identification si disponible
    'difficulty': [list(difficulty_mapping.keys())[list(difficulty_mapping.values()).index(label)] for label in predicted_labels]
})

# Enregistrer en CSV
output_df.to_csv('predicted_difficulties4.csv', index=False)





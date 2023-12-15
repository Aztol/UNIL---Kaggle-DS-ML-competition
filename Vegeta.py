import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from tqdm import trange
import nltk
import tokenizer
import re
from nltk.tokenize import word_tokenize
import string
from sklearn.metrics import accuracy_score

MAX_LEN = 128
batch_size = 16
difficulty_mapping = {
    'A1': 0,
    'A2': 1,
    'B1': 2,
    'B2': 3,
    'C1': 4,
    'C2': 5
}
# Charger le meilleur modèle
# Remplacez 'best_model_path' par le chemin de votre meilleur modèle sauvegardé
model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=len(difficulty_mapping))

# Load the state dictionary
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
device = torch.device("cpu")
model.to(device)

# Charger le nouveau jeu de données
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
output_df.to_csv('predicted_difficulties.csv', index=False)

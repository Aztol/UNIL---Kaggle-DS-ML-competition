import streamlit as st
import pandas as pd
import torch
from transformers import CamembertTokenizer, AutoModelForSequenceClassification
import numpy as np

# Autres bibliothèques nécessaires

difficulty_mapping = {
    0: 'A1',
    1: 'A2',
    2: 'B1',
    3: 'B2',
    4: 'C1',
    5: 'C2'
}

model_path = '/Users/Gaetan_1/Desktop/model_fromage/model_weights.pt'  # Le dossier où se trouve votre fichier model_weights.pt
tokenizer_path = '/Users/Gaetan_1/Desktop/model_fromage/tokenizer'  # Le dossier où se trouve votre tokenizer

# Charger le tokenizer
tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

# Charger le modèle
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()  # Mettez le modèle en mode d'évaluation


tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)

MAX_LEN = 512  # Définir la longueur maximale

st.title("Détecteur de niveau CEFR")

user_text = st.text_area("Entrez votre texte ici:", height=250)

if st.button("Soumettre"):
    try:
        # Tokenisation
        input_ids = tokenizer.encode(user_text, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True, truncation=True)
        attention_masks = [float(i > 0) for i in input_ids]

        # Conversion en tenseurs
        input_ids = torch.tensor([input_ids])
        attention_masks = torch.tensor([attention_masks])

        # Évaluation du modèle
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        predicted_label_index = np.argmax(logits, axis=1).flatten()[0]
        predicted_label = difficulty_mapping[predicted_label_index]
        # Afficher les résultats (traduire predicted_label en une étiquette significative)
        st.success(f"Prédiction : {predicted_label}")
    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")

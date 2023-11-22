from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("camembert-base")

print("Modèle chargé : ", model)

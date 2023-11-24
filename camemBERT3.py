import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import CamembertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import torch

df = pd.read_csv('training_data.csv')
df.head()


train_texts, val_texts, train_labels, val_labels = train_test_split(df['sentence'], df['difficulty'], test_size=0.2)



tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)


# Tenseurs pour les textes d'entraînement
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])

train_labels = torch.tensor(pd.get_dummies(train_labels).values.tolist())


# Créer un DataLoader
batch_size = 16  # Vous pouvez choisir une taille de lot appropriée en fonction de votre configuration et de la mémoire disponible.
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))


# Charger le modèle CamemBERT
  # Remplacez 3 par le nombre de catégories dans votre tâche de classification
num_labels = 6
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Configuration de l'entraînement
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3  # Définissez le nombre d'époques ici
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()

# Boucle d'entraînement
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        print(len(batch))
        batch = tuple(t.to(device) for t in batch)  # Assurez-vous que tout est transféré sur le bon périphérique
        print(len(batch))
        b_input_ids, b_input_mask, b_labels = batch
        print(b_input_ids.shape) 
        print(b_input_mask.shape)
        print(b_labels.shape)
        # Convertir les étiquettes booléennes en entiers
        b_labels = b_labels.long()
        print(b_labels.shape)
        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss}")

    print(b_input_ids.shape)  # Doit être quelque chose comme (16, seq_length)
    print(b_labels.shape)    # Doit être (16,)
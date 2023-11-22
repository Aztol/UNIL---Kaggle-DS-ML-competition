import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Téléchargement des ressources NLTK, si nécesf
#saire
nltk.download('punkt')
nltk.download('stopwords')

# Chargement des stopwords français
french_stopwords = set(stopwords.words('french'))

def preprocess_text(text):
    """Nettoie et tokenise le texte."""
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères non-alphabétiques
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenisation
    tokens = word_tokenize(text, language="french")
    
    # Suppression des stopwords et de la ponctuation
    tokens = [mot for mot in tokens if mot not in french_stopwords and mot not in string.punctuation]
    
    # Retourne les tokens nettoyés sous forme de chaîne de caractères
    return ' '.join(tokens)

# Charger les données depuis le fichier CSV
df = pd.read_csv('training_data.csv')

# Appliquer le nettoyage à chaque phrase dans la colonne 'sentence'
df['cleaned_text'] = df['sentence'].apply(preprocess_text)

# Sauvegarder les données nettoyées dans un nouveau fichier
df.to_csv('training_data_cleaned.csv', index=False)


#def calculate_levels_score(df, levels):       
    # Check corresponding words and calculate levels_score
    
    #a1a2_words = levels['a1a2'].tolist()
    #b1b2_words = levels['b1b2'].tolist()
    #c1c2_words = levels['c1c2'].tolist()

    #levels_score = 0
    #for index, row in df.iterrows():
        #levels_score = 0
        #current_sentence = row['sentence']
        #word = word_tokenize(current_sentence)
        #if word in a1a2_words:
            #levels_score += 1
        #elif word in b1b2_words:
            #levels_score += 2
        #elif word in c1c2_words:
            #levels_score += 3
        #df.at[index, 'levels_score'] = levels_score
    #return df
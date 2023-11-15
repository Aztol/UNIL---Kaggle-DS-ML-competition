#Cela est une modification
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv("training_data.csv")

print(df.head())
print(df.isnull().sum())

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('french')]

    text = ' '.join(tokens)
    return text

df['clean_text'] = df['sentence'].apply(preprocess_text)

print(df.head())
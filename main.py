#Voici le main, dans lequel on executera les functions
from data_cleaning import *
import nltk

df = pd.read_csv("training_data.csv")
levels = pd.read_csv("vocabulaires")

nltk.download('stopwords')
nltk.download('punkt')

print(df.head())
print(levels.head())
print(df.isnull().sum())

df['clean_text'] = df['sentence'].apply(preprocess_text)
df['stemmer_sentence'] = df['sentence'].apply(stem_sentences)

add_length_column(df)

print(df.head())
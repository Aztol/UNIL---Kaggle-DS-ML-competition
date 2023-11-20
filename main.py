#Voici le main, dans lequel on executera les functions
from data_cleaning import *
import nltk

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv("training_data.csv")
levels = pd.read_csv("vocabulaires.csv", sep=';')


print(nltk.data.path)

print(df.head())
print(levels.head())
print(df.isnull().sum())

#df['clean_text'] = df['sentence'].apply(preprocess_text)
#df['stemmer_sentence'] = df['sentence'].apply(stem_sentences)
df = calculate_levels_score(df, levels)
print((df['levels_score'] != 0.0).sum())

add_length_column(df)

print(df.head())
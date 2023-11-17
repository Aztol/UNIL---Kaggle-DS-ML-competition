#Voici le main, dans lequel on executera les functions
from data_cleaning import *
import nltk

df = pd.read_csv("training_data.csv")
nltk.download('stopwords')
nltk.download('punkt')
print(df.head())
print(df.isnull().sum())
df['clean_text'] = df['sentence'].apply(preprocess_text)

print(df.head())
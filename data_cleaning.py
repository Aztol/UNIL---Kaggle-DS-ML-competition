import re
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer

import string





#french_stopwords = set(stopwords.words('french'))

def preprocess_text(text):
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text, language="french")
    tokens = [mot for mot in tokens if mot.lower() not in stopwords.words('french') and mot not in string.punctuation]


    #stemmer = SnowballStemmer('french')
    #text = [stemmer.stem(mot) for mot in tokens]
    return text

def add_length_column(dataframe):

    dataframe['length'] = dataframe['sentence'].apply(lambda x: len(x))

def stem_sentences(sentence):
    # Tokenize the sentence and apply stemming using Porter Stemmer
    stemmer = SnowballStemmer('french')
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def calculate_levels_score(df, levels):
    # Check corresponding words and calculate levels_score
    
    a1a2_words = levels['a1a2'].tolist()
    b1b2_words = levels['b1b2'].tolist()
    c1c2_words = levels['c1c2'].tolist()

    levels_score = 0
    for index, row in df.iterrows():
        levels_score = 0
        current_sentence = row['sentence']
        word = word_tokenize(current_sentence)
        if word in a1a2_words:
            levels_score += 1
        elif word in b1b2_words:
            levels_score += 2
        elif word in c1c2_words:
            levels_score += 3
        df.at[index, 'levels_score'] = levels_score
    return df
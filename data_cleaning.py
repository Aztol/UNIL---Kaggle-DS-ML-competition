import re
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
import nltk
import string

#

#french_stopwords = set(stopwords.words('french'))

def preprocess_text(text):
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text, language="french")
    tokens = [mot for mot in tokens if mot.lower() not in stopwords.words('french') and mot not in string.punctuation]
    #tokens = [mot for mot in tokens if mot.lower() not in french_stopwords and mot not in string.punctuation]


    #stemmer = SnowballStemmer('french')
    #text = [stemmer.stem(mot) for mot in tokens]
    return text


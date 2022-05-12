
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from matplotlib import pyplot as plt

df = pd.read_csv('../web_scrape_description_final.csv', error_bad_lines=False)

## tokenize the description to split individual words into tokens
df['Web_Description_Tokenized'] = df['Web_Description'].apply(word_tokenize)

##convert text into lower case
df['Web_Description_Tokenized'] = df['Web_Description_Tokenized'].apply(lambda x: [word.lower().strip() for word in x])

##remove stopwords
stops = stopwords.words("english")
df['Web_Description_Tokenized'] = df['Web_Description_Tokenized'].apply(lambda x: [word for word in x if word not in stops])

##remove punctuations
table = str.maketrans("", "", string.punctuation)
df['Web_Description_NoPunctuation'] = df['Web_Description_Tokenized'].apply(lambda x: [word.translate(table) for word in x])
##.apply(lambda x: [word for word in x if word not in punc])


df['Web_Description_Clean'] = df['Web_Description_NoPunctuation'].apply(lambda x: [word for word in x if word])

##add pos tags (useful for lemmatization)
df['Web_Description_PosTag'] = df['Web_Description_Clean'].apply(nltk.tag.pos_tag)

## define a function to get pos tags that can be used in lemmatization 
from nltk.corpus import wordnet
def get_pos_tag(tag):
  if tag.startswith('J'):
    return wordnet.ADJ
  elif tag.startswith('V'):
    return wordnet.VERB
  elif tag.startswith('N'):
    return wordnet.NOUN
  elif tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN


##call get_pos_tag function for creating (word, tag) pair to be used in lemmatization
df['Web_Description_WordnetPos'] = df['Web_Description_PosTag'].apply(lambda x: [(word, get_pos_tag(pos_tag)) for (word, pos_tag) in x])


##lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['Web_Description_Lemmatized'] = df['Web_Description_WordnetPos'].apply(lambda x: [lemmatizer.lemmatize(word, tag) for word, tag in x])


##join the lemmatized words into a single text
df['Web_Description_LemmatizedString'] = [' '.join(map(str,l)) for l in df['Web_Description_Lemmatized']]

## save the cleaned df as csv
df.to_csv('./lemmatized_data.csv', index=False)
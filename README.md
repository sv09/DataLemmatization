# Data Lemmatization

This code uses nltk lemmatizer to lemmatize text from google web scrape result. This is part of the Dark Net Market Research project, and the google web scrape data is the description of the product names found on Silk Road 2.0 marketplace.

Packages used:
- word_tokenize from nltk.tokenize for tokenizing each word in the text.
- stopwords from nltk.corpus for excluding commonly used words from the text.
- wordnet from nltk.corpus for tagging words by parts of speech (pos).
- WordNetLemmatizer from nltk.stem for returning the base form of a word.

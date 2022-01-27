import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import dask.dataframe as ddf
import time
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet


def df_melt(df):
    df = df.melt(value_vars=['Positive_Review', 'Negative_Review'], var_name = 'label', value_name = 'review')
    df["label"].replace({"Positive_Review": 1, "Negative_Review": 0}, inplace=True) #, inplace=True
    return df

def clean(df):

    stopwords = ENGLISH_STOP_WORDS
    df = df_melt(df)
    df["review"] = df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    df['review'] = df['review'].str.lower()
    df['review'].apply(lambda x: [item for item in x if item not in stopwords])
    df['review'] = df['review'].str.replace('\d+', '', regex=True)
    df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def stemming(df):
    # Import the function to perform stemming
    # Call the stemmer
    new_column = []
    porter = PorterStemmer()    # Transform the array of tweets to tokens

    tokens = [word_tokenize(review) for review in df['review']]
    # Stem the list of tokens
    stemmed_tokens = [[porter.stem(word) for word in review] for review in tokens]
    for l in stemmed_tokens:
        stem_sentence = " ".join(l)
        new_column.append(stem_sentence)
        #print(stem_sentence)
    #print(stemmed_tokens)
    df['stem_review'] = new_column
    return df

def prepare():

    df = pd.read_csv("data/Hotel_Reviews.csv")

    df = clean(df)
    df = stemming(df)
    
    

    return df
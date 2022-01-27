import time
import dask.dataframe as ddf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

def df_melt(df):
    df = df.melt(value_vars=['Positive_Review', 'Negative_Review'], var_name = 'label', value_name = 'text')
    df["label"].replace({"Positive_Review": 1, "Negative_Review": 0}, inplace=True)
    #df = df[["text", "label"]]
    return df

# def remove_emails(text):
#         regex =  r'\S*@\S*\s?'
#         return re.sub(regex, '', text)

# def remove_newlinechars(text):
#     regex = r'\s+'
#     return re.sub(regex, ' ', text)

# def tokenize(text):
#     tokens = nltk.word_tokenize(text)  
#     return list(filter(lambda word: word.isalnum(), tokens))

# def remove_stopwords(words):
#     stop_words = ENGLISH_STOP_WORDS
#     filtered = filter(lambda word: word not in stop_words, words)  
#     return list(filtered)

# def clean_text(df):
#     df["cleaned_text"] = df.text.map(lambda text:text.lower()).map(remove_emails).map(remove_newlinechars).map(remove_stopwords)
#     return df

# def dask_cleaning(df):

#     df = df_melt(df)
#     df = df.assign(text=df["text"]).assign(label=df["label"])

#     print(df.info())
#     print(df.head())

#     dask_dataframe = ddf.from_pandas(df, npartitions=6)

#     result = dask_dataframe.map_partitions(clean_text, meta=df)
#     df = result.compute()
#     return df

def tryout(df_original):
    
    df_original = df_melt(df_original)

    import pandas as pd
    df = pd.DataFrame()
    df = df.assign(text=df_original["text"]).assign(target=df_original["label"])

    import re
    # test_text = df_original.text[0]

    def remove_emails(text):
    
    # Remove any email address in text with
    
        regex =  r'\S*@\S*\s?'
        return re.sub(regex, '', text)

    # test_text = remove_emails(df_original.text[0])
    # print(test_text)


    def remove_newlinechars(text):

        #Substitute any newline chars with a whitespace

        regex = r'\s+'
        return re.sub(regex, ' ', text)

    # test_text = remove_newlinechars(test_text)
    # print(test_text)

    import nltk

    def tokenize(text):
    
        #Tokenize text

        tokens = nltk.word_tokenize(text)  
        return list(filter(lambda word: word.isalnum(), tokens))

    # test_text = tokenize(test_text)
    # print(test_text)

    from nltk.corpus import stopwords

    stop_words = stopwords.words("english")

    ## Add some common words from text
    # stop_words.extend(["from","subject","summary","keywords", "article"])

    def remove_stopwords(words):
        # Remove stop words from the list of words
    
        filtered = filter(lambda word: word not in stop_words, words)  
        return list(filtered)

    # test_text = remove_stopwords(test_text)
    # print(test_text)

    import time
    def clean_text(df):

        # Take in a Dataframe, and process it

        df["cleaned_text"] = df.text.map(lambda text:text.lower()).map(remove_emails).map(remove_newlinechars).map(remove_stopwords)
        return df

    # t0 = time.time()
    df = clean_text(df)
    # t1 = time.time()
    # print("Time to process without Dask {}".format(t1-t0))
    # 169.84 sec

    import dask.dataframe as ddf

    dask_dataframe = ddf.from_pandas(df, npartitions=6)

    print('\nCleaning the dataframe\n')
    t0 = time.time()
    result = dask_dataframe.map_partitions(clean_text, meta=df)
    df = result.compute()
    t1 = time.time()
    print("Time to process with Dask {}".format(t1-t0))

    return df
    # 58.795


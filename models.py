from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from dask_ml.feature_extraction.text import HashingVectorizer
import dask.dataframe as dd

import pandas as pd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import time
import pickle

def tfidf(df):
    # Build the vectorizer
    vect = TfidfVectorizer(ngram_range=(1, 2), 
                            max_features=500, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['stem_review'].values.astype('U'))
    # Create paRSE matrix from the vectorizer
    X = vect.transform(df['stem_review'].values.astype('U'))
    # Create a DataFrame
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df

def dask_model(df):
    df = df.head(50000)
    print('\n')
    print('Applying tfidf...')
    print('\n')

    X = tfidf(df)
    y = df.label

    from sklearn import decomposition
    from sklearn.pipeline import Pipeline

    print('\n')
    print('Applying models without Dask...')
    print('\n')

    t0 = time.time()
    logistic = LogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca),
                       ('logistic', logistic)])

    grid = dict(pca__n_components=[50, 100, 250],
            logistic__C=[1e-4, 1.0, 1e4],
            logistic__penalty=['l2'])

    estimator = GridSearchCV(pipe, grid, n_jobs=-1)
    estimator.fit(X, y)
    t1 = time.time()
    print('\n')
    print('Best parameters without Dask: ',estimator.best_params_)
    print('Best score without Dask: ', estimator.best_score_)
    print("Time to process without Dask {}".format(t1-t0))
    print('\n')

    print('\n')
    print('Applying models with Dask...')
    print('\n')

    t0 = time.time()
    logistic = DaskLogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca),
                       ('logistic', logistic)])

    grid = dict(pca__n_components=[50, 100, 250],
            logistic__C=[1e-4, 1.0, 1e4],
            logistic__penalty=['l2'])

    destimator = DaskGridSearchCV(pipe, grid)
    destimator.fit(X, y)
    t1 = time.time()
    print('\n')
    print('Best parameters: ',destimator.best_params_)
    print('Best score with Dask: ', destimator.best_score_)
    print("Time to process with Dask {}".format(t1-t0))
    print('\n')

    

# def dask_model(df):

#     df = df.head(10000)
#     print('\n')
#     print('Applying tfidf...')
#     print('\n')
#     X = tfidf(df)
#     y = df.label
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#     print('\n')
#     print('Applying model without Dask...')
#     print('\n')

#     t0 = time.time()
#     logistic = LogisticRegression(C = 1, penalty = 'l2')
    
#     logistic.fit(X_train, y_train)
#     y_pred = logistic.predict(X_test)
#     t1 = time.time()
#     print('\n')
#     print('\nClassification report:\n')
#     print('Without Dask: ', classification_report(y_test, y_pred))
#     print("Time to process without Dask {}".format(t1-t0))
#     print('\n')

#     #######################################################################################################
#     # DASK
#     #######################################################################################################

#     print('\n')
#     print('Applying models with Dask...')
#     print('\n')

#     t0 = time.time()
#     logistic_dask = DaskLogisticRegression(C = 0.0001, penalty = 'l2')
    
#     logistic_dask.fit(X_train, y_train)
#     y_pred_dask = logistic_dask.predict(X_test)
#     t1 = time.time()
#     print('\n')
#     print('\nClassification report:\n')
#     print('Without Dask: ', classification_report(y_test, y_pred_dask))
#     print("Time to process with Dask {}".format(t1-t0))
#     print('\n')


from sklearn.exceptions import ConvergenceWarning


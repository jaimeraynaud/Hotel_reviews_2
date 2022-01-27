import pandas as pd
import numpy as np
from numpy import array

# TextMining
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GRU
from keras.layers.embeddings import Embedding

from matplotlib import pyplot as plt

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM

def neural_network(df):
    #df = df.head(10000)
    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    def preprocess_text(sen):
        # Removing html tags, punctuations and numbers, single character removal and removing multiple spaces
        sentence = remove_tags(sen)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    X = []
    sentences = list(df['stem_review'])
    for sen in sentences:
        X.append(preprocess_text(sen))

    y = df.label
    y = y.factorize()[0]
    y = np.array(list(y))
    print(y)

    epo = 6
    bs = 256

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
    print(y_train)
    print(y_test)

    # Neural NetWorks

    max_words=5000

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    print(X_train[3])

    maxlen = 50   # tried 100 , 500 bad results

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

######################## describe the batch size and number of epochs chosen and add extra or remove layers in the convolutional (add)
    # Convolutional NN
    # Without layer mods scores: 0.9135, 0.9195
    # create the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=maxlen)) #converts words into vectors reducing dimensions
        #  text is 1 dimensional, so Conv1D
        #  the kernel in this case is a vector of length 5, not a 2 dimensional matrix
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu')) #creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputsBEST RESULT filters=64 instead of 128: 0.916
        #  the pooling layer in this case is also 1 dimensional
    model.add(MaxPooling1D(pool_size=2)) #selects the most prominent features new pool_size, new maxpooling instead of global pooling
    model.add(Flatten()) #converts data into 1dim vector new
    model.add(Dense(10, activation='relu')) #applies matrix mult to preceding layer new 0.915 0.914 0.9129, tryed with 32 but decreases accuracy
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=bs, epochs=epo, verbose=1, validation_split=0.2)
    score = model.evaluate(X_test, y_test, verbose=1)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print('epochs: ',epo)
    print('batch size: ',bs)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    #Do some prediction
    # instance="Positive happy good clean room"
    # print(instance)
    # instance = tokenizer.texts_to_sequences(instance)

    # flat_list = []
    # for sublist in instance:
    #     for item in sublist:
    #         flat_list.append(item)

    # flat_list = [flat_list]

    # instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    # predicted = model.predict(instance)
    # y_pred = np.argmax(predicted, axis=1)
    # print(y_pred)

    # #Use the model to predict on new data
    # predicted = model.predict(X_test)

    # # Choose the class with higher probability 
    # y_pred = np.argmax(predicted, axis=1)
    # print(y_pred)

    # # Compute and print the confusion matrix
    # print(confusion_matrix(y_test, y_pred))

    # # Create the performance report
    # print(classification_report(y_test, y_pred))

    # Recurrent NN

    # create the model
    # Without layer mods scores: 0.9135, 0.9195

    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=maxlen)) #converts words into vectors reducing dimensions #trainable = True => drops the accuracy 
    model.add(Dense(32, activation='relu', name='Dense1')) #applies matrix mult to preceding layer #new drops accuracy to 0.86 0.62 0.917
    model.add(Dropout(rate = 0.25)) #new #ignore neurons to prevent overfitting
    model.add(LSTM(64, return_sequences=True, dropout=0.15, name='LSTM')) #capable of learning order dependence modified
    model.add(GRU(64, return_sequences=False, dropout=0.15, name='GRU')) #solve vanishing gradient problem new AFTER MODS 0.917
    model.add(Dense(64, name='Dense2')) #new2
    model.add(Dropout(rate = 0.25)) #new2
    model.add(Dense(32, name='Dense3')) #new2 0.91699 0.9185 0.914, NEW RECORD AFTER CHANGING TO 64: 0.921
    model.add(Dense(1, activation='sigmoid', name = 'Output'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())


    history = model.fit(X_train, y_train, batch_size=bs, epochs=epo, verbose=1, validation_split=0.2)
    score = model.evaluate(X_test, y_test, verbose=1)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()


    #Do some prediction
    # instance="Positive happy good clean room"
    # print(instance)
    # instance = tokenizer.texts_to_sequences(instance)

    # flat_list = []
    # for sublist in instance:
    #     for item in sublist:
    #         flat_list.append(item)

    # flat_list = [flat_list]

    # instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    # predicted = model.predict(instance)
    # y_pred = np.argmax(predicted, axis=1)
    # print(y_pred)

    # #Use the model to predict on new data
    # predicted = model.predict(X_test)

    # # Choose the class with higher probability 
    # y_pred = np.argmax(predicted, axis=1)
    # print(y_pred)

    # # Compute and print the confusion matrix
    # print(confusion_matrix(y_test, y_pred))

    # # Create the performance report
    # print(classification_report(y_test, y_pred))
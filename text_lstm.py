#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 13:47:54 2018

@author: hezb
"""
        
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Activation, Dropout, Flatten
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# hyper-parameters
MAX_WORD_NUM = 1000
embedding_dim = 100


train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
encoder = OneHotEncoder()
y = encoder.fit_transform(train['class'].values.reshape(-1,1)).toarray()


print("Tokenization")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['word_seg'].append(test['word_seg']))
vocab = tokenizer.word_index
vocab_size = len(vocab)

print("word to vocab index")
x = tokenizer.texts_to_sequences(train['word_seg'])
x_test = tokenizer.texts_to_sequences(test['word_seg'])

print("padding")
x = pad_sequences(x, maxlen= MAX_WORD_NUM, padding='post')
x_test = pad_sequences(x_test, maxlen=MAX_WORD_NUM, padding='post')

print("spliting training data")
X, X_val, Y, Y_val = train_test_split(x, y, test_size = 0.1, random_state = 1)


input_layer = Input(shape = (MAX_WORD_NUM, ), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_WORD_NUM, trainable = True)(input_layer)
xi = Dropout(0.3)(embedding_layer)
xi = LSTM(units=128, return_sequences= True)(embedding_layer)
xi = Dropout(0.3)(xi)
xi = Flatten()(xi)
xi = Dense(units=19)(xi)

output = Activation('softmax')(xi)

model = Model(inputs = input_layer, outputs = output)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 64
epochs  =20

model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

x1 = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(embedding_layer)
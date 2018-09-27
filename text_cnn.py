#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:59:18 2018

@author: hezb
"""

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Conv1D, Activation, GlobalMaxPool1D, concatenate, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# hyper-parameters
MAX_WORD_NUM = 1000
embedding_dim = 512


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


input_layer = Input(shape = (MAX_WORD_NUM,), dtype = 'int32', name = 'input')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length= MAX_WORD_NUM, trainable = True)(input_layer)

conv_output = []
for filter_ in [1,2,3,4,5,6]:
    conv = Conv1D(256, filter_, padding='same')(embedding_layer)
    conv = Activation('relu')(conv)
    conv = GlobalMaxPool1D()(conv)
    conv_output.append(conv)

conv_output = concatenate(conv_output)
full_connected_layer = Dense(256)(conv_output)
full_connected_layer = Dropout(0.3)(full_connected_layer)
full_connected_layer = Activation('relu')(full_connected_layer)
full_connected_layer = Dense(72)(full_connected_layer)
full_connected_layer = Activation('relu')(full_connected_layer)
full_connected_layer = Dense(19)(full_connected_layer)

output_layer = Activation('softmax')(full_connected_layer)

model = Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer=Adam(), metrics=['accuracy'], loss='binary_crossentropy')

print("training")
model.fit(x=X, y = Y, epochs=2)

train_pred = model.predict(X)
valid_pred = model.predict(X_val)

train_pred = np.argmax(train_pred, axis = 1)
valid_pred = np.argmax(valid_pred, axis = 1)
train_true = np.argmax(Y, axis = 1)
valid_true = np.argmax(Y_val, axis = 1)

train_acc = accuracy_score(train_pred, train_true)
valid_acc = accuracy_score(valid_pred, valid_true)

print(train_acc, valid_acc)
'''
training_score = model.evaluate(X, Y)
valid_score = model.evaluate(X_val, Y_val)
print(training_score, valid_score)

Y_preds = model.predict(x_test)
Y_preds = np.argmax(Y_preds, axis = 1)
test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (Y_preds + 1).astype(int)
#test_pred.to_csv('../sub/sub_textcnn.csv',index=None)
'''
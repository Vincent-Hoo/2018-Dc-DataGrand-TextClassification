#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 09:05:27 2018

@author: hezb
"""

import pandas as pd
import numpy as np
import fasttext
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time

def func(x):
    return x.word_seg + " __label__" + str(x['class'])

def convertProba(preds):
    proba = np.zeros((len(preds), n_classes))
    for i in range(len(preds)):
        pred_class = int(preds[i][0][0])
        pred_proba = preds[i][0][1]
        proba[i][pred_class-1] = pred_proba
        remain_proba = (1-pred_proba) / (n_classes-1)
        for j in range(n_classes):
            if j == pred_class - 1:
                continue
            proba[i][j] = remain_proba
    return proba


train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

x = train['word_seg'].values
x_test = test['word_seg'].values
y = train['class'].values

n_folds = 5
n_classes = 19

kf = KFold(n_splits = n_folds, random_state=2018)


oof_train = np.zeros((train.shape[0], n_classes))
oof_test = np.zeros((test.shape[0], n_classes))
oof_test_skf = np.zeros((n_folds, test.shape[0], n_classes))

t1 = time.time()


for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(i)
    kf_x_train = x[train_index]
    kf_y_train = y[train_index]
    kf_x_test = x[test_index]
    
    for j in range(kf_x_train.shape[0]):
        kf_x_train[j] += ' __label__' + str(kf_y_train[j])
        
    fasttext_train = pd.DataFrame(kf_x_train, columns = ['word_seg'])
    fasttext_train.to_csv('tuning/fasttext_trainset.txt', index = None, header = None)
    
    clf = fasttext.supervised(input_file = 'tuning/fasttext_trainset.txt', output = 'tuning/fasttext_model', label_prefix = '__label__', 
                         bucket = 2000000, word_ngrams = 1, ws = 20, epoch=40, silent = 0)
    
    valid_preds = clf.predict_proba(kf_x_test.tolist())
    test_preds = clf.predict_proba(x_test.tolist())
    valid_preds_proba = convertProba(valid_preds)
    test_preds_proba = convertProba(test_preds)
    
    oof_train[test_index, 0 : n_classes] = valid_preds_proba
    oof_test_skf[i, :, :] = test_preds_proba
    


t2=time.time()
print("time use:",(t2-t1) / 60)



train_preds = np.argmax(oof_train, axis = 1)
train_preds += 1
acc = accuracy_score(train_preds, y)
print(acc)

oof_test[:, 0 : n_classes] = np.mean(oof_test_skf, axis = 0)


oof_train_df = pd.DataFrame(oof_train)
oof_test_df = pd.DataFrame(oof_test)
oof_train_df.to_csv('../data/stacking/train/ft.csv', index = None, header = None)
oof_test_df.to_csv('../data/stacking/test/ft.csv', index = None, header = None)

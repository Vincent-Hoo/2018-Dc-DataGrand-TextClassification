#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 22:48:22 2018

@author: hezb
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.cross_validation import train_test_split

def fun(zi_list_str):
    zi_list = zi_list_str.split(' ')
    new_zi_list = []
    for zi in zi_list:
        new_zi_list.append('z' + zi)
    return ' '.join(new_zi_list)



t1=time.time()
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

train['article'] = train['article'].apply(fun)
test['article'] = test['article'].apply(fun)

train['merged'] = train['article'] + train['word_seg']
test['merged'] = test['article'] + test['word_seg']


column="merged"
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,3),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
x = vec.fit_transform(train[column])
x_test = vec.transform(test[column])
y = (train["class"] - 1).astype(int)


clf = RandomForestClassifier(n_estimators=50)
#clf = LogisticRegression(C=4, dual=True, n_jobs = 4)
#clf = LinearSVC()
#lin_clf.fit(x,y)

X, X_val, Y, Y_val = train_test_split(x, y, test_size = 0.1, random_state = 1, stratify = y)

clf.fit(X, Y)
Y_preds = clf.predict(X_val)
acc = accuracy_score(Y_preds, Y_val)
f1_macro = f1_score(Y_val, Y_preds, average = 'macro')
f1_micro = f1_score(Y_val, Y_preds, average = 'micro')
print(acc, f1_macro, f1_micro)


clf.fit(x, y)
Y_preds = clf.predict(x_test)
test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (Y_preds + 1).astype(int)
#test_pred.to_csv('../sub/sub_tfidf_svm_zi_merged_baseline.csv',index=None)
t2=time.time()
print("time use:",(t2-t1) / 60)


'''
xgb_param = {'max_depth':7,
             'subsample': 0.6,
             'colsample_bytree':0.6,
             'colsample_bylevel':0.6,
             'eta':0.02,
             'alpha': 1.5,
             'lambda': 0.8,
             'objective':'multi:softmax',
             'eval_metric': 'logloss',
             'num_class': 19,
	     'silent': 0
             }


dtrain = xgb.DMatrix(X, label = Y)
dval = xgb.DMatrix(X_val, label = Y_val)
bst = xgb.train(xgb_param, dtrain, 200)
train_pre = bst.predict(dtrain)
val_pre = bst.predict(dval)
train_accuracy = accuracy_score(Y, train_pre)
val_accuracy = accuracy_score(Y_val, val_pre)
print(f1_score(y, train_pre, average = 'micro'))


dtest = xgb.DMatrix(test_term_doc)
preds = bst.predict(dtest)
preds = preds.astype(int)


X, X_val, Y, Y_val = train_test_split(x, y, test_size = 0.1, random_state = 1, stratify = y)


lgb_train = lgb.Dataset(X, Y)
lgb_eval = lgb.Dataset(X_val, Y_val, reference = lgb_train)

lgb_params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': 19,
              'metric': 'multi_error',
              'num_leaves': 200,
              'min_data_in_leaf': 100,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'bagging_fraction': 0.8,
              'bagging_freq': 5,
              'lambda_l1': 0.8,
              'lambda_l2': 0.5*2,
              'min_gain_to_split': 0.2,
              'verbose': 5,
              'is_unbalance': True
              }
print('start')
t1 = time.time()

lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=10)

t2=time.time()
print("time use:",(t2-t1) / 60)


Y_preds = lgb_model.predict(X_val)
Y_preds = np.argmax(Y_preds, axis = 1)
acc = accuracy_score(Y_preds, Y_val)
acc



Y_preds = lgb_model.predict(x_test)

Y_preds = np.argmax(Y_preds, axis = 1)
test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (Y_preds + 1).astype(int)
test_pred.to_csv('../sub/sub_lgb_baseline.csv',index=None)
'''
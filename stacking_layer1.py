#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 18:19:45 2018

@author: hezb
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import pandas as pd, numpy as np
import time
import pickle

def fun(zi_list_str):
    zi_list = zi_list_str.split(' ')
    new_zi_list = []
    for zi in zi_list:
        new_zi_list.append('z' + zi)
    return ' '.join(new_zi_list)

train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

train['article'] = train['article'].apply(fun)
test['article'] = test['article'].apply(fun)

train['merged'] = train['article'] + train['word_seg']
test['merged'] = test['article'] + test['word_seg']

'''
column="merged"
vec = TfidfVectorizer(ngram_range=(1,3),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
x = vec.fit_transform(train[column])
x_test = vec.transform(test[column])
y = train['class'].values
'''
#pickle.dump(x, open('../data/feat/train_tfidf', 'wb'), protocol = 4)
#pickle.dump(x_test, open('../data/feat/test_tfidf', 'wb'), protocol = 4)

x = pickle.load(open('../data/feat/train_tfidf', 'rb'), encoding = 'latin4')
x_test = pickle.load(open('../data/feat/test_tfidf', 'rb'), encoding = 'latin4')
print('data has been prepared')

n_folds = 5
n_classes = 19

kf = KFold(n_splits = n_folds, random_state=2018)
clfs = [SGDClassifier(loss='hinge', n_jobs = 8, max_iter=100, tol=0.03),
        LogisticRegression(C = 4, dual = False, n_jobs = 8, penalty='l2', multi_class='ovr', solver='sag'), 
          LogisticRegression(C = 4, dual = False, n_jobs = 8, penalty='l2', multi_class='ovr', solver='lbfgs'),
          LogisticRegression(C = 4, dual = False, n_jobs = 8, penalty='l2', multi_class='multinomial', solver='sag'),
          LinearSVC(dual=False, C = 4),
          PassiveAggressiveClassifier(random_state=0),
          Perceptron(random_state=0, n_jobs = 8, max_iter=1000, tol=0.03)
          ]

name = ['sgd', 'lr_ovr_sag', 'lr_ovr_lbfgs', 'lr_mul_sag', 'linsvc', 'passiveAgg', 'perceptron']

for ind in range(6, len(clfs)):
    clf = clfs[ind]
    oof_train = np.zeros((train.shape[0], n_classes))
    oof_test = np.zeros((test.shape[0], n_classes))
    oof_test_skf = np.zeros((n_folds, test.shape[0], n_classes))
    
    t1 = time.time()
    print(name[ind])
    
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        print(i)
        kf_x_train = x[train_index]
        kf_y_train = y[train_index]
        kf_x_test = x[test_index]
        
        clf.fit(kf_x_train, kf_y_train)
        
        #oof_train[test_index, 0 : n_classes] = clf.predict_proba(kf_x_test)
        #oof_test_skf[i, :, :] = clf.predict_proba(x_test)
        if ind != 0 and ind != 5 and ind != 6:
            oof_train[test_index, 0 : n_classes] = clf.predict_proba(kf_x_test)
            oof_test_skf[i, :, :] = clf.predict_proba(x_test)
        else:
            oof_train[test_index, 0 : n_classes] = clf.decision_function(kf_x_test)
            oof_test_skf[i, :, :] = clf.decision_function(x_test)
    
    t2=time.time()
    print("time use:",(t2-t1) / 60)
    
    train_preds = np.argmax(oof_train, axis = 1)
    train_preds += 1
    acc = accuracy_score(train_preds, y)
    print(acc)
    
    oof_test[:, 0 : n_classes] = np.mean(oof_test_skf, axis = 0)

    
    oof_train_df = pd.DataFrame(oof_train)
    oof_test_df = pd.DataFrame(oof_test)
    oof_train_df.to_csv('../data/stacking/train/%s.csv'%name[ind], index = None, header = None)
    oof_test_df.to_csv('../data/stacking/test/%s.csv'%name[ind], index = None, header = None)

'''
#oof_train = oof_train.astype(int)


from sklearn.cross_validation import train_test_split
import lightgbm as lgb

X, X_val, Y, Y_val = train_test_split(oof_train, y, test_size = 0.1, random_state = 1, stratify = y)
lgb_train = lgb.Dataset(X, Y)
lgb_eval = lgb.Dataset(X_val, Y_val, reference = lgb_train)

lgb_params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': 19,
              'metric': 'multi_error',
              'num_leaves': 200,
              'min_data_in_leaf': 100,
              'learning_rate': 0.05,
              'feature_fraction': 0.8,
              'bagging_fraction': 0.8,
              'bagging_freq': 5,
              'lambda_l1': 0.8,
              'lambda_l2': 0.5*2,
              'min_gain_to_split': 0.2,
              'verbose': 5,
              'is_unbalance': True
              }

t1 = time.time()

lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=10)

t2=time.time()
print("time use:",(t2-t1) / 60)


Y_preds = lgb_model.predict(X_val)
Y_preds = np.argmax(Y_preds, axis = 1)
acc = accuracy_score(Y_preds, Y_val)
acc


Y_preds = lgb_model.predict(oof_test)

Y_preds = np.argmax(Y_preds, axis = 1)
test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (Y_preds + 1).astype(int)
#test_pred.to_csv('../sub/sub_stacking.csv',index=None)



svm_clf = RandomForestClassifier()
svm_clf.fit(oof_train, y)

train_pre = svm_clf.predict(oof_train)
print('train acc: ' + str(accuracy_score(train_pre, y))) 
np.unique(train_pre)

test_pre = svm_clf.predict(oof_test)
np.unique(test_pre)

test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (test_pre).astype(int)
#test_pred.to_csv('../sub/sub_stacking.csv',index=None)
'''
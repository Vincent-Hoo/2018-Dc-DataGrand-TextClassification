#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:54:05 2018

@author: inplus-dm
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import time
import pickle

train_lr_mul_sag = pd.read_csv('../data/stacking/train/lr_mul_sag.csv', header = None).values
train_lr_ovr_sag = pd.read_csv('../data/stacking/train/lr_ovr_sag.csv', header = None).values
train_lr_ovr_lbfgs = pd.read_csv('../data/stacking/train/lr_ovr_lbfgs.csv', header = None).values
train_sgd = pd.read_csv('../data/stacking/train/sgd.csv', header = None).values
train_linsvc = pd.read_csv('../data/stacking/train/linsvc.csv', header = None).values
train_ft = pd.read_csv('../data/stacking/train/ft.csv', header = None).values
train_passiveAgg = pd.read_csv('../data/stacking/train/passiveAgg.csv', header = None).values
train_perceptron = pd.read_csv('../data/stacking/train/perceptron.csv', header = None).values
train_lstm = pd.read_csv('../data/stacking/train/perceptron.csv', header = None).values

test_lr_mul_sag = pd.read_csv('../data/stacking/test/lr_mul_sag.csv', header = None).values
test_lr_ovr_sag = pd.read_csv('../data/stacking/test/lr_ovr_sag.csv', header = None).values
test_lr_ovr_lbfgs = pd.read_csv('../data/stacking/test/lr_ovr_lbfgs.csv', header = None).values
test_sgd = pd.read_csv('../data/stacking/test/sgd.csv', header = None).values
test_linsvc = pd.read_csv('../data/stacking/test/linsvc.csv', header = None).values
test_ft = pd.read_csv('../data/stacking/test/ft.csv', header = None).values
test_passiveAgg = pd.read_csv('../data/stacking/test/passiveAgg.csv', header = None).values
test_perceptron = pd.read_csv('../data/stacking/test/perceptron.csv', header = None).values
test_lstm = pd.read_csv('../data/stacking/test/perceptron.csv', header = None).values

x = np.column_stack((train_linsvc, train_lr_mul_sag, train_lr_ovr_lbfgs, train_lr_ovr_sag, train_sgd, train_passiveAgg, train_perceptron, train_lstm))
x_test = np.column_stack((test_linsvc, test_lr_mul_sag, test_lr_ovr_lbfgs, test_lr_ovr_sag, test_sgd, test_passiveAgg, test_perceptron, test_lstm))
#y = pd.read_csv('../data/train_set.csv')['class'].values - 1
test = pd.read_csv('../data/test_set.csv')
y = pickle.load(open('../data/train_label', 'rb'), encoding='latin1')

X, X_val, Y, Y_val = train_test_split(x, y, test_size = 0.1, random_state = 1, stratify = y)
lgb_train = lgb.Dataset(X, Y)
lgb_eval = lgb.Dataset(X_val, Y_val, reference = lgb_train)


lgb_params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': 19,
              'metric': 'multi_error',
              'num_leaves': 50,
              'min_data_in_leaf': 100,
              'learning_rate': 0.05,
              'feature_fraction': 0.8,
              'bagging_fraction': 0.8,
              'bagging_freq': 1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.8,
              'min_gain_to_split': 0.2,
              'verbose_eval': False,
              'is_unbalance': True
              }

t1 = time.time()

lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=80)

t2=time.time()
print("time use:",(t2-t1) / 60)


train_preds = lgb_model.predict(X)
train_preds = np.argmax(train_preds, axis = 1)
train_acc = accuracy_score(train_preds, Y)
print('training accuracy is %s'%str(train_acc))

val_preds = lgb_model.predict(X_val)
val_preds = np.argmax(val_preds, axis = 1)
val_acc = accuracy_score(val_preds, Y_val)
print('validation accuracy is %s'%str(val_acc))






Y_preds = lgb_model.predict(x_test)
Y_preds = np.argmax(Y_preds, axis = 1)
test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (Y_preds + 1).astype(int)
test_pred.to_csv('../sub/sub_stacking_five_models_v2.csv',index=None)

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

data = train.append(test)
data = data.reset_index()
del data['index']
'''
data['word_cnt'] = data['word_seg'].apply(lambda x: len(x.split(' ')))
data['zi_cnt'] = data['article'].apply(lambda x: len(x.split(' ')))

sentences = []
for index, row in data.iterrows():
    sentences.append(row['word_seg'])
    
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

countvectorizer = CountVectorizer()
mat = countvectorizer.fit_transform(sentences)

sentences = data['word_seg'].values
for i in range(sentences.shape[0]):
    sentences[i] = sentences[i].split(" ")
'''
from gensim.models import Word2Vec, KeyedVectors    
model = KeyedVectors.load_word2vec_format('')  
#model = Word2Vec(sentences, size=100, window=10, min_count=0, sg=1, hs=1)
#model.wv.save_word2vec_format('../data/wordvec_dim100')

def doc2vec(sentence):
    words = sentence.split(" ")
    docvec = np.zeros(100)
    for word in words:
        docvec += model.wv[word]
    docvec /= len(words)
    return docvec

X = np.zeros((train.shape[0], 100))
for i in range(X.shape[0]):
    X[i] = doc2vec(train.loc[i, 'word_seg'])

Y = train['class'].values

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score




classifiers = {'LR': LogisticRegression(),
               'SVM': SVC(),
               'NB': GaussianProcessClassifier(),
               'GBDT': GradientBoostingClassifier()
               }


for clf_name in classifiers:
    print(clf_name)
    classifier = classifiers[clf_name]
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_val)
    score = f1_score(Y_val, Y_pred, average = 'micro')
    print(clf_name, score)
    
lr_clf = LogisticRegression(C = 4, dual = True)

lr_clf.fit(X, Y)
#Y_pred = lr_clf.predict(X_val)
#score = f1_score(Y_val, Y_pred, average = 'micro')


X_test = np.zeros((test.shape[0], 100))

for i in range(X_test.shape[0]):
    X_test[i] = doc2vec(test.loc[i, 'word_seg'])
    
Y_pred = lr_clf.predict(X_test)

res_df = pd.DataFrame(columns = ['id', 'class'])
res_df.id = test.id.values
res_df['class'] = Y_pred

res_df.to_csv('../sub/sub_wordvec_lr.csv', index = None)
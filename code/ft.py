import fasttext
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
'''
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

def func(x):
    return x.word_seg + " __label__" + str(x['class'])


train['word_seg'] = train.apply(func, axis = 1)

Y = train['class'].values
#fasttext_train = train.loc[:, ['word_seg']]

X_train, X_val, Y_train, Y_val = train_test_split(train, Y, test_size = 0.2)
fasttext_train = X_train.loc[:, ['word_seg']]
fasttext_val = X_val.loc[:, ['word_seg']]

fasttext_train.to_csv('tuning/fasttext_trainset.txt', index = None, header = None)
'''











clf = fasttext.supervised(input_file = 'tuning/fasttext_trainset.txt', output = 'tuning/fasttext_model', label_prefix = '__label__', 
                         bucket = 2000000, word_ngrams = 1, ws = 20, epoch=40, silent = 0)


train_data = X_train.word_seg.values.tolist()
val_data = X_val.word_seg.values.tolist()

res = clf.predict(train_data)
y_pred = []
for i in res:
    y_pred.append(int(i[0]))
train_acc = accuracy_score(Y_train, y_pred)

res = clf.predict(val_data)
y_pred = []
for i in res:
    y_pred.append(int(i[0]))
val_acc = accuracy_score(Y_val, y_pred)

print(train_acc, val_acc)





clf = fasttext.supervised(input_file = 'fasttext_trainset.txt', output = 'fasttext_model', label_prefix = '__label__', 
                         bucket = 2000000, word_ngrams = 2, ws = 20, epoch=40, silent = 0)


#test_data = test.word_seg.values.tolist()
test = pd.read_csv('../data/test_set.csv')
test_data = test.word_seg.values.tolist()

res = clf.predict(test_data)
y_pred = []
for i in res:
    y_pred.append(int(i[0]))

test_pred = pd.DataFrame(columns = ['id', 'class'])
test_pred.id = test.id.values
test_pred['class'] = (y_pred)
test_pred.to_csv('../sub/sub_fasttext_baseline.csv',index=None)
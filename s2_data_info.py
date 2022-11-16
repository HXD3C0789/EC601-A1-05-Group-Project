import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
from sklearn import metrics
from scipy import stats
import pandas as pd
from sklearn import preprocessing
import math
import lightgbm as lgbm
from matplotlib import pyplot
import random
import torch
import warnings

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
warnings.filterwarnings("ignore")


data = pd.read_csv('Amex date pickled/data_processed_1m.csv')
# print(label['target'])
data = data.sample(frac=1)

rr = []
pp = []

#random.shuffle(data)
kk = [i for i in range(0,len(data))]
#print(data['target'])
label = np.array(data['target'])
data_test = data[600:]
label_test = label[600:]
print(data_test)
data_train = data[0:200]
train_label = label[0:200]
print(data_train)

for i in range(0, 700):
    #print(i)
    test_d = np.array(data_train[str(i)])
    test_d = preprocessing.normalize([test_d])[0]
    r, p = stats.pointbiserialr(test_d, train_label)
    if p > 5:
        test_d1 = np.power(test_d, 2)
        test_d1 = preprocessing.normalize([test_d1])[0]
        r, p = stats.pointbiserialr(test_d1, train_label)
        temp = np.power(data_test[str(i)],2)
        temp[~np.isfinite(temp)] = 0
        temp = preprocessing.normalize([temp])[0]

        data_test[str(i)] = temp
    if p > 5:
        test_d1 = np.power(test_d, 0.5)
        test_d1[~np.isfinite(test_d1)] = 0
        test_d1 = preprocessing.normalize([test_d1])[0]
        r, p = stats.pointbiserialr(test_d1, train_label)
        temp = np.power(data_test[str(i)],0.5)
        temp[~np.isfinite(temp)] = 0
        temp = preprocessing.normalize([temp])[0]
        data_test[str(i)] = temp
        if p > 5:
            test_d1 = np.log(test_d)
            test_d1[~np.isfinite(test_d1)] = 0
            test_d1 = preprocessing.normalize([test_d1])[0]
            temp = np.log(data_test[str(i)])
            temp[~np.isfinite(temp)] = 0
            temp = preprocessing.normalize([temp])[0]

            data_test[str(i)] = temp
            r, p = stats.pointbiserialr(test_d1, train_label)
    rr.append(r)
    pp.append(p)

n = [i for i in range(0, 700)]
#print(len(n))
plt.scatter(n, pp)
plt.ylabel('p-value')
plt.show()
plt.scatter(n, rr)
plt.ylabel('correlation')
plt.show()
'''
log_m = sklearn.linear_model.LogisticRegression()
log_m.fit(data_train,train_label)
pr_l = log_m.predict(data_test)
print(sklearn.metrics.accuracy_score(pr_l,label_test))
print(metrics.confusion_matrix(label_test,pr_l))
'''

'''
dtrain=xgb.DMatrix(data,label=train_label)
dtest=xgb.DMatrix(data_test)
params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]
bst=xgb.train(params,dtrain,num_boost_round=5,evals=watchlist)
ypred=bst.predict(dtest)
y_pred = (ypred >= 0.5)*1

print ('AUC: %.4f' % metrics.roc_auc_score(label_test,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(label_test,y_pred))
print ('Recall: %.4f' % metrics.recall_score(label_test,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(label_test,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(label_test,y_pred))
print(metrics.confusion_matrix(label_test,y_pred))
'''
import re
data_train = data_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
lg = lgbm.LGBMClassifier(n_jobs=16)
lg.fit(data_train,train_label)
pr_l = lg.predict(data_test)
print(metrics.confusion_matrix(label_test,pr_l))
print(sklearn.metrics.accuracy_score(pr_l,label_test))

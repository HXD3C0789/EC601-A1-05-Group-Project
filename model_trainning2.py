import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path="/Users/jianxiaoyang/Documents/EC601 software/american express/data/"
data=pd.read_csv(r'%s/data_processed.csv'%path,index_col=0)
data=np.array(data)
X=data[:,:-1]
y=data[:,-1]

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,stratify=y)

from sklearn.model_selection import GridSearchCV

'''
logistic：
{'C': 10}
best score:0.979419
'''
# from sklearn.linear_model import LogisticRegression
# print('logistic test:')
# logistic=LogisticRegression(max_iter=100000,penalty='l2')
'''
可以直接用模型跑，设定C=1是不会错的，大不了有点过拟合
'''
'''
model=logistic.fit(Xtrain,ytrain)
train_score=logistic.score(Xtrain,ytrain)
test_score=logistic.score(Xtest,ytest)
print("train score:%.6f"%train_score)
print("test score:%.6f"%test_score)   #0.984
'''

'''
如果用GridSearch，前面就没有必要分Xtrain，Xtest了，因为这里的cv就在分片
输出每一次跑的参数：
for p, s in zip(model.cv_results_['params'],model.cv_results_['mean_test_score']):
    print(p,s)
'''
# grid={'C':[0.1,1,5,10]}
# clf=GridSearchCV(estimator=logistic,param_grid=grid,cv=5)
# clf.fit(Xtrain,ytrain)
# print(clf.best_params_)
# print('best score:%.6f'%clf.best_score_)

'''
svm
目前跑出来是：
{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
best score:0.979899
'''
# from sklearn import svm
# print("\nSVM test:")
# SVM=svm.SVC()
# grid=[
#     {'kernel':['linear'],'C':[0.1,1,5,10]},\
#     {'kernel':['rbf'],'C':[0.1,1,5,10,100],'gamma':[0.01,0.1,1]},\
#     {'kernel':['poly'],'C':[0.1,1,5,10,100],'gamma':[0.01,0.1,1]}
# ]
# clf=GridSearchCV(estimator=SVM,param_grid=grid,cv=3)
# clf.fit(Xtrain,ytrain)
# print(clf.best_params_)
# print('best score:%.6f'%clf.best_score_)

'''
决策森林：
{'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_split': 3}
best score:0.957885
'''
# from sklearn.tree import DecisionTreeClassifier
# print('decision tree test:')
# decisionTree=DecisionTreeClassifier()
# grid={'max_features':['sqrt','log2'],'min_samples_split':[2,3,4,5],'criterion':['entropy','gini']}
# clf=GridSearchCV(estimator=decisionTree,param_grid=grid)
# clf.fit(Xtrain,ytrain)
# print(clf.best_params_)
# print('best score:%.6f'%clf.best_score_)

'''
简单用MLP神经网络跑一下，这个是要重点调参的，参数很多(目前先放了2层，50*20)，先测试一下直接用Xtrain，Xtest,
score:0.9813352476669059
'''
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(2000,2000,540,50),activation='logistic',solver='sgd',alpha=0.0001,\
                 learning_rate='adaptive',learning_rate_init=0.1,power_t=0.5,tol=0.0001,max_iter=100)
mlp.fit(Xtrain,ytrain)
ypred=mlp.predict(Xtest)
res=np.array(ytest[ytest==ypred])
print('score:%.3f'%(len(res)/len(ytest)))
print ('AUC: %.4f' % metrics.roc_auc_score(ytest,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(ytest,ypred))
print ('Recall: %.4f' % metrics.recall_score(ytest,ypred))
print ('F1-score: %.4f' %metrics.f1_score(ytest,ypred))
print ('Precesion: %.4f' %metrics.precision_score(ytest,ypred))
print(metrics.confusion_matrix(ytest,ypred))


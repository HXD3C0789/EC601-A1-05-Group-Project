import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path="/Users/jianxiaoyang/Documents/EC601 software/american express/data/"
data=pd.read_csv(r'%s/data_processed.csv'%path,index_col=0)
data=np.array(data)
X=data[:,:-1]
y=data[:,-1]

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,stratify=y)

from sklearn.linear_model import LogisticRegression
print('logistic test:')
logistic=LogisticRegression(max_iter=10000,penalty='l2')
model=logistic.fit(Xtrain,ytrain)
train_score=logistic.score(Xtrain,ytrain)
test_score=logistic.score(Xtest,ytest)
print("train score:%.6f"%train_score)
print("test score:%.6f"%test_score)

# from sklearn.neural_network import MLPClassifier
# mlp=MLPClassifier(hidden_layer_sizes=(700,100,30),activation='logistic',solver='sgd',alpha=0.0001,\
#                  learning_rate='adaptive',learning_rate_init=0.1,power_t=0.5,tol=0.0001,max_iter=100000)
# mlp.fit(Xtrain,ytrain)
# ypred=mlp.predict(Xtest)
# res=np.array(ytest[ytest==ypred])
# print('score:%.3f'%(len(res)/len(ytest)))
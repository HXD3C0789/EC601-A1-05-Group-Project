import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path="/Users/jianxiaoyang/Documents/EC601 software/american express/data"


Xtrain=pd.read_csv(r'%s/train_data_S.csv'%path,header=0)
def StringToInt(df):
    dict_63={'CL':0,"CO":1,"CR":2,"XL":3,"XM":4,"XZ":5}
    dict_64={np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}
    df['D_63']=df['D_63'].apply(lambda x:dict_63[x]).astype(dtype='int')
    df['D_64'] = df['D_64'].apply(lambda x: dict_64[x]).astype(dtype='int')

def denoise(df):
    for col in df.columns:
        if(col not in ['customer_ID', 'S_2', 'D_63', 'D_64']):
            df[col]=np.floor(df[col]*100)

StringToInt(Xtrain)
denoise(Xtrain)

# Xtrain.to_csv(r'/Users/jianxiaoyang/Documents/EC601 software/american express/data/train_data_S_c1.csv')

#缺失值直接置0，缺失值大于75%整列去掉
threshold=Xtrain.shape[0]*0.75
Xtrain.dropna(thresh=int(threshold),axis=1,inplace=True)
Xtrain.fillna(0,inplace=True)
# Xtrain.to_csv(r'/Users/jianxiaoyang/Documents/EC601 software/american express/data/train_data_S_c2.csv')
#日期没什么用，扔了,第一列本身是索引，扔了
Xtrain.drop(['S_2'],axis=1,inplace=True)
Xtrain.drop(Xtrain.columns[0],axis=1,inplace=True)  #10000*191

'''
need groupby, ytrain is set by id, but Xtrain is not, Xtrain has duplicate id
'''
Xtrain=Xtrain.groupby("customer_ID").agg(['mean', 'std', 'min', 'max', 'sum'])
Xtrain.columns=['_'.join(x) for x in Xtrain.columns] #826*780
Xtrain.fillna(0,inplace=True) #合并后会出现缺失值，std中
# Xtrain.to_csv(r'/Users/jianxiaoyang/Documents/EC601 software/american express/data/train_data_S_c3.csv')

ytrain=pd.read_csv(r'%s/train_labels_S.csv'%path,header=0,index_col=0)
data=pd.merge(Xtrain,ytrain,how='left',on=['customer_ID'])
# data.to_csv(r'/Users/jianxiaoyang/Documents/EC601 software/american express/data/data_S_c4.csv')

# #对数据进行PCA分解
Xtrain_data=data.iloc[:,3:-1] #排掉unnamed和id,date,ytrain
from PCA import PCA
pca=PCA(Xtrain_data)
compare=pca.SVDdecompose()

# print(compare)
# import matplotlib.pyplot as plt
# n=range(1,len(compare))
# plt.plot(n,compare[:-1])
# plt.show()
'''
by drawing the eigenvalue-feature figure, find out that 700 has the value of 1e15, so it is the dividing point
'''
Xtrain_pca,t=pca.PCAcompose(700)
# Xtrain_pca=pd.DataFrame(Xtrain_pca)
# Xtrain_pca.to_csv(r'/Users/jianxiaoyang/Documents/EC601 software/american express/data/train_data_S_c3.csv')
ytrain=data.iloc[:,-1]

'''
current datasets:826*700
'''
data=pd.DataFrame(Xtrain_pca)
data['target']=ytrain
data.to_csv(r'%s/data_processed.csv'%path)



import pandas as pd
path="/Users/jianxiaoyang/Documents/EC601 software/american express/data"
Xdata=pd.read_csv(r'%s/train_data.csv'%path,nrows=10000)
ydata=pd.read_csv(r'%s/train_labels.csv'%path,nrows=10000)
Xdata.to_csv(r'%s/train_data_S.csv'%path)
ydata.to_csv(r'%s/train_labels_S.csv'%path)
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score#从Sklearn指标中 引入准确率
from sklearn import metrics
print('begin')
df=pd.read_csv('AXB_383_gene_default.csv')
columns=df.columns
label=df['label']
        #df=df.drop(['id'],axis=1)
#df=df.drop(['0_mirna'],axis=1)
#df=df.drop(['1_gene'],axis=1)
#df=df.drop(['label'],axis=1)
df=df.sample(frac=1).reset_index(drop=True)#首先要打乱df
#print(df)
X=df.values
Y=label.values
print('read_end')
from sklearn.model_selection import KFold
import numpy as np
KF=KFold(n_splits=5)  #建立5折交叉验证方法  查一下KFold函数的参数
count=0
for train_index,test_index in KF.split(X):
    print("TRAIN:",train_index,"TEST:",test_index)
    X_train,X_test=X[train_index],X[test_index]
    #Y_train,Y_test=Y[train_index],Y[test_index]
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    #Y_train=pd.DataFrame(Y_train)
    #Y_test=pd.DataFrame(Y_test)
    count+=1
    X_train.columns=columns
    X_test.columns=columns
    X_train.to_csv('X_train{}.csv'.format(count),index=None)
    X_test.to_csv('X_test{}.csv'.format(count),index=None)

    
    #print(X_train,X_test)
    #print(Y_train,Y_test)

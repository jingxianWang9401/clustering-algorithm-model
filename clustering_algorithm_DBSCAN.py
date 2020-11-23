# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:13:14 2020

@author: wangjingxian
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd



data=pd.read_csv('E:\data_mining\eye_classification\data\eeg_train.csv')
train_data=data.iloc[:,0:14] 
trainingLabels=data.iloc[:,[14]] 

'''
data=pd.read_csv('E:/data_mining/clustering_algorithm_model/data/iris.csv')
train_data=data.ix[:,1:5]
'''

#iris=load_iris()

dbscan=DBSCAN()

#dbscan_model=dbscan.fit(iris.data)

dbscan.fit(train_data)

label = dbscan.labels_

print(label)


#print ("聚类中心\n", (dbscan.cluster_centers_))

quantity = pd.Series(dbscan.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(dbscan.labels_)

res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(data.iloc[res0.index]))
data0=data.iloc[res0.index]
print(data0)



res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(data.iloc[res1.index]))

res2 = resSeries[resSeries.values == -1]
print("聚类后类别为-1的数据\n",(data.iloc[res2.index]))



#all_predictions=dbscan_model.predict(iris.data)
'''
pca=PCA(n_components=2).fit(iris.data)
pca_2d=pca.transform(iris.data)
'''

pca=PCA(n_components=2).fit(train_data)
pca_2d=pca.transform(train_data)




for i in range(0,pca_2d.shape[0]):
    if dbscan.labels_[i] ==0:
        c1=plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r',marker='+')
    elif dbscan.labels_[i] ==1:
        c2=plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif dbscan.labels_[i] ==-1:
        c3=plt.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')

plt.legend([c1,c2,c3],['Cluster1','Cluster2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()

    
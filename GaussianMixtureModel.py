# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:58:45 2020

@author: wangjingxian
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

'''
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3], 
                  random_state = 0)
'''

data=pd.read_csv('E:/data_mining/v_a_guiyihua.csv')
X=data.ix[:,1:3]


##设置gmm函数
gmm = GaussianMixture(n_components=3, covariance_type='spherical').fit(X)
##训练数据
y_pred = gmm.predict(X)

print(y_pred)

predicted_label=gmm.predict([[0.320347155,0.478602869]])
print('预测标签为：',predicted_label)

#print ("聚类中心\n", (gmm.cluster_centers_))

##每个数据的分类
#lables = gmm.labels_
#print('标签预测为：',lables)

##总共的标签分类
labels_unique = np.unique(y_pred)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters聚类数量为 : %d" % n_clusters_)

#print ("聚类中心\n", (spectral_clustering.cluster_centers_))
quantity = pd.Series(y_pred).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类的数据
resSeries = pd.Series(y_pred)
res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(data.iloc[res0.index]))

res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(data.iloc[res1.index]))

res2 = resSeries[resSeries.values == 2]
print("聚类后类别为2的数据\n",(data.iloc[res2.index]))


'''
##绘图
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
'''
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:42:07 2020

@author: wangjingxian
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import spectral_clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle  ##python自带的迭代器模块
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

'''
##产生随机数据的中心
centers = [[1, 1], [-1, -1], [1, -1]]
##产生的数据个数
n_samples=3000
##生产数据
X, lables_true = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.6, 
                  random_state =0)
'''

#data=pd.read_csv('E:\data_mining\loudian_problem\data\dataset3.csv')
#X=data.ix[:,7]


data=pd.read_csv('E:\data_mining\eye_classification\data\eeg_train.csv')
X=data.iloc[:,0:14] 
trainingLabels=data.iloc[:,[14]] 



#scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
#X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则

##变换成矩阵，输入必须是对称矩阵
metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(X)).astype(np.int32)
metrics_metrix += -1 * metrics_metrix.min()
##设置谱聚类函数
n_clusters_= 2
lables = spectral_clustering(metrics_metrix,n_clusters=n_clusters_)

print('数据聚类标签为：',lables)

'''
predicted_label=spectral_clustering.predict([[0.320347155,0.478602869]])
print('预测标签为：',predicted_label)
'''


labels_unique = np.unique(lables)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters聚类数量为 : %d" % n_clusters_)

#print ("聚类中心\n", (spectral_clustering.cluster_centers_))
quantity = pd.Series(lables).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))
#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(lables)
res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(data.iloc[res0.index]))


res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(data.iloc[res1.index]))



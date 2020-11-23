# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:39:27 2020

@author: wangjingxian
"""

# -*- coding:utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data=pd.read_csv('E:\data_mining\loudian_problem\data\dataset3.csv')

X=data.ix[:,7]


scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则

##带宽，也就是以某个点为核心时的搜索半径
bandwidth = estimate_bandwidth(X_dataScale, quantile=0.2, n_samples=500)
##设置均值偏移函数
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
##训练数据
ms.fit(X_dataScale)
##每个点的标签
labels = ms.labels_
print(labels)

'''
predicted_label=ms.predict([[0.320347155,0.478602869]])
print('预测标签为：',predicted_label)
'''

##簇中心的点的集合
cluster_centers = ms.cluster_centers_
print('cluster_centers:',cluster_centers)
##总共的标签分类
labels_unique = np.unique(labels)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

print ("聚类中心\n", (ms.cluster_centers_))
quantity = pd.Series(ms.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))
#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(ms.labels_)
res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(data.iloc[res0.index]))
data0=data.iloc[res0.index]
data0_dianliu=data0.ix[:,7]
max0_dianliu=max(data0_dianliu)
min0_dianliu=min(data0_dianliu)
print('类别0的最大最小值为：\n',min0_dianliu,max0_dianliu)

res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(data.iloc[res1.index]))
data1=data.iloc[res1.index]
data1_dianliu=data1.ix[:,7]
max1_dianliu=max(data1_dianliu)
min1_dianliu=min(data1_dianliu)
print('类别1的最大最小值为：\n',min1_dianliu,max1_dianliu)

res2 = resSeries[resSeries.values == 2]
print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data2=data.iloc[res2.index]
data2_dianliu=data2.ix[:,7]
max2_dianliu=max(data2_dianliu)
min2_dianliu=min(data2_dianliu)
print('类别2的最大最小值为：\n',min2_dianliu,max2_dianliu)

res3 = resSeries[resSeries.values == 3]
print("聚类后类别为3的数据\n",(data.iloc[res3.index]))



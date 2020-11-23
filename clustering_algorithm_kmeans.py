# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:50:10 2020

@author: wangjingxian
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

data=pd.read_csv('E:/data_mining/v_a_guiyihua.csv')

print(data.head())

print(data.describe())


train_data=data.ix[:,1:3]



print(train_data)

model=KMeans(n_clusters=3)

model.fit(train_data)

label = model.labels_

print(label)

predicted_label=model.predict([[0.320347155,0.478602869]])
print('预测标签为：',predicted_label)

# 建立模型。n_clusters参数用来设置分类个数，即K值，这里表示将样本分为两类。
#clf_KMeans = KMeans(n_clusters=3, max_iter=10)
# 模型训练。得到预测值。
print ("聚类中心\n", (model.cluster_centers_))
quantity = pd.Series(model.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))
#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(model.labels_)
res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(data.iloc[res0.index]))

res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(data.iloc[res1.index]))

res2 = resSeries[resSeries.values == 2]
print("聚类后类别为2的数据\n",(data.iloc[res2.index]))

'''
#可以用下面的语句非常简便的返回打印聚类之后每个类别的样本
res0 = data[(model.labels_ == 0)]
print('聚类完成后类别标签为0的样本：',res0)
res1 = data[(model.labels_ == 1)]
print('聚类完成后类别标签为1的样本：',res1)
res2 = data[(model.labels_ == 2)]
print('聚类完成后类别标签为2的样本：',res2)
'''

'''
iris_df=datasets.load_iris()

print(dir(iris_df))

print(iris_df.target)

print(iris_df.target_names)

label={0:'red',1:'blue',2:'green'}

x_axis=iris_df.data[:,0]
y_axis=iris_df.data[:,2]

print(x_axis)
print(y_axis)

plt.scatter(x_axis,y_axis,c=iris_df.target)
plt.show()

#紫罗兰色：山鸢尾，绿色：维吉尼亚鸢尾，黄色：变色鸢尾

model=KMeans(n_clusters=3)

model.fit(iris_df.data)

label = model.labels_ # 可以输出聚类之后数据的标签
print(label)

predicted_label=model.predict([[7.2,3.5,0.8,1.6]])

all_predictions=model.predict(iris_df.data)

print(predicted_label)
print(all_predictions)
'''
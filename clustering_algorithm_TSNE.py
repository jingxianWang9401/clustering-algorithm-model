# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:01:39 2020

@author: wangjingxian
"""

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

'''
data=pd.read_csv('E:/data_mining/v_a_guiyihua.csv')
train_data=data.ix[:,1:3]
'''

iris_df=datasets.load_iris()

model=TSNE(learning_rate=100)

transformed=model.fit_transform(iris_df.data)
#transformed=model.fit_transform(train_data)


'''
#predicted_label=transformed.predict([[7.2,3.5,0.8,1.6]])

all_predictions=transformed.predict(iris_df.data)

#print(predicted_label)
print(all_predictions)
'''
x_axis=transformed[:,0]

y_axis=transformed[:,1]

plt.scatter(x_axis,y_axis,c=iris_df.target)

plt.show()


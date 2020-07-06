#!/usr/bin/env python
# coding: utf-8
# Sangam2019 High Accuracy Traffic Prediction via PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import decomposition
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

le=LabelEncoder()
lr=LinearRegression()

pca=decomposition.PCA()

traffic=pd.read_csv("Train.csv")
traffic
print(traffic.shape)
traffic["is_holiday"].value_counts()
traffic.weather_type=le.fit_transform(traffic.weather_type)
traffic.head()
list(le.classes_)

traffic.loc[traffic['is_holiday']!='None','is_holiday']=1
traffic.loc[traffic['is_holiday']=='None','is_holiday']=0
traffic
traffic['is_holiday'].value_counts()
traffic['date_time'].describe()
traffic[['date','time']]=traffic.date_time.str.split(expand=True)
traffic.head()

col=traffic.columns.tolist()
col=col[-1:]+col[:-1]
col=col[-1:]+col[:-1]
col

traffic=traffic[col]
traffic=traffic.drop('date_time',axis=1)
traffic.head()

traffic['date']= pd.to_datetime(traffic['date'])
traffic.info()
traffic1=traffic
traffic1.head()

traffic1['date']=traffic1['date'].dt.weekday
traffic1['date']=traffic1['date'].astype(int)
traffic1.head()

traffic1['time']=traffic1['time'].str.replace(':00','')
traffic1['time']=traffic1['time'].astype(int)
traffic1.head()
traffic1.info()
traffic1.loc[traffic1['date']=='6','is_holiday']=1

traffic2=traffic1
traffic2=traffic2.drop(['weather_description'],axis=1)
traffic2.head()

traffic2[['temperature','rain_p_h','snow_p_h']]=traffic2[['temperature','rain_p_h','snow_p_h']].astype(int)
traffic2.info()

x_train,x_test,y_train,y_test=train_test_split(traffic2.drop('traffic_volume',axis=1),traffic2['traffic_volume'],test_size=0.2,random_state=5)

lr.fit(x_train,y_train)
prd=lr.predict(x_test)
lr.score(x_test,y_test)
r_sq=r2_score(y_test,prd)
r_sq

pca.n_components=2
pca_data=pca.fit_transform(traffic2)
print('Shape of pca_reduced.shape: ',pca_data.shape)

data=np.vstack((pca_data.T,traffic2['traffic_volume'])).T    #T for tranpose, without which h-stack needs to be used
trafficpca=pd.DataFrame(data,columns=('Col1','Col2','Labels'))
trafficpca.head()

x_train1,x_test1,y_train1,y_test1=train_test_split(trafficpca.drop('Labels',axis=1),trafficpca['Labels'],test_size=0.1,random_state=5)
lr.fit(x_train1,y_train1)
prdpca=lr.predict(x_test1)
prdpca
prdpca.shape

lr.score(x_test1,y_test1)
r_sq1=r2_score(y_test1,prdpca)
r_sq1

accuracy_score(y_test1,prdpca.round())

custom_x=[4,9,0,121,89,2,329,1,1,288,0,0,40,1]

traffic_test=pd.read_csv("Test.csv")
traffic_test.head()
traffic_test.weather_type=le.fit_transform(traffic_test.weather_type)
traffic_test.loc[traffic_test['is_holiday']!='None','is_holiday']=1
traffic_test.loc[traffic_test['is_holiday']=='None','is_holiday']=0
traffic_test.head()
traffic_test[['date','time']]=traffic_test.date_time.str.split(expand=True)
traffic_test.head()

col=traffic_test.columns.tolist()
col=col[-1:]+col[:-1]
col=col[-1:]+col[:-1]
col

traffic_test=traffic_test[col]
traffic_test=traffic_test.drop('date_time',axis=1)
traffic_test.head()
traffic_test['date']= pd.to_datetime(traffic_test['date'])

traffic_test1=traffic_test
traffic_test1.head()
traffic_test1['date']=traffic_test1['date'].dt.weekday
traffic_test1['date']=traffic_test1['date'].astype(int)
traffic_test1['time']=traffic_test1['time'].str.replace(':00','')
traffic_test1['time']=traffic_test1['time'].astype(int)
traffic_test1.head()
traffic_test1.loc[traffic_test1['date']=='6','is_holiday']=1

traffic_test2=traffic_test1
traffic_test2=traffic_test2.drop(['weather_description'],axis=1)
traffic_test2.head()
traffic_test2[['temperature','rain_p_h','snow_p_h']]=traffic_test2[['temperature','rain_p_h','snow_p_h']].astype(int)
traffic_test2.info()

pca.n_components=2
pca_data=pca.fit_transform(traffic_test2)
print('Shape of pca_reduced.shape: ',pca_data.shape)
pca_data

data=np.vstack((pca_data.T)).T    #T for tranpose, without which h-stack needs to be used
traffic_test_pca=pd.DataFrame(data,columns=('Col1','Col2'))
traffic_test_pca.head()

prdpca_test=lr.predict(traffic_test_pca)
prdpca_test.shape

traffic_test_xl = pd.read_csv("Test.csv")
traffic_test_xl.head()
traffic_test_xl['traffic_volume'] = prdpca_test
traffic_test_xl.head()
traffic_test_xl1 = traffic_test_xl[['date_time','traffic_volume']]
traffic_test_xl1.head()
traffic_test_xl1.to_csv('output1.csv',index=False)

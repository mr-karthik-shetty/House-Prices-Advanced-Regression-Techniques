# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:51:08 2018

@author: reach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
Train=pd.read_csv('train.csv')
Test=pd.read_csv('test.csv')

#first check skew of distribution
sb.distplot(Train['SalePrice'],kde=False)
print(Train['SalePrice'].skew())
"""we use log to reduce the skew of the distribution"""
sb.distplot(np.log(Train['SalePrice']),kde=False)
print(np.log(Train['SalePrice']).skew())
target=np.log(Train["SalePrice"])

#extract numerical features
numerical_features=Train.select_dtypes(include=[np.number])

#find correlation of numerical features with saleprice
corr=numerical_features.corr()
print(corr["SalePrice"].sort_values(ascending=False)[:5])
print(corr["SalePrice"].sort_values(ascending=False)[-5:])

"""we use these correlation stats to analyse each feature individually"""

#FEATURE 1(OVERALL QUALITY)
Train["OverallQual"].unique()
#create pivot table and show its bar graph
quality_pivot=Train.pivot_table(index='OverallQual',values='SalePrice',aggfunc=np.median)
quality_pivot.plot(kind='bar',color='blue')
"""We observe that median sale prices strictly increase as Overall Quality increases"""

#FEATURE 2(ABOVE GROUND LIVING AREA)
plt.scatter(x=Train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

#FEATURE 3(GARAGE AREA)
plt.scatter(x=Train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
"""we remove outliers to increase accuracy"""
Train=Train[Train['GarageArea']<1200]
plt.scatter(x=Train['GarageArea'], y=np.log(Train['SalePrice']))
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

#handling missing data
Train.isnull().sum().sort_values(ascending=False)[:25]
"""Drop features with large numberof missing values"""
Train=Train.dropna(axis=1,thresh=Train.shape[0]*0.67)
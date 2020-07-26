# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:41:24 2020

@author: Vimal PM
"""

#importing neccessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
#loading the datasets
Zoo=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\KNN\Zoo.csv")
#columns names
Zoo.columns
#Index(['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne',
      # 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
    #   'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']
#visualizing the data's using barplot and histogram
plt.hist(Zoo.hair)#the animals which don't have hair is  more(zeros are more)
plt.hist(Zoo.feathers)#the animals which dont have feathers is more
plt.hist(Zoo.eggs)#the animals which had eggs is more
plt.hist(Zoo.milk)#the animals which dont have milk is more
plt.hist(Zoo.airborne)#more number of animals which is airborne
sns.barplot(x="aquatic",y="predator",data=Zoo)
sns.barplot(x="toothed",y="backbone",data=Zoo)
sns.barplot(x="breathes",y="venomous",data=Zoo)
plt.hist(Zoo.fins)
plt.hist(Zoo.legs)
plt.hist(Zoo.domestic)
sns.barplot(x="tail",y="catsize",data=Zoo)
#next I would like to see the bussiness moments values  using describe() such as mean,median,mode,std,etc...
Zoo.describe()
             hair    feathers        eggs  ...    domestic     catsize        type
count  101.000000  101.000000  101.000000  ...  101.000000  101.000000  101.000000
mean     0.425743    0.198020    0.584158  ...    0.128713    0.435644    2.831683
std      0.496921    0.400495    0.495325  ...    0.336552    0.498314    2.102709
min      0.000000    0.000000    0.000000  ...    0.000000    0.000000    1.000000
25%      0.000000    0.000000    0.000000  ...    0.000000    0.000000    1.000000
50%      0.000000    0.000000    1.000000  ...    0.000000    0.000000    2.000000
75%      1.000000    0.000000    1.000000  ...    0.000000    1.000000    4.000000
max      1.000000    1.000000    1.000000  ...    1.000000    1.000000    7.000000

#getting the train test data's
train,test=train_test_split(Zoo,test_size=0.3)#70% of train data and 30% of test data
#defining a for loop function to get the accuracy of train test data's
#metioning my K values I want to use
x=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33]
for k in x:
    if k%2!=0:
        neigh=KNC(n_neighbors=k)
        neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
        print("train_accuracy"+str(k)," : "+str(np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])))
        print("test_accuracy"+str(k)," : "+str(np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])))

######below shows the accuracy values that I got"
#train_accuracy1  : 1.0
#test_accuracy1  : 1.0
#train_accuracy3  : 0.9714285714285714
#test_accuracy3  : 0.9354838709677419
#train_accuracy5  : 0.9428571428571428
#test_accuracy5  : 0.9032258064516129
#train_accuracy7  : 0.9428571428571428
#test_accuracy7  : 0.9032258064516129
#train_accuracy9  : 0.8571428571428571
#test_accuracy9  : 0.8709677419354839
#train_accuracy11  : 0.7571428571428571
#test_accuracy11  : 0.8387096774193549
#train_accuracy13  : 0.7714285714285715
#test_accuracy13  : 0.8387096774193549
#train_accuracy15  : 0.7571428571428571
#test_accuracy15  : 0.8387096774193549
#train_accuracy17  : 0.7571428571428571
#test_accuracy17  : 0.8387096774193549
#train_accuracy19  : 0.7428571428571429
#test_accuracy19  : 0.8387096774193549
#train_accuracy21  : 0.6857142857142857
#test_accuracy21  : 0.7419354838709677
#train_accuracy23  : 0.5857142857142857
#test_accuracy23  : 0.5483870967741935
#train_accuracy25  : 0.5857142857142857
#test_accuracy25  : 0.5483870967741935
#train_accuracy27  : 0.5857142857142857
#test_accuracy27  : 0.5483870967741935
#train_accuracy29  : 0.5714285714285714
#test_accuracy29  : 0.5161290322580645
#train_accuracy31  : 0.5571428571428572
#test_accuracy31  : 0.4838709677419355
#train_accuracy33  : 0.5571428571428572
#test_accuracy33  : 0.4838709677419355
        
#by looking on the accuracy I can say that my model is a generalized model.       

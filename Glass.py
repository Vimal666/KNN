# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:34:15 2020

@author: Vimal PM
"""
#Importing the neccessary libraries
import pandas as pd
import numpy as np
#importing knn algorithm and train,test from sklearn module
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split    
import seaborn as sns
import matplotlib.pyplot as plt

#opening the dataset using read_csv()

glass=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\KNN\glass.csv")
glass.columns
#Index(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
#going for visualizations using histogram,barplot
plt.hist(glass.RI)
plt.hist(glass.Na)
plt.hist(glass.Fe)
plt.hist(glass.Type)
sns.barplot(x="Mg",y="Si",data=glass)
sns.barplot(x="Al",y="K",data=glass)
sns.barplot(x="Ca",y="Type",data=glass)

#next I would like to see the bussiness moments values  using describe() such as mean,median,mode,std,etc...
glass.describe()
               RI          Na          Mg  ...          Ba          Fe        Type
count  214.000000  214.000000  214.000000  ...  214.000000  214.000000  214.000000
mean     1.518365   13.407850    2.684533  ...    0.175047    0.057009    2.780374
std      0.003037    0.816604    1.442408  ...    0.497219    0.097439    2.103739
min      1.511150   10.730000    0.000000  ...    0.000000    0.000000    1.000000
25%      1.516523   12.907500    2.115000  ...    0.000000    0.000000    1.000000
50%      1.517680   13.300000    3.480000  ...    0.000000    0.000000    2.000000
75%      1.519157   13.825000    3.600000  ...    0.000000    0.100000    3.000000
max      1.533930   17.380000    4.490000  ...    3.150000    0.510000    7.000000

#getting the train and test data's
train,test=train_test_split(glass,test_size=0.2)#80% for train and 20% for test
#creating a model and mentioning the K values I want to classify in a "for loop" function
x=[1,3,5,7,9,11,13,15,17,19,21]
for k in  x:
    if k%2!=0:
        neigh=KNC(n_neighbors=k)
        neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
        print("train_accuracy"+str(k)+" : "+str(np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])))
        print("test_accuracy"+str(k)+" : "+str(np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])))
#below shows the Accuracy that I got
#train_accuracy1 : 1.0
#test_accuracy1 : 0.627906976744186
#train_accuracy3 : 0.8187134502923976
#test_accuracy3 : 0.5581395348837209
#train_accuracy5 : 0.7543859649122807
#test_accuracy5 : 0.5581395348837209
#train_accuracy7 : 0.7251461988304093
#test_accuracy7 : 0.5813953488372093
#train_accuracy9 : 0.7017543859649122
#test_accuracy9 : 0.5116279069767442
#train_accuracy11 : 0.695906432748538
#test_accuracy11 : 0.5116279069767442
#train_accuracy13 : 0.6842105263157895
#test_accuracy13 : 0.5348837209302325
#train_accuracy15 : 0.6900584795321637
#test_accuracy15 : 0.5116279069767442
#train_accuracy17 : 0.6900584795321637
#test_accuracy17 : 0.5348837209302325
#train_accuracy19 : 0.6900584795321637
#test_accuracy19 : 0.5813953488372093
#train_accuracy21 : 0.672514619883041
#test_accuracy21 : 0.5348837209302325    


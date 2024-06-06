# -*- coding: utf-8 -*-
"""ANN-test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qc6cAALOrOW4m6xT9Rg_NkU7D6q7VXuf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# to develpo Deep Learning AAN architecture :
# we need to follow same rules like Machine learning only model development is different

path = '/content/drive/MyDrive/top mentor deep learning/breast-cancer.csv'
df = pd.read_csv(path)
df.head()

df.isnull().sum()

df  = df.drop(['Unnamed: 32','id'], axis = 1)

df.head()

df['diagnosis'] = df['diagnosis'].map({'M':1 , 'B':0}).astype(int)
df.head()

# splitting the data into x and y variable
x = df.iloc[0: , 1:]
y = df.iloc[: ,:1]

y.head()

x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=42)

x_train.head()

x_test.head()

y_train.head()

y_test.head()

"""while developing the AAN architecture it is important to scale down part of the data**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,sigmoid

ann =  Sequential()
# Sequential is the model architecture

x.shape

ann.add(Dense(units =6,kernel_initializer='he_uniform',activation='relu',input_dim = x.shape[1]))
ann.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
ann.add(Dense(units=1,kernel_initializer='glorot_uniform',activation = 'sigmoid'))

ann.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

reg = ann.fit(x_train,y_train,validation_split=0.1,batch_size=10,epochs=50)

x_train

x_test

y_pred = ann.predict(x_test)
c = []
for i in y_pred:
  if  i[0]> 0.5:
    c.append(1)
  else:
    c.append(0)

c

import sklearn
from sklearn.metrics import accuracy_score

print(f'accuracy :{accuracy_score(y_test,c)}')









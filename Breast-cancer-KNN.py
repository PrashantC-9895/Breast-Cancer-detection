import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNN:
    def __init__(self,path):
        self.path=path
        self.df=pd.read_csv(self.path)
        #print(self.df)
        self.model = KNeighborsClassifier(n_neighbors=5)

    def split_data(self):
        try:
            x = self.df.iloc[ : , 1:]
            y = self.df.iloc[: , 0]
            print(x.shape)
            print(y.shape)
            X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state=42)
            return X_train,X_test,y_train,y_test

        except Exception as e:
            print(f'error in main : {e.__str__()}')


    def model_training(self,X_train,X_test, y_train, y_test):
        try:
            self.model.fit(X_train,y_train)
            y_train_pred = self.model.predict(X_train)
            print(f'the training accuracy : {accuracy_score(y_train,y_train_pred)}')
            y_test_pred=self.model.predict(X_test)
            print(f'test accuracy :{accuracy_score(y_test,y_test_pred)}')


        except Exception as e:
            print(f'error in main : {e.__str__()}')


    def preprocessing(self):
        try:
            print(self.df.isnull().sum())
            self.df = self.df.drop(['Unnamed: 32','id'], axis = 1)  # removing unwanted column
            print(self.df.isnull().sum())
            print(self.df.shape[1])  #checking ttaht 2 rows are removed or not

            X_train, X_test, y_train, y_test = self.split_data()
            print(f'X_train shape :{X_train.shape} X_test shape :{X_test.shape}')
            print(f'y_train shape :{y_train.shape} y_test shape :{y_test.shape}')
            self.model_training(X_train,X_test, y_train, y_test)

        except Exception as e:
            print(f'error in main : {e.__str__()}')



if __name__ == '__main__':
    try:

        obj=KNN('C:/Users/pcrid/TOP MENTOR ALL/TMPC/Breast-Cancer-detection/breast-cancer.csv')
        obj.preprocessing()

    except Exception as e:
        print(f'error in main : {e.__str__()}')


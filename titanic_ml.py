import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as model
from sklearn.metrics import accuracy_score

data=pd.read_csv('Titanic-Dataset.csv')

data

gender=data.groupby('Sex')['Survived'].agg(['mean','sum','count'])

print(gender)

data.isnull().sum()

data.isnull().sum().sum()

data.describe()

data.info()

data['Sex']=data['Sex'].replace({'male':0,'female':1})

data

data['Age'].fillna(data['Age'].median(),inplace=True)

data['Age'].isnull().sum()

data.drop(columns=['Cabin'],inplace=True)
data.drop(columns=['Name'],inplace=True)
data.drop(columns=['Ticket'],inplace=True)

data.info()

data.isnull().sum()

data['Embarked']

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data['Embarked']=data['Embarked'].replace({'S':0,'C':1,'Q':2})

data.isnull().sum().sum()

data.info()

x=data.drop(['Survived','PassengerId'],axis=1)
y=data['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

regression=model()
regression.fit(x_train,y_train)

prediction=regression.predict(x_test)

accuracy_score(prediction,y_test)

print(prediction)

import pickle
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(regression, f)
print("✅ Model saved as titanic_model.pkl")

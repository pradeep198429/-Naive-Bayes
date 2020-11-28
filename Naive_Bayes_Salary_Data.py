import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import train_test_split
 
data1=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Naive_Bayes\\SalaryData_Train.csv")
data2=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Naive_Bayes\\SalaryData_Test.csv")

trainX=data1.iloc[:,0:13]
trainY=data1.iloc[:,13]
testX=data2.iloc[:,0:13]
testY=data2.iloc[:,13]
columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']
from sklearn import preprocessing
for i in columns:
    number=preprocessing.LabelEncoder()
    trainX[i]=number.fit_transform(trainX[i])
    testX[i]=number.fit_transform(testX[i])
    
model=GaussianNB()
model.fit(trainX,trainY)

y_pred=model.predict(testX)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(testY,y_pred)#0.7942
confusion_matrix(testY,y_pred)
classification_report(testY,y_pred)

model2=MultinomialNB()
model2.fit(trainX,trainY)
y_pred2=model2.predict(trainX)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(trainY,y_pred2)#0.7749
confusion_matrix(trainY,y_pred2)
classification_report(trainY,y_pred2)

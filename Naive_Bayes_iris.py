import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Naive_Bayes\\iris.csv")

train,test= train_test_split(df,test_size=0.3)

model=GaussianNB()
model.fit(train.iloc[:,0:4],train.iloc[:,4])

#To find train and test accuracy:

train_acc=np.mean(model.predict(train.iloc[:,0:4])==train.iloc[:,4])
test_acc=np.mean(model.predict(test.iloc[:,0:4])==test.iloc[:,4])

X=df.iloc[:,0:4]
y=df.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model1=GaussianNB()

model1.fit(X_train,y_train)

y_pred=model1.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred.flatten())#0.9111111111111111
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)


############################################################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Naive_Bayes\\iris.csv")

train,test=train_test_split(df,test_size=0.3)

model=MultinomialNB()
model.fit(train.iloc[:,0:4],train.iloc[:,4])

#To find train and test accuracy:

train_acc=np.mean(model.predict(train.iloc[:,0:4])==train.iloc[:,4])
train_acc
test_acc=np.mean(model.predict(test.iloc[:,0:4])==test.iloc[:,4])
test_acc

X=df.iloc[:,0:4]
X.shape
y=df.iloc[:,4]
y.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model2=MultinomialNB()
model2.fit(X_train,y_train)

y_pred2=model2.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred2)# 0.8444444444444444
confusion_matrix(y_test,y_pred2)
classification_report(y_test,y_pred2)

##############################################################################################################################################





















































































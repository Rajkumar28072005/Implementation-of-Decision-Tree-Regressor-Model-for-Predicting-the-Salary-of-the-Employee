# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset. 
7.Predict the values of array. 8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:RAJKUMAR G
RegisterNumber:  212223230166
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![image](https://github.com/user-attachments/assets/a751f846-46a3-42c1-9af8-b5cbcd202225)


![image](https://github.com/user-attachments/assets/e2a5716c-6323-4ea2-b8c5-21b4abe5a29a)


![image](https://github.com/user-attachments/assets/6a56d05b-988c-48fb-8b07-2c0cc13920ed)


![image](https://github.com/user-attachments/assets/ecadde52-f75b-4bff-a4bd-398b0210383b)



![image](https://github.com/user-attachments/assets/469491f7-5b8a-47fa-b30c-99cc2aca5872)

![image](https://github.com/user-attachments/assets/8be0e1c1-f5c2-4240-a005-0679e056770e)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

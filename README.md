# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```Python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Kanagavel A K
RegisterNumber: 212223230096
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv('Salary.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![image](https://github.com/user-attachments/assets/3b6b816d-73c6-4da0-b506-f5c665e1c63a)

![image](https://github.com/user-attachments/assets/56937260-6d26-4eeb-b66f-66f3b923e2d7)

![image](https://github.com/user-attachments/assets/0cc14947-b825-4a25-b916-e933591b8037)

![image](https://github.com/user-attachments/assets/a3fc2f01-1379-4278-8751-7202a4a3aeb5)

![image](https://github.com/user-attachments/assets/b59a4b46-4e61-4606-860c-1456e7c19d1c)

![image](https://github.com/user-attachments/assets/284f5a32-4dde-4017-be1d-05ae23227b56)

![image](https://github.com/user-attachments/assets/9ab6343b-2f76-41a6-abd4-eeebcb92b55b)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset 
2.Data Preprocessing 
3.Feature and Target Selection 
4.Split the Data into Training and Testing Sets 
5.Build and Train the Decision Tree Model 
6.Make Predictions 
7.Evaluate the Model


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shasmithaa Sankar
RegisterNumber: 212224040311
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/26d6f166-bdb5-4308-a812-fccf39490441)
![image](https://github.com/user-attachments/assets/e88c0132-4c09-4f3c-b515-50c5485542fb)
![image](https://github.com/user-attachments/assets/edc5d62a-dca0-4bf3-83a4-2c2c48593939)
![image](https://github.com/user-attachments/assets/e17bf9c1-134a-4ae4-a0bf-4c748a27dbc8)
![image](https://github.com/user-attachments/assets/16736447-acce-4c19-8d84-b78f70c55132)
![image](https://github.com/user-attachments/assets/73315605-c33b-424d-836a-a618c6a2a2e9)
![image](https://github.com/user-attachments/assets/37ace385-6b76-48fe-966e-a9b74a53483d)
![image](https://github.com/user-attachments/assets/c67e10ee-a0fd-4908-a16e-87bea1ed92d5)
![image](https://github.com/user-attachments/assets/e7d8528d-fa83-4fea-a4a4-605fc83dc621)








## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

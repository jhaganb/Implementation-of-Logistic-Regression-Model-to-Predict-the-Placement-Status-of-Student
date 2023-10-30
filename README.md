# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Jhagan B
RegisterNumber:  212220040066
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:

Placement Data:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/24a204ed-2620-4ce6-be4c-a4b59f45faea)

Salary Data:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/29bf55ed-a38d-49ae-89ac-857e8241d233)

Checking the null() function:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/b9f5a432-88bf-4b02-9edb-565ed21444cf)

Data Duplicate:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/fd06e3b4-d49e-4966-8008-83c01fe5face)

Print Data:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/1bd768dc-522f-41f6-aa25-5e7a8cc920eb)

Data-Status:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/87d442fb-0f45-43af-ab1c-a9df0013f303)

Y_prediction array:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/4918f130-3192-41a5-81f8-3c5b11f36c62)

Accuracy value:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/5633a8d4-ecc2-4fed-808b-3f81d0b5b3fa)

Confusion array:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/6e3a74cf-9602-4899-9324-7444ff3be9c3)

Classification Report:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/31ac2852-57ca-4898-9f87-06548b1e790f)

Prediction of LR:

![image](https://github.com/jhaganb/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/63654882/0e554bb3-016c-4c7c-b1d7-a76656d67a52)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

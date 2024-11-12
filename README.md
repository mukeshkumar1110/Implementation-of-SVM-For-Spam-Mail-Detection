# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Mukesh Kumar S
RegisterNumber: 212223240099
*/
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrices
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data Result
![image](https://github.com/user-attachments/assets/0028c826-0a01-4a21-9180-86fecc7da253)

### Data Head
![image](https://github.com/user-attachments/assets/a8091103-bd85-4b66-b430-894c4a2f0e09)

### Data Information
![image](https://github.com/user-attachments/assets/b63639a6-d7ab-45e7-8ae8-2d2272ac8b7d)

### Null Data
![image](https://github.com/user-attachments/assets/69919a25-c326-49a1-99e1-db8fd985ddea)

### Y Pred
![image](https://github.com/user-attachments/assets/5455458c-f526-49ac-b757-10d195256182)

### Accuracy
![image](https://github.com/user-attachments/assets/6d2c74f9-3dc9-463f-95c3-7edd021ca5af)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import necessary libraries (pandas, numpy, matplotlib, sklearn) for data manipulation, visualization, and linear regression.

2.Load the dataset (student_scores.csv) using pandas, then separate the features (X) and target variable (Y), where X represents hours studied and Y represents scores.

3.Split the data into training and testing sets using train_test_split with a 2/3 training and 1/3 testing ratio.

4.Initialize a LinearRegression model, train it using the training data (X_train, Y_train), and make predictions on the test data (Y_pred).

5.Calculate the model's performance using:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

6.Plot the training set and test set results with scatter plots, and overlay the regression line to show the model's fit on both the training and testing data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head(10)
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
##plotting for test data
plt.scatter(x_test,y_test,color="grey")
plt.plot(x_test,y_pred,color="purple")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:

### Head
![Screenshot 2025-03-05 225409](https://github.com/user-attachments/assets/eb01a188-a7c0-401f-a38e-de6422d7e9e9)
### Tail
![Screenshot 2025-03-05 225429](https://github.com/user-attachments/assets/9d3a05fd-bfb4-43c1-9885-35206f5e73f6)
### X Values
![Screenshot 2025-03-05 225448](https://github.com/user-attachments/assets/5e8daf0a-6563-425d-85dd-03b64bd841ba)
### Y Values
![Screenshot 2025-03-05 225505](https://github.com/user-attachments/assets/1b216402-94fa-47dd-90f2-38f79ed4301c)
### Y_Predicted Values
![Screenshot 2025-03-05 225708](https://github.com/user-attachments/assets/5792a960-7bc6-46b7-b915-f0bb02dab370)
### Y_Test Values
![Screenshot 2025-03-05 225752](https://github.com/user-attachments/assets/bafb8ffb-52c1-4894-9638-e4a8d62b7157)
### MSE,MAE AND RMSE
![Screenshot 2025-03-05 225816](https://github.com/user-attachments/assets/aa347465-140c-4fa2-8a95-fd83c25645ae)
### Training Set
![Screenshot 2025-03-05 225841](https://github.com/user-attachments/assets/9f8b488c-5d47-4d80-b68a-449834bbd6b6)
### Testing Set
![Screenshot 2025-03-05 225905](https://github.com/user-attachments/assets/f319fa47-0d6e-40cc-8794-7d1ccd61ef0c)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

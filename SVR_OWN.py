import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading dataset
dataset = pd.read_csv("Position_Salaries.csv");
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Fitting into SVR
from sklearn.svm import SVR
#'rbf' for non linear also use 'poly'
regressor = SVR(kernal='rbf')
regressor.fit(x,y)

#Predicting new results
y_predict = regressor.predict(6.5)

#Visualizing the SVR results
plt.scatter(x,y,color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

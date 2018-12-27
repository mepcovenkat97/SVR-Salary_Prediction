import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading dataset
dataset = pd.read_csv("Position_Salaries.csv");
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
y_train = sc_X.transform(x_test)
sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(y_train)
y_train = sc_Y.transform(y_test)

#Fitting into SVR
from sklearn.svm import SVR
#'rbf' for non linear also use 'poly'
regressor = SVR(gamma='scale', C=1.0, epsilon=0.2,degree = 3)
regressor.fit(x,y)

#Predicting new results
y_predict = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing the SVR results
plt.scatter(x,y,color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

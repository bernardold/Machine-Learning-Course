# -*- coding: utf-8 -*-
"""
Created on Sat May 19 02:35:29 2018

@author: Bernardo Duarte
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # Positing's been mapped as level
y = dataset.iloc[:, 2].values

# Comment chunk to use the whole dataset to train the machine
# Splitting dataset into training set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting the regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result with Decision Tree  Regression
y_pred = regressor.predict(6.5)

# Needs more points to predict and know what happens between the levels to plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression )')
plt.xlabel('Position level')
plt.ylabel('Salary') 
plt.show()
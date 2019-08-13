# Multiple Linear Regression

# Importing the libraries
import os
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(os.path.join(os.path.abspath(''), 'MultipleLinearRegression', 'Garch.csv'))
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(), [1])], remainder="passthrough")
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# printing result
print(y_pred)

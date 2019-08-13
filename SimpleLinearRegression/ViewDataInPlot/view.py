# imports
import os
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv(os.path.join(os.path.abspath(''), 'SimpleLinearRegression', 'bigcity.csv'))
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# results on plot
plt.scatter(X, y, color='red')
plt.title('Data set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

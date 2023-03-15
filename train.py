#import skitlearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#get the data named Salary_Data.csv
dataset = pd.read_csv('Salary_Data.csv')

#split the data into X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#print accuracy

from sklearn.metrics import r2_score

#get the accuracy

acc=r2_score(y_test,y_pred)
#save the model as a pickle in a file
import pickle
pickle.dump(regressor,open('model.pkl','wb'))


print(acc)


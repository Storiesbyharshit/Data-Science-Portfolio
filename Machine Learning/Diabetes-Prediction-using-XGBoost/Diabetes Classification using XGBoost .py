# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
orignaldata = pd.read_csv('diabetes.csv')
data = pd.read_csv('diabetes.csv')



# check if any null value is present
data.isnull().values.any()



true_count = len(data.loc[data['Outcome'] == True])
false_count = len(data.loc[data['Outcome'] == False])


X = data.iloc[:, [0,1,2,3,4,5,6,7]].values
y = data.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#dealing with missing data
from sklearn.preprocessing import Imputer

fill_values = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate = 0.05 , max_depth = 3 , min_child_weight = 1 , gamma = 0.1 , colsample_bytree = 0.3)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()





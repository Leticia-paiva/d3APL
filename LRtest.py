from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from LogisticRegression import LogisticRegression


wine = load_wine()
X = wine['data']
y = wine['target']

print(X.shape)
print(y.shape)
print(f'Labels: {np.unique(y)}')


# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train.shape = {X_train.shape}')
print(f'y_train.shape = {y_train.shape}')
print(f'X_test.shape = {X_test.shape}')
print(f'y_test.shape = {y_test.shape}')

df = pd.DataFrame(X_train, columns=wine['feature_names'])
df

df.describe().transpose()

df.info()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression()
clf

print(clf)

clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred))

plot_confusion_matrix(clf, X_test, y_test, normalize='true')
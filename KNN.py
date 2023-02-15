from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import pandas as pd
import numpy as np


def import_data():
    df = pd.read_excel(r'Damages rate.xlsx')
    df = df.replace(['Y', 'F'], 1)
    df = df.replace(['N', 'B'], 0)
    df = df.replace('S', 2)
    df['Total Bid Amount'] = df['Total Bid Amount'].apply(lambda x: x / 1000000)
    df['Net Change Order Amount'] = df['Net Change Order Amount'].apply(lambda x: x / 1000000)
    df = df.astype('float32')
    return df


def split_data(data):
    X = data.values[:, :8]
    Y = data.values[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    lab_enc = preprocessing.LabelEncoder()
    y_train = lab_enc.fit_transform(y_train)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    return knn


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred


dataFrame = import_data()
X, Y, X_train, X_test, y_train, y_test = split_data(dataFrame)
# Cleaning
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

knn = train(X_train, y_train)
y_pred = prediction(X_test, knn)

# Stats
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
result = sm.OLS(y_test, y_pred).fit()
print("--------------------------   Stats   -----------------------------------")
print("MAE = " + str(mae))
print("MSE = " + str(mse))
print("R^2 = " + str(result.rsquared))


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns
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
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred


def calculate_residuals(model, features, label):
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    return df_results


def linear_assumption(model, features, label):
    print('Linear Relationship between the Target and the Feature', '\n')

    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)

    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()


def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    from statsmodels.stats.diagnostic import normal_ad
    print('The error terms are normally distributed', '\n')

    df_results = calculate_residuals(model, features, label)
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test: ', p_value)

    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')

    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()

    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')


def multicollinearity_assumption(model, features, label, feature_names=None):
    print('Little to no multicollinearity among predictors')

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
    plt.title('Correlation of Variables')
    plt.show()

    VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
    for idx, vif in enumerate(VIF):
        print('{0}: {1}'.format(feature_names[idx], vif))

    # Gathering and printing total cases of possible or definite multicollinearity
    possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
    definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
    print()
    print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
    print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
    print()

    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print('Assumption satisfied')
        else:
            print('Assumption possibly satisfied')
            print()
            print('Coefficient interpretability may be problematic')

    else:
        print('Assumption not satisfied')
        print()
        print('Coefficient interpretability will be problematic')


def autocorrelation_assumption(model, features, label):
    print('No Autocorrelation', '\n')

    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')


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

reg = train(X_train,y_train)
y_pred = prediction(X_test, reg)

# Stats
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("--------------------------   Stats   -----------------------------------")
print("Equation: y = " + str(reg.intercept_) + " + " + str(reg.coef_[0]) + "x1 " + str(reg.coef_[1]) + "x2" + " + " +
      str(reg.coef_[2]) + "x3" + " + " + str(reg.coef_[3]) + "x4 " + str(reg.coef_[4]) + "x5 " + str(reg.coef_[5]) +
      "x6 " + str(reg.coef_[6]) + "x7" + " + " + str(reg.coef_[7]) + "x8")
print("MAE = " + str(mae))
print("MSE = " + str(mse))
print("R^2 = " + str(reg.score(X_train, y_train)))

# Assumptions
print("--------------------------   Assumption 1   -----------------------------------")
linear_assumption(reg, X_train, y_train)
print("--------------------------   Assumption 2   -----------------------------------")
normal_errors_assumption(reg, X_train, y_train)
print("--------------------------   Assumption 3   -----------------------------------")
feature_names = ['BID_DA', 'Road System Type', 'Total Bid Amount', 'Net Change Order Amount',
                 'Pending Change Order Amount', 'Auto Liquidated Damage Indicator', 'Funding Indicator',
                 'Total Adjustment Days']
multicollinearity_assumption(reg, X_train, y_train, feature_names=feature_names)
print("--------------------------   Assumption 4   -----------------------------------")
autocorrelation_assumption(reg, X_train, y_train)

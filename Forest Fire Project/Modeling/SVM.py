from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.robust.scale import mad


# Prepping data for modeling

df = pd.read_csv('C:/Users/Taylor/OneDrive/Desktop/GitHub local repo/Forest-Fire/Forest Fire Project/forestfires.csv')

# ln(x+1) transformation on target

df['area_ln'] = np.log1p(df['area'])

X_svm = df[['FFMC', 'DMC', 'DC', 'ISI', 'wind', 'temp', 'RH']].apply(zscore)

y = df['area_ln']

X_train, X_test, y_train, y_test = train_test_split(
    X_svm,
    y,
    test_size = .3,
    random_state = 11
)

# Modeling

svm_model = SVR()

svm_model.fit(X_train, y_train)

svm_preds = svm_model.predict(X_test)

svm_r2 = r2_score(y_test, svm_preds)
svm_rmse = root_mean_squared_error(y_test, svm_preds)

mad_value = mad(residuals)

#Plotting results

plt.scatter(y_test, svm_preds)
plt.plot(
    [min(y_test), max(y_test)], 
    [min(y_test), max(y_test)], 
    color = 'red', 
    linestyle = '--'
    )
plt.show()

# Plotting residuals

residuals = y_test - svm_preds

plt.scatter(svm_preds, residuals)
plt.axhline(
    y = 0,
    color = 'red',
    linestyle = '--'
)
plt.show()

"""
Imbalance of the target variable is probably creating a systematic bias in the model
which might explain the downward sloping line.  There aren't enough values for the model
to train on so the high number of 0's are skewing it
"""

# Redoing model with same predictors used by the original people except rain

X = df[['temp', 'RH', 'wind']].apply(zscore)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size  = .3,
    random_state = 13
)

svm_model02 = SVR()

svm_model02.fit(X_train, y_train)

svm_preds02 = svm_model02.predict(X_test)

svm_r2_02 = r2_score(y_test, svm_preds02)

residuals = y_test - svm_preds02

mad_value = mad(residuals)

plt.scatter(y_test, svm_preds02)
plt.plot(
    [min(y_test), max(y_test)], 
    [min(y_test), max(y_test)], 
    color = 'red', 
    linestyle = '--'
    )
plt.show()

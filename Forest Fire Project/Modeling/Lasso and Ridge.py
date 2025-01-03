import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

df = pd.read_csv('C:/Users/Taylor/OneDrive/Desktop/GitHub local repo/Forest-Fire/Forest Fire Project/forestfires.csv')

df_corr = df.iloc[:, 4:13]

cor_matrix = df_corr.corr()

sns.heatmap(cor_matrix, annot = True)
plt.show()

"""
Somewhat high positive correlation between DC and DMC then a moderate
negative correlation with RH and temp
"""
#Preparing data for linear regression

df_cat = pd.get_dummies(
    df[['month', 'day']],
    drop_first = True
)

df_standard = df.iloc[:, 4:13].apply(zscore)

linear_df = pd.concat([df_cat, df_standard], axis = 1)

#Baseline LR

X = linear_df.drop(columns = 'area')
y = linear_df['area']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .3,
    random_state = 9
)

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

base_mae = mean_absolute_error(y_test, lr_preds)
base_mse = mean_squared_error(y_test, lr_preds)
base_r2 = r2_score(y_test, lr_preds)
base_rmse = root_mean_squared_error(y_test, lr_preds)

coef_df = pd.DataFrame({
    'Features': X.columns,
    'Coefficients': lr_model.coef_
})

coef_df.loc[-1] = ['Intercept', lr_model.intercept_]


#Applying k-fold cv

cv_model = LinearRegression()

cv_scores  = cross_val_score(
    cv_model, 
    X, 
    y, 
    cv = 10, 
    scoring = 'r2'
    )
cv_scores = -cv_scores

mse_avg = cv_scores.mean()

r2_avg = cv_scores.mean()

#Ridge Method

ridge_model = Ridge()

ridge_model.fit(X_train, y_train)

ridge_preds = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_preds)
ridge_r2 = r2_score(y_test, ridge_preds)
ridge_rmse = root_mean_squared_error(y_test, ridge_preds)

#Lasso Method

lasso_model = Lasso(random_state = 10)

lasso_model.fit(X_train, y_train)

lasso_preds = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_preds)
lasso_r2 = r2_score(y_test, lasso_preds)
lasso_rmse = root_mean_squared_error(y_test, lasso_preds)

#Ridge Parameter Tuning

parameters = {
    'alpha': [100, 200, 1000, 2000, 5000],
    'max_iter': [500, 1000, 2000, 5000],
    'tol': [.001, .0001, .00001, .000001]
}

ridge02 = Ridge()

ridge_parameters = GridSearchCV(
    ridge02, 
    parameters,
    scoring = 'neg_mean_squared_error',
    cv = 5
    )

ridge_parameters.fit(X, y)

ridge_parameters.best_params_

ridge_model02 = Ridge(
    alpha = 2000, 
    max_iter = 500, 
    tol = .001
    )

ridge_model02.fit(X_train, y_train)

ridge_preds02 = ridge_model02.predict(X_test)

ridge_r2_02 = r2_score(y_test, ridge_preds02)
ridge_rmse02 = root_mean_squared_error(y_test, ridge_preds02)

ridge_model02.coef_

#Lasso hyperparameter tuning

lasso_parameters = {
    'alpha': [.001, .01, .1, .5, 1, 10, 100],
    'max_iter': [500, 1000, 2000, 5000],
    'tol': [.001, .0001, .00001, .000001],
    'selection': ['random', 'cyclic']
}

lasso02 = Lasso()

lasso_best = GridSearchCV(
    lasso02, 
    lasso_parameters,
    scoring = 'neg_mean_squared_error',
    cv = 5
    )

lasso_best.fit(X, y)

lasso_best.best_params_

lasso_model02 = Lasso(
    alpha = .1,
    max_iter = 500,
    selection = 'random',
    tol = .001
)

lasso_model02.fit(X_train, y_train)

lasso_preds02 = lasso_model02.predict(X_test)

lasso_r2_02 = r2_score(y_test, lasso_preds02)
lasso_rmse02 = root_mean_squared_error(y_test, lasso_preds02)
lasso_mse02 = mean_squared_error(y_test, lasso_preds02)

lasso_model02.coef_

X_train.columns[21]

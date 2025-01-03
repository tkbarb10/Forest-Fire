import smogn
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from statsmodels.robust.scale import mad

df_smogn = smogn.smoter(
    data = df,
    y = 'area',
    samp_method = 'extreme'
)

df_smogn['area'].value_counts()

df_smogn.columns

X = df_smogn[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']]

y = df_smogn['area']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .3,
    random_state = 10
)

svm_model = SVR()

svm_model.fit(X_train, y_train)

svm_preds = svm_model.predict(X_test)

svm_r2 = r2_score(y_test, svm_preds)
svm_rmse = root_mean_squared_error(y_test, svm_preds)

residuals = y_test - svm_preds

svm_mad = mad(residuals)
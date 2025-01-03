import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


df = pd.read_csv('C:/Users/Taylor/OneDrive/Desktop/GitHub local repo/Forest-Fire/Forest Fire Project/forestfires.csv')

df.head()

"""
Rain column is mostly 0 so dropping that
"""

df.drop(columns = 'rain', inplace = True)

# Histogram of temperatures

df['temp'].plot(kind = 'hist', edgecolor = 'black', bins = 20)
plt.show()

# Histogram of wind

df['wind'].plot(kind = 'hist', edgecolor = 'black', bins = 20)
plt.show()

#Bar graphs of month and days

months = df['month'].value_counts()
months.plot(kind = 'bar')
plt.show()

days = df['day'].value_counts()
days.plot(kind = 'bar')
plt.show()

#Histogram of area

df['area'].plot(kind = 'hist', bins = 30)
plt.show()

#Histogram of area with removing the outliers (can vary range
#to zoom in on lower end)

df['area'].plot(
    kind = 'hist', 
    bins = 30, 
    edgecolor = 'black', 
    range = (0.1, 200)
    )
plt.show()

#Boxplot

df['area'].plot(kind = 'box')
plt.show()

#Set ylim to drop a few of the extremes

df['area'].plot(kind = 'box', ylim = (-1, 300))
plt.show()


#Following are Histograms of the various measures of fire index values

df['FFMC'].plot(kind = 'hist', edgecolor = 'black')
plt.show()

df['DMC'].plot(kind = 'hist', edgecolor = 'black')
plt.show()

df['DC'].plot(kind = 'hist', edgecolor = 'black')
plt.show()

df['ISI'].plot(kind = 'hist', edgecolor = 'black')
plt.show()

df['RH'].plot(kind = 'hist', edgecolor = 'black')
plt.show()

#Comparing these indexes with boxplots

df_index = df.loc[:, ['FFMC', 'DMC', 'DC', 'ISI', 'RH']]

df_index.plot(kind = 'box')
plt.show()

#Normalizing

df_index = df_index.apply(zscore)

df_index.plot(kind = 'box')
plt.show()

"""
FFMC measure shows that some areas seem to be pretty wet on top
but overall all measures seem to indicate similar levels of dryness
"""

df[df['FFMC'] <= 75]

"""
Upon further examination, only one area can be considered wet,
the others range from moderately moit to semi-dry.  So the outliers
show that the vast majority of instances are very dry
"""

#Combining X and Y features into Areas

index = df[['X', 'Y']].value_counts().index.tolist()

area_mapping = {
    coord: f"Area {i+1}" for i, coord in enumerate(index)
    }

df['Area_Name'] = df.apply(
    lambda row: area_mapping[(row['X'], row['Y'])], 
    axis = 1
    )


#Encoding and standardizing data frame to prepare for modeling

df_cat = pd.get_dummies(
    df[['month', 'day', 'Area_Name']], 
    drop_first = True
    )

df_standard = df.iloc[:, 4:12].apply(zscore)

df_modeling = pd.concat([df_cat, df_standard], axis = 1)


# Baseline linear regression

X = df_standard.iloc[:, 0:7]
y = df_modeling['area']

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

mean = y.mean()
print(mean)

mae = mean_absolute_error(y, predictions)
print(mae)

mse = mean_squared_error(y, predictions)
print(mse)

r2 = r2_score(y, predictions)
print(r2)

"""
MAE and MSE are small, below one, but the r^2 is .015 so these
predictors don't really explain any of the variance.  
They also suffer from a lot of outliers and a large number
of 0 values in the area feature.  Will attempt to mitigate through
outlier elimination and oversampling
"""

#Dropping the two outliers where area is over 500 HA

df.drop(index = [238, 415], inplace = True)

X.iloc[:, 0:52] = X.iloc[:, 0:52].astype(bool).astype(int)

y = df_modeling['area']

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = .3, 
    random_state = 1
)

lr_model = sm.OLS(y, X).fit()

lr_model.summary()

y_pred = lr_model.predict(X_test)

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predictor (X)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

#OLS assumes a linear relationship, and plotting the residuals
#shows the model has a pattern of bias in a specific direction
#indicating a non-linear relationship.  Attempting Polynomial
#Regression first

#Polynomial transformation sharply increases number of features,
#so doing poly trans, then PCA decomp

X.drop(columns = 'const', inplace = True)

poly = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly.fit_transform(X)

pca = PCA(n_components = 50)
X_reduced = pca.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced,
    y,
    test_size = .3,
    random_state = 2
)

model_pca = LinearRegression(fit_intercept = True)

result_pca = model_pca.fit(X_train, y_train)

pca_preds = result_pca.predict(X_test)

r2_pca = r2_score(y_test, pca_preds)
print(r2_pca)

#That didn't work, we'll try Ensemble methods

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X,
    y,
    test_size = .3,
    random_state = 3
)

rf_model = RandomForestRegressor(random_state = 3)
rf_model.fit(X_train_rf, y_train_rf)

importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature Names': X.columns,
    'Importances': importances
})

importance_df = importance_df.sort_values(
    by = 'Importances',
    ascending = False
    )

# Running again, this time making sure min_samples_leaf is greater than one to eliminate sparse features

rf_model01 = RandomForestRegressor(
    min_samples_leaf = 5, 
    random_state = 4
    )

rf_model01.fit(X_train_rf, y_train_rf)

importances01 = rf_model01.feature_importances_

importance_df01 = pd.DataFrame({
    'Feature Names': X.columns,
    'Importances': importances01
})

importance_df01 = importance_df01.sort_values(
    by = 'Importances',
    ascending = False
    )


# Trying polynomial regression again with top 10 features

X_top = importance_df01.iloc[0:9]

columns = X_top['Feature Names'].tolist()

subset = X.loc[:, columns]

poly = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly.fit_transform(subset)


X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size = .3,
    random_state = 5
)

model_importances = LinearRegression(fit_intercept = True)
model_importances.fit(X_train, y_train)

preds = model_importances.predict(X_test)

r2 = r2_score(y_test, preds)
print(r2)

"""
insert shrug emoji, lets see what the predictions for the 
Random Forest model are
"""

#Random Forest Model Predictions

rf_preds = rf_model01.predict(X_test_rf)

r2 = r2_score(y_test_rf, rf_preds)
print(r2)

mse = mean_squared_error(y_test, rf_preds)
print(mse)
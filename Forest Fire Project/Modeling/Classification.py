import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = pd.read_csv('C:/Users/Taylor/OneDrive/Desktop/GitHub local repo/Forest-Fire/Forest Fire Project/forestfires.csv')

df.head()

df.drop(columns = 'rain', inplace = True)

#Combining X and Y features into Areas

index = df[['X', 'Y']].value_counts().index.tolist()

area_mapping = {
    coord: f"Area {i+1}" for i, coord in enumerate(index)
    }

df['Area_Name'] = df.apply(
    lambda row: area_mapping[(row['X'], row['Y'])], 
    axis = 1
    )

#(Changing drop_first to True to avoid multicollinearity)

df_cat = pd.get_dummies(
    df[['month', 'day', 'Area_Name']], 
    drop_first = True
    )

df_standard = df.iloc[:, 4:12].apply(zscore)

df_modeling = pd.concat([df_cat, df_standard], axis = 1)

#Dropping the two outliers over 500


#Starting over and attempting this as a classifaction problem

#Converting 'area' feature to 0/1

"""
Add the two outliers back for this
"""

df['area'] = df['area'].apply(lambda x: 1 if x > 0 else x)

df['area'] = df['area'].astype(int)

#Preparing data for modeling

df_cat = pd.get_dummies(
    df[['month', 'day', 'Area_Name']], 
    drop_first = True
    )

df_standard = df.iloc[:, 4:12]

df_modeling = pd.concat([df_cat, df_standard], axis = 1)

#Attempt Random Forest again

y_class = df['area']

X_train, X_test, y_train, y_test = train_test_split(
    df_modeling,
    y_class,
    test_size = .3,
    random_state = 6
)

rf_model = RandomForestClassifier(
    min_samples_leaf = 5,
    oob_score = True,
    random_state = 6
    )

rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_test)

print(classification_report(y_test, preds))

"""
The most prominent value in the target class is 1, which makes
up 52% of the total.  Acccuracy of the model is 56% so barely
better than baseline.  Let's drop some of the features that 
are mostly 0 and see if we can improve this
"""

df_modeling.drop(
    columns = ['month_dec', 'month_nov', 
               'month_jan', 'month_may'], inplace = True
               )

subset = df['Area_Name'].value_counts()[df['Area_Name'].value_counts() < 10].index.tolist()

subset = ['Area_Name_' + i for i in subset]

df_modeling.drop(columns = subset, inplace = True)

#RF Modeling round 2

X_train, X_test, y_train, y_test = train_test_split(
    df_modeling,
    y_class,
    test_size = .3,
    random_state = 7
)

rf_model02 = RandomForestClassifier(
    min_samples_leaf = 5,
    oob_score = True,
    random_state = 7
    )

rf_model02.fit(X_train, y_train)

preds02 = rf_model02.predict(X_test)

print(classification_report(y_test, preds02))

"""
Well that lowered the accuracy of the model so not the most effective tactic
"""

df_modeling_const = sm.add_constant(df_modeling)

boolean_columns = df_modeling_const.select_dtypes(include = 'bool').astype(int).columns
df_modeling_const[boolean_columns] = df_modeling_const[boolean_columns].astype(int)

vif = pd.DataFrame()
vif['Feature'] = df_modeling_const.columns
vif['VIF'] = [variance_inflation_factor(df_modeling_const.values, i) for i in range(df_modeling_const.shape[1])]

vif = vif.sort_values(by = 'VIF', ascending = False)

subset = vif[1:7]['Feature'].values

df_modeling.drop(columns = subset, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(
    df_modeling,
    y_class,
    test_size = .3,
    random_state = 8
)

rf_model03 = RandomForestClassifier(
    min_samples_leaf = 5,
    oob_score = True,
    random_state = 8
    )

rf_model03.fit(X_train, y_train)

preds03 = rf_model03.predict(X_test)

print(classification_report(y_test, preds03))

"""
Captured more of the areas that had been burned but at the 
expense of everything else.  Resetting and starting from scratch
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    df, 
    columns = ['month', 'day', 'Area_Name'], 
    drop_first = False,
    prefix_sep = '_'
    )

df_standard = df.iloc[:, 4:12].apply(zscore)

df_modeling = pd.concat([df_cat, df_standard], axis = 1)

df_modeling.drop(columns = ['X', 'Y'], inplace = True)

# Baseline linear regression

X = df_modeling.iloc[:, 0:7]
y = df_modeling['area']

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
print(mae)

mse = mean_squared_error(y, predictions)
print(mse)

r2 = r2_score(y, predictions)
print(r2)

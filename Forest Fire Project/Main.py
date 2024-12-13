import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


df = pd.read_csv('C:/Users/Taylor/OneDrive/Desktop/GitHub local repo/Forest-Fire/Forest Fire Project/forestfires.csv')

# ln(x+1) transformation on target

df['area_ln'] = np.log1p(df['area'])

# Checking transformed target against other numerical predictors

df.columns

df.plot(kind = 'scatter', x = 'area_ln', y = 'ISI')
plt.show()

sns.regplot(x = 'area_ln', y = 'ISI', data = df, lowess = True)
plt.show()

sns.regplot(x = 'area_ln', y = 'wind', data = df, order = 3, scatter = True)
plt.show()

# All flat curves so transforming ISI since only one with right skew

df['ISI'].plot(kind = 'hist')
plt.show()

df['ISI_ln'] = np.log1p(df['ISI'])

#Plotting log transformation

df.plot(kind = 'scatter', x = 'area_ln', y = 'ISI_ln')
plt.show()


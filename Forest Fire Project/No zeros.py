df_no_zero = df[df['area'] > 0]

# SVM modeling

X = df_no_zero[['FFMC', 'DMC', 'DC', 'ISI', 'wind', 'temp', 'RH']].apply(zscore)
y = zscore(df_no_zero['area'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .25,
    random_state = 14
)

svm = SVR()

svm.fit(X_train, y_train)

preds = svm.predict(X_test)

r2_no_zero = r2_score(y_test, preds)

plt.scatter(y_test, preds)
plt.plot(
    [min(y_test), max(y_test)], 
    [min(y_test), max(y_test)], 
    color = 'red', 
    linestyle = '--'
    )
plt.show()

# Drop the outliers and redo

df_filtered = df_no_zero[df_no_zero['area'] < 100]

X = df_filtered[['FFMC', 'DMC', 'DC', 'ISI', 'wind', 'temp', 'RH']].apply(zscore)
y = zscore(df_filtered['area'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .25,
    random_state = 15
)

svm = SVR()

svm.fit(X_train, y_train)

preds = svm.predict(X_test)

r2_no_zero = r2_score(y_test, preds)

plt.scatter(y_test, preds)
plt.plot(
    [min(y_test), max(y_test)], 
    [min(y_test), max(y_test)], 
    color = 'red', 
    linestyle = '--'
    )
plt.show()


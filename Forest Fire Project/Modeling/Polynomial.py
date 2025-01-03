from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay

features = df[['FFMC', 'DMC', 'DC', 'ISI', 'wind', 'temp', 'RH']]
y = df['area']

poly = PolynomialFeatures(
    degree = 2,
    interaction_only = True,
    include_bias = False
    )

X_poly = poly.fit_transform(features)

interaction_df = pd.DataFrame(
    X_poly, 
    columns = poly.get_feature_names_out(features.columns)
    )

# Attempting RF ensemble

X_train, X_test, y_train, y_test = train_test_split(
    interaction_df,
    y,
    test_size = .3,
    random_state = 16
)

rf_model = RandomForestRegressor(random_state = 16)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_preds)

# Parameter tuning

parameters = {
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'max_features': [2, 'log2', 'sqrt'],
    'min_samples_leaf': [2, 5, 10]
}

rf_model02 = RandomForestRegressor()

best_params = GridSearchCV(
    rf_model02,
    param_grid = parameters,
    cv = 5,
    scoring = 'neg_mean_squared_error',
    n_jobs = -1
)

best_params.fit(X_train, y_train)

best_params.best_params_

rf_best = RandomForestRegressor(
    n_estimators = 100,
    max_features = 2,
    min_samples_leaf = 10,
    min_samples_split = 2,
    random_state = 20,
    oob_score = True
)

rf_best.fit(X_train, y_train)

preds = rf_best.predict(X_test)

r2_best = r2_score(y_test, preds)

rf_best.feature_importances_

importance_df = pd.DataFrame()
importance_df['Features'] = X_train.columns
importance_df['Coefficient'] = rf_best.feature_importances_

sorted_df = importance_df.sort_values(
    by = 'Coefficient',
    ascending = False
    )

best_columns = sorted_df['Features'].head(10)

# Partial dependence Plot

PartialDependenceDisplay.from_estimator(
    rf_best,
    X_train,
    features = best_columns,
    kind = 'average',
    grid_resolution = 75,
    n_cols = 5
)

plt.suptitle('Partial Dependence Plots')
#plt.tight_layout()
plt.show()
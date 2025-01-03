**Data variations used in modeling**

- ln(x + 1) transformation of target variable
- Dropping all rows with values of 0 for area
- Using all predictors, one hot encoding for the categorical variables and standarizing the numerical ones
- Oversampling of the target variable to balance out the 0's using SMOTE for regression
- Created interaction terms using a polynomial transformation
- Feature engineering using top 10 features based on relative predictive power from a Random Forest ensemble
- Performed PCA decomposition on data frame
- Attempted ln(x + 1) transformation on predictors
- Left predictors in original state
- Transformed target into binary for classification attempt
- Calculated variation inflation factors for each predictors and used the predictors with low VIF for modeling

**Models Attempted**

- Linear Regression
- Random Forest ensemble for Regression
- Random Forest Ensemble for Classification
- Lasso and Ridge Regression
- Polynomial Regression
- Support Vector Machines

**Metrics Used**

- Mean Squared Error
- R^2 
- Root Mean Squared Error
- Mean Absolute Deviation
- Accuracy
- F-score
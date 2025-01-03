**Results**

The lowest Median Absolute Deviation (MAD), the metric chosen by the original authors, was 1.13. This was achieved using an SVM model, an ln(x+1) transformation of the target variable, and standardized predictors: 'FFMC', 'DMC', 'DC', 'ISI', 'wind', 'temp', and 'RH'. However, the best R^2 value achieved across all models and data variations remained negative. This indicates that while the model predicts values close to the median of the true values, it fails to capture the variability in the data.

When reformulated as a classification problem, the best accuracy achieved was 56%, slightly better than random chance, but nothing to write home about

**Limitations**

The primary limitation is the lack of sufficient data. The dataset has a small number of samples, with approximately 50% of them imbalanced toward zero. This imbalance skews predictions toward lower values, explaining the low MAD and MSE scores but the negative R^2 values.

**Suggestions**

Better success might be found in trying to determine the risk to a particular area rather than trying to predict how much fire damage an area might sustain
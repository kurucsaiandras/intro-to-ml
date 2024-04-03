import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np
from linear_regression import *
from sklearn.model_selection import KFold

# Define the KFold for the outer loop
kf_outer = KFold(n_splits=10, shuffle=True, random_state=42)

# Define a range of lambda values to explore in the inner loop
lambda_range = np.logspace(-4, 4, 50)

# Placeholder for recording results
results = []

# Outer loop
for train_index, test_index in kf_outer.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Inner loop to find optimal lambda using nested cross-validation
    inner_cv_scores = []
    for alpha in lambda_range:
        inner_ridge = Ridge(alpha=alpha)
        # Perform inner cross-validation
        inner_scores = cross_val_score(inner_ridge, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
        inner_cv_scores.append(np.mean(inner_scores))
    
    # Find the optimal lambda for this outer fold
    optimal_lambda_inner = lambda_range[np.argmax(inner_cv_scores)]
    ridge_optimal_inner = Ridge(alpha=optimal_lambda_inner)
    ridge_optimal_inner.fit(X_train, y_train)
    
    # Evaluate on the test set of the outer loop
    y_pred = ridge_optimal_inner.predict(X_test)
    test_error = mean_squared_error(y_test, y_pred)
    
    # Record the optimal lambda and test error for this outer fold
    results.append((optimal_lambda_inner, test_error))

results_df = pd.DataFrame(results, columns=['Optimal Lambda', 'Test Error'])

print(results_df)
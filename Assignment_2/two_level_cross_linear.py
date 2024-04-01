import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np
from linear_regression import *
from sklearn.model_selection import KFold, GridSearchCV

# Outer loop: K1 = 10 folds
kf_outer = KFold(n_splits=10, shuffle=True, random_state=42)

# Inner loop: K2 = 10 folds for finding the optimal lambda
kf_inner = KFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {'alpha': np.logspace(-4, 4, 50)}

results = []

# Outer loop
for train_index, test_index in kf_outer.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Inner loop: Grid search to find the optimal lambda
    model = GridSearchCV(Ridge(), param_grid, cv=kf_inner, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    
    # Optimal lambda for this fold
    optimal_lambda = model.best_params_['alpha']
    
    # Train model on the entire training set using the optimal lambda
    model_optimal = Ridge(alpha=optimal_lambda)
    model_optimal.fit(X_train, y_train)
    
    # Evaluate on the test set to obtain the squared loss per observation
    y_pred = model_optimal.predict(X_test)
    test_error = np.mean((y_test - y_pred) ** 2) 

    # Store the results
    results.append({'optimal_lambda': optimal_lambda, 'test_error': test_error})

results_df = pd.DataFrame(results)
results_df.index += 1  
results_df.index.name = 'i'
results_df.rename(columns={'optimal_lambda': 'λ*_i', 'test_error': 'Eitest'}, inplace=True)
print(results_df)
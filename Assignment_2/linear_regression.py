import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('Life-Expectancy-Data.csv')

# Drop the 'Country' column
data = data.drop(['Country'], axis=1)

# Target Life Expectancy
X = data.drop('Life_expectancy', axis=1)
y = data['Life_expectancy']

# one-of-K encoding of Region and standardization of numerical features
# Identify numerical columns (excluding 'Region' and 'Year' as Year will be treated as a categorical due to its nature in this context)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pipeline for numerical features
numerical = Pipeline([
    ('std_scaler', StandardScaler())
])

# Complete preprocessing pipeline
transformer = ColumnTransformer([
    ('num', numerical, numerical_cols),
    ('cat', OneHotEncoder(), ['Region'])
])

# Apply transformations
X_normalized = transformer.fit_transform(X)

# Define a range of lambda values
lambda_values = np.logspace(-4, 4, 50)

# Placeholder for storing cross-validation scores
cv_scores = []

# Perform 10-fold cross-validation for each value of lambda (alpha)
for alpha in lambda_values:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_normalized, y, scoring='neg_mean_squared_error', cv=10)
    rmse = np.sqrt(-scores)
    cv_scores.append(rmse.mean())

# Find the lambda value with the minimum average RMSE
min_error = np.min(cv_scores)
optimal_lambda = lambda_values[np.argmin(cv_scores)]

print(optimal_lambda)

# Plotting the generalization error (RMSE) as a function of lambda
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, cv_scores, marker='o', linestyle='-', color='blue', label='RMSE per Lambda')
plt.axvline(x=optimal_lambda, color='red', linestyle='--', label=f'Optimal Lambda = {optimal_lambda:.2f}')
plt.xscale('log')
plt.xlabel('Lambda (Regularization Strength)')
plt.ylabel('Estimated Generalization Error (RMSE)')
plt.title('Generalization Error as a Function of Lambda in Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()


# Fit the Ridge Regression model with the optimal lambda value
ridge_optimal = Ridge(alpha=optimal_lambda)
ridge_optimal.fit(X_normalized, y)

# Extract the coefficients
coeff = ridge_optimal.coef_
feature_names = numerical_cols + list(transformer.named_transformers_['cat'].get_feature_names_out())
# Create a DataFrame to display feature names and their corresponding coefficients
coef_df = pd.DataFrame(data={'Feature': feature_names, 'Coefficient': coeff})
coef_df.sort_values(by='Coefficient', key=abs, ascending=False).reset_index(drop=True)

print(coef_df)



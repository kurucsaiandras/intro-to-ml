import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv('intro-to-ml/Life-Expectancy-Data.csv')

data_transformed = data.drop(columns=['Country'])

# Define the columns for transformations
categorical_features = ['Region']
numeric_features = data_transformed.columns.drop(['Region', 'Life_expectancy'])

# Create transformers for the pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')  # Drop first to avoid dummy variable trap

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
X_transformed = preprocessor.fit_transform(data_transformed.drop(columns=['Life_expectancy']))
y = data_transformed['Life_expectancy']

# Convert transformed X back to a DataFrame for better readability
column_names = list(numeric_features) + \
               list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

X_transformed_df = pd.DataFrame(X_transformed, columns=column_names)

##########################################
#b.2
lambda_values = np.logspace(-4, 4, 50)

# K-Fold cross-validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store the average MSE for each value of lambda
avg_mse_scores = []

for lambda_value in lambda_values:
    model = Ridge(alpha=lambda_value)
    
    # Negative MSE scores, because cross_val_score returns negative values for MSE (the higher, the better)
    mse_scores = cross_val_score(model, X_transformed, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Convert back to positive MSE and calculate the average
    avg_mse_scores.append(-np.mean(mse_scores))

# Find the lambda value that minimizes the MSE
optimal_lambda = lambda_values[np.argmin(avg_mse_scores)]
min_mse = min(avg_mse_scores)

optimal_lambda, min_mse
# plt.figure(figsize=(10, 6))
# plt.semilogx(lambda_values, avg_mse_scores, marker='o', linestyle='-', color='b')
# plt.xlabel('Lambda (Regularization strength)')
# plt.ylabel('Mean Squared Error (Generalization Error)')
# plt.title('Generalization Error vs. Regularization Strength')
# plt.axvline(x=optimal_lambda, color='r', linestyle='--', label=f'Optimal λ = {optimal_lambda:.2f}')
# plt.legend()
# plt.grid(True)
# plt.show()

##################################
#a.3
# Train the Ridge regression model with the optimal lambda value
model_optimal = Ridge(alpha=optimal_lambda)
model_optimal.fit(X_transformed, y)

# Get the model's coefficients (weights) and the intercept
weights = model_optimal.coef_
intercept = model_optimal.intercept_

# Display the weights and the intercept
print(weights, intercept)
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn import model_selection

def get_norm_params(set, exclude_cols):
    means = set.mean(axis=0)
    stds = set.std(axis=0)
    # Not normalizing columns with binary data
    means[exclude_cols] = 0
    stds[exclude_cols] = 1
    return means, stds

path = 'Assignment_2/output/REGR/'

dataset = pd.read_csv("Life-Expectancy-Data.csv")
attributeNames = np.asarray(dataset.columns)

X = dataset.drop(["Life_expectancy", "Country"], axis=1)

X = pd.get_dummies(X, dtype=float).values

# The property we want to predict

### TODO: Choose this for regression
y = dataset["Life_expectancy"].values

# Identify binary columns (assuming binary columns contain only 0s and 1s)
binary_columns = [col for col in range(X.shape[1]) if len(np.unique(X[:,col])) == 2]

your_params= np.logspace(-4, 4, 50)

models = []

for p in your_params:
    models.append(Ridge(p))

# Outer Cross validation
K_o = 10
CV_o = model_selection.KFold(n_splits=K_o,shuffle=True, random_state=42)

# Lists for saving the results
E_gen_o = [] # Outer generalization errors
### TODO: Create a list also to save the optimal parameters for the models
your_param_results = []


num_fold = 1
for train_val_idx, test_idx in CV_o.split(X):
    # extract training and validation set for current CV fold
    X_train_val = X[train_val_idx,:]

    # Inner cross validation
    K_i = 10
    CV_i = model_selection.KFold(n_splits=K_i,shuffle=True, random_state=42)

    # List to store errors for the model types
    E_gen_i = [0] * len(models)

    for train_idx, val_idx in CV_i.split(X_train_val):
        # extract training and test set for current CV fold
        X_train = X[train_idx,:]

        means, stds = get_norm_params(X_train, binary_columns)

        X_train = (X_train - means) / stds
        y_train = y[train_idx]
        X_val = (X[val_idx,:] - means) / stds
        y_val = y[val_idx]

        for i, model in enumerate(models):
            print(f'Training model {i}')
            # Train the model using X_train and y_train
            model.fit(X_train, y_train)  # This line trains the model
    
            # Get y_est by running X_val through the trained model
            y_est = model.predict(X_val)  # This line makes predictions
            
            ### TODO: Gen error for regression:
            E_gen_i[i] += np.square(y_val - y_est).sum()/y_val.shape[0]
        # Dividing generalization error with number of folds
    E_gen_i = [E_gen / K_i for E_gen in E_gen_i]

    # Choosing best model
    best_model_idx = E_gen_i.index(min(E_gen_i))
    best_model = models[best_model_idx]

    # Log
    print(f'Best model in fold {num_fold}:')
    print(f'\t<Your_parameter>:\t{your_params[best_model_idx]}') ### TODO: Rename to your paramter
    print(f'\tGen. error:\t{E_gen_i[best_model_idx]}')

    # Prepare data for training
    means, stds = get_norm_params(X_train_val, binary_columns)

    X_train_val = (X_train_val - means) / stds
    y_train_val = y[train_val_idx]
    X_test = (X[test_idx,:] - means) / stds
    y_test = y[test_idx]

    best_model.fit(X_train_val, y_train_val)  # Use the best model found in the inner CV loop
    
    # Get y_est by running X_test through the trained model
    y_est = best_model.predict(X_test)  # Make predictions on the test set

    E_gen_o.append(np.square(y_test - y_est).sum()/y_test.shape[0])

    your_param_results.append(your_params[best_model_idx])  ### TODO: Rename to your paramter

    # Save true and estimated values for further analization
    np.save(path + f'fold_{num_fold}_y_true.npy', y_test)
    np.save(path + f'fold_{num_fold}_y_est.npy', y_est)

    num_fold += 1

# Save generalization error and model parameters for the outer folds
print("Generalization errors:")
print(E_gen_o)
print("Optimal numbers of lambdas:") ### TODO: Rename to your paramter
print(your_param_results) ### TODO: Rename to your paramter

np.save(path + 'E_gen_o.npy', np.asarray(E_gen_o))
np.save(path + 'your_param_results.npy', np.asarray(your_param_results)) ### TODO: Rename to your paramter

results_df = pd.DataFrame({
    'Fold': range(1, K_o + 1),
    'Optimal Lambda (λ*ᵢ)': your_param_results,
    'Estimated Generalization Error (Eᵢ_test)': E_gen_o
})

print(results_df)
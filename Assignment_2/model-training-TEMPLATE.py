import pandas as pd
import numpy as np
from sklearn import model_selection

def get_norm_params(set, exclude_cols):
    means = set.mean(axis=0)
    stds = set.std(axis=0)
    # Not normalizing columns with binary data
    means[exclude_cols] = 0
    stds[exclude_cols] = 1
    return means, stds

# Path to save output to
### TODO: Create your own folder
path = 'Assignment_2/output/<your_folder>/'

dataset = pd.read_csv("Life-Expectancy-Data.csv")
attributeNames = np.asarray(dataset.columns)
# Dropping country because ot would generate a lot of attributes
### NOTE: Rasmus will drop more columns here probably

### TODO: Choose this for regression
X = dataset.drop(["Life_expectancy", "Country"], axis=1)

### TODO: Choose this for classification
X = dataset.drop(["Economy_status_Developed", "Economy_status_Developing", "Country"], axis=1)

# One hot key encode Region
X = pd.get_dummies(X, dtype=float).values

# The property we want to predict

### TODO: Choose this for regression
y = dataset["Life_expectancy"].values

### TODO: Choose this for classification
y = dataset["Economy_status_Developed"].values

# Identify binary columns (assuming binary columns contain only 0s and 1s)
binary_columns = [col for col in range(X.shape[1]) if len(np.unique(X[:,col])) == 2]

### TODO: Instantiate your models with different parameters (e.g. lambdas or num of hidden layers)
your_params = ...
models = []
for p in your_params:
    models.append(MODEL(p)) ### TODO: create the model with parameter p

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
            ### TODO: Train the model with using X_train and y_train
            ### TODO: Get y_est with running X_val through the trained model 
            y_est = ...

            ### TODO: CHOOSE ONE OF THESE:
            
            ### TODO: Gen error for regression:
            E_gen_i[i] += np.square(y_val - y_est).sum()/y_val.shape[0]

            ### TODO: Gen error for classification:
            y_est = (y_est > 0.5)
            e = y_est != y_val
            E_gen_i[i] += (sum(e) / len(y_val))
    
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

    # Train and evaluate model
    ### Train the model with using X_train_val and y_train_val
    ### Get y_est with running X_test through the trained model 
    y_est = ...

    ### TODO: CHOOSE ONE OF THESE:
    ### TODO: Gen error for regression:
    E_gen_o.append(np.square(y_test - y_est).sum()/y_test.shape[0])

    ### TODO: Gen error for classification:
    y_est = (y_est > 0.5)
    e = y_est != y_test
    E_gen_o.append(sum(e) / len(y_test))
    
    ### TODO: Save the optimal model parameter for this fold
    your_param_results.append(your_params[best_model_idx])  ### TODO: Rename to your paramter

    # Save true and estimated values for further analization
    np.save(path + f'fold_{num_fold}_y_true.npy', y_test)
    np.save(path + f'fold_{num_fold}_y_est.npy', y_est)

    num_fold += 1

# Save generalization error and model parameters for the outer folds
print("Generalization errors:")
print(E_gen_o)
print("Optimal numbers of <your_param>:") ### TODO: Rename to your paramter
print(your_param_results) ### TODO: Rename to your paramter

np.save(path + 'E_gen_o.npy', np.asarray(E_gen_o))
np.save(path + '<your_param>.npy', np.asarray(your_param_results)) ### TODO: Rename to your paramter
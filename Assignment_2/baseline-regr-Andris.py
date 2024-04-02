import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm

def get_norm_params(set, exclude_cols):
    means = set.mean(axis=0)
    stds = set.std(axis=0)
    # Not normalizing columns with binary data
    means[exclude_cols] = 0
    stds[exclude_cols] = 1
    return means, stds

# Path to save output to
path = 'Assignment_2/output/baseline-regr/'

dataset = pd.read_csv("Life-Expectancy-Data.csv")
attributeNames = np.asarray(dataset.columns)
# Dropping country because it would generate a lot of attributes
### NOTE: Rasmus will drop more columns here probably
X = dataset.drop(["Life_expectancy", "Country"], axis=1)

# One hot key encode Region
X = pd.get_dummies(X, dtype=float).values

# The property we want to predict
y = dataset["Life_expectancy"].values

# Identify binary columns (assuming binary columns contain only 0s and 1s)
binary_columns = [col for col in range(X.shape[1]) if len(np.unique(X[:,col])) == 2]

# Outer Cross validation
K_o = 10
CV_o = model_selection.KFold(n_splits=K_o,shuffle=True, random_state=42)

# Lists for saving the results
E_gen_o = [] # Outer generalization errors

num_fold = 1
for train_idx, test_idx in CV_o.split(X):

    y_train = y[train_idx]
    y_test = y[test_idx]

    y_est = np.ones(y_test.shape) * y_train.mean(axis=0)

    E_gen_o.append(np.square(y_test - y_est).sum()/y_test.shape[0])

    # Save true and estimated values for further analization
    np.save(path + f'fold_{num_fold}_y_true.npy', y_test)
    np.save(path + f'fold_{num_fold}_y_est.npy', y_est)

    num_fold += 1

# Save generalization error and model parameters for the outer folds
print("Generalization errors:")
print(E_gen_o)

np.save(path + 'E_gen_o.npy', np.asarray(E_gen_o))
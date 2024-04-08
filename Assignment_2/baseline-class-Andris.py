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
path = 'Assignment_2/output/baseline-class/'

dataset = pd.read_csv("Life-Expectancy-Data.csv")
attributeNames = np.asarray(dataset.columns)

X = dataset.drop(["Economy_status_Developed", "Economy_status_Developing", 
                  "Country", "GDP_per_capita", "Life_expectancy",
                  "Hepatitis_B","Infant_deaths", 
                  "Region"], axis=1).values

# The property we want to predict
y = dataset["Economy_status_Developed"].values

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
    y_est = (y_est > 0.5)
    e = y_est != y_test
    E_gen_o.append(sum(e) / len(y_test))

    # Save true and estimated values for further analization
    np.save(path + f'fold_{num_fold}_y_true.npy', y_test)
    np.save(path + f'fold_{num_fold}_y_est.npy', y_est)

    num_fold += 1

# Save generalization error and model parameters for the outer folds
print("Generalization errors:")
print(E_gen_o)

np.save(path + 'E_gen_o.npy', np.asarray(E_gen_o))
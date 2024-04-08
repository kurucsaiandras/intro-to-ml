import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm

# ANN with arbitrary number of hidden layers
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_of_hidden):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_of_hidden)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Apply Xavier initialization to the model's parameters
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)

# Trains an ANN with the given number of epochs (or until reaching tolerance)
def train(model, X, y, epochs=10000, tolerance=1e-6):
    # Apply initialization to all model parameters
    model.apply(initialize_weights)

    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()
    old_loss = 1e6

    for i in range(epochs):
        y_est = model(X).squeeze()  # forward pass, predict labels on training set
        loss = loss_fn(y_est, y)  # determine loss
        loss_value = loss.cpu().data.numpy()  # get numpy array instead of tensor

        # Convergence check, see if the percentual loss decrease is within
        # tolerance:
        p_delta_loss = np.abs(loss_value - old_loss) / old_loss
        if p_delta_loss < tolerance:
            break
        old_loss = loss_value

        # do backpropagation of loss and optimize weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0: print(f'Epoch: {i}\tLoss: {loss_value}')

    return model

def get_norm_params(set, exclude_cols):
    means = set.mean(axis=0)
    stds = set.std(axis=0)
    # Not normalizing columns with binary data
    means[exclude_cols] = 0
    stds[exclude_cols] = 1
    return means, stds

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Path to save output to
path = 'Assignment_2/output/ANN-class/'

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

# Generate ANN models with different number of hidden layers
ANNs = []
input_size = X.shape[1]
hidden_size = 20
output_size = 1
num_of_hidden = [1, 3, 5, 7]
for h in num_of_hidden:
    ANNs.append(ANN(input_size, hidden_size, output_size, h))

# Outer Cross validation
K_o = 10
CV_o = model_selection.KFold(n_splits=K_o,shuffle=True, random_state=42)

# Lists for saving the results
E_gen_o = []
nums_of_hidden = []

progress_bar = tqdm(total=K_o)
num_fold = 1
for train_val_idx, test_idx in CV_o.split(X):
    # extract training and test set for current CV fold
    X_train_val = X[train_val_idx,:]

    # Inner cross validation
    K_i = 10
    CV_i = model_selection.KFold(n_splits=K_i,shuffle=True, random_state=42)

    # List to store errors for the model types
    E_gen_i = [0] * len(ANNs)

    for train_idx, val_idx in CV_i.split(X_train_val):
        # extract training and test set for current CV fold
        X_train = X[train_idx,:]

        means, stds = get_norm_params(X_train, binary_columns)

        X_train = torch.Tensor((X_train - means) / stds)
        y_train = torch.Tensor(y[train_idx])
        X_val = torch.Tensor((X[val_idx,:] - means) / stds)
        y_val = torch.Tensor(y[val_idx]).type(dtype=torch.uint8).to(device)
        
        for i, model in enumerate(ANNs):
            print(f'Training model {i}')
            net = train(model, X_train, y_train, epochs=100)
            y_sigmoid = net(X_val.to(device)).squeeze()
            y_est = (y_sigmoid > 0.5).type(dtype=torch.uint8)
            e = y_est != y_val
            E_gen_i[i] += (sum(e).type(torch.float) / len(y_val)).cpu().data.numpy()
        
        progress_bar.update(1.0/K_i)
    
    # Dividing generalization error with number of folds
    E_gen_i = [E_gen / K_i for E_gen in E_gen_i]

    # Choosing best model
    best_model_idx = E_gen_i.index(min(E_gen_i))
    best_model = ANNs[best_model_idx]

    # Log
    print(f'Best model in fold {num_fold}:')
    print(f'\tHidden layers:\t{num_of_hidden[best_model_idx]}')
    print(f'\tGen. error:\t{E_gen_i[best_model_idx]}')

    # Prepare data for training
    means, stds = get_norm_params(X_train_val, binary_columns)

    X_train_val = torch.Tensor((X_train_val - means) / stds)
    y_train_val = torch.Tensor(y[train_val_idx])
    X_test = torch.Tensor((X[test_idx,:] - means) / stds)
    y_test = torch.Tensor(y[test_idx]).type(dtype=torch.uint8).to(device)

    # Train and evaluate model
    net = train(model, X_train_val, y_train_val, epochs=100)
    y_sigmoid = net(X_test.to(device)).squeeze()
    y_est = (y_sigmoid > 0.5).type(dtype=torch.uint8)
    e = y_est != y_test
    E_gen_o.append((sum(e).type(torch.float) / len(y_val)).cpu().data.numpy())
    nums_of_hidden.append(num_of_hidden[best_model_idx])

    # Save true and estimated values for further analization
    np.save(path + f'fold_{num_fold}_y_true.npy', y_test.cpu().data.numpy())
    np.save(path + f'fold_{num_fold}_y_est.npy', y_est.cpu().data.numpy())

    num_fold += 1

# Save generalization error and model parameters for the outer folds
print("Generalization errors:")
print(E_gen_o)
print("Optimal numbers of hidden layers:")
print(nums_of_hidden)

np.save(path + 'E_gen_o.npy', np.asarray(E_gen_o))
np.save(path + 'nums_of_hidden.npy', np.asarray(nums_of_hidden))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import plotly.graph_objects as go

dataset = pd.read_csv("Life-Expectancy-Data.csv")

raw_data = dataset.values

cols = range(0, 21)
X = raw_data[:, cols]

attributeNames = np.asarray(dataset.columns[cols])
shortAttribNames = np.asarray(['C', 'R', 'Y', 'Id', 'U', 'Am', 'Ac', 'He', 'Me', 'B', 'P', 'D', 'Hi', 'G', 'Pp', 'T10', 'T5', 'S', 'Dd', 'Dg', 'L'])

# Extract class names to python list,
# then encode with integers (dict)
classLabels = X[:,0]
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Delete Country, Region and Economy status for PCA
del_attribs = [0, 1, 18, 19]
X_ = np.delete(X, del_attribs, axis=1).astype(float)
attributeNames_ = np.delete(attributeNames, del_attribs, axis=0)
shortAttribNames_ = np.delete(shortAttribNames, del_attribs, axis=0)
# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X_ - np.ones((N, 1)) * X_.mean(axis=0)
Y = Y * (1 / np.std(Y, 0))

# PCA by computing SVD of Y
U, S, Vt = svd(Y, full_matrices=False)
V = Vt.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()

# Create a heatmap of the first 8 principal components
V_90 = V[:,0:8]
plt.figure(figsize=(10, 6))
sns.heatmap(V_90, annot=True, fmt=".2f", cmap="viridis")
plt.xlabel("Principal Components")
plt.ylabel("Attributes")
plt.title("Principal directions of the components")
plt.yticks(ticks=np.arange(len(attributeNames_)) + 0.5, labels=attributeNames_, rotation=0, ha='right')
plt.xticks(ticks=np.arange(8) + 0.5, labels=range(1, 9))
plt.subplots_adjust(left=0.3)

# Plot attribute coefficients in principal component space
plt.figure()
for att in range(V.shape[1]):
    plt.arrow(0, 0, V[att, 0], V[att, 1])
    plt.text(V[att, 0], V[att, 1], shortAttribNames_[att])
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel("PC" + str(1))
plt.ylabel("PC" + str(2))
plt.grid()
# Add a unit circle
plt.plot(
    np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01))
)
plt.title("Attribute coefficients")
plt.axis("equal")

# Compute the projection onto the principal components
Z = U * S
# Plot projection
plt.figure()
plt.plot(Z[dataset['Economy_status_Developed'] == 0, 0], Z[dataset['Economy_status_Developed'] == 0, 1], ".", alpha=0.5)
plt.plot(Z[dataset['Economy_status_Developed'] == 1, 0], Z[dataset['Economy_status_Developed'] == 1, 1], ".", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection to PC1 and PC2")
plt.legend(['Developing', 'Developed'])
plt.axis("equal")

# Plot projection colored by country
plt.figure()
cmap = plt.get_cmap('hsv')
for i in range(len(y)):
    plt.plot(Z[i, 0], Z[i, 1], ".", alpha=0.5, color=cmap(y[i]))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection to PC1 and PC2")
plt.axis("equal")

plt.show()
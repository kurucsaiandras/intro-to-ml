#normal distributed? Bar plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

# Load the dataset
dataset = pd.read_csv("Life-Expectancy-Data.csv")

# Display basic information to verify loading
#dataset.info(), dataset.head()

# Continuous attributes list
continuous_attributes = [
    'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption', 
    'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita', 
    'Population_mln', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Schooling', 
    'Life_expectancy'
]

# Selecting the continuous data from the dataset
X = dataset[continuous_attributes]

n_attributes = len(continuous_attributes)
rows = n_attributes // 4 + (1 if n_attributes % 4 > 0 else 0)  

figure, axes = plt.subplots(nrows=rows, ncols=4, figsize=(18, 2*rows))

# Iterate through each attribute and plot on the subplot
for i, attribute in enumerate(continuous_attributes):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    ax.hist(X[attribute].dropna(), bins=20, color='blue', edgecolor='black')
    ax.set_title(attribute, fontsize=10)  # Smaller title size

# Hide
total_plots = rows * 4
for i in range(n_attributes, total_plots):
    axes.flat[i].set_visible(False)


plt.tight_layout(pad=2.0, h_pad=3.0)  # Adjust padding for tighter layout and more space between rows
plt.show()

#Based on the graph we can detect some outliers and applying masks for some attributes 
# Create an outlier mask based on these criteria
outlier_mask = (X['Incidents_HIV'] > 3) | \
                (X['GDP_per_capita'] > 20000) | \
                (X['Thinness_five_nine_years'] > 10) | \
                (X['Polio'] < 60) | \
                (X['Diphtheria'] < 40)

#Create a valid mask
valid_mask = ~outlier_mask

#Filter the dataset based on the valid mask
X_filtered = X[valid_mask]

#Visualize the results after applying the mask, focusing on the adjusted attributes
adjusted_attributes = ['Incidents_HIV', 'GDP_per_capita', 'Thinness_five_nine_years', 'Polio', 'Diphtheria']
figure, axes = plt.subplots(nrows=1, ncols=5, figsize=(18, 6))
for i, attribute in enumerate(adjusted_attributes):
    axes[i].hist(X_filtered[attribute].dropna(), bins=20, color='blue', edgecolor='black')
    axes[i].set_title(attribute, fontsize=10)

plt.tight_layout(pad=2.0)
plt.show()

#Print the original and new dataset sizes to see how many rows were considered outliers and removed
original_size = X.shape[0]
filtered_size = X_filtered.shape[0]
print(original_size, filtered_size)


# Plot histograms for the entire dataset after outlier removal
figure, axes = plt.subplots(nrows=rows, ncols=4, figsize=(18, 2*rows))

# Iterate through each attribute and plot on the subplot
for i, attribute in enumerate(continuous_attributes):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    ax.hist(X_filtered[attribute].dropna(), bins=20, color='blue', edgecolor='black')
    ax.set_title(attribute, fontsize=10)  # Smaller title size

#Hide
total_plots = rows * 4
for i in range(n_attributes, total_plots):
    axes.flat[i].set_visible(False)


plt.tight_layout(pad=2.0, h_pad=3.0)  
plt.show()

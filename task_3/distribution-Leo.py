#normal distributed? Bar plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Reload the dataset since the environment has been reset
data = pd.read_csv("Life_Expectancy_Data.csv")

# Define the continuous attributes (excluding 'Life expectancy' which is the target variable)
continuous_attributes = [
    'Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol',
    'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
    'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
    ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
    ' thinness 5-9 years', 'Income composition of resources', 'Schooling'
]

# Extract the continuous attributes
X_corrected = data[continuous_attributes]

# Define a function to plot histograms in batches
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20), constrained_layout=True)
fig.suptitle('Histograms', fontsize=25)

# Plot each histogram in the grid
for i, ax in enumerate(axes.flatten()):
    if i < len(continuous_attributes):
        # Plot the histogram for the current attribute
        sns.histplot(X_corrected[continuous_attributes[i]], kde=True, ax=ax)
        # Set title with the name of the current attribute
        ax.set_title(continuous_attributes[i], fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        # Hide the axes for subplots that do not have an attribute to display
        ax.axis('off')

# Define expected ranges for the specified attributes
# expected_ranges = {
#     'infant deaths': (2, 50),
#     'percentage expenditure': (0, 1000),
#     'Measles ': (2, 60),
#     'under-five deaths ': (2, 75),
#     ' HIV/AIDS': (0.5, 4),
#     'GDP': (0, 20000),
#     'Population': (0.02e8, 0.4e8)
# }
# Define expected ranges for the specified attributes
expected_ranges = {
    'infant deaths': (0, 25),
    'percentage expenditure': (0, 1000),
    'Measles ': (0, 40),
    'under-five deaths ': (0, 30),
    ' HIV/AIDS': (0, 3),
    'GDP': (0, 20000),
    'Population': (0, 0.2e8)
}

# Apply a mask to remove outliers based on the expected ranges
mask = pd.Series(True, index=X_corrected.index)  # Start with a mask that includes all rows
for attribute, (low, high) in expected_ranges.items():
    mask &= X_corrected[attribute].between(low, high)

# Apply the mask to the dataset
X_no_outliers = X_corrected[mask]

# Compare the shapes of the original and the cleaned datasets
original_shape = X_corrected.shape
cleaned_shape = X_no_outliers.shape

original_shape, cleaned_shape


print(original_shape, cleaned_shape) 
# (2938, 19) (55, 19)

# Set up the matplotlib figure with a layout of 5 rows and 4 columns for histograms of the cleaned dataset
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20), constrained_layout=True)
fig.suptitle('Histograms without Outliers', fontsize=25)

# Plot each histogram in the grid for the dataset with outliers removed
for i, ax in enumerate(axes.flatten()):
    if i < len(continuous_attributes):
        # Plot the histogram for the current attribute
        sns.histplot(X_no_outliers[continuous_attributes[i]], kde=True, ax=ax)
        # Set title with the name of the current attribute
        ax.set_title(continuous_attributes[i], fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        # Hide the axes for subplots that do not have an attribute to display
        ax.axis('off')

# Show the plot
plt.show()

# Show the plot
plt.show()
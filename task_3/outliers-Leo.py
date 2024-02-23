#outliers ? Box plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv("Life_Expectancy_Data.csv")

corrected_continuous_attributes = [
    'Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol',
    'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
    'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
    ' HIV/AIDS', 'GDP', ' thinness  1-19 years',
    ' thinness 5-9 years', 'Income composition of resources', 'Schooling'
]
X_corrected = dataset[corrected_continuous_attributes].copy()
#I removed Population from the data plus all the categorical attributes

# Set up the matplotlib figure with a layout of 5 rows and 4 columns
rows = 5
cols = 4
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20), constrained_layout=True)
fig.suptitle('Boxplots to detect outliers', fontsize=15)

# Plot each boxplot in the grid
for i, ax in enumerate(axes.flatten()):
    if i < len(corrected_continuous_attributes):
        # Plot the boxplot for the current attribute
        sns.boxplot(x=X_corrected[corrected_continuous_attributes[i]], ax=ax)
        # Set title with the name of the current attribute
        ax.set_title(corrected_continuous_attributes[i], fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        # Hide the axes for subplots that do not have an attribute to display
        ax.axis('off')
    
plt.show()

# OUTLIERS: 
# Adult Mortality: There are several data points that lie above the upper whisker, indicating potential outliers.

# Infant Deaths - Under-Five Deaths: Many data points are outside and above the upper whisker.

# Percentage Expenditure: This attribute has a significant number of data points beyond the upper whisker, suggesting outliers.

# Hepatitis B: There are outliers on both ends, below the lower whisker and above the upper whisker.

# Measles: A large number of data points are above the upper whisker, indicating many potential outliers.

# Polio: There are several outliers below the lower whisker.

# Diphtheria: There are outliers present below the lower whisker.

# HIV/AIDS: Numerous outliers can be seen above the upper whisker.

# GDP: There is a spread of data points far above the upper whisker, indicating the presence of outliers.

# Thinness 1-19 years - Thinness 5-9 years:: There are several outliers on the higher end beyond the upper whisker.



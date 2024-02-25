#outliers ? Box plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

dataset = pd.read_csv("Life-Expectancy-Data.csv")

#dataset.head()

continuous_attributes = [
                                     'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption', 
                                    'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita', 
                                    'Population_mln', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Schooling', 
                                    'Life_expectancy'
                                    ]



continuous_data = dataset[continuous_attributes]
X = dataset[continuous_attributes]
# Rename the columns to fit in the plot
X = X.rename(columns={
    'Thinness_ten_nineteen_years': 'Thinness_10/19_y/O',
    'Thinness_five_nine_years': 'Thinness_5/9_y/O'
})

# Box plot of the continuous attributes before standardization
plt.figure(figsize=(14, 7))
plt.title("Life Expectancy Data")
plt.boxplot(X.values, labels=X.columns, vert=False)
plt.xticks(rotation=45)
plt.xlabel('Attribute Values')
plt.ylabel('Attributes')
plt.show()


# Standardize the data using z-score normalization
X_standardized = zscore(X.values, ddof=1)

# Box plot of the continuous attributes after standardization
plt.figure(figsize=(14, 7))
plt.title("Life Expectancy Data: Boxplot After Standardization")
plt.boxplot(X_standardized, labels=X.columns, vert=False)
plt.xticks(rotation=45)
plt.xlabel('Z-score')
plt.ylabel('Attributes')
plt.show()

#Identifyng outliers

# Calculate z-scores for each attribute
count = zscore(X, ddof=1)

# Convert to a DataFrame for easier handling
count_df = pd.DataFrame(count, columns=X.columns)
# Identify outliers (absolute z-score > 3)
outliers = np.abs(count_df) > 3

# Count the number of outliers in each attribute
outlier_counts = outliers.sum().sort_values(ascending=False)

# Display the attributes with their outlier counts, excluding those with zero outliers
outlier_counts[outlier_counts > 0]

print(outlier_counts)

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



import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '')
from utils.read_dataset import read

df = pd.read_csv("Life-Expectancy-Data-Old.csv")
attributeNames = np.asarray(df.columns)

# Count 'nan' values for each property
nan_counts = df.isna().sum()

plt.figure(figsize=(10, 6))
nan_counts.plot(kind='bar', color='b')
plt.title('Count of NaN Values for Each Property')
plt.xlabel('Properties')
plt.ylabel('Count of NaN Values')
plt.xticks(range(len(attributeNames)), attributeNames, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Check if there are any missing years for any country
df = pd.read_csv("Life-Expectancy-Data.csv")
attributeNames = np.asarray(df.columns)

country_counts = df.iloc[:, 0].value_counts()
underrep_countries = country_counts[country_counts != 16]
print(f'Countries with missing years: {underrep_countries.size}')

# Check if one hot key encoding is valid
iscorrect = ((df['Economy_status_Developed'] == 1) & (df['Economy_status_Developing'] == 0) |
            (df['Economy_status_Developed'] == 0) & (df['Economy_status_Developing'] == 1)).all()
print(f'Is the one hot key encoding correct: {iscorrect}')

# Check if percent and per 1000 data is valid
per_1000_attribs = ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Measles', 'Incidents_HIV']
percent_attribs = ['Hepatitis_B', 'Polio', 'Diphtheria', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years']

for attr in per_1000_attribs:
    iscorrect = ((df[attr] >= 0) & (df[attr] <= 1000)).all()
    print(f'{attr} is valid: {iscorrect}')

for attr in percent_attribs:
    iscorrect = ((df[attr] >= 0) & (df[attr] <= 100)).all()
    print(f'{attr} is valid: {iscorrect}')
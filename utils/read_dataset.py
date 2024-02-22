import numpy as np
import pandas as pd

def read():
    dataset = pd.read_csv("Life_Expectancy_Data.csv")

    raw_data = dataset.values

    cols = range(0, 22)
    X = raw_data[:, cols]

    attributeNames = np.asarray(dataset.columns[cols])
    
    return attributeNames, X
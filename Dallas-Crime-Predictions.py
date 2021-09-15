""" 
Dallas Crime Predictions
Sept 14th 2021
Austin Caudill

"""

# Begin Timer for script
import time
start_time = time.time()

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

print("Load Successful.")
# ===============================

data = pd.read_csv("Police_Incidents.csv", parse_dates=['Date1 of Occurrence'])
# Columns 3, 41, 52, 57, 60, 74 have mixed data types. Examine.
# print(data.iloc[:, [3, 3]])

data.rename(columns={'Date1 of Occurrence': 'Date'}, inplace=True)
print('First date: ', str(data.Date.describe(datetime_is_numeric=True)['min']))
print('Last date: ', str(data.Date.describe(datetime_is_numeric=True)['max']))

# Split dataframe into train and test datasets.
train, test = train_test_split(data, test_size=0.33, random_state=42)

print("Success")
print("--- %s seconds ---" % (time.time() - start_time))
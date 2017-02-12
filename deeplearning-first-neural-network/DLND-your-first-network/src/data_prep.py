# This code is from Udacity Deep Learning Coursework
# Lesson: Intro to Neural Networks.Implementing Gradient descent

import numpy as np
import pandas as pd


admissions = pd.read_csv('../data/binary.csv')
print ("Input data:\n", admissions.head(5))
#make dummy variables
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)
print ("Dummy data: \n", data.head(5))

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

print ("Standardized data: \n", data.head(5))

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

print ("Complete test data: \n", test_data.head(5))

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

print ("Test data :\n", features_test.head(5))

print ("Test output :\n", targets_test.head(5))

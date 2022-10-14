#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:12:39 2022

@author: JosephNavelski
"""


# Loading Packages
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib

import scipy as sp
import scipy.linalg

import sklearn
# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt


# Define the seed so that results can be reproduced
seed = 123
rand_state = 123

import sklearn.datasets as dt
pd.set_option('display.max_columns', 20)

seed = 123
random.seed(123)

df = dt.load_iris()
print(df.feature_names)
print(df.data)

print(df.target_names)
print(df.target)

df = pd.DataFrame(data= np.c_[df['target'], df['data']],
                  columns= ['Species'] + df['feature_names'])

a = np.random.choice([0,1], size=(50,), p=[1/3, 2/3])
b = np.random.choice([0,2], size=(50,), p=[1/3, 2/3])
c = np.random.choice([0,3], size=(50,), p=[1/3, 2/3])

df1 = pd.DataFrame(data=np.r_[a,b,c], columns=['type'])
df = pd.concat([df,df1],axis=1)

'''
features = pd.get_dummies(df['type'])
features.columns = ['x0','x1','x2','x3']
df = pd.concat([df,features],axis=1)
df = df.drop('x0', axis=1)
'''

df.value_counts()
df.describe()

df.type = df.type.astype('str')
df.type = df.type.astype('category')

# Run a Random Forest
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df.iloc[:,1:6], df.iloc[:,0], test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Saving feature names for later use
feature_list = list(df.columns[1:6])


from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 6), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];




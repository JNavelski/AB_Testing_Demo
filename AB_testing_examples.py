#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:10:01 2022

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

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

seed = 123
random.seed(123)
print(random.random())

rand = np.random.RandomState(seed)    

dist_list = ['uniform','normal','exponential','lognormal','chisquare','beta']
param_list = ['-1,1','0,1','1','0,1','2','0.5,0.9']
colors_list = ['green','blue','yellow','cyan','magenta','pink']

fig,ax = plt.subplots(nrows=2, ncols=3,figsize=(12,7))
plt_ind_list = np.arange(6)+231

for dist, plt_ind, param, colors in zip(dist_list, plt_ind_list, param_list, colors_list):
    x = eval('rand.'+dist+'('+param+',5000)') 
    
    plt.subplot(plt_ind)
    plt.hist(x,bins=50,color=colors)
    plt.title(dist)

fig.subplots_adjust(hspace=0.4,wspace=.3) 
plt.suptitle('Sampling from Various Distributions',fontsize=20)
plt.show()

map_colors = plt.cm.get_cmap('RdYlBu')
fig,ax = plt.subplots(nrows=2, ncols=3,figsize=(16,7))
plt_ind_list = np.arange(6)+231

for noise,plt_ind in zip([0,0.1,1,10,100,1000],plt_ind_list): 
    x,y = dt.make_regression(n_samples=1000,
                             n_features=2,
                             noise=noise,
                             random_state=rand_state) 
    
    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(x[:,0],
                                  x[:,1],
                                  c=y,
                                  vmin=min(y),
                                  vmax=max(y),
                                  s=35,
                                  cmap=color_map)
    
    plt.title('noise: '+str(noise))
    plt.colorbar(my_scatter_plot)
    
fig.subplots_adjust(hspace=0.3,wspace=.3)
plt.suptitle('make_regression() With Different Noise Levels',fontsize=20)
plt.show()

fig = plt.figure(figsize=(18,5))

x,y = dt.make_friedman1(n_samples=1000,n_features=5,random_state=rand_state)
ax = fig.add_subplot(131, projection='3d')
my_scatter_plot = ax.scatter(x[:,0], x[:,1],x[:,2], c=y, cmap=color_map)
fig.colorbar(my_scatter_plot)
plt.title('make_friedman1')

x,y = dt.make_friedman2(n_samples=1000,random_state=rand_state)
ax = fig.add_subplot(132, projection='3d')
my_scatter_plot = ax.scatter(x[:,0], x[:,1],x[:,2], c=y, cmap=color_map)
fig.colorbar(my_scatter_plot)
plt.title('make_friedman2')

x,y = dt.make_friedman3(n_samples=1000,random_state=rand_state)
ax = fig.add_subplot(133, projection='3d')
my_scatter_plot = ax.scatter(x[:,0], x[:,1],x[:,2], c=y, cmap=color_map)
fig.colorbar(my_scatter_plot)
plt.suptitle('make_friedman?() for Non-Linear Data',fontsize=20)
plt.title('make_friedman3')

plt.show()

fig,ax = plt.subplots(nrows=1, ncols=3,figsize=(16,5))
plt_ind_list = np.arange(3)+131

for class_sep,plt_ind in zip([0.1,1,10],plt_ind_list):
    x,y = dt.make_classification(n_samples=1000,
                                 n_features=2,
                                 n_repeated=0,
                                 class_sep=class_sep,
                                 n_redundant=0,
                                 random_state=rand_state)
    
    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(x[:,0],
                                  x[:,1],
                                  c=y,
                                  vmin=min(y),
                                  vmax=max(y),
                                  s=35,
                                  cmap=color_map_discrete)
    plt.title('class_sep: '+str(class_sep))

fig.subplots_adjust(hspace=0.3,wspace=.3)
plt.suptitle('make_classification() With Different class_sep Values',fontsize=20)
plt.show()




import sklearn.datasets as dt
pd.set_option('display.max_columns', 20)

seed = 123
random.seed(123)

# Generatng Normal Data for AB Testing
x1 = np.random.normal(0, 1, 1000)
x2 = np.random.normal(-1, 1, 1000)
x3 = np.random.normal(3, 1, 1000)
x4 = np.random.normal(0, 3, 1000)

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=x1, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(2, 110, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

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

features = pd.get_dummies(df['type'])
features.columns = ['x0','x1','x2','x3']
df = pd.concat([df,features],axis=1)
df = df.drop('x0', axis=1)

df.value_counts()
df.describe()

# Run a Random Forest
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df.iloc[:,1:9], df.iloc[:,0], test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Saving feature names for later use
feature_list = list(df.columns[1:9])

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))


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









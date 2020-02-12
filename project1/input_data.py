# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat

disease_mat=loadmat('../project1/heart_disease.mat',squeeze_me=True);

X = disease_mat['X']
y = disease_mat['y']
M = disease_mat['M']
N = disease_mat['N']
C = disease_mat['C']
attributeNames = disease_mat['attributeNames']
classNames = disease_mat['classNames']

mean_X = X.mean(axis=0)
std_X = X.std(axis=0)
median_X = np.median(X,axis=0)

# Display results
print('Matrix:',X)
print('Mean:',mean_X)
print('Standard Deviation:',std_X)
print('Median:',median_X)






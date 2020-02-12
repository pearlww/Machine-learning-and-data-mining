# -*- coding: utf-8 -*-

from input_data import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend,scatter
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1



# Plot PCA of the data
f = figure()
title('PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# =============================================================================
# f = figure()
# title('PCA')
# for c in range(C):
#     # select indices belonging to class c:
#     class_mask = y==c
#     plot(Z[class_mask,i], Z[class_mask,i],'o', alpha=.5)
# legend(classNames)
# xlabel('PC{0}'.format(i+1))
# 
# f = figure()
# title('PCA')
# for c in range(C):
#     # select indices belonging to class c:
#     class_mask = y==c
#     scatter(Z[class_mask,i], Z[class_mask,i],Z[class_mask,k],'o', alpha=.5)
# legend(classNames)
# xlabel('PC{0}'.format(i+1))
# =============================================================================

# Output result to screen
show()

"""
Created on Tue Sep 24 16:21:11 2019

@author: ZoengGamming
"""


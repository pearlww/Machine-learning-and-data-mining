#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

import xlrd


# In[70]:


disease_mat=loadmat('../project1/heart_disease.mat',squeeze_me=True);

X = disease_mat['X']
y = disease_mat['y']
M = disease_mat['M']
N = disease_mat['N']
C = disease_mat['C']
attributeNames = disease_mat['attributeNames']
classNames = disease_mat['classNames']
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)
Y = Y*(1/np.std(Y,0))

mean_X = X.mean(axis=0)
std_X = X.std(axis=0)
median_X = np.median(X,axis=0)
print('Matrix:',X)
print('Mean:',mean_X)
print('Standard Deviation:',std_X)
print('Median:',median_X)
print('Standlized:',Y)


# In[4]:


attributeNames


# In[5]:


classNames


# In[21]:


plt.boxplot(X)
plt.xticks(range(1,10),attributeNames)
plt.title('SAHD data set - boxplot')
plt.show()


# In[77]:


fig = plt.figure(figsize=(10,8))
fig.subplots_adjust(hspace=0.3 ,wspace=0.5) 

for i in range(0,4):
    plt.subplot(2,4,i+1)
    plt.xlabel(attributeNames[i])
    plt.boxplot(X[:,i])    
for i in range(5,9):
    plt.subplot(2,4,i)
    plt.xlabel(attributeNames[i])
    plt.boxplot(X[:,i])    
plt.show()

df = pd.read_excel('../project1/heart_disease_data1.xlsx')
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

#remove the outliers
#mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
#df[mask] = np.nan


# In[78]:


df


# In[76]:


sns.set(style="ticks")


g = sns.pairplot(df,hue="heart disease")
g.map(plt.scatter)
g.add_legend()


# In[7]:


# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.8

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# In[18]:



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
f = plt.figure()
plt.title('PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()

ind = [0, 1, 2]
colors = ['blue', 'red']

f = plt.figure()
ax = f.add_subplot(111, projection='3d') #Here the mpl_toolkits is used
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(Z[class_mask,ind[0]], Z[class_mask,ind[1]], Z[class_mask,ind[2]], c=colors[c])

ax.view_init(30, 220)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(classNames)


# In[40]:


r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('SAHD: attribute standard deviations')

## Investigate how standardization affects PCA

# Try this *later* (for last), and explain the effect
#X_s = X.copy() # Make a to be "scaled" version of X
#X_s[:, 2] = 100*X_s[:, 2] # Scale/multiply attribute C with a factor 100
# Use X_s instead of X to in the script below to see the difference.
# Does it affect the two columns in the plot equally?


# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.8
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
plt.title('SAHD: Effect of standardization')
nrows=3
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.xlabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames)
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()


# In[19]:


U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# We saw that the first 6 components explaiend more than 80
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3,4,5]
Vk=V[:,:6]

Z = Y @ Vk

legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .15
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=.2)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('SAHD: PCA Component Coefficients')
plt.show()

print('The principal directions:')
print(Vk)
print('PC1:')
print(V[:,0])
print('Z')
print(Z)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





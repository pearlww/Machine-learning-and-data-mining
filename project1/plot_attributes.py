# -*- coding: utf-8 -*-
# Imports the numpy and xlrd package, then runs the ex2_1_1 code
#from input_data_excel import *
import pandas as pd
# (requires data structures from ex. 2.1.1)
import seaborn as sns
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

sns.set(style="ticks")
# Data attributes to be plotted
i = 5#3
j = 6

df = pd.read_excel('../project1/heart_disease_data1.xlsx')


g = sns.pairplot(df,hue=y)
g.map(plt.scatter);
g.add_legend();

#iris = sns.load_dataset("iris")
#g = sns.PairGrid(iris, hue="species")
#g.map_diag(plt.hist)
#g.map_offdiag(plt.scatter)
#g.add_legend();


##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
# =============================================================================
# plot(X[:, i], X[:, j], 'o')
# 
# # %%
# # Make another more fancy plot that includes legend, class labels, 
# # attribute names, and a title.
# f = figure()
# title('heart disease data')
# 
# for c in range(C):
#     # select indices belonging to class c:
#     class_mask = y==c
#     plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)
# 
# legend(classNames)
# xlabel(attributeNames[i])
# ylabel(attributeNames[j])
# =============================================================================

# Output result to screen
show()



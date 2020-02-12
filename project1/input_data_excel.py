# -*- coding: utf-8 -*-
# exercise 2.1.1
import numpy as np
import xlrd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd
import matplotlib.pyplot as plt
# Load xls sheet with data
doc = xlrd.open_workbook('../project1/heart_disease_data.xlsx').sheet_by_index(0)

# Extract attribute names (1st row, column 2 to 11)
attributeNames = doc.row_values(0, 1, 11)

# Extract class names to python list,
# then encode with integers (dict)
#chd_classLabels = doc.col_values(10, 1, 463)
#chd_classNames = sorted(set(chd_classLabels))
#classDict = dict(zip(classNames, range(5)))

#y -> y_chd_label

# Extract vector y, convert to NumPy array
y_chd_label = doc.col_values(10,1,463)
y_chd_classNames = sorted(set(y_chd_label))
#list -> numpy array
y_chd_label = np.asarray(y_chd_label)

famhlist_Labels = doc.col_values(5,1,463)
famhlist_Names = sorted(set(famhlist_Labels))
famhlist_Dict = dict(zip(famhlist_Names, range(2)))

# Preallocate memory, then extract excel data to matrix X
X = np.empty((462, 10))
for i, col_id in enumerate(range(1, 11)):
    if(i == 4):
        X[:, 4] = np.asarray([famhlist_Dict[value] for value in famhlist_Labels])
    else: 
        X[:, i] = np.asarray(doc.col_values(col_id, 1, 463))
    
# Compute values of N, M and C.
N = len(y_chd_label)
M = len(attributeNames)
C = len(y_chd_classNames)

print('Ran PCA data input')
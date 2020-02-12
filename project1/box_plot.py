# -*- coding: utf-8 -*-
# Exercise 4.2.3

from matplotlib.pyplot import boxplot, xticks, ylabel, title, show

# requires data from exercise 4.2.1
from input_data import *

boxplot(X)
xticks(range(1,10),attributeNames)
ylabel('scalar')
title('SAHD data set - boxplot')
show()

print('Ran Exercise 4.2.3')


#Prepare the dataset
#Load the siabetes dataset
#ref. https:scikit-learn.org/stable/datasets/index.html#diabets-dataset

from sklearn import datasets
diabetes = datasets.load_diabetes()

orig_X = diabetes.data
print(type(orig_X))
print(orig_X.shape)
print(orig_X[:,5])
print(diabetes.feature_names)

import numpy as np
X = diabetes.data[:,np.newaxis,2]
print(X.shape)
print(X[:5])
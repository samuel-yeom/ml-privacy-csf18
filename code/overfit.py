from __future__ import division, print_function

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut

'''Calculates the empirical root mean squared error. algorithm is a function
that takes in X and y and returns a model.'''
def get_empirical_error(algorithm, X, y):
    model = algorithm(X, y)
    mse = mean_squared_error(y, model.predict(X))
    return sqrt(mse)

'''Calculates the cross validation root mean squared error. If the dataset
is large, uses 10-fold cross validation. Otherwise, uses leave-one-out cross
validation. algorithm is a function that takes in X and y and returns a model.
'''
def get_cross_validation_error(algorithm, X, y):
    errors = []
    if len(y) > 10000:
        print('Data contains more than 10000 rows. Using 10-fold cross validation...')
        splitter = KFold(n_splits=10, shuffle=True, random_state=4294967295) #2**32 - 1
    else:
        print('Using leave-one-out cross validation...')
        splitter = LeaveOneOut()
    
    for train_indices, test_indices in splitter.split(X):
        train_X, train_y = X[train_indices], y[train_indices]
        test_X, test_y = X[test_indices], y[test_indices]
        
        model = algorithm(train_X, train_y)
        mse = mean_squared_error(test_y, model.predict(test_X))
        errors.append(mse)
    
    return sqrt(sum(errors) / len(errors))
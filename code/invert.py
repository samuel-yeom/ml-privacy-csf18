from __future__ import division

import numpy as np
from scipy.sparse import issparse, vstack

from inclusion import TRAIN #for reductions
from logprob import Logprob

BOTTOM = -1 #failed to make a guess

'''Takes in a 1-D numpy array counts and normalizes it so that the resulting
array corresponds to proportions instead. The output is 1-D numpy array of
Logprob.'''
def normalize(counts):
    total = counts.sum()
    freqs = [Logprob(val / total) for val in counts]
    return np.array(freqs)

'''Returns a 1-D numpy array containing Logprob multipliers to be used as
weights during attribute inference. Computes the z-scores corresponding to the
1-D numpy array errors, with rmse used as the standard deviation. The returned
multipliers represent e^(-z^2 / 2), which is the probability density function
of the standard normal distribution at z, without the normalization factor.'''
def get_multipliers(errors, rmse):
    zs = errors / rmse
    return np.array([Logprob(-z**2 / 2, True) for z in zs])

'''Returns a 1-D numpy array representing the result of attribute inference on
the samples described by X and y. The output array contains integers in
range(num_variants), where num_variants is the number of possible
values of the target attribute.

If the target attribute is binary, num_variants == 2, len(target_cols) == 1,
and target_cols contains the index of the column in X that contains the target
attribute. Then, the j-th element of the output array is 0 if the value of the
target attribute for the j-th row in X and y has been guessed to be 0, and
vice versa.

If the target attribute is one-hot encoded, num_variants == len(target_cols),
and target_cols contains the indices of the columns in X that correspond to the
target attribute. Then, the j-th element of the output array is i if the value
of the target attribute for the data point represented by the j-th row of X and
y has been guessed to be the value that corresponds to the column
X[:, target_cols[i]].

For example, suppose target_cols = np.array([13, 14, 15]) and X[:,13], X[:,14],
and X[:, 15] correspond to 'vkorc1=CC', 'vkorc1=CT', and 'vkorc1=TT',
respectively. If the j-th element of the output is 0, it means that the target
attribute for the j-th row in X and y was guessed to be the column in X that
corresponds to target_cols[0] == 13, which is 'vkorc1=CC'.

model: a model that was trained on the training data, where model.predict(X)
       returns the prediction y
dist: a 1-D numpy array of Logprob; dist[i] represents the a priori probability
      of i (if target is a binary attribute) or the value that corresponds to
      target_cols[i] (if one-hot encoded)
X: a 2-D numpy matrix that corresponds to the input of model
y: a 1-D numpy array that corresponds to the output of model
target_cols: 1-D numpy array of indices of columns in X that correspond to the
             target attribute
rmse: root mean squared error used to compute the multiplier'''
def sklearn_invert(model, dist, X, y, target_cols, rmse):
    assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions
    
    if issparse(X): #deal with sparse matrices correctly
        stack = vstack
    else:
        stack = np.stack
    
    guesses = []
    
    assert len(target_cols) > 0
    one_hot = (len(target_cols) > 1) #whether the target attribute was one-hot encoded (binary otherwise)
    num_variants = len(target_cols) if one_hot else 2 #number of possible values of the target
    
    for i in range(X.shape[0]): #iterate over the rows of X and y
        row_X = stack([X[i] for _ in range(num_variants)]) #create copies of X[i]
        if one_hot:
            row_X[:, target_cols] = np.eye(num_variants) #fill in with all possible values of target (one-hot encoded)
        else: #fill in with all possible values of target (binary)
            row_X[0, target_cols] = 0
            row_X[1, target_cols] = 1
        row_y = np.repeat(y[i], num_variants)
        
        errors = row_y - model.predict(row_X)
        likelihood_scores = dist * get_multipliers(errors, rmse)
        guess = np.where(likelihood_scores == max(likelihood_scores))[0][0] #an integer in range(num_variants)
        guesses.append(guess)
    
    return np.array(guesses)

'''Returns a 1-D numpy array representing the result of attribute inference on
the samples described by X and y, with access to the membership oracle. The
output array contains integers in range(num_variants) + [BOTTOM], where
num_variants is the number of possible values of the target attribute.

If the target attribute is binary, num_variants == 2, len(target_cols) == 1,
and target_cols contains the index of the column in X that contains the target
attribute. Then, the j-th element of the output array is 0 if the value of the
target attribute for the j-th row in X and y has been guessed to be 0, and
vice versa.

If the target attribute is one-hot encoded, num_variants == len(target_cols),
and target_cols contains the indices of the columns in X that correspond to the
target attribute. Then, the j-th element of the output array is i if the value
of the target attribute for the data point represented by the j-th row of X and
y has been guessed to be the value that corresponds to the column
X[:, target_cols[i]].

For example, suppose target_cols = np.array([13, 14, 15]) and X[:,13], X[:,14],
and X[:, 15] correspond to 'vkorc1=CC', 'vkorc1=CT', and 'vkorc1=TT',
respectively. If the j-th element of the output is 0, it means that the target
attribute for the j-th row in X and y was guessed to be the column in X that
corresponds to target_cols[0] == 13, which is 'vkorc1=CC'.

oracle: membership oracle; a function that takes in a 2-D numpy matrix X and
        a 1-D numpy array y such that X.shape[0] == y.shape[0]; outputs
        a 1-D numpy array whose size is X.shape[0]; output[i] is TRAIN if the
        membership oracle predicts that the data point represented by X[i] and
        y[i] is in the training set and is TEST otherwise
dist: a 1-D numpy array of Logprob; dist[i] represents the a priori probability
      of i (if target is a binary attribute) or the value that corresponds to
      target_cols[i] (if one-hot encoded)
X: a 2-D numpy matrix that corresponds to the input of model
y: a 1-D numpy array that corresponds to the output of model
target_cols: 1-D numpy array of indices of columns in X that correspond to the
             target attribute
only_one: if True, checks only the value of the target attribute with the
          highest a priori probabiltiy; otherwise, checks all possible values
          of the target attribute'''
def sklearn_reduce(oracle, dist, X, y, target_cols, only_one=False):
    assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions
    
    if issparse(X): #deal with sparse matrices correctly
        stack = vstack
    else:
        stack = np.stack
    
    assert len(target_cols) > 0
    one_hot = (len(target_cols) > 1) #whether the target attribute was one-hot encoded (binary otherwise)
    num_variants = len(target_cols) if one_hot else 2 #number of possible values of the target
    
    if only_one: #check only the argmax target value
        argmax = dist.argmax() #index of target value with highest a priori probability
        new_X = X.copy()
        if one_hot:
            new_X[:, target_cols] = np.eye(num_variants)[argmax] #one-hot encoding of the argmax value
        else:
            new_X[:, target_cols] = argmax #argmax value in [0, 1]
        inc_res = oracle(new_X, y)
        guesses = np.array([argmax if res == TRAIN else BOTTOM for res in inc_res])
        return guesses
            
    else: #check all possible values of the target attribute
        guesses = []
        
        argsort = dist.argsort() #indices of target values in order of highest a priori probability
        for i in range(X.shape[0]): #iterate over the rows of X and y
            row_X = stack([X[i] for _ in range(num_variants)]) #create copies of X[i]
            if one_hot:
                row_X[:, target_cols] = np.eye(num_variants) #fill in with all possible values of target (one-hot encoded)
            else: #fill in with all possible values of target (binary)
                row_X[0, target_cols] = 0
                row_X[1, target_cols] = 1
            row_y = np.repeat(y[i], num_variants)
            
            inc_res = oracle(row_X, row_y)
            for i in argsort:
                if inc_res[i] == TRAIN:
                    guess = i
                    break
            else:
                guess = BOTTOM
            
            guesses.append(guess)
        
        return np.array(guesses)
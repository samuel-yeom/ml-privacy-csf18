from __future__ import division, print_function

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import scale
from scipy import sparse

import warnings
import argparse
import csv
import sys
import os

import tree, linreg
import overfit
import inclusion
import invert

'''Runs membership inference on all rows in data represented by X and y, and returns
the proportion guessed to be in the training set.'''
def sklearn_do_inclusion(model, X, y, r_emp, r_cv):
    assert X.shape[0] == y.shape[0]
    num_rows = X.shape[0]
    
    results = inclusion.sklearn_inclusion(model, X, y, r_emp, r_cv)
    num_train = np.count_nonzero(results == inclusion.TRAIN)
    
    return num_train / num_rows

'''Runs attribute inference on all rows in data represented by X and y, and
returns the proportion guessed correctly (matches the corresponding row in t).

If target attribute is binary, t contains integers in [0, 1], and target_cols
contains the index of the column in X that contains the target attribute.

If target attribute is a one-hot encoded, t contains integers in
range(len(target_cols)), where target_cols is a 1-D numpy array of indices of
columns in X that correspond to the target attribute. If the j-th element of t
is i, it means that the value of the target attribute for the data point
represented by the j-th row of X and y corresponds to the column
X[:, target_cols[i]].

For example, suppose target_cols = np.array([13, 14, 15]) and X[:,13], X[:,14],
and X[:, 15] correspond to 'vkorc1=CC', 'vkorc1=CT', and 'vkorc1=TT',
respectively. If the j-th element of the output is 0, it means that the target
attribute of the j-th row in X and y is the column in X that corresponds to
target_cols[0] == 13, which is 'vkorc1=CC'.
'''
def sklearn_do_inversion(model, dist, X, y, t, target_cols, rmse):
    assert X.shape[0] == y.shape[0] == t.shape[0]
    num_rows = X.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
        results = invert.sklearn_invert(model, dist, X, y, target_cols, rmse)
    num_correct = np.count_nonzero(results == t)
    
    return num_correct / num_rows

def sklearn_do_reduction(oracle, dist, X, y, t, target_cols):
    assert X.shape[0] == y.shape[0] == t.shape[0]
    num_rows = X.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
        results = invert.sklearn_reduce(oracle, dist, X, y, target_cols)
    num_correct = np.count_nonzero(results == t)
    
    return num_correct / num_rows

'''Returns the values of the target attribute for the data points in X.

X: a 2-D numpy matrix
target_str: name of a discrete attribute whose values we want to extract
featnames: names of the features, in the order that matches the order of the
           columns of X

t: a 1-D numpy array; for more details, read the documentation for
   sklearn_do_inversion
target_cols: a 1-D numpy array that shows which columns of X correspond to
             target_str
dist: a 1-D numpy array of Logprob; dist[i] represents the a priori probability
      of i (if target is a binary attribute) or the value that corresponds to
      target_cols[i] (if one-hot encoded)'''
def extract_target(X, target_str, featnames):
    target_cols = np.array([i for i, featname in enumerate(featnames) if featname.startswith(target_str)])
    
    if len(target_cols) >= 2: #one-hot encoded
        rows, cols, vals = sparse.find(X[:, target_cols]) #rows, columns, and values of nonzero values in X that have to do with target_str
        assert np.array_equal(np.sort(rows), np.arange(X.shape[0])) #check that one-hot encoding was used
        assert np.array_equal(vals, np.ones(X.shape[0])) #check that one-hot encoding was used
        t = cols[np.argsort(rows)] #read the documentation in sklearn_do_inversion for more details about the format of t
        dist = invert.normalize(np.squeeze(np.array(X[:, target_cols].sum(axis=0))))
    
    else:
        assert len(target_cols) == 1 #check that only one column corresponds to target_str
        
        t = X[:, target_cols]
        if sparse.issparse(t):
            t = t.todense()
        t = np.squeeze(np.array(t))
        
        vals, counts = np.unique(t, return_counts=True)
        assert np.array_equal(np.unique(t), np.arange(2)) #check that the target attribute is binary
        dist = invert.normalize(counts)
    
    return t, target_cols, dist

'''Reads outfile and figures out the last rseed recorded in the file. Returns
the first unrecorded rseed. If this rseed would be greater than or equal to
iters, returns None instead.

attack represents which attack is being run. It must be one of 'inc'
(membership inference), 'inv' (attribute inference), or 'red' (reduction).

outfile must be in CSV format. The first field of each line must be rseed,
and the lines must be sorted in ascending order of rseed.'''
def prepare_outfile(outfile, attack, iters):
    assert attack in ['inc', 'inv', 'red']
    
    print('Running {} iterations with different random seeds...'.format(iters))
    
    try:
        with open(outfile) as f:
            for line in f:
                pass
        last_line = line
        last_used_rseed = int(last_line.split(',')[0])
        start_rseed = last_used_rseed + 1 #first unused random seed
        
        if start_rseed >= iters:
            print('Full data already exists.')
            return None
        else:
            print('Incomplete data found. Resuming with rseed={}...'.format(start_rseed))
            return start_rseed
    
    except IOError: #file does not exist
        with open(outfile, 'w') as f:
            if attack == 'inc':
                f.write('rseed,r_emp,r_cv,train_TRAIN,test_TRAIN\n')
            else:
                f.write('rseed,r_emp,r_cv,train_correct,test_correct\n')
        return 0
    
    except ValueError: #last_used_seed is not an integer (only the header line exists)
        return 0

'''Runs membership inference n times, where n is the value of the input
variable iters, and writes the result to the file whose path is specified by
outfile. Each iteration uses a different random seed for splitting the data
into training and test sets.

algorithm: a function that uses the training data to train a model
X: a 2-D numpy matrix containing the inputs to the model
y: a 1-D numpy array containing the actual values of the attribute predicted
   by the model
one_error: if True, assumes that the adversary does not know r_cv
r_emp: empirical root mean squared error
r_cv: cross validation root mean squared error
iters: number of times to run attribute inference
outfile: path of the file to which the results will be written

The result file will be in CSV format and will contain the following for each
iteration of attribute inference.

rseed: random seed used to split the data into training and test sets
r_emp: empirical root mean squared error
r_cv: cross validation root mean squared error
train_TRAIN: proportion of the training set data that was guessed to be in the
             training set
test_TRAIN: proportion of the test set data that was guessed to be in the
            test set
'''
def iterate_inclusion_and_write(algorithm, X, y, one_error, r_emp, r_cv, iters, outfile):
    start_rseed = prepare_outfile(outfile, 'inc', iters)
    if start_rseed is None: #full data already exists in outfile
        return
    
    adv_r_cv = None if one_error else r_cv
    
    with open(outfile, 'a') as f:
        for rseed in range(start_rseed, iters): #repeat with different random seeds
            train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=rseed)
            
            model = algorithm(train_X, train_y)
            train_TRAIN = sklearn_do_inclusion(model, train_X, train_y, r_emp, adv_r_cv)
            test_TRAIN = sklearn_do_inclusion(model, test_X, test_y, r_emp, adv_r_cv)
            
            output = '{},{:.4f},{:.4f},{:.6f},{:.6f}\n'.format(rseed, r_emp, r_cv, train_TRAIN, test_TRAIN)
            f.write(output)
    
    print('Result written to {}'.format(outfile))

'''Runs attribute inference n times, where n is the value of the input variable
iters, and writes the result to the file whose path is specified by outfile.
Each iteration uses a different random seed for splitting the data into
training and test sets.

algorithm: a function that takes training data as input and returns a model
X: a 2-D numpy matrix containing the inputs to the model
y: a 1-D numpy array containing the actual values of the attribute predicted
   by the model
target_str: name of a discrete attribute whose values the adversary wants to
            guess
featnames: names of the attributes, in the order that matches the order of the
           columns of X
one_error: if True, assumes that the adversary does not know r_cv
r_emp: empirical root mean squared error
r_cv: cross validation root mean squared error
iters: number of times to run attribute inference
outfile: path of the file to which the results will be written

The result file will be in CSV format and will contain the following for each
iteration of attribute inference.

rseed: random seed used to split the data into training and test sets
r_emp: empirical root mean squared error
r_cv: cross validation root mean squared error
train_correct: proportion of the training set data whose target attribute was
               guessed correctly
test_correct: proportion of the test set data whose target attribute was
              guessed correctly
'''
def iterate_inversion_and_write(algorithm, X, y, target_str, featnames, one_error, r_emp, r_cv, iters, outfile):
    start_rseed = prepare_outfile(outfile, 'inv', iters)
    if start_rseed is None: #full data already exists in outfile
        return
    
    t, target_cols, dist = extract_target(X, target_str, featnames)
    
    adv_r_cv = r_emp if one_error else r_cv
    
    with open(outfile, 'a') as f:
        for rseed in range(start_rseed, iters): #repeat with different random seeds
            train_X, test_X, train_y, test_y, train_t, test_t = train_test_split(X, y, t, random_state=rseed)
            model = algorithm(train_X, train_y)
            
            #Calculate attribute inference accuracy
            train_correct = sklearn_do_inversion(model, dist, train_X, train_y, train_t, target_cols, r_emp)
            test_correct = sklearn_do_inversion(model, dist, test_X, test_y, test_t, target_cols, adv_r_cv)
            
            output = '{},{:.4f},{:.4f},{:.6f},{:.6f}\n'.format(rseed, r_emp, r_cv, train_correct, test_correct)
            f.write(output)
    
    print('Result written to {}'.format(outfile))

def iterate_reduction_and_write(algorithm, X, y, target_str, featnames, one_error, r_emp, r_cv, iters, outfile):
    start_rseed = prepare_outfile(outfile, 'red', iters)
    if start_rseed is None:
        return
    
    t, target_cols, dist = extract_target(X, target_str, featnames)
    
    adv_r_cv = None if one_error else r_cv
    
    with open(outfile, 'a') as f:
        for rseed in range(start_rseed, iters): #repeat with different random seeds
            train_X, test_X, train_y, test_y, train_t, test_t = train_test_split(X, y, t, random_state=rseed)
            model = algorithm(train_X, train_y)
            oracle = lambda X, y: inclusion.sklearn_inclusion(model, X, y, r_emp, adv_r_cv)
            
            train_correct = sklearn_do_reduction(oracle, dist, train_X, train_y, train_t, target_cols)
            test_correct = sklearn_do_reduction(oracle, dist, test_X, test_y, test_t, target_cols)
            
            output = '{},{:.4f},{:.4f},{:.6f},{:.6f}\n'.format(rseed, r_emp, r_cv, train_correct, test_correct)
            f.write(output)
    
    print('Result written to {}'.format(outfile))

'''The below functions load data from file and return a tuple
(X, y, featnames), where X is a 2-D numpy matrix, y is a 1-D numpy array,
and featnames is a list such that X.shape[0] == y.shape[0]
and X.shape[1] == len(featnames).'''
def load_eyedata(data_folder):
    datafile = '{}/eyedata.csv'.format(data_folder)
    data = np.loadtxt(datafile, skiprows=1, delimiter=',')
    data = scale(data)
    X, y = data[:, :-1], data[:, -1]
    featnames = np.array(list(map(lambda i: '{:03}'.format(i), range(X.shape[1])))) #feature names are strings 000, 001, ...
    return X, y, featnames

def load_iwpc(data_folder):
    datafile = '{}/iwpc-scaled.csv'.format(data_folder)
    col_types = {'race': str,
                 'age': float,
                 'height': float,
                 'weight': float,
                 'amiodarone': int,
                 'decr': int,
                 'cyp2c9': str,
                 'vkorc1': str,
                 'dose': float}
    X, y = [], []
    with open(datafile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for col_name in reader.fieldnames:
                col_type = col_types[col_name]
                row[col_name] = col_type(row[col_name]) #cast to correct type
                if col_name == 'dose':
                    y.append(row[col_name])
                    del row[col_name]
            X.append(row)
    
    dv = DictVectorizer()
    X = dv.fit_transform(X)
    y = np.array(y)
    featnames = np.array(dv.get_feature_names())
    return X, y, featnames

def load_netflix(data_folder):
    datafile = '{}/netflix.npz'.format(data_folder)
    rated_movie = 11092 #ID of movie whose rating the model will predict
    with np.load(datafile) as f:
        #binary sparse matrix; watched[i] corresponds to the user with ID i; watched[:,j] corresponds to the movie with ID j
        watched = sparse.csr_matrix((f['data'], f['indices'], f['indptr']))
    
    ratingfile = '{}/netflix-{:07}.txt'.format(data_folder, rated_movie)
    user_ratings = np.loadtxt(ratingfile, skiprows=1, delimiter=',', usecols=[0, 1], dtype=int)
    users = user_ratings[:,0]
    ratings = user_ratings[:,1]
    watched = watched[users] #only take the rows that correspond to users who have rated the movie
    
    featnames = np.delete(np.arange(watched.shape[1]), [0, rated_movie]) #remove the rated movie from the input features
    X = watched[:,featnames] #only take the columns that correspond to input features (movies)
    y = scale(ratings.astype(float))
    featnames = np.vectorize(lambda i: '{:07}'.format(i))(featnames) #feature names are strings 0000001, 0000002, ...
    return X, y, featnames

'''If the directories for the path outfile does not exist, create them.'''
def conditional_makedirs(outfile):
    dirname = os.path.dirname(outfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if __name__ == '__main__':
    data_folder = '../data'
    results_folder = '../results-sklearn'
    
    parser = argparse.ArgumentParser(description='By default, computes the empirical and cross validation errors without doing anything else. Use --inc for membership inference, --inv for attribute inference, and --red for reduction.')
    parser.add_argument('data', choices=['eyedata', 'iwpc', 'netflix'], help='Specify the data to use')
    parser.add_argument('model', choices=['tree', 'linreg'], help='Specify the machine learning algorithm')
    parser.add_argument('param', type=float, help='Depth of the decision tree, or lambda (alpha) in Ridge linear regression')
    parser.add_argument('--target', help='Name of the target attribute; necessary for attribute inference and reduction; ignored otherwise')
    parser.add_argument('--one-error', action='store_true', help='Assume that the adversary does not know r_cv')
    parser.add_argument('--iters', type=int, metavar='n', default=100, help='Number of iterations to use; each iteration uses a different random seed for splitting the data into training and test sets')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--inc', type=float, nargs=2, metavar=('r_emp', 'r_cv'), help='Perform membership inference using r_emp and r_cv as training set and test set standard errors, respectively')
    group.add_argument('--inv', type=float, nargs=2, metavar=('r_emp', 'r_cv'), help='Perform attribute inference using r_emp and r_cv as training set and test set standard errors, respectively')
    group.add_argument('--red', type=float, nargs=2, metavar=('r_emp', 'r_cv'), help='Perform the reduction from membership to attribute using r_emp and r_cv as the errors for the membership oracle')
    args = parser.parse_args()
    
    dataset = args.data
    model_type = args.model
    param = args.param #parameter for training model
    target_str = args.target #IMPORTANT: target_str must not be a prefix of any other feature name
    one_error = args.one_error
    iters = args.iters
    
    #argparse ensures that at most one of the following will be not None
    inc = args.inc #if not None, do membership inference
    inv = args.inv #if not None, do attribute inference
    red = args.red #if not None, do membership-to-attribute reduction
    
    if dataset == 'eyedata':
        X, y, featnames = load_eyedata(data_folder)
    if dataset == 'iwpc':
        X, y, featnames = load_iwpc(data_folder)
    if dataset == 'netflix':
        X, y, featnames = load_netflix(data_folder)
    
    if model_type == 'tree':
        param = int(param)
        max_depth = param
        algorithm = lambda X, y: tree.sklearn_train_tree(X, y, max_depth)
    if model_type == 'linreg':
        alpha = param
        algorithm = lambda X, y: linreg.sklearn_train_linreg(X, y, alpha)
    
    #argparse ensures that at most one of {inc, inv, red} will be not None
    if inc is not None or inv is not None or red is not None:
        test_error = 'unknown-test-error' if one_error else 'known-test-error'
        #Membership inference
        if inc is not None:
            r_emp = inc[0]
            r_cv = inc[1]
            outfile = '{}/{}/membership/{}/{}-{}.csv'.format(results_folder, dataset, test_error, model_type, param)
            conditional_makedirs(outfile)
            iterate_inclusion_and_write(algorithm, X, y, one_error, r_emp, r_cv, iters, outfile)
        
        else:
            if target_str is None:
                print('Attribute inference and reduction require that the target attribute be passed through the --target option')
                sys.exit(2)
            
            #Attribute inference
            if inv is not None:
                r_emp = inv[0]
                r_cv = inv[1]
                outfile = '{}/{}/attribute/{}/{}/{}-{}.csv'.format(results_folder, dataset, target_str, test_error, model_type, param)
                conditional_makedirs(outfile)
                iterate_inversion_and_write(algorithm, X, y, target_str, featnames, one_error, r_emp, r_cv, iters, outfile)
    
            #Membership-to-attribute reduction
            else:
                r_emp = red[0]
                r_cv = red[1]
                outfile = '{}/{}/reduction/{}/{}/{}-{}.csv'.format(results_folder, dataset, target_str, test_error, model_type, param)
                conditional_makedirs(outfile)
                iterate_reduction_and_write(algorithm, X, y, target_str, featnames, one_error, r_emp, r_cv, iters, outfile)
    
    #Compute errors
    else:
        r_emp = overfit.get_empirical_error(algorithm, X, y) #root mean squared
        r_cv = overfit.get_cross_validation_error(algorithm, X, y) #root mean squared
        outfile = '{}/{}/{}-errors.txt'.format(results_folder, dataset, model_type)
        conditional_makedirs(outfile)
        with open(outfile, 'a') as f:
            f.write('{} {:.4f} {:.4f}\n'.format(param, r_emp, r_cv))
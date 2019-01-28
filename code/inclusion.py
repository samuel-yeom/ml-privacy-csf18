import numpy as np

TRAIN = 100
TEST = 200

'''Returns the probability density function of the Gaussian distribution
N(0, sd^2) at x, without the 1/sqrt(2*pi) normalization factor. x can be either
a scalar or a numpy array.'''
def gaussian_pdf(sd, x):
    if sd <= 0:
        raise ValueError('standard deviation must be positive but is {}'.format(sd))
    else: #sd > 0
        return np.e ** (-0.5*(x/sd)**2) / sd

'''Returns a 1-D numpy array. Each element in the array is TRAIN if the
membership adversary guesses that the corresponding data point was used to
train model. Otherwise, the element is TEST.

errors: a 1-D numpy array where each element contains the error of the model
        on a particular data point
r_emp: empirical root mean squared error
r_cv: cross validation root mean squared error

If r_cv is None, the decision boundary is at abs(error) = r_emp. Otherwise,
we compare the values of the probability density functions of normal
distributions that correspond to r_emp and r_cv.'''
def sklearn_decide(errors, r_emp, r_cv):
    if r_cv is None:
        return np.where(abs(errors) < r_emp, TRAIN, TEST)
    else:
        return np.where(gaussian_pdf(r_emp, errors) > gaussian_pdf(r_cv, errors), TRAIN, TEST)

'''Returns a 1-D numpy array. Each element in the array is TRAIN if the
membership adversary guesses that the corresponding data point in the arrays
X and y was used to train model. Otherwise, the element is TEST.

model: a model that was trained on the training data, where model.predict(X)
       returns a numpy array with the predictions
X: a 2-D numpy array where each row contains the values of the model's
   input variables for a data point
y: a 1-D numpy array where each element contains the value of the model's
   output variables for a data point
r_emp: empirical root mean squared error of the model
r_cv: cross validation root mean squared error of the model; if None, we assume
       that r_emp == r_cv'''
def sklearn_inclusion(model, X, y, r_emp, r_cv):
    pred_vals = model.predict(X)
    actual_vals = y
    errors = actual_vals - pred_vals
    return sklearn_decide(errors, r_emp, r_cv)
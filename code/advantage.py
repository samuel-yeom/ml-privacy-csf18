from __future__ import division, print_function
import numpy as np
from math import erf

'''Returns the probability density function of the normal distribution with
mean mu and standard deviation sigma.'''
def gaussian(mu, sigma):
    return lambda x: np.exp(-x**2/(2*sigma**2)) / ((2*np.pi)**0.5 * sigma)

'''The following functions compute advantage of the membership adversary if
the test standard error is sd times the training standard error.'''
def theo_adv(sd): #known test error
    temp = sd * np.sqrt(np.log(sd) / (sd**2 - 1))
    return erf(temp) - erf(temp / sd)

def theo_adv2(sd): #unknown test error
    return erf(0.5**0.5) - erf(0.5**0.5 / sd)
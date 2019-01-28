import math

'''Immutable class that represents probability as the natural log of the
probability. Supports comparison, addition, multiplication, and division.
Use value() to recover the actual probability. As usual, comparing floats for
equality may not behave as expected.'''
class Logprob(object):
    ninf = float('-inf')
    tolerance = 1e-15 #floating point arithmetic error tolerance
    
    '''log represents whether prob is the natural log of the intended
    probability. If log is False, prob is considered the actual intended
    probability.'''
    def __init__(self, prob, log=False):
        if log: #prob is log of probability
            logprob = float(prob)
            if logprob > 0:
                raise ValueError('log of probability must be nonpositive')
            self.logprob = logprob
        else: #prob is actual probability
            if prob < 0 or prob > 1:
                raise ValueError('probability must be in the interval [0, 1]')
            if prob == 0:
                self.logprob = Logprob.ninf
            else:
                self.logprob = math.log(prob)
    
    def value(self):
        return math.exp(self.logprob)
    
    def __hash__(self):
        return hash(self.logprob)
    
    def __eq__(self, other):
        return self.logprob == other.logprob
    
    def __ne__(self, other):
        return self.logprob != other.logprob
    
    def __lt__(self, other):
        return self.logprob < other.logprob
    
    def __le__(self,other):
        return self.logprob <= other.logprob
    
    def __gt__(self, other):
        return self.logprob > other.logprob
    
    def __ge__(self, other):
        return self.logprob >= other.logprob
    
    def __add__(self, other):
        if self < other:
            return other + self
        else:
            if self.logprob == Logprob.ninf: #self.value == other.value == 0
                return Logprob(0)
            else: #formula on Wikipedia page for "Log probability" (as of 9 Oct 2016)
                logprob = self.logprob
                logprob += math.log1p(math.exp(other.logprob - self.logprob))
                if logprob > 0:
                    if logprob <= Logprob.tolerance: #float arithmetic rounding error
                        logprob = 0
                    else:
                        raise ValueError('Logprob addition resulted in probability {:.20f}'.format(math.exp(logprob)))
                return Logprob(logprob, True)
    
    def __mul__(self, other):
        return Logprob(self.logprob + other.logprob, True)
    
    def __div__(self, other):
        if other.logprob == Logprob.ninf:
            raise ZeroDivisionError('Logprob division by zero')
        logprob = self.logprob - other.logprob
        if logprob > 0:
            if logprob <= Logprob.tolerance: #float arithmetic rounding error
                logprob = 0
            else:
                raise ValueError('Logprob division resulted in probability {:.20f}'.format(math.exp(logprob)))
        return Logprob(logprob, True)
    
    def __repr__(self):
        return str(self.value())
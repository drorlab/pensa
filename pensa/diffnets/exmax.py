"""
Copyright 2015 by Washington University in Saint Louis. Authored by S. Joshua Swamidass. 
A license to use for strictly non-commerical use is granted.
Derivative work is not permited without prior written authorization.
All other rights reserved.
"""

from scipy import inf, asarray, array, rand, zeros, prod, where, allclose
from scipy.stats import pearsonr

def distribution_of_sum(P, ignore_idx=set()):
    """
    Given a set of binomial random variables parameterized by a vector P.
    Ignoring variables in ignore_idx...
    What is the distribution of their sum?
    
    Output is the discreet distribution D, where the probability of a
    specific sum s is D[s].
    O(N^2) time in length of P
    
    For example, using this P...
    >>> P = [0.5, 0.25, 0.5]

    >>> distribution_of_sum(P)
    array([ 0.1875,  0.4375,  0.3125,  0.0625])

    Ignoring the 2nd (1 in zero indexing) variable, we have...
    >>> distribution_of_sum(P, [1]) 
    array([ 0.25,  0.5 ,  0.25,  0.  ])
    """
    P = asarray(P)
    N = P.shape[0]
    P1 = 1 - P #convenience variable that is 1 minus P

    #starting distribution of the sum (no variables in summation) is D[0] = 1 and D[>0] = 0
    #in otherwords, we know the sum is zero.
    D = zeros(N + 1)
    D[0] = 1



    for n in range(N):
        if n in ignore_idx:
            continue #skip P[n] if n in ignore_idx

        #Convolution of the distribution of the nth variable with the summation distribution for the first n-1 variables.
        C = D * P1[n] 
        C[1:] += D[:-1] * P[n]
        
        D = C #the distribution of the sum of the first n variables
        
    #D is now the distribution of sums for all variables
    return D


def expectation_range_CUBIC(P, lower, upper):
    """
    Given a set of binomial random variables parameterized by a vector P.
    Conditioned on the number successes between lower and upper (inclusive).
    What is the expectation of each random variable?
    
    Output is a vector E of expectations.
    O(N^3) time in length of P

    >>> R = rand(10)  # a random vector of probabilities 10 elements long.
    >>>
    >>> lower, upper = 3, 6
    >>>
    >>> EC = expectation_range_CUBIC(R, lower, upper)
    >>> EE = expectation_range_EXP(R, lower, upper)
    
    >>> correlation, pvalue = pearsonr(EE, EC)
    >>> correlation > .99 and pvalue < .01
    True
    
    This shows that both versions yield results that are > 99% correlated.
    """
    P = asarray(P)
    N = P.shape[0]
    E = zeros(N)
    if upper == 0:
        return E

    D = distribution_of_sum(P)
    PROB_IN_RANGE = sum(D[max(0, lower): min(N, upper + 1)])

    for i in range(N):
        #Implementation described in paper
        D = distribution_of_sum(P, [i])
        PROB_IN_RANGE_CONDITION_ON_I = sum(D[max(0, lower - 1): min(N, upper)])  
        E[i] = P[i] * PROB_IN_RANGE_CONDITION_ON_I / PROB_IN_RANGE

        #Alternative, equivalent implementation
        #POS = sum(D[max(0, lower - 1): min(N, upper)]) * P[i]
        #NEG = sum(D[max(0, lower): min(N, upper + 1)]) * (1 - P[i])
        #E[i] = POS/ (POS + NEG)
    return E


def expectation_range_EXP(P, lower, upper):
    """
    Given a set of binomial random variables parameterized by a vector P.
    Condition on the number successes between lower and upper (inclusive).
    What is the expectation of each random variable?
    
    This version is slow, but more conceptually clear.
    
    Output is a vector E of expectations.
    O(2^N) time in length of P
    
    This version suffers from floating point error, and should not be used
    for anything other than testing.
    """
    P = asarray(P)
    N = P.shape[0]
    P1 = 1 - P
    E = zeros(N)
    if upper == 0:
        return E

    import itertools

    D = 0
    
    for S in itertools.product(*tuple([[1, 0]] * N)): #iterate over all binary vectors of length N
        SUM = sum(S)
        if SUM >= upper or SUM <= lower:
            continue #skip the vectors without the right sum
                
        S = array(S)
        
        p = prod(where(S, P, P1)) #probability of S according to P

        E += p * S #summing up this vector's contribution to the final expectation
        D += p #sum up this contribution to the denominator
        
    E = E / D #normalize by dividing by $\sum Pr(S)$ 
    return E

def expectation_or_LINEAR(P, E_or):
    """
    Given a set of binomial random variables parameterized by a vector P.
    Conditioned on E[at least one success] = E_or
    What is the expectation of each random variable?

    Output is a vector E of expectations.    

    All the implementations should produce the same results.

    >>> R = rand(10)
    >>>
    >>> EL = expectation_or_LINEAR(R, 1)
    >>> EC = expectation_or_CUBIC(R, 1)
    >>> EE = expectation_E_EXP(R, 1)

    >>> correlation, pvalue = pearsonr(EL, EC)
    >>> correlation > .99 and pvalue < .01
    True

    >>> allclose(EL, EE)
    True
    >>> correlation, pvalue = pearsonr(EL, EE)
    >>> correlation > .99 and pvalue < .01
    True

    This shows that all versions yield results that are > 99% correlated.
      
    And we know the results for some simple cases.
    
    >>> expectation_or([0.5, 0.5], 1)
    array([ 0.66666667,  0.66666667])
    >>> expectation_or([0.5, 0.5], .75)
    array([ 0.5,  0.5])
    """
    P = asarray(P)
    # boundary case that is easy to compuate and would cause problems if we
    # let it pass through
    if any(P == 1):
        return P * E_or

    # probability of success at this index, but failure or success at all
    # others
    P1 = P
    # probability of failure at this index, but success at one or more others
    P0 = (1 - prod(1 - P) / (1 - P)) * (1 - P)
    # given that at least one is success, what is the probability of success
    # for this index
    PS = P1 / (P1 + P0)
    # but the true probability is E_or, so adjust expectations down and return
    return PS * E_or


def expectation_or_CUBIC(P, E_or):
    """
    Given a set of binomial random variables parameterized by a vector P.
    Conditioned on E[at least one success] = E_or
    What is the expectation of each random variable?

    Output is a vector E of expectations.    
    
    alternate, equivalent implementation for error checking
    the problem with this implementation is that it is O(N^3) time
    """
    return expectation_range(P, 1, inf) * E_or


def expectation_E_EXP(P, E_or):
    """
    Given a set of binomial random variables parameterized by a vector P.
    Conditioned on E[at least one success] = E_or
    What is the expectation of each random variable?

    Output is a vector E of expectations.    
 
    alternate, equivalent implementation for error checking
    the problem with this implementation is that it is exponential time
    """
    P = asarray(P)
    P1 = 1 - P
    N = P.shape[0]
    E = zeros(N)
    import itertools
    for S in itertools.product(*tuple([[1, 0]] * N)): #iterate over all binary vectors of length N
        S = array(S)
        p = prod(where(S, P, P1)) #compute the probability according to P of vector
        E += p * S #accumulate the probability-weighted average
    E = E * E_or / (1 - prod(P1)) #divide by the probability of getting at least 1 success and multiply times E_or
    return E


expectation_or = expectation_or_LINEAR
expectation_range = expectation_range_CUBIC



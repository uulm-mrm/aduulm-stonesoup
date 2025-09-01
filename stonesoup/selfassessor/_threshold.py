import numpy as np
import math
from scipy.stats import norm


def calc_threshold_n_diff(n_b, n_s, alpha):
    """Calculates DC-threshold for different opinions.

    This method calculates the threshold to check if two opinions are
    different.
    Therefore, the given confidence is used.
    After calculating the threshold, the DC between two opinions can be
    calculated and if the value does exceed the threshold then the two
    opinions can be considered different with a confidence of 1-alpha.
    This method uses the algorithm from:
    https://www.jstor.org/stable/2684318

    Parameters
    ----------
    n_b : int or list or dict, array_like
        Number of bins of the opinion.
    n_s : int or list or dict, array_like
        Number of samples used for the creation of the opinion.
        This can be obtained by summing the evidence of all bins of the
        opinion.
    alpha : float or list or dict, array_like
        Used to define wanted confidence 1-alpha.

    Raises
    ------
    ValueError
        If input arrays have different length or inputs have different type.

    Returns
    -------
    theta : float or list or dict, array_like
        DC-threshold above which two opinions are considered to be different.
    """
    # if any of the given values is a scalar it is converted to a list
    if not ((type(n_s) is list) or (type(n_s) is dict)):
        n_s = [n_s]
    if not ((type(n_b) is list) or (type(n_b) is dict)):
        n_b = [n_b]
    if not ((type(alpha) is list) or (type(alpha) is dict)):
        alpha = [alpha]
    if not (len(n_b) == len(n_s) == len(alpha)):
        raise ValueError('Given input values for DC confidence-based threshold calculation must have same length.')

    if type(n_s) == type(n_b) == type(alpha) == dict:
        type_tmp = 'dict'
        theta = {}
    elif type(n_s) == type(n_b) == type(alpha) == list:
        type_tmp = 'list'
        theta = np.zeros(len(n_s))
    else:
        raise ValueError('Given input variables for DC confidence-based threshold must have the same type.')

    for index, value in enumerate(n_s):

        if type_tmp == 'dict':
            idx = value
        elif type_tmp == 'list':
            idx = index

        beta = 1 - (1 - alpha[idx]) / (2 * n_b[idx])
        z = norm.ppf(beta)
        d2n = z ** 2 / n_b[idx] * (1 - 1 / n_b[idx])
        d = math.sqrt(d2n / n_s[idx])
        theta[idx] = d / 2 * n_b[idx] * (1 - n_b[idx] / (n_b[idx] + n_s[idx]))

    if isinstance(theta, dict):
        return theta
    elif len(theta) > 1:
        return theta
    else:
        return theta[0]


def calc_threshold_op_diff(op_X, alpha):
    """Calculates DC-threshold for different opinions.

    This method uses the similar named method above.
    Instead of getting the sample size and the number of bins as parameters,
    the values are extracted from the given opinion.


    Parameters
    ----------
    op_X : MultiOpinion, array_like
        Opinion for the evaluation of the tracking performance.
    alpha : float, array_like
        Used to define wanted confidence 1-alpha.

    Returns
    -------
    theta : float, array_like
        DC-threshold above which two opinions are considered different.

    """
    if not (type(op_X) is list):
        op_X = [op_X]

    if isinstance(op_X[0] , dict):
        n_s = [int(sum(op.get_evidence())) if int(sum(op.get_evidence())) > 0 else 1 for op in op_X[0].values()]
        n_b = [op_X_i.W for op_X_i in op_X[0].values()]
    else:
        n_s = [int(sum(op.get_evidence())) if int(sum(op.get_evidence())) > 0 else 1 for op in op_X]
        n_b = [op_X_i.W for op_X_i in op_X]

    return calc_threshold_n_diff(n_b, n_s, alpha)

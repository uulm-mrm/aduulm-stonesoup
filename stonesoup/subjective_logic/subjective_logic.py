"""Docstring for the subjective_logic.py module.

This class implements a subjective logic toolbox. The algorithms are from
    [Jøsang 2016]: 'Subjective Logic - A formalism for reasoning under uncertainty'
    from Audun Jøsang in 2016.
The references to equations, pages ect. are related to this book.

Application is seen in combination with the Tracking Toolbox 'Stone Soup'
"""

from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.stats import beta
from numpy.typing import ArrayLike
from scipy.stats import dirichlet


def dc(w_a, w_b, components: bool = False):
    """Calculate degree of conflict (dc) of two given opinions.

    Uses equation (4.63) p. 80

    Parameters
    ----------
    w_a, w_b : MultiOpinion
        Given opinions between which the degree of conflict is calculated.
    components: boolean, optional
        Decides if dc is returned in components
        (projected distance and conjunctive uncertainty) or as one value.

    Returns
    -------
    float
        degree of conflict of the two given opinions

    """
    if components:
        projected_distance = \
            0.5*sum(abs(w_a.get_proj_prob()-w_b.get_proj_prob()))
        conjunctive_certainty = (1-w_a.uncertainty) * (1-w_b.uncertainty)
        return [projected_distance, conjunctive_certainty]
    else:
        return 0.5*sum(abs(w_a.get_proj_prob()-w_b.get_proj_prob())) * \
            (1-w_a.uncertainty) * (1-w_b.uncertainty)

def ud(w_a, w_b):
    """Calculate uncertainty differential of opinion a with respect to
        opinion b.

    Uncertainty differential as in [Jøsang 2016].

    Parameters
    ----------
    w_a, w_b : MultiOpinion
        Given opinions

    Returns
    -------
    float
        Uncertainty differential of opinion a with respect to opinion b
    """
    u_a = w_a.uncertainty
    u_b = w_b.uncertainty

    return u_a / (u_a + u_b)


def ud_general(idx: int, ws):
    """Calculate uncertainty differential of opinion idx with respect to the
        other opinions in ws.

    Extension of uncertainty differential as in [Jøsang 2016] (only for two
    opinions) to an arbitrary number of opinions.

    Parameters
    ----------
    idx : int
        Index of the MultiOpinion (in the ws list) with respect to which
        we calculate the uncertainty differential
    ws : array_like, MultiOpinion
        Given opinions

    Returns
    -------
    float
        Uncertainty differential of opinion with index idx with respect to
        all other opinions in array ws
    """
    u_idx = ws[idx].uncertainty
    us = [ws[idx_tmp].uncertainty for idx_tmp in range(len(ws))]

    return u_idx / sum(us)


def fusion_acbf(w_a, w_b):
    """Calculate aleatory cumulative fusion.

    Uses equation (12.14) and (12.15) p. 226

    Parameters
    ----------
    w_a : MultiOpinion
        Prior known opinion to which the new opinion is fused.
    w_b : MultiOpinion
        New Opinion which is fused into the existing opinion.

    Returns
    -------
    w_ab : MultiOpinion
        Fused opinion containing evidence of both opinions.

    """
    u_a = w_a.uncertainty
    u_b = w_b.uncertainty
    b_a = w_a.belief
    b_b = w_b.belief
    a_a = w_a.baseRate
    a_b = w_b.baseRate

    if math.isclose(u_a, 0, abs_tol=10**-5) and \
            math.isclose(u_b, 0, abs_tol=10**-5):
        # gamma values are 0.5 as stated on p. 227
        g_a = 0.5
        g_b = 0.5
        b_ab = g_a*b_a + g_b*b_b
        u_ab = 0
        a_ab = g_a*a_a + g_b*a_b
    else:
        n = u_a+u_b-u_a*u_b
        b_ab = (b_a*u_b + b_b*u_a) / n
        u_ab = u_a*u_b / n
        if math.isclose(u_a, 1, abs_tol=10**-5) and \
                math.isclose(u_b, 1, abs_tol=10**-5):
            a_ab = 0.5 * (a_a + a_b)
        else:
            a_ab = (a_a*u_b + a_b*u_a - (a_a+a_b)*u_a*u_b) / (u_a+u_b-2*u_a*u_b)

    try:
        if isinstance(w_a, BiOpinion):
            w_ab = BiOpinion(b_ab[0], b_ab[1], a_ab[0], u_ab)
        elif isinstance(w_a, MultiOpinion):
            w_ab = MultiOpinion(b_ab, a_ab, u_ab)
    except ValueError:
        # output false values
        # this should not happen, if nonetheless the wrong values are printed
        # for debugging and rescaled forcefully to ensure a working fusion
        # after intensive testing, this should be removed
        print(f'b: {b_ab}')
        print(f'a: {a_ab}')
        print(f'u: {u_ab}')
        # rescale
        b_ab = b_ab / (sum(b_ab) + u_ab)
        u_ab = u_ab / (sum(b_ab) + u_ab)
        a_ab = a_ab / sum(a_ab)
        # try again
        if isinstance(w_a, BiOpinion):
            w_ab = BiOpinion(b_ab[0], b_ab[1], a_ab[0], u_ab)
        elif isinstance(w_a, MultiOpinion):
            w_ab = MultiOpinion(b_ab, a_ab, u_ab)

    return w_ab


def unfusion_cu(w_ab, w_b):
    """Calculate cumulative unfusion.

    Uses equation (13.1) and (13.2) p. 238

    Parameters
    ----------
    w_ab : MultiOpinion
        Fused opinion containing evidence of both opinions.
    w_b : MultiOpinion
        New Opinion which is unfused from the existing opinion.

    Returns
    -------
    MultiOpinion
        Prior known opinion without the evidence added by new opinion.

    """
    u_ab = w_ab.uncertainty
    u_b = w_b.uncertainty
    b_ab = w_ab.belief
    b_b = w_b.belief

    if (u_ab == 0) and (u_b == 0):
        # gamma values are assumed to be 0.5 as stated on p. 227
        g_b = 0.5
        g_ab = 0.5
        b_a = g_b*b_ab - g_ab*b_b
        u_a = 0
    else:
        n = u_b-u_ab+u_b*u_ab
        b_a = (b_ab*u_b-b_b*u_ab) / n
        u_a = u_b*u_ab / n
    if isinstance(w_ab, BiOpinion):
        w_a = BiOpinion(b_a[0], b_a[1], w_ab.baseRate[0], u_a)
    elif isinstance(w_ab, MultiOpinion):
        w_a = MultiOpinion(b_a, w_ab.baseRate, u_a)
    else:
        raise Exception("Error case in cumulative unfusion calculation!")

    return w_a

def fusion_averaging(w_a, w_b):
    """Calculate averaging belief fusion.

    Uses equation (12.14) and (12.15) p. 226

    Parameters
    ----------
    w_a : MultiOpinion
        Prior known opinion to which the new opinion is fused.
    w_b : MultiOpinion
        New Opinion which is fused into the existing opinion.

    Returns
    -------
    w_ab : MultiOpinion
        Fused opinion containing evidence of both opinions.

    """
    u_a = w_a.uncertainty
    u_b = w_b.uncertainty
    b_a = w_a.belief
    b_b = w_b.belief
    a_a = w_a.baseRate
    a_b = w_b.baseRate
    # check if the uncertainty value is near to zero
    if math.isclose(u_a, 0, abs_tol=10 ** -5) and \
                    math.isclose(u_b, 0, abs_tol=10 ** -5):
        # gamma values are 0.5 as stated on p. 227
        g_a = 0.5
        g_b = 0.5
        b_ab = g_a*b_a + g_b*b_b
        u_ab = 0
        a_ab = g_a*a_a + g_b+a_b
    else:
        denominator = u_a + u_b
        b_numerator = b_a * u_b + b_b * u_a
        b_ab = b_numerator/denominator
        u_numerator = 2 * u_a * u_b
        u_ab = u_numerator/denominator
        a_ab = 0.5 * (a_a + a_b)
    try:
        if isinstance(w_a, BiOpinion):
            w_ab = BiOpinion(b_ab[0], b_ab[1], a_ab[0], u_ab)
        elif isinstance(w_a, MultiOpinion):
            w_ab = MultiOpinion(b_ab, a_ab, u_ab)
    except ValueError:
        # output false values
        # this should not happen, if nonetheless the wrong values are printed
        # for debugging and rescaled forcefully to ensure a working fusion
        # after intensive testing, this should be removed
        print(f'b: {b_ab}')
        print(f'a: {a_ab}')
        print(f'u: {u_ab}')
        # rescale
        b_ab = b_ab / (sum(b_ab) + u_ab)
        u_ab = u_ab / (sum(b_ab) + u_ab)
        a_ab = a_ab / sum(a_ab)
        # try again
        if isinstance(w_a, BiOpinion):
            w_ab = BiOpinion(b_ab[0], b_ab[1], a_ab[0], u_ab)
        elif isinstance(w_a, MultiOpinion):
            w_ab = MultiOpinion(b_ab, a_ab, u_ab)

    return w_ab

def unfusion_averaging(w_ab, w_b):
    """Calculate averaging unfusion.

    Parameters
    ----------
    w_ab : MultiOpinion
        Fused opinion containing evidence of both opinions.
    w_b : MultiOpinion
        New Opinion which is unfused from the existing opinion.

    Returns
    -------
    MultiOpinion
        Prior known opinion without the evidence added by new opinion.

    """
    u_ab = w_ab.uncertainty
    u_b = w_b.uncertainty
    b_ab = w_ab.belief
    b_b = w_b.belief
    if (u_ab == 0) and (u_b == 0):
        # gamma values are assumed to be 0.5 as stated on p. 227
        g_b = 0.5
        g_ab = 0.5
        b_a = g_b * b_ab - g_ab * b_b
        u_a = 0
    else:
        b_numerator = 2 * b_ab * u_b - b_b * u_ab
        denominator = 2 * u_b - u_ab
        u_numerator = u_b * u_ab
        b_a = b_numerator / denominator
        u_a = u_numerator / denominator

    if isinstance(w_ab, BiOpinion):
        w_fused = BiOpinion(b_a[0], b_a[1], w_ab.baseRate[0], u_a)
    elif isinstance(w_ab, MultiOpinion):
        w_fused = MultiOpinion(b_a, w_ab.baseRate, u_a)
    else:
        raise Exception("Error case in cumulative unfusion calculation!")

    return w_fused

def fusion_acbf_expanded(w_all):
    """Calculate aleatory cumulative fusion of multisensor-system.

    Uses equation of the paper multi source trust revision by Audun Josang

    Parameters
    ----------
    w_all : MultiOpinion
        Prior known opinions to which the new opinion is fused.

    Returns
    -------
    w_fused : MultiOpinion
        Fused opinion containing evidence of all opinions.

    """

    b_numerator = [0 for x in w_all[0].belief]
    denominator1 = 0
    denominator2 = len(w_all) - 1
    a_numerator1 = [0 for x in w_all[0].baseRate]
    a_numerator2 = [0 for x in w_all[0].baseRate]
    u_total = 1
    uncertainty_array = []

    # Case2: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8009635
    for i, w_regarded1 in enumerate(w_all):
        uncertainty_array.append(w_regarded1.uncertainty)

    if (0 in uncertainty_array):
        u_fused = 0
        a_fused = 0
        b_fused = 0
        for i, w_regarded1 in enumerate(w_all):
            if (math.isclose(w_regarded1.uncertainty, 0, abs_tol=10 ** -5)):
                b_fused = b_fused + 1 / len(uncertainty_array) * w_regarded1.belief
                a_fused = a_fused + 1 / len(uncertainty_array) * w_regarded1.baseRate
            else:
                b_fused = b_fused + w_regarded1.uncertainty / (np.sum(uncertainty_array)) * w_regarded1.belief
                a_fused = a_fused + w_regarded1.uncertainty / (np.sum(uncertainty_array)) * w_regarded1.baseRate

        b_fused = [sum(x) for x in zip(b_fused)]

        if isinstance(w_all[0], BiOpinion):
            w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
        elif isinstance(w_all[0], MultiOpinion):
            w_fused = MultiOpinion(b_fused, a_fused, u_fused)
    else:
        for i, w_regarded1 in enumerate(w_all):
            baseRate_temp = w_regarded1.baseRate
            a_numerator2 = a_numerator2 + w_regarded1.baseRate
            belief_temp = w_regarded1.belief
            u_total = u_total * w_regarded1.uncertainty
            uncertainty_temp = 1
            for j, w_regarded2 in enumerate(w_all):
                if j != i:
                    uncertainty_temp = uncertainty_temp * w_regarded2.uncertainty
                    belief_temp = belief_temp * w_regarded2.uncertainty
                    baseRate_temp = baseRate_temp * w_regarded2.uncertainty
            b_numerator = b_numerator + belief_temp
            denominator1 = denominator1 + uncertainty_temp
            a_numerator1 = a_numerator1 + baseRate_temp
        denominator2 = denominator2 * u_total
        a_numerator2 = a_numerator2 * u_total
        b_fused = b_numerator/(denominator1-denominator2)
        u_fused = u_total/(denominator1-denominator2)
        if (all(math.isclose(val, 1, abs_tol=10 ** -5) for val in uncertainty_array)):
            a_fused = 0
            for i, w_regarded1 in enumerate(w_all):
                a_fused = a_fused + w_regarded1.baseRate / len(w_all)
        else:
            a_fused = (a_numerator1 - a_numerator2)/(denominator1-len(w_all)*u_total)
        try:
            if isinstance(w_all[0], BiOpinion):
                w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
            elif isinstance(w_all[0], MultiOpinion):
                w_fused = MultiOpinion(b_fused, a_fused, u_fused)
        except ValueError:
            # output false values
            # this should not happen, if nonetheless the wrong values are printed
            # for debugging and rescaled forcefully to ensure a working fusion
            # after intensive testing, this should be removed
            print(f'b: {b_fused}')
            print(f'a: {a_fused}')
            print(f'u: {u_fused}')
            # rescale
            b_fused = b_fused / (sum(b_fused) + u_fused)
            u_fused = u_fused / (sum(b_fused) + u_fused)
            a_fused = a_fused / sum(a_fused)
            # try again
            if isinstance(w_all[0], BiOpinion):
                w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
            elif isinstance(w_all[0], MultiOpinion):
                w_fused = MultiOpinion(b_fused, a_fused, u_fused)

    return w_fused


def fusion_averaged_expanded(w_all):
    """Calculate averaging fusion of multisensor-system.

    Uses equation of the paper:
        A. J⊘sang, J. Zhang and D. Wang, "Multi-source trust revision,"
        2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017,
        pp. 1-8, doi: 10.23919/ICIF.2017.8009635.

    Parameters
    ----------
    w_all : MultiOpinion
        Prior known opinions to which the new opinion is fused.

    Returns
    -------
    w_fused : MultiOpinion
        Fused opinion containing evidence of all opinions.

    """

    b_numerator = [0 for x in w_all[0].belief]
    denominator = 0
    u_numerator = len(w_all)
    a_numerator1 = [0 for x in w_all[0].baseRate]
    uncertainty_array = []
    for i, w_regarded1 in enumerate(w_all):
        uncertainty_array.append(w_regarded1.uncertainty)
    # Case2: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8009635
    if(0 in uncertainty_array ):
        u_fused = 0
        a_fused = 0
        b_fused = 0
        for i, w_regarded1 in enumerate(w_all):
            if(math.isclose(w_regarded1.uncertainty, 0, abs_tol=10 ** -5) ):
                b_fused = b_fused + 1/len(uncertainty_array) * w_regarded1.belief
                a_fused = a_fused + 1/len(uncertainty_array) * w_regarded1.baseRate
            else:
                b_fused =  b_fused +  w_regarded1.uncertainty/(np.sum(uncertainty_array))*w_regarded1.belief
                a_fused =  a_fused +  w_regarded1.uncertainty/(np.sum(uncertainty_array))*w_regarded1.baseRate

        b_fused = [sum(x) for x in zip(b_fused)]

        if isinstance(w_all[0], BiOpinion):
            w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
        elif isinstance(w_all[0], MultiOpinion):
            w_fused = MultiOpinion(b_fused, a_fused, u_fused)
    else:
        for i, w_regarded1 in enumerate(w_all):
            a_numerator1 = a_numerator1 + w_regarded1.baseRate
            belief_temp = w_regarded1.belief
            u_numerator = u_numerator * w_regarded1.uncertainty
            uncertainty_temp = 1
            for j, w_regarded2 in enumerate(w_all):
                if j != i:
                    uncertainty_temp = uncertainty_temp * w_regarded2.uncertainty
                    belief_temp = belief_temp * w_regarded2.uncertainty
            b_numerator = b_numerator + belief_temp
            denominator = denominator + uncertainty_temp
        b_fused = b_numerator/denominator
        u_fused = u_numerator/denominator
        a_fused = (a_numerator1 / len(w_all))
        try:
            if isinstance(w_all[0], BiOpinion):
                w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
            elif isinstance(w_all[0], MultiOpinion):
                w_fused = MultiOpinion(b_fused, a_fused, u_fused)
        except ValueError:
            print(f'b: {b_fused}')
            print(f'a: {a_fused}')
            print(f'u: {u_fused}')
            # rescale
            b_fused = b_fused / (sum(b_fused)+u_fused)
            u_fused = u_fused / (sum(b_fused)+u_fused)
            a_fused = a_fused / sum(b_fused)
            # try again
            if isinstance(w_all[0], BiOpinion):
                w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
            elif isinstance(w_all[0], MultiOpinion):
                w_fused = MultiOpinion(b_fused, a_fused, u_fused)

    return w_fused


def fusion_weighted_belief(w_all):
    """Calculate weighted fusion of multisensor-system.

    Uses equation of the paper:
        A. J⊘sang, J. Zhang and D. Wang, "Multi-source trust revision,"
        2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017,
        pp. 1-8, doi: 10.23919/ICIF.2017.8009635.

    Parameters
    ----------
    w_all : MultiOpinion
            Prior known opinions to which the new opinion is fused.

    Returns
    -------
    w_fused : MultiOpinion
              Fused opinion containing evidence of all opinions.

    """
    b_numerator = 0
    denominator_1 = 0
    denominator_2 = len(w_all)
    N = len(w_all)
    uncertainty_sum = 0
    uncertainty_product = 1
    a_numerator = 0
    uncertainty_array = []

    # Case 2: https://arxiv.org/pdf/1805.01388.pdf
    for i, w_regarded1 in enumerate(w_all):
        uncertainty_array.append(w_regarded1.uncertainty)

    if(0 in uncertainty_array ):

        u_fused = 0
        a_fused = 0
        b_fused = 0
        for i, w_regarded1 in enumerate(w_all):
            if(math.isclose(w_regarded1.uncertainty, 0, abs_tol=10 ** -5) ):
                b_fused = b_fused + 1/len(uncertainty_array) * w_regarded1.belief
                a_fused = a_fused + 1/len(uncertainty_array) * w_regarded1.baseRate
                print("a_fused")
            else:
                b_fused =  b_fused +  w_regarded1.uncertainty/(np.sum(uncertainty_array))*w_regarded1.belief
                a_fused =  a_fused +  w_regarded1.uncertainty/(np.sum(uncertainty_array))*w_regarded1.baseRate

        b_fused = [sum(x) for x in zip(b_fused)]

        if isinstance(w_all[0], BiOpinion):
            w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
        elif isinstance(w_all[0], MultiOpinion):
            w_fused = MultiOpinion(b_fused, a_fused, u_fused)
    elif( all(val>=0.9999999 for val in uncertainty_array)):
        b_fused = 0
        u_fused = 1
        a_fused = 0
        for i, w_regarded1 in enumerate(w_all):
            a_fused = a_fused + w_regarded1.baseRate/len(w_all)

        if isinstance(w_all[0], BiOpinion):
            w_fused = BiOpinion(0,0, a_fused[0], u_fused)
        elif isinstance(w_all[0], MultiOpinion):
            w_fused = MultiOpinion(np.zeros(len(a_fused)), a_fused, u_fused)
    else:

        for i, w_regarded1 in enumerate(w_all):
            uncertainty_temp = 1
            uncertainty_sum = uncertainty_sum + w_regarded1.uncertainty
            uncertainty_product = uncertainty_product * w_regarded1.uncertainty
            denominator_2 = denominator_2 * w_regarded1.uncertainty
            b_temp = w_regarded1.belief * (1-w_regarded1.uncertainty)
            a_numerator = a_numerator + w_regarded1.baseRate * (1 - w_regarded1.uncertainty)
            for j, w_regarded2 in enumerate(w_all):
                if j != i:
                    uncertainty_temp = uncertainty_temp * w_regarded2.uncertainty
            denominator_1 = denominator_1 + uncertainty_temp
            b_numerator = b_numerator + b_temp * uncertainty_temp
        denominator = denominator_1 - denominator_2
        b_fused = b_numerator/denominator
        u_fused = ((N-uncertainty_sum)*uncertainty_product)/denominator
        a_fused = a_numerator/ (N - uncertainty_sum)
        try:
            if isinstance(w_all[0], BiOpinion):
                w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
            elif isinstance(w_all[0], MultiOpinion):
                w_fused = MultiOpinion(b_fused, a_fused, u_fused)
        except ValueError:
            print(f'b: {b_fused}')
            print(f'a: {a_fused}')
            print(f'u: {u_fused}')
            # rescale
            b_fused = b_fused / (sum(b_fused)+u_fused)
            u_fused = u_fused / (sum(b_fused)+u_fused)
            a_fused = a_fused / sum(b_fused)
            # try again
            if isinstance(w_all[0], BiOpinion):
                w_fused = BiOpinion(b_fused[0], b_fused[1], a_fused[0], u_fused)
            elif isinstance(w_all[0], MultiOpinion):
                w_fused = MultiOpinion(b_fused, a_fused, u_fused)

    return w_fused

def binomial_multiplication(w_x, w_y):
    """Calculate binomial multiplication of opinions.

    Uses equation (7.1) p. 102

    Parameters
    ----------
    w_x : BiOpinion
        First binomial Opinion.
    w_y : BiOpinion
        Second binomial Opinion.

    Raises
    ------
    ValueError
        Raised if the provided opinions are not binomial.

    Returns
    -------
    w_xandy : BiOpinion
        Multiplied binomial opinion.

    """
    # check if all components contain two elements
    if any([len(b) != 2 for b in [w_x.belief, w_x.baseRate, w_y.belief,
                                  w_y.baseRate]]):
        raise ValueError('The provided opinions are not binomial.')
    # get values from inputs for better formulas afterwards
    b_x = w_x.belief[0]
    b_y = w_y.belief[0]
    d_x = w_x.belief[1]
    d_y = w_y.belief[1]
    a_x = w_x.baseRate[0]
    a_y = w_y.baseRate[0]
    u_x = w_x.uncertainty
    u_y = w_y.uncertainty

    # calculate value for multiplied opinion
    b_xay = b_x*b_y + ((1-a_x)*a_y*b_x*u_y + a_x*(1-a_y)*u_x*b_y)/(1 - a_x*a_y)
    d_xay = d_x + d_y - d_x*d_y
    u_xay = u_x*u_y + ((1-a_y)*b_x*u_y + (1-a_x)*u_x*b_y)/(1 - a_x*a_y)
    a_xay = a_x*a_y

    # create opinion from values
    w_xandy = BiOpinion(b_xay, d_xay, a_xay, u_xay)
    return w_xandy


class Opinion(ABC):
    """Abstract class of subjective logic opinion.

    Attributes
    ----------
    belief : array_like, float
             states the belief in the different possible values of X
    baseRate: array_like, float
              states the default belief in the different possible values of X
              without additional evidence
    uncertainty : float
                  states the uncertainty about the belief
    W : int
        states the dimensionality of the opinion
    """

    def __init__(self):
        """Construct abstract class Opinion.

        Initializes standard values for an opinion

        Returns
        -------
        Opinion

        """
        self.belief = np.array([0, 0])
        self.baseRate = np.array([0.5, 0.5])
        self.uncertainty = np.array([1, 1])
        self.W = 2

    def get_proj_prob(self):
        """Calculate projected probability of opinion.

        Uses equation (3.12) p. 30

        Returns
        -------
        array_like, float
            projected probability for each element of X

        """
        return self.belief + self.baseRate*self.uncertainty

    def get_proj_var(self):
        """Calculate projected variance of opinion.

        Uses equation (3.13) p. 30

        Returns
        -------
        array_like, float
            projected covariance matrix for X

        """
        p = self.get_proj_prob()
        return p*(1-p)*self.uncertainty / (self.W + self.uncertainty)

    def get_evidence(self):
        """Calculate the evidence for each Value in X.

        Uses equation (3.23) p. 37

        Returns
        -------
        r : array_like, int
            Amount of evidence supporting the value of X

        """
        if self.uncertainty != 0:
            r = self.W * self.belief / self.uncertainty
        else:
            r = self.belief * math.inf
        return r

    @abstractmethod
    def trust_discount(self, p_disc: float = 0.9):
        """Abstract method to calculate the trust discount.

        Returns
        -------
        None.

        """

    @abstractmethod
    def is_valid(self):
        """Abstract method to check for additivity requirement.

        Returns
        -------
        None.

        """


class MultiOpinion(Opinion):
    """Implementation of abstract class of subjective logic multinomial opinion.

    Described in chapter 3.5 p. 30

    Attributes
    ----------
    belief : array_like, float
             states the belief in the different possible values of X
    baseRate: array_like, float
              states the default belief in the different possible values of X
              without additional evidence
    uncertainty : float
                  states the uncertainty about the belief
    W : int
        states the dimensionality of the opinion
    """

    def __init__(self, br: ArrayLike, a: ArrayLike, u: float = 1, evidence: bool = False):
        """Construct multinomial opinions.

        Creates object of type MultiOpinion.

        Parameters
        ----------
        br : array_like, float or int
            either states the belief in the different possible values of X
            or the evidence supporting the different possible values of X.
        a : array_like, float
            base rate
            states the default belief in the different possible values of X
            without additional evidence.
        u : float, optional (not needed for evidence representation)
            states the uncertainty about the belief.
        evidence : bool, optional
            states if the Opinion is defined using the
            evidence notation eq (3.23) p. 37 or using the belief notation.
            The default is False and thus the belief notation.

        Returns
        -------
        MultiOpinion.

        """
        super().__init__()
        self.W = np.size(br)
        self.baseRate = a
        if evidence:
            self.uncertainty = self.W / (self.W + sum(br))
            self.belief = br / (self.W + sum(br))
        else:
            self.uncertainty = u
            self.belief = br
        if not self.is_valid():
            raise ValueError('Input parameters do not comply to the additivity '
                             + 'requirement.\n' + f'b={self.belief} '
                                                  f'u={self.uncertainty}, '
                                                  f'a={self.baseRate}')

    def trust_discount(self, p_disc: float = 0.9):
        """Calculate the trust discount.

        Updates the value of belief of the calling object using trust discount.
        Uses equation (14.6) p. 256

        Parameters
        ----------
        p_disc : float, optional
            States the discount probability of the opinion. The default is 0.9.

        Returns
        -------
        MultiOpinion
            The opinion itself to be used in the code directly

        """
        self.belief *= p_disc
        self.uncertainty = 1-sum(self.belief)
        return self

    def is_valid(self):
        """Check for additivity requirements.

        Checks if the additivity formulas are given:
            sum(b)+u=1 equation (2.6) p. 14
            sum(a)=1   equation (2.8) p. 15
        Compares 5 significant figures

        Returns
        -------
        bool
            states if the opinion has valid attributes.

        """

        return math.isclose(self.uncertainty +
                            sum(self.belief), 1, abs_tol=10**-5) and \
            math.isclose(sum(self.baseRate), 1, abs_tol=10**-5)

    def calculate_dirichlet_pdf(self, p):
        """Calculate values of Dirichlet-PDF for given p-values of this opinion.

        Parameters
        ----------
        p : array_like, float
            vector of probability over which to plot the Beta-PDF
            The sum of p must be equal to one.

        Returns
        -------
        beta_val : array_like, float
                   Values, which represent the Beta PDF for this probability

        """

        alpha = self.get_evidence() + self.baseRate * self.W
        dirichlet_val = dirichlet.pdf(p, alpha)

        return dirichlet_val


class BiOpinion(MultiOpinion):
    """Implementation of child class of subjective logic multinomial opinion.

    This class is used as a wrapper for binomial opinions.
    Internally these can be modeled as multinomial opinions which is used here.
    The classical disbelief is the second value in the belief array.

    Important: If you want to get the value of belief or disbelief you have to
    use BiOpinion.belief[0] and BiOpinion.belief[1] respectively.
    The base rate for the belief is also obtained using BiOpinion.baseRate[0].


    Attributes
    ----------
    belief : array_like, float
             states the belief in the different possible values of X
    baseRate: array_like, float
              states the default belief in the different possible values of X
              without additional evidence
    uncertainty : float
                  states the uncertainty about the belief
    W : int
        states the dimensionality of the opinion, which is 2
    """

    def __init__(self, br: ArrayLike, ds: ArrayLike, a: ArrayLike, u: float, evidence: bool = False):
        """Construct binomial opinions.

        Creates object of type BiOpinion.

        Parameters
        ----------
        br : array_like, float or int
            either states the belief in the different possible values of X
            or the evidence supporting the different possible values of X.
        ds : array_like, float or int
            either states the disbelief in the different possible values of X
            or the evidence opposing the different possible values of X.
        a : array_like, float
            base rate
            states the default belief in the different possible values of X
            without additional evidence.
        u : float
            states the uncertainty about the belief.
        evidence : bool, optional
            states if the Opinion is defined using the
            evidence notation eq (3.23) p. 37 or using the belief notation.
            The default is False and thus the belief notation.

        Returns
        -------
        BiOpinion.

        """
        if (np.size(br) > 1) or (np.size(ds) > 1) or (np.size(a) > 1) or \
                (np.size(u) > 1):
            raise ValueError('Input parameters must be scalar.')
        b = np.array([br, ds])
        a = np.array([a, 1-a])
        super().__init__(b, a, u, evidence)

    def calculate_beta_pdf(self, p: ArrayLike):
        """Calculate values of Beta-PDF for given p-values of this opinion.

        Parameters
        ----------
        p : array_like, float
            vector of probability over which to plot the Beta-PDF

        Returns
        -------
        beta_val : array_like, float
            Values, which represent the Beta PDF for this probability

        """
        [a, b] = self.get_evidence() + self.baseRate*self.W
        beta_val = beta.pdf(p, a, b)

        return beta_val

    def get_projected_prob(self):
        """Calculate the projected probability.

        Returns
        -------
        p : array_like, float
            Value, which represent the projected probability

        """
        p = self.belief[0]+self.baseRate[0]*self.uncertainty
        return p

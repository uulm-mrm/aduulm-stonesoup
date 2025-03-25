"""
This class implements a Gauss variable class.
This class creates discretized versions of distributions. Therefore the borders
of this bins and their probability are calculated.
For 1D the distribution of the values themself is regarded.
For higher dimensions (or if wanted also for 1D) the distance of the normal distributed
values to their mean is regarded. This results in a chi-squared distribution of the
squared distance with degrees of freedom equal to the dimensionality.

In all cases the assumed distribution is a normal distribution with mean = 0 and sigma = 1
To compare this distribution to others normalize the other distribution or use the
Mahalanobis distance for the N-dimensional case.
"""

from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.special import gamma, gammainc
from scipy.stats.distributions import chi2
import numpy as np

class RandomVariable(ABC):
    """Abstract class of a random variable.

    Attributes
    ----------
    number_bins : int
        Number of bins in which the distribution is discretized.
    borders_bins : array_like, float
        Upper borders of the bins. The border for the highest bin is infinity and NOT
        included in this array.
    probability_bins : array_like, float
        Probability for the bins.

    """

    def __init__(self, number_bins=3, sigma_border=3):
        """Construct abstract class RandomVariable.

        This class implements a template for any distribution and is not bound
        to a specific one.
        The only prerequisites are that the distribution in this class is
        separated in Bins with defined upper bounds and these bounds are saved in
        the attribute borders_bins in ascending order.
        The specific distribution is defined by the calculation of the probability
        of each bin in the subclasses.

        Parameters
        ----------
        number_bins : int, optional
            Number of bins in which the distribution is discretized.
            The default is 3.
        sigma_border : float, optional
            Distance after which only one bin is considered. The default is 3.

        Returns
        -------
        RandomVariable

        """
        self.number_bins = number_bins
        [self.borders_bins, self.probability_bins] = \
            self.generate_probabilities_for_bins(number_bins, sigma_border)

    @abstractmethod
    def generate_probabilities_for_bins(self, num_bins=3, sigma_border=3):
        """Generate probabilities and borders for the bins of the object.

        Parameters
        ----------
        number_bins : int, optional
            Number of bins in which the distribution is discretized.
            The default is 3.
        sigma_border : float, optional
            Distance after which only one bin is considered. The default is 3.

        Returns
        -------
        None.

        """

    def check_value_for_bin(self, measurement, weight=1.0):
        """Check in which one-dimensional bin the provided value lies.

        Parameters
        ----------
        measurement : float
            Measurement to check its membership of the bins.

        Returns
        -------
        position : array_like, int
            One-hot-vector of the position of the measurement in the bins.

        """
        # initialize with zeros
        position = np.zeros(self.number_bins)
        # if measurement lies in last bin
        if measurement > self.borders_bins[-1]:
            position[-1] = weight
        # for each upper border
        for i, border in enumerate(self.borders_bins):
            # if measurement is smaller than upper border increase the value at
            # this position and break to only find the smallest upper border
            if measurement <= border:
                position[i] = weight
                break
        return position


class GaussVariable1D(RandomVariable):
    """Class of a 1-dimensional Gauss variable.

    Attributes
    ----------
    number_bins : int
        Number of bins in which the Gauss distribution is discretized.
    borders_bins : array_like, float
        Upper borders of the bins. The border for the highest bin is infinity and NOT
        included in this array. There is always one bin for the whole region outside the
        three sigma interval on both sides: (-inf, -3] and [3, inf).
    probability_bins : array_like, float
        Probability for the bins.

    """

    def generate_probabilities_for_bins(self, num_bins=3, sigma_border=3):
        """Generate probabilities and borders for the bins of the object.

        Parameters
        ----------
        number_bins : int, optional
            Number of bins in which the Gauss distribution is discretized.
            The default is 3.
        sigma_border : float, optional
            Distance after which only one bin is considered. The default is 3.

        Returns
        -------
        list
            bins_borders : array_like, float
                Upper borders of the bins.
            bins_probabilities : array_like, float
                Probabilities of the bins.
        """
        bins_probabilities = np.zeros(num_bins)
        bins_borders = np.zeros(num_bins-1)
        # normal distribution with mu=0, and sigma=1
        rv = norm(0, 1)
        # distance between each upper border
        sigma_increment = 2*sigma_border / (num_bins - 2)
        # set first upper border as -3
        interval_upper_bound = -sigma_border
        # calculate probability for lowest bin, for values lower than -sigma_border
        bins_probabilities[0] = rv.cdf(interval_upper_bound)
        bins_borders[0] = interval_upper_bound
        # repeat for each following bin
        for idx in range(1, num_bins - 1):
            interval_lower_bound = interval_upper_bound
            interval_upper_bound += sigma_increment
            bins_probabilities[idx] = rv.cdf(interval_upper_bound) - \
                rv.cdf(interval_lower_bound)
            bins_borders[idx] = interval_upper_bound
        # calculate probability for highest bin, for values higher than 3
        bins_probabilities[-1] = 1 - rv.cdf(interval_upper_bound)

        return [bins_borders, bins_probabilities]


class GaussVariableND(RandomVariable):
    """Class of a N-dimensional Gauss variable.

    The bins in this class do not follow a grid-like structure for higher dimensions.
    The bins are classified using the distance to the mean value (here: 0 due to prior
    normalization). Therefore the membership to a certain bin means that the distance of
    the value to 0 is less than the upper border and more than the lower border of the bin.
    This essentially results in a discretized chi-squared-distribution. This is calculated
    using the Gamma distribution.

    Attributes
    ----------
    dim : int
        Number of dimensions this GaussVariable is about.
    number_bins : int
        Number of bins in which the Gauss distribution is discretized.
    borders_bins : array_like, float
        Upper borders of the bins. The border for the highest bin is infinity and NOT
        included in this array. There is always one bin for the whole region outside the
        three sigma interval: (3, inf).
    probability_bins : array_like, float
        Probability for the bins.
    volume: array_like, float
        Enclosed volume for each bin. Uses dimensionality and borders as parameters.

    """

    def __init__(self, dim=2, number_bins=2, sigma_border=3):
        """Construct object of class GaussVariableND.

        Parameters
        ----------
        dim : int, optional
            Number of dimensions this GaussVariable is about. The default is 2.
        number_bins : int, optional
            Number of bins in which the Gauss distribution is discretized.
            The default is 2.
        sigma_border : float, optional
            Distance after which only one bin is considered. The default is 3.

        Returns
        -------
        GaussVariableND

        """
        self.dim = dim
        super().__init__(number_bins, sigma_border)
        self.volume = self.calculate_volume()

    def prob_ndim_sigma_interval(self, sigma):
        """Calculate the probability that a value lies inside the given sigma radius.

        This method uses the regularized incomplete Gamma-function of the upper bound.
        This uses the function described in https://w.wiki/nes
        This calculates essentially the chi-squared-CDF with self.dim degrees
        of freedom and the argument sigma^2. This is equal to the gamma-CDF with
        parameters self.dim/2 and sigma^2/2.

        Parameters
        ----------
        sigma : float
            Distance up to which the probability of the value is calculated.

        Returns
        -------
        p : float
            Probability that a value lies inside the given sigma radius.

        """
        # p = gammainc(self.dim/2, sigma**2/2)
        # should be the same, see comment in method description
        p = chi2.cdf(sigma**2, self.dim)
        return p

    def calculate_volume(self):
        """Calculate Volume for each bin.

        This calculates the Volume enclosed by each bin. Last bin has no volume
        associated because it is unbounded.
        This uses the equation for an n-dimensional sphere given via
        https://dlmf.nist.gov/5.19#E4

        Returns
        -------
        vol : array_like, float
            Volume which each bin encloses. Last bin has no volume associated because it
            is unbounded.

        """
        # constant factor for multiple reuses
        fac = np.pi**(self.dim/2) / gamma(self.dim/2 + 1)
        # calculate the volume of the sphere inside the corresponding radius
        vol = np.array([fac * r**self.dim for r in self.borders_bins])
        return vol

    def generate_probabilities_for_bins(self, num_bins=2, sigma_border=3):
        """Generate probabilities and borders for the bins of the object.

        Parameters
        ----------
        number_bins : int, optional
            Number of bins in which the Gauss distribution is discretized.
            The default is 2.
        sigma_border : float, optional
            Distance after which only one bin is considered. The default is 3.

        Returns
        -------
        list
            bins_borders : array_like, float
                Upper borders of the bins.
            bins_probabilities : array_like, float
                Probabilities of the bins.
        """
        bins_probabilities = np.zeros(num_bins)
        bins_borders = np.zeros(num_bins-1)
        # distance between each upper border
        sigma_increment = sigma_border / (num_bins - 1)

        # set up the probabilities of the bins
        interval_upper_bound = sigma_increment
        # calculate probability for lowest bin, for values lower than first upper bound
        bins_probabilities[0] = self.prob_ndim_sigma_interval(interval_upper_bound)
        bins_borders[0] = interval_upper_bound
        # repeat for each following bin
        for idx in range(1, num_bins - 1):
            interval_lower_bound = interval_upper_bound
            interval_upper_bound += sigma_increment
            bins_probabilities[idx] = self.prob_ndim_sigma_interval(interval_upper_bound) - \
                self.prob_ndim_sigma_interval(interval_lower_bound)
            bins_borders[idx] = interval_upper_bound
        # calculate probability for highest bin, for values higher than highest_upper_bound
        bins_probabilities[-1] = 1 - self.prob_ndim_sigma_interval(interval_upper_bound)

        # borders in 1D chi2 space
        return [bins_borders, bins_probabilities]
